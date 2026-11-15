"""Command-line interface: ``arl run | compare | report | list | scenarios | calibrate-judge``.

Exit codes: 0 success, 1 a gate failed (cases failed in ``run``, a regression in
``compare``, a problem in ``matrix``), 2 invalid input or configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .agents.loading import load_agent
from .agents.simulated import FAILURE_MODES
from .errors import ArlError, ConfigError
from .evaluators import EVALUATOR_NAMES
from .faults import FAULT_PROFILES
from .loader import SUFFIXES, load_suite, parse_document, resolve_cases
from .metrics import compute_metrics
from .models import GuardConfig, PriceTable, RunConfig, Suite, default_price_table
from .report import (
    DISCLAIMER,
    money,
    number,
    plain,
    rate,
    render_calibration,
    render_comparison,
    render_run_list,
    render_run_report,
    to_json,
)
from .runner import run_suite
from .scenarios import ALL, builtin_suite_names, load_builtin_suite
from .storage import RunStore, resolve_run, validate_run_name

DEFAULT_RUNS_DIR = "runs"


def _load_suite(reference: str) -> Suite:
    if reference == ALL or reference in builtin_suite_names():
        return load_builtin_suite(reference)
    return load_suite(reference)


def _load_prices(path: str | None) -> PriceTable:
    if path is None:
        return default_price_table()
    file = Path(path)
    fmt = SUFFIXES.get(file.suffix.lower())
    if fmt is None or not file.is_file():
        raise ConfigError(f"{file}: expected an existing .yaml, .yml or .json price table")
    try:
        return PriceTable.model_validate(parse_document(file.read_text("utf-8"), fmt, str(file)))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ConfigError(f"{file}: invalid price table ({where}: {first['msg']})") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{file}: cannot be read as UTF-8 text") from exc


def _emit(text: str, out: str | None) -> None:
    if out is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return
    target = Path(out)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write {target}: {exc.strerror}") from exc
    print(f"wrote {target}")


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    suite = _load_suite(args.suite)
    agent = load_agent(args.agent, args.agent_name, args.agent_version)
    if args.name is not None:
        validate_run_name(args.name)
    if not 0.0 <= args.fail_under <= 1.0:
        raise ConfigError("--fail-under must be between 0 and 1")
    try:
        config = RunConfig(
            seed=args.seed,
            trials=args.trials,
            timeout_s=args.timeout,
            max_tool_calls=args.max_tool_calls,
            fault_profile=args.faults,
            prices=_load_prices(args.prices),
            guard=GuardConfig(
                enabled=not args.no_guard,
                allow_network=args.allow_network,
                allow_subprocess=args.allow_subprocess,
                strict_reads=args.strict_reads,
            ),
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ConfigError(f"invalid option {first['loc'][-1]}: {first['msg']}") from exc
    run = run_suite(suite, agent, config, run_id=args.name)
    metrics = compute_metrics(run)
    saved: Path | None = None
    if not args.no_save:
        store = RunStore(args.runs_dir)
        if args.name is None and store.exists(run.run_id):
            run.run_id = validate_run_name(f"{run.run_id}-{len(store.names()) + 1}")
        saved = store.save(run, overwrite=args.overwrite)

    if args.format == "json":
        payload = metrics.model_dump(mode="json")
        payload["saved_to"] = str(saved) if saved else None
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        profile = f" | fault profile {config.fault_profile}" if config.fault_profile else ""
        print(
            f"Suite {plain(suite.name, 60)} ({metrics.cases} cases) | agent "
            f"{plain(run.agent_name, 60)} {plain(run.agent_version, 30)} | seed {config.seed} | "
            f"{config.trials} trial(s){profile}"
        )
        print()
        failures = {result.case_id: result for result in reversed(run.results) if not result.passed}
        for stat in metrics.case_stats:
            if stat.outcome == "pass":
                if not args.quiet:
                    print(f"  PASS   {plain(stat.case_id, 80)}")
                continue
            label = "FAIL " if stat.outcome == "fail" else "FLAKY"
            detail = ""
            failed = failures.get(stat.case_id)
            if failed is not None:
                evaluation = next(item for item in failed.evaluations if not item.passed)
                problem = evaluation.failures()[0]
                detail = f"  {evaluation.evaluator}: {plain(problem.message, 110)}"
            trials = f" ({stat.passes}/{stat.trials})" if stat.trials > 1 else ""
            print(f"  {label}  {plain(stat.case_id, 80)}{trials}{detail}")
        print()
        print(f"Pass rate          {rate(metrics.pass_rate)}")
        print(f"Task success       {rate(metrics.task_success_rate, False)}")
        print(f"Tool-call accuracy {rate(metrics.tool_call_accuracy, False)}")
        latency = metrics.latency
        print(
            f"Latency (accounted) p50 {number(latency.p50_ms, 0)} ms, "
            f"p95 {number(latency.p95_ms, 0)} ms, p99 {number(latency.p99_ms, 0)} ms"
        )
        print(
            f"Tokens             {metrics.input_tokens:,} in, {metrics.output_tokens:,} out; "
            f"estimated cost {money(metrics.estimated_cost_usd)} "
            f"(estimate, price table '{plain(metrics.price_table, 40)}')"
        )
        print(
            f"Tool calls         {metrics.tool_calls:,}, retries {metrics.retries:,}, "
            f"injected faults {metrics.injected_faults:,}"
        )
        if metrics.taxonomy:
            top = ", ".join(f"{name} {count}" for name, count in list(metrics.taxonomy.items())[:6])
            print(f"Failure categories {top}")
        if saved is not None:
            print(f"Saved run          {saved}")
        print()
        print(DISCLAIMER)
    passed = metrics.pass_rate.rate if metrics.pass_rate.rate is not None else 0.0
    return 0 if passed >= args.fail_under else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    from .regression import compare_runs

    store = RunStore(args.runs_dir)
    report = compare_runs(
        resolve_run(args.baseline, store),
        resolve_run(args.candidate, store),
        alpha=args.alpha,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    _emit(to_json(report) if args.format == "json" else render_comparison(report), args.out)
    if report.verdict == "regression" or (args.strict and report.verdict == "needs_review"):
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = resolve_run(args.run, RunStore(args.runs_dir))
    metrics = compute_metrics(run)
    _emit(to_json(metrics) if args.format == "json" else render_run_report(run, metrics), args.out)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    infos, problems = RunStore(args.runs_dir).list()
    if args.format == "json":
        rows = [{**info.__dict__, "pass_rate": info.pass_rate} for info in infos]
        print(json.dumps(rows, indent=2))
    else:
        sys.stdout.write(render_run_list(infos))
    for problem in problems:
        print(f"warning: {plain(problem, 300)}", file=sys.stderr)
    return 0


def _cmd_scenarios(args: argparse.Namespace) -> int:
    if args.suite is None:
        suites = [load_builtin_suite(name) for name in builtin_suite_names()]
        if args.format == "json":
            rows = [
                {
                    "suite": suite.name,
                    "cases": len(suite.cases),
                    "categories": suite.categories(),
                    "description": " ".join(suite.description.split()),
                }
                for suite in suites
            ]
            print(json.dumps(rows, indent=2))
            return 0
        for suite in suites:
            print(f"{suite.name} ({len(suite.cases)} cases)")
            print(f"  {' '.join(suite.description.split())}")
            print(f"  categories: {', '.join(suite.categories())}")
            print()
        total = sum(len(suite.cases) for suite in suites)
        print(f"{total} cases in {len(suites)} suites. Use '{ALL}' to run every suite at once.")
        print(f"Evaluators: {', '.join(EVALUATOR_NAMES)}")
        print(f"Simulated failure modes: {', '.join(FAILURE_MODES)}")
        print(f"Fault profiles for --faults: {', '.join(sorted(FAULT_PROFILES))}")
        return 0
    suite = _load_suite(args.suite)
    cases = [
        {
            "id": case.id,
            "category": case.category,
            "description": case.description,
            "checks": case.substantive_checks(),
            "faults": [fault.kind for fault in case.faults],
        }
        for case in resolve_cases(suite)
    ]
    if args.format == "json":
        print(json.dumps({"suite": suite.name, "cases": cases}, indent=2))
        return 0
    print(f"{plain(suite.name, 60)}: {len(cases)} cases")
    for case in cases:
        faults = f" [faults: {', '.join(case['faults'])}]" if case["faults"] else ""
        print(f"  {plain(case['id'], 60):38s} {case['category']:28s}{faults}")
        if case["description"]:
            print(f"      {plain(case['description'], 150)}")
        print(f"      checks: {', '.join(case['checks'])}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .judge import MockJudge, ReviewPolicy, calibrate
    from .judge.base import Judge
    from .judge.calibration import builtin_calibration_set, load_calibration_set

    dataset = (
        builtin_calibration_set()
        if args.dataset == "builtin"
        else load_calibration_set(args.dataset)
    )
    judge: Judge
    if args.judge == "mock":
        judge = MockJudge()
    else:
        from .judge.anthropic_backend import AnthropicJudge

        judge = AnthropicJudge(args.judge_model or "", enabled=args.allow_llm_judge)
    try:
        policy = ReviewPolicy(
            min_confidence=args.min_confidence,
            ambiguity_margin=args.ambiguity_margin,
            samples=args.samples,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ConfigError(f"invalid option {first['loc'][-1]}: {first['msg']}") from exc
    report = calibrate(judge, dataset.rubric, dataset.items, policy, dataset.description)
    _emit(to_json(report) if args.format == "json" else render_calibration(report), args.out)
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    from .detection import render_matrix, run_detection_matrix

    matrix = run_detection_matrix(seed=args.seed)
    if args.format == "json":
        payload = {
            "ok": matrix.ok,
            "seed": matrix.seed,
            "suites": matrix.suites,
            "reference_cases": matrix.reference_cases,
            "reference_failures": matrix.reference_failures,
            "rows": [
                {**row.__dict__, "ok": row.ok, "primary": list(row.primary)} for row in matrix.rows
            ],
            "category_coverage": {
                name: {"target": target, "failed": failed, "cases": total}
                for name, (target, failed, total) in matrix.category_coverage.items()
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_matrix(matrix))
    return 0 if matrix.ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import write_demo_runs

    paths = write_demo_runs(args.runs_dir, trials=args.trials, seed=args.seed)
    for path in paths:
        print(f"wrote {path}")
    print(f"{len(paths)} demo runs. Try: arl list --runs-dir {args.runs_dir}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise ConfigError(
            "the dashboard needs the 'ui' extra: pip install 'agent-reliability-lab[ui]'"
        ) from exc
    app = Path(__file__).parent / "dashboard" / "app.py"
    command = ["streamlit", "run", str(app), "--browser.gatherUsageStats", "false"]
    if args.port is not None:
        command += ["--server.port", str(args.port)]
    if args.headless:
        command += ["--server.headless", "true"]
    if args.runs_dir is not None:
        command += ["--", "--runs-dir", args.runs_dir]
    sys.argv = command
    streamlit_cli.main()
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arl",
        description="Agent Reliability Lab: test whether an AI agent is reliable, safe and "
        "affordable under difficult conditions.",
        epilog=DISCLAIMER,
    )
    parser.add_argument("--version", action="version", version=f"arl {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def runs_dir(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--runs-dir", default=DEFAULT_RUNS_DIR, help="directory of stored runs (default: runs)"
        )

    run = commands.add_parser("run", help="run an agent against a suite and store the result")
    run.add_argument(
        "--suite", required=True, help=f"built-in suite name, '{ALL}', or a .yaml/.json suite file"
    )
    run.add_argument(
        "--agent",
        default="reference",
        help="'reference', 'faulty:<mode>[@rate]', 'module:attr' or 'file.py:attr'",
    )
    run.add_argument("--agent-name", help="override the agent's name in the stored run")
    run.add_argument("--agent-version", help="override the agent's version in the stored run")
    run.add_argument("--trials", type=int, default=1, help="trials per case (default 1)")
    run.add_argument("--seed", type=int, default=0, help="run seed (default 0)")
    run.add_argument(
        "--timeout", type=float, default=10.0, help="wall-clock seconds per case (default 10)"
    )
    run.add_argument("--max-tool-calls", type=int, default=50, help="tool calls allowed per case")
    run.add_argument(
        "--faults", choices=sorted(FAULT_PROFILES), help="inject a fault profile into every tool"
    )
    run.add_argument("--prices", help="price table file (.yaml/.json) for cost estimates")
    run.add_argument("--name", help="name for the stored run (letters, digits, . _ -)")
    run.add_argument("--overwrite", action="store_true", help="replace a stored run of that name")
    run.add_argument("--no-save", action="store_true", help="do not store the run")
    run.add_argument("--no-guard", action="store_true", help="disable the containment guard")
    run.add_argument(
        "--allow-network", action="store_true", help="let agent code open network connections"
    )
    run.add_argument(
        "--allow-subprocess", action="store_true", help="let agent code start processes"
    )
    run.add_argument(
        "--strict-reads", action="store_true", help="also block file reads outside the sandbox"
    )
    run.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="exit with status 1 if the pass rate is below this (default 1.0; 0 never fails)",
    )
    run.add_argument("--format", choices=["text", "json"], default="text")
    run.add_argument("--quiet", action="store_true", help="only list cases that did not pass")
    runs_dir(run)
    run.set_defaults(handler=_cmd_run)

    compare = commands.add_parser("compare", help="compare two runs and give a regression verdict")
    compare.add_argument("baseline", help="stored run name or path to a run .json file")
    compare.add_argument("candidate", help="stored run name or path to a run .json file")
    compare.add_argument("--alpha", type=float, default=0.05, help="significance level")
    compare.add_argument(
        "--tolerance", type=float, default=0.10, help="smallest relative cost/latency change"
    )
    compare.add_argument("--seed", type=int, default=0, help="seed for the bootstrap")
    compare.add_argument(
        "--strict", action="store_true", help="also exit with status 1 on 'needs_review'"
    )
    compare.add_argument("--format", choices=["md", "json"], default="md")
    compare.add_argument("--out", help="write the report to this file")
    runs_dir(compare)
    compare.set_defaults(handler=_cmd_compare)

    report = commands.add_parser("report", help="render a report for a stored run")
    report.add_argument("run", help="stored run name or path to a run .json file")
    report.add_argument("--format", choices=["md", "json"], default="md")
    report.add_argument("--out", help="write the report to this file")
    runs_dir(report)
    report.set_defaults(handler=_cmd_report)

    listing = commands.add_parser("list", help="list stored runs")
    listing.add_argument("--format", choices=["text", "json"], default="text")
    runs_dir(listing)
    listing.set_defaults(handler=_cmd_list)

    scenarios = commands.add_parser("scenarios", help="list built-in suites or the cases of one")
    scenarios.add_argument("suite", nargs="?", help="suite name or file to list in detail")
    scenarios.add_argument("--format", choices=["text", "json"], default="text")
    scenarios.set_defaults(handler=_cmd_scenarios)

    calibrate = commands.add_parser(
        "calibrate-judge", help="measure a judge against labelled examples"
    )
    calibrate.add_argument(
        "--dataset", default="builtin", help="calibration file, or 'builtin' for the demo set"
    )
    calibrate.add_argument("--judge", choices=["mock", "anthropic"], default="mock")
    calibrate.add_argument("--judge-model", help="model id for --judge anthropic")
    calibrate.add_argument(
        "--allow-llm-judge",
        action="store_true",
        help="required for --judge anthropic: sends data to the API and costs money",
    )
    calibrate.add_argument("--min-confidence", type=float, default=0.75)
    calibrate.add_argument("--ambiguity-margin", type=float, default=0.1)
    calibrate.add_argument("--samples", type=int, default=1, help="readings per item")
    calibrate.add_argument("--format", choices=["md", "json"], default="md")
    calibrate.add_argument("--out", help="write the report to this file")
    calibrate.set_defaults(handler=_cmd_calibrate)

    matrix = commands.add_parser(
        "matrix", help="run the detection matrix over the simulated faulty agents"
    )
    matrix.add_argument("--seed", type=int, default=0)
    matrix.add_argument("--format", choices=["md", "json"], default="md")
    matrix.set_defaults(handler=_cmd_matrix)

    demo = commands.add_parser("demo", help="write the deterministic demo runs to a directory")
    demo.add_argument("--trials", type=int, default=2)
    demo.add_argument("--seed", type=int, default=0)
    runs_dir(demo)
    demo.set_defaults(handler=_cmd_demo)

    dashboard = commands.add_parser("dashboard", help="open the Streamlit dashboard")
    dashboard.add_argument("--runs-dir", help="stored runs to show (default: generated demo runs)")
    dashboard.add_argument("--port", type=int)
    dashboard.add_argument("--headless", action="store_true")
    dashboard.set_defaults(handler=_cmd_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``arl`` console script."""
    args = build_parser().parse_args(argv)
    try:
        result: int = args.handler(args)
        return result
    except ArlError as exc:
        print(f"error: {plain(exc, 2000)}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
