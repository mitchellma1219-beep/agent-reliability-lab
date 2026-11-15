"""Streamlit dashboard for Agent Reliability Lab.

Run with ``arl dashboard`` or ``streamlit run app.py -- --runs-dir runs``. Without a runs
directory it shows demo data generated locally, so it works offline.

Everything that originates from an agent (answers, tool arguments, error messages) is shown
with ``st.code``, ``st.json`` or inside a table cell. It is never passed to a Markdown
renderer, so an answer cannot inject links, images or HTML into this page.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import streamlit as st

from agent_reliability_lab.dashboard import charts, data
from agent_reliability_lab.metrics import RunMetrics, compute_metrics
from agent_reliability_lab.models import CaseResult, RunResult, ToolCall
from agent_reliability_lab.regression import compare_runs
from agent_reliability_lab.report import (
    DISCLAIMER,
    VERDICT_TEXT,
    inline,
    plain,
    render_comparison,
    render_run_report,
)
from agent_reliability_lab.taxonomy import DESCRIPTIONS
from agent_reliability_lab.text import strip_control, truncate

VIEWS = ("Runs", "Metrics", "Traces", "Failures", "Trends", "Compare")
COST_COLUMN = "mean est. cost per case-trial (USD)"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def runs_dir_from_args() -> str | None:
    """``--runs-dir`` after ``--`` on the command line, else ``ARL_RUNS_DIR``, else demo data."""
    arguments = sys.argv[1:]
    if "--runs-dir" in arguments:
        position = arguments.index("--runs-dir")
        if position + 1 < len(arguments):
            return arguments[position + 1]
    return os.environ.get("ARL_RUNS_DIR") or None


@st.cache_data(show_spinner="Loading runs...")
def cached_runs(runs_dir: str | None, signature: str) -> tuple[list[RunResult], list[str], str]:
    return data.load_runs(runs_dir)


@st.cache_data(show_spinner=False)
def cached_metrics(fingerprint: str, _run: RunResult) -> RunMetrics:
    return compute_metrics(_run)


def directory_signature(runs_dir: str | None) -> str:
    """Changes whenever a stored run is added, removed or rewritten."""
    if not runs_dir or not os.path.isdir(runs_dir):
        return "demo"
    entries = []
    for name in sorted(os.listdir(runs_dir)):
        if name.endswith(".json"):
            stat = os.stat(os.path.join(runs_dir, name))
            entries.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(entries)


def theme() -> str | None:
    try:
        kind = st.context.theme.type
    except Exception:
        return None
    return kind if kind in ("light", "dark") else None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def untrusted(text: object, limit: int = 4000) -> None:
    """Show text that came from an agent or a run file, verbatim and inert."""
    st.code(truncate(strip_control(str(text)), limit) or "(empty)", language=None)


def plot(figure: Any, key: str) -> None:
    st.plotly_chart(figure, key=key, config=PLOT_CONFIG)


def kpi_row(metrics: RunMetrics) -> None:
    columns = st.columns(6)
    columns[0].metric("Pass rate", percent(metrics.pass_rate.rate))
    columns[1].metric("Task success", percent(metrics.task_success_rate.rate))
    columns[2].metric("Tool-call accuracy", percent(metrics.tool_call_accuracy.rate))
    p95 = metrics.latency.p95_ms
    columns[3].metric("p95 latency", "n/a" if p95 is None else f"{p95:,.0f} ms")
    cost = metrics.estimated_cost_usd
    columns[4].metric("Estimated cost", "n/a" if cost is None else f"${cost:,.4f}")
    columns[5].metric("Retry rate", percent(metrics.retry_rate))
    low, high = metrics.pass_rate.low, metrics.pass_rate.high
    if low is not None and high is not None:
        st.caption(
            f"Pass rate: {metrics.pass_rate.successes} of {metrics.pass_rate.total} cases passed "
            f"every trial and every check (Wilson 95% interval {low:.1%} to {high:.1%}). "
            "Latency is accounted time. Cost is an estimate from the price table "
            f"{inline(metrics.price_table, 40)}."
        )


def view_runs(runs: list[RunResult], metrics: list[RunMetrics], source: str) -> None:
    st.header("Runs")
    st.caption(source)
    st.dataframe(
        data.runs_frame(runs, metrics),
        hide_index=True,
        column_config={
            "pass rate": st.column_config.ProgressColumn(
                "pass rate", format="percent", min_value=0, max_value=1
            ),
            "task success": st.column_config.NumberColumn("task success", format="percent"),
            "tool-call accuracy": st.column_config.NumberColumn(
                "tool-call accuracy", format="percent"
            ),
            "p95 latency (ms)": st.column_config.NumberColumn("p95 latency (ms)", format="%.0f"),
            "est. cost (USD)": st.column_config.NumberColumn("est. cost (USD)", format="%.4f"),
        },
    )
    st.info(DISCLAIMER, icon=":material/info:")


def view_metrics(run: RunResult, metrics: RunMetrics) -> None:
    st.header("Metrics")
    kpi_row(metrics)
    if metrics.flaky_cases:
        st.warning(
            f"Flaky: {len(metrics.flaky_cases)} case(s) passed some trials and failed others.",
            icon=":material/warning:",
        )
        st.dataframe(
            {"flaky case": [plain(case, 100) for case in metrics.flaky_cases]}, hide_index=True
        )

    st.subheader("Pass rate by category")
    categories = data.categories_frame(metrics)
    plot(charts.rate_bars(categories, "category", "pass rate", theme()), "category-bars")
    with st.expander("Table view: categories"):
        st.dataframe(categories, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Checks by evaluator")
        st.dataframe(
            data.evaluators_frame(metrics),
            hide_index=True,
            column_config={
                "pass rate": st.column_config.NumberColumn("pass rate", format="percent")
            },
        )
    with right:
        st.subheader("Error taxonomy")
        taxonomy = data.taxonomy_frame(metrics)
        if taxonomy.empty:
            st.success("No failures in this run.", icon=":material/check_circle:")
        else:
            plot(
                charts.count_bars(taxonomy, "failure category", "case-trials", theme()),
                "taxonomy-bars",
            )
            with st.expander("Table view: failure categories"):
                taxonomy["meaning"] = [
                    DESCRIPTIONS.get(name, "") for name in taxonomy["failure category"]
                ]
                st.dataframe(taxonomy, hide_index=True)

    st.subheader("Latency, tokens and cost")
    latency = metrics.latency
    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        (
            ("p50", latency.p50_ms),
            ("p90", latency.p90_ms),
            ("p95", latency.p95_ms),
            ("p99", latency.p99_ms),
            ("max", latency.max_ms),
            ("mean", latency.mean_ms),
        ),
        strict=True,
    ):
        column.metric(f"Latency {label}", "n/a" if value is None else f"{value:,.0f} ms")
    usage = st.columns(4)
    usage[0].metric("Tokens in", f"{metrics.input_tokens:,}")
    usage[1].metric("Tokens out", f"{metrics.output_tokens:,}")
    usage[2].metric("Tool calls", f"{metrics.tool_calls:,}")
    usage[3].metric("Injected faults", f"{metrics.injected_faults:,}")

    st.subheader("Cases")
    st.dataframe(data.cases_frame(metrics), hide_index=True)
    st.download_button(
        "Download this run's report (Markdown)",
        render_run_report(run, metrics),
        file_name=f"{run.run_id}.md",
        mime="text/markdown",
    )


def pick_result(run: RunResult, key: str, only_failures: bool = False) -> CaseResult | None:
    results = [r for r in run.results if not (only_failures and r.passed)]
    if not results:
        return None
    labels = [
        f"{'PASS' if r.passed else 'FAIL'}  {plain(r.case_id, 80)}  (trial {r.trial})"
        for r in results
    ]
    chosen = st.selectbox(
        "Case and trial", range(len(results)), format_func=labels.__getitem__, key=key
    )
    return results[chosen]


def show_trace(result: CaseResult) -> None:
    trace = result.trace
    columns = st.columns(5)
    columns[0].metric("Result", "PASS" if result.passed else "FAIL")
    columns[1].metric("Ended as", trace.status)
    columns[2].metric("Tool calls", len(trace.tool_calls))
    columns[3].metric("Latency", f"{trace.latency_ms:,.0f} ms")
    cost = result.estimated_cost_usd
    columns[4].metric("Estimated cost", "n/a" if cost is None else f"${cost:,.4f}")

    st.subheader("Final answer")
    untrusted(trace.output)
    for error in trace.errors:
        st.error(f"{error.kind}: {inline(error.message, 300)}", icon=":material/error:")
    for event in trace.security_events:
        st.error(
            f"Containment: {inline(event.kind, 60)} {inline(event.detail, 200)}",
            icon=":material/gpp_bad:",
        )

    st.subheader("Evaluations")
    st.dataframe(data.evaluations_frame(result), hide_index=True)

    steps = data.steps_frame(trace)
    st.subheader("Timeline")
    if steps.empty:
        st.caption("No steps were recorded.")
    else:
        plot(charts.timeline(steps, theme()), "trace-timeline")
        with st.expander("Table view: steps", expanded=True):
            st.dataframe(steps, hide_index=True)

    tool_calls = [step for step in trace.steps() if isinstance(step, ToolCall)]
    if tool_calls:
        st.subheader("Tool calls")
        for call in tool_calls:
            state = "ok" if call.ok else "failed"
            title = f"Step {call.index}: {inline(call.tool, 60)} ({state}, attempt {call.attempt})"
            with st.expander(title):
                st.caption("Arguments")
                st.json(call.arguments, expanded=True)
                if call.error is not None:
                    st.caption("Error")
                    untrusted(f"{call.error.code}: {call.error.message}", 600)
                else:
                    st.caption("Result")
                    st.json(call.result, expanded=False)
                if call.fault:
                    st.caption(f"Injected fault: {call.fault}")
    with st.expander("State of the simulated world after the run"):
        st.json(trace.final_state, expanded=False)


def view_traces(run: RunResult) -> None:
    st.header("Trace viewer")
    result = pick_result(run, "trace-case")
    if result is None:
        st.caption("This run has no results.")
        return
    show_trace(result)


def view_failures(run: RunResult) -> None:
    st.header("Failure examples")
    failures = data.failures_frame(run)
    if failures.empty:
        st.success("Every case passed in this run.", icon=":material/check_circle:")
        return
    st.caption(f"{len(failures)} failing case-trial(s) of {len(run.results)}.")
    st.dataframe(failures, hide_index=True)
    result = pick_result(run, "failure-case", only_failures=True)
    if result is None:
        return
    for evaluation in result.evaluations:
        for finding in evaluation.failures():
            st.error(
                f"{evaluation.evaluator} / {finding.code}: {inline(finding.message, 400)}",
                icon=":material/error:",
            )
    st.subheader("Final answer")
    untrusted(result.trace.output)
    with st.expander("Steps"):
        st.dataframe(data.steps_frame(result.trace), hide_index=True)


def pick_series(runs: list[RunResult], key: str) -> list[RunResult]:
    keys = data.series_keys(runs)
    labels = [plain(f"{agent} on suite {suite}", 120) for agent, suite in keys]
    chosen = st.selectbox(
        "Agent and suite", range(len(keys)), format_func=labels.__getitem__, key=key
    )
    return data.series_runs(runs, keys[chosen])


def view_trends(runs: list[RunResult]) -> None:
    st.header("Historical trends")
    series = pick_series(runs, "trend-series")
    frame = data.trend_frame(series)
    if len(frame) < 2:
        st.caption("At least two runs of the same agent on the same suite are needed for a trend.")
    panels = (
        ("Pass rate", "pass rate", True, "", 0),
        ("Task success rate", "task success", True, "", 0),
        ("p95 latency", "p95 latency (ms)", False, " ms", 0),
        ("Mean estimated cost per case-trial", COST_COLUMN, False, " USD", 4),
    )
    for row in (panels[:2], panels[2:]):
        for column, (title, value, as_rate, unit, decimals) in zip(st.columns(2), row, strict=True):
            with column:
                st.subheader(title)
                if frame[value].notna().any():
                    figure = charts.trend_line(
                        frame, value, theme(), as_rate=as_rate, unit=unit, decimals=decimals
                    )
                    plot(figure, f"trend-{value}")
                else:
                    st.caption("No data.")
    with st.expander("Table view: history", expanded=True):
        st.dataframe(frame, hide_index=True)


def view_compare(runs: list[RunResult]) -> None:
    st.header("Version comparison")
    series = pick_series(runs, "compare-series")
    if len(series) < 2:
        st.caption("At least two runs of the same agent on the same suite are needed.")
        return
    labels = [data.run_label(run) for run in series]
    left, right = st.columns(2)
    with left:
        baseline_index = st.selectbox(
            "Baseline",
            range(len(series)),
            index=len(series) - 2,
            format_func=labels.__getitem__,
            key="baseline",
        )
    with right:
        candidate_index = st.selectbox(
            "Candidate",
            range(len(series)),
            index=len(series) - 1,
            format_func=labels.__getitem__,
            key="candidate",
        )
    baseline, candidate = series[baseline_index], series[candidate_index]
    report = compare_runs(baseline, candidate)
    message = VERDICT_TEXT[report.verdict]
    if report.verdict == "regression":
        st.error(message, icon=":material/error:")
    elif report.verdict == "needs_review":
        st.warning(message, icon=":material/warning:")
    elif report.verdict == "improved":
        st.success(message, icon=":material/check_circle:")
    else:
        st.info(message, icon=":material/info:")
    for reason in report.reasons:
        st.text(f"- {reason}")
    for warning in report.warnings:
        st.text(f"warning: {warning}")

    columns = st.columns(4)
    columns[0].metric("Baseline pass rate", percent(report.baseline_pass_rate.rate))
    columns[1].metric("Candidate pass rate", percent(report.candidate_pass_rate.rate))
    columns[2].metric("Stopped passing", report.stopped_passing)
    columns[3].metric("Started passing", report.started_passing)
    st.caption(f"Exact McNemar test, two-sided: p = {report.p_value:.4f} (alpha {report.alpha:g}).")

    st.subheader("Pass rate by category")
    frame = data.comparison_frame(compute_metrics(baseline), compute_metrics(candidate))
    changed = frame[frame["baseline"] != frame["candidate"]]
    if changed.empty:
        st.caption(f"The pass rate is the same in all {len(frame)} categories.")
    else:
        st.caption(
            f"Showing the {len(changed)} of {len(frame)} categories whose pass rate differs. "
            "The table view lists all of them."
        )
        plot(charts.comparison_bars(changed, theme()), "comparison-bars")
    with st.expander("Table view: categories"):
        st.dataframe(frame, hide_index=True)

    st.subheader("Cases that changed")
    changes = data.changes_frame(report)
    if changes.empty:
        st.caption("No case changed outcome.")
    else:
        st.dataframe(changes, hide_index=True)
    st.download_button(
        "Download the regression report (Markdown)",
        render_comparison(report),
        file_name=f"compare-{baseline.run_id}-vs-{candidate.run_id}.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(
        page_title="Agent Reliability Lab", page_icon=":material/science:", layout="wide"
    )
    runs_dir = runs_dir_from_args()
    runs, problems, source = cached_runs(runs_dir, directory_signature(runs_dir))
    st.sidebar.title("Agent Reliability Lab")
    view = st.sidebar.radio("View", VIEWS, key="view")
    for problem in problems:
        st.sidebar.warning(f"Skipped: {inline(problem, 300)}", icon=":material/warning:")
    if not runs:
        st.title("Agent Reliability Lab")
        st.warning(
            "No stored runs were found. Create one with `arl run`.", icon=":material/warning:"
        )
        return
    metrics = [cached_metrics(run.fingerprint(), run) for run in runs]
    labels = [data.run_label(run) for run in runs]
    selected = st.sidebar.selectbox(
        "Run", range(len(runs)), index=len(runs) - 1, format_func=labels.__getitem__, key="run"
    )
    run, run_metrics = runs[selected], metrics[selected]
    st.sidebar.caption("The run selection applies to Metrics, Traces and Failures.")
    if run.notes:
        st.sidebar.text(truncate(strip_control(run.notes), 300))

    if view == "Runs":
        view_runs(runs, metrics, source)
    elif view == "Metrics":
        view_metrics(run, run_metrics)
    elif view == "Traces":
        view_traces(run)
    elif view == "Failures":
        view_failures(run)
    elif view == "Trends":
        view_trends(runs)
    else:
        view_compare(runs)


main()
