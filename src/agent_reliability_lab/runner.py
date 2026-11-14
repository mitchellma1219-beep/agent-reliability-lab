"""Runs an agent against a suite and records what happened.

For every case and trial the runner builds a fresh simulated world, derives a seed from
``(run seed, case id, trial)``, runs the agent in a worker thread with a wall-clock timeout,
assembles the trace from what the sandbox recorded, and applies every evaluator that has
something to check.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import __version__
from .agents.base import Agent, AgentContext, AgentResponse, infer_status
from .errors import ConfigError, HarnessAbort
from .evaluators import DEFAULT_EVALUATORS
from .evaluators.base import EvalContext, Evaluator, finding
from .faults import FAULT_PROFILES, FaultInjector, derive_seed
from .filler import build_context
from .loader import resolve_cases
from .models import (
    AgentTask,
    Case,
    CaseResult,
    Evaluation,
    FaultSpec,
    RunConfig,
    RunResult,
    Status,
    Suite,
    Trace,
    TraceError,
)
from .sandbox import Limits, Sandbox
from .taxonomy import failure_categories
from .tools import BUILTIN_TOOLS, FILE_TOOLS, ToolSpec, World, resolve_tools

MAX_METADATA_CHARS = 10_000
StatusSource = Literal["declared", "inferred", "harness"]


def build_task(case: Case) -> AgentTask:
    """The agent-visible part of a case. ``expected`` and ``attack`` are never included."""
    return AgentTask(
        input=case.input,
        instructions=list(case.instructions),
        context=build_context(case),
        output_schema=case.output_schema,
        sim=case.sim,
    )


@dataclass
class _Outcome:
    value: Any = None
    error: BaseException | None = None
    abort: HarnessAbort | None = None
    exited: bool = False
    timed_out: bool = False
    finished: bool = False
    events: list[Any] = field(default_factory=list)


def _execute(
    agent: Agent,
    ctx: AgentContext,
    timeout_s: float,
    wrap: Callable[[Callable[[], None]], Callable[[], None]] | None = None,
) -> _Outcome:
    """Run the agent in a daemon thread and wait at most ``timeout_s`` seconds for it."""
    outcome = _Outcome()

    def target() -> None:
        try:
            outcome.value = agent.run(ctx)
            outcome.finished = True
        except HarnessAbort as exc:
            outcome.abort = exc
        except SystemExit:
            outcome.exited = True
        except Exception as exc:
            outcome.error = exc

    body = wrap(target) if wrap is not None else target
    worker = threading.Thread(target=body, name="arl-agent", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        outcome.timed_out = True
    return outcome


def _faults_for(case: Case, config: RunConfig) -> list[FaultSpec]:
    faults = [*case.faults, *config.extra_faults]
    if config.fault_profile is not None:
        if config.fault_profile not in FAULT_PROFILES:
            raise ConfigError(
                f"unknown fault profile {config.fault_profile!r}; "
                f"choose from {', '.join(sorted(FAULT_PROFILES))}"
            )
        faults.extend(FAULT_PROFILES[config.fault_profile])
    return faults


def _response(
    outcome: _Outcome, sandbox: Sandbox, config: RunConfig, timeout_s: float
) -> tuple[str, bool, Status, StatusSource, list[TraceError], dict[str, Any]]:
    """Normalise whatever the agent produced into output, status, errors and metadata."""
    errors: list[TraceError] = []
    output = ""
    status: Status = "failed"
    source: StatusSource = "harness"
    metadata: dict[str, Any] = {}
    if outcome.timed_out:
        errors.append(TraceError(kind="timeout", message=f"no answer within {timeout_s:g} s"))
    elif outcome.abort is not None:
        errors.append(TraceError(kind="harness_abort", message=outcome.abort.message))
    elif outcome.exited:
        errors.append(TraceError(kind="agent_exit", message="the agent tried to exit the process"))
    elif outcome.error is not None:
        message = f"{type(outcome.error).__name__}: {str(outcome.error)[:500]}"
        errors.append(TraceError(kind="agent_exception", message=message))
    elif isinstance(outcome.value, AgentResponse):
        output = outcome.value.output
        metadata = dict(outcome.value.metadata)
        if outcome.value.status is not None:
            status, source = outcome.value.status, "declared"
        else:
            status, source = infer_status(output), "inferred"
    elif isinstance(outcome.value, str):
        output = outcome.value
        status, source = infer_status(output), "inferred"
    else:
        kind = type(outcome.value).__name__
        errors.append(
            TraceError(kind="invalid_output", message=f"the agent returned {kind}, not text")
        )
    if sandbox.abort_reason is not None and not any(e.kind == "harness_abort" for e in errors):
        errors.append(
            TraceError(
                kind="harness_abort", message=f"stopped by the harness: {sandbox.abort_reason}"
            )
        )
    truncated = len(output) > config.max_output_chars
    if truncated:
        errors.append(
            TraceError(
                kind="invalid_output",
                message=f"output of {len(output)} characters exceeds the limit of "
                f"{config.max_output_chars} and was truncated",
            )
        )
        output = output[: config.max_output_chars]
    metadata = {**sandbox.annotations, **metadata}
    if len(json.dumps(metadata, default=str)) > MAX_METADATA_CHARS:
        metadata = {"dropped": "agent metadata was too large to store"}
    return output, truncated, status, source, errors, metadata


def run_case(
    case: Case,
    agent: Agent,
    config: RunConfig | None = None,
    trial: int = 0,
    *,
    evaluators: Sequence[Evaluator] | None = None,
    registry: Mapping[str, ToolSpec] | None = None,
) -> CaseResult:
    """Run one (already resolved) case once and evaluate the result."""
    config = config or RunConfig()
    registry = BUILTIN_TOOLS if registry is None else registry
    seed = derive_seed(config.seed, case.id, trial)
    tools = resolve_tools(list(case.tools or []), registry)
    task = build_task(case)
    needs_files = bool(case.environment.files) or bool(FILE_TOOLS & set(tools))
    workdir = Path(tempfile.mkdtemp(prefix="arl-case-")) if needs_files else None
    timeout_s = case.timeout_s if case.timeout_s is not None else config.timeout_s
    try:
        world = World.from_environment(case.environment, workdir)
        sandbox = Sandbox(
            world,
            tools,
            FaultInjector(_faults_for(case, config), seed),
            Limits(max_tool_calls=config.max_tool_calls, max_llm_calls=config.max_llm_calls),
        )
        ctx = AgentContext(task, sandbox, seed)
        started = time.perf_counter()
        outcome = _execute(agent, ctx, timeout_s)
        wall_ms = (time.perf_counter() - started) * 1000.0
        sandbox.close("timeout" if outcome.timed_out else "finished")
        output, truncated, status, source, errors, metadata = _response(
            outcome, sandbox, config, timeout_s
        )
        trace = Trace(
            case_id=case.id,
            trial=trial,
            seed=seed,
            agent_name=str(getattr(agent, "name", type(agent).__name__))[:100],
            agent_version=str(getattr(agent, "version", "0"))[:40],
            tool_calls=list(sandbox.tool_calls),
            llm_calls=list(sandbox.llm_calls),
            waits=list(sandbox.waits),
            output=output,
            output_truncated=truncated,
            status=status,
            status_source=source,
            usage=sandbox.usage(),
            latency_ms=sandbox.now_ms,
            wall_ms=wall_ms,
            errors=errors,
            security_events=list(sandbox.events),
            final_state=world.snapshot(),
            agent_metadata=json.loads(json.dumps(metadata, default=str)),
        )
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)

    cost = config.prices.estimate(trace.usage)
    eval_ctx = EvalContext(
        prices=config.prices, tools=registry, context=tuple(task.context), estimated_cost_usd=cost
    )
    evaluations: list[Evaluation] = []
    for evaluator in DEFAULT_EVALUATORS if evaluators is None else evaluators:
        if not evaluator.applies(case, trace):
            continue
        try:
            evaluations.append(evaluator.evaluate(case, trace, eval_ctx))
        except Exception as exc:
            evaluations.append(
                Evaluation(
                    evaluator=evaluator.name,
                    passed=False,
                    score=0.0,
                    findings=[
                        finding(
                            "evaluator_error",
                            f"{type(exc).__name__}: {str(exc)[:300]} (the check could not run)",
                        )
                    ],
                )
            )
    return CaseResult(
        case_id=case.id,
        category=case.category,
        trial=trial,
        seed=seed,
        passed=all(item.passed for item in evaluations),
        evaluations=evaluations,
        failure_categories=failure_categories(evaluations),
        estimated_cost_usd=cost,
        trace=trace,
    )


def _slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in text)
    return cleaned.strip("-.") or "x"


def default_run_id(suite: Suite, agent: Agent, created_at: dt.datetime) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    name = _slug(str(getattr(agent, "name", "agent")))[:30]
    version = _slug(str(getattr(agent, "version", "0")))[:15]
    return f"{_slug(suite.name)[:30]}-{name}-{version}-{stamp}"


def run_suite(
    suite: Suite,
    agent: Agent,
    config: RunConfig | None = None,
    *,
    run_id: str | None = None,
    created_at: dt.datetime | None = None,
    evaluators: Sequence[Evaluator] | None = None,
    registry: Mapping[str, ToolSpec] | None = None,
    progress: Callable[[CaseResult], None] | None = None,
) -> RunResult:
    """Run every case of ``suite`` for ``config.trials`` trials."""
    config = config or RunConfig()
    created = created_at or dt.datetime.now(dt.UTC)
    results: list[CaseResult] = []
    for case in resolve_cases(suite):
        for trial in range(config.trials):
            result = run_case(case, agent, config, trial, evaluators=evaluators, registry=registry)
            results.append(result)
            if progress is not None:
                progress(result)
    return RunResult(
        run_id=run_id or default_run_id(suite, agent, created),
        created_at=created.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        arl_version=__version__,
        suite_name=suite.name,
        suite_version=suite.version,
        suite_description=suite.description,
        agent_name=str(getattr(agent, "name", type(agent).__name__))[:100],
        agent_version=str(getattr(agent, "version", "0"))[:40],
        config=config,
        results=results,
    )
