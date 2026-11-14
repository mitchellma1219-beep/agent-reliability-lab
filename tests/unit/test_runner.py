"""The runner: timeouts, crashes, unusable answers, budgets, seeds, evaluators and run ids."""

from __future__ import annotations

import contextlib
import datetime as dt
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent_reliability_lab import (
    AgentContext,
    AgentResponse,
    FunctionAgent,
    ScriptedAgent,
    ToolError,
    reference_agent,
    run_case,
    run_suite,
)
from agent_reliability_lab.errors import ConfigError
from agent_reliability_lab.evaluators import DEFAULT_EVALUATORS, EVALUATOR_NAMES
from agent_reliability_lab.evaluators.base import EvalContext, verdict
from agent_reliability_lab.faults import derive_seed
from agent_reliability_lab.models import (
    ID_PATTERN,
    Case,
    CaseResult,
    Evaluation,
    FaultSpec,
    RunConfig,
    Suite,
    Trace,
)
from agent_reliability_lab.runner import MAX_METADATA_CHARS, build_task, default_run_id
from helpers import case_data, codes, make_case, make_suite

ANSWER = "Refunds are available within 30 days of delivery."


def kinds(result: CaseResult) -> list[str]:
    return [error.kind for error in result.trace.errors]


def join_agent_threads() -> None:
    for thread in threading.enumerate():
        if thread.name == "arl-agent":
            thread.join(10)


# --------------------------------------------------------------------------- timeouts


def test_a_timeout_is_reported_and_the_late_agent_cannot_change_anything() -> None:
    release = threading.Event()
    late: list[str] = []

    def stuck(ctx: AgentContext) -> str:
        ctx.tools.call("get_order", {"order_id": "A-1001"})
        release.wait(30)
        for action in (
            lambda: ctx.tools.call("issue_refund", {"order_id": "A-1001", "amount": 5}),
            lambda: ctx.record_llm_call(input_tokens=10**6, output_tokens=10**6),
            lambda: ctx.wait(1000),
            lambda: ctx.annotate("late", True),
        ):
            try:
                action()
            except BaseException as exc:
                late.append(type(exc).__name__)
        return ANSWER

    started = time.perf_counter()
    result = run_case(make_case(), FunctionAgent(stuck), RunConfig(timeout_s=0.3))
    elapsed = time.perf_counter() - started
    assert 0.3 <= elapsed < 10
    snapshot = result.model_dump_json()
    release.set()
    join_agent_threads()
    assert late == ["HarnessAbort"] * 4
    assert result.model_dump_json() == snapshot
    assert (
        kinds(result) == ["timeout"] and "no answer within 0.3 s" in result.trace.errors[0].message
    )
    trace = result.trace
    assert (trace.status, trace.status_source, trace.output) == ("failed", "harness", "")
    assert [call.tool for call in trace.tool_calls] == ["get_order"] and trace.usage.llm_calls == 0
    assert trace.final_state["refunds"] == [] and trace.agent_metadata == {}
    assert not result.passed and result.failure_categories == ["timeout", "wrong_outcome"]
    assert codes(result, "task_success") == ["timeout", "wrong_status", "missing_text"]


def test_a_case_timeout_overrides_the_run_timeout() -> None:
    release = threading.Event()
    agent = FunctionAgent(lambda ctx: ANSWER if release.wait(30) else "")
    quick = make_case(timeout_s=0.2)
    started = time.perf_counter()
    result = run_case(quick, agent, RunConfig(timeout_s=60))
    assert time.perf_counter() - started < 10 and kinds(result) == ["timeout"]
    assert "no answer within 0.2 s" in result.trace.errors[0].message
    release.set()
    join_agent_threads()
    assert run_case(quick, agent, RunConfig(timeout_s=60)).passed


# --------------------------------------------------------------------------- crashes and exits


def test_an_exception_is_recorded_and_fails_the_case() -> None:
    def crash(ctx: AgentContext) -> str:
        ctx.tools.call("get_order", {"order_id": "A-1001"})
        raise RuntimeError("boom " + "x" * 2000)

    result = run_case(make_case(), FunctionAgent(crash))
    assert kinds(result) == ["agent_exception"]
    message = result.trace.errors[0].message
    assert (
        message.startswith("RuntimeError: boom xxx") and len(message) == len("RuntimeError: ") + 500
    )
    assert len(result.trace.tool_calls) == 1 and result.trace.status == "failed"
    assert not result.passed and "agent_crash" in result.failure_categories


@pytest.mark.parametrize(
    ("raised", "kind", "text"),
    [
        (SystemExit(3), "agent_exit", "tried to exit"),
        (KeyboardInterrupt(), "agent_exception", "KeyboardInterrupt"),
        (GeneratorExit(), "agent_exception", "GeneratorExit"),
        (MemoryError("out of memory"), "agent_exception", "MemoryError: out of memory"),
        (RecursionError("too deep"), "agent_exception", "RecursionError: too deep"),
    ],
)
def test_exits_and_base_exceptions_do_not_escape_the_runner(
    raised: BaseException, kind: str, text: str
) -> None:
    def agent(ctx: AgentContext) -> str:
        raise raised

    result = run_case(make_case(), FunctionAgent(agent))
    assert kinds(result) == [kind] and text in result.trace.errors[0].message
    assert not result.passed and result.failure_categories == ["agent_crash", "wrong_outcome"]


def test_a_tool_error_the_agent_does_not_handle_is_an_agent_exception() -> None:
    result = run_case(
        make_case(), FunctionAgent(lambda ctx: ctx.tools.call("get_order", {"order_id": "A-0"}))
    )
    assert kinds(result) == ["agent_exception"] and result.trace.errors[0].message.startswith(
        "ToolError:"
    )
    assert result.trace.tool_calls[0].error is not None


# --------------------------------------------------------------------------- unusable answers


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, "NoneType"),
        (42, "int"),
        (b"30 days", "bytes"),
        (["30 days"], "list"),
        ({"output": "30 days"}, "dict"),
    ],
)
def test_a_non_text_answer_is_unusable(value: Any, kind: str) -> None:
    result = run_case(make_case(), FunctionAgent(lambda ctx: value))
    assert kinds(result) == ["invalid_output"]
    assert result.trace.errors[0].message == f"the agent returned {kind}, not text"
    assert result.trace.output == "" and not result.passed
    assert result.failure_categories == ["unusable_output", "wrong_outcome"]


def test_an_oversized_answer_is_truncated_and_fails() -> None:
    huge = ANSWER + " " + "x" * 5000
    result = run_case(make_case(), FunctionAgent(lambda ctx: huge), RunConfig(max_output_chars=200))
    assert result.trace.output == huge[:200] and result.trace.output_truncated
    assert kinds(result) == ["invalid_output"]
    assert (
        f"output of {len(huge)} characters exceeds the limit of 200"
        in result.trace.errors[0].message
    )
    assert not result.passed and result.failure_categories == ["unusable_output"]
    exact = run_case(
        make_case(), FunctionAgent(lambda ctx: ANSWER.ljust(200)), RunConfig(max_output_chars=200)
    )
    assert exact.passed and not exact.trace.output_truncated


def test_metadata_is_merged_copied_and_bounded() -> None:
    def agent(ctx: AgentContext) -> AgentResponse:
        ctx.annotate("stage", "lookup")
        ctx.annotate("shared", "from annotate")
        return AgentResponse(output=ANSWER, metadata={"shared": "from response", "steps": [1, 2]})

    trace = run_case(make_case(), FunctionAgent(agent)).trace
    assert trace.agent_metadata == {"stage": "lookup", "shared": "from response", "steps": [1, 2]}

    def bloated(ctx: AgentContext) -> AgentResponse:
        return AgentResponse(output=ANSWER, metadata={"blob": "x" * (MAX_METADATA_CHARS + 1)})

    result = run_case(make_case(), FunctionAgent(bloated))
    assert result.trace.agent_metadata == {"dropped": "agent metadata was too large to store"}
    assert result.passed


def test_an_invalid_agent_response_is_an_agent_exception() -> None:
    result = run_case(
        make_case(), FunctionAgent(lambda ctx: AgentResponse(output=ANSWER, status="done"))
    )  # type: ignore[arg-type]
    assert (
        kinds(result) == ["agent_exception"] and "ValidationError" in result.trace.errors[0].message
    )


# --------------------------------------------------------------------------- runaway agents


def test_a_runaway_tool_loop_is_stopped_even_if_the_agent_swallows_everything() -> None:
    def polite(ctx: AgentContext) -> str:
        while True:
            try:
                ctx.tools.call("get_order", {"order_id": "A-1001"})
            except Exception:
                continue

    result = run_case(make_case(), FunctionAgent(polite), RunConfig(max_tool_calls=7))
    assert (
        kinds(result) == ["harness_abort"]
        and "more than 7 tool calls" in result.trace.errors[0].message
    )
    assert len(result.trace.tool_calls) == 7 and "runaway_loop" in result.failure_categories

    def stubborn(ctx: AgentContext) -> str:
        for _ in range(20):
            with contextlib.suppress(BaseException):
                ctx.tools.call("get_order", {"order_id": "A-1001"})
        return ANSWER

    result = run_case(make_case(), FunctionAgent(stubborn), RunConfig(max_tool_calls=7))
    assert kinds(result) == ["harness_abort"]
    assert result.trace.errors[0].message == "stopped by the harness: tool_call_budget"
    assert len(result.trace.tool_calls) == 7 and result.trace.output == ANSWER and not result.passed


def test_the_model_call_budget_is_enforced() -> None:
    def chatty(ctx: AgentContext) -> str:
        while True:
            ctx.record_llm_call(input_tokens=10, output_tokens=10)

    result = run_case(make_case(), FunctionAgent(chatty), RunConfig(max_llm_calls=5))
    assert (
        kinds(result) == ["harness_abort"]
        and "more than 5 model calls" in result.trace.errors[0].message
    )
    assert result.trace.usage.llm_calls == 5 and result.trace.usage.input_tokens == 50


# --------------------------------------------------------------------------- evaluators


class Exploding:
    name = "exploding"

    def applies(self, case: Case, trace: Trace) -> bool:
        return True

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        raise ZeroDivisionError("division by zero in a custom check " + "y" * 1000)


class Skipped(Exploding):
    name = "skipped"

    def applies(self, case: Case, trace: Trace) -> bool:
        return False


class AlwaysPasses:
    name = "always_passes"

    def applies(self, case: Case, trace: Trace) -> bool:
        return True

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        return verdict(
            self.name, [], details={"cost": ctx.estimated_cost_usd, "documents": len(ctx.context)}
        )


def test_an_evaluator_that_crashes_fails_the_case_instead_of_the_run() -> None:
    def priced(ctx: AgentContext) -> str:
        ctx.record_llm_call(model="sim-1", input_tokens=1000, output_tokens=0)
        return ANSWER

    evaluators = [AlwaysPasses(), Exploding(), Skipped()]
    result = run_case(make_case(), FunctionAgent(priced), evaluators=evaluators)
    assert [item.evaluator for item in result.evaluations] == ["always_passes", "exploding"]
    broken = result.evaluations[1]
    assert (
        not broken.passed
        and broken.score == 0.0
        and [f.code for f in broken.findings] == ["evaluator_error"]
    )
    assert broken.findings[0].message.startswith("ZeroDivisionError: division by zero")
    assert (
        broken.findings[0].message.endswith("(the check could not run)")
        and len(broken.findings[0].message) < 400
    )
    assert not result.passed and result.failure_categories == ["evaluator_error"]
    assert result.evaluations[0].details == {"cost": pytest.approx(0.001), "documents": 0}


def test_only_applicable_evaluators_run_by_default() -> None:
    result = run_case(make_case(), ScriptedAgent(output=ANSWER, status="completed"))
    assert [item.evaluator for item in result.evaluations] == ["task_success", "containment"]
    assert [evaluator.name for evaluator in DEFAULT_EVALUATORS] == list(EVALUATOR_NAMES)
    none = run_case(make_case(), ScriptedAgent(output="wrong"), evaluators=[])
    assert none.passed and none.evaluations == []


# --------------------------------------------------------------------------- faults and config


def test_unknown_fault_profiles_and_tools_are_configuration_errors() -> None:
    with pytest.raises(
        ConfigError, match="unknown fault profile 'nope'; choose from degraded, flaky-tools"
    ):
        run_case(make_case(), ScriptedAgent(output=ANSWER), RunConfig(fault_profile="nope"))
    with pytest.raises(ConfigError, match="unknown tool"):
        run_case(make_case(tools=["get_order", "launch_rocket"]), ScriptedAgent(output=ANSWER))


def test_extra_faults_and_profiles_are_applied_reproducibly() -> None:
    lookup = [("get_order", {"order_id": "A-1001"})]
    fault = FaultSpec(kind="permanent_error", tool="get_order")
    failed = run_case(make_case(), ScriptedAgent(lookup, ANSWER), RunConfig(extra_faults=[fault]))
    call = failed.trace.tool_calls[0]
    assert (
        not call.ok
        and call.fault == "permanent_error"
        and call.error is not None
        and not call.error.retryable
    )
    clean = run_case(make_case(), ScriptedAgent(lookup, ANSWER))
    assert clean.trace.tool_calls[0].ok and clean.trace.tool_calls[0].fault is None

    suite = make_suite(*[case_data(id=f"case-{n}") for n in range(12)])
    config = RunConfig(seed=5, trials=3, fault_profile="flaky-tools")
    agent = ScriptedAgent(lookup * 3, ANSWER, "completed")
    first, second = run_suite(suite, agent, config), run_suite(suite, agent, config)
    assert first.fingerprint() == second.fingerprint()
    faults = [
        call.fault for result in first.results for call in result.trace.tool_calls if call.fault
    ]
    assert faults and len(faults) < 3 * 36
    other = run_suite(suite, agent, RunConfig(seed=6, trials=3, fault_profile="flaky-tools"))
    assert other.fingerprint() != first.fingerprint()


# --------------------------------------------------------------------------- seeds and isolation


def test_seeds_depend_on_run_seed_case_and_trial() -> None:
    suite = make_suite(case_data(), case_data(id="case-2"))
    run = run_suite(suite, ScriptedAgent(output=ANSWER), RunConfig(seed=9, trials=3))
    assert [(r.case_id, r.trial) for r in run.results] == [
        ("case-1", 0), ("case-1", 1), ("case-1", 2), ("case-2", 0), ("case-2", 1), ("case-2", 2)
    ]  # fmt: skip
    assert [r.seed for r in run.results] == [
        derive_seed(9, r.case_id, r.trial) for r in run.results
    ]
    assert len({r.seed for r in run.results}) == 6
    assert all(r.trace.seed == r.seed and r.trace.trial == r.trial for r in run.results)
    other = run_suite(suite, ScriptedAgent(output=ANSWER), RunConfig(seed=10, trials=3))
    assert {r.seed for r in other.results}.isdisjoint({r.seed for r in run.results})


def test_every_trial_starts_from_a_fresh_world() -> None:
    refund = [("issue_refund", {"order_id": "A-1002", "amount": 10})]
    suite = make_suite(case_data())
    run = run_suite(suite, ScriptedAgent(refund, ANSWER), RunConfig(trials=3))
    for result in run.results:
        refunds = result.trace.final_state["refunds"]
        assert isinstance(refunds, list) and len(refunds) == 1
    assert len({str(result.trace.final_state["refunds"]) for result in run.results}) == 1


def test_progress_is_reported_after_every_trial() -> None:
    seen: list[tuple[str, int, bool]] = []
    suite = make_suite(
        case_data(), case_data(id="case-2", expected={"answer": {"contains_all": ["never"]}})
    )
    run = run_suite(
        suite,
        ScriptedAgent(output=ANSWER),
        RunConfig(trials=2),
        progress=lambda result: seen.append((result.case_id, result.trial, result.passed)),
    )
    assert seen == [
        ("case-1", 0, True),
        ("case-1", 1, True),
        ("case-2", 0, False),
        ("case-2", 1, False),
    ]
    assert [r.passed for r in run.results] == [True, True, False, False]


FILE_CASE = {
    "tools": ["read_file", "write_file"],
    "environment": {"files": {"notes/policy.txt": ANSWER}},
}


def case_directories() -> set[str]:
    return {path.name for path in Path(tempfile.gettempdir()).glob("arl-case-*")}


def test_a_case_with_files_gets_a_temporary_directory_that_is_removed() -> None:
    before = case_directories()
    seen: dict[str, Any] = {}

    def agent(ctx: AgentContext) -> str:
        seen["read"] = ctx.tools.call("read_file", {"path": "notes/policy.txt"})
        ctx.tools.call("write_file", {"path": "out/summary.txt", "content": "done"})
        seen["during"] = case_directories() - before
        return ANSWER

    result = run_case(make_case(**FILE_CASE), FunctionAgent(agent))
    assert result.passed and seen["read"] == {"path": "notes/policy.txt", "content": ANSWER}
    assert len(seen["during"]) == 1
    files = {"notes/policy.txt": ANSWER, "out/summary.txt": "done"}
    assert result.trace.final_state["files"] == files
    assert case_directories() == before
    no_files = run_case(make_case(), FunctionAgent(lambda ctx: ANSWER))
    assert no_files.trace.final_state["files"] == {}


def test_escaping_through_the_file_tools_is_refused_and_fails_containment() -> None:
    before = case_directories()
    refused: list[str] = []

    def agent(ctx: AgentContext) -> str:
        for path in ("../../etc/passwd", "/etc/passwd", "notes/../../outside.txt"):
            try:
                ctx.tools.call("read_file", {"path": path})
            except ToolError as exc:
                refused.append(exc.code)
        try:
            ctx.tools.call("write_file", {"path": "../escaped.txt", "content": "x"})
        except ToolError as exc:
            refused.append(exc.code)
        return ANSWER

    result = run_case(make_case(**FILE_CASE), FunctionAgent(agent))
    assert refused == ["permission_denied"] * 4
    assert [event.kind for event in result.trace.security_events] == ["path_escape"] * 4
    assert not result.passed and result.failed_evaluators() == ["containment"]
    assert result.failure_categories == ["sandbox_violation"]
    assert result.trace.final_state["files"] == {"notes/policy.txt": ANSWER}
    assert case_directories() == before
    assert not Path(tempfile.gettempdir(), "escaped.txt").exists()


def test_the_temporary_directory_is_removed_after_a_crash_or_a_timeout() -> None:
    before = case_directories()
    release = threading.Event()

    def crash(ctx: AgentContext) -> str:
        ctx.tools.call("write_file", {"path": "partial.txt", "content": "half"})
        raise RuntimeError("boom")

    def stuck(ctx: AgentContext) -> str:
        release.wait(30)
        return ANSWER

    assert kinds(run_case(make_case(**FILE_CASE), FunctionAgent(crash))) == ["agent_exception"]
    timed_out = run_case(make_case(**FILE_CASE), FunctionAgent(stuck), RunConfig(timeout_s=0.2))
    assert kinds(timed_out) == ["timeout"]
    release.set()
    join_agent_threads()
    assert case_directories() == before


def test_build_task_leaves_out_the_oracle() -> None:
    case = make_case(
        instructions=[{"id": "tone", "text": "Be brief."}],
        context=[{"id": "note", "title": "Note", "text": "Synthetic note."}],
        output_schema={"type": "object"},
        expected={"answer": {"contains_all": ["ORACLE-ONLY"]}, "status": "completed"},
        attack={
            "kind": "direct",
            "goal": "ATTACK-ONLY",
            "indicators": {"output_contains": ["pwned"]},
        },
    )
    task = build_task(case)
    dumped = task.model_dump_json()
    assert "ORACLE-ONLY" not in dumped and "ATTACK-ONLY" not in dumped and "pwned" not in dumped
    assert task.input == case.input and [doc.id for doc in task.context] == ["note"]
    assert task.output_schema == {"type": "object"} and task.sim is None
    assert not hasattr(task, "expected") and not hasattr(task, "attack")


# --------------------------------------------------------------------------- run ids and metadata


@pytest.mark.parametrize(
    ("name", "version", "expected"),
    [
        ("support-agent", "1.2.0", "unit-suite-support-agent-1.2.0-20261114T080000Z"),
        ("../../etc/passwd", "1.0/beta", "unit-suite-etc-passwd-1.0-beta-20261114T080000Z"),
        ("héllo wörld", "β", "unit-suite-h-llo-w-rld-x-20261114T080000Z"),
        ("代理", "一", "unit-suite-x-x-20261114T080000Z"),
        ("", "", "unit-suite-x-x-20261114T080000Z"),
        ("a" * 200, "9" * 200, f"unit-suite-{'a' * 30}-{'9' * 15}-20261114T080000Z"),
        ("x\x00y\n`z|", "..", "unit-suite-x-y--z-x-20261114T080000Z"),
    ],
)
def test_default_run_ids_are_always_safe_identifiers(
    name: str, version: str, expected: str
) -> None:
    suite = make_suite(case_data())
    agent = FunctionAgent(lambda ctx: ANSWER, name=name, version=version)
    created = dt.datetime(2026, 11, 14, 8, 0, 0, tzinfo=dt.UTC)
    run_id = default_run_id(suite, agent, created)
    assert run_id == expected and re.fullmatch(ID_PATTERN, run_id)
    run = run_suite(suite, agent, created_at=created)
    assert run.run_id == expected and run.created_at == "2026-11-14T08:00:00Z"
    assert run.agent_name == name[:100] and run.agent_version == version[:40]


def test_run_metadata_and_explicit_ids(core_suite: Suite) -> None:
    created = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    run = run_suite(core_suite, reference_agent(), run_id="my-run", created_at=created)
    assert (run.run_id, run.created_at) == ("my-run", "2026-01-01T19:04:05Z")
    assert (run.suite_name, run.suite_version) == (core_suite.name, core_suite.version)
    assert run.suite_description == core_suite.description and run.arl_version == "0.1.0"
    assert run.config == RunConfig() and len(run.results) == len(core_suite.cases)
    with pytest.raises(ValueError, match="run_id"):
        run_suite(core_suite, reference_agent(), run_id="../escape")


def test_agents_without_a_name_or_version_still_run() -> None:
    class Bare:
        def run(self, ctx: AgentContext) -> str:
            return ANSWER

    result = run_case(make_case(), Bare())  # type: ignore[arg-type]
    assert result.passed and (result.trace.agent_name, result.trace.agent_version) == ("Bare", "0")
    run = run_suite(make_suite(case_data()), Bare())  # type: ignore[arg-type]
    assert run.agent_name == "Bare" and run.run_id.startswith("unit-suite-agent-0-")
