"""The sandbox: validation, recording, budgets and isolation between agent and fixtures."""

from __future__ import annotations

import pytest

from agent_reliability_lab.errors import HarnessAbort, ToolError
from agent_reliability_lab.models import Environment
from agent_reliability_lab.sandbox import Limits, Sandbox
from agent_reliability_lab.tools import BUILTIN_TOOLS, World, resolve_tools
from helpers import ORDERS


def sandbox(limits: Limits | None = None, tools: list[str] | None = None) -> Sandbox:
    environment = Environment.model_validate({"records": {"orders": ORDERS}})
    names = tools or ["get_order", "calculator", "send_email", "search_docs"]
    return Sandbox(World.from_environment(environment), resolve_tools(names), limits=limits)


def test_successful_call_is_recorded_with_simulated_latency() -> None:
    box = sandbox()
    result = box.call("get_order", {"order_id": "A-1001"})
    assert result["status"] == "shipped"
    (call,) = box.tool_calls
    assert (call.tool, call.ok, call.attempt, call.is_retry) == ("get_order", True, 1, False)
    assert call.arguments == {"order_id": "A-1001"} and call.result == result
    assert call.latency_ms == BUILTIN_TOOLS["get_order"].latency_ms == box.now_ms
    assert call.sink is False and call.fault is None


def test_results_are_copies_so_the_agent_cannot_edit_the_world() -> None:
    box = sandbox()
    first = box.call("get_order", {"order_id": "A-1001"})
    first["status"] = "tampered"
    assert box.call("get_order", {"order_id": "A-1001"})["status"] == "shipped"
    assert box.tool_calls[0].result["status"] == "shipped"  # type: ignore[index,call-overload]


def test_arguments_are_copied_at_call_time() -> None:
    box = sandbox()
    arguments = {"order_id": "A-1001"}
    box.call("get_order", arguments)
    arguments["order_id"] = "changed-later"
    assert box.tool_calls[0].arguments == {"order_id": "A-1001"}


@pytest.mark.parametrize(
    "name, arguments, code",
    [
        ("teleport", {}, "unknown_tool"),
        ("get_order", {}, "invalid_arguments"),
        ("get_order", {"order_id": 7}, "invalid_arguments"),
        ("get_order", {"order_id": "A-1", "x": 1}, "invalid_arguments"),
        ("get_order", ["A-1001"], "invalid_arguments"),
        ("get_order", {"order_id": {1, 2}}, "invalid_arguments"),
        ("get_order", {"order_id": float("nan")}, "invalid_arguments"),
        ("get_order", {"order_id": "x" * 70_000}, "invalid_arguments"),
        ("get_order", {"order_id": "Z-404"}, "not_found"),
        ("calculator", {"expression": "1/0"}, "invalid_request"),
    ],
)
def test_bad_calls_raise_tool_errors_and_are_still_recorded(
    name: str, arguments: object, code: str
) -> None:
    box = sandbox()
    with pytest.raises(ToolError) as caught:
        box.call(name, arguments)  # type: ignore[arg-type]
    assert caught.value.code == code
    (call,) = box.tool_calls
    assert call.ok is False and call.error is not None and call.error.code == code
    assert len(str(call.arguments)) < 1000, "oversized arguments must not be stored whole"


def test_non_string_tool_names_do_not_crash_the_sandbox() -> None:
    box = sandbox()
    with pytest.raises(ToolError) as caught:
        box.call(12345, {})  # type: ignore[arg-type]
    assert caught.value.code == "unknown_tool" and box.tool_calls[0].tool == "12345"


def test_call_budget_stops_a_runaway_loop_even_through_except_exception() -> None:
    box = sandbox(Limits(max_tool_calls=5))
    swallowed = 0
    with pytest.raises(HarnessAbort) as caught:
        while True:
            try:
                box.call("get_order", {"order_id": "missing"})
            except Exception:
                swallowed += 1
    assert swallowed == 5 and len(box.tool_calls) == 5
    assert caught.value.reason == "tool_call_budget" and box.abort_reason == "tool_call_budget"
    assert not isinstance(caught.value, Exception)


def test_closed_sandbox_refuses_everything() -> None:
    box = sandbox()
    box.close("timeout")
    for action in (
        lambda: box.call("get_order", {"order_id": "A-1001"}),
        lambda: box.record_llm_call(model="m", input_tokens=1, output_tokens=1),
        lambda: box.wait(10),
        lambda: box.annotate("late", "value"),
    ):
        with pytest.raises(HarnessAbort) as caught:
            action()
        assert caught.value.reason == "closed"
    assert box.tool_calls == [] and box.annotations == {}


def test_attempts_and_retries_are_tracked_per_identical_call() -> None:
    box = sandbox()
    for _ in range(2):
        with pytest.raises(ToolError):
            box.call("get_order", {"order_id": "Z-1"})
    box.call("get_order", {"order_id": "A-1001"})
    box.call("get_order", {"order_id": "A-1001"})
    flags = [(call.attempt, call.is_retry) for call in box.tool_calls]
    assert flags == [(1, False), (2, True), (1, False), (2, False)]


def test_large_results_are_truncated_in_the_trace_but_not_for_the_agent() -> None:
    environment = Environment.model_validate(
        {"documents": [{"id": "big", "text": "refund " + "x" * 30_000}]}
    )
    box = Sandbox(
        World.from_environment(environment),
        resolve_tools(["search_docs"]),
        limits=Limits(max_result_chars=1000),
    )
    result = box.call("search_docs", {"query": "refund"})
    assert len(result["results"][0]["text"]) > 30_000
    call = box.tool_calls[0]
    assert call.result_truncated and call.result["_truncated"] is True  # type: ignore[index,call-overload]


def test_model_calls_waits_and_usage() -> None:
    box = sandbox(Limits(max_llm_calls=2))
    box.record_llm_call(model="a", input_tokens=100, output_tokens=10, latency_ms=200, note="plan")
    box.wait(500, "back off")
    box.record_llm_call(model="b", input_tokens=50, output_tokens=5, latency_ms=100)
    assert box.now_ms == 800
    usage = box.usage()
    assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (150, 15, 2)
    assert usage.by_model["a"].calls == 1 and usage.by_model["b"].input_tokens == 50
    assert [step.index for step in (*box.llm_calls, *box.waits)] == [0, 2, 1]
    with pytest.raises(HarnessAbort) as caught:
        box.record_llm_call(model="a", input_tokens=1, output_tokens=1)
    assert caught.value.reason == "step_budget"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": -1, "output_tokens": 0},
        {"input_tokens": 0, "output_tokens": 0, "latency_ms": -5},
    ],
)
def test_negative_usage_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        sandbox().record_llm_call(model="m", **kwargs)


@pytest.mark.parametrize("ms", [-1, 3_600_001, float("inf")])
def test_wait_must_be_reasonable(ms: float) -> None:
    with pytest.raises(ValueError, match="between 0 and one hour"):
        sandbox().wait(ms)


def test_annotations_are_json_copies_and_bounded() -> None:
    box = sandbox()
    payload = {"modes": ["a"]}
    box.annotate("info", payload)
    payload["modes"].append("b")
    assert box.annotations == {"info": {"modes": ["a"]}}
    for number in range(100):
        box.annotate(f"key-{number}", number)
    assert len(box.annotations) == 50
    box.annotate("info", "updated")
    assert box.annotations["info"] == "updated"


def test_handler_bugs_become_tool_errors_not_crashes() -> None:
    from dataclasses import replace

    def broken(ctx: object, args: dict) -> object:
        raise RuntimeError("bug in the tool")

    def not_json(ctx: object, args: dict) -> object:
        return {1, 2, 3}

    for handler in (broken, not_json):
        spec = replace(BUILTIN_TOOLS["calculator"], handler=handler)
        box = Sandbox(World(), {"calculator": spec})
        with pytest.raises(ToolError) as caught:
            box.call("calculator", {"expression": "1"})
        assert caught.value.code == "internal_error" and box.tool_calls[0].ok is False
