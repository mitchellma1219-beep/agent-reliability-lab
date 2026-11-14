"""The agent protocol, the adapters, status inference and loading agents by reference."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_reliability_lab import (
    Agent,
    AgentContext,
    AgentResponse,
    FunctionAgent,
    ScriptedAgent,
    ToolError,
    faulty_agent,
    load_builtin_suite,
    reference_agent,
    run_case,
    run_suite,
)
from agent_reliability_lab.agents import (
    FAILURE_MODES,
    Behaviour,
    Message,
    ModelTurn,
    ScriptedCall,
    SimulatedAgent,
    ToolLoopAgent,
    ToolRequest,
    infer_status,
)
from agent_reliability_lab.agents.loading import load_agent
from agent_reliability_lab.errors import ConfigError
from agent_reliability_lab.models import RunConfig
from agent_reliability_lab.tools import ToolInfo
from helpers import case_data, make_case, make_suite

ANSWER = "Refunds are available within 30 days of delivery."


# --------------------------------------------------------------------------- protocol and context


def test_the_protocol_is_structural() -> None:
    class Mine:
        name = "mine"
        version = "0.1"

        def run(self, ctx: AgentContext) -> str:
            return ANSWER

    assert isinstance(Mine(), Agent) and isinstance(reference_agent(), Agent)
    assert isinstance(FunctionAgent(lambda ctx: ""), Agent) and isinstance(ScriptedAgent(), Agent)
    assert not isinstance(object(), Agent)
    result = run_case(make_case(), Mine())
    assert result.passed and (result.trace.agent_name, result.trace.agent_version) == (
        "mine",
        "0.1",
    )


def test_the_agent_sees_the_task_but_never_the_expected_answer() -> None:
    seen: dict[str, object] = {}

    def probe(ctx: AgentContext) -> str:
        seen["fields"] = sorted(type(ctx.task).model_fields)
        seen["dump"] = ctx.task.model_dump_json()
        seen["tools"] = ctx.tools.names()
        seen["spec"] = ctx.tools.spec("get_order")
        seen["missing"] = ctx.tools.spec("no_such_tool")
        seen["attributes"] = [name for name in vars(ctx) if "sandbox" not in name.lower()]
        return ANSWER

    case = make_case(
        expected={"answer": {"contains_all": ["30 days"], "not_contains": ["SECRET-ORACLE-TEXT"]}},
        attack={
            "kind": "direct",
            "goal": "SECRET-ATTACK-GOAL",
            "indicators": {"output_contains": ["pwned"]},
        },
    )
    assert run_case(case, FunctionAgent(probe)).passed
    assert seen["fields"] == ["context", "input", "instructions", "output_schema", "sim"]
    assert "SECRET-ORACLE-TEXT" not in str(seen["dump"]) and "SECRET-ATTACK-GOAL" not in str(
        seen["dump"]
    )
    assert seen["tools"] == case.tools and seen["missing"] is None
    spec = seen["spec"]
    assert (
        isinstance(spec, ToolInfo)
        and spec.name == "get_order"
        and "order_id" in str(spec.parameters)
    )
    assert seen["attributes"] == ["task", "tools", "seed"]


def test_tools_accept_a_dict_keywords_or_both() -> None:
    results: list[object] = []

    def agent(ctx: AgentContext) -> str:
        results.append(ctx.tools.call("get_order", {"order_id": "A-1001"})["status"])
        results.append(ctx.tools.call("get_order", order_id="A-1002")["status"])
        results.append(
            ctx.tools.call("calculator", {"expression": "1 + 1"}, expression="2 * 21")["value"]
        )
        try:
            ctx.tools.call("get_order", order_id="A-9999")
        except ToolError as exc:
            results.append(exc.code)
        return ANSWER

    result = run_case(make_case(), FunctionAgent(agent))
    assert results == ["shipped", "delivered", 42, "not_found"]
    assert [call.arguments for call in result.trace.tool_calls][2] == {"expression": "2 * 21"}


def test_context_helpers_report_usage_time_and_reproducible_randomness() -> None:
    draws: list[tuple[float, float, float]] = []

    def agent(ctx: AgentContext) -> AgentResponse:
        before = ctx.elapsed_ms
        ctx.record_llm_call(
            input_tokens=1000, output_tokens=200, latency_ms=350.0, model="sim-1", note="plan"
        )
        ctx.wait(500.0, "back-off")
        ctx.annotate("stage", {"step": 2})
        draws.append((ctx.uniform("a"), ctx.uniform("a"), ctx.uniform("b")))
        return AgentResponse(
            output=ANSWER, status="completed", metadata={"elapsed": ctx.elapsed_ms - before}
        )

    first = run_case(make_case(), FunctionAgent(agent), RunConfig(seed=3))
    again = run_case(make_case(), FunctionAgent(agent), RunConfig(seed=3))
    other = run_case(make_case(), FunctionAgent(agent), RunConfig(seed=4), trial=1)
    assert first.trace.usage.input_tokens == 1000 and first.trace.usage.output_tokens == 200
    assert first.trace.latency_ms == 850.0 and first.trace.waits[0].reason == "back-off"
    assert first.trace.agent_metadata == {"stage": {"step": 2}, "elapsed": 850.0}
    assert first.estimated_cost_usd == pytest.approx(1000 / 1e6 * 1.0 + 200 / 1e6 * 5.0)
    (a1, a2, b), second, third = draws
    assert a1 == a2 != b and 0.0 <= a1 < 1.0 and second == draws[0] and third != draws[0]
    assert first.seed == again.seed != other.seed


# --------------------------------------------------------------------------- status inference


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Your order A-1001 has shipped.", "completed"),
        ("", "completed"),
        ("I was unable to complete the refund.", "failed"),
        ("The order service is temporarily unavailable, so nothing was changed.", "failed"),
        ("An error occurred while looking that up; please try again later.", "failed"),
        ("I can't share that information.", "refused"),
        ("I am not permitted to issue refunds above 500 USD.", "refused"),
        ("That would be against our policy.", "refused"),
        ("I could not find an order with that number.", "abstained"),
        ("The documents do not mention a warranty for that product.", "abstained"),
        ("There is no record of that customer.", "abstained"),
        ("Which order do you mean, A-1001 or A-1002?", "needs_clarification"),
        ("Could you confirm the order number?", "needs_clarification"),
        ("I could not complete the refund. Which order did you mean?", "failed"),
        ("I can't share that. Which order do you mean?", "refused"),
        ("I could not find it. Could you confirm the order number?", "abstained"),
    ],
)
def test_status_inference(text: str, status: str) -> None:
    assert infer_status(text) == status


def test_a_declared_status_always_wins_over_the_wording() -> None:
    case = make_case(expected={"status": "needs_clarification"})
    declared = run_case(
        case,
        FunctionAgent(lambda ctx: AgentResponse(output="Order?", status="needs_clarification")),
    )
    assert declared.passed and declared.trace.status_source == "declared"
    inferred = run_case(
        case, FunctionAgent(lambda ctx: AgentResponse(output="Which order do you mean?"))
    )
    assert inferred.passed and inferred.trace.status_source == "inferred"
    plain = run_case(case, FunctionAgent(lambda ctx: "Order A-1001 has shipped."))
    assert not plain.passed and plain.trace.status == "completed"


# --------------------------------------------------------------------------- adapters


def test_scripted_agent_replays_calls_ignores_tool_errors_and_reports_usage() -> None:
    agent = ScriptedAgent(
        [
            ("get_order", {"order_id": "A-9999"}),
            ScriptedCall("get_order", {"order_id": "A-1001"}),
            ("nope", {}),
        ],
        ANSWER,
        "completed",
        name="replay",
        version="3",
        input_tokens=10,
        output_tokens=5,
        latency_ms=7.0,
    )
    trace = run_case(make_case(), agent).trace
    assert [(call.tool, call.ok) for call in trace.tool_calls] == [
        ("get_order", False),
        ("get_order", True),
        ("nope", False),
    ]
    assert [call.error.code for call in trace.tool_calls if call.error] == [
        "not_found",
        "unknown_tool",
    ]
    assert (trace.usage.input_tokens, trace.usage.output_tokens, trace.usage.llm_calls) == (
        10,
        5,
        1,
    )
    assert trace.llm_calls[0].model == "scripted" and trace.output == ANSWER
    silent = run_case(make_case(), ScriptedAgent(output=ANSWER)).trace
    assert silent.llm_calls == [] and silent.status_source == "inferred"


class FakeModel:
    """A scripted chat model: returns the prepared turns one after another."""

    def __init__(self, turns: Sequence[ModelTurn]) -> None:
        self.turns = list(turns)
        self.seen: list[list[Message]] = []
        self.tools: list[str] = []

    def __call__(self, messages: Sequence[Message], tools: Sequence[ToolInfo]) -> ModelTurn:
        self.seen.append(list(messages))
        self.tools = [tool.name for tool in tools]
        return self.turns[min(len(self.seen), len(self.turns)) - 1]


def test_tool_loop_agent_runs_tools_until_the_model_answers() -> None:
    model = FakeModel(
        [
            ModelTurn("Let me look.", [ToolRequest("get_order", {"order_id": "A-1001"})], 100, 20),
            ModelTurn(
                "",
                [
                    ToolRequest("get_order", {"order_id": "A-9999"}),
                    ToolRequest("search_docs", {"query": "refund"}),
                ],
                150,
                20,
            ),
            ModelTurn("Order A-1001 has shipped. " + ANSWER, (), 200, 30, "completed"),
        ]
    )
    case = make_case(
        instructions=[
            {"id": "tone", "text": "Be brief.", "priority": "system"},
            {"id": "cite", "text": "Mention the order id.", "priority": "user"},
        ],
        context=[{"id": "note", "title": "Note", "text": "Pat is a long-standing customer."}],
        output_schema={"type": "object"},
        expected={
            "answer": {"contains_all": ["shipped", "30 days"]},
            "tool_calls": {"required": [{"tool": "get_order"}]},
        },
    )
    agent = ToolLoopAgent(model, name="loop", version="1", model_name="sim-1")
    result = run_case(case, agent)
    trace = result.trace
    assert [call.tool for call in trace.tool_calls] == ["get_order", "get_order", "search_docs"]
    assert (trace.usage.llm_calls, trace.usage.input_tokens, trace.usage.output_tokens) == (
        3,
        450,
        70,
    )
    assert trace.status == "completed" and trace.status_source == "declared"
    assert result.estimated_cost_usd == pytest.approx(450 / 1e6 + 70 * 5 / 1e6)
    opening = model.seen[0]
    assert [message.role for message in opening] == ["system", "user"]
    assert opening[0].content == "Be brief."
    user = opening[1].content
    assert user.startswith(case.input) and "Mention the order id." in user
    assert "[document note] Note\nPat is a long-standing customer." in user
    assert 'Answer with JSON matching this schema: {"type": "object"}' in user
    final = model.seen[-1]
    assert [message.role for message in final] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
    ]
    assert (
        final[3].content.startswith('get_order: {"') and '"status": "shipped"' in final[3].content
    )
    assert '"error": "not_found"' in final[4].content and '"retryable": false' in final[4].content
    assert model.tools == case.tools


def test_tool_loop_agent_gives_up_after_max_turns_without_running_away() -> None:
    model = FakeModel([ModelTurn("", [ToolRequest("calculator", {"expression": "1+1"})], 1, 1)])
    result = run_case(make_case(), ToolLoopAgent(model, max_turns=4))
    assert len(model.seen) == 4 and len(result.trace.tool_calls) == 4
    assert result.trace.status == "failed" and "allowed number of steps" in result.trace.output
    assert result.trace.errors == [] and not result.passed


def test_tool_loop_agent_without_instructions_sends_only_the_user_message() -> None:
    model = FakeModel([ModelTurn(ANSWER)])
    assert run_case(make_case(), ToolLoopAgent(model)).passed
    assert [message.role for message in model.seen[0]] == ["user"]
    assert model.seen[0][0].content == make_case().input


# --------------------------------------------------------------------------- simulated agents


def test_simulated_agents_need_a_sim_block() -> None:
    result = run_case(make_case(), reference_agent())
    assert not result.passed and result.trace.errors[0].kind == "agent_exception"
    assert "simulated agents need a 'sim' block" in result.trace.errors[0].message
    assert result.failure_categories == ["agent_crash", "wrong_outcome"]


def test_faulty_agent_construction() -> None:
    agent = faulty_agent("no_retry", "leaks_canary")
    assert agent.name == "faulty-no-retry+leaks-canary" and agent.version == "1.0.0"
    assert agent.behaviour.active_modes() == ["no_retry", "leaks_canary"]
    assert faulty_agent("slow", name="v2", version="2.0.0").name == "v2"
    assert Behaviour().active_modes() == [] and reference_agent().behaviour == Behaviour()
    assert sorted(FAILURE_MODES) == sorted(field for field in Behaviour.__dataclass_fields__)
    with pytest.raises(ConfigError, match="unknown failure mode"):
        faulty_agent("no_such_mode")
    with pytest.raises(ConfigError, match="unknown failure mode"):
        faulty_agent()
    for rate in (-0.1, 1.5):
        with pytest.raises(ConfigError, match="between 0 and 1"):
            SimulatedAgent(activation_rate=rate)


def test_activation_rate_makes_a_fault_intermittent_but_reproducible() -> None:
    suite = load_builtin_suite("faults")
    config = RunConfig(seed=11, trials=6)
    flaky = faulty_agent("no_retry", activation_rate=0.5)
    first = run_suite(suite, flaky, config)
    second = run_suite(suite, faulty_agent("no_retry", activation_rate=0.5), config)
    assert first.fingerprint() == second.fingerprint()
    outcomes = [result.passed for result in first.by_case()["fault-timeout-once"]]
    assert True in outcomes and False in outcomes
    never = run_suite(suite, faulty_agent("no_retry", activation_rate=0.0), RunConfig(seed=11))
    assert all(result.passed for result in never.results)


# --------------------------------------------------------------------------- loading by reference


def test_load_builtin_agents() -> None:
    assert load_agent("reference").name == "reference-agent"
    assert load_agent(" reference ", name="mine", version="9.9").version == "9.9"
    faulty = load_agent("faulty:no_retry+slow@0.25")
    assert isinstance(faulty, SimulatedAgent) and faulty.activation_rate == 0.25
    assert faulty.behaviour.active_modes() == ["no_retry", "slow"]
    assert load_agent("faulty:hallucinates", name="x").name == "x"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("unknown", "unknown agent 'unknown'"),
        ("", "unknown agent ''"),
        ("faulty:", "unknown failure mode"),
        ("faulty:nope", "unknown failure mode"),
        ("faulty:no_retry@often", "invalid activation rate"),
        ("faulty:no_retry@7", "between 0 and 1"),
        ("no_such_package_arl.module:agent", "cannot import"),
        ("json:no_such_attribute", "has no attribute"),
        ("json:decoder", "is not an agent"),
        ("missing/agent.py:agent", "no such agent file"),
    ],
)
def test_load_agent_errors(spec: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_agent(spec)


AGENT_FILE = '''
from agent_reliability_lab import AgentResponse

ANSWER = "Refunds are available within 30 days of delivery."


def answer(ctx):
    """A plain function taking the context."""
    return ANSWER


class Support:
    name = "support"
    version = "2.1.0"

    def run(self, ctx):
        return AgentResponse(output=ANSWER, status="completed")


class Nameless:
    def run(self, ctx):
        return ANSWER


instance = Support()


def build():
    """A factory with no arguments."""
    return Support()


def build_with_default(flag=True):
    return Support()


not_an_agent = 42
'''


@pytest.fixture
def agent_file(tmp_path: Path) -> Path:
    path = tmp_path / "my_agent.py"
    path.write_text(textwrap.dedent(AGENT_FILE), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("attribute", "name", "version"),
    [
        ("answer", "answer", "0"),
        ("Support", "support", "2.1.0"),
        ("instance", "support", "2.1.0"),
        ("build", "support", "2.1.0"),
        ("build_with_default", "support", "2.1.0"),
        ("Nameless", "Nameless", "0"),
    ],
)
def test_load_an_agent_from_a_file(
    agent_file: Path, attribute: str, name: str, version: str
) -> None:
    agent = load_agent(f"{agent_file}:{attribute}")
    assert (agent.name, agent.version) == (name, version)
    assert run_case(make_case(), agent).passed


def test_load_agent_from_a_file_reports_problems(agent_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is not an agent"):
        load_agent(f"{agent_file}:not_an_agent")
    with pytest.raises(ConfigError, match="has no attribute 'missing'"):
        load_agent(f"{agent_file}:missing")
    with pytest.raises(ConfigError, match="no such agent file"):
        load_agent(f"{tmp_path}:answer")


def test_load_an_agent_from_an_importable_module(
    agent_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(agent_file.parent))
    monkeypatch.delitem(sys.modules, "my_agent", raising=False)
    agent = load_agent("my_agent:Support", version="3.0.0")
    assert (agent.name, agent.version) == ("support", "3.0.0")
    sys.modules.pop("my_agent", None)


def test_a_loaded_agent_runs_a_whole_suite(agent_file: Path) -> None:
    suite = make_suite(case_data(), case_data(id="case-2", input="Refund window?"))
    run = run_suite(suite, load_agent(f"{agent_file}:answer"), RunConfig(trials=2))
    assert [result.passed for result in run.results] == [True] * 4 and run.agent_name == "answer"
