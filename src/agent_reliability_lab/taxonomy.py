"""Error taxonomy: turns evaluator findings into a small set of failure categories."""

from __future__ import annotations

from .models import Evaluation

EVALUATOR_CATEGORY: dict[str, str] = {
    "task_success": "task_failure",
    "tool_call_accuracy": "tool_use_error",
    "instruction_following": "instruction_violation",
    "groundedness": "ungrounded_answer",
    "hallucination": "hallucination",
    "output_schema": "schema_violation",
    "workflow": "workflow_incomplete",
    "injection_resistance": "prompt_injection",
    "canary_leak": "data_exfiltration",
    "recovery": "recovery_failure",
    "latency_budget": "latency_budget_exceeded",
    "cost_budget": "cost_budget_exceeded",
    "containment": "sandbox_violation",
}

CODE_CATEGORY: dict[str, str] = {
    "timeout": "timeout",
    "agent_exception": "agent_crash",
    "agent_exit": "agent_crash",
    "harness_abort": "runaway_loop",
    "invalid_output": "unusable_output",
    "wrong_status": "wrong_outcome",
    "missing_call": "tool_selection_error",
    "unexpected_call": "tool_selection_error",
    "unknown_tool": "tool_selection_error",
    "wrong_order": "tool_order_error",
    "invalid_arguments": "tool_argument_error",
    "forbidden_call": "forbidden_tool_call",
    "too_many_calls": "tool_overuse",
    "claimed_action_not_performed": "false_completion_claim",
    "evaluator_error": "evaluator_error",
}

DESCRIPTIONS: dict[str, str] = {
    "task_failure": "the answer is wrong or incomplete",
    "wrong_outcome": "the agent ended in the wrong state (for example answered instead of asking)",
    "tool_use_error": "tool calls did not match what the task needs",
    "tool_selection_error": "a needed tool was not called, or an unneeded one was",
    "tool_argument_error": "a tool was called with invalid arguments",
    "tool_order_error": "tools were called in the wrong order",
    "forbidden_tool_call": "a tool the task forbids was called",
    "tool_overuse": "more tool calls than the task allows",
    "instruction_violation": "a system or user instruction was not followed",
    "ungrounded_answer": "a statement is not supported by the evidence the agent saw",
    "hallucination": "a specific value in the answer has no source",
    "false_completion_claim": "the answer claims an action that never happened",
    "schema_violation": "the output is not valid JSON for the required schema",
    "workflow_incomplete": "a multi-step workflow was not carried out fully or in order",
    "prompt_injection": "an injected instruction was obeyed",
    "data_exfiltration": "a canary string reached the output or a sink",
    "recovery_failure": "a tool fault was not handled gracefully or safely",
    "latency_budget_exceeded": "the run took longer than its latency budget",
    "cost_budget_exceeded": "the run cost more than its budget (estimated)",
    "sandbox_violation": "the agent tried to leave the sandbox",
    "timeout": "the agent did not answer within the wall-clock timeout",
    "agent_crash": "the agent raised an exception or tried to exit",
    "runaway_loop": "the harness stopped a runaway loop",
    "unusable_output": "the agent returned something that is not usable text",
    "evaluator_error": "an evaluator could not run (counted as a failure)",
}


def failure_categories(evaluations: list[Evaluation]) -> list[str]:
    """Sorted, de-duplicated failure categories for the failed evaluations."""
    categories: set[str] = set()
    for evaluation in evaluations:
        if evaluation.passed:
            continue
        specific = {
            CODE_CATEGORY[item.code] for item in evaluation.failures() if item.code in CODE_CATEGORY
        }
        categories |= specific or {
            EVALUATOR_CATEGORY.get(evaluation.evaluator, evaluation.evaluator)
        }
    return sorted(categories)
