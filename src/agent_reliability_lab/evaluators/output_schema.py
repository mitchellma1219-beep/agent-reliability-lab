"""Output schema validation: the final answer must be JSON that satisfies the case's schema."""

from __future__ import annotations

import json

from ..models import Case, Evaluation, Trace
from ..schema import validate
from .base import EvalContext, finding, verdict


class OutputSchema:
    """Strict: the whole output must be one JSON document, with no prose or code fence.

    Only applied when the agent claims to have completed the task. If it asked a question,
    refused or failed explicitly, there is no payload to validate and ``task_success``
    decides whether that outcome was acceptable.
    """

    name = "output_schema"

    def applies(self, case: Case, trace: Trace) -> bool:
        return case.output_schema is not None and trace.status == "completed"

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        schema = case.output_schema or {}
        try:
            payload = json.loads(trace.output)
        except RecursionError:
            return verdict(self.name, [finding("not_json", "output is nested too deeply to parse")])
        except ValueError as exc:
            return verdict(self.name, [finding("not_json", f"output is not valid JSON: {exc}")])
        problems = validate(schema, payload)
        return verdict(
            self.name,
            [finding("schema_violation", problem) for problem in problems],
            details={"violations": len(problems)},
        )
