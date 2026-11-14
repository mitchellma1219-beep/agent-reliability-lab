---
name: agent-reliability-lab
description: Test and evaluate an autonomous AI agent for reliability, accuracy, prompt-injection resistance, fault recovery, cost and latency, and compare two versions of an agent for regressions. Use when the user wants to check whether an agent is ready to ship, run adversarial or fault-injection scenarios against an agent, write a golden dataset of test cases for an agent, investigate why an agent run failed, calibrate an LLM judge, or decide whether a new agent version is worse than the previous one.
license: MIT
compatibility: Requires Python 3.11+ with pydantic, jsonschema, numpy and PyYAML. The dashboard additionally needs streamlit, plotly and pandas.
---

# Agent Reliability Lab

Every number you report must come from the bundled script. Do not estimate pass rates,
significance, cost or latency yourself, and do not describe a run you have not executed.

## Run

`python scripts/arl.py <command> ...` (path relative to this skill directory). The script uses
the installed `agent_reliability_lab` package, or the repository `src/` when run from a
checkout.

| Task | Command |
|---|---|
| See the built-in suites and their cases | `scenarios` or `scenarios SUITE` |
| Run an agent against a suite | `run --suite SUITE --agent AGENT [--trials N] [--seed S] [--name NAME]` |
| Run under random tool faults | `run --suite SUITE --agent AGENT --faults flaky-tools` |
| Report on a stored run | `report RUN [--format md\|json] [--out FILE]` |
| Compare two versions | `compare BASELINE CANDIDATE [--strict] [--out FILE]` |
| List stored runs | `list` |
| Check that the evaluators catch known mistakes | `matrix` |
| Measure a judge against labelled answers | `calibrate-judge [--dataset FILE]` |

`SUITE` is `core`, `injection`, `robustness`, `faults`, `all`, or a path to a `.yaml` / `.json`
suite file. `AGENT` is `reference`, `faulty:<mode>`, or the user's own agent as
`package.module:attribute` or `path/to/file.py:attribute`. Runs are stored in `runs/`
(`--runs-dir` changes that). Exit code 1 means a gate failed (cases failed, or a regression
was found); exit code 2 means the input was rejected, with the reason on stderr. Relay it.

## Workflow

1. **Connect the agent.** The agent must be a Python object with `name`, `version` and
   `run(ctx)`, or a function `(ctx) -> str`. It calls tools only through `ctx.tools.call(...)`
   and reports model usage with `ctx.record_llm_call(...)`. See `references/suite-format.md`
   for a minimal adapter. If the user has no adapter yet, write one with them first.
2. **Start with `core`**, then `robustness`, `faults` and `injection`. Use `--trials 3` or
   more for a non-deterministic agent, so flaky cases show up.
3. **Read the failures before the summary.** For each failing case report the evaluator, the
   finding and what the agent actually did (`report RUN`). Group failures by the error
   taxonomy in the report rather than listing sixty lines.
4. **For a new version, run the same suite with the same seed and trials and use `compare`.**
   Report the verdict word for word together with its reasons. `needs_review` is not a pass.
5. **For the user's own tasks, write a suite file** (`references/suite-format.md`) and
   validate it by running it. A case with no real check is rejected on load; fix the case,
   do not delete the check.

## Rules

- Passing a suite shows how the agent behaved on those synthetic cases. Never say or imply
  that it proves the agent is reliable or secure. Say what was tested and what was not.
- The suites target simulated tools only. Do not point the lab, or an agent under test, at
  real systems, real customer data or real credentials.
- Cost figures are estimates from reported tokens and the price table in use. Say so, and
  ask the user for their real prices (`--prices FILE`) before quoting money.
- The guard that blocks network, subprocess and file writes from agent code is a tripwire,
  not a sandbox. Only use `--allow-network` or `--no-guard` when the user asks for it.
- The LLM judge is off by default. Do not enable it (`--allow-llm-judge`) unless the user
  explicitly asks; it sends data to an external API and costs money. Calibrate a judge
  before trusting it, and keep verdicts marked `needs_human_review` out of any pass rate.
- If a result surprises you, check `references/evaluators.md` for the evaluator's known
  false positives and false negatives before blaming the agent.

More detail: `references/evaluators.md`, `references/suite-format.md`,
`references/interpreting-results.md`.
