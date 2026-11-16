# Agent Reliability Lab

A testing and evaluation framework for autonomous AI agents. It runs an agent against
scenario suites built on simulated tools and measures whether the agent is accurate, follows
instructions, resists prompt injection, recovers from tool faults and stays inside its cost and
latency budgets. It also compares two versions of an agent and gives a regression verdict.

It ships as a Python package, the `arl` command-line tool, a Streamlit dashboard and an Agent
Skill (`skills/agent-reliability-lab`). Everything runs offline and deterministically from a
seed; the optional LLM judge is off by default.

## What it checks

Four built-in suites, 60 cases in total, all against simulated tools with fictional data:

| Suite | Cases | What it exercises |
|---|---|---|
| `core` | 17 | lookups, arithmetic, questions answered from documents, structured output, instruction following, multi-step workflows |
| `robustness` | 18 | conflicting instructions, questions the documents cannot answer, ambiguous requests, long contexts with distractors, unexpected types and malformed JSON from tools |
| `faults` | 15 | timeouts, rate limits, transient and permanent errors, silently wrong responses, partial completion, retry exhaustion, dependency failures |
| `injection` | 10 | direct and indirect prompt injection (in the request, in retrieved documents, in tool results), with a canary planted in the system instructions |

Thirteen deterministic evaluators score every run: `task_success`, `tool_call_accuracy`,
`instruction_following`, `groundedness`, `hallucination`, `output_schema`, `workflow`,
`injection_resistance`, `canary_leak`, `recovery`, `latency_budget`, `cost_budget` and
`containment`. None of them calls a model. Their known false positives and false negatives
are listed in [skills/agent-reliability-lab/references/evaluators.md](skills/agent-reliability-lab/references/evaluators.md).

## Do the evaluators catch real mistakes?

`arl matrix` runs a reference agent and fifteen deliberately faulty agents (one per failure
mode: follows injections, hallucinates, calls the wrong tool, ignores instructions, never
retries, retries forever, leaks the canary, stops half way, is slow, is expensive, and so on)
over every suite, and checks that each fault is caught by the evaluator designed for it,
without false alarms on the reference agent. With seed 0 the reference agent passes 60/60
cases and every failure mode is detected with no escapes and no false alarms.

## Install

```bash
git clone https://github.com/mitchellma1219-beep/agent-reliability-lab.git
cd agent-reliability-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"     # ".[dev]" without the dashboard, "." for the library alone
```

Python 3.12 or newer (the containment guard relies on thread audit events that Python 3.11
does not emit). Runtime dependencies: pydantic, jsonschema, numpy, PyYAML. The `ui` extra
adds Streamlit, Plotly and pandas.

## Command line

```bash
arl scenarios                                   # the built-in suites and their cases
arl run --suite core --agent reference --name v1
arl run --suite all --agent faulty:hallucinates --name v2 --fail-under 0
arl compare v1 v2                               # regression verdict with reasons
arl report v1 --format md                       # failures grouped by error taxonomy
arl run --suite core --agent reference --faults flaky-tools   # random tool faults
arl matrix                                      # detection matrix (see above)
arl calibrate-judge                             # measure a judge against labelled answers
arl demo && arl dashboard                       # demo runs and the Streamlit dashboard
```

Your own agent is passed as `package.module:attribute` or `path/to/file.py:attribute`. It is
an object with `name`, `version` and `run(ctx)` (or a function `(ctx) -> str`), calls tools
through `ctx.tools.call(...)` and reports model usage with `ctx.record_llm_call(...)`. Your own
cases go in a YAML or JSON suite file; the format is in
[skills/agent-reliability-lab/references/suite-format.md](skills/agent-reliability-lab/references/suite-format.md).

Exit code 1 means a gate failed (cases below `--fail-under`, or a regression); exit code 2
means the input was rejected, with the reason on stderr.

## Limits

- Passing a suite shows how an agent behaved on these synthetic cases. It does not prove the
  agent is reliable or secure.
- The guard that blocks network, subprocess and file writes from agent code is an in-process
  tripwire, not a sandbox. Do not run agent code you do not trust.
- Cost figures are estimates from reported tokens and a price table; pass your own prices
  with `--prices`.
- The LLM judge sends data to an external API and costs money; it runs only with
  `--allow-llm-judge`, and verdicts it is unsure about are kept out of pass rates.

## Related project: CommerceGuard

[projects/commerceguard-v4](projects/commerceguard-v4/README.md) applies the same idea, that an
agent should not grade its own work, to digital advertising. Four roles (Data Analyst,
Strategy Planner, Scenario Simulator, Independent Auditor) review weekly Facebook and Google Ads
spend and propose a budget shift, and the auditor rejects proposals that break hard rules. It is
read-only, with a 40-week walk-forward backtest and 17 tests.

## Quality checks

CI runs `ruff`, `mypy --strict`, the test suite with a coverage gate of 85% on Python 3.12 and
3.13, plus `bandit` and `pip-audit`.

## License

MIT
