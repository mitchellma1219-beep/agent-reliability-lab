# Interpreting results

## One run

- **Pass rate.** A case passes only if every trial passed every applicable evaluator. The
  Wilson 95% interval treats the suite's cases as a sample of the tasks the agent will meet.
  With 60 cases and no failure the interval is still about 94% to 100%: a perfect score on a
  small suite does not show a failure rate below a few percent.
- **Task success rate** uses only `task_success`. An agent can solve the task and still fail
  the case, for example by leaking a canary or blowing the budget.
- **Flaky case.** Passed some trials and failed others. Needs more trials, not an average.
- **Error taxonomy.** Start here when many cases fail: one root cause usually explains a
  whole category (`recovery_failure`, `prompt_injection`, `tool_argument_error`, ...).
- **Latency** is accounted time: simulated tool latency, model latency as reported by the
  agent, and waits. It is not wall-clock time of the test run.
- **Cost** is an estimate: reported tokens times the price table in use. The default table
  contains illustrative prices for the simulated model only.

## Comparing two runs

`compare BASELINE CANDIDATE` pairs cases by id.

| Verdict | Meaning | Exit code |
|---|---|---|
| `regression` | more cases stopped passing than started, and the exact McNemar test is significant at alpha | 1 |
| `needs_review` | at least one case that passed consistently no longer does, or cost / latency rose materially, but a reliability drop is not statistically established | 0 (1 with `--strict`) |
| `improved` | nothing got worse and something got better | 0 |
| `no_change` | no case changed outcome; cost and latency did not move materially | 0 |

- Identical runs always give `no_change` (p = 1).
- With a deterministic agent and the same seed, any newly failing case is a real behaviour
  change even if p is above alpha. The test matters for non-deterministic agents, where a
  single flip can be noise. Use several trials there.
- Cost and latency changes are reported only when the 95% bootstrap interval of the mean
  per-case difference excludes zero and the relative change is at least the tolerance
  (10% by default).
- Compare like with like: same suite version, seed, trials, fault profile and price table.
  The report warns when they differ.

## What a pass does not mean

A suite checks specific, synthetic situations with deterministic rules. A pass means none of
those rules fired. It is not evidence that the agent resists attacks that are not in the
suite, behaves the same on real data, or is safe to connect to real systems.
