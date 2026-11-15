"""Demo runs: the release history of a fictional agent, generated deterministically.

The dashboard and the README use these runs so that everything can be tried offline. They
are produced by the bundled simulated agents on the bundled suites with fixed seeds and
fixed timestamps, so the same code always yields the same runs. They describe no real
product and no real measurements.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from .agents.simulated import faulty_agent, reference_agent
from .models import RunConfig, RunResult
from .runner import run_suite
from .scenarios import ALL, load_builtin_suite
from .storage import RunStore

DEMO_AGENT = "demo-support-agent"


@dataclass(frozen=True)
class DemoVersion:
    version: str
    modes: tuple[str, ...]
    activation_rate: float
    created_at: str
    story: str


DEMO_HISTORY: tuple[DemoVersion, ...] = (
    DemoVersion(
        "0.8.0",
        ("no_retry", "ignores_instruction"),
        1.0,
        "2026-09-01T09:00:00Z",
        "First prototype: gives up on the first tool error and ignores system instructions.",
    ),
    DemoVersion("0.9.0", ("no_retry",), 1.0, "2026-09-15T09:00:00Z", "Instruction handling fixed."),
    DemoVersion("1.0.0", (), 1.0, "2026-10-01T09:00:00Z", "Bounded retries added. Clean."),
    DemoVersion(
        "1.1.0",
        ("follows_injection",),
        1.0,
        "2026-10-15T09:00:00Z",
        "A refactor made the agent obey instructions found in documents and tool results.",
    ),
    DemoVersion(
        "1.2.0",
        ("hallucinates",),
        0.5,
        "2026-11-01T09:00:00Z",
        "Injection fixed, but the agent now sometimes invents answers (flaky).",
    ),
    DemoVersion("1.3.0", (), 1.0, "2026-11-10T09:00:00Z", "Clean again."),
)


def demo_run_id(version: str) -> str:
    return f"{DEMO_AGENT}-{version}"


def generate_demo_runs(trials: int = 2, seed: int = 0, suite: str = ALL) -> list[RunResult]:
    """Run every demo version against a built-in suite. Deterministic for a given seed."""
    loaded = load_builtin_suite(suite)
    config = RunConfig(seed=seed, trials=trials)
    runs: list[RunResult] = []
    for item in DEMO_HISTORY:
        agent = (
            faulty_agent(
                *item.modes,
                name=DEMO_AGENT,
                version=item.version,
                activation_rate=item.activation_rate,
            )
            if item.modes
            else reference_agent(DEMO_AGENT, item.version)
        )
        created = dt.datetime.strptime(item.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
        run = run_suite(loaded, agent, config, run_id=demo_run_id(item.version), created_at=created)
        run.notes = f"Demo data. {item.story}"
        runs.append(run)
    return runs


def write_demo_runs(out_dir: str | Path, trials: int = 2, seed: int = 0) -> list[Path]:
    """Generate the demo runs and store them in ``out_dir`` (existing demo runs are replaced)."""
    store = RunStore(out_dir)
    return [store.save(run, overwrite=True) for run in generate_demo_runs(trials, seed)]
