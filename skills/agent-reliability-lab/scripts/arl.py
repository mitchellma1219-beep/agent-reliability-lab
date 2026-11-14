#!/usr/bin/env python3
"""Skill entry point: forwards to the agent_reliability_lab command line."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from agent_reliability_lab.cli import main
except ModuleNotFoundError:  # running from a repository checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from agent_reliability_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
