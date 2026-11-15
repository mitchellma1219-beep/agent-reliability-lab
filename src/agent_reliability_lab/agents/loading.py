"""Resolving an agent from a command-line reference.

``reference``
    the clean simulated agent
``faulty:<mode>[+<mode>...][@<rate>]``
    a simulated agent with deliberately introduced failure modes; ``@0.5`` makes the fault
    appear in about half of the trials
``package.module:attribute`` or ``path/to/file.py:attribute``
    your own agent. The attribute may be an agent object, an agent class or a factory with no
    arguments, or a function ``(ctx) -> str``.

Loading your own agent imports and runs that Python code with your permissions, exactly as
``python file.py`` would.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .adapters import FunctionAgent
from .base import Agent
from .simulated import FAILURE_MODES, faulty_agent, reference_agent


def _from_object(target: Any, spec: str) -> Agent:
    if inspect.isclass(target):
        target = target()
    elif callable(target) and not hasattr(target, "run"):
        parameters = [
            parameter
            for parameter in inspect.signature(target).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if parameters:
            name = getattr(target, "__name__", "function-agent")
            return FunctionAgent(target, name=name.replace("_", "-"))
        target = target()
    if not callable(getattr(target, "run", None)):
        raise ConfigError(f"{spec!r} is not an agent: it has no run(ctx) method")
    if not isinstance(getattr(target, "name", None), str):
        target.name = type(target).__name__
    if not isinstance(getattr(target, "version", None), str):
        target.version = "0"
    agent: Agent = target
    return agent


def _import(location: str) -> Any:
    if location.endswith(".py") or "/" in location or "\\" in location:
        path = Path(location)
        if not path.is_file():
            raise ConfigError(f"no such agent file: {location}")
        module_name = f"arl_user_agent_{abs(hash(str(path.resolve())))}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ConfigError(f"cannot load an agent from {location}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(location)
    except ImportError as exc:
        raise ConfigError(f"cannot import {location!r}: {exc}") from exc


def load_agent(spec: str, name: str | None = None, version: str | None = None) -> Agent:
    """Build an agent from a reference string (see the module documentation)."""
    spec = spec.strip()
    agent: Agent
    if spec == "reference":
        agent = reference_agent()
    elif spec.startswith("faulty:"):
        body, _, rate_text = spec[len("faulty:") :].partition("@")
        modes = [mode for mode in body.split("+") if mode]
        try:
            rate = float(rate_text) if rate_text else 1.0
        except ValueError as exc:
            raise ConfigError(f"invalid activation rate in {spec!r}") from exc
        agent = faulty_agent(*modes, activation_rate=rate)
    elif ":" in spec:
        location, _, attribute = spec.rpartition(":")
        module = _import(location)
        if not hasattr(module, attribute):
            raise ConfigError(f"{location!r} has no attribute {attribute!r}")
        agent = _from_object(getattr(module, attribute), spec)
    else:
        raise ConfigError(
            f"unknown agent {spec!r}: use 'reference', 'faulty:<mode>' "
            f"(modes: {', '.join(FAILURE_MODES)}) or 'module:attribute'"
        )
    if name is not None:
        agent.name = name
    if version is not None:
        agent.version = version
    return agent
