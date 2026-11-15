"""The in-process containment tripwire.

Every hostile action in this file is chosen so that it is harmless even if the guard failed
to block it: file operations only touch pytest's temporary directory, network attempts only
name the loopback address, and the only programs started are ``python -c pass``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_reliability_lab import AgentContext, FunctionAgent, guard, run_case
from agent_reliability_lab.guard import MAX_EVENTS, GuardViolation
from agent_reliability_lab.models import CaseResult, GuardConfig, RunConfig, SecurityEvent
from helpers import make_case

ANSWER = "Refunds are available within 30 days of delivery."


def guarded(
    action: Callable[[], Any], config: GuardConfig | None = None, workdir: Path | None = None
) -> tuple[Any, list[SecurityEvent]]:
    """Run ``action`` in a guarded thread; return what it returned or raised, and the events."""
    events: list[SecurityEvent] = []
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = action()
        except BaseException as exc:
            box["value"] = exc

    thread = threading.Thread(target=guard.wrap(target, config or GuardConfig(), workdir, events))
    thread.start()
    thread.join(30)
    assert not thread.is_alive()
    return box.get("value"), events


def run_agent(
    function: Callable[[AgentContext], Any], config: RunConfig | None = None
) -> CaseResult:
    return run_case(make_case(), FunctionAgent(function), config)


def blocked(value: Any, events: list[SecurityEvent], kind: str) -> bool:
    return isinstance(value, GuardViolation) and [event.kind for event in events] == [kind]


# --------------------------------------------------------------------------- filesystem


def test_writes_outside_the_sandbox_directory_are_blocked(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    target = tmp_path / "created.txt"
    actions: dict[str, Callable[[], Any]] = {
        "write_text": lambda: target.write_text("x", encoding="utf-8"),
        "open_append": lambda: open(victim, "a"),  # noqa: SIM115
        "open_update": lambda: open(victim, "r+"),  # noqa: SIM115
        "os_open": lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
        "remove": lambda: os.remove(victim),
        "unlink": lambda: victim.unlink(),
        "rename_away": lambda: os.rename(victim, tmp_path / "renamed.txt"),
        "rename_out_of_sandbox": lambda: os.rename(sandbox, tmp_path / "moved"),
        "mkdir": lambda: (tmp_path / "newdir").mkdir(),
        "rmtree": lambda: shutil.rmtree(tmp_path / "sandbox-sibling", ignore_errors=False),
        "copyfile": lambda: shutil.copyfile(victim, tmp_path / "copy.txt"),
        "symlink": lambda: os.symlink(victim, tmp_path / "link"),
        "chmod": lambda: os.chmod(victim, 0o777),
        "truncate": lambda: os.truncate(victim, 0),
    }
    for name, action in actions.items():
        value, events = guarded(action, workdir=sandbox)
        assert blocked(value, events, "filesystem_write_blocked"), name
        assert str(tmp_path) in events[0].detail, name
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sandbox", "victim.txt"]
    assert "blocked by the Agent Reliability Lab guard (filesystem_write_blocked)" in str(value)


def test_writes_inside_the_sandbox_directory_are_allowed(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    def work() -> str:
        (sandbox / "nested").mkdir()
        note = sandbox / "nested" / "note.txt"
        note.write_text("inside", encoding="utf-8")
        os.rename(note, sandbox / "moved.txt")
        (sandbox / "nested").rmdir()
        return (sandbox / "moved.txt").read_text(encoding="utf-8")

    assert guarded(work, workdir=sandbox) == ("inside", [])
    assert [path.name for path in sandbox.iterdir()] == ["moved.txt"]


def test_symlinks_and_dot_dot_cannot_be_used_to_leave_the_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (sandbox / "way-out").symlink_to(outside)
    lookalike = tmp_path / "sandbox-evil"
    lookalike.mkdir()
    for target in (
        sandbox / "way-out" / "f.txt",
        sandbox / ".." / "outside" / "g.txt",
        lookalike / "h.txt",
    ):
        value, events = guarded(
            lambda target=target: target.write_text("x", encoding="utf-8"), workdir=sandbox
        )
        assert blocked(value, events, "filesystem_write_blocked"), target
    assert list(outside.iterdir()) == [] and list(lookalike.iterdir()) == []


def test_without_a_sandbox_directory_no_write_is_allowed(tmp_path: Path) -> None:
    value, events = guarded(lambda: (tmp_path / "x.txt").write_text("x", encoding="utf-8"))
    assert blocked(value, events, "filesystem_write_blocked") and not (tmp_path / "x.txt").exists()


def test_reads_are_allowed_by_default_and_blocked_under_strict_reads(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("fixture text, not a real secret", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "ok.txt").write_text("inside", encoding="utf-8")
    assert guarded(lambda: secret.read_text(encoding="utf-8"), workdir=sandbox) == (
        "fixture text, not a real secret",
        [],
    )
    assert guarded(lambda: sorted(os.listdir(tmp_path)), workdir=sandbox) == (
        ["sandbox", "secret.txt"],
        [],
    )
    strict = GuardConfig(strict_reads=True)
    for action in (
        lambda: secret.read_text(encoding="utf-8"),
        lambda: os.listdir(tmp_path),
        lambda: list(os.scandir(tmp_path)),
    ):
        value, events = guarded(action, strict, sandbox)
        assert blocked(value, events, "filesystem_read_blocked")
    assert guarded(lambda: (sandbox / "ok.txt").read_text(encoding="utf-8"), strict, sandbox) == (
        "inside",
        [],
    )
    assert guarded(lambda: len(Path(os.__file__).read_bytes()) > 0, strict, sandbox) == (True, [])
    allowed = GuardConfig(strict_reads=True, extra_read_roots=[str(tmp_path)])
    assert guarded(lambda: secret.read_text(encoding="utf-8"), allowed, sandbox)[1] == []


def test_the_null_device_is_exempt() -> None:
    def work() -> str:
        with open(os.devnull, "w") as sink:
            sink.write("discarded")
        return "ok"

    assert guarded(work, GuardConfig(strict_reads=True)) == ("ok", [])


def test_only_real_bytecode_caches_are_exempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importable = tmp_path / "importable"
    elsewhere = tmp_path / "elsewhere"
    importable.mkdir()
    elsewhere.mkdir()
    (importable / "arl_guard_probe_module.py").write_text("VALUE = 41 + 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(importable))
    monkeypatch.delitem(sys.modules, "arl_guard_probe_module", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    def first_import() -> int:
        import arl_guard_probe_module

        return int(arl_guard_probe_module.VALUE)

    assert guarded(first_import) == (42, [])
    cached = list((importable / "__pycache__").iterdir())
    assert (
        len(cached) == 1
        and cached[0].name.startswith("arl_guard_probe_module.")
        and cached[0].suffix == ".pyc"
    )
    sys.modules.pop("arl_guard_probe_module", None)

    cache = importable / "__pycache__"
    assert guarded(lambda: (cache / "other.cpython-313.pyc.12345").write_bytes(b"\x00"))[1] == []
    for target in (
        cache / "notes.txt",
        cache / "payload.pyc.txt",
        elsewhere / "__pycache__",
        tmp_path / "__pycache__",
    ):
        action = (
            target.mkdir
            if target.name == "__pycache__"
            else lambda target=target: target.write_text("x")
        )
        value, events = guarded(action)
        assert blocked(value, events, "filesystem_write_blocked"), target
    (elsewhere / "__pycache__").mkdir()
    value, events = guarded(
        lambda: (elsewhere / "__pycache__" / "m.cpython-313.pyc").write_bytes(b"\x00")
    )
    assert blocked(value, events, "filesystem_write_blocked")
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        [cached[0].name, "other.cpython-313.pyc.12345"]
    )


# --------------------------------------------------------------------------- network and processes


def test_network_access_is_blocked() -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    actions: dict[str, Callable[[], Any]] = {
        "connect": lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.2),
        "resolve": lambda: socket.getaddrinfo("127.0.0.1", 9),
        "bind": lambda: socket.socket().bind(("127.0.0.1", 0)),
        "urllib": lambda: opener.open("http://127.0.0.1:9/", timeout=0.2),
        "gethostbyname": lambda: socket.gethostbyname("127.0.0.1"),
    }
    for name, action in actions.items():
        value, events = guarded(action)
        assert blocked(value, events, "network_blocked"), (name, value)
    value, events = guarded(lambda: socket.gethostname() and socket.socket().close())
    assert events == [] and not isinstance(value, BaseException)


def test_network_can_be_allowed_explicitly() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def connect() -> str:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            return "connected"

    try:
        assert guarded(connect, GuardConfig(allow_network=True)) == ("connected", [])
        value, events = guarded(connect)
        assert blocked(value, events, "network_blocked")
    finally:
        server.close()


def test_starting_processes_is_blocked_unless_allowed() -> None:
    command = [sys.executable, "-c", "pass"]
    actions: dict[str, Callable[[], Any]] = {
        "subprocess.run": lambda: subprocess.run(command, check=False),
        "os.system": lambda: os.system("true"),
        "os.kill": lambda: os.kill(os.getpid(), 0),
    }
    for name, action in actions.items():
        value, events = guarded(action)
        assert blocked(value, events, "subprocess_blocked"), name
    value, events = guarded(
        lambda: subprocess.run(command, check=False).returncode, GuardConfig(allow_subprocess=True)
    )
    assert (value, events) == (0, [])


# --------------------------------------------------------------------------- threads


def test_threads_started_by_a_guarded_thread_inherit_the_guard(tmp_path: Path) -> None:
    target = tmp_path / "from-child.txt"
    caught: list[BaseException] = []

    def child() -> None:
        try:
            target.write_text("x", encoding="utf-8")
        except BaseException as exc:
            caught.append(exc)

    def parent() -> str:
        worker = threading.Thread(target=child)
        worker.start()
        worker.join(10)
        grandchildren = threading.Thread(target=lambda: threading.Thread(target=child).start())
        grandchildren.start()
        grandchildren.join(10)
        return "done"

    value, events = guarded(parent)
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and not thread.daemon:
            thread.join(10)
    assert value == "done" and not target.exists()
    assert len(caught) == 2 and all(isinstance(exc, GuardViolation) for exc in caught)
    assert [event.kind for event in events] == ["filesystem_write_blocked"] * 2


def test_threads_that_are_not_agent_threads_are_unaffected(tmp_path: Path) -> None:
    go, done = threading.Event(), threading.Event()
    target = tmp_path / "from-bystander.txt"

    def bystander() -> None:
        go.wait(10)
        target.write_text("written", encoding="utf-8")
        done.set()

    other = threading.Thread(target=bystander)
    other.start()

    def agent() -> bool:
        go.set()
        return done.wait(10)

    value, events = guarded(agent)
    other.join(10)
    assert (value, events) == (True, []) and target.read_text(encoding="utf-8") == "written"
    (tmp_path / "after.txt").write_text(
        "the guard is gone when the agent finishes", encoding="utf-8"
    )


def test_the_guard_is_released_after_the_agent_finishes_or_crashes(tmp_path: Path) -> None:
    def crash() -> None:
        raise RuntimeError("boom")

    events: list[SecurityEvent] = []
    wrapped = guard.wrap(crash, GuardConfig(), None, events)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped()
    (tmp_path / "main-thread.txt").write_text("fine", encoding="utf-8")
    assert events == [] and threading.get_ident() not in guard._guards


def test_recorded_events_are_capped(tmp_path: Path) -> None:
    def hammer() -> int:
        refused = 0
        for number in range(MAX_EVENTS + 50):
            try:
                (tmp_path / f"f{number}.txt").write_text("x", encoding="utf-8")
            except GuardViolation:
                refused += 1
        return refused

    value, events = guarded(hammer)
    assert value == MAX_EVENTS + 50 and len(events) == MAX_EVENTS and list(tmp_path.iterdir()) == []


def test_a_violation_is_a_permission_error_with_a_bounded_detail(tmp_path: Path) -> None:
    long_name = tmp_path / ("n" * 200) / ("m" * 200)
    value, events = guarded(lambda: long_name.write_text("x", encoding="utf-8"))
    assert isinstance(value, PermissionError) and isinstance(value, OSError)
    assert len(events[0].detail) <= len("open: ") + 200


def test_threads_started_behind_the_threading_modules_back_are_blocked() -> None:
    import _thread

    value, events = guarded(lambda: _thread.start_new_thread(lambda: None, ()))
    assert blocked(value, events, "thread_blocked")


def test_a_stale_guard_is_dropped_when_a_thread_identifier_is_reused() -> None:
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    events: list[SecurityEvent] = []
    stale = guard._ThreadGuard(GuardConfig(), (), (), (), events, weakref.ref(finished))
    ident = threading.get_ident()
    guard._guards[ident] = stale  # as if the finished agent thread had had this thread's id
    try:
        guard._hook("os.system", ("true",))  # would raise if the stale guard still applied
    finally:
        guard._guards.pop(ident, None)
    assert events == [] and not stale.alive(ident)


def test_adoption_only_applies_to_the_thread_that_was_started() -> None:
    events: list[SecurityEvent] = []
    current = threading.current_thread()
    parent = guard._ThreadGuard(
        GuardConfig(), ("/w",), ("/r",), ("/i",), events, weakref.ref(current)
    )
    unstarted = threading.Thread(target=lambda: None)
    guard._pending[unstarted] = parent
    try:
        assert guard._adopt(threading.get_ident()) is None
        assert threading.get_ident() not in guard._guards
        guard._hook("os.system", ("true",))  # this thread is not the pending one: not guarded
    finally:
        guard._pending.pop(unstarted, None)
    guard._pending[current] = parent
    try:
        adopted = guard._adopt(threading.get_ident())
    finally:
        guard._guards.pop(threading.get_ident(), None)
        guard._pending.pop(current, None)
    assert adopted is not None and adopted is not parent and adopted.events is events
    assert (adopted.write_roots, adopted.read_roots, adopted.import_roots) == (
        ("/w",),
        ("/r",),
        ("/i",),
    )
    assert current not in guard._pending and events == []


def test_path_arguments_that_are_not_paths_are_ignored() -> None:
    assert guard._real(7) is None and guard._real(None) is None and guard._real(3.5) is None
    assert guard._real("/tmp/\x00embedded-nul") is None
    assert guard._real(b"/") == "/" and guard._real(Path("/")) == "/"

    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no text")

    events: list[SecurityEvent] = []
    state = guard._ThreadGuard(
        GuardConfig(), (), (), (), events, weakref.ref(threading.current_thread())
    )
    with pytest.raises(GuardViolation):
        guard._block(state, "network_blocked", "socket.connect", Unprintable())
    assert events[0].detail == "socket.connect: <unprintable>"


# --------------------------------------------------------------------------- through the runner


def test_a_hostile_agent_is_contained_and_fails_the_containment_check(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    attempts: list[str] = []

    def hostile(ctx: AgentContext) -> str:
        for action in (
            lambda: (tmp_path / "stolen.txt").write_text(ctx.task.input, encoding="utf-8"),
            lambda: victim.unlink(),
            lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.2),
            lambda: subprocess.run([sys.executable, "-c", "pass"], check=False),
        ):
            try:
                action()
            except OSError as exc:
                attempts.append(type(exc).__name__)
        return ANSWER

    result = run_agent(hostile)
    assert attempts == ["GuardViolation"] * 4
    assert [event.kind for event in result.trace.security_events] == [
        "filesystem_write_blocked",
        "filesystem_write_blocked",
        "network_blocked",
        "subprocess_blocked",
    ]
    assert not result.passed and result.failed_evaluators() == ["containment"]
    assert result.failure_categories == ["sandbox_violation"]
    containment = result.evaluation("containment")
    assert containment is not None and len(containment.findings) == 4
    assert (
        victim.read_text(encoding="utf-8") == "keep me" and not (tmp_path / "stolen.txt").exists()
    )


def test_an_unhandled_violation_is_also_an_agent_exception(tmp_path: Path) -> None:
    result = run_agent(lambda ctx: (tmp_path / "x.txt").write_text("x", encoding="utf-8"))
    assert [error.kind for error in result.trace.errors] == ["agent_exception"]
    assert result.trace.errors[0].message.startswith("GuardViolation: blocked by")
    assert result.failure_categories == ["agent_crash", "sandbox_violation", "wrong_outcome"]
    assert not (tmp_path / "x.txt").exists()


def test_guard_options_come_from_the_run_configuration(tmp_path: Path) -> None:
    target = tmp_path / "allowed-when-disabled.txt"

    def writer(ctx: AgentContext) -> str:
        target.write_text("only ever inside pytest's temporary directory", encoding="utf-8")
        return ANSWER

    off = run_agent(writer, RunConfig(guard=GuardConfig(enabled=False)))
    assert off.passed and off.trace.security_events == [] and target.exists()
    target.unlink()
    on = run_agent(writer)
    assert not on.passed and not target.exists()

    def reader(ctx: AgentContext) -> str:
        return (
            ANSWER
            if Path(tmp_path, "nothing-here.txt").exists() or os.listdir(tmp_path) == []
            else "?"
        )

    assert run_agent(reader).passed
    strict = run_agent(reader, RunConfig(guard=GuardConfig(strict_reads=True)))
    assert [event.kind for event in strict.trace.security_events] == ["filesystem_read_blocked"]

    def spawner(ctx: AgentContext) -> str:
        subprocess.run([sys.executable, "-c", "pass"], check=True)
        return ANSWER

    assert run_agent(spawner, RunConfig(guard=GuardConfig(allow_subprocess=True))).passed
    assert not run_agent(spawner).passed


def test_ordinary_agent_code_is_not_disturbed_by_the_guard() -> None:
    def ordinary(ctx: AgentContext) -> str:
        import colorsys  # an import that may not have happened yet
        import json

        async def answer() -> str:
            await asyncio.sleep(0)
            return ANSWER

        workers = [threading.Thread(target=lambda: json.dumps({"n": 1})) for _ in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        with contextlib.suppress(Exception):
            colorsys.rgb_to_hsv(0.1, 0.2, 0.3)
        return asyncio.run(answer())

    result = run_agent(ordinary)
    assert result.passed and result.trace.security_events == [] and result.trace.errors == []
