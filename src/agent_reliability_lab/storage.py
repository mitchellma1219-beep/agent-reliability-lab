"""JSON run storage with safe paths.

Runs are stored as ``<root>/<name>.json``. A run name is an identifier, never a path: it may
contain letters, digits, dot, dash and underscore only, so ``../x`` or an absolute path can
not be used to write or read outside the runs directory. Symbolic links are not followed.
Files are written atomically and nothing is ever unpickled.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .errors import StorageError
from .models import ID_PATTERN, SCHEMA_VERSION, RunResult

MAX_RUN_BYTES = 200_000_000
_NAME = re.compile(ID_PATTERN)
_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{n}" for n in range(1, 10)]
    + [f"lpt{n}" for n in range(1, 10)]
)


def validate_run_name(name: object) -> str:
    """Return ``name`` if it is a safe run name, otherwise raise :class:`StorageError`."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise StorageError(
            "invalid run name: use 1-100 letters, digits, '.', '-' or '_', starting with a "
            "letter or digit"
        )
    if ".." in name or name.endswith(".") or name.split(".")[0].lower() in _RESERVED:
        raise StorageError(f"invalid run name {name!r}")
    return name


@dataclass(frozen=True)
class RunInfo:
    """Summary of a stored run for listings."""

    name: str
    suite: str
    agent: str
    version: str
    created_at: str
    cases: int
    trials: int
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def parse_run(text: str, source: str) -> RunResult:
    """Parse stored JSON into a run, with errors reported as :class:`StorageError`."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise StorageError(f"{source}: not valid JSON") from exc
    if not isinstance(data, dict):
        raise StorageError(f"{source}: not a stored run")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise StorageError(
            f"{source}: schema version {version!r} is not supported (expected {SCHEMA_VERSION})"
        )
    try:
        return RunResult.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise StorageError(f"{source}: not a valid run ({where}: {first['msg']})") from exc


def load_run_file(path: str | Path) -> RunResult:
    """Load a run from an explicit file path chosen by the user."""
    file = Path(path)
    if file.is_symlink():
        raise StorageError(f"{file}: refusing to follow a symbolic link")
    if not file.is_file():
        raise StorageError(f"{file}: no such run file")
    if file.stat().st_size > MAX_RUN_BYTES:
        raise StorageError(f"{file}: larger than {MAX_RUN_BYTES} bytes")
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StorageError(f"{file}: cannot be read as UTF-8 text") from exc
    return parse_run(text, str(file))


class RunStore:
    """A directory of stored runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        """The file for a run name, after checking that it stays inside the store."""
        safe = validate_run_name(name)
        path = self.root / f"{safe}.json"
        if path.resolve().parent != self.root.resolve():
            raise StorageError(f"run {safe!r} would be stored outside {self.root}")
        return path

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def save(self, run: RunResult, *, overwrite: bool = False) -> Path:
        """Write the run atomically and return its path."""
        target = self.path_for(run.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise StorageError(f"{target}: refusing to write through a symbolic link")
        if target.exists() and not overwrite:
            raise StorageError(f"run {run.run_id!r} already exists; choose another name")
        payload = run.model_dump_json()
        handle, temporary = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return target

    def load(self, name: str) -> RunResult:
        path = self.path_for(name)
        if not path.is_file():
            raise StorageError(f"no stored run named {name!r} in {self.root}")
        return load_run_file(path)

    def names(self) -> list[str]:
        """Names of the stored runs, oldest file name first."""
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_file() and not path.is_symlink() and _NAME.fullmatch(path.stem):
                found.append(path.stem)
        return found

    def list(self) -> tuple[list[RunInfo], list[str]]:
        """Summaries of the readable runs, plus messages for files that could not be read."""
        infos: list[RunInfo] = []
        problems: list[str] = []
        for name in self.names():
            try:
                run = self.load(name)
            except StorageError as exc:
                problems.append(str(exc))
                continue
            infos.append(
                RunInfo(
                    name=name,
                    suite=run.suite_name,
                    agent=run.agent_name,
                    version=run.agent_version,
                    created_at=run.created_at,
                    cases=len(run.case_ids()),
                    trials=run.config.trials,
                    passed=sum(1 for result in run.results if result.passed),
                    total=len(run.results),
                )
            )
        infos.sort(key=lambda info: (info.created_at, info.name))
        return infos, problems


def resolve_run(reference: str, store: RunStore) -> RunResult:
    """Load a run given either a stored name or an explicit path to a ``.json`` file."""
    if reference.endswith(".json") or os.sep in reference or "/" in reference:
        return load_run_file(reference)
    return store.load(reference)
