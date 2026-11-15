"""End-to-end tests for the ``arl`` command line. Everything runs offline and deterministically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_reliability_lab.cli import build_parser, main


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_scenarios_lists_builtin_suites(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, "scenarios")
    assert code == 0
    assert "core" in out and "injection" in out


def test_scenarios_detail_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, "scenarios", "core")
    assert code == 0 and out.strip()
    code, out, _ = _run(capsys, "scenarios", "core", "--format", "json")
    assert code == 0
    json.loads(out)


def test_scenarios_unknown_suite_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _run(capsys, "scenarios", "no-such-suite")
    assert code == 2
    assert "error" in err


def test_run_store_list_report_and_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = str(tmp_path / "runs")
    code, out, _ = _run(capsys, "run", "--suite", "core", "--name", "base", "--runs-dir", runs)
    assert code == 0, out
    code, out, _ = _run(
        capsys,
        "run",
        "--suite",
        "core",
        "--agent",
        "faulty:ignores_instruction",
        "--name",
        "cand",
        "--fail-under",
        "0",
        "--quiet",
        "--runs-dir",
        runs,
    )
    assert code == 0, out

    code, out, _ = _run(capsys, "list", "--runs-dir", runs)
    assert code == 0 and "base" in out and "cand" in out
    code, out, _ = _run(capsys, "list", "--format", "json", "--runs-dir", runs)
    assert code == 0
    json.loads(out)

    code, out, _ = _run(capsys, "report", "base", "--runs-dir", runs)
    assert code == 0 and out.strip()
    report_file = tmp_path / "report.json"
    code, _, _ = _run(
        capsys, "report", "base", "--format", "json", "--out", str(report_file), "--runs-dir", runs
    )
    assert code == 0
    json.loads(report_file.read_text(encoding="utf-8"))

    code, out, _ = _run(capsys, "compare", "base", "cand", "--runs-dir", runs)
    assert code in (0, 1)
    assert out.strip()
    code, out, _ = _run(
        capsys, "compare", "base", "base", "--format", "json", "--strict", "--runs-dir", runs
    )
    assert code == 0
    json.loads(out)


def test_run_json_without_saving(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(
        capsys,
        "run",
        "--suite",
        "core",
        "--no-save",
        "--format",
        "json",
        "--faults",
        sorted(_fault_profiles())[0],
        "--fail-under",
        "0",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 0
    json.loads(out)
    assert not any(tmp_path.iterdir())


def test_run_fails_under_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = _run(
        capsys,
        "run",
        "--suite",
        "injection",
        "--agent",
        "faulty:follows_injection",
        "--no-save",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 1


def test_run_rejects_a_bad_agent_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(
        capsys,
        "run",
        "--suite",
        "core",
        "--agent",
        "faulty:not_a_mode",
        "--runs-dir",
        str(tmp_path),
    )
    assert code == 2 and "error" in err


def test_demo_writes_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "demo"
    code, _, _ = _run(capsys, "demo", "--trials", "1", "--runs-dir", str(runs))
    assert code == 0
    assert list(runs.glob("*.json"))


def test_calibrate_judge_with_the_mock_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run(capsys, "calibrate-judge")
    assert code in (0, 1) and out.strip()
    out_file = tmp_path / "calibration.json"
    code, _, _ = _run(capsys, "calibrate-judge", "--format", "json", "--out", str(out_file))
    assert code in (0, 1)
    json.loads(out_file.read_text(encoding="utf-8"))


def test_calibrate_judge_refuses_the_api_without_consent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _, err = _run(capsys, "calibrate-judge", "--judge", "anthropic")
    assert code == 2 and "error" in err


def test_matrix_markdown_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, "matrix")
    assert code == 0 and "Detection matrix" in out
    code, out, _ = _run(capsys, "matrix", "--format", "json")
    assert code == 0
    assert json.loads(out)["ok"] is True


def _fault_profiles() -> list[str]:
    from agent_reliability_lab.cli import FAULT_PROFILES

    return list(FAULT_PROFILES)
