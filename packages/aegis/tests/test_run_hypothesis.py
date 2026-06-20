from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


def _load_script() -> Any:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_hypothesis.py"
    spec = importlib.util.spec_from_file_location("run_hypothesis", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_hypothesis = _load_script()


def _base_spec() -> dict[str, Any]:
    return {
        "id": "olympus59_unit",
        "type": "combo",
        "universe": ["BTC/USDT"],
        "predeclared_signals": ["carry_minus_cost"],
        "params": {"lookback": 20},
        "cost_model": {
            "fee_bps": 1.0,
            "slippage_bps": 2.0,
            "funding_bps_per_period": 0.0,
            "funding_label": "N/A for synthetic fixture",
        },
        "benchmark": "cash",
        "data_source": "local_fixture",
        "trial_n": 3,
        "survivor_light": False,
        "trust": {
            "registry_scope": "private",
            "predeclared": True,
            "review_gate": True,
            "export_contains_private_spec_data": False,
            "live_or_network_required": False,
            "no_live": True,
            "read_only": True,
        },
        "discipline": {
            "t_plus_1_execution": True,
            "locked_oos": True,
            "walk_forward": True,
            "full_costs": True,
            "multiple_testing": True,
            "survivor_ceiling": False,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _set_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    public_root = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    task_dir = private_root / "incubating" / "olympus59"
    public_root.mkdir()
    private_root.mkdir()
    task_dir.mkdir(parents=True)
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(run_hypothesis, "REPO_ROOT", public_root)
    return public_root, private_root, task_dir


def test_cli_runs_private_spec_and_appends_global_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, _private_root, task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = task_dir / "specs" / "unit.json"
    registry_path = task_dir / "hypothesis_registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "spec_id": "older",
                "trial_n": 2,
                "verdict": "NO_EDGE",
                "time": "2026-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(spec_path, _base_spec())

    assert run_hypothesis.run_cli([str(spec_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["global_trial_n"] == 5
    assert summary["state"] == "INSUFFICIENT"
    result_path = Path(summary["result_path"])
    assert result_path.is_relative_to(task_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["spec"]["id"] == "olympus59_unit"
    assert result["spec"]["type"] == "combo"
    assert result["verdict"]["verdict"] == "INSUFFICIENT"
    assert result["verdict"]["candidate_count_n"] == 3
    assert result["verdict"]["multiple_testing"]["hypothesis_trial_count_n"] == 3

    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["spec_id"] == "olympus59_unit"
    assert rows[-1]["trial_n"] == 3
    assert rows[-1]["verdict"] == "INSUFFICIENT"


def test_cli_rejects_public_spec_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_root, _private_root, _task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = public_root / "incubating" / "olympus59" / "unit.json"
    _write_json(spec_path, _base_spec())

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "inside the public repository" in capsys.readouterr().err


def test_cli_rejects_spec_outside_olympus59(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, private_root, _task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = private_root / "incubating" / "olympus60" / "unit.json"
    _write_json(spec_path, _base_spec())

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "spec JSON must live under" in capsys.readouterr().err


def test_cli_rejects_output_dir_outside_olympus59(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, private_root, task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = task_dir / "unit.json"
    _write_json(spec_path, _base_spec())
    bad_output = private_root / "incubating" / "olympus60"

    assert run_hypothesis.run_cli([str(spec_path), "--private-dir", str(bad_output)]) == 2

    assert "may only write under" in capsys.readouterr().err


def test_cli_rejects_missing_discipline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, _private_root, task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = task_dir / "unit.json"
    payload = _base_spec()
    payload["discipline"]["locked_oos"] = False
    _write_json(spec_path, payload)

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "discipline.locked_oos must be true" in capsys.readouterr().err


def test_cli_rejects_survivor_light_without_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, _private_root, task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = task_dir / "unit.json"
    payload = _base_spec()
    payload["survivor_light"] = True
    payload["discipline"]["survivor_ceiling"] = False
    _write_json(spec_path, payload)

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "survivor_light specs require" in capsys.readouterr().err


def test_cli_rejects_live_or_network_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _public_root, _private_root, task_dir = _set_roots(tmp_path, monkeypatch)
    spec_path = task_dir / "unit.json"
    payload = _base_spec()
    payload["data_source"] = "https://example.test/private-feed"
    _write_json(spec_path, payload)

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "forbidden local-only marker" in capsys.readouterr().err


def test_cli_requires_existing_private_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_STRATEGIES_ROOT", raising=False)

    assert run_hypothesis.run_cli([os.devnull]) == 2
