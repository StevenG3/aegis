from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from aegis.lean_execution_validation import validate_lean_execution_report


def _load_script() -> Any:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "lean_execution_validation.py"
    spec = importlib.util.spec_from_file_location("lean_execution_validation", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lean_cli = _load_script()


def _spec() -> dict[str, Any]:
    return {
        "id": "lean-spx-vrp-unit",
        "source_hypothesis_id": "olympus85c_spx_vrp",
        "source_aegis_state": "EDGE",
        "engine": "lean",
        "mode": "backtest",
        "live_trading": False,
        "read_only": True,
        "data_adequacy": "limited",
        "unlock_condition": "broker-native paper fills and paid PIT option chains",
    }


def _report() -> dict[str, Any]:
    return {
        "engine": "lean",
        "live_trading": False,
        "executable_fills": False,
        "order_count": 42,
        "total_fees": 123.0,
        "total_slippage": 45.0,
        "metrics": {
            "annualized_return": 0.08,
            "sharpe": 1.2,
            "max_drawdown": -0.12,
        },
        "benchmark": {
            "annualized_return": 0.05,
            "sharpe": 0.6,
        },
    }


def test_validate_lean_execution_report_passes_costed_offline_report() -> None:
    result = validate_lean_execution_report(spec=_spec(), report=_report())

    assert result["state"] == "EDGE"
    assert result["verdict"] == "EXECUTION_VALID_PENDING_FINAL_GATE"
    assert result["data_adequacy"] == "limited"
    assert result["safety"]["live_trading"] is False
    assert result["execution_metrics"]["order_count"] == 42


def test_validate_lean_execution_report_fails_when_benchmark_not_beaten() -> None:
    report = _report()
    report["metrics"]["sharpe"] = 0.3

    result = validate_lean_execution_report(spec=_spec(), report=report)

    assert result["state"] == "NO_EDGE"
    assert result["verdict"] == "EXECUTION_FAIL"
    assert "benchmark" in result["reason"]


def test_validate_lean_execution_report_blocks_live_or_uncosted_report() -> None:
    report = _report()
    report["live_trading"] = True
    report["total_slippage"] = 0.0

    result = validate_lean_execution_report(spec=_spec(), report=report)

    assert result["state"] == "INSUFFICIENT"
    assert result["data_adequacy"] == "blocked"
    assert "live_trading" in result["reason"]


def test_lean_cli_writes_private_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_root = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    task_dir = private_root / "incubating" / "olympus85"
    public_root.mkdir()
    task_dir.mkdir(parents=True)
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(lean_cli, "REPO_ROOT", public_root)
    spec_path = task_dir / "lean-spec.json"
    report_path = task_dir / "lean-report.json"
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    assert lean_cli.run_cli([str(spec_path), str(report_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "EDGE"
    result_path = Path(summary["result_path"])
    assert result_path.is_relative_to(task_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["verdict"]["verdict"] == "EXECUTION_VALID_PENDING_FINAL_GATE"


def test_lean_cli_rejects_public_spec_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_root = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    public_root.mkdir()
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(lean_cli, "REPO_ROOT", public_root)
    spec_path = public_root / "lean-spec.json"
    report_path = private_root / "incubating" / "olympus85" / "lean-report.json"
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    assert lean_cli.run_cli([str(spec_path), str(report_path)]) == 2

    assert "inside the public repository" in capsys.readouterr().err
