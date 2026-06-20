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


def _microstructure_observations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closes = [100.0, 101.0, 103.0, 105.0, 100.0, 96.0, 98.0, 101.0, 97.0, 94.0]
    btc_closes = [100.0, 101.0, 102.0, 106.0, 109.0, 104.0, 100.0, 103.0, 106.0, 101.0]
    for symbol, scale, survivor_status in (
        ("BTC/USDT:USDT", 1.0, "active"),
        ("DELISTED/USDT:USDT", 0.2, "delisted"),
    ):
        for index, close in enumerate(closes):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": 1_700_000_000 + index * 14_400,
                    "close": max(1.0, close * scale),
                    "btc_close": btc_closes[index],
                    "open_interest": max(1.0, 1_000.0 - index * 35.0),
                    "funding_rate": 0.0002,
                    "buy_volume": 10.0,
                    "sell_volume": 30.0,
                    "bid_ask_spread_bps": 5.0,
                    "top_depth_usd": 100_000.0,
                    "quote_volume_usd": 2_000_000.0,
                    "order_book_event_rate_per_hour": 0.0,
                    "survivor_status": survivor_status,
                }
            )
    return rows


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
    spec_path = private_root / "scatter" / "unit.json"
    _write_json(spec_path, _base_spec())

    assert run_hypothesis.run_cli([str(spec_path)]) == 2

    assert "must be under" in capsys.readouterr().err


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

    assert "same ${AEGIS_STRATEGIES_ROOT}" in capsys.readouterr().err


def test_cli_runs_named_microstructure_runner_from_olympus60(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_root = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    task_dir = private_root / "incubating" / "olympus60"
    public_root.mkdir()
    private_root.mkdir()
    task_dir.mkdir(parents=True)
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(run_hypothesis, "REPO_ROOT", public_root)
    spec_path = task_dir / "specs" / "microstructure.json"
    payload = _base_spec()
    payload.update(
        {
            "id": "olympus60_microstructure_unit",
            "type": "event",
            "runner": "microstructure_perp",
            "universe": ["BTC/USDT:USDT", "DELISTED/USDT:USDT"],
            "predeclared_signals": [
                "funding_sign",
                "oi_price_divergence",
                "orderflow_imbalance",
            ],
            "trial_n": 4,
            "survivor_light": True,
            "benchmark": "buy_and_hold",
            "data_source": "synthetic_offline_microstructure_fixture",
        }
    )
    payload["discipline"]["survivor_ceiling"] = True
    payload["cost_model"]["funding_label"] = "perp funding debited from observations"
    payload["params"] = {
        "observations": _microstructure_observations(),
        "grid": {
            "funding_abs_bps": [1.0, 2.0],
            "imbalance_abs": [0.2],
            "oi_drop_abs": [0.02],
            "score_threshold": [1, 2],
        },
        "locked_oos_fraction": 0.8,
        "fold_count": 4,
        "pbo_splits": 4,
    }
    _write_json(spec_path, payload)

    assert run_hypothesis.run_cli([str(spec_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["trial_n"] == 4
    assert summary["result_path"].startswith(str(task_dir))
    result = json.loads(Path(summary["result_path"]).read_text(encoding="utf-8"))
    assert result["payload"]["strategy"] == "microstructure_perp_funding_oi_orderflow"
    assert result["payload"]["multiple_testing"]["method"] == "BH-FDR + CSCV_PBO"
    assert result["payload"]["safety"]["network"] is False
    assert result["payload"]["safety"]["perp_funding_counted"] is True


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
