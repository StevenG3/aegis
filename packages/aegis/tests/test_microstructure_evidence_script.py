from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "microstructure_evidence.py"
    spec = importlib.util.spec_from_file_location("microstructure_evidence_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["microstructure_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


def load_impulse_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "microstructure_impulse_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "microstructure_impulse_evidence_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["microstructure_impulse_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


class FakeBinanceUsdm:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, object]]:
        start = since or 1_700_000_000_000
        return [
            {"timestamp": start + index * 14_400_000, "fundingRate": 0.0002}
            for index in range(min(limit or 6, 6))
        ]

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        start = since or 1_700_000_000_000
        return [
            {
                "timestamp": start + index * 14_400_000,
                "openInterestAmount": 1000.0 - index * 10.0,
            }
            for index in range(min(limit or 6, 6))
        ]

    def fapiPublicGetKlines(self, params: dict[str, object]) -> list[list[object]]:
        start = int(cast(int, params["startTime"]))
        limit = min(int(cast(int, params.get("limit", 6))), 6)
        return [
            [
                start + index * 14_400_000,
                "100",
                "110",
                "90",
                str(100 + index),
                "40",
                start + index * 14_400_000 + 1,
                "4000",
                "0",
                "15",
            ]
            for index in range(limit)
        ]


def test_run_from_env_uses_private_olympus60(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_SYMBOLS", "BTC/USDT:USDT, ftt/usdt:usdt")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_DELISTED_CRASH", "FTT/USDT:USDT")

    run = module._run_from_env()

    assert run.symbols == ("BTC/USDT:USDT", "FTT/USDT:USDT")
    assert run.output_dir == private_root / "incubating" / "olympus60"
    assert run.delisted_crash_symbols == ("FTT/USDT:USDT",)


def test_alignment_uses_funding_price_flow_and_open_interest() -> None:
    module = load_script()
    rows = module._align_observations(
        symbol="BTC/USDT:USDT",
        funding=[
            {"timestamp": 1000, "funding_rate": 0.0001},
            {"timestamp": 2000, "funding_rate": -0.0002},
        ],
        price_flow=[
            {
                "timestamp": 900,
                "close": 100.0,
                "buy_volume": 10.0,
                "sell_volume": 20.0,
                "quote_volume_usd": 3000.0,
            },
            {
                "timestamp": 1900,
                "close": 101.0,
                "buy_volume": 30.0,
                "sell_volume": 10.0,
                "quote_volume_usd": 4000.0,
            },
        ],
        open_interest=[
            {"timestamp": 800, "open_interest": 1000.0},
            {"timestamp": 1800, "open_interest": 900.0},
        ],
        btc_reference=[
            {"timestamp": 850, "close": 99.0},
            {"timestamp": 1850, "close": 102.0},
        ],
        survivor_status="active",
    )

    assert len(rows) == 2
    assert rows[0]["close"] == 100.0
    assert rows[1]["open_interest"] == 900.0
    assert rows[1]["buy_volume"] == 30.0
    assert rows[1]["btc_close"] == 102.0
    assert rows[1]["quote_volume_usd"] == 4000.0


def test_main_with_fake_ccxt_writes_private_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()
    fake_ccxt = SimpleNamespace(binanceusdm=lambda config: FakeBinanceUsdm(config))
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_SYMBOLS", "BTC/USDT:USDT")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_START", "2024-01-01T00:00:00+00:00")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_END", "2024-01-05T00:00:00+00:00")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_MAX_BARS", "6")

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    json_path = Path(summary["json"])
    assert json_path.is_relative_to(private_root / "incubating" / "olympus60")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["coverage"]["funding_rows"] > 0
    assert payload["report"]["multiple_testing"]["method"] == "BH-FDR + CSCV_PBO"
    assert payload["verdict"]["survivor_ceiling_applied"] is True
    assert payload["input"]["btc_reference_symbol"] == "BTC/USDT:USDT"


def test_impulse_main_with_fake_ccxt_writes_olympus65_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_impulse_script()
    fake_ccxt = SimpleNamespace(binanceusdm=lambda config: FakeBinanceUsdm(config))
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_SYMBOLS", "BTC/USDT:USDT")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_START", "2024-01-01T00:00:00+00:00")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_END", "2024-01-05T00:00:00+00:00")
    monkeypatch.setenv("MICROSTRUCTURE_EVIDENCE_MAX_BARS", "6")

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    json_path = Path(summary["json"])
    assert json_path.is_relative_to(private_root / "incubating" / "olympus65")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["briefing"] == "CODEX_OLYMPUS_65_MICROSTRUCTURE_IMPULSE_REAL_CCXT"
    assert payload["comparison"]["impulse"]["candidate_count_n"] > payload["comparison"]["base"][
        "candidate_count_n"
    ]
    assert payload["predeclared"]["survivor_light_ceiling"] is True
    assert payload["impulse"]["verdict"]["survivor_ceiling_applied"] is True
