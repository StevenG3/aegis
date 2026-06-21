from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "crypto_cross_sectional_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "crypto_cross_sectional_evidence_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["crypto_cross_sectional_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


class _FakeExchange:
    def __init__(self) -> None:
        self.tickers: dict[str, dict[str, object]]
        self.tickers = {
            "BTC/USDT:USDT": {"quoteVolume": 1_000_000_000},
            "ETH/USDT:USDT": {"baseVolume": 500_000, "last": 2_000},
            "USDC/USDT:USDT": {"quoteVolume": 900_000_000},
            "WETH/USDT:USDT": {"quoteVolume": 800_000_000},
            "ABCUP/USDT:USDT": {"quoteVolume": 700_000_000},
            "WIF/USDT:USDT": {"quoteVolume": 600_000_000},
        }

    def fetch_tickers(self) -> dict[str, dict[str, object]]:
        return self.tickers


def test_ranked_candidates_exclude_stable_wrapped_and_leveraged_but_keep_wif() -> None:
    evidence = load_script()
    markets: dict[str, dict[str, Any]] = {
        "BTC/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "BTC",
        },
        "ETH/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "ETH",
        },
        "USDC/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "USDC",
        },
        "WETH/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "WETH",
        },
        "ABCUP/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "ABCUP",
        },
        "WIF/USDT:USDT": {
            "swap": True,
            "quote": "USDT",
            "linear": True,
            "base": "WIF",
        },
    }

    ranked = evidence._ranked_candidates(
        _FakeExchange(),
        markets=markets,
        cross_listed_bases={"BTC", "ETH", "WIF"},
        prefetch_symbols=10,
    )

    assert ranked == [
        ("BTC/USDT:USDT", 2),
        ("ETH/USDT:USDT", 2),
        ("WIF/USDT:USDT", 2),
    ]


def test_bars_from_ohlcv_attaches_daily_funding_and_proxy_volume() -> None:
    evidence = load_script()
    day0 = evidence._parse_date("2021-01-01")
    ohlcv = [
        [day0, 100.0, 110.0, 90.0, 105.0, 1_000.0],
        [day0 + 86_400_000, 105.0, 112.0, 100.0, 110.0, 1_000.0],
    ]

    bars = evidence._bars_from_ohlcv(
        ohlcv,
        funding_by_day={day0: 0.001},
        exchange_count=2,
    )

    assert len(bars) == 2
    assert bars[0].quote_volume_usd == 105_000.0
    assert bars[0].funding_rate == 0.001
    assert bars[1].funding_rate == 0.0
    assert bars[0].exchange_count == 2
