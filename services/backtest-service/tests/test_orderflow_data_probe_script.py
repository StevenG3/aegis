from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_script():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "orderflow_data_probe.py"
    spec = importlib.util.spec_from_file_location("orderflow_data_probe_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["orderflow_data_probe_script"] = module
    spec.loader.exec_module(module)
    return module


class FakeExchange:
    has = {
        "fetchTrades": True,
        "fetchOrderBook": True,
        "fetchFundingRateHistory": True,
    }
    rateLimit = 50

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def fetch_trades(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, object]]:
        if since is not None:
            return [
                {
                    "timestamp": since,
                    "price": 100,
                    "amount": 0.1,
                    "side": "buy",
                    "id": "since",
                }
            ]
        count = min(limit or 2, 2)
        rows = [
            {
                "timestamp": 1_700_000_000_000 + index * 1000,
                "price": 100 + index,
                "amount": 0.1 + index,
                "side": "buy" if index % 2 == 0 else "sell",
                "id": str(index),
            }
            for index in range(count)
        ]
        return rows

    def fetch_order_book(self, symbol: str, limit: int | None = None) -> dict[str, object]:
        return {"bids": [[99.0, 1.0]], "asks": [[101.0, 2.0]], "nonce": 1}

    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, object]]:
        return [{"timestamp": 1_700_000_000_000, "fundingRate": 0.0001}]


def test_orderflow_probe_builds_availability_report() -> None:
    module = load_script()
    fake_ccxt = SimpleNamespace(binance=lambda config: FakeExchange(config))

    report = module.build_report(
        fake_ccxt,
        exchanges=["binance"],
        symbol="BTC/USDT",
        swap_symbol="BTC/USDT:USDT",
        trade_limit=2,
        book_limit=5,
        since_hours=24,
    )

    row = report["exchanges"][0]
    assert row["exchange"] == "binance"
    assert row["trades"]["availability"] == "available"
    assert row["trades"]["has_taker_side"] is True
    assert row["trades"]["history_depth_conclusion"] == "partial_recent_history_since_supported"
    assert row["trades"]["since_probe"]["sample_count"] == 1
    assert row["order_book"]["availability"] == "available"
    assert row["order_book"]["historical_order_book"] is False
    assert row["funding"]["availability"] == "available"
    assert report["conclusion"]["historical_backtest_feasibility"] == "limited"
    assert "No API keys" in report["human_readable"]
