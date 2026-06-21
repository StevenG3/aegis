from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "polymarket_5m_forward_collector.py"
    spec = importlib.util.spec_from_file_location("polymarket_forward_collector", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["polymarket_forward_collector"] = module
    spec.loader.exec_module(module)
    return module


class FakeForwardClient:
    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.gamma_api_base_url = "https://gamma.example"

    def iter_events(
        self,
        *,
        limit: int,
        max_events: int,
        sleep_seconds: float,
        order: str,
        ascending: bool,
        closed: bool,
        active: bool,
        tag_slug: str,
    ) -> list[dict[str, object]]:
        _ = limit, max_events, sleep_seconds, order, ascending, closed, active, tag_slug
        end = datetime.now(UTC) + timedelta(seconds=70)
        start_ts = int(end.timestamp()) - 300
        return [
            {
                "slug": f"btc-updown-5m-{start_ts}",
                "markets": [
                    {
                        "conditionId": "condition-1",
                        "slug": f"btc-updown-5m-{start_ts}",
                        "question": "Bitcoin Up or Down - synthetic 5m",
                        "outcomes": '["Up", "Down"]',
                        "outcomePrices": '["0.5", "0.5"]',
                        "clobTokenIds": '["up-token", "down-token"]',
                        "resolutionSource": "https://data.chain.link/streams/btc-usd",
                        "endDate": end.isoformat().replace("+00:00", "Z"),
                        "closed": False,
                    }
                ],
            }
        ]

    def get_order_book(self, token_id: str) -> dict[str, object]:
        if token_id == "up-token":
            return {
                "bids": [{"price": "0.84", "size": "10"}],
                "asks": [{"price": "0.86", "size": "11"}],
            }
        return {
            "bids": [{"price": "0.14", "size": "10"}],
            "asks": [{"price": "0.16", "size": "11"}],
        }

    def _get_json(self, url: str) -> list[dict[str, object]]:
        _ = url
        start_ts = 1_900_000_000
        return [
            {
                "markets": [
                    {
                        "outcomes": '["Up", "Down"]',
                        "outcomePrices": '["1", "0"]',
                        "conditionId": "condition-1",
                        "slug": f"btc-updown-5m-{start_ts}",
                    }
                ]
            }
        ]


def test_forward_collector_captures_decision_window_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(module, "PolymarketDataApiClient", FakeForwardClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polymarket_5m_forward_collector.py",
            "--max-iterations",
            "1",
            "--duration-seconds",
            "0",
            "--interval-seconds",
            "0",
            "--sleep-seconds",
            "0",
            "--depth-levels",
            "2",
            "--disable-chainlink-rtds",
        ],
    )

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    output_dir = Path(summary["output_dir"])
    assert output_dir.is_relative_to(private_root / "incubating" / "olympus63")
    assert summary["snapshot_records_written"] == 2
    assert summary["wallet_order_account_connected"] is False
    rows = [
        json.loads(line)
        for line in Path(summary["last_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["outcome"] for row in rows} == {"Up", "Down"}
    assert all(row["record_type"] == "snapshot" for row in rows)
    assert rows[0]["chainlink_reference_status"] == "chainlink_rtds_not_available"


def test_settlement_record_marks_missing_chainlink_source() -> None:
    module = load_script()
    market = module.ActiveBtc5mMarket(
        condition_id="condition-1",
        slug="btc-updown-5m-1900000000",
        title="Bitcoin Up or Down - synthetic 5m",
        start_ts=1_900_000_000,
        end_ts=1_900_000_300,
        outcomes=("Up", "Down"),
        token_ids=("up-token", "down-token"),
        resolution_source="https://data.chain.link/streams/btc-usd",
    )
    config = module.ForwardCollectorConfig(
        output_dir=Path("/tmp/unused"),
        interval_seconds=0,
        duration_seconds=0,
        max_iterations=1,
        market_page_size=1,
        max_markets=1,
        timeout_seconds=1,
        sleep_seconds=0,
        depth_levels=2,
        decision_window_start_seconds=90,
        decision_window_end_seconds=30,
        settlement_lag_seconds=0,
        chainlink_price_url_template=None,
        chainlink_rtds_enabled=False,
        chainlink_tick_match_tolerance_seconds=10,
    )

    record = module.settlement_record(
        FakeForwardClient(user_agent="ua", timeout_seconds=1),
        config,
        {},
        market,
        datetime.now(UTC),
    )

    assert record is not None
    assert record["record_type"] == "settlement"
    assert record["settlement_direction"] == "Up"
    assert record["chainlink_start_price"] is None
    assert record["chainlink_start_status"] == "chainlink_historical_source_not_configured"


def test_chainlink_rtds_message_parser_accepts_btc_usd_tick() -> None:
    module = load_script()

    tick = module.chainlink_tick_from_rtds_message(
        json.dumps(
            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "payload": {
                    "symbol": "btc/usd",
                    "timestamp": 1_900_000_001_234,
                    "value": 100123.45,
                },
            }
        ),
        datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert tick is not None
    assert tick.symbol == "btc/usd"
    assert tick.price == 100123.45
    assert tick.price_ts_ms == 1_900_000_001_234


def test_chainlink_rtds_message_parser_accepts_batched_snapshot() -> None:
    module = load_script()

    tick = module.chainlink_tick_from_rtds_message(
        json.dumps(
            {
                "topic": "crypto_prices_chainlink",
                "type": "subscribe",
                "payload": {
                    "data": [
                        {"timestamp": 1_900_000_001_000, "value": 100000.0},
                        {"timestamp": 1_900_000_002_000, "value": 100010.0},
                    ]
                },
            }
        ),
        datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert tick is not None
    assert tick.price == 100010.0
    assert tick.price_ts_ms == 1_900_000_002_000
