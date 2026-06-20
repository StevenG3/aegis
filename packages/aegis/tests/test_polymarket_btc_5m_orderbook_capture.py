from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "polymarket_btc_5m_orderbook_capture.py"
    spec = importlib.util.spec_from_file_location("polymarket_book_capture_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["polymarket_book_capture_script"] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def iter_closed_markets(
        self,
        *,
        limit: int,
        max_markets: int,
        sleep_seconds: float,
        order: str,
        ascending: bool,
        closed: bool,
    ) -> list[dict[str, object]]:
        _ = limit, max_markets, sleep_seconds, order, ascending, closed
        return [
            {
                "conditionId": "condition-1",
                "slug": "btc-updown-5m-1800000000",
                "question": "Bitcoin Up or Down - synthetic 5m",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.5", "0.5"]',
                "clobTokenIds": '["up-token", "down-token"]',
                "endDate": "2099-01-15T08:05:00Z",
                "closed": False,
            },
        ]

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
        return [
            {
                "slug": "btc-updown-5m-1800000000",
                "markets": [
                    {
                        "conditionId": "condition-1",
                        "slug": "btc-updown-5m-1800000000",
                        "question": "Bitcoin Up or Down - synthetic 5m",
                        "outcomes": '["Up", "Down"]',
                        "outcomePrices": '["0.5", "0.5"]',
                        "clobTokenIds": '["up-token", "down-token"]',
                        "endDate": "2099-01-15T08:05:00Z",
                        "closed": False,
                    }
                ],
            },
            {
                "conditionId": "condition-2",
                "slug": "not-btc",
                "question": "Some other market",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.5", "0.5"]',
                "clobTokenIds": '["yes-token", "no-token"]',
                "endDate": "2099-01-15T08:05:00Z",
                "closed": False,
            },
        ]

    def get_order_book(self, token_id: str) -> dict[str, object]:
        if token_id == "up-token":
            return {
                "bids": [{"price": "0.80", "size": "10"}, {"price": "0.79", "size": "5"}],
                "asks": [{"price": "0.82", "size": "7"}, {"price": "0.83", "size": "3"}],
            }
        return {
            "bids": [{"price": "0.18", "size": "8"}],
            "asks": [{"price": "0.20", "size": "4"}],
        }


def test_capture_writes_private_jsonl_order_book_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setattr(module, "PolymarketDataApiClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polymarket_btc_5m_orderbook_capture.py",
            "--max-iterations",
            "1",
            "--interval-seconds",
            "0",
            "--duration-seconds",
            "0",
            "--sleep-seconds",
            "0",
            "--depth-levels",
            "2",
        ],
    )

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    output_dir = Path(summary["output_dir"])
    assert output_dir.is_relative_to(private_root / "incubating" / "olympus61")
    assert summary["records_written"] == 2
    assert summary["wallet_order_account_connected"] is False
    jsonl_path = Path(summary["last_jsonl"])
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    up = next(row for row in rows if row["token_id"] == "up-token")
    assert up["outcome"] == "Up"
    assert up["best_bid"] == "0.8"
    assert up["best_ask"] == "0.82"
    assert up["spread_bps"] == "246.9135802469135802469135802"
    assert up["bid_depth_usd_top_n"] == "11.95"
    assert up["ask_depth_usd_top_n"] == "8.23"
    assert up["source"] == "polymarket_clob_book_public_read_only"


def test_snapshot_record_handles_missing_side_without_fake_depth() -> None:
    module = load_script()
    record = module.build_snapshot_record(
        module.ActiveBtcMarket(
            condition_id="condition-1",
            slug="btc-updown-5m-1800000000",
            title="Bitcoin Up or Down - synthetic 5m",
            end_ts=None,
            outcomes=("Up", "Down"),
            token_ids=("up-token", "down-token"),
        ),
        outcome="Up",
        token_id="up-token",
        raw_book={"bids": [{"price": "0.4", "size": "2"}], "asks": []},
        captured_at=module.datetime(2027, 1, 15, tzinfo=module.UTC),
        depth_levels=5,
    )

    assert record["best_bid"] == "0.4"
    assert record["best_ask"] is None
    assert record["spread_bps"] is None
    assert record["ask_depth_usd_top_n"] == "0"
