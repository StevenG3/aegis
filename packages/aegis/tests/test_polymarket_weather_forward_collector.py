from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aegis.gefs_weather_probability import GefsCycle, StationForecastSample


def _load_collector() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "polymarket_weather_forward_collector.py"
    spec = importlib.util.spec_from_file_location(
        "polymarket_weather_forward_collector",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["polymarket_weather_forward_collector"] = module
    spec.loader.exec_module(module)
    return module


def _event() -> dict[str, Any]:
    source = "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA"
    return {
        "slug": "highest-temperature-in-nyc-on-june-24-2026",
        "closed": False,
        "endDate": "2026-06-25T04:00:00Z",
        "resolutionSource": source,
        "description": (
            "Temperatures are measured in whole degrees Fahrenheit. Revisions are accepted "
            "until the first datapoint for the following date has been finalized."
        ),
        "markets": [
            {
                "slug": "highest-temperature-in-nyc-on-june-24-2026-84-85f",
                "question": "Will the high be between 84-85°F?",
                "description": "whole degrees Fahrenheit",
                "resolutionSource": source,
                "clobTokenIds": '["yes-token", "no-token"]',
            }
        ],
    }


def test_weather_forward_collector_writes_true_book_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _load_collector()
    cycle = GefsCycle(issue_time=datetime(2026, 6, 23, 18, tzinfo=UTC), cycle_hour=18)
    samples = [
        StationForecastSample(
            "gep01",
            daily_max_f=84.2,
            rounded_daily_max_f=84,
            lead_hours_used=(6,),
        ),
        StationForecastSample(
            "gep02",
            daily_max_f=85.2,
            rounded_daily_max_f=85,
            lead_hours_used=(6,),
        ),
        StationForecastSample(
            "gep03",
            daily_max_f=90.0,
            rounded_daily_max_f=90,
            lead_hours_used=(6,),
        ),
    ]

    monkeypatch.setattr(module, "_fetch_event_by_slug", lambda _slug: _event())
    monkeypatch.setattr(
        module,
        "station_daily_samples_from_gefs",
        lambda **_kwargs: (samples, cycle, (6,)),
    )

    def fake_get_json(url: str) -> dict[str, object]:
        if "yes-token" in url:
            return {"asks": [{"price": "0.40", "size": "10"}, {"price": "0.50", "size": "5"}]}
        if "no-token" in url:
            return {"asks": [{"price": "0.55", "size": "8"}]}
        return {}

    monkeypatch.setattr(module, "_get_json", fake_get_json)
    rows, summary_path = module.collect_weather_forward_snapshot(
        output_dir=tmp_path,
        target_dates=(date(2026, 6, 24),),
        city_slugs=("nyc",),
        members=("gep01", "gep02", "gep03"),
        max_workers=1,
    )

    assert rows == 1
    assert summary_path.exists()
    jsonl = next(tmp_path.glob("date=*/hour=*/polymarket_weather_forward.jsonl"))
    line = jsonl.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["yes_ask"] == 0.4
    assert row["no_ask"] == 0.55
    assert row["forecast_issue_ts"] < row["decision_ts"]
    assert row["market_end_ts"] == 1782360000
    assert row["price_source"] == "current CLOB /book true ask/depth snapshot"


def test_book_for_token_uses_lowest_ask_and_depth(monkeypatch: Any) -> None:
    module = _load_collector()
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda _url: {
            "asks": [
                {"price": "0.70", "size": "2"},
                {"price": "0.60", "size": "3"},
            ],
            "bids": [
                {"price": "0.20", "size": "5"},
                {"price": "0.30", "size": "7"},
            ],
        },
    )

    book = module._book_for_token("token")

    assert book.best_price == 0.60
    assert book.best_size == 3.0
    assert book.depth_usd == pytest.approx(3.2)
    assert book.levels == 2
    assert book.best_bid == 0.30
    assert book.best_bid_size == 7.0
    assert book.bid_depth_usd == pytest.approx(3.1)
    assert book.bid_levels == 2
