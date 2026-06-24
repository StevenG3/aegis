from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from aegis.polymarket_weather_precision_rule import (
    WeatherPrecisionRuleConfig,
    precision_data_blocked_report,
    run_weather_precision_rule,
)


def test_weather_precision_rule_blocks_forecast_lookahead() -> None:
    rows = [
        {
            "event_slug": "event",
            "market_slug": "market",
            "city": "nyc",
            "station": "KLGA",
            "decision_ts": 100,
            "forecast_issue_ts": 100,
            "entry_window": "morning_local",
            "forecast_yes_probability": 0.70,
            "yes_ask": 0.20,
            "no_ask": 0.82,
            "actual_yes_won": True,
        }
    ]
    report = run_weather_precision_rule(
        rows,
        config=WeatherPrecisionRuleConfig(min_observations=1),
    )
    assert report["state"] == "INSUFFICIENT"
    assert report["data_adequacy"] == "blocked"
    assert report["coverage"]["excluded_reasons"] == {"forecast_issue_not_before_decision": 1}


def test_weather_precision_rule_uses_net_ev_not_win_rate() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "event_slug": f"event-{index}",
                "market_slug": f"market-{index}",
                "city": "miami",
                "station": "KMIA",
                "decision_ts": 10_000 + index,
                "forecast_issue_ts": 9_000 + index,
                "entry_window": "morning_local",
                "forecast_yes_probability": 0.30,
                "yes_ask": 0.70,
                "no_ask": 0.90,
                "actual_yes_won": index >= 9,
            }
        )
    report = run_weather_precision_rule(
        rows,
        config=WeatherPrecisionRuleConfig(
            expensive_yes_min=(0.65,),
            forecast_yes_max=(0.40,),
            yes_ask_max=(),
            tail_yes_ask_max=(),
            tail_yes_min=(),
            entry_windows=("morning_local",),
            min_observations=10,
            pbo_splits=4,
        ),
    )
    assert report["standard_metrics"]["trades"] == 10
    assert report["standard_metrics"]["wins"] == 9
    assert report["standard_metrics"]["win_rate"] == 0.9
    assert report["standard_metrics"]["mean_net_return"] < 0.0
    assert report["state"] == "NO_EDGE"


def test_weather_precision_rule_applies_50bps_slippage_cost() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "event_slug": f"event-{index}",
                "market_slug": f"market-{index}",
                "city": "nyc",
                "station": "KLGA",
                "decision_ts": 20_000 + index,
                "forecast_issue_ts": 19_000 + index,
                "entry_window": "morning_local",
                "forecast_yes_probability": 0.70,
                "yes_ask": 0.20,
                "no_ask": 0.82,
                "actual_yes_won": True,
            }
        )
    report = run_weather_precision_rule(
        rows,
        config=WeatherPrecisionRuleConfig(
            yes_ask_max=(0.30,),
            forecast_yes_min=(0.60,),
            expensive_yes_min=(),
            forecast_yes_max=(),
            tail_yes_ask_max=(),
            tail_yes_min=(),
            entry_windows=("morning_local",),
            min_observations=8,
            pbo_splits=4,
            slippage_rate=0.005,
        ),
    )
    assert report["standard_metrics"]["mean_slippage_cost"] == 0.001
    assert report["standard_metrics"]["wins"] == 8


def test_weather_precision_data_blocked_report_sets_adequacy() -> None:
    report = precision_data_blocked_report(
        reason="missing point-in-time historical forecast archive",
        coverage={"events_found": 4},
        config=WeatherPrecisionRuleConfig(),
    )
    assert report["state"] == "INSUFFICIENT"
    assert report["data_adequacy"] == "blocked"
    assert report["unlock_condition"] == "missing point-in-time historical forecast archive"


def test_weather_precision_evidence_gate_blocks_when_archive_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / (
        "polymarket_weather_precision_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("weather_precision_evidence", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["weather_precision_evidence"] = module
    spec.loader.exec_module(module)

    monkeypatch.delenv("AEGIS_WEATHER_PIT_FORECAST_ARCHIVE", raising=False)
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
    monkeypatch.delenv("OWM_API_KEY", raising=False)
    monkeypatch.delenv("WEATHERAPI_API_KEY", raising=False)
    gate = module._point_in_time_forecast_gate()
    assert gate["available"] is False
    assert "issue_ts < decision_ts" in gate["reason"]

    missing = tmp_path / "missing.jsonl"
    monkeypatch.setenv("AEGIS_WEATHER_PIT_FORECAST_ARCHIVE", str(missing))
    gate = module._point_in_time_forecast_gate()
    assert gate["available"] is False
    assert gate["local_archive_path"] == str(missing)
