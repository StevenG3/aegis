from __future__ import annotations

from aegis.polymarket_weather_relative_value import (
    WeatherRelativeValueConfig,
    parse_temperature_bucket,
    run_weather_relative_value_firstpass,
    settlement_source_alignment,
    station_from_wunderground_url,
)


def test_parse_temperature_bucket_variants() -> None:
    below = parse_temperature_bucket("83°F or below")
    above = parse_temperature_bucket("86°F or higher")
    assert below is not None
    assert above is not None
    assert below.upper_f == 83
    assert above.lower_f == 86
    span = parse_temperature_bucket("84-85°F")
    assert span is not None
    assert span.lower_f == 84
    assert span.upper_f == 85
    assert span.contains(84)
    assert not span.contains(86)


def test_station_and_settlement_source_alignment() -> None:
    source = "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA"
    assert station_from_wunderground_url(source) == "KLGA"
    event = {
        "resolutionSource": source,
        "description": (
            "The resolution source for this market measures temperatures to whole degrees "
            "Fahrenheit. Revisions will be considered until the first datapoint for the "
            "following date has been published."
        ),
        "markets": [{"resolutionSource": source, "description": "whole degrees Fahrenheit"}],
    }
    alignment = settlement_source_alignment(event)
    assert alignment["aligned"] is True
    assert alignment["station"] == "KLGA"


def test_forecast_issue_must_precede_decision() -> None:
    rows = [
        {
            "event_slug": "highest-temperature-in-nyc-on-june-22-2026",
            "market_slug": "bucket",
            "city": "nyc",
            "station": "KLGA",
            "bucket_label": "84-85°F",
            "decision_ts": 100,
            "forecast_issue_ts": 100,
            "model_probability": 0.8,
            "yes_ask": 0.5,
            "actual_won": True,
        }
    ]
    report = run_weather_relative_value_firstpass(
        rows,
        config=WeatherRelativeValueConfig(min_observations=1),
    )
    assert report["verdict"] == "INSUFFICIENT"
    assert report["coverage"]["excluded_reasons"] == {"forecast_issue_not_before_decision": 1}


def test_weather_relative_value_runner_applies_ask_costs_and_gates() -> None:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "event_slug": f"event-{idx}",
                "market_slug": f"market-{idx}",
                "city": "miami",
                "station": "KMIA",
                "bucket_label": "94-95°F",
                "decision_ts": 1_000 + idx,
                "forecast_issue_ts": 900 + idx,
                "model_probability": 0.80,
                "yes_ask": 0.60,
                "yes_bid": 0.58,
                "actual_won": idx < 8,
            }
        )
    report = run_weather_relative_value_firstpass(
        rows,
        config=WeatherRelativeValueConfig(
            edge_thresholds=(0.05, 0.10),
            min_observations=10,
            pbo_splits=4,
        ),
    )
    assert report["status"] == "OK"
    assert report["standard_metrics"]["trades"] == 12
    assert report["standard_metrics"]["wins"] == 8
    assert report["standard_metrics"]["mean_yes_ask"] == 0.60
