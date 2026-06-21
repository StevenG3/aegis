from __future__ import annotations

import pytest

from aegis.polymarket_relative_value import (
    RelativeValueConfig,
    run_relative_value_calibration,
)


def test_well_calibrated_bucket_is_no_edge() -> None:
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.extend(
            _market_rows(
                index,
                up_mid=0.75,
                down_mid=0.25,
                settlement="Up" if index < 15 else "Down",
            )
        )

    report = run_relative_value_calibration(
        rows,
        config=RelativeValueConfig(
            time_to_close_seconds=(60,),
            probability_bins=((0.70, 0.80),),
            min_markets=20,
            min_observations=10,
            min_bucket_observations=5,
        ),
    )

    assert report["verdict"] == "NO_EDGE"
    bucket = report["calibration_buckets"][0]
    assert bucket["n"] == 20
    assert bucket["wins"] == 15
    assert bucket["calibrated"] is True
    assert bucket["fdr_miscalibrated"] is False


def test_missing_minimum_observations_is_insufficient() -> None:
    report = run_relative_value_calibration(
        [],
        config=RelativeValueConfig(min_markets=1, min_observations=1),
    )

    assert report["verdict"] == "INSUFFICIENT"
    assert "no forward" in str(report["reason"])


def test_target_time_calibration_does_not_use_later_price_snapshot() -> None:
    rows = _market_rows(0, up_mid=0.75, down_mid=0.25, settlement="Up")
    future_rows = rows + [
        _snapshot(
            "btc-updown-5m-1900000000",
            1_900_000_000,
            1_900_000_300,
            seconds_to_close=5,
            outcome="Up",
            bid=0.01,
            ask=0.03,
        ),
        _snapshot(
            "btc-updown-5m-1900000000",
            1_900_000_000,
            1_900_000_300,
            seconds_to_close=5,
            outcome="Down",
            bid=0.97,
            ask=0.99,
        ),
    ]
    config = RelativeValueConfig(
        time_to_close_seconds=(60,),
        probability_bins=((0.70, 0.80),),
        target_tolerance_seconds=10,
        min_markets=1,
        min_observations=1,
        min_bucket_observations=1,
    )

    original = run_relative_value_calibration(rows, config=config)
    mutated = run_relative_value_calibration(future_rows, config=config)

    assert mutated["calibration_buckets"][0]["mean_implied_probability"] == pytest.approx(
        original["calibration_buckets"][0]["mean_implied_probability"]
    )


def test_positive_miscalibration_below_ask_is_not_exploitable_edge() -> None:
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.extend(
            _market_rows(
                index,
                up_mid=0.55,
                down_mid=0.45,
                up_bid=0.15,
                up_ask=0.95,
                settlement="Up" if index < 18 else "Down",
            )
        )

    report = run_relative_value_calibration(
        rows,
        config=RelativeValueConfig(
            time_to_close_seconds=(60,),
            probability_bins=((0.50, 0.60),),
            min_markets=20,
            min_observations=10,
            min_bucket_observations=5,
        ),
    )

    assert report["verdict"] == "NO_EDGE"
    bucket = report["calibration_buckets"][0]
    assert bucket["fdr_miscalibrated"] is True
    assert bucket["favorite_edge_after_ask"] < 0
    assert bucket["underdog_edge_after_ask"] < 0
    assert bucket["max_edge_after_ask"] < 0


def test_overpriced_favorite_can_be_suggestive_for_underdog_after_ask() -> None:
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.extend(
            _market_rows(
                index,
                up_mid=0.65,
                down_mid=0.35,
                settlement="Up" if index < 2 else "Down",
            )
        )

    report = run_relative_value_calibration(
        rows,
        config=RelativeValueConfig(
            time_to_close_seconds=(60,),
            probability_bins=((0.60, 0.70),),
            min_markets=20,
            min_observations=10,
            min_bucket_observations=5,
        ),
    )

    bucket = report["calibration_buckets"][0]
    assert bucket["fdr_miscalibrated"] is True
    assert bucket["underdog_edge_after_ask"] > 0
    assert report["verdict"] == "SUGGESTIVE_NEEDS_PAID_CONFIRM"


def _market_rows(
    index: int,
    *,
    up_mid: float,
    down_mid: float,
    settlement: str,
    up_bid: float | None = None,
    up_ask: float | None = None,
) -> list[dict[str, object]]:
    start_ts = 1_900_000_000 + index * 300
    end_ts = start_ts + 300
    slug = f"btc-updown-5m-{start_ts}"
    bid_up = up_mid - 0.01 if up_bid is None else up_bid
    ask_up = up_mid + 0.01 if up_ask is None else up_ask
    return [
        _snapshot(
            slug,
            start_ts,
            end_ts,
            seconds_to_close=60,
            outcome="Up",
            bid=bid_up,
            ask=ask_up,
        ),
        _snapshot(
            slug,
            start_ts,
            end_ts,
            seconds_to_close=60,
            outcome="Down",
            bid=down_mid - 0.01,
            ask=down_mid + 0.01,
        ),
        {
            "record_type": "settlement",
            "slug": slug,
            "condition_id": f"condition-{index}",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "settlement_direction": settlement,
        },
    ]


def _snapshot(
    slug: str,
    start_ts: int,
    end_ts: int,
    *,
    seconds_to_close: int,
    outcome: str,
    bid: float,
    ask: float,
) -> dict[str, object]:
    timestamp_ms = (end_ts - seconds_to_close) * 1000
    return {
        "record_type": "snapshot",
        "slug": slug,
        "condition_id": f"condition-{slug}",
        "outcome": outcome,
        "captured_ts_ms": timestamp_ms,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "seconds_to_close": seconds_to_close,
        "best_bid": str(bid),
        "best_ask": str(ask),
        "bid_levels": [{"price": str(bid), "size": "100"}],
        "ask_levels": [{"price": str(ask), "size": "100"}],
    }
