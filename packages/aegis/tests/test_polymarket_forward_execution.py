from __future__ import annotations

from aegis.polymarket_forward_execution import (
    ForwardExecutionConfig,
    run_forward_execution_backtest,
)


def test_forward_execution_requires_chainlink_reference_prices() -> None:
    rows = [
        {
            "record_type": "snapshot",
            "slug": "btc-updown-5m-1900000000",
            "condition_id": "condition-1",
            "outcome": "Up",
            "captured_ts_ms": 1_900_000_070_000,
            "start_ts": 1_900_000_000,
            "end_ts": 1_900_000_300,
            "seconds_to_close": 80,
            "actual_settlement_source": "https://data.chain.link/streams/btc-usd",
            "best_ask": "0.88",
            "ask_levels": [{"price": "0.88", "size": "100"}],
        },
        {
            "record_type": "settlement",
            "slug": "btc-updown-5m-1900000000",
            "condition_id": "condition-1",
            "start_ts": 1_900_000_000,
            "end_ts": 1_900_000_300,
            "actual_settlement_source": "https://data.chain.link/streams/btc-usd",
            "settlement_direction": "Up",
        },
    ]

    report = run_forward_execution_backtest(
        rows,
        config=ForwardExecutionConfig(min_markets=1),
    )

    assert report["status"] == "INSUFFICIENT"
    assert report["verdict"] == "INSUFFICIENT"
    assert "Chainlink" in str(report["reason"])


def test_forward_execution_fail_closed_without_verified_settlement_source() -> None:
    rows = [
        _snapshot(
            "btc-updown-5m-1900000000",
            1_900_000_000,
            1_900_000_300,
            timestamp_ms=1_900_000_220_000,
            seconds_to_close=80,
            outcome="Up",
            chainlink_start=100_000.0,
            chainlink_reference=100_100.0,
            ask="0.86",
            bid="0.85",
            settlement_source=None,
        ),
        {
            "record_type": "settlement",
            "slug": "btc-updown-5m-1900000000",
            "condition_id": "condition-1",
            "start_ts": 1_900_000_000,
            "end_ts": 1_900_000_300,
            "settlement_direction": "Up",
        },
    ]

    report = run_forward_execution_backtest(rows, config=ForwardExecutionConfig(min_markets=1))

    assert report["status"] == "INSUFFICIENT"
    assert report["coverage"]["missing_settlement_source_markets"] == 1
    assert "verified Chainlink Data Streams settlement source" in str(report["reason"])


def test_forward_execution_fail_closed_on_mismatched_settlement_source() -> None:
    rows = [
        _snapshot(
            "btc-updown-5m-1900000000",
            1_900_000_000,
            1_900_000_300,
            timestamp_ms=1_900_000_220_000,
            seconds_to_close=80,
            outcome="Up",
            chainlink_start=100_000.0,
            chainlink_reference=100_100.0,
            ask="0.86",
            bid="0.85",
            settlement_source="https://example.com/not-the-settlement-source",
        ),
        {
            "record_type": "settlement",
            "slug": "btc-updown-5m-1900000000",
            "condition_id": "condition-1",
            "start_ts": 1_900_000_000,
            "end_ts": 1_900_000_300,
            "actual_settlement_source": "https://example.com/not-the-settlement-source",
            "settlement_direction": "Up",
        },
    ]

    report = run_forward_execution_backtest(rows, config=ForwardExecutionConfig(min_markets=1))

    assert report["status"] == "INSUFFICIENT"
    assert report["coverage"]["mismatched_settlement_source_markets"] == 1
    assert report["coverage"]["markets"] == 0


def test_forward_execution_uses_delayed_ask_fill_and_preclose_bid_exit() -> None:
    rows: list[dict[str, object]] = []
    for offset in range(4):
        start_ts = 1_900_000_000 + offset * 300
        end_ts = start_ts + 300
        slug = f"btc-updown-5m-{start_ts}"
        rows.extend(
            [
                _snapshot(
                    slug,
                    start_ts,
                    end_ts,
                    timestamp_ms=(end_ts - 80) * 1000,
                    seconds_to_close=80,
                    outcome="Up",
                    chainlink_start=100_000.0,
                    chainlink_reference=100_100.0,
                    ask="0.85",
                    bid="0.84",
                ),
                _snapshot(
                    slug,
                    start_ts,
                    end_ts,
                    timestamp_ms=(end_ts - 79) * 1000,
                    seconds_to_close=79,
                    outcome="Up",
                    chainlink_start=100_000.0,
                    chainlink_reference=100_100.0,
                    ask="0.86",
                    bid="0.85",
                ),
                _snapshot(
                    slug,
                    start_ts,
                    end_ts,
                    timestamp_ms=(end_ts - 20) * 1000,
                    seconds_to_close=20,
                    outcome="Up",
                    chainlink_start=100_000.0,
                    chainlink_reference=100_100.0,
                    ask="0.97",
                    bid="0.96",
                ),
                {
                    "record_type": "settlement",
                    "slug": slug,
                    "condition_id": f"condition-{offset}",
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "actual_settlement_source": "https://data.chain.link/streams/btc-usd",
                    "settlement_direction": "Up",
                },
            ]
        )

    report = run_forward_execution_backtest(
        rows,
        config=ForwardExecutionConfig(notional_usd=25.0, min_markets=4, venue_geoblocked=True),
    )

    assert report["status"] == "OK"
    best = report["best_preclose_candidate"]
    assert best["trades"] == 4
    trade = best["sample_trades"][0]
    assert trade["entry_timestamp_ms"] > trade["signal_timestamp_ms"]
    assert trade["entry_average_price"] == 0.86
    assert trade["exit_average_price"] == 0.96
    assert trade["exit_mode"] == "preclose_30s"


def test_forward_execution_can_use_separate_chainlink_tick_rows() -> None:
    start_ts = 1_900_100_000
    end_ts = start_ts + 300
    slug = f"btc-updown-5m-{start_ts}"
    rows = [
        {
            "record_type": "chainlink_price",
            "symbol": "btc/usd",
            "source": "polymarket_rtds_crypto_prices_chainlink",
            "price": 100_000.0,
            "price_ts_ms": start_ts * 1000,
        },
        {
            "record_type": "chainlink_price",
            "symbol": "btc/usd",
            "source": "polymarket_rtds_crypto_prices_chainlink",
            "price": 100_080.0,
            "price_ts_ms": (end_ts - 80) * 1000,
        },
        _snapshot_without_chainlink(
            slug,
            start_ts,
            end_ts,
            timestamp_ms=(end_ts - 80) * 1000,
            seconds_to_close=80,
            outcome="Up",
            ask="0.86",
            bid="0.85",
        ),
        _snapshot_without_chainlink(
            slug,
            start_ts,
            end_ts,
            timestamp_ms=(end_ts - 79) * 1000,
            seconds_to_close=79,
            outcome="Up",
            ask="0.87",
            bid="0.86",
        ),
        _snapshot_without_chainlink(
            slug,
            start_ts,
            end_ts,
            timestamp_ms=(end_ts - 20) * 1000,
            seconds_to_close=20,
            outcome="Up",
            ask="0.97",
            bid="0.96",
        ),
        {
            "record_type": "settlement",
            "slug": slug,
            "condition_id": "condition-tick",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "actual_settlement_source": "https://data.chain.link/streams/btc-usd",
            "settlement_direction": "Up",
        },
    ]

    report = run_forward_execution_backtest(
        rows,
        config=ForwardExecutionConfig(notional_usd=25.0, min_markets=1, venue_geoblocked=True),
    )

    assert report["status"] == "OK"
    assert report["coverage"]["chainlink_ready_markets"] == 1


def _snapshot(
    slug: str,
    start_ts: int,
    end_ts: int,
    *,
    timestamp_ms: int,
    seconds_to_close: int,
    outcome: str,
    chainlink_start: float,
    chainlink_reference: float,
    ask: str,
    bid: str,
    settlement_source: str | None = "https://data.chain.link/streams/btc-usd",
) -> dict[str, object]:
    row: dict[str, object] = {
        "record_type": "snapshot",
        "slug": slug,
        "condition_id": f"condition-{slug}",
        "outcome": outcome,
        "captured_ts_ms": timestamp_ms,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "seconds_to_close": seconds_to_close,
        "chainlink_start_price": chainlink_start,
        "chainlink_reference_price": chainlink_reference,
        "best_ask": ask,
        "best_bid": bid,
        "ask_levels": [{"price": ask, "size": "100"}],
        "bid_levels": [{"price": bid, "size": "100"}],
    }
    if settlement_source is not None:
        row["actual_settlement_source"] = settlement_source
    return row


def _snapshot_without_chainlink(
    slug: str,
    start_ts: int,
    end_ts: int,
    *,
    timestamp_ms: int,
    seconds_to_close: int,
    outcome: str,
    ask: str,
    bid: str,
) -> dict[str, object]:
    return {
        "record_type": "snapshot",
        "slug": slug,
        "condition_id": f"condition-{slug}",
        "outcome": outcome,
        "captured_ts_ms": timestamp_ms,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "seconds_to_close": seconds_to_close,
        "actual_settlement_source": "https://data.chain.link/streams/btc-usd",
        "best_ask": ask,
        "best_bid": bid,
        "ask_levels": [{"price": ask, "size": "100"}],
        "bid_levels": [{"price": bid, "size": "100"}],
    }
