from __future__ import annotations

from typing import Any

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec
from aegis.microstructure_perp_runner import run_microstructure_perp_from_spec


def _bars(
    *,
    event_rate: float | None = None,
    include_btc: bool = True,
    spread_bps: float | None = 5.0,
    top_depth_usd: float | None = 100_000.0,
    quote_volume_usd: float | None = 2_000_000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closes = [100.0, 101.0, 103.0, 105.0, 100.0, 96.0, 98.0, 101.0, 97.0, 94.0]
    btc_closes = [100.0, 101.0, 102.0, 106.0, 109.0, 104.0, 100.0, 103.0, 106.0, 101.0]
    for index, close in enumerate(closes):
        row: dict[str, Any] = {
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_700_000_000 + index * 14_400,
            "close": close,
            "open_interest": max(1.0, 1_000.0 - index * 35.0),
            "funding_rate": 0.0002,
            "buy_volume": 10.0,
            "sell_volume": 30.0,
            "survivor_status": "active",
        }
        if spread_bps is not None:
            row["bid_ask_spread_bps"] = spread_bps
        if top_depth_usd is not None:
            row["top_depth_usd"] = top_depth_usd
        if quote_volume_usd is not None:
            row["quote_volume_usd"] = quote_volume_usd
        if include_btc:
            row["btc_close"] = btc_closes[index]
        if event_rate is not None:
            row["order_book_event_rate_per_hour"] = event_rate
        rows.append(row)
    for index in range(10):
        row = {
            "symbol": "DELISTED/USDT:USDT",
            "timestamp": 1_700_000_000 + index * 14_400,
            "close": max(1.0, 20.0 - index),
            "open_interest": max(1.0, 500.0 - index * 20.0),
            "funding_rate": -0.0001,
            "buy_volume": 25.0,
            "sell_volume": 10.0,
            "survivor_status": "delisted",
        }
        if spread_bps is not None:
            row["bid_ask_spread_bps"] = spread_bps
        if top_depth_usd is not None:
            row["top_depth_usd"] = top_depth_usd
        if quote_volume_usd is not None:
            row["quote_volume_usd"] = quote_volume_usd
        if include_btc:
            row["btc_close"] = btc_closes[index]
        if event_rate is not None:
            row["order_book_event_rate_per_hour"] = event_rate
        rows.append(row)
    return rows


def _spec(
    observations: list[dict[str, Any]], extra_params: dict[str, object] | None = None
) -> HypothesisSpec:
    params: dict[str, object] = {
        "observations": observations,
        "grid": {
            "funding_abs_bps": [1.0, 2.0],
            "imbalance_abs": [0.2],
            "oi_drop_abs": [0.02],
            "score_threshold": [1, 2],
        },
        "locked_oos_fraction": 0.8,
        "fold_count": 4,
        "pbo_splits": 4,
        "annualization_periods": 365 * 6,
    }
    if extra_params:
        params.update(extra_params)
    return HypothesisSpec(
        key="microstructure_unit",
        hypothesis_type="event",
        universe=("BTC/USDT:USDT", "DELISTED/USDT:USDT"),
        predeclared_signals=("funding_sign", "oi_price_divergence", "orderflow_imbalance"),
        params=params,
        cost_model={
            "fee_bps": 1.0,
            "slippage_bps": 2.0,
            "funding_bps_per_period": 0.0,
            "funding_label": "perp funding debited from observations",
        },
        benchmark="buy_and_hold",
        data_source="synthetic_offline_microstructure_fixture",
        trial_count_n=8,
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=True,
    )


def test_microstructure_runner_reports_fdr_pbo_and_full_cost_safety() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars()))

    assert payload["status"] == "OK"
    assert payload["candidate_count_n"] == 8
    assert payload["multiple_testing"]["method"] == "BH-FDR + CSCV_PBO"
    assert "pbo_after_survivors" in payload["multiple_testing"]
    assert payload["safety"]["t_plus_1_execution"] is True
    assert payload["safety"]["perp_funding_counted"] is True
    assert payload["standard_metrics"]["net_cost"] > 0
    assert "buy_and_hold" in payload["benchmark_metrics"]
    assert "DELISTED/USDT:USDT" in payload["universe"]["usable_symbols"]
    controls = payload["research_controls"]["control_grid"]
    assert controls[0]["btc_impulse"]["enabled"] is True
    assert controls[0]["liquidity_guard"]["enabled"] is True
    assert payload["event_log"]
    first_log = payload["event_log"][0]
    assert "btc_impulse_pass" in first_log
    assert "funding_cost" in first_log
    assert "fee_cost" in first_log
    assert "slippage_cost" in first_log
    passing_logs = [
        entry
        for entry in payload["event_log"]
        if entry["excluded_reason"] == "" and entry["symbol"] == "BTC/USDT:USDT"
    ]
    assert passing_logs
    assert all(entry["btc_impulse_pass"] is True for entry in passing_logs)
    assert all(entry["liquidity_guard_pass"] is True for entry in passing_logs)
    assert any(entry["oi_price_divergence"] is True for entry in passing_logs)
    assert any(abs(float(entry["order_flow_imbalance"])) >= 0.2 for entry in passing_logs)


def test_microstructure_runner_counts_predeclared_control_grid() -> None:
    payload = run_microstructure_perp_from_spec(
        _spec(
            _bars(),
            extra_params={
                "control_grid": [
                    {
                        "name": "impulse_loose",
                        "btc_impulse": {
                            "enabled": True,
                            "lookback_bars": 3,
                            "return_threshold": 0.01,
                            "zscore_threshold": 0.0,
                        },
                        "liquidity_guard": {
                            "enabled": True,
                            "max_spread_bps": 25.0,
                            "min_top_depth_usd": 50_000.0,
                            "min_quote_volume_usd": 1_000_000.0,
                        },
                        "entry_window": {},
                    },
                    {
                        "name": "impulse_strict",
                        "btc_impulse": {
                            "enabled": True,
                            "lookback_bars": 6,
                            "return_threshold": 0.02,
                            "zscore_threshold": 1.0,
                        },
                        "liquidity_guard": {
                            "enabled": True,
                            "max_spread_bps": 25.0,
                            "min_top_depth_usd": 50_000.0,
                            "min_quote_volume_usd": 1_000_000.0,
                        },
                        "entry_window": {},
                    },
                ]
            },
        )
    )

    assert payload["candidate_count_n"] == 16
    assert payload["multiple_testing"]["control_grid_candidates"] == 2
    assert len(payload["research_controls"]["control_grid"]) == 2
    assert {entry["control"] for entry in payload["event_log"]} == {
        "impulse_loose",
        "impulse_strict",
    }


def test_microstructure_runner_data_blocks_high_orderbook_event_rate() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(event_rate=20_000.0)))

    assert payload["status"] == "INSUFFICIENT"
    assert "data-block" in payload["reason"]
    assert "BTC/USDT:USDT" in payload["safety"]["excluded_data_blocked_symbols"]
    assert payload["event_log"][0]["excluded_reason"] == "orderbook_event_rate_data_blocked"


def test_microstructure_runner_records_missing_btc_reference() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(include_btc=False)))

    assert payload["status"] == "OK"
    assert {entry["excluded_reason"] for entry in payload["event_log"]} == {"btc_close_missing"}
    assert all(entry["btc_impulse_pass"] is False for entry in payload["event_log"])


def test_microstructure_runner_records_spread_guard_fail() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(spread_bps=100.0)))

    assert payload["status"] == "OK"
    reasons = {entry["excluded_reason"] for entry in payload["event_log"]}
    assert "spread_guard_fail" in reasons
    assert all(entry["liquidity_guard_pass"] is False for entry in payload["event_log"])


def test_microstructure_runner_records_depth_guard_fail() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(top_depth_usd=10_000.0)))

    assert payload["status"] == "OK"
    reasons = {entry["excluded_reason"] for entry in payload["event_log"]}
    assert "top_depth_guard_fail" in reasons
    assert all(entry["liquidity_guard_pass"] is False for entry in payload["event_log"])


def test_microstructure_runner_records_quote_volume_guard_fail() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(quote_volume_usd=100_000.0)))

    assert payload["status"] == "OK"
    reasons = {entry["excluded_reason"] for entry in payload["event_log"]}
    assert "quote_volume_guard_fail" in reasons
    assert all(entry["liquidity_guard_pass"] is False for entry in payload["event_log"])


def test_microstructure_runner_data_blocks_globally_missing_spread_and_depth() -> None:
    payload = run_microstructure_perp_from_spec(
        _spec(_bars(spread_bps=None, top_depth_usd=None))
    )

    assert payload["status"] == "OK"
    availability = payload["research_controls"]["liquidity_data_availability"]
    assert availability["spread_data_blocked"] is True
    assert availability["top_depth_data_blocked"] is True
    passing_logs = [entry for entry in payload["event_log"] if entry["excluded_reason"] == ""]
    assert passing_logs
    assert all(entry["liquidity_guard_pass"] is True for entry in passing_logs)
    assert all(
        entry["liquidity_data_blocked"] == ("bid_ask_spread_bps", "top_depth_usd")
        for entry in passing_logs
    )


def test_microstructure_runner_uses_t_plus_1_entry_inside_predeclared_window() -> None:
    observations = _bars()
    start = observations[5]["timestamp"]
    end = observations[6]["timestamp"]
    payload = _spec(
        observations,
        extra_params={"entry_window": {"start_timestamp": start, "end_timestamp": end}},
    )

    result = run_microstructure_perp_from_spec(payload)

    logs = result["event_log"]
    assert any(entry["excluded_reason"] == "outside_predeclared_entry_window" for entry in logs)
    inside = [
        entry
        for entry in logs
        if start <= entry["timestamp"] <= end and entry["excluded_reason"] == ""
    ]
    assert inside
    assert all(entry["entry_timestamp"] > entry["timestamp"] for entry in inside)
    assert all(entry["exit_timestamp"] > entry["entry_timestamp"] for entry in inside)


def test_microstructure_runner_rejects_entries_outside_predeclared_window() -> None:
    observations = _bars()
    payload = _spec(
        observations,
        extra_params={
            "entry_window": {
                "start_timestamp": observations[-1]["timestamp"] + 1,
                "end_timestamp": observations[-1]["timestamp"] + 2,
            }
        },
    )

    result = run_microstructure_perp_from_spec(payload)

    assert result["status"] == "OK"
    assert {
        entry["excluded_reason"]
        for entry in result["event_log"]
    } == {"outside_predeclared_entry_window"}
