from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from aegis.domestic_options_retest import (
    DeltaNeutralTrade,
    DomesticOptionsConfig,
    OptionTrade,
    delta_neutral_return_after_costs,
    evaluate_data_feasibility,
    option_buyer_return_after_costs,
    requirement_keys,
    run_synthetic_mechanism_verdict,
    trial_count,
)


def test_data_gate_fails_closed_when_free_stack_lacks_required_option_data() -> None:
    report = evaluate_data_feasibility(
        {
            "underlying_history": True,
            "commodity_daily_underlying": True,
            "fee_schedule": True,
        }
    )

    assert report["verdict"] == "INSUFFICIENT"
    mechanisms = cast(Mapping[str, Mapping[str, object]], report["mechanisms"])
    commodity = mechanisms["commodity_adx_option_buyer"]
    mo_im = mechanisms["mo_im_delta_neutral"]
    assert commodity["verdict"] == "INSUFFICIENT"
    commodity_missing = cast(list[str], commodity["missing_requirements"])
    mo_im_missing = cast(list[str], mo_im["missing_requirements"])
    assert "pit_option_chain" in commodity_missing
    assert "option_bid_ask_quotes" in commodity_missing
    assert mo_im["verdict"] == "INSUFFICIENT"
    assert "mo_intraday_option_quotes" in mo_im_missing
    assert "rebalance_execution_costs" in mo_im_missing


def test_all_requirements_available_marks_data_ready_without_edge_claim() -> None:
    available: dict[str, bool] = {key: True for key in requirement_keys("mo_im_delta_neutral")}

    report = evaluate_data_feasibility(available)

    assert report["verdict"] == "DATA_READY"
    assert report["max_positive_verdict"] == "SUGGESTIVE"
    assert report["source_report_parameters_are_in_sample"] is True


def test_report_optimized_parameters_are_only_grid_points() -> None:
    config = DomesticOptionsConfig()

    assert 22.0 in config.adx_thresholds
    assert 13 in config.ema_fast_windows
    assert 34 in config.ema_slow_windows
    assert 0.35 in config.delta_min_values
    assert 0.65 in config.delta_max_values
    assert 0.40 in config.hard_stop_values
    assert 1.5 in config.atr_stop_multipliers
    assert 30 in config.min_dte_values
    assert 0.08 in config.iv_drop_stop_values
    assert trial_count(config, "commodity_adx_option_buyer", product_count=8) > 8
    assert trial_count(config, "mo_im_delta_neutral", product_count=8) > trial_count(
        config,
        "commodity_adx_option_buyer",
        product_count=8,
    )


def test_option_buyer_uses_t_plus_one_and_ask_to_bid_costs() -> None:
    trade = OptionTrade(
        symbol="SHFE.au.synthetic",
        signal_timestamp=1,
        entry_timestamp=2,
        exit_timestamp=10,
        entry_ask=10.0,
        exit_bid=12.0,
        quantity=2,
        multiplier=100.0,
        fee=6.0,
        slippage=4.0,
    )

    assert option_buyer_return_after_costs(trade) == pytest.approx((2400 - 2000 - 10) / 2000)


def test_option_buyer_rejects_same_bar_entry() -> None:
    with pytest.raises(ValueError, match="t\\+1"):
        option_buyer_return_after_costs(
            OptionTrade(
                symbol="CZCE.TA.synthetic",
                signal_timestamp=5,
                entry_timestamp=5,
                exit_timestamp=8,
                entry_ask=10.0,
                exit_bid=11.0,
            )
        )


def test_delta_neutral_counts_rebalance_costs_and_margin_capital() -> None:
    trade = DeltaNeutralTrade(
        symbol="CFFEX.MO.synthetic",
        signal_timestamp=1,
        entry_timestamp=2,
        exit_timestamp=3,
        option_entry_ask=100.0,
        option_exit_bid=130.0,
        option_quantity=4,
        option_multiplier=100.0,
        hedge_pnl=800.0,
        hedge_fee=120.0,
        hedge_slippage=80.0,
        margin_capital=80_000.0,
        option_fee=20.0,
        option_slippage=30.0,
    )

    expected = ((130 - 100) * 4 * 100 + 800 - 120 - 80 - 20 - 30) / 80_000
    assert delta_neutral_return_after_costs(trade) == pytest.approx(expected)


def test_synthetic_runner_reports_insufficient_when_pbo_is_invalid() -> None:
    report = run_synthetic_mechanism_verdict(
        [[0.01] * 30, [0.0] * 30],
        mechanism="commodity_adx_option_buyer",
        config=DomesticOptionsConfig(pbo_splits=32, min_trades=30),
    )

    assert report.verdict == "INSUFFICIENT"
    assert "invalid PBO" in report.reason
