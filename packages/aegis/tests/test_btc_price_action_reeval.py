from __future__ import annotations

from typing import cast

import pytest

from aegis.btc_price_action_reeval import (
    DEFAULT_PRICE_ACTION_COST_MODEL,
    ExternalContext,
    PriceActionConfig,
    PriceActionParams,
    _execute_trade,
    predeclared_price_action_params,
    report_to_dict,
    run_btc_price_action_reeval,
    simulate_price_action,
)
from aegis.combo_indicator_search import ComboBar, ComboCostModel


def _bar(
    index: int,
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
) -> ComboBar:
    price_open = close if open_ is None else open_
    return ComboBar(
        timestamp=index,
        open=float(price_open),
        high=float(high if high is not None else max(price_open, close) * 1.01),
        low=float(low if low is not None else min(price_open, close) * 0.99),
        close=float(close),
        volume=float(volume),
    )


def _config() -> PriceActionConfig:
    return PriceActionConfig(
        train_bars=30,
        test_bars=10,
        step_bars=10,
        locked_oos_fraction=0.30,
        min_is_folds=3,
        min_trades=1,
        lookbacks=(5, 8),
        risk_rewards=(1.0, 1.2),
        max_holds=(4,),
        sma_window=5,
        daily_sma_window=10,
        atr_period=5,
        volume_window=3,
        risk_diff_bootstrap_samples=50,
        risk_diff_bootstrap_block_bars=5,
    )


def test_predeclared_price_action_grid_counts_all_parameter_trials() -> None:
    params = predeclared_price_action_params(_config())

    assert len(params) == 4
    assert {param.key for param in params} == {
        "lookback_5_rr_1p0_hold_4",
        "lookback_5_rr_1p2_hold_4",
        "lookback_8_rr_1p0_hold_4",
        "lookback_8_rr_1p2_hold_4",
    }


def test_signal_enters_next_bar_open_not_confirmation_close() -> None:
    bars = [_bar(index, 100 + index * 0.1, volume=1000) for index in range(20)]
    bars[13] = _bar(13, 111.0, high=112.0, low=109.0, volume=2000)
    bars[14] = _bar(14, 111.5, high=112.0, low=101.5, volume=2200)
    bars[15] = _bar(15, 120.0, open_=115.0, high=121.0, low=114.0, volume=1000)
    params = PriceActionParams(lookback=5, risk_reward=1.0, max_hold=4)

    result = simulate_price_action(
        bars,
        params,
        start=15,
        end=19,
        config=_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert result.trades
    trade = result.trades[0]
    assert trade.signal_index == 14
    assert trade.entry_index == 15
    assert trade.entry == 115.0
    assert trade.entry != bars[14].close


def test_same_bar_stop_and_target_is_conservative_stop() -> None:
    bars = [
        _bar(0, 100),
        _bar(1, 100, open_=100, high=106, low=94),
    ]
    signal: dict[str, int | float | str] = {
        "side": 1,
        "setup": "synthetic",
        "signal_index": 0,
        "stop": 95.0,
        "risk_reward": 1.0,
    }

    trade = _execute_trade(
        bars,
        ExternalContext(),
        signal,
        entry_index=1,
        max_exit_index=1,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert trade.exit_reason == "stop"
    assert trade.exit_price == 95.0
    assert trade.net_return == pytest.approx(-0.05)


def test_short_trade_debits_positive_funding() -> None:
    bars = [
        _bar(0, 100),
        _bar(1, 100, open_=100, high=101, low=99),
        _bar(2, 98, open_=98, high=99, low=96),
    ]
    external = ExternalContext(funding_by_timestamp={1: 0.001, 2: 0.001})
    signal: dict[str, int | float | str] = {
        "side": -1,
        "setup": "synthetic",
        "signal_index": 0,
        "stop": 102.0,
        "risk_reward": 1.0,
    }

    trade = _execute_trade(
        bars,
        external,
        signal,
        entry_index=1,
        max_exit_index=2,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert trade.funding_cost == pytest.approx(0.002)
    assert trade.net_return == pytest.approx(trade.gross_return - 0.002)


def test_report_uses_bh_fdr_and_reports_private_boundary() -> None:
    bars = [_bar(index, 100 + index * 0.2, volume=1000 + index) for index in range(150)]
    report = run_btc_price_action_reeval(
        bars,
        config=_config(),
        cost_model=DEFAULT_PRICE_ACTION_COST_MODEL,
    )
    payload = report_to_dict(report)
    safety = cast(dict[str, object], payload["safety"])

    assert report.candidate_count_n == 4
    assert report.multiple_testing["candidate_count_n"] == 4
    assert report.multiple_testing["risk_diff_test"] == (
        "paired block bootstrap reused from risk_disciplined_beta"
    )
    assert safety["hermes_source_in_public"] is False
    assert report.external_coverage["oi_policy"] == (
        "no forward fill; use same-timestamp OI, else same-timestamp futures/spot "
        "volume ratio proxy if available"
    )
