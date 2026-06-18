from __future__ import annotations

import pytest

from aegis.combo_indicator_search import (
    ComboBar,
    ComboCostModel,
    ComboSearchConfig,
    benjamini_hochberg,
    predeclared_indicators,
    predeclared_rules,
    run_combo_indicator_search,
    simulate_rule,
)


def _bars(closes: list[float]) -> list[ComboBar]:
    return [
        ComboBar(
            timestamp=index,
            open=float(close),
            high=float(close) * 1.01,
            low=float(close) * 0.99,
            close=float(close),
            volume=1000.0 + index,
        )
        for index, close in enumerate(closes)
    ]


def _tiny_config() -> ComboSearchConfig:
    return ComboSearchConfig(
        train_bars=30,
        test_bars=10,
        step_bars=10,
        locked_oos_fraction=0.30,
        min_is_folds=3,
        top_k_oos=2,
        rsi_periods=(3,),
        ma_periods=(3,),
        ma_cross_pairs=(),
        macd_params=(),
        roc_periods=(3,),
        tsmom_periods=(),
        atr_periods=(3,),
        bollinger_periods=(),
        realized_vol_periods=(),
        volume_z_periods=(3,),
        obv_periods=(),
    )


def test_predeclared_search_space_counts_every_symbol_rule_trial() -> None:
    config = _tiny_config()
    bars = _bars([100 + index * 0.2 for index in range(120)])
    report = run_combo_indicator_search(
        {"BTC/USDT": bars, "ETH/USDT": bars},
        config=config,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert report.search_space_n == len(predeclared_rules(config)) * 2
    assert report.indicator_count == len(predeclared_indicators(config))
    assert report.multiple_testing["trial_count_n"] == report.search_space_n


def test_combo_signal_uses_prior_close_and_next_open_execution() -> None:
    config = _tiny_config()
    bars = _bars([100, 100, 100, 120, 60, 66, 70, 72])
    rule = predeclared_rules(config)[0]
    result = simulate_rule(
        bars,
        rule,
        start=4,
        end=6,
        config=config,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert result.first_execution_index == 4
    assert result.positions[0] == 1
    assert result.returns[0] == pytest.approx(66 / 60 - 1)


def test_locked_oos_is_after_selector_range() -> None:
    config = _tiny_config()
    bars = _bars([100 + index * 0.1 for index in range(150)])
    report = run_combo_indicator_search(
        {"BTC/USDT": bars},
        config=config,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert report.status == "OK"
    assert report.locked_oos_start == 105
    for metrics_name in report.selected_for_oos:
        assert metrics_name.startswith("BTC/USDT::")


def test_costs_reduce_combo_returns() -> None:
    config = _tiny_config()
    bars = _bars([100, 100, 100, 120, 132, 145, 160, 176])
    rule = predeclared_rules(config)[0]
    no_cost = simulate_rule(
        bars,
        rule,
        start=4,
        end=7,
        config=config,
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )
    with_cost = simulate_rule(
        bars,
        rule,
        start=4,
        end=7,
        config=config,
        cost_model=ComboCostModel(fee_bps=10, slippage_bps=5),
    )

    assert with_cost.metrics.net_cost > 0
    assert with_cost.metrics.total_return < no_cost.metrics.total_return


def test_bh_fdr_corrects_for_multiple_combo_trials() -> None:
    assert benjamini_hochberg([0.001, 0.02, 0.20, 0.90], alpha=0.10) == [
        True,
        True,
        False,
        False,
    ]
    assert benjamini_hochberg([0.04, 0.05, 0.06, 0.07], alpha=0.05) == [
        False,
        False,
        False,
        False,
    ]
