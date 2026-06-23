from __future__ import annotations

import pytest

from aegis.crypto_tsmom import (
    CostModel,
    CryptoBar,
    TsmomConfig,
    benjamini_hochberg,
    run_crypto_tsmom_walk_forward,
    sign_test_p_value,
    simulate_tsmom,
)


def _bars(closes: list[float]) -> list[CryptoBar]:
    return [
        CryptoBar(timestamp=index, open=float(close), close=float(close))
        for index, close in enumerate(closes)
    ]


def test_tsmom_uses_prior_close_signal_and_next_open_execution() -> None:
    bars = _bars([100, 100, 110, 80, 60, 66])
    result = simulate_tsmom(
        bars,
        lookback=2,
        start=3,
        end=5,
        config=TsmomConfig(lookbacks=(2,), annualization_periods=365),
        cost_model=CostModel(fee_bps=0, slippage_bps=0),
    )

    assert result.first_trade_index == 3
    assert result.positions == (1, 0)
    assert result.returns[0] == pytest.approx(60 / 80 - 1)
    assert result.returns[1] == pytest.approx(0.0)


def test_walk_forward_freezes_lookback_before_oos_execution() -> None:
    bars = _bars([100 + index * 0.5 for index in range(90)])
    report = run_crypto_tsmom_walk_forward(
        {"BTC/USDT": bars, "ETH/USDT": bars},
        config=TsmomConfig(
            lookbacks=(3, 5),
            train_bars=30,
            test_bars=15,
            step_bars=15,
            annualization_periods=365,
        ),
        cost_model=CostModel(fee_bps=0, slippage_bps=0),
    )

    assert report.status == "OK"
    assert len(report.windows) >= 3
    for window in report.windows:
        assert window.selected_lookback in {3, 5}
        assert window.selector_max_bar_seen < window.first_oos_execution_bar


def test_costs_and_funding_are_counted_against_returns() -> None:
    bars = _bars([100, 100, 110, 121, 133.1, 146.41])
    no_cost = simulate_tsmom(
        bars,
        lookback=2,
        start=3,
        end=5,
        config=TsmomConfig(lookbacks=(2,), annualization_periods=365),
        cost_model=CostModel(fee_bps=0, slippage_bps=0, funding_bps_per_period=0),
    )
    with_cost = simulate_tsmom(
        bars,
        lookback=2,
        start=3,
        end=5,
        config=TsmomConfig(lookbacks=(2,), annualization_periods=365),
        cost_model=CostModel(fee_bps=10, slippage_bps=5, funding_bps_per_period=2),
    )

    assert with_cost.metrics.net_cost > 0
    assert with_cost.metrics.total_return < no_cost.metrics.total_return


def test_position_change_charges_one_way_cost_not_round_trip() -> None:
    bars = _bars([100, 100, 110, 121, 133.1, 146.41])
    result = simulate_tsmom(
        bars,
        lookback=2,
        start=3,
        end=5,
        config=TsmomConfig(lookbacks=(2,), annualization_periods=365),
        cost_model=CostModel(fee_bps=10, slippage_bps=5, funding_bps_per_period=0),
    )

    assert result.positions == (1, 1)
    assert result.costs == pytest.approx((0.0015, 0.0))
    assert result.returns[0] == pytest.approx(133.1 / 121 - 1 - 0.0015)


def test_sign_test_and_fdr_are_explicit_not_all_window_gates() -> None:
    assert sign_test_p_value([0.1, 0.2, 0.3, -0.1, 0.4]) < 1.0
    assert benjamini_hochberg([0.01, 0.04, 0.5], alpha=0.10) == [True, True, False]
