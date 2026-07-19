from __future__ import annotations

from aegis.backtest_core import CostModel
from aegis.volume_divergence_rr import (
    VolumeDivergenceBar,
    VolumeDivergenceConfig,
    run_volume_divergence_rr,
)


def _bar(
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> VolumeDivergenceBar:
    return VolumeDivergenceBar(
        timestamp=index * 4 * 60 * 60 * 1000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def test_lower_low_with_lower_volume_enters_next_bar_and_takes_profit() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=99, low=90, close=96, volume=80),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(lookback_bars=2, allow_short=False),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.signal_timestamp == bars[4].timestamp
    assert trade.entry_timestamp == bars[5].timestamp
    assert trade.stop_price == 90
    assert trade.take_profit_price == 114
    assert trade.exit_reason == "take_profit"
    assert trade.net_return_after_costs == 0.1875


def test_higher_high_with_lower_volume_enters_short_next_bar() -> None:
    bars = [
        _bar(0, open_=100, high=102, low=98, close=100, volume=300),
        _bar(1, open_=100, high=106, low=99, close=105, volume=250),
        _bar(2, open_=105, high=110, low=103, close=108, volume=120),
        _bar(3, open_=108, high=109, low=104, close=106, volume=220),
        _bar(4, open_=106, high=120, low=105, close=118, volume=90),
        _bar(5, open_=118, high=119, low=115, close=116, volume=300),
        _bar(6, open_=116, high=117, low=111, close=112, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(lookback_bars=2, allow_long=False),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short"
    assert trade.signal_timestamp == bars[4].timestamp
    assert trade.entry_timestamp == bars[5].timestamp
    assert trade.stop_price == 120
    assert trade.take_profit_price == 112
    assert trade.exit_reason == "take_profit"


def test_higher_volume_divergence_does_not_trigger() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=99, low=90, close=96, volume=180),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(lookback_bars=2, allow_short=False),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert result.trades == ()


def test_same_bar_stop_and_take_profit_uses_stop_first() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=99, low=90, close=96, volume=80),
        _bar(5, open_=96, high=115, low=89, close=99, volume=300),
        _bar(6, open_=99, high=100, low=98, close=99, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(lookback_bars=2, allow_short=False),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_timestamp == bars[5].timestamp
    assert trade.exit_timestamp == bars[5].timestamp
    assert trade.exit_reason == "stop_loss"
    assert trade.net_return_after_costs == -0.0625


def test_fixed_account_risk_sizing_scales_two_r_win_to_two_percent() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=99, low=90, close=96, volume=80),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            reward_risk=2.0,
            risk_per_trade_fraction=0.01,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.stop_distance_fraction == 0.0625
    assert trade.position_notional_fraction == 0.16
    assert round(trade.net_return_after_costs, 10) == 0.02


def test_max_position_notional_caps_fixed_risk_sizing() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=99, low=90, close=96, volume=80),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            reward_risk=2.0,
            risk_per_trade_fraction=0.01,
            max_position_notional_fraction=0.1,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.position_notional_fraction == 0.1
    assert round(trade.net_return_after_costs, 10) == 0.0125


def test_rsi_bullish_divergence_can_be_required_for_long() -> None:
    bars = [
        _bar(0, open_=100, high=102, low=99, close=100, volume=300),
        _bar(1, open_=100, high=101, low=89, close=90, volume=250),
        _bar(2, open_=90, high=92, low=79, close=80, volume=100),
        _bar(3, open_=80, high=97, low=79, close=95, volume=220),
        _bar(4, open_=95, high=96, low=78, close=91, volume=80),
        _bar(5, open_=91, high=120, low=90, close=110, volume=300),
        _bar(6, open_=110, high=140, low=109, close=138, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            rsi_period=2,
            require_rsi_divergence=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.previous_extreme_rsi is not None
    assert trade.signal_extreme_rsi is not None
    assert trade.signal_extreme_rsi > trade.previous_extreme_rsi


def test_missing_bullish_rsi_divergence_rejects_lower_volume_lower_low() -> None:
    bars = [
        _bar(0, open_=100, high=102, low=99, close=100, volume=300),
        _bar(1, open_=100, high=101, low=89, close=90, volume=250),
        _bar(2, open_=90, high=92, low=79, close=80, volume=100),
        _bar(3, open_=80, high=82, low=79, close=79, volume=220),
        _bar(4, open_=79, high=80, low=78, close=70, volume=80),
        _bar(5, open_=91, high=120, low=90, close=110, volume=300),
        _bar(6, open_=110, high=140, low=109, close=138, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            rsi_period=2,
            require_rsi_divergence=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert result.trades == ()


def test_rsi_bearish_divergence_can_be_required_for_short() -> None:
    bars = [
        _bar(0, open_=100, high=101, low=98, close=100, volume=300),
        _bar(1, open_=100, high=112, low=99, close=110, volume=250),
        _bar(2, open_=110, high=120, low=109, close=120, volume=100),
        _bar(3, open_=120, high=119, low=104, close=105, volume=220),
        _bar(4, open_=105, high=122, low=104, close=109, volume=80),
        _bar(5, open_=109, high=110, low=95, close=100, volume=300),
        _bar(6, open_=100, high=101, low=70, close=75, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            rsi_period=2,
            require_rsi_divergence=True,
            allow_long=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short"
    assert trade.previous_extreme_rsi is not None
    assert trade.signal_extreme_rsi is not None
    assert trade.signal_extreme_rsi < trade.previous_extreme_rsi


def test_macd_histogram_bullish_divergence_can_be_required_for_long() -> None:
    bars = [
        _bar(0, open_=120, high=122, low=119, close=120, volume=300),
        _bar(1, open_=120, high=121, low=114, close=115, volume=290),
        _bar(2, open_=115, high=116, low=109, close=110, volume=280),
        _bar(3, open_=110, high=111, low=99, close=100, volume=270),
        _bar(4, open_=100, high=101, low=84, close=85, volume=180),
        _bar(5, open_=85, high=97, low=85, close=95, volume=260),
        _bar(6, open_=95, high=96, low=89, close=90, volume=250),
        _bar(7, open_=90, high=91, low=83, close=88, volume=120),
        _bar(8, open_=88, high=110, low=87, close=108, volume=300),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            require_macd_histogram_divergence=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.previous_extreme_macd_histogram is not None
    assert trade.signal_extreme_macd_histogram is not None
    assert trade.signal_extreme_macd_histogram > trade.previous_extreme_macd_histogram


def test_missing_macd_histogram_divergence_rejects_lower_volume_lower_low() -> None:
    bars = [
        _bar(0, open_=100, high=102, low=99, close=100, volume=300),
        _bar(1, open_=100, high=101, low=89, close=90, volume=250),
        _bar(2, open_=90, high=92, low=79, close=80, volume=100),
        _bar(3, open_=80, high=82, low=79, close=79, volume=220),
        _bar(4, open_=79, high=80, low=78, close=70, volume=80),
        _bar(5, open_=91, high=120, low=90, close=110, volume=300),
        _bar(6, open_=110, high=140, low=109, close=138, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            require_macd_histogram_divergence=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert result.trades == ()


def test_liquidity_sweep_can_be_required_for_long() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=101, low=90, close=96, volume=80),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            require_liquidity_sweep=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    assert result.trades[0].liquidity_sweep_pass is True


def test_missing_liquidity_sweep_rejects_lower_volume_lower_low() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=101, low=90, close=94, volume=80),
        _bar(5, open_=96, high=100, low=95, close=99, volume=300),
        _bar(6, open_=99, high=115, low=98, close=114, volume=310),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            require_liquidity_sweep=True,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert result.trades == ()


def test_choch_confirmation_delays_entry_until_next_bar() -> None:
    bars = [
        _bar(0, open_=100, high=105, low=99, close=102, volume=300),
        _bar(1, open_=102, high=103, low=97, close=101, volume=250),
        _bar(2, open_=101, high=103, low=95, close=96, volume=100),
        _bar(3, open_=96, high=100, low=96, close=98, volume=220),
        _bar(4, open_=98, high=101, low=90, close=96, volume=80),
        _bar(5, open_=96, high=105, low=95, close=104, volume=300),
        _bar(6, open_=104, high=147, low=103, close=146, volume=310),
        _bar(7, open_=146, high=148, low=145, close=147, volume=320),
    ]
    result = run_volume_divergence_rr(
        bars,
        symbol="TEST/USDT",
        config=VolumeDivergenceConfig(
            lookback_bars=2,
            require_liquidity_sweep=True,
            require_choch_confirmation=True,
            choch_lookback_bars=2,
            allow_short=False,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.signal_timestamp == bars[4].timestamp
    assert trade.choch_confirmation_timestamp == bars[5].timestamp
    assert trade.entry_timestamp == bars[6].timestamp
    assert trade.choch_level == 103
