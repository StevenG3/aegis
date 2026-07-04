from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from aegis.backtest_core import MAX_TRIAL_COUNT_DEFAULT, CostModel
from aegis.vibecoding_factor_factory import (
    StrategyParams,
    VibeBar,
    VibeFactoryConfig,
    compute_factor_table,
    conservative_bar_return,
    generate_strategy_signals,
    run_vibecoding_factor_factory,
    simulate_strategy,
)


def _bars(
    *,
    count: int = 180,
    drift: float = 0.002,
    wave: float = 0.01,
    start_price: float = 100.0,
) -> list[VibeBar]:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    price = start_price
    out: list[VibeBar] = []
    for index in range(count):
        timestamp = int((start + timedelta(hours=index)).timestamp() * 1000)
        wiggle = wave * math.sin(index / 4.0)
        open_price = price
        close = max(1.0, open_price * (1.0 + drift + wiggle))
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        out.append(
            VibeBar(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1_000_000.0 + index,
            )
        )
        price = close
    return out


def _small_config(
    *, max_trial_count: int | None = MAX_TRIAL_COUNT_DEFAULT
) -> VibeFactoryConfig:
    return VibeFactoryConfig(
        train_bars_1h=50,
        test_bars_1h=25,
        step_bars_1h=25,
        train_bars_4h=50,
        test_bars_4h=25,
        step_bars_4h=25,
        min_trades=1,
        min_oos_windows=2,
        pbo_splits=4,
        max_trial_count=max_trial_count,
    )


def test_factor_values_do_not_use_future_bars() -> None:
    bars = _bars(count=140)
    original = compute_factor_table(bars)
    mutated = list(bars)
    mutated[80] = replace(mutated[80], close=mutated[80].close * 10.0)

    changed = compute_factor_table(mutated)

    assert changed["ret_12"][40] == original["ret_12"][40]
    assert changed["rsi_14"][40] == original["rsi_14"][40]


def test_range_breakout_uses_prior_high_not_current_high() -> None:
    bars = _bars(count=80, drift=0.0, wave=0.0)
    mutable = list(bars)
    for index, bar in enumerate(mutable):
        mutable[index] = replace(bar, high=100.0, low=99.0, close=99.5, open=99.5)
    mutable[30] = replace(mutable[30], close=101.0, open=101.0, high=10_000.0)
    params = StrategyParams(
        "range_compression_breakout",
        {"range_window": 20, "breakout_window": 20, "max_range_pct": 200.0},
    )

    signals = generate_strategy_signals(mutable, params)

    assert signals[30] is True


def test_t_plus_one_trade_log_entry_after_signal() -> None:
    bars = _bars(count=100, drift=0.004, wave=0.0)
    params = StrategyParams(
        "roc_momentum",
        {"roc_lookback": 12, "roc_threshold": 0.0, "trend_ma": 50},
    )

    simulation = simulate_strategy(bars, params, config=_small_config())

    assert simulation.trade_log
    first = cast(dict[str, Any], simulation.trade_log[0])
    assert str(first["entry_timestamp"]) > str(first["signal_timestamp"])


def test_costs_reduce_strategy_return() -> None:
    bars = _bars(count=120, drift=0.004, wave=0.0)
    params = StrategyParams(
        "roc_momentum",
        {"roc_lookback": 12, "roc_threshold": 0.0, "trend_ma": 50},
    )

    no_cost = simulate_strategy(
        bars,
        params,
        config=_small_config(),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )
    with_cost = simulate_strategy(
        bars,
        params,
        config=_small_config(),
        cost_model=CostModel(fee_bps=20.0, slippage_bps=10.0),
    )

    assert sum(with_cost.returns) < sum(no_cost.returns)
    assert with_cost.net_cost > no_cost.net_cost


def test_same_bar_stop_and_take_profit_is_conservative() -> None:
    value, reason = conservative_bar_return(
        open_price=100.0,
        high=110.0,
        low=90.0,
        next_open=105.0,
        take_profit_pct=0.05,
        stop_loss_pct=0.03,
    )

    assert reason == "stop_loss"
    assert value == pytest.approx(-0.03)


def test_no_significant_ic_skips_strategy_stage_without_positive_verdict() -> None:
    flat = _bars(count=120, drift=0.0, wave=0.0)
    result = run_vibecoding_factor_factory(
        {"BTCUSDT:1h": flat},
        config=_small_config(max_trial_count=150),
        cost_model=CostModel(fee_bps=10.0, slippage_bps=5.0),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    factor = cast(dict[str, Any], result["factor_ic"])
    strategy = cast(dict[str, Any], result["strategy_backtest"])

    assert result["verdict"] in {"NO_EDGE", "INSUFFICIENT"}
    assert factor["allowed_strategy_families"] == []
    assert strategy["walk_forward_windows"] == []


def test_factory_reports_trial_count_and_survivor_ceiling() -> None:
    result = run_vibecoding_factor_factory(
        {
            "BTCUSDT:1h": _bars(count=180, drift=0.004, wave=0.004),
            "ETHUSDT:1h": _bars(count=180, drift=0.003, wave=0.006, start_price=80.0),
            "SOLUSDT:1h": _bars(count=180, drift=0.002, wave=0.008, start_price=40.0),
        },
        config=_small_config(max_trial_count=400),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    multiple = cast(dict[str, Any], result["multiple_testing"])
    safety = cast(dict[str, Any], result["safety"])
    assert int(multiple["candidate_count_n"]) > 0
    assert safety["survivor_light_ceiling"] is True
    assert result["verdict"] != "ROBUST"
