from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from aegis.backtest_core import CostModel
from aegis.crypto_cross_sectional import (
    CrossSectionalCryptoBar,
    CrossSectionalCryptoConfig,
    run_crypto_cross_sectional_momentum,
)


def _bars(
    symbol_index: int,
    *,
    days: int = 90,
    drift: float,
    stable: bool = False,
    funding_rate: float = 0.0,
    volume: float = 100_000_000.0,
) -> list[CrossSectionalCryptoBar]:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    price = 100.0 + symbol_index * 20.0
    out: list[CrossSectionalCryptoBar] = []
    for day in range(days):
        timestamp = int((start + timedelta(days=day)).timestamp() * 1000)
        wiggle = 0.004 * math.sin(day / 3.0 + symbol_index)
        open_price = price
        close_price = max(1.0, open_price * (1.0 + drift + wiggle))
        out.append(
            CrossSectionalCryptoBar(
                timestamp=timestamp,
                open=open_price,
                close=close_price,
                quote_volume_usd=volume + symbol_index * 1_000_000.0,
                funding_rate=funding_rate,
                market_cap_usd=None,
                exchange_count=2,
                listed_at=int(start.timestamp() * 1000),
                is_stable=stable,
            )
        )
        price = close_price
    return out


def _dataset() -> dict[str, list[CrossSectionalCryptoBar]]:
    return {
        "BTC/USDT:USDT": _bars(0, drift=0.0010),
        "ETH/USDT:USDT": _bars(1, drift=0.0020),
        "SOL/USDT:USDT": _bars(2, drift=0.0030),
        "XRP/USDT:USDT": _bars(3, drift=-0.0010),
        "DOGE/USDT:USDT": _bars(4, drift=-0.0020),
        "ADA/USDT:USDT": _bars(5, drift=-0.0030),
    }


def _config() -> CrossSectionalCryptoConfig:
    return CrossSectionalCryptoConfig(
        momentum_lookback_days=6,
        skip_recent_days=1,
        vol_lookback_days=6,
        rebalance_days=7,
        target_annual_volatility=0.10,
        funding_gate_annualized=0.30,
        funding_lookback_days=3,
        min_history_days=8,
        liquidity_top_n=6,
        liquidity_pool_n=6,
        min_market_cap_usd=1_000_000_000.0,
        min_proxy_volume_usd=10_000_000.0,
        min_exchange_count=2,
        locked_oos_fraction=0.50,
        pbo_splits=4,
        bootstrap_iterations=60,
        min_years=0.0,
        require_regime_years=(),
    )


def _run(
    data: dict[str, list[CrossSectionalCryptoBar]],
    *,
    config: CrossSectionalCryptoConfig | None = None,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    result = run_crypto_cross_sectional_momentum(
        data,
        config=config or _config(),
        cost_model=cost_model or CostModel(fee_bps=5.0, slippage_bps=5.0),
        survivor_light=True,
    )
    return dict(result)


def test_t_plus_one_decision_timestamp_precedes_execution_timestamp() -> None:
    result = _run(_dataset())

    first = cast(dict[str, Any], result["rebalance_sample"][0])
    assert first["execution_timestamp"] > first["decision_timestamp"]


def test_first_rebalance_does_not_use_execution_bar_close_for_signal() -> None:
    data = _dataset()
    original = _run(data)
    mutated = _dataset()
    first_execution_index = max(
        _config().min_history_days,
        _config().momentum_lookback_days + _config().skip_recent_days + 2,
        _config().vol_lookback_days + 2,
    )
    series = mutated["ADA/USDT:USDT"]
    bar = series[first_execution_index]
    series[first_execution_index] = replace(bar, close=bar.close * 50.0)

    original_first = cast(dict[str, Any], original["rebalance_sample"][0])
    mutated_first = cast(dict[str, Any], _run(mutated)["rebalance_sample"][0])

    assert mutated_first["weights"] == pytest.approx(original_first["weights"])


def test_portfolio_is_dollar_neutral_at_rebalance() -> None:
    result = _run(_dataset())

    first = cast(dict[str, Any], result["rebalance_sample"][0])
    assert first["long_count"] > 0
    assert first["short_count"] > 0
    assert abs(float(first["net"])) < 1e-12


def test_survivor_light_ceiling_is_reported_for_free_data_mode() -> None:
    result = _run(_dataset())

    safety = cast(dict[str, Any], result["safety"])
    assert safety["survivor_light_ceiling_required"] is True
    assert result["verdict"] != "ROBUST"


def test_costs_and_funding_reduce_net_result() -> None:
    data = _dataset()
    no_cost = _run(data, cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0))
    with_cost = _run(data, cost_model=CostModel(fee_bps=20.0, slippage_bps=10.0))

    no_cost_metrics = cast(dict[str, float], no_cost["standard_metrics"])
    with_cost_metrics = cast(dict[str, float], with_cost["standard_metrics"])
    assert with_cost_metrics["net_cost"] > 0.0
    assert with_cost_metrics["total_return"] < no_cost_metrics["total_return"]


def test_stablecoin_and_single_exchange_symbols_are_excluded() -> None:
    data = _dataset()
    data["USDC/USDT:USDT"] = _bars(6, drift=0.0, stable=True)
    single_exchange = _bars(7, drift=0.004)
    data["NEW/USDT:USDT"] = [replace(bar, exchange_count=1) for bar in single_exchange]

    result = _run(data, config=replace(_config(), liquidity_top_n=8, liquidity_pool_n=8))
    first = cast(dict[str, Any], result["rebalance_sample"][0])

    assert first["universe_count"] == 6
