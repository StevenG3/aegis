from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from aegis.backtest_core import (
    BacktestDiscipline,
    CostModel,
    HypothesisSpec,
    benjamini_hochberg,
    bootstrap_mean_ci,
    metrics_from_returns,
    pbo,
    run_backtest,
    sign_test_p_value,
    trade_scorecard,
    trade_scorecard_to_dict,
)

Verdict = Literal["SUGGESTIVE_NEEDS_PAID_CONFIRM", "SUGGESTIVE", "NO_EDGE", "INSUFFICIENT"]


@dataclass(frozen=True)
class CrossSectionalCryptoBar:
    timestamp: int
    open: float
    close: float
    quote_volume_usd: float
    funding_rate: float = 0.0
    market_cap_usd: float | None = None
    exchange_count: int = 1
    listed_at: int | None = None
    is_stable: bool = False
    is_wrapped: bool = False
    is_leveraged: bool = False


@dataclass(frozen=True)
class CrossSectionalCryptoConfig:
    momentum_lookback_days: int = 30
    skip_recent_days: int = 1
    vol_lookback_days: int = 30
    rebalance_days: int = 7
    target_annual_volatility: float = 0.10
    funding_gate_annualized: float | None = 0.30
    funding_lookback_days: int = 7
    min_history_days: int = 180
    liquidity_top_n: int = 30
    liquidity_pool_n: int = 50
    min_market_cap_usd: float = 1_000_000_000.0
    min_proxy_volume_usd: float = 50_000_000.0
    min_exchange_count: int = 2
    long_short_fraction: float = 1 / 3
    locked_oos_fraction: float = 0.40
    annualization_periods: int = 365
    pbo_splits: int = 4
    pbo_threshold: float = 0.20
    fdr_alpha: float = 0.10
    bootstrap_iterations: int = 600
    min_years: float = 4.0
    require_regime_years: tuple[int, ...] = (2021, 2022, 2023, 2024)


@dataclass(frozen=True)
class Simulation:
    returns: tuple[float, ...]
    cash_returns: tuple[float, ...]
    equal_weight_long_returns: tuple[float, ...]
    btc_buy_hold_returns: tuple[float, ...]
    trade_returns: tuple[float, ...]
    timestamps: tuple[int, ...]
    turnover: float
    net_cost: float
    funding_pnl: float
    rebalance_count: int
    breadth: tuple[int, ...]
    rank_ic: tuple[float, ...]
    rebalance_log: tuple[Mapping[str, object], ...]


MAIN_CONFIG = CrossSectionalCryptoConfig()
DEFAULT_COST_MODEL = CostModel(
    fee_bps=5.0,
    slippage_bps=5.0,
    funding_label="perp funding debited/credited from daily funding observations",
)


def run_crypto_cross_sectional_momentum(
    bars_by_symbol: Mapping[str, Sequence[CrossSectionalCryptoBar]],
    *,
    config: CrossSectionalCryptoConfig = MAIN_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    survivor_light: bool = True,
    data_source: str = "caller_supplied_crypto_perp_bars",
) -> Mapping[str, Any]:
    spec = crypto_cross_sectional_hypothesis_spec(
        bars_by_symbol,
        config=config,
        cost_model=cost_model,
        survivor_light=survivor_light,
        data_source=data_source,
        runner=lambda: _run_impl(
            bars_by_symbol,
            config=config,
            cost_model=cost_model,
            survivor_light=survivor_light,
        ),
    )
    return cast(Mapping[str, Any], run_backtest(spec).payload)


def crypto_cross_sectional_hypothesis_spec(
    bars_by_symbol: Mapping[str, Sequence[CrossSectionalCryptoBar]],
    *,
    config: CrossSectionalCryptoConfig = MAIN_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    survivor_light: bool = True,
    data_source: str = "caller_supplied_crypto_perp_bars",
    runner: Callable[[], object] | None = None,
) -> HypothesisSpec:
    universe = tuple(sorted(bars_by_symbol)) or ("<empty>",)
    return HypothesisSpec(
        key="olympus67_crypto_cross_sectional_momentum",
        hypothesis_type="factor",
        universe=universe,
        predeclared_signals=(
            "cross_sectional_momentum_30_1",
            "inverse_realized_vol_30",
            "funding_crowding_gate",
        ),
        params={
            "mode": "confirmation",
            "momentum_lookback_days": config.momentum_lookback_days,
            "skip_recent_days": config.skip_recent_days,
            "vol_lookback_days": config.vol_lookback_days,
            "rebalance_days": config.rebalance_days,
            "target_annual_volatility": config.target_annual_volatility,
            "funding_gate_annualized": config.funding_gate_annualized,
            "locked_oos_fraction": config.locked_oos_fraction,
        },
        cost_model=cost_model,
        benchmark="cash_neutral+equal_weight_long_beta+btc_buy_hold",
        data_source=data_source,
        trial_count_n=1,
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=survivor_light,
        runner=runner,
    )


def robustness_configs() -> tuple[CrossSectionalCryptoConfig, ...]:
    configs: list[CrossSectionalCryptoConfig] = []
    for lookback in (30, 60, 90):
        for rebalance_days in (7, 1):
            for target_vol in (0.10, 0.15):
                for funding_gate in (None, 0.30, 0.60):
                    configs.append(
                        replace(
                            MAIN_CONFIG,
                            momentum_lookback_days=lookback,
                            rebalance_days=rebalance_days,
                            target_annual_volatility=target_vol,
                            funding_gate_annualized=funding_gate,
                        )
                    )
    return tuple(configs)


def _run_impl(
    bars_by_symbol: Mapping[str, Sequence[CrossSectionalCryptoBar]],
    *,
    config: CrossSectionalCryptoConfig,
    cost_model: CostModel,
    survivor_light: bool,
) -> Mapping[str, Any]:
    aligned = _aligned_bars(bars_by_symbol)
    if len(aligned) < 2:
        return _insufficient("no aligned daily bars", config, cost_model, survivor_light)
    sim = _simulate(aligned, config=config, cost_model=cost_model)
    if not sim.returns:
        return _insufficient(
            "no tradable OOS returns after point-in-time universe and funding gates",
            config,
            cost_model,
            survivor_light,
        )
    oos_start = int(len(sim.returns) * (1.0 - config.locked_oos_fraction))
    oos_returns = sim.returns[oos_start:]
    oos_cash = sim.cash_returns[oos_start:]
    oos_equal = sim.equal_weight_long_returns[oos_start:]
    oos_btc = sim.btc_buy_hold_returns[oos_start:]
    if len(oos_returns) < max(30, config.pbo_splits):
        return _insufficient(
            "insufficient locked-OOS returns for confirmation gates",
            config,
            cost_model,
            survivor_light,
            simulation=sim,
        )

    years = len(oos_returns) / config.annualization_periods
    regime_years = _year_set(sim.timestamps[oos_start:])
    power = _power_self_check(years=max(years, 1 / config.annualization_periods), trials=1)
    strategy_metrics = metrics_from_returns(
        oos_returns,
        annualization_periods=config.annualization_periods,
        turnover=sim.turnover,
        net_cost=sim.net_cost,
        oos_vs_buy_hold_window_win_rate=_positive_share(
            [left - right for left, right in zip(oos_returns, oos_btc, strict=False)]
        ),
    )
    cash_metrics = metrics_from_returns(
        oos_cash,
        annualization_periods=config.annualization_periods,
        turnover=0.0,
        net_cost=0.0,
    )
    equal_metrics = metrics_from_returns(
        oos_equal,
        annualization_periods=config.annualization_periods,
        turnover=0.0,
        net_cost=0.0,
    )
    btc_metrics = metrics_from_returns(
        oos_btc,
        annualization_periods=config.annualization_periods,
        turnover=0.0,
        net_cost=0.0,
    )
    p_value = sign_test_p_value(oos_returns, alternative="greater")
    fdr_pass = benjamini_hochberg([p_value], alpha=config.fdr_alpha, tie_policy="rank")[0]
    robustness = _robustness(aligned, base_config=config, cost_model=cost_model)
    pbo_report = _pbo_report(robustness, pbo_splits=config.pbo_splits)
    sharpe_ci = _bootstrap_sharpe_ci(
        oos_returns,
        annualization_periods=config.annualization_periods,
        iterations=config.bootstrap_iterations,
    )
    trade_stats = trade_scorecard_to_dict(trade_scorecard(sim.trade_returns[oos_start:]))
    benchmarks_pass = (
        strategy_metrics.total_return > cash_metrics.total_return
        and strategy_metrics.sharpe > equal_metrics.sharpe
        and strategy_metrics.sharpe > btc_metrics.sharpe
    )
    underpowered = years < config.min_years or not set(config.require_regime_years).issubset(
        regime_years
    )
    if underpowered:
        verdict: Verdict = "INSUFFICIENT"
        reason = (
            "confirmation sample lacks required years/regimes for the predeclared power gate"
        )
    elif sharpe_ci["p05"] is None or float(sharpe_ci["p05"]) <= 0.0:
        verdict = "NO_EDGE"
        reason = "net Sharpe bootstrap CI includes <= 0"
    elif (
        fdr_pass
        and bool(pbo_report.get("valid", False))
        and _float(pbo_report.get("pbo"), 1.0) <= config.pbo_threshold
        and benchmarks_pass
    ):
        verdict = "SUGGESTIVE_NEEDS_PAID_CONFIRM" if survivor_light else "SUGGESTIVE"
        reason = "main specification passed FDR, PBO, bootstrap CI, and benchmark gates"
    else:
        verdict = "NO_EDGE"
        reason = "main specification failed one or more FDR, PBO, CI, or benchmark gates"
    return {
        "status": "INSUFFICIENT" if verdict == "INSUFFICIENT" else "OK",
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "point-in-time venue universe with delisted contracts, complete historical funding, "
            "and non-survivor-limited liquidity/market-cap coverage"
        ),
        "candidate_count_n": 1,
        "raw_is_survivors": int(statistics.fmean(oos_returns) > 0.0),
        "fdr_is_survivors": int(fdr_pass),
        "standard_metrics": _metrics_to_dict(strategy_metrics),
        "benchmark_metrics": {
            "cash_neutral": _metrics_to_dict(cash_metrics),
            "equal_weight_long_beta": _metrics_to_dict(equal_metrics),
            "btc_buy_hold": _metrics_to_dict(btc_metrics),
        },
        "trade_scorecard": trade_stats,
        "multiple_testing": {
            "method": "main_spec_BH_FDR_G1 + robustness_grid_CSCV_PBO",
            "candidate_count_n": 1,
            "p_value": p_value,
            "fdr_alpha": config.fdr_alpha,
            "fdr_after": int(fdr_pass),
            "pbo": pbo_report,
            "robustness_grid_n": len(robustness),
        },
        "power_self_check": {
            **power,
            "sample_years": years,
            "observed_sharpe": strategy_metrics.sharpe,
            "sharpe_bootstrap_ci": sharpe_ci,
            "regime_years_observed": sorted(regime_years),
            "required_regime_years": list(config.require_regime_years),
            "underpowered": underpowered,
        },
        "ic": {
            "mean_rank_ic": statistics.fmean(sim.rank_ic) if sim.rank_ic else 0.0,
            "observations": len(sim.rank_ic),
            "breadth_mean": statistics.fmean(sim.breadth) if sim.breadth else 0.0,
            "breadth_min": min(sim.breadth) if sim.breadth else 0,
        },
        "robustness_grid": _robustness_summary(robustness),
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "t_plus_1_execution": True,
            "locked_oos": True,
            "full_costs": True,
            "funding_counted": True,
            "funding_label": cost_model.funding_label,
            "survivor_light_ceiling_required": survivor_light,
        },
        "coverage": _coverage(aligned),
        "rebalance_sample": list(sim.rebalance_log[:5]),
    }


def _simulate(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    *,
    config: CrossSectionalCryptoConfig,
    cost_model: CostModel,
) -> Simulation:
    symbols = tuple(sorted(aligned))
    length = min(len(aligned[symbol]) for symbol in symbols)
    start = max(
        config.min_history_days,
        config.momentum_lookback_days + config.skip_recent_days + 2,
        config.vol_lookback_days + 2,
    )
    current_weights = {symbol: 0.0 for symbol in symbols}
    current_month: tuple[int, int] | None = None
    month_universe: tuple[str, ...] = ()
    returns: list[float] = []
    cash_returns: list[float] = []
    equal_returns: list[float] = []
    btc_returns: list[float] = []
    trade_returns: list[float] = []
    timestamps: list[int] = []
    breadth: list[int] = []
    rank_ic: list[float] = []
    rebalance_log: list[Mapping[str, object]] = []
    turnover_total = 0.0
    net_cost = 0.0
    funding_total = 0.0
    rebalance_count = 0
    for execution_index in range(start, length - 1):
        decision_index = execution_index - 1
        bar = aligned[symbols[0]][execution_index]
        month = _year_month(bar.timestamp)
        if month != current_month:
            current_month = month
            month_universe = _select_universe(aligned, decision_index, config)
        if (execution_index - start) % config.rebalance_days == 0:
            new_weights, log = _target_weights(
                aligned,
                month_universe,
                decision_index=decision_index,
                config=config,
            )
            turnover = sum(
                abs(new_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
                for symbol in symbols
            )
            turnover_total += turnover
            trade_cost = turnover * cost_model.one_way_cost
            net_cost += trade_cost
            current_weights = {symbol: new_weights.get(symbol, 0.0) for symbol in symbols}
            rebalance_count += 1
            rebalance_log.append(
                {
                    "decision_timestamp": aligned[symbols[0]][decision_index].timestamp,
                    "execution_timestamp": bar.timestamp,
                    **log,
                    "turnover": turnover,
                }
            )
        else:
            trade_cost = 0.0
        asset_returns = {
            symbol: _open_return(aligned[symbol], execution_index) for symbol in symbols
        }
        gross = sum(current_weights[symbol] * asset_returns[symbol] for symbol in symbols)
        funding = sum(
            -current_weights[symbol] * aligned[symbol][execution_index].funding_rate
            for symbol in symbols
        )
        funding_total += funding
        daily = gross + funding - trade_cost
        returns.append(daily)
        trade_returns.append(daily)
        cash_returns.append(0.0)
        eligible = tuple(symbol for symbol in month_universe if symbol in asset_returns)
        equal_returns.append(
            statistics.fmean(asset_returns[symbol] for symbol in eligible) if eligible else 0.0
        )
        btc_symbol = _btc_symbol(symbols)
        btc_returns.append(asset_returns.get(btc_symbol, 0.0))
        timestamps.append(bar.timestamp)
        breadth.append(sum(1 for weight in current_weights.values() if weight != 0.0))
        ic = _rank_ic(
            aligned,
            month_universe,
            decision_index,
            execution_index,
            config=config,
        )
        if ic is not None:
            rank_ic.append(ic)
    return Simulation(
        returns=tuple(returns),
        cash_returns=tuple(cash_returns),
        equal_weight_long_returns=tuple(equal_returns),
        btc_buy_hold_returns=tuple(btc_returns),
        trade_returns=tuple(trade_returns),
        timestamps=tuple(timestamps),
        turnover=turnover_total,
        net_cost=net_cost,
        funding_pnl=funding_total,
        rebalance_count=rebalance_count,
        breadth=tuple(breadth),
        rank_ic=tuple(rank_ic),
        rebalance_log=tuple(rebalance_log),
    )


def _target_weights(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    universe: Sequence[str],
    *,
    decision_index: int,
    config: CrossSectionalCryptoConfig,
) -> tuple[dict[str, float], Mapping[str, object]]:
    signals: list[tuple[str, float, float, float]] = []
    for symbol in universe:
        series = aligned[symbol]
        momentum = _momentum(series, decision_index, config)
        vol = _realized_vol(series, decision_index, config.vol_lookback_days)
        funding = _funding_annualized(series, decision_index, config.funding_lookback_days)
        if momentum is None or vol <= 0.0:
            continue
        signals.append((symbol, momentum, vol, funding))
    if len(signals) < 3:
        return {}, {"universe_count": len(universe), "long_count": 0, "short_count": 0}
    ordered = sorted(signals, key=lambda item: item[1])
    leg_count = max(1, int(len(ordered) * config.long_short_fraction))
    shorts = ordered[:leg_count]
    longs = ordered[-leg_count:]
    if config.funding_gate_annualized is not None:
        gate = config.funding_gate_annualized
        longs = [item for item in longs if item[3] <= gate]
        shorts = [item for item in shorts if item[3] >= -gate]
    weights: dict[str, float] = {}
    _assign_leg(weights, longs, target=0.5)
    _assign_leg(weights, shorts, target=-0.5)
    scale = _vol_scale(aligned, weights, decision_index, config)
    scaled = {symbol: weight * scale for symbol, weight in weights.items()}
    return scaled, {
        "universe_count": len(universe),
        "signal_count": len(signals),
        "long_count": len(longs),
        "short_count": len(shorts),
        "gross": sum(abs(value) for value in scaled.values()),
        "net": sum(scaled.values()),
        "vol_scale": scale,
        "weights": dict(sorted(scaled.items())),
    }


def _assign_leg(
    weights: dict[str, float], leg: Sequence[tuple[str, float, float, float]], *, target: float
) -> None:
    if not leg:
        return
    inv_vol = [(symbol, 1.0 / vol) for symbol, _momentum, vol, _funding in leg if vol > 0.0]
    total = sum(value for _symbol, value in inv_vol)
    if total <= 0.0:
        return
    for symbol, value in inv_vol:
        weights[symbol] = weights.get(symbol, 0.0) + target * value / total


def _select_universe(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    decision_index: int,
    config: CrossSectionalCryptoConfig,
) -> tuple[str, ...]:
    candidates: list[tuple[str, float]] = []
    for symbol, series in aligned.items():
        current = series[decision_index]
        if current.is_stable or current.is_wrapped or current.is_leveraged:
            continue
        listed_at = current.listed_at if current.listed_at is not None else series[0].timestamp
        if current.timestamp - listed_at < config.min_history_days * 86_400_000:
            continue
        if current.exchange_count < config.min_exchange_count:
            continue
        volume = _median_volume(series, decision_index, 30)
        if volume <= 0.0:
            continue
        if current.market_cap_usd is None:
            if volume < config.min_proxy_volume_usd:
                continue
        elif current.market_cap_usd < config.min_market_cap_usd:
            continue
        candidates.append((symbol, volume))
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[
        : config.liquidity_pool_n
    ]
    return tuple(symbol for symbol, _volume in ranked[: config.liquidity_top_n])


def _aligned_bars(
    bars_by_symbol: Mapping[str, Sequence[CrossSectionalCryptoBar]],
) -> dict[str, tuple[CrossSectionalCryptoBar, ...]]:
    clean: dict[str, tuple[CrossSectionalCryptoBar, ...]] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
        if len(ordered) >= 2:
            clean[str(symbol)] = ordered
    if not clean:
        return {}
    common = sorted(
        set.intersection(*(set(bar.timestamp for bar in bars) for bars in clean.values()))
    )
    if len(common) < 2:
        return {}
    common_set = set(common)
    return {
        symbol: tuple(bar for bar in bars if bar.timestamp in common_set)
        for symbol, bars in clean.items()
    }


def _momentum(
    series: Sequence[CrossSectionalCryptoBar],
    decision_index: int,
    config: CrossSectionalCryptoConfig,
) -> float | None:
    end = decision_index - config.skip_recent_days
    start = end - config.momentum_lookback_days
    if start < 0:
        return None
    start_close = series[start].close
    if start_close <= 0.0:
        return None
    return series[end].close / start_close - 1.0


def _realized_vol(
    series: Sequence[CrossSectionalCryptoBar], decision_index: int, lookback: int
) -> float:
    start = decision_index - lookback
    if start < 0:
        return 0.0
    returns = [
        series[index].close / series[index - 1].close - 1.0
        for index in range(start + 1, decision_index + 1)
        if series[index - 1].close > 0.0
    ]
    return statistics.pstdev(returns) * math.sqrt(365) if len(returns) > 1 else 0.0


def _vol_scale(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    weights: Mapping[str, float],
    decision_index: int,
    config: CrossSectionalCryptoConfig,
) -> float:
    if not weights:
        return 0.0
    start = max(1, decision_index - config.vol_lookback_days + 1)
    returns: list[float] = []
    for index in range(start, decision_index + 1):
        value = 0.0
        for symbol, weight in weights.items():
            series = aligned[symbol]
            if series[index - 1].close > 0.0:
                value += weight * (series[index].close / series[index - 1].close - 1.0)
        returns.append(value)
    stdev = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    annual_vol = stdev * math.sqrt(config.annualization_periods)
    if annual_vol <= 0.0:
        return 0.0
    return config.target_annual_volatility / annual_vol


def _funding_annualized(
    series: Sequence[CrossSectionalCryptoBar], decision_index: int, lookback: int
) -> float:
    start = max(0, decision_index - lookback + 1)
    values = [series[index].funding_rate for index in range(start, decision_index + 1)]
    return statistics.fmean(values) * 365 if values else 0.0


def _median_volume(
    series: Sequence[CrossSectionalCryptoBar], decision_index: int, lookback: int
) -> float:
    start = max(0, decision_index - lookback + 1)
    return statistics.median(
        series[index].quote_volume_usd for index in range(start, decision_index + 1)
    )


def _open_return(series: Sequence[CrossSectionalCryptoBar], execution_index: int) -> float:
    current = series[execution_index].open
    nxt = series[execution_index + 1].open
    return nxt / current - 1.0 if current > 0.0 else 0.0


def _rank_ic(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    universe: Sequence[str],
    decision_index: int,
    execution_index: int,
    *,
    config: CrossSectionalCryptoConfig,
) -> float | None:
    pairs: list[tuple[float, float]] = []
    for symbol in universe:
        signal = _momentum(aligned[symbol], decision_index, config)
        if signal is None:
            continue
        pairs.append((signal, _open_return(aligned[symbol], execution_index)))
    if len(pairs) < 3:
        return None
    signal_ranks = _ranks([left for left, _right in pairs])
    forward_ranks = _ranks([right for _left, right in pairs])
    return _pearson(signal_ranks, forward_ranks)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    for rank, (index, _value) in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 0.0 else 0.0


def _robustness(
    aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]],
    *,
    base_config: CrossSectionalCryptoConfig,
    cost_model: CostModel,
) -> tuple[tuple[CrossSectionalCryptoConfig, Simulation], ...]:
    rows: list[tuple[CrossSectionalCryptoConfig, Simulation]] = []
    for config in robustness_configs():
        adjusted = replace(
            config,
            min_history_days=base_config.min_history_days,
            locked_oos_fraction=base_config.locked_oos_fraction,
            bootstrap_iterations=base_config.bootstrap_iterations,
        )
        rows.append((adjusted, _simulate(aligned, config=adjusted, cost_model=cost_model)))
    return tuple(rows)


def _pbo_report(
    robustness: Sequence[tuple[CrossSectionalCryptoConfig, Simulation]],
    *,
    pbo_splits: int,
) -> Mapping[str, object]:
    trials = [
        simulation.returns
        for _config, simulation in robustness
        if len(simulation.returns) >= pbo_splits
    ]
    if len(trials) < 2:
        return {"valid": False, "reason": "PBO requires at least two robustness trials", "pbo": 1.0}
    min_len = min(len(trial) for trial in trials)
    aligned = [tuple(trial[-min_len:]) for trial in trials]
    try:
        report = dict(pbo(aligned, n_splits=pbo_splits))
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "pbo": 1.0}
    report["valid"] = True
    return report


def _robustness_summary(
    robustness: Sequence[tuple[CrossSectionalCryptoConfig, Simulation]],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for config, simulation in robustness:
        metrics = metrics_from_returns(
            simulation.returns,
            annualization_periods=config.annualization_periods,
            turnover=simulation.turnover,
            net_cost=simulation.net_cost,
        )
        rows.append(
            {
                "lookback": config.momentum_lookback_days,
                "rebalance_days": config.rebalance_days,
                "target_vol": config.target_annual_volatility,
                "funding_gate": config.funding_gate_annualized,
                "observations": len(simulation.returns),
                "sharpe": metrics.sharpe,
                "total_return": metrics.total_return,
                "positive": metrics.total_return > 0.0,
            }
        )
    return rows


def _bootstrap_sharpe_ci(
    returns: Sequence[float], *, annualization_periods: int, iterations: int
) -> Mapping[str, float | int | None]:
    if not returns:
        return {"p05": None, "p50": None, "p95": None, "iterations": iterations}
    means = bootstrap_mean_ci(
        returns,
        iterations=iterations,
        include_iterations=True,
    )
    stdev = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    if stdev <= 0.0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "iterations": iterations}
    return {
        "p05": None
        if means["p05"] is None
        else float(means["p05"]) / stdev * math.sqrt(annualization_periods),
        "p50": None
        if means["p50"] is None
        else float(means["p50"]) / stdev * math.sqrt(annualization_periods),
        "p95": None
        if means["p95"] is None
        else float(means["p95"]) / stdev * math.sqrt(annualization_periods),
        "iterations": iterations,
    }


def _power_self_check(*, years: float, trials: int) -> Mapping[str, object]:
    z_power_80 = 0.8416212336
    # q=0.10. For G=1 z_(0.90)=1.28155; for larger G use a conservative approximation.
    alpha_tail = 1.0 - 0.10 / max(1, trials)
    z_alpha = 1.2815515655 if trials == 1 else _normal_quantile(alpha_tail)
    return {
        "method": "(z_(1-q/G)+z_power_0.8)/sqrt(years)",
        "fdr_q": 0.10,
        "power": 0.80,
        "trials": trials,
        "detectable_annualized_sharpe": (z_alpha + z_power_80) / math.sqrt(years),
    }


def _normal_quantile(probability: float) -> float:
    # Acklam-style approximation, sufficient for reporting a power threshold.
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    a = (-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((c[0] * q + c[1]) * q + c[2]) * q + c[3]) / (
            ((d[0] * q + d[1]) * q + d[2]) * q + 1.0
        )
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((c[0] * q + c[1]) * q + c[2]) * q + c[3]) / (
            ((d[0] * q + d[1]) * q + d[2]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((a[0] * r + a[1]) * r + a[2]) * r + a[3])
        * q
        / ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + 1.0)
    )


def _coverage(aligned: Mapping[str, tuple[CrossSectionalCryptoBar, ...]]) -> Mapping[str, object]:
    if not aligned:
        return {"symbols": 0, "rows_min": 0, "rows_max": 0}
    lengths = [len(values) for values in aligned.values()]
    return {
        "symbols": len(aligned),
        "rows_min": min(lengths),
        "rows_max": max(lengths),
        "symbols_with_market_cap": sum(
            1 for bars in aligned.values() if any(bar.market_cap_usd is not None for bar in bars)
        ),
        "symbols_with_exchange_count_ge_2": sum(
            1 for bars in aligned.values() if any(bar.exchange_count >= 2 for bar in bars)
        ),
    }


def _insufficient(
    reason: str,
    config: CrossSectionalCryptoConfig,
    cost_model: CostModel,
    survivor_light: bool,
    *,
    simulation: Simulation | None = None,
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": 1,
        "standard_metrics": {},
        "benchmark_metrics": {},
        "multiple_testing": {
            "candidate_count_n": 1,
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": reason},
        },
        "power_self_check": {
            "underpowered": True,
            "reason": reason,
        },
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "t_plus_1_execution": True,
            "locked_oos": True,
            "full_costs": True,
            "funding_counted": True,
            "funding_label": cost_model.funding_label,
            "survivor_light_ceiling_required": survivor_light,
        },
        "coverage": {
            "returns": 0 if simulation is None else len(simulation.returns),
            "rebalance_count": 0 if simulation is None else simulation.rebalance_count,
        },
        "config": _config_to_dict(config),
    }


def _config_to_dict(config: CrossSectionalCryptoConfig) -> Mapping[str, object]:
    return {
        "momentum_lookback_days": config.momentum_lookback_days,
        "skip_recent_days": config.skip_recent_days,
        "vol_lookback_days": config.vol_lookback_days,
        "rebalance_days": config.rebalance_days,
        "target_annual_volatility": config.target_annual_volatility,
        "funding_gate_annualized": config.funding_gate_annualized,
        "min_history_days": config.min_history_days,
        "liquidity_top_n": config.liquidity_top_n,
    }


def _metrics_to_dict(metrics: Any) -> dict[str, float]:
    return {
        "annualized_return": metrics.annualized_return,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "positive_period_win_rate": metrics.positive_period_win_rate,
        "oos_vs_buy_hold_window_win_rate": metrics.oos_vs_buy_hold_window_win_rate,
        "annualized_turnover": metrics.annualized_turnover,
        "net_cost": metrics.net_cost,
    }


def _year_month(timestamp_ms: int) -> tuple[int, int]:
    value = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return value.year, value.month


def _year_set(timestamps: Sequence[int]) -> set[int]:
    return {datetime.fromtimestamp(timestamp / 1000, tz=UTC).year for timestamp in timestamps}


def _btc_symbol(symbols: Sequence[str]) -> str:
    for symbol in symbols:
        if symbol.upper().startswith("BTC/") or symbol.upper().startswith("BTCUSDT"):
            return symbol
    return symbols[0] if symbols else ""


def _positive_share(values: Sequence[float]) -> float:
    return sum(1 for value in values if value > 0.0) / len(values) if values else 0.0


def _float(value: object, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default
