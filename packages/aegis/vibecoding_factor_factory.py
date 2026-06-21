from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from aegis.backtest_core import (
    CostModel,
    benjamini_hochberg,
    deflated_sharpe_threshold,
    metrics_from_returns,
    normal_two_sided_p,
    pbo,
    sign_test_p_value,
    trade_scorecard,
    trade_scorecard_to_dict,
)

StrategyFamily = Literal[
    "rsi_trend_reversal",
    "roc_momentum",
    "range_compression_breakout",
    "rsi_roc_dual_confirmation",
]
FactoryVerdict = Literal["SUGGESTIVE", "NO_EDGE", "INSUFFICIENT"]


@dataclass(frozen=True)
class VibeBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class VibeFactoryConfig:
    factor_horizons: tuple[int, ...] = (1, 3, 6, 12)
    train_bars_1h: int = 2160
    test_bars_1h: int = 720
    step_bars_1h: int = 720
    train_bars_4h: int = 1080
    test_bars_4h: int = 360
    step_bars_4h: int = 360
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    pbo_threshold: float = 0.50
    min_trades: int = 30
    min_oos_windows: int = 3
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    survivor_light: bool = True


DEFAULT_FACTORY_CONFIG = VibeFactoryConfig()


@dataclass(frozen=True)
class StrategyParams:
    family: StrategyFamily
    values: Mapping[str, float | int]


@dataclass(frozen=True)
class StrategySimulation:
    returns: tuple[float, ...]
    signals: tuple[bool, ...]
    trade_returns: tuple[float, ...]
    trade_log: tuple[Mapping[str, object], ...]
    turnover: float
    net_cost: float
    exit_reasons: Mapping[str, int]


FACTOR_NAMES = (
    "ret_3",
    "ret_6",
    "ret_12",
    "roc_12",
    "roc_slope",
    "rsi_7",
    "rsi_14",
    "rsi_wilder_14",
    "rsi_slope_3",
    "rsi_zscore_100",
    "range_20",
    "range_50",
    "range_position_20",
    "close_vs_ma_200",
    "ma_50_slope",
    "atr_pct_14",
    "realized_vol_24",
)

DEFAULT_COST_MODEL = CostModel(
    fee_bps=10.0,
    slippage_bps=5.0,
    funding_label="N/A for Binance spot long-only; no perp funding is modeled",
)


def run_vibecoding_factor_factory(
    frames_by_key: Mapping[str, Sequence[VibeBar]],
    *,
    config: VibeFactoryConfig = DEFAULT_FACTORY_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    generated_at: datetime | None = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(UTC)
    frames = {
        key: _sorted_bars(bars)
        for key, bars in sorted(frames_by_key.items())
        if len(bars) > 0
    }
    factor_trial_count = len(frames) * len(FACTOR_NAMES) * len(config.factor_horizons)
    strategy_trial_count = sum(
        len(strategy_grid(family)) * len(frames) for family in _strategy_families()
    )
    total_trial_count = factor_trial_count + strategy_trial_count
    if not frames:
        return _insufficient(
            generated,
            "no OHLCV frames supplied",
            total_trial_count=total_trial_count,
            factor_trial_count=factor_trial_count,
            strategy_trial_count=strategy_trial_count,
            config=config,
            cost_model=cost_model,
        )

    factor_rows = _factor_ic_rows(frames, config)
    factor_p_values = [_object_float(row["p_value"]) for row in factor_rows]
    factor_bh = benjamini_hochberg(factor_p_values, alpha=config.fdr_alpha)
    for row, passed in zip(factor_rows, factor_bh, strict=True):
        row["bh_fdr_pass"] = bool(passed)
    allowed_families = _allowed_families(factor_rows)

    strategy_rows, window_rows, pbo_rows = _walk_forward_strategy_rows(
        frames,
        allowed_families=allowed_families,
        config=config,
        cost_model=cost_model,
    )
    all_p_values = factor_p_values + [_object_float(row["p_value"]) for row in strategy_rows]
    all_bh = benjamini_hochberg(all_p_values, alpha=config.fdr_alpha) if all_p_values else []
    factor_fdr_survivors = sum(1 for passed in all_bh[: len(factor_p_values)] if passed)
    strategy_fdr_flags = all_bh[len(factor_p_values) :]
    for row, passed in zip(strategy_rows, strategy_fdr_flags, strict=True):
        row["bh_fdr_pass_total_scope"] = bool(passed)

    selected_oos_returns = tuple(
        value
        for row in window_rows
        for value in cast(tuple[float, ...], row["oos_returns"])
    )
    selected_buy_hold = tuple(
        value
        for row in window_rows
        for value in cast(tuple[float, ...], row["buy_hold_returns"])
    )
    selected_trades = tuple(
        value
        for row in window_rows
        for value in cast(tuple[float, ...], row["trade_returns"])
    )
    total_turnover = sum(_object_float(row["turnover"]) for row in window_rows)
    total_net_cost = sum(_object_float(row["net_cost"]) for row in window_rows)
    annualization = _dominant_annualization(frames)
    metrics = metrics_from_returns(
        selected_oos_returns,
        annualization_periods=annualization,
        turnover=total_turnover,
        net_cost=total_net_cost,
        oos_vs_buy_hold_window_win_rate=_window_win_rate(window_rows),
    )
    benchmark = metrics_from_returns(
        selected_buy_hold,
        annualization_periods=annualization,
        turnover=0.0,
        net_cost=0.0,
    )
    scorecard = trade_scorecard(selected_trades)
    positive_assets = {
        str(row["asset"])
        for row in window_rows
        if _compound(cast(tuple[float, ...], row["oos_returns"])) > 0
    }
    strategy_fdr_survivors = 0
    for row in strategy_rows:
        if (
            bool(row.get("bh_fdr_pass_total_scope"))
            and _object_float(row.get("net_ev", 0.0)) > 0.0
            and _pbo_passes(pbo_rows, str(row["family"]), str(row["key"]), config)
        ):
            strategy_fdr_survivors += 1
    insufficient = (
        len(selected_trades) < config.min_trades
        or _oos_window_count(window_rows) < config.min_oos_windows
    )
    green = (
        strategy_fdr_survivors > 0
        and _any_pbo_pass(pbo_rows, config)
        and metrics.total_return > 0
        and len(positive_assets) >= 2
        and not insufficient
    )
    verdict: FactoryVerdict
    reason: str
    if insufficient and not factor_rows:
        verdict = "INSUFFICIENT"
        reason = "insufficient data for factor IC and walk-forward evaluation"
    elif insufficient and strategy_fdr_survivors > 0:
        verdict = "INSUFFICIENT"
        reason = "positive-looking candidate failed minimum trade/window power gates"
    elif green:
        verdict = "SUGGESTIVE"
        reason = "all health gates passed, but survivor-light single-venue data caps at SUGGESTIVE"
    else:
        verdict = "NO_EDGE"
        reason = "no strategy survived full-scope BH-FDR, PBO, costs, and OOS health gates"
    return {
        "generated_at": generated.isoformat(),
        "briefing": "CODEX_OLYMPUS_69_VIBECODING_FACTOR_FACTORY",
        "status": "OK" if verdict != "INSUFFICIENT" else "INSUFFICIENT",
        "verdict": verdict,
        "reason": reason,
        "ev_newness_statement": (
            "Free-data technical indicators (RSI/ROC/MA/Range) are an already-disproved "
            "class in Olympus #38/#44/#45/#46/#49; expected outcome after full costs, "
            "FDR, and PBO is no survivor. Value is a reusable honest factory and a "
            "defensible close-out for this class."
        ),
        "data": {
            "frames": {key: _frame_summary(bars) for key, bars in frames.items()},
            "spot_funding": "N/A",
            "closed_klines_only": True,
        },
        "discipline": _discipline(config, cost_model),
        "factor_ic": {
            "trial_count": factor_trial_count,
            "rows": factor_rows,
            "bh_fdr_survivors_total_scope": factor_fdr_survivors,
            "allowed_strategy_families": sorted(allowed_families),
        },
        "strategy_backtest": {
            "trial_count": strategy_trial_count,
            "families_skipped_by_factor_gate": sorted(
                set(_strategy_families()).difference(allowed_families)
            ),
            "strategy_rows": strategy_rows,
            "walk_forward_windows": _public_windows(window_rows),
            "pbo": pbo_rows,
            "strategy_fdr_survivors_total_scope": strategy_fdr_survivors,
        },
        "standard_metrics": _metrics_dict(metrics),
        "benchmark_metrics": _metrics_dict(benchmark),
        "trade_scorecard": trade_scorecard_to_dict(scorecard),
        "multiple_testing": {
            "candidate_count_n": total_trial_count,
            "factor_ic_trials": factor_trial_count,
            "strategy_trials": strategy_trial_count,
            "fdr_alpha": config.fdr_alpha,
            "fdr_scope": "all factor-IC plus all strategy x asset x timeframe x parameter trials",
            "raw_min_p": min(all_p_values) if all_p_values else None,
            "fdr_survivors": factor_fdr_survivors + strategy_fdr_survivors,
            "deflated_sharpe_threshold": deflated_sharpe_threshold(
                trial_count=max(total_trial_count, 1),
                observations=max(len(selected_oos_returns), 1),
            ),
        },
        "health": {
            "score": (
                "GREEN"
                if green
                else ("INSUFFICIENT" if verdict == "INSUFFICIENT" else "AMBER")
            ),
            "paper_candidate_only": True,
            "green_requires": [
                "BH-FDR pass",
                "PBO < 0.5",
                "full-cost EV > 0",
                "enough OOS windows",
                "two assets or cross-regime consistency",
            ],
            "positive_assets": sorted(positive_assets),
            "oos_windows": _oos_window_count(window_rows),
            "trades": len(selected_trades),
        },
        "known_limits": [
            (
                "Strategy 4 fake breakout intentionally skipped because the source brief was not "
                "available and fake-breakout rules are especially leakage-prone."
            ),
            (
                "Single Binance spot venue is survivor-light and caps any positive result at "
                "SUGGESTIVE/PAPER_CANDIDATE_ONLY."
            ),
            "This is research/backtest only; no live, account, wallet, or order path is present.",
        ],
        "safety": {
            "read_only": True,
            "orders_or_wallets": False,
            "credentials_read": False,
            "survivor_light_ceiling": config.survivor_light,
            "max_positive_verdict": "SUGGESTIVE",
        },
    }


def compute_factor_table(bars: Sequence[VibeBar]) -> Mapping[str, tuple[float | None, ...]]:
    frame = _columns(_sorted_bars(bars))
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    rsi_14 = _rsi(close, 14, wilder=False)
    table: dict[str, tuple[float | None, ...]] = {
        "ret_3": _pct_change(close, 3),
        "ret_6": _pct_change(close, 6),
        "ret_12": _pct_change(close, 12),
        "roc_12": _pct_change(close, 12),
        "roc_slope": _diff(_pct_change(close, 12), 3),
        "rsi_7": _rsi(close, 7, wilder=False),
        "rsi_14": rsi_14,
        "rsi_wilder_14": _rsi(close, 14, wilder=True),
        "rsi_slope_3": _diff(rsi_14, 3),
        "rsi_zscore_100": _zscore(rsi_14, 100),
        "range_20": _range_pct(high, low, close, 20),
        "range_50": _range_pct(high, low, close, 50),
        "range_position_20": _range_position(high, low, close, 20),
        "close_vs_ma_200": _close_vs_ma(close, 200),
        "ma_50_slope": _pct_change(_sma(close, 50), 12),
        "atr_pct_14": _atr_pct(high, low, close, 14),
        "realized_vol_24": _rolling_stdev(_returns(close), 24),
    }
    # Volume is deliberately touched to keep the factor set honest about available data:
    # no volume-derived factor is predeclared for #69, so it is not returned.
    _ = volume
    return table


def strategy_grid(family: StrategyFamily) -> tuple[StrategyParams, ...]:
    params: list[StrategyParams] = []
    if family == "rsi_trend_reversal":
        for period in (7, 14):
            for oversold in (30, 40):
                for exit_rsi in (55, 65):
                    for trend_ma in (50, 200):
                        params.append(
                            StrategyParams(
                                family,
                                {
                                    "rsi_period": period,
                                    "oversold": oversold,
                                    "exit_rsi": exit_rsi,
                                    "trend_ma": trend_ma,
                                },
                            )
                        )
    elif family == "roc_momentum":
        for lookback in (12, 24):
            for threshold in (0.0, 0.02):
                for trend_ma in (50, 200):
                    params.append(
                        StrategyParams(
                            family,
                            {
                                "roc_lookback": lookback,
                                "roc_threshold": threshold,
                                "trend_ma": trend_ma,
                            },
                        )
                    )
    elif family == "range_compression_breakout":
        for range_window in (20, 50):
            for breakout_window in (20, 50):
                for max_range_pct in (0.05, 0.10):
                    params.append(
                        StrategyParams(
                            family,
                            {
                                "range_window": range_window,
                                "breakout_window": breakout_window,
                                "max_range_pct": max_range_pct,
                            },
                        )
                    )
    else:
        for rsi_period in (7, 14):
            for roc_lookback in (12, 24):
                for roc_threshold in (0.0, 0.02):
                    for min_rsi in (50, 55):
                        params.append(
                            StrategyParams(
                                family,
                                {
                                    "rsi_period": rsi_period,
                                    "roc_lookback": roc_lookback,
                                    "roc_threshold": roc_threshold,
                                    "min_rsi": min_rsi,
                                },
                            )
                        )
    return tuple(params)


def generate_strategy_signals(
    bars: Sequence[VibeBar],
    params: StrategyParams,
) -> tuple[bool, ...]:
    ordered = _sorted_bars(bars)
    columns = _columns(ordered)
    close = columns["close"]
    high = columns["high"]
    low = columns["low"]
    values = params.values
    if params.family == "rsi_trend_reversal":
        rsi = _rsi(close, _int_param(values, "rsi_period"), wilder=False)
        ma = _sma(close, _int_param(values, "trend_ma"))
        return _stateful_signals(
            [
                _valid(rsi[index])
                and _valid(rsi[index - 1] if index > 0 else None)
                and _valid(ma[index])
                and close[index] > cast(float, ma[index])
                and cast(float, rsi[index - 1]) < _float_param(values, "oversold")
                and cast(float, rsi[index]) >= _float_param(values, "oversold")
                for index in range(len(close))
            ],
            [
                _valid(rsi[index])
                and (
                    cast(float, rsi[index]) >= _float_param(values, "exit_rsi")
                    or (_valid(ma[index]) and close[index] < cast(float, ma[index]))
                )
                for index in range(len(close))
            ],
        )
    if params.family == "roc_momentum":
        roc = _pct_change(close, _int_param(values, "roc_lookback"))
        ma = _sma(close, _int_param(values, "trend_ma"))
        return _stateful_signals(
            [
                _valid(roc[index])
                and _valid(ma[index])
                and close[index] > cast(float, ma[index])
                and cast(float, roc[index]) > _float_param(values, "roc_threshold")
                for index in range(len(close))
            ],
            [
                _valid(roc[index])
                and (
                    cast(float, roc[index]) <= 0.0
                    or (_valid(ma[index]) and close[index] < cast(float, ma[index]))
                )
                for index in range(len(close))
            ],
        )
    if params.family == "range_compression_breakout":
        range_pct = _range_pct(high, low, close, _int_param(values, "range_window"))
        prior_high = _shift(_rolling_max(high, _int_param(values, "breakout_window")), 1)
        prior_low = _shift(_rolling_min(low, _int_param(values, "breakout_window")), 1)
        return _stateful_signals(
            [
                _valid(range_pct[index])
                and _valid(prior_high[index])
                and cast(float, range_pct[index]) <= _float_param(values, "max_range_pct")
                and close[index] > cast(float, prior_high[index])
                for index in range(len(close))
            ],
            [
                _valid(prior_low[index]) and close[index] < cast(float, prior_low[index])
                for index in range(len(close))
            ],
        )
    rsi = _rsi(close, _int_param(values, "rsi_period"), wilder=False)
    roc = _pct_change(close, _int_param(values, "roc_lookback"))
    return _stateful_signals(
        [
            _valid(rsi[index])
            and _valid(roc[index])
            and cast(float, rsi[index]) >= _float_param(values, "min_rsi")
            and cast(float, roc[index]) > _float_param(values, "roc_threshold")
            for index in range(len(close))
        ],
        [
            _valid(rsi[index])
            and _valid(roc[index])
            and (cast(float, rsi[index]) < 50.0 or cast(float, roc[index]) <= 0.0)
            for index in range(len(close))
        ],
    )


def simulate_strategy(
    bars: Sequence[VibeBar],
    params: StrategyParams,
    *,
    config: VibeFactoryConfig = DEFAULT_FACTORY_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> StrategySimulation:
    ordered = _sorted_bars(bars)
    signals = generate_strategy_signals(ordered, params)
    returns = [0.0 for _ in ordered]
    trades: list[float] = []
    trade_log: list[Mapping[str, object]] = []
    exit_reasons: dict[str, int] = {"signal": 0, "stop_loss": 0, "take_profit": 0, "end": 0}
    position = False
    entry_index: int | None = None
    entry_factor = 1.0
    turnover = 0.0
    net_cost = 0.0
    for index in range(1, max(len(ordered) - 1, 1)):
        desired = signals[index - 1]
        cost = 0.0
        if desired != position:
            cost = cost_model.one_way_cost
            net_cost += cost
            turnover += 1.0
            if desired:
                entry_index = index
                entry_factor = 1.0 - cost
            elif entry_index is not None:
                trade_return = entry_factor * (1.0 - cost) - 1.0
                trades.append(trade_return)
                trade_log.append(
                    _trade_row(
                        ordered,
                        params,
                        signal_index=index - 1,
                        entry_index=entry_index,
                        exit_index=index,
                        net_return=trade_return,
                        reason="signal",
                    )
                )
                exit_reasons["signal"] += 1
                entry_index = None
        position = desired
        bar_return = 0.0
        if position:
            raw_return, reason = conservative_bar_return(
                open_price=ordered[index].open,
                high=ordered[index].high,
                low=ordered[index].low,
                next_open=ordered[index + 1].open,
                take_profit_pct=config.take_profit_pct,
                stop_loss_pct=config.stop_loss_pct,
            )
            bar_return = raw_return - cost
            entry_factor *= 1.0 + raw_return
            if reason in {"stop_loss", "take_profit"}:
                exit_reasons[reason] += 1
                position = False
                signals_list = list(signals)
                if index < len(signals_list):
                    signals_list[index] = False
                signals = tuple(signals_list)
                if entry_index is not None:
                    trade_return = entry_factor * (1.0 - cost_model.one_way_cost) - 1.0
                    net_cost += cost_model.one_way_cost
                    turnover += 1.0
                    trades.append(trade_return)
                    trade_log.append(
                        _trade_row(
                            ordered,
                            params,
                            signal_index=max(entry_index - 1, 0),
                            entry_index=entry_index,
                            exit_index=index,
                            net_return=trade_return,
                            reason=reason,
                        )
                    )
                    entry_index = None
        else:
            bar_return = -cost
        returns[index] = bar_return
    if position and entry_index is not None and len(ordered) > 1:
        exit_reasons["end"] += 1
        trade_return = entry_factor * (1.0 - cost_model.one_way_cost) - 1.0
        net_cost += cost_model.one_way_cost
        turnover += 1.0
        trades.append(trade_return)
        trade_log.append(
            _trade_row(
                ordered,
                params,
                signal_index=max(entry_index - 1, 0),
                entry_index=entry_index,
                exit_index=len(ordered) - 1,
                net_return=trade_return,
                reason="end",
            )
        )
    return StrategySimulation(
        returns=tuple(returns),
        signals=signals,
        trade_returns=tuple(trades),
        trade_log=tuple(trade_log),
        turnover=turnover,
        net_cost=net_cost,
        exit_reasons=exit_reasons,
    )


def conservative_bar_return(
    *,
    open_price: float,
    high: float,
    low: float,
    next_open: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[float, str]:
    stop = open_price * (1.0 - stop_loss_pct)
    take = open_price * (1.0 + take_profit_pct)
    hit_stop = low <= stop
    hit_take = high >= take
    if hit_stop and hit_take:
        return stop / open_price - 1.0, "stop_loss"
    if hit_stop:
        return stop / open_price - 1.0, "stop_loss"
    if hit_take:
        return take / open_price - 1.0, "take_profit"
    return next_open / open_price - 1.0, "signal"


def _factor_ic_rows(
    frames: Mapping[str, Sequence[VibeBar]],
    config: VibeFactoryConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, bars in frames.items():
        closes = tuple(bar.close for bar in bars)
        table = compute_factor_table(bars)
        for factor_name in FACTOR_NAMES:
            values = table[factor_name]
            for horizon in config.factor_horizons:
                x: list[float] = []
                y: list[float] = []
                for index in range(0, len(bars) - horizon):
                    value = values[index]
                    if value is None or closes[index] <= 0:
                        continue
                    future = closes[index + horizon] / closes[index] - 1.0
                    if math.isfinite(value) and math.isfinite(future):
                        x.append(float(value))
                        y.append(float(future))
                corr = _spearman(x, y)
                observations = len(x)
                t_value = _corr_t(corr, observations)
                p_value = normal_two_sided_p(t_value) if observations >= 4 else 1.0
                rows.append(
                    {
                        "key": key,
                        "asset": _asset_from_key(key),
                        "timeframe": _timeframe_from_key(key),
                        "factor": factor_name,
                        "horizon_bars": horizon,
                        "observations": observations,
                        "rank_ic": corr,
                        "t_value": t_value,
                        "p_value": p_value,
                        "bh_fdr_pass": False,
                    }
                )
    return rows


def _walk_forward_strategy_rows(
    frames: Mapping[str, Sequence[VibeBar]],
    *,
    allowed_families: set[StrategyFamily],
    config: VibeFactoryConfig,
    cost_model: CostModel,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    strategy_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    pbo_rows: list[dict[str, object]] = []
    for key, bars in frames.items():
        timeframe = _timeframe_from_key(key)
        train_bars, test_bars, step_bars = _window_sizes(timeframe, config)
        starts = _window_starts(len(bars), train_bars, test_bars, step_bars)
        buy_hold = _buy_hold_returns(bars)
        for family in _strategy_families():
            grid = strategy_grid(family)
            simulations = [
                simulate_strategy(bars, params, config=config, cost_model=cost_model)
                for params in grid
            ]
            for params, sim in zip(grid, simulations, strict=True):
                oos_start = int(len(sim.returns) * 0.60)
                excess = tuple(
                    sim.returns[index] - buy_hold[index]
                    for index in range(oos_start, min(len(sim.returns), len(buy_hold)))
                )
                p_value = _safe_sign_test_greater(excess)
                strategy_rows.append(
                    {
                        "key": key,
                        "asset": _asset_from_key(key),
                        "timeframe": timeframe,
                        "family": family,
                        "params": dict(params.values),
                        "p_value": p_value,
                        "net_ev": statistics.fmean(excess) if excess else 0.0,
                        "trades": len(sim.trade_returns),
                        "exit_reasons": dict(sim.exit_reasons),
                        "skipped_by_factor_gate": family not in allowed_families,
                    }
                )
            if family not in allowed_families:
                continue
            pbo_rows.append(_pbo_row(key, family, simulations, config))
            for start in starts:
                train_slice = range(start, start + train_bars)
                test_slice = range(start + train_bars, start + train_bars + test_bars)
                selected_index = max(
                    range(len(simulations)),
                    key=lambda idx: _sharpe_for_range(simulations[idx].returns, train_slice),
                )
                selected = simulations[selected_index]
                oos_returns = tuple(selected.returns[index] for index in test_slice)
                oos_buy_hold = tuple(buy_hold[index] for index in test_slice)
                window_rows.append(
                    {
                        "key": key,
                        "asset": _asset_from_key(key),
                        "timeframe": timeframe,
                        "family": family,
                        "train_period": _period(bars, start, start + train_bars - 1),
                        "test_period": _period(
                            bars,
                            start + train_bars,
                            start + train_bars + test_bars - 1,
                        ),
                        "selected_params": dict(grid[selected_index].values),
                        "is_sharpe": _sharpe_for_range(selected.returns, train_slice),
                        "oos_sharpe": _sharpe_for_range(selected.returns, test_slice),
                        "oos_return": _compound(oos_returns),
                        "buy_hold_return": _compound(oos_buy_hold),
                        "oos_returns": oos_returns,
                        "buy_hold_returns": oos_buy_hold,
                        "trade_returns": selected.trade_returns,
                        "turnover": selected.turnover,
                        "net_cost": selected.net_cost,
                        "sample_trades": selected.trade_log[:3],
                    }
                )
    return strategy_rows, window_rows, pbo_rows


def _pbo_row(
    key: str,
    family: StrategyFamily,
    simulations: Sequence[StrategySimulation],
    config: VibeFactoryConfig,
) -> dict[str, object]:
    trials = [sim.returns for sim in simulations if len(sim.returns) >= config.pbo_splits]
    if len(trials) < 2:
        return {
            "key": key,
            "family": family,
            "valid": False,
            "reason": "fewer than two parameter trials with enough observations",
            "pbo": None,
        }
    try:
        result = pbo(trials, n_splits=config.pbo_splits)
    except ValueError as exc:
        return {"key": key, "family": family, "valid": False, "reason": str(exc), "pbo": None}
    return {
        "key": key,
        "family": family,
        "valid": True,
        "pbo": result["pbo"],
        "split_count": result["split_count"],
        "trial_count": result["trial_count"],
    }


def _allowed_families(rows: Sequence[Mapping[str, object]]) -> set[StrategyFamily]:
    passed = {str(row["factor"]) for row in rows if bool(row.get("bh_fdr_pass"))}
    allowed: set[StrategyFamily] = set()
    if any(name.startswith("rsi") for name in passed):
        allowed.add("rsi_trend_reversal")
    if any(name.startswith(("roc", "ret")) or name == "ma_50_slope" for name in passed):
        allowed.add("roc_momentum")
    if any(name.startswith("range") for name in passed):
        allowed.add("range_compression_breakout")
    if (
        any(name.startswith("rsi") for name in passed)
        and any(name.startswith(("roc", "ret")) for name in passed)
    ):
        allowed.add("rsi_roc_dual_confirmation")
    return allowed


def _strategy_families() -> tuple[StrategyFamily, ...]:
    return (
        "rsi_trend_reversal",
        "roc_momentum",
        "range_compression_breakout",
        "rsi_roc_dual_confirmation",
    )


def _stateful_signals(entries: Sequence[bool], exits: Sequence[bool]) -> tuple[bool, ...]:
    in_position = False
    out: list[bool] = []
    for entry, exit_signal in zip(entries, exits, strict=True):
        if in_position and exit_signal:
            in_position = False
        if not in_position and entry:
            in_position = True
        out.append(in_position)
    return tuple(out)


def _columns(bars: Sequence[VibeBar]) -> dict[str, tuple[float, ...]]:
    return {
        "open": tuple(bar.open for bar in bars),
        "high": tuple(bar.high for bar in bars),
        "low": tuple(bar.low for bar in bars),
        "close": tuple(bar.close for bar in bars),
        "volume": tuple(bar.volume for bar in bars),
    }


def _sorted_bars(bars: Sequence[VibeBar]) -> tuple[VibeBar, ...]:
    return tuple(sorted(bars, key=lambda bar: bar.timestamp))


def _pct_change(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for index, value in enumerate(values):
        prior = values[index - window] if index >= window else None
        if value is None or prior is None or prior == 0:
            out.append(None)
        else:
            out.append(float(value) / float(prior) - 1.0)
    return tuple(out)


def _returns(values: Sequence[float]) -> tuple[float | None, ...]:
    return _pct_change(values, 1)


def _diff(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for index, value in enumerate(values):
        prior = values[index - window] if index >= window else None
        out.append(None if value is None or prior is None else float(value) - float(prior))
    return tuple(out)


def _sma(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            out.append(None)
            continue
        sample = values[index + 1 - window : index + 1]
        if any(value is None for value in sample):
            out.append(None)
        else:
            out.append(statistics.fmean(cast(Sequence[float], sample)))
    return tuple(out)


def _rolling_stdev(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            out.append(None)
            continue
        sample = [
            float(value)
            for value in values[index + 1 - window : index + 1]
            if value is not None
        ]
        out.append(statistics.pstdev(sample) if len(sample) == window else None)
    return tuple(out)


def _rolling_max(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    return tuple(
        None if index + 1 < window else max(values[index + 1 - window : index + 1])
        for index in range(len(values))
    )


def _rolling_min(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    return tuple(
        None if index + 1 < window else min(values[index + 1 - window : index + 1])
        for index in range(len(values))
    )


def _shift(values: Sequence[float | None], periods: int) -> tuple[float | None, ...]:
    return tuple(
        None if index < periods else values[index - periods] for index in range(len(values))
    )


def _rsi(values: Sequence[float], period: int, *, wilder: bool) -> tuple[float | None, ...]:
    gains: list[float] = [0.0]
    losses: list[float] = [0.0]
    for index in range(1, len(values)):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    out: list[float | None] = [None for _ in values]
    if len(values) <= period:
        return tuple(out)
    avg_gain = statistics.fmean(gains[1 : period + 1])
    avg_loss = statistics.fmean(losses[1 : period + 1])
    out[period] = _rsi_value(avg_gain, avg_loss)
    for index in range(period + 1, len(values)):
        if wilder:
            avg_gain = (avg_gain * (period - 1) + gains[index]) / period
            avg_loss = (avg_loss * (period - 1) + losses[index]) / period
        else:
            avg_gain = statistics.fmean(gains[index + 1 - period : index + 1])
            avg_loss = statistics.fmean(losses[index + 1 - period : index + 1])
        out[index] = _rsi_value(avg_gain, avg_loss)
    return tuple(out)


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    relative = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative))


def _zscore(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for index, value in enumerate(values):
        if value is None or index + 1 < window:
            out.append(None)
            continue
        sample = [
            float(item)
            for item in values[index + 1 - window : index + 1]
            if item is not None
        ]
        if len(sample) != window:
            out.append(None)
            continue
        stdev = statistics.pstdev(sample)
        out.append(None if stdev == 0 else (float(value) - statistics.fmean(sample)) / stdev)
    return tuple(out)


def _range_pct(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int,
) -> tuple[float | None, ...]:
    max_high = _rolling_max(high, window)
    min_low = _rolling_min(low, window)
    out: list[float | None] = []
    for index, price in enumerate(close):
        if max_high[index] is None or min_low[index] is None or price == 0:
            out.append(None)
        else:
            out.append((cast(float, max_high[index]) - cast(float, min_low[index])) / price)
    return tuple(out)


def _range_position(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int,
) -> tuple[float | None, ...]:
    max_high = _rolling_max(high, window)
    min_low = _rolling_min(low, window)
    out: list[float | None] = []
    for index, price in enumerate(close):
        top = max_high[index]
        bottom = min_low[index]
        if top is None or bottom is None or top == bottom:
            out.append(None)
        else:
            out.append((price - bottom) / (top - bottom))
    return tuple(out)


def _close_vs_ma(close: Sequence[float], window: int) -> tuple[float | None, ...]:
    ma = _sma(close, window)
    return tuple(
        None if ma_value is None or ma_value == 0 else close[index] / ma_value - 1.0
        for index, ma_value in enumerate(ma)
    )


def _atr_pct(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int,
) -> tuple[float | None, ...]:
    true_ranges: list[float] = []
    for index in range(len(close)):
        if index == 0:
            true_ranges.append(high[index] - low[index])
        else:
            true_ranges.append(
                max(
                    high[index] - low[index],
                    abs(high[index] - close[index - 1]),
                    abs(low[index] - close[index - 1]),
                )
            )
    atr = _sma(true_ranges, window)
    return tuple(
        None if value is None or close[index] == 0 else value / close[index]
        for index, value in enumerate(atr)
    )


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    return _pearson(_ranks(x), _ranks(y))


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for _, index in ordered[cursor:end]:
            ranks[index] = average
        cursor = end
    return tuple(ranks)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    x_dev = [value - x_mean for value in x]
    y_dev = [value - y_mean for value in y]
    denom = math.sqrt(sum(value * value for value in x_dev) * sum(value * value for value in y_dev))
    if denom == 0:
        return 0.0
    return sum(a * b for a, b in zip(x_dev, y_dev, strict=True)) / denom


def _corr_t(corr: float, observations: int) -> float:
    if observations < 4 or abs(corr) >= 1.0:
        return 0.0
    return corr * math.sqrt((observations - 2) / max(1e-12, 1.0 - corr * corr))


def _safe_sign_test_greater(values: Sequence[float]) -> float:
    non_zero = [value for value in values if value != 0.0]
    n = len(non_zero)
    if n <= 1024:
        return sign_test_p_value(non_zero, alternative="greater")
    wins = sum(1 for value in non_zero if value > 0)
    mean = n * 0.5
    stdev = math.sqrt(n * 0.25)
    z_value = (wins - mean) / stdev if stdev > 0 else 0.0
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def _sharpe_for_range(values: Sequence[float], indices: range) -> float:
    sample = [values[index] for index in indices if index < len(values)]
    if len(sample) < 2:
        return 0.0
    stdev = statistics.pstdev(sample)
    return statistics.fmean(sample) / stdev if stdev > 0 else 0.0


def _buy_hold_returns(bars: Sequence[VibeBar]) -> tuple[float, ...]:
    returns = [0.0 for _ in bars]
    for index in range(1, max(len(bars) - 1, 1)):
        returns[index] = bars[index + 1].open / bars[index].open - 1.0
    return tuple(returns)


def _window_sizes(
    timeframe: str,
    config: VibeFactoryConfig,
) -> tuple[int, int, int]:
    if timeframe == "4h":
        return config.train_bars_4h, config.test_bars_4h, config.step_bars_4h
    return config.train_bars_1h, config.test_bars_1h, config.step_bars_1h


def _window_starts(total: int, train: int, test: int, step: int) -> tuple[int, ...]:
    starts: list[int] = []
    index = 0
    while index + train + test <= total:
        starts.append(index)
        index += step
    return tuple(starts)


def _period(bars: Sequence[VibeBar], start: int, end: int) -> Mapping[str, str]:
    return {
        "start": _iso(bars[max(0, start)].timestamp),
        "end": _iso(bars[min(len(bars) - 1, end)].timestamp),
    }


def _trade_row(
    bars: Sequence[VibeBar],
    params: StrategyParams,
    *,
    signal_index: int,
    entry_index: int,
    exit_index: int,
    net_return: float,
    reason: str,
) -> Mapping[str, object]:
    return {
        "family": params.family,
        "params": dict(params.values),
        "signal_timestamp": _iso(bars[signal_index].timestamp),
        "entry_timestamp": _iso(bars[entry_index].timestamp),
        "exit_timestamp": _iso(bars[exit_index].timestamp),
        "net_return_after_costs": net_return,
        "exit_reason": reason,
    }


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def _compound(values: Sequence[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _window_win_rate(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    wins = 0
    for row in rows:
        if _object_float(row["oos_return"]) > _object_float(row["buy_hold_return"]):
            wins += 1
    return wins / len(rows)


def _oos_window_count(rows: Sequence[Mapping[str, object]]) -> int:
    return len(rows)


def _object_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value).__name__}")


def _dominant_annualization(frames: Mapping[str, Sequence[VibeBar]]) -> int:
    timeframes = [_timeframe_from_key(key) for key in frames]
    return 6 * 365 if timeframes.count("4h") > timeframes.count("1h") else 24 * 365


def _metrics_dict(metrics: Any) -> Mapping[str, float]:
    return {
        "annualized_return": float(metrics.annualized_return),
        "total_return": float(metrics.total_return),
        "max_drawdown": float(metrics.max_drawdown),
        "sharpe": float(metrics.sharpe),
        "sortino": float(metrics.sortino),
        "calmar": float(metrics.calmar),
        "positive_period_win_rate": float(metrics.positive_period_win_rate),
        "oos_vs_buy_hold_window_win_rate": float(metrics.oos_vs_buy_hold_window_win_rate),
        "annualized_turnover": float(metrics.annualized_turnover),
        "net_cost": float(metrics.net_cost),
    }


def _public_windows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    public: list[Mapping[str, object]] = []
    for row in rows:
        public.append(
            {
                "key": row["key"],
                "family": row["family"],
                "train_period": row["train_period"],
                "test_period": row["test_period"],
                "selected_params": row["selected_params"],
                "is_sharpe": row["is_sharpe"],
                "oos_sharpe": row["oos_sharpe"],
                "oos_return": row["oos_return"],
                "buy_hold_return": row["buy_hold_return"],
                "sample_trades": row["sample_trades"],
            }
        )
    return public


def _pbo_passes(
    rows: Sequence[Mapping[str, object]],
    family: str,
    key: str,
    config: VibeFactoryConfig,
) -> bool:
    for row in rows:
        if row.get("family") == family and row.get("key") == key and bool(row.get("valid")):
            value = row.get("pbo")
            return isinstance(value, float) and value < config.pbo_threshold
    return False


def _any_pbo_pass(rows: Sequence[Mapping[str, object]], config: VibeFactoryConfig) -> bool:
    return any(
        bool(row.get("valid"))
        and isinstance(row.get("pbo"), float)
        and cast(float, row["pbo"]) < config.pbo_threshold
        for row in rows
    )


def _asset_from_key(key: str) -> str:
    return key.split(":", 1)[0]


def _timeframe_from_key(key: str) -> str:
    return key.rsplit(":", 1)[-1] if ":" in key else "1h"


def _valid(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _int_param(values: Mapping[str, float | int], key: str) -> int:
    return int(values[key])


def _float_param(values: Mapping[str, float | int], key: str) -> float:
    return float(values[key])


def _frame_summary(bars: Sequence[VibeBar]) -> Mapping[str, object]:
    return {
        "bars": len(bars),
        "start": _iso(bars[0].timestamp) if bars else None,
        "end": _iso(bars[-1].timestamp) if bars else None,
    }


def _discipline(config: VibeFactoryConfig, cost_model: CostModel) -> Mapping[str, object]:
    return {
        "closed_klines_only": True,
        "signal_t_entry_t_plus_1": True,
        "rolling_breakout_high_low_shifted_1": True,
        "same_bar_take_profit_stop_loss": "conservative_stop_loss_first",
        "full_costs": {
            "fee_bps": cost_model.fee_bps,
            "slippage_bps": cost_model.slippage_bps,
            "funding": cost_model.funding_label,
        },
        "walk_forward": {
            "1h": {
                "train": config.train_bars_1h,
                "test": config.test_bars_1h,
                "step": config.step_bars_1h,
            },
            "4h": {
                "train": config.train_bars_4h,
                "test": config.test_bars_4h,
                "step": config.step_bars_4h,
            },
        },
    }


def _insufficient(
    generated: datetime,
    reason: str,
    *,
    total_trial_count: int,
    factor_trial_count: int,
    strategy_trial_count: int,
    config: VibeFactoryConfig,
    cost_model: CostModel,
) -> Mapping[str, Any]:
    return {
        "generated_at": generated.isoformat(),
        "briefing": "CODEX_OLYMPUS_69_VIBECODING_FACTOR_FACTORY",
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "discipline": _discipline(config, cost_model),
        "multiple_testing": {
            "candidate_count_n": total_trial_count,
            "factor_ic_trials": factor_trial_count,
            "strategy_trials": strategy_trial_count,
        },
        "safety": {
            "read_only": True,
            "orders_or_wallets": False,
            "credentials_read": False,
            "survivor_light_ceiling": config.survivor_light,
        },
    }
