from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from aegis.backtest_core import (
    CostModel,
    ReturnMetrics,
    benjamini_hochberg,
    deflated_sharpe_threshold,
    metrics_from_returns,
    pbo,
    sign_test_p_value,
    trade_scorecard,
    trade_scorecard_to_dict,
)

TradeMode = Literal["long_flat", "long_short"]
Verdict = Literal["EDGE", "NO_EDGE", "INSUFFICIENT"]


@dataclass(frozen=True)
class FuturesBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    roll_marker: bool = False


@dataclass(frozen=True)
class FuturesAdxConfig:
    adx_thresholds: tuple[float, ...] = (18.0, 22.0, 26.0)
    ema_pairs: tuple[tuple[int, int], ...] = ((13, 34), (20, 55))
    trade_modes: tuple[TradeMode, ...] = ("long_flat", "long_short")
    train_bars: int = 756
    test_bars: int = 252
    step_bars: int = 252
    annualization_periods: int = 252
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    min_bars: int = 900
    min_trades: int = 20
    oos_start_fraction: float = 0.70
    survivor_light: bool = True


@dataclass(frozen=True)
class CandidateKey:
    symbol: str
    adx_threshold: float
    ema_fast: int
    ema_slow: int
    mode: TradeMode

    def label(self) -> str:
        return (
            f"{self.symbol}|adx={self.adx_threshold:g}|ema={self.ema_fast}/"
            f"{self.ema_slow}|mode={self.mode}"
        )


@dataclass(frozen=True)
class SimulationResult:
    returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    positions: tuple[int, ...]
    costs: tuple[float, ...]
    entry_timestamps: tuple[int, ...]
    turnover: float
    net_cost: float
    trade_returns: tuple[float, ...]


@dataclass(frozen=True)
class WalkCandidateResult:
    returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    trade_returns: tuple[float, ...]
    turnover: float
    net_cost: float
    window_count: int
    window_win_rate: float
    first_oos_entry_timestamp: int | None
    selector_max_signal_timestamp: int | None


DEFAULT_CONFIG = FuturesAdxConfig()
DEFAULT_COST_MODEL = CostModel(
    fee_bps=2.0,
    slippage_bps=3.0,
    funding_label="N/A for listed domestic futures; no perp funding",
)


def trial_count(symbols: Sequence[str], config: FuturesAdxConfig = DEFAULT_CONFIG) -> int:
    return (
        len(tuple(symbols))
        * len(config.adx_thresholds)
        * len(config.ema_pairs)
        * len(config.trade_modes)
    )


def validate_bars(
    bars_by_symbol: Mapping[str, Sequence[FuturesBar]],
    *,
    required_symbols: Sequence[str],
    config: FuturesAdxConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    missing = [symbol for symbol in required_symbols if symbol not in bars_by_symbol]
    too_short = {
        symbol: len(bars_by_symbol[symbol])
        for symbol in required_symbols
        if symbol in bars_by_symbol and len(bars_by_symbol[symbol]) < config.min_bars
    }
    malformed = {
        symbol: "non-positive OHLC or unsorted timestamps"
        for symbol in required_symbols
        if symbol in bars_by_symbol and not _bars_are_valid(bars_by_symbol[symbol])
    }
    ok = not missing and not too_short and not malformed
    return {
        "status": "OK" if ok else "INSUFFICIENT",
        "missing_symbols": missing,
        "too_short_symbols": too_short,
        "malformed_symbols": malformed,
        "required_symbols": tuple(required_symbols),
        "bar_counts": {symbol: len(bars_by_symbol.get(symbol, ())) for symbol in required_symbols},
    }


def run_underlying_adx_prefilter(
    bars_by_symbol: Mapping[str, Sequence[FuturesBar]],
    *,
    required_symbols: Sequence[str],
    config: FuturesAdxConfig = DEFAULT_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    data_source: str,
    roll_method: str,
) -> dict[str, object]:
    symbols = tuple(required_symbols)
    feasibility = validate_bars(bars_by_symbol, required_symbols=symbols, config=config)
    candidate_n = trial_count(symbols, config)
    if feasibility["status"] != "OK":
        return _insufficient(
            reason="missing or invalid daily futures OHLCV for required underlying symbols",
            symbols=symbols,
            config=config,
            cost_model=cost_model,
            data_source=data_source,
            roll_method=roll_method,
            candidate_n=candidate_n,
            feasibility=feasibility,
        )

    candidates: list[dict[str, object]] = []
    p_values: list[float] = []
    oos_trials: list[tuple[float, ...]] = []
    benchmark_by_candidate: list[tuple[float, ...]] = []
    for symbol in symbols:
        bars = tuple(bars_by_symbol[symbol])
        for adx_threshold in config.adx_thresholds:
            for ema_fast, ema_slow in config.ema_pairs:
                for mode in config.trade_modes:
                    key = CandidateKey(symbol, adx_threshold, ema_fast, ema_slow, mode)
                    walk = _walk_forward_candidate(
                        bars,
                        key=key,
                        config=config,
                        cost_model=cost_model,
                    )
                    p_value = sign_test_p_value(
                        _paired_excess(walk.returns, walk.benchmark_returns),
                        alternative="greater",
                    )
                    p_values.append(p_value)
                    oos_returns = tuple(float(value) for value in walk.returns)
                    oos_trials.append(oos_returns)
                    benchmark_by_candidate.append(
                        tuple(float(value) for value in walk.benchmark_returns)
                    )
                    metrics = metrics_from_returns(
                        oos_returns,
                        annualization_periods=config.annualization_periods,
                        turnover=walk.turnover,
                        net_cost=walk.net_cost,
                        oos_vs_buy_hold_window_win_rate=walk.window_win_rate,
                        nonpositive_annualized_return=0.0,
                    )
                    scorecard = trade_scorecard(walk.trade_returns)
                    candidates.append(
                        {
                            "key": key.label(),
                            "symbol": symbol,
                            "adx_threshold": adx_threshold,
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "mode": mode,
                            "p_value": p_value,
                            "trade_count": scorecard.total_trades,
                            "metrics": _metrics_dict(metrics),
                            "trade_scorecard": trade_scorecard_to_dict(scorecard),
                            "window_count": walk.window_count,
                            "window_win_rate": walk.window_win_rate,
                            "first_oos_entry_timestamp": walk.first_oos_entry_timestamp,
                            "selector_max_signal_timestamp": walk.selector_max_signal_timestamp,
                        }
                    )

    fdr_pass = benjamini_hochberg(p_values, alpha=config.fdr_alpha)
    pbo_report = _pbo_or_invalid(oos_trials, config)
    best_index = _best_candidate_index(candidates, fdr_pass)
    best = candidates[best_index] if best_index is not None else None
    diagnostic_best_index = max(
        range(len(candidates)),
        key=lambda index: _candidate_sharpe(candidates[index]),
    )
    fdr_survivors = sum(1 for passed in fdr_pass if passed)
    pbo_value = _finite_float(pbo_report.get("pbo"), default=1.0)
    best_excess_mean = (
        statistics.fmean(_paired_excess(oos_trials[best_index], benchmark_by_candidate[best_index]))
        if best_index is not None
        else 0.0
    )
    positive = (
        best_index is not None
        and fdr_pass[best_index]
        and pbo_report.get("valid") is True
        and pbo_value < 0.5
        and best_excess_mean > 0.0
    )
    verdict: Verdict = "EDGE" if positive else "NO_EDGE"
    if any(len(trial) < config.pbo_splits for trial in oos_trials):
        verdict = "INSUFFICIENT"
    display_verdict = "SUGGESTIVE" if positive and config.survivor_light else verdict
    return {
        "status": "OK",
        "verdict": display_verdict,
        "standard_verdict": verdict,
        "reason": _reason(
            positive=positive,
            survivor_light=config.survivor_light,
            fdr_survivors=fdr_survivors,
            pbo_valid=pbo_report.get("valid") is True,
        ),
        "symbols": symbols,
        "data": {
            "source": data_source,
            "roll_method": roll_method,
            "bar_counts": feasibility["bar_counts"],
            "survivor_light": config.survivor_light,
        },
        "config": _config_dict(config),
        "cost_model": _cost_dict(cost_model),
        "candidate_count_n": candidate_n,
        "raw_survivors": sum(1 for p_value in p_values if p_value < config.fdr_alpha),
        "fdr_survivors": fdr_survivors,
        "best_candidate": best,
        "diagnostic_best_candidate": candidates[diagnostic_best_index],
        "multiple_testing": {
            "candidate_count_n": candidate_n,
            "fdr_alpha": config.fdr_alpha,
            "p_values_min": min(p_values) if p_values else None,
            "fdr_pass": fdr_pass,
            "pbo": pbo_report,
            "deflated_sharpe_threshold": deflated_sharpe_threshold(
                trial_count=max(candidate_n, 1),
                observations=min((len(trial) for trial in oos_trials), default=0),
            ),
        },
        "candidates": candidates,
        "benchmarks": {
            "cash": 0.0,
            "buy_hold": "per-symbol t+1 open-to-open benchmark",
        },
        "inference_to_options": _options_inference(display_verdict),
        "safety": {
            "live_trading": False,
            "broker_gui": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
            "max_positive_verdict": "SUGGESTIVE",
        },
    }


def simulate_candidate(
    bars: Sequence[FuturesBar],
    *,
    key: CandidateKey,
    start: int,
    end: int,
    cost_model: CostModel,
    annualization_periods: int = 252,
) -> SimulationResult:
    if key.ema_fast >= key.ema_slow:
        raise ValueError("EMA fast window must be less than slow window")
    ordered = tuple(bars)
    warmup = max(key.ema_slow, 14 * 2) + 1
    start = max(start, warmup + 1)
    end = min(end, len(ordered) - 1)
    ema_fast = _ema([bar.close for bar in ordered], key.ema_fast)
    ema_slow = _ema([bar.close for bar in ordered], key.ema_slow)
    adx = _adx(ordered, period=14)
    returns: list[float] = []
    benchmark_returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    entry_timestamps: list[int] = []
    trade_returns: list[float] = []
    turnover = 0.0
    prev_position = 0
    current_trade_return = 0.0
    in_trade = False
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        position = _position_from_signal(
            ema_fast[decision_index],
            ema_slow[decision_index],
            adx[decision_index],
            threshold=key.adx_threshold,
            mode=key.mode,
        )
        position_change = abs(position - prev_position)
        if position_change:
            turnover += position_change
            if in_trade and prev_position != 0:
                trade_returns.append(current_trade_return)
                current_trade_return = 0.0
                in_trade = False
            if position != 0:
                entry_timestamps.append(ordered[execution_index].timestamp)
                in_trade = True
        gross = position * (ordered[execution_index + 1].open / ordered[execution_index].open - 1.0)
        trade_cost = position_change * cost_model.round_trip_bps / 10_000.0
        net = gross - trade_cost
        if position != 0:
            current_trade_return += net
        returns.append(net)
        benchmark_returns.append(
            ordered[execution_index + 1].open / ordered[execution_index].open - 1.0
        )
        positions.append(position)
        costs.append(trade_cost)
        prev_position = position
    if in_trade:
        trade_returns.append(current_trade_return)
    return SimulationResult(
        returns=tuple(returns),
        benchmark_returns=tuple(benchmark_returns),
        positions=tuple(positions),
        costs=tuple(costs),
        entry_timestamps=tuple(entry_timestamps),
        turnover=turnover,
        net_cost=sum(costs),
        trade_returns=tuple(trade_returns),
    )


def _walk_forward_candidate(
    bars: Sequence[FuturesBar],
    *,
    key: CandidateKey,
    config: FuturesAdxConfig,
    cost_model: CostModel,
) -> WalkCandidateResult:
    starts = _window_starts(len(bars), config)
    returns: list[float] = []
    benchmark_returns: list[float] = []
    trade_returns: list[float] = []
    window_wins = 0
    total_turnover = 0.0
    total_cost = 0.0
    first_entry: int | None = None
    for train_start, train_end, test_start, test_end in starts:
        result = simulate_candidate(
            bars,
            key=key,
            start=test_start,
            end=test_end,
            cost_model=cost_model,
            annualization_periods=config.annualization_periods,
        )
        returns.extend(result.returns)
        benchmark_returns.extend(result.benchmark_returns)
        trade_returns.extend(result.trade_returns)
        total_turnover += result.turnover
        total_cost += result.net_cost
        if sum(result.returns) > sum(result.benchmark_returns):
            window_wins += 1
        if result.entry_timestamps and first_entry is None:
            first_entry = result.entry_timestamps[0]
        assert train_start < train_end <= test_start < test_end
    return WalkCandidateResult(
        returns=tuple(returns),
        benchmark_returns=tuple(benchmark_returns),
        trade_returns=tuple(trade_returns),
        turnover=total_turnover,
        net_cost=total_cost,
        window_count=len(starts),
        window_win_rate=window_wins / len(starts) if starts else 0.0,
        first_oos_entry_timestamp=first_entry,
        selector_max_signal_timestamp=starts[0][1] - 1 if starts else None,
    )


def _window_starts(
    bar_count: int,
    config: FuturesAdxConfig,
) -> tuple[tuple[int, int, int, int], ...]:
    windows: list[tuple[int, int, int, int]] = []
    train = config.train_bars
    test = config.test_bars
    step = config.step_bars
    start = 0
    while start + train + test <= bar_count:
        train_start = start
        train_end = start + train
        test_start = train_end
        test_end = test_start + test
        windows.append((train_start, train_end, test_start, test_end))
        start += step
    return tuple(windows)


def _position_from_signal(
    fast: float | None,
    slow: float | None,
    adx_value: float | None,
    *,
    threshold: float,
    mode: TradeMode,
) -> int:
    if fast is None or slow is None or adx_value is None or adx_value < threshold:
        return 0
    if fast > slow:
        return 1
    if fast < slow and mode == "long_short":
        return -1
    return 0


def _ema(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    if window < 1:
        raise ValueError("EMA window must be positive")
    alpha = 2.0 / (window + 1.0)
    out: list[float | None] = []
    current: float | None = None
    for index, value in enumerate(values):
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        out.append(current if index >= window - 1 else None)
    return tuple(out)


def _adx(bars: Sequence[FuturesBar], *, period: int) -> tuple[float | None, ...]:
    if period < 1:
        raise ValueError("ADX period must be positive")
    true_ranges: list[float] = [0.0]
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for index in range(1, len(bars)):
        current = bars[index]
        previous = bars[index - 1]
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    out: list[float | None] = [None for _ in bars]
    dx_values: list[float | None] = [None for _ in bars]
    for index in range(period, len(bars)):
        tr_sum = sum(true_ranges[index - period + 1 : index + 1])
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * sum(plus_dm[index - period + 1 : index + 1]) / tr_sum
        minus_di = 100.0 * sum(minus_dm[index - period + 1 : index + 1]) / tr_sum
        denom = plus_di + minus_di
        dx_values[index] = 0.0 if denom == 0 else 100.0 * abs(plus_di - minus_di) / denom
    for index in range(period * 2 - 1, len(bars)):
        window = [value for value in dx_values[index - period + 1 : index + 1] if value is not None]
        if len(window) == period:
            out[index] = statistics.fmean(window)
    return tuple(out)


def _paired_excess(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> tuple[float, ...]:
    n = min(len(strategy_returns), len(benchmark_returns))
    return tuple(
        float(strategy_returns[index]) - float(benchmark_returns[index]) for index in range(n)
    )


def _best_candidate_index(
    candidates: Sequence[Mapping[str, object]],
    fdr_pass: Sequence[bool],
) -> int | None:
    eligible = [index for index, passed in enumerate(fdr_pass) if passed]
    if not eligible:
        return None
    return max(eligible, key=lambda index: _candidate_sharpe(candidates[index]))


def _candidate_sharpe(candidate: Mapping[str, object]) -> float:
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    value = metrics.get("sharpe", 0.0)
    return float(value) if isinstance(value, (float, int)) else 0.0


def _pbo_or_invalid(
    trials: Sequence[Sequence[float]],
    config: FuturesAdxConfig,
) -> dict[str, object]:
    try:
        result: dict[str, object] = dict(pbo(trials, n_splits=config.pbo_splits))
        result["valid"] = True
        return result
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "pbo": 1.0, "n_splits": config.pbo_splits}


def _bars_are_valid(bars: Sequence[FuturesBar]) -> bool:
    previous: int | None = None
    for bar in bars:
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            return False
        if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
            return False
        if previous is not None and bar.timestamp <= previous:
            return False
        previous = bar.timestamp
    return True


def _metrics_dict(metrics: ReturnMetrics) -> dict[str, float]:
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


def _config_dict(config: FuturesAdxConfig) -> dict[str, object]:
    return {
        "adx_thresholds": config.adx_thresholds,
        "ema_pairs": config.ema_pairs,
        "trade_modes": config.trade_modes,
        "train_bars": config.train_bars,
        "test_bars": config.test_bars,
        "step_bars": config.step_bars,
        "annualization_periods": config.annualization_periods,
        "fdr_alpha": config.fdr_alpha,
        "pbo_splits": config.pbo_splits,
        "min_bars": config.min_bars,
        "min_trades": config.min_trades,
        "survivor_light": config.survivor_light,
    }


def _cost_dict(cost_model: CostModel) -> dict[str, object]:
    return {
        "fee_bps": cost_model.fee_bps,
        "slippage_bps": cost_model.slippage_bps,
        "funding_bps_per_period": cost_model.funding_bps_per_period,
        "funding_label": cost_model.funding_label,
    }


def _insufficient(
    *,
    reason: str,
    symbols: tuple[str, ...],
    config: FuturesAdxConfig,
    cost_model: CostModel,
    data_source: str,
    roll_method: str,
    candidate_n: int,
    feasibility: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "standard_verdict": "INSUFFICIENT",
        "reason": reason,
        "symbols": symbols,
        "data": {
            "source": data_source,
            "roll_method": roll_method,
            "feasibility": feasibility,
            "survivor_light": config.survivor_light,
        },
        "config": _config_dict(config),
        "cost_model": _cost_dict(cost_model),
        "candidate_count_n": candidate_n,
        "raw_survivors": 0,
        "fdr_survivors": 0,
        "multiple_testing": {"candidate_count_n": candidate_n},
        "best_candidate": None,
        "diagnostic_best_candidate": None,
        "benchmarks": {"cash": 0.0, "buy_hold": "not evaluated"},
        "inference_to_options": _options_inference("INSUFFICIENT"),
        "safety": {
            "live_trading": False,
            "broker_gui": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
            "max_positive_verdict": "SUGGESTIVE",
        },
    }


def _reason(
    *,
    positive: bool,
    survivor_light: bool,
    fdr_survivors: int,
    pbo_valid: bool,
) -> str:
    if positive and survivor_light:
        return "underlying directional core passed gates but is capped to SUGGESTIVE by data limits"
    if positive:
        return "underlying directional core passed FDR/PBO/EV gates"
    if fdr_survivors == 0:
        return "no predeclared underlying EMA/ADX candidate survived BH-FDR"
    if not pbo_valid:
        return "FDR candidates exist but PBO is invalid or underpowered"
    return "underlying EMA/ADX candidates failed PBO or positive excess EV gate"


def _options_inference(verdict: str) -> str:
    if verdict == "NO_EDGE":
        return (
            "Underlying directional core has no edge after costs; naked long options would add "
            "theta and wider spreads, so mechanism① carries the burden of proving convexity adds "
            "value before option-data spend is justified."
        )
    if verdict in {"EDGE", "SUGGESTIVE"}:
        return (
            "Underlying directional core evidence does not prove the option version; option theta, "
            "IV crush, and bid/ask drag still require PIT option-chain validation."
        )
    return "No inference to options is made because the underlying data gate is insufficient."


def _finite_float(value: object, *, default: float) -> float:
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return default
