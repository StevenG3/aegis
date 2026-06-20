from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from aegis.backtest_core import (
    benjamini_hochberg,
    metrics_from_returns,
    pbo,
    sign_test_p_value,
    trade_scorecard,
    trade_scorecard_to_dict,
)

MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR = 15_000.0


@dataclass(frozen=True)
class MicrostructureBar:
    symbol: str
    timestamp: int
    close: float
    open_interest: float
    funding_rate: float
    buy_volume: float
    sell_volume: float
    order_book_event_rate_per_hour: float = 0.0
    survivor_status: str = "active"


@dataclass(frozen=True)
class MicrostructureCandidate:
    name: str
    funding_abs_bps: float
    imbalance_abs: float
    oi_drop_abs: float
    score_threshold: int


@dataclass(frozen=True)
class CandidateResult:
    symbol: str
    candidate: MicrostructureCandidate
    returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    trade_returns: tuple[float, ...]
    fold_excess: tuple[float, ...]
    p_value: float
    turnover: float
    net_cost: float


def run_microstructure_perp_from_spec(spec: Any) -> Mapping[str, Any]:
    """Run the offline perp microstructure hypothesis against private spec params."""
    params = _mapping(spec.params, "params")
    bars = _bars_from_params(params)
    if not bars:
        return _insufficient_payload("params.observations is empty")

    filtered, excluded_symbols = _exclude_data_blocked_symbols(bars)
    if not filtered:
        return _insufficient_payload(
            "all symbols excluded by order_book_event_rate_per_hour data-block",
            excluded_symbols=excluded_symbols,
        )

    by_symbol = _group_bars(filtered)
    usable_symbols = tuple(sorted(by_symbol))
    if not usable_symbols:
        return _insufficient_payload("no usable symbol histories after filtering")

    grid = _candidate_grid(params)
    if not grid:
        return _insufficient_payload("predeclared microstructure grid is empty")

    costs = _costs(spec.cost_model)
    locked_oos_fraction = _float_param(params, "locked_oos_fraction", 0.40)
    fold_count = max(2, _int_param(params, "fold_count", 4))
    annualization = max(1, _int_param(params, "annualization_periods", 365 * 3))
    fdr_alpha = _float_param(params, "fdr_alpha", 0.10)
    pbo_splits = _int_param(params, "pbo_splits", 4)
    pbo_threshold = _float_param(params, "pbo_threshold", 0.20)

    results = tuple(
        _evaluate_candidate(
            candidate,
            symbol=symbol,
            bars=bars,
            costs=costs,
            locked_oos_fraction=locked_oos_fraction,
            fold_count=fold_count,
        )
        for symbol, bars in sorted(by_symbol.items())
        for candidate in grid
    )
    valid_results = tuple(result for result in results if result.returns)
    if not valid_results:
        return _insufficient_payload(
            "insufficient locked-OOS t+1 samples after applying grid",
            excluded_symbols=excluded_symbols,
            candidate_count=max(_spec_trial_count(spec), len(grid) * len(by_symbol)),
        )

    p_values = [result.p_value for result in valid_results]
    fdr_flags = benjamini_hochberg(p_values, alpha=fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(valid_results, pbo_splits=pbo_splits)
    pbo_value = _number_from_mapping(pbo_report, "pbo", 1.0)
    pbo_pass = pbo_value <= pbo_threshold
    survivors = [
        result
        for result, fdr_pass in zip(valid_results, fdr_flags, strict=True)
        if fdr_pass and pbo_pass and _beats_benchmark(result, annualization)
    ]
    best = max(valid_results, key=_candidate_rank)
    strategy_metrics = metrics_from_returns(
        best.returns,
        annualization_periods=annualization,
        turnover=best.turnover,
        net_cost=best.net_cost,
        oos_vs_buy_hold_window_win_rate=_positive_share(best.fold_excess),
    )
    benchmark_metrics = metrics_from_returns(
        best.benchmark_returns,
        annualization_periods=annualization,
        turnover=0.0,
        net_cost=0.0,
    )
    scorecard = trade_scorecard(best.trade_returns)
    verdict = "EDGE" if survivors else "NO_EDGE"
    status = "OK"
    reason = (
            "predeclared microstructure symbol-grid survived BH-FDR and PBO"
        if survivors
        else "no predeclared microstructure symbol-grid survived BH-FDR, PBO, and buy-hold gates"
    )
    candidate_count_n = max(_spec_trial_count(spec), len(grid) * len(by_symbol))
    return {
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "strategy": "microstructure_perp_funding_oi_orderflow",
        "standard_metrics": _metrics_to_dict(strategy_metrics),
        "benchmark_metrics": {"buy_and_hold": _metrics_to_dict(benchmark_metrics)},
        "trade_scorecard": trade_scorecard_to_dict(scorecard),
        "candidate_count_n": candidate_count_n,
        "raw_is_survivors": sum(1 for result in valid_results if _candidate_rank(result) > 0.0),
        "fdr_is_survivors": sum(1 for value in fdr_flags if value),
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": candidate_count_n,
            "tested_candidates": len(valid_results),
            "pooled_grid_candidates": len(grid),
            "symbol_count": len(by_symbol),
            "fdr_alpha": fdr_alpha,
            "fdr_before": sum(1 for result in valid_results if result.p_value < fdr_alpha),
            "fdr_after": sum(1 for value in fdr_flags if value),
            "pbo_before_survivors": sum(1 for value in fdr_flags if value),
            "pbo_after_survivors": len(survivors),
            "pbo_threshold": pbo_threshold,
            "pbo": pbo_report,
        },
        "safety": {
            "local_file_only": True,
            "network": False,
            "live": False,
            "t_plus_1_execution": True,
            "locked_oos": True,
            "walk_forward": True,
            "full_costs": True,
            "perp_funding_counted": True,
            "survivor_light_ceiling_required": True,
            "order_book_event_rate_cap_per_hour": MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR,
            "excluded_data_blocked_symbols": excluded_symbols,
        },
        "best_candidate": _best_candidate_to_dict(best),
        "universe": {
            "usable_symbols": usable_symbols,
            "excluded_data_blocked_symbols": excluded_symbols,
            "survivor_light": True,
        },
    }


def _bars_from_params(params: Mapping[str, object]) -> tuple[MicrostructureBar, ...]:
    raw = params.get("observations", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("params.observations must be a list of bar objects")
    return tuple(_bar_from_mapping(_mapping(item, "observation")) for item in raw)


def _bar_from_mapping(item: Mapping[str, object]) -> MicrostructureBar:
    return MicrostructureBar(
        symbol=_required_str(item, "symbol"),
        timestamp=_required_int(item, "timestamp"),
        close=_positive_float(item, "close"),
        open_interest=_nonnegative_float(item, "open_interest"),
        funding_rate=_float_value(item, "funding_rate"),
        buy_volume=_nonnegative_float(item, "buy_volume"),
        sell_volume=_nonnegative_float(item, "sell_volume"),
        order_book_event_rate_per_hour=_nonnegative_float(
            item, "order_book_event_rate_per_hour", default=0.0
        ),
        survivor_status=_str_value(item.get("survivor_status", "active")),
    )


def _exclude_data_blocked_symbols(
    bars: Sequence[MicrostructureBar],
) -> tuple[tuple[MicrostructureBar, ...], tuple[str, ...]]:
    blocked = {
        bar.symbol
        for bar in bars
        if bar.order_book_event_rate_per_hour > MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR
    }
    kept = tuple(bar for bar in bars if bar.symbol not in blocked)
    return kept, tuple(sorted(blocked))


def _group_bars(
    bars: Sequence[MicrostructureBar],
) -> dict[str, tuple[MicrostructureBar, ...]]:
    grouped: dict[str, list[MicrostructureBar]] = {}
    for bar in bars:
        grouped.setdefault(bar.symbol, []).append(bar)
    return {
        symbol: tuple(sorted(items, key=lambda item: item.timestamp))
        for symbol, items in grouped.items()
    }


def _candidate_grid(params: Mapping[str, object]) -> tuple[MicrostructureCandidate, ...]:
    grid = _mapping(params.get("grid", {}), "params.grid")
    funding_values = _float_list(grid.get("funding_abs_bps", (1.0, 3.0)))
    imbalance_values = _float_list(grid.get("imbalance_abs", (0.10, 0.20)))
    oi_drop_values = _float_list(grid.get("oi_drop_abs", (0.02, 0.05)))
    score_values = _int_list(grid.get("score_threshold", (1, 2)))
    candidates: list[MicrostructureCandidate] = []
    for funding_bps, imbalance, oi_drop, score in product(
        funding_values, imbalance_values, oi_drop_values, score_values
    ):
        candidates.append(
            MicrostructureCandidate(
                name=(
                    f"fund{funding_bps:g}_imb{imbalance:g}_"
                    f"oidrop{oi_drop:g}_score{score}"
                ),
                funding_abs_bps=funding_bps,
                imbalance_abs=imbalance,
                oi_drop_abs=oi_drop,
                score_threshold=score,
            )
        )
    return tuple(candidates)


def _evaluate_candidate(
    candidate: MicrostructureCandidate,
    *,
    symbol: str,
    bars: Sequence[MicrostructureBar],
    costs: Mapping[str, float],
    locked_oos_fraction: float,
    fold_count: int,
) -> CandidateResult:
    (
        strategy_returns,
        benchmark_returns,
        trade_returns,
        total_turnover,
        total_cost,
    ) = _symbol_returns(
        candidate,
        bars=bars,
        costs=costs,
        locked_oos_fraction=locked_oos_fraction,
    )
    fold_excess = _fold_excess(strategy_returns, benchmark_returns, fold_count)
    return CandidateResult(
        symbol=symbol,
        candidate=candidate,
        returns=tuple(strategy_returns),
        benchmark_returns=tuple(benchmark_returns),
        trade_returns=tuple(trade_returns),
        fold_excess=fold_excess,
        p_value=sign_test_p_value(fold_excess, alternative="greater"),
        turnover=total_turnover,
        net_cost=total_cost,
    )


def _symbol_returns(
    candidate: MicrostructureCandidate,
    *,
    bars: Sequence[MicrostructureBar],
    costs: Mapping[str, float],
    locked_oos_fraction: float,
) -> tuple[list[float], list[float], list[float], float, float]:
    if len(bars) < 6:
        return [], [], [], 0.0, 0.0
    start = max(1, int(len(bars) * (1.0 - locked_oos_fraction)))
    strategy: list[float] = []
    benchmark: list[float] = []
    trades: list[float] = []
    turnover = 0.0
    net_cost = 0.0
    previous_position = 0
    for index in range(start, len(bars) - 2):
        signal = _signal(candidate, previous=bars[index - 1], current=bars[index])
        entry = bars[index + 1]
        exit_bar = bars[index + 2]
        gross = signal * (exit_bar.close / entry.close - 1.0)
        position_change = abs(signal - previous_position)
        trade_cost = position_change * costs["one_way_cost"]
        funding_cost = signal * entry.funding_rate
        net = gross - trade_cost - funding_cost
        strategy.append(net)
        benchmark.append(exit_bar.close / entry.close - 1.0)
        turnover += float(position_change)
        net_cost += trade_cost + abs(funding_cost)
        if signal != 0:
            trades.append(net)
        previous_position = signal
    return strategy, benchmark, trades, turnover, net_cost


def _signal(
    candidate: MicrostructureCandidate,
    *,
    previous: MicrostructureBar,
    current: MicrostructureBar,
) -> int:
    score = 0
    funding_bps = current.funding_rate * 10_000.0
    if funding_bps >= candidate.funding_abs_bps:
        score -= 1
    elif funding_bps <= -candidate.funding_abs_bps:
        score += 1

    price_change = current.close / previous.close - 1.0
    oi_change = (
        current.open_interest / previous.open_interest - 1.0
        if previous.open_interest > 0
        else 0.0
    )
    if price_change > 0.0 and oi_change <= -candidate.oi_drop_abs:
        score -= 1
    elif price_change < 0.0 and oi_change <= -candidate.oi_drop_abs:
        score += 1

    total_volume = current.buy_volume + current.sell_volume
    imbalance = (
        (current.buy_volume - current.sell_volume) / total_volume if total_volume > 0 else 0.0
    )
    if imbalance >= candidate.imbalance_abs:
        score += 1
    elif imbalance <= -candidate.imbalance_abs:
        score -= 1

    if score >= candidate.score_threshold:
        return 1
    if score <= -candidate.score_threshold:
        return -1
    return 0


def _fold_excess(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    fold_count: int,
) -> tuple[float, ...]:
    n = min(len(strategy_returns), len(benchmark_returns))
    if n == 0:
        return ()
    folds = min(fold_count, n)
    fold_size = max(1, n // folds)
    values: list[float] = []
    for start in range(0, n, fold_size):
        stop = min(n, start + fold_size)
        strategy = math.prod(1.0 + value for value in strategy_returns[start:stop]) - 1.0
        benchmark = math.prod(1.0 + value for value in benchmark_returns[start:stop]) - 1.0
        values.append(strategy - benchmark)
    return tuple(values)


def _pbo_report(
    results: Sequence[CandidateResult],
    *,
    pbo_splits: int,
) -> Mapping[str, object]:
    trials = [result.fold_excess for result in results if result.fold_excess]
    if len(trials) < 2:
        return {"valid": False, "reason": "PBO requires at least two candidates", "pbo": 1.0}
    min_len = min(len(trial) for trial in trials)
    effective_splits = min(pbo_splits, min_len)
    if effective_splits % 2 != 0:
        effective_splits -= 1
    if effective_splits < 4:
        return {
            "valid": False,
            "reason": "PBO requires at least four OOS fold observations",
            "pbo": 1.0,
            "trial_count": len(trials),
            "observation_count": min_len,
        }
    aligned = [tuple(trial[:min_len]) for trial in trials]
    report = dict(pbo(aligned, n_splits=effective_splits))
    report["valid"] = True
    return report


def _beats_benchmark(result: CandidateResult, annualization_periods: int) -> bool:
    strategy = metrics_from_returns(
        result.returns,
        annualization_periods=annualization_periods,
        turnover=result.turnover,
        net_cost=result.net_cost,
    )
    benchmark = metrics_from_returns(
        result.benchmark_returns,
        annualization_periods=annualization_periods,
        turnover=0.0,
        net_cost=0.0,
    )
    return strategy.total_return > benchmark.total_return and strategy.sharpe > benchmark.sharpe


def _candidate_rank(result: CandidateResult) -> float:
    return statistics.fmean(result.fold_excess) if result.fold_excess else -math.inf


def _costs(cost_model: object) -> Mapping[str, float]:
    model = _mapping(cost_model, "cost_model")
    fee_bps = _float_value(model, "fee_bps")
    slippage_bps = _float_value(model, "slippage_bps")
    return {"one_way_cost": (fee_bps + slippage_bps) / 10_000.0}


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


def _candidate_to_dict(candidate: MicrostructureCandidate) -> dict[str, float | int | str]:
    return {
        "name": candidate.name,
        "funding_abs_bps": candidate.funding_abs_bps,
        "imbalance_abs": candidate.imbalance_abs,
        "oi_drop_abs": candidate.oi_drop_abs,
        "score_threshold": candidate.score_threshold,
    }


def _best_candidate_to_dict(result: CandidateResult) -> dict[str, float | int | str]:
    payload = _candidate_to_dict(result.candidate)
    payload["symbol"] = result.symbol
    return payload


def _spec_trial_count(spec: Any) -> int:
    value = getattr(spec, "trial_count_n", 0)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def _insufficient_payload(
    reason: str,
    *,
    excluded_symbols: Sequence[str] = (),
    candidate_count: int = 0,
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "candidate_count_n": candidate_count,
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": candidate_count,
            "fdr_after": 0,
            "pbo_after_survivors": 0,
        },
        "safety": {
            "local_file_only": True,
            "network": False,
            "live": False,
            "excluded_data_blocked_symbols": tuple(excluded_symbols),
        },
    }


def _positive_share(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value > 0.0) / len(values)


def _number_from_mapping(raw: Mapping[str, object], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _mapping(raw: object, name: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _str_value(raw: object) -> str:
    return str(raw).strip() if raw is not None else ""


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_value(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _positive_float(raw: Mapping[str, object], key: str) -> float:
    value = _float_value(raw, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _nonnegative_float(
    raw: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    if key not in raw and default is not None:
        return default
    value = _float_value(raw, key)
    if value < 0.0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def _float_param(params: Mapping[str, object], key: str, default: float) -> float:
    if key not in params:
        return default
    return _float_value(params, key)


def _int_param(params: Mapping[str, object], key: str, default: int) -> int:
    if key not in params:
        return default
    return _required_int(params, key)


def _float_list(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("grid values must be lists")
    values = tuple(float(value) for value in raw)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("grid float values must be nonnegative finite numbers")
    return values


def _int_list(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("grid values must be lists")
    values = tuple(int(value) for value in raw)
    if not values or any(value < 1 for value in values):
        raise ValueError("grid integer values must be positive")
    return values
