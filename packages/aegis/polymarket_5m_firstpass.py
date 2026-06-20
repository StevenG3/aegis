from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aegis.backtest_core import benjamini_hochberg, pbo, sign_test_p_value

Direction = Literal["Up", "Down"]
ExitMode = Literal["settlement", "preclose_30s"]

MOVE_THRESHOLDS = (50.0, 70.0, 100.0, 150.0)
ENTRY_WINDOWS = ((150, 90), (120, 60))
PRICE_BANDS = ((0.70, 0.90), (0.80, 0.95), (0.85, 0.99))
EXIT_MODES: tuple[ExitMode, ...] = ("settlement", "preclose_30s")
UNMODELED_EXECUTION_COSTS = (
    "buy_at_ask_not_mid_or_last",
    "historical_bid_ask_spread_missing",
    "historical_depth_missing",
    "last_second_btc_reversal",
    "fak_non_fill",
    "quote_stale_or_sparse",
)


@dataclass(frozen=True)
class PricePoint:
    timestamp: int
    price: float


@dataclass(frozen=True)
class Polymarket5mObservation:
    condition_id: str
    slug: str
    title: str
    start_ts: int
    end_ts: int
    settlement_direction: Direction
    btc_move_usd: float
    btc_direction: Direction
    up_prices: tuple[PricePoint, ...]
    down_prices: tuple[PricePoint, ...]


@dataclass(frozen=True)
class FirstpassCandidate:
    name: str
    move_threshold_usd: float
    window_start_seconds: int
    window_end_seconds: int
    min_price: float
    max_price: float
    exit_mode: ExitMode


@dataclass(frozen=True)
class FirstpassTrade:
    condition_id: str
    slug: str
    direction: Direction
    decision_timestamp: int
    seconds_to_close: int
    entry_price: float
    exit_price: float
    settlement_direction: Direction
    btc_move_usd: float
    net_return: float
    outcome_correct: bool


@dataclass(frozen=True)
class CandidateResult:
    candidate: FirstpassCandidate
    trades: tuple[FirstpassTrade, ...]
    fold_excess: tuple[float, ...]
    p_value: float


def run_polymarket_5m_firstpass(observations: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
    parsed = tuple(_observation_from_mapping(row) for row in observations)
    if not parsed:
        return _insufficient("no aligned BTC 5m Polymarket observations")
    candidates = _candidate_grid()
    results = tuple(_evaluate_candidate(candidate, parsed) for candidate in candidates)
    valid_results = tuple(result for result in results if result.trades)
    if not valid_results:
        return _insufficient(
            "no entries passed the predeclared optimistic first-pass grid",
            market_count=len(parsed),
            candidate_count=len(candidates),
        )

    p_values = [result.p_value for result in valid_results]
    fdr_flags = benjamini_hochberg(p_values, alpha=0.10, tie_policy="rank")
    pbo_report = _pbo_report(valid_results, pbo_splits=4)
    pbo_value = _number_from_mapping(pbo_report, "pbo", 1.0)
    survivors = [
        result
        for result, fdr_pass in zip(valid_results, fdr_flags, strict=True)
        if fdr_pass and pbo_value <= 0.20 and _mean_return(result.trades) > 0.0
    ]
    best = max(valid_results, key=lambda result: statistics.fmean(result.fold_excess))
    verdict = "SUGGESTIVE_NEEDS_EXECUTION_VALIDATION" if survivors else "NO_EDGE"
    reason = (
        "optimistic observed-price first pass survived FDR/PBO but execution costs are unmodeled"
        if survivors
        else "no optimistic observed-price candidate survived FDR/PBO with positive mean return"
    )
    threshold_counts = _threshold_entry_counts(valid_results)
    return {
        "status": "OK",
        "verdict": verdict,
        "reason": reason,
        "strategy": "polymarket_btc_5m_firstpass_optimistic",
        "candidate_count_n": len(candidates),
        "raw_is_survivors": sum(
            1 for result in valid_results if _mean_return(result.trades) > 0.0
        ),
        "fdr_is_survivors": sum(1 for value in fdr_flags if value),
        "standard_metrics": _metrics(best.trades),
        "benchmark_metrics": _benchmarks(parsed, best),
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": len(candidates),
            "tested_candidates": len(valid_results),
            "fdr_alpha": 0.10,
            "fdr_before": sum(1 for p_value in p_values if p_value < 0.10),
            "fdr_after": sum(1 for value in fdr_flags if value),
            "pbo_threshold": 0.20,
            "pbo_after_survivors": len(survivors),
            "pbo": pbo_report,
        },
        "coverage": {
            "market_count": len(parsed),
            "date_range": _date_range(parsed),
            "entry_count": sum(len(result.trades) for result in valid_results),
            "entry_count_by_move_threshold": threshold_counts,
        },
        "best_candidate": _candidate_to_dict(best.candidate),
        "optimistic_boundary": {
            "optimistic_only": True,
            "positive_verdict_ceiling": "SUGGESTIVE_NEEDS_EXECUTION_VALIDATION",
            "unmodeled_execution_costs": UNMODELED_EXECUTION_COSTS,
            "robust_or_edge_claim_allowed": False,
        },
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
        "sample_trades": [_trade_to_dict(trade) for trade in best.trades[:10]],
    }


def _candidate_grid() -> tuple[FirstpassCandidate, ...]:
    candidates: list[FirstpassCandidate] = []
    for move in MOVE_THRESHOLDS:
        for window_start, window_end in ENTRY_WINDOWS:
            for lower, upper in PRICE_BANDS:
                for exit_mode in EXIT_MODES:
                    candidates.append(
                        FirstpassCandidate(
                            name=(
                                f"move{move:g}_w{window_start}-{window_end}_"
                                f"p{lower:g}-{upper:g}_{exit_mode}"
                            ),
                            move_threshold_usd=move,
                            window_start_seconds=window_start,
                            window_end_seconds=window_end,
                            min_price=lower,
                            max_price=upper,
                            exit_mode=exit_mode,
                        )
                    )
    return tuple(candidates)


def _evaluate_candidate(
    candidate: FirstpassCandidate,
    observations: Sequence[Polymarket5mObservation],
) -> CandidateResult:
    trades = tuple(
        trade
        for observation in sorted(observations, key=lambda row: row.end_ts)
        if (trade := _trade_for_observation(candidate, observation)) is not None
    )
    fold_excess = _fold_excess(trades, fold_count=4)
    return CandidateResult(
        candidate=candidate,
        trades=trades,
        fold_excess=fold_excess,
        p_value=sign_test_p_value(fold_excess, alternative="greater"),
    )


def _trade_for_observation(
    candidate: FirstpassCandidate,
    observation: Polymarket5mObservation,
) -> FirstpassTrade | None:
    if abs(observation.btc_move_usd) < candidate.move_threshold_usd:
        return None
    direction = observation.btc_direction
    prices = observation.up_prices if direction == "Up" else observation.down_prices
    entry = _entry_price_point(candidate, prices, observation.end_ts)
    if entry is None:
        return None
    exit_price = _exit_price(candidate, prices, entry.timestamp, observation)
    if exit_price is None:
        return None
    outcome_correct = observation.settlement_direction == direction
    net_return = exit_price / entry.price - 1.0
    return FirstpassTrade(
        condition_id=observation.condition_id,
        slug=observation.slug,
        direction=direction,
        decision_timestamp=entry.timestamp,
        seconds_to_close=observation.end_ts - entry.timestamp,
        entry_price=entry.price,
        exit_price=exit_price,
        settlement_direction=observation.settlement_direction,
        btc_move_usd=observation.btc_move_usd,
        net_return=net_return,
        outcome_correct=outcome_correct,
    )


def _entry_price_point(
    candidate: FirstpassCandidate,
    prices: Sequence[PricePoint],
    end_ts: int,
) -> PricePoint | None:
    eligible = [
        point
        for point in sorted(prices, key=lambda value: value.timestamp)
        if candidate.window_end_seconds
        <= end_ts - point.timestamp
        <= candidate.window_start_seconds
        and candidate.min_price <= point.price <= candidate.max_price
    ]
    return eligible[0] if eligible else None


def _exit_price(
    candidate: FirstpassCandidate,
    prices: Sequence[PricePoint],
    entry_ts: int,
    observation: Polymarket5mObservation,
) -> float | None:
    if candidate.exit_mode == "settlement":
        return 1.0 if observation.settlement_direction == observation.btc_direction else 0.0
    target_ts = observation.end_ts - 30
    eligible = [
        point for point in prices if entry_ts < point.timestamp <= target_ts and point.price > 0.0
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda point: point.timestamp).price


def _fold_excess(trades: Sequence[FirstpassTrade], fold_count: int) -> tuple[float, ...]:
    if not trades:
        return ()
    folds = min(fold_count, len(trades))
    fold_size = max(1, len(trades) // folds)
    values: list[float] = []
    for start in range(0, len(trades), fold_size):
        stop = min(len(trades), start + fold_size)
        values.append(statistics.fmean(trade.net_return for trade in trades[start:stop]))
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
            "reason": "PBO requires at least four fold observations",
            "pbo": 1.0,
            "trial_count": len(trials),
            "observation_count": min_len,
        }
    aligned = [tuple(trial[:min_len]) for trial in trials]
    report = dict(pbo(aligned, n_splits=effective_splits))
    report["valid"] = True
    return report


def _benchmarks(
    observations: Sequence[Polymarket5mObservation],
    best: CandidateResult,
) -> Mapping[str, object]:
    random_win_rate = 0.5
    no_impulse_trades = _no_impulse_trades(best.candidate, observations)
    return {
        "no_trade": {"mean_return": 0.0},
        "random_direction": {"expected_win_rate": random_win_rate, "mean_return": 0.0},
        "no_impulse_filter": {
            "trades": len(no_impulse_trades),
            "mean_return": _mean_return(no_impulse_trades),
            "note": "Same price/window/exit candidate without the BTC move threshold.",
        },
    }


def _no_impulse_trades(
    candidate: FirstpassCandidate,
    observations: Sequence[Polymarket5mObservation],
) -> tuple[FirstpassTrade, ...]:
    relaxed = FirstpassCandidate(
        name=f"{candidate.name}_no_impulse",
        move_threshold_usd=0.0,
        window_start_seconds=candidate.window_start_seconds,
        window_end_seconds=candidate.window_end_seconds,
        min_price=candidate.min_price,
        max_price=candidate.max_price,
        exit_mode=candidate.exit_mode,
    )
    return tuple(
        trade
        for observation in sorted(observations, key=lambda row: row.end_ts)
        if (trade := _trade_for_observation(relaxed, observation)) is not None
    )


def _metrics(trades: Sequence[FirstpassTrade]) -> Mapping[str, float | int]:
    returns = [trade.net_return for trade in trades]
    wins = [value for value in returns if value > 0.0]
    losses = [value for value in returns if value < 0.0]
    return {
        "trades": len(trades),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "total_return_sum": sum(returns),
        "win_rate": len(wins) / len(returns) if returns else 0.0,
        "average_win": statistics.fmean(wins) if wins else 0.0,
        "average_loss": statistics.fmean(losses) if losses else 0.0,
        "max_drawdown_sum": _max_drawdown(returns),
    }


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def _threshold_entry_counts(results: Sequence[CandidateResult]) -> dict[str, int]:
    counts: dict[str, int] = {f"{threshold:g}": 0 for threshold in MOVE_THRESHOLDS}
    for result in results:
        counts[f"{result.candidate.move_threshold_usd:g}"] += len(result.trades)
    return dict(sorted(counts.items()))


def _date_range(observations: Sequence[Polymarket5mObservation]) -> Mapping[str, int | None]:
    if not observations:
        return {"start_ts": None, "end_ts": None}
    return {
        "start_ts": min(row.start_ts for row in observations),
        "end_ts": max(row.end_ts for row in observations),
    }


def _mean_return(trades: Sequence[FirstpassTrade]) -> float:
    return statistics.fmean(trade.net_return for trade in trades) if trades else 0.0


def _number_from_mapping(raw: Mapping[str, object], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _observation_from_mapping(raw: Mapping[str, object]) -> Polymarket5mObservation:
    settlement_direction = _direction(raw["settlement_direction"])
    btc_direction = _direction(raw["btc_direction"])
    return Polymarket5mObservation(
        condition_id=_required_str(raw, "condition_id"),
        slug=_required_str(raw, "slug"),
        title=_required_str(raw, "title"),
        start_ts=_required_int(raw, "start_ts"),
        end_ts=_required_int(raw, "end_ts"),
        settlement_direction=settlement_direction,
        btc_move_usd=_required_float(raw, "btc_move_usd"),
        btc_direction=btc_direction,
        up_prices=_price_points(raw.get("up_prices", ())),
        down_prices=_price_points(raw.get("down_prices", ())),
    )


def _price_points(raw: object) -> tuple[PricePoint, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("price history must be a list")
    points: list[PricePoint] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("price point must be an object")
        timestamp = _required_int(item, "timestamp")
        price = _required_float(item, "price")
        if timestamp > 0 and 0.0 <= price <= 1.0:
            points.append(PricePoint(timestamp=timestamp, price=price))
    return tuple(sorted(points, key=lambda point: point.timestamp))


def _direction(value: object) -> Direction:
    text = str(value).strip().lower()
    if text == "up":
        return "Up"
    if text == "down":
        return "Down"
    raise ValueError("direction must be Up or Down")


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _candidate_to_dict(candidate: FirstpassCandidate) -> dict[str, float | int | str]:
    return {
        "name": candidate.name,
        "move_threshold_usd": candidate.move_threshold_usd,
        "window_start_seconds": candidate.window_start_seconds,
        "window_end_seconds": candidate.window_end_seconds,
        "min_price": candidate.min_price,
        "max_price": candidate.max_price,
        "exit_mode": candidate.exit_mode,
    }


def _trade_to_dict(trade: FirstpassTrade) -> dict[str, float | int | str | bool]:
    return {
        "condition_id": trade.condition_id,
        "slug": trade.slug,
        "direction": trade.direction,
        "decision_timestamp": trade.decision_timestamp,
        "seconds_to_close": trade.seconds_to_close,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "settlement_direction": trade.settlement_direction,
        "btc_move_usd": trade.btc_move_usd,
        "net_return": trade.net_return,
        "outcome_correct": trade.outcome_correct,
    }


def _insufficient(
    reason: str,
    *,
    market_count: int = 0,
    candidate_count: int = len(_candidate_grid()),
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "strategy": "polymarket_btc_5m_firstpass_optimistic",
        "candidate_count_n": candidate_count,
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": candidate_count,
            "fdr_after": 0,
            "pbo_after_survivors": 0,
        },
        "coverage": {"market_count": market_count, "entry_count": 0},
        "optimistic_boundary": {
            "optimistic_only": True,
            "positive_verdict_ceiling": "SUGGESTIVE_NEEDS_EXECUTION_VALIDATION",
            "unmodeled_execution_costs": UNMODELED_EXECUTION_COSTS,
            "robust_or_edge_claim_allowed": False,
        },
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }
