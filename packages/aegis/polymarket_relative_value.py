from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aegis.backtest_core import benjamini_hochberg
from aegis.polymarket_forward_execution import Direction, ForwardMarket, parse_forward_markets

Verdict = Literal["SUGGESTIVE_NEEDS_PAID_CONFIRM", "NO_EDGE", "INSUFFICIENT"]

TIME_TO_CLOSE_SECONDS = (30, 60, 120)
PROBABILITY_BINS = ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95))
TRIAL_COUNT_N = len(TIME_TO_CLOSE_SECONDS) * len(PROBABILITY_BINS)


@dataclass(frozen=True)
class RelativeValueConfig:
    time_to_close_seconds: tuple[int, ...] = TIME_TO_CLOSE_SECONDS
    probability_bins: tuple[tuple[float, float], ...] = PROBABILITY_BINS
    target_tolerance_seconds: float = 20.0
    min_markets: int = 100
    min_observations: int = 50
    min_bucket_observations: int = 5
    fdr_alpha: float = 0.10
    survivor_light: bool = True


@dataclass(frozen=True)
class CalibrationObservation:
    slug: str
    target_seconds: int
    snapshot_timestamp_ms: int
    seconds_to_close: float
    favorite: Direction
    implied_probability: float
    favorite_bid: float | None
    favorite_ask: float | None
    favorite_mid: float
    underdog_bid: float | None
    underdog_ask: float | None
    underdog_mid: float | None
    favorite_won: bool


def run_relative_value_calibration(
    rows: Sequence[Mapping[str, object]],
    *,
    config: RelativeValueConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = RelativeValueConfig()
    markets = parse_forward_markets(rows)
    coverage = _coverage(markets)
    if not markets:
        return _insufficient("no forward Polymarket rows", coverage, config)
    if int(coverage["settled_markets"]) < config.min_markets:
        return _insufficient(
            f"settled forward markets {coverage['settled_markets']} < min_markets "
            f"{config.min_markets}",
            coverage,
            config,
        )
    observations = _calibration_observations(markets, config)
    if len(observations) < config.min_observations:
        return _insufficient(
            f"calibration observations {len(observations)} < min_observations "
            f"{config.min_observations}",
            {**coverage, "calibration_observations": len(observations)},
            config,
        )
    bucket_rows = _bucket_rows(observations, config)
    p_values = [_float(row["p_value"], default=1.0) for row in bucket_rows]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha, tie_policy="rank")
    fdr_discoveries = [
        {**row, "fdr_miscalibrated": flag}
        for row, flag in zip(bucket_rows, fdr_flags, strict=True)
    ]
    raw_miscalibrated = [
        row
        for row in fdr_discoveries
        if _int(row["n"]) >= config.min_bucket_observations and not bool(row["calibrated"])
    ]
    fdr_miscalibrated = [
        row
        for row in fdr_discoveries
        if _int(row["n"]) >= config.min_bucket_observations and bool(row["fdr_miscalibrated"])
    ]
    exploitable = [
        row
        for row in fdr_miscalibrated
        if _float(row.get("max_edge_after_ask"), default=-1.0) > 0.0
    ]
    if not fdr_miscalibrated:
        verdict: Verdict = "NO_EDGE"
        reason = (
            "Polymarket favorite implied probabilities are not significantly "
            "miscalibrated after BH-FDR across predeclared time/bucket tests"
        )
    elif exploitable:
        verdict = "SUGGESTIVE_NEEDS_PAID_CONFIRM"
        reason = (
            "one or more calibration buckets are FDR-miscalibrated and remain above "
            "the relevant side's ask; survivor-light cap applies"
        )
    else:
        verdict = "NO_EDGE"
        reason = (
            "calibration deviations exist before FDR or before spread, but no "
            "predeclared bucket shows exploitable post-ask edge"
        )
    return {
        "status": "OK",
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "longer multi-regime forward capture with confirmed full market enumeration "
            "and non-survivor-limited coverage"
        ),
        "candidate_count_n": TRIAL_COUNT_N,
        "raw_is_survivors": len(raw_miscalibrated),
        "fdr_is_survivors": len(exploitable),
        "coverage": {
            **coverage,
            "calibration_observations": len(observations),
            "targets": list(config.time_to_close_seconds),
            "probability_bins": [list(value) for value in config.probability_bins],
        },
        "standard_metrics": {
            "mean_implied_probability": statistics.fmean(
                observation.implied_probability for observation in observations
            ),
            "actual_favorite_win_rate": _win_rate(observations),
            "mean_favorite_spread": _mean_spread(observations),
            "max_positive_edge_after_ask": max(
                (_float(row.get("max_edge_after_ask"), default=0.0) for row in bucket_rows),
                default=0.0,
            ),
        },
        "benchmark_metrics": {
            "benchmark": "well_calibrated_market",
            "expected_edge_after_spread": 0.0,
        },
        "multiple_testing": {
            "method": "binomial calibration tests + BH-FDR",
            "candidate_count_n": TRIAL_COUNT_N,
            "fdr_alpha": config.fdr_alpha,
            "fdr_after": len(fdr_miscalibrated),
            "exploitable_after_ask": len(exploitable),
            "tested_buckets": len(
                [
                    row
                    for row in bucket_rows
                    if _int(row["n"]) >= config.min_bucket_observations
                ]
            ),
        },
        "calibration_buckets": fdr_discoveries,
        "deviation_test": {
            "status": "skipped" if not fdr_miscalibrated else "evaluated_from_calibration",
            "reason": (
                "calibration kill-switch found no FDR miscalibration"
                if not fdr_miscalibrated
                else "post-ask upper bound reported per bucket"
            ),
        },
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "execution_or_fill_modeling": False,
            "settlement_only_for_labels": True,
            "survivor_light_ceiling_required": config.survivor_light,
        },
    }


def _calibration_observations(
    markets: Sequence[ForwardMarket], config: RelativeValueConfig
) -> tuple[CalibrationObservation, ...]:
    observations: list[CalibrationObservation] = []
    for market in markets:
        if market.settlement_direction is None:
            continue
        for target in config.time_to_close_seconds:
            up = _nearest_snapshot(market, "Up", target, config.target_tolerance_seconds)
            down = _nearest_snapshot(market, "Down", target, config.target_tolerance_seconds)
            if up is None and down is None:
                continue
            up_mid = _implied_mid(up)
            down_mid = _implied_mid(down)
            if up_mid is None and down_mid is None:
                continue
            if up_mid is None:
                if down_mid is None:
                    continue
                up_probability = 1.0 - down_mid
            elif down_mid is None:
                up_probability = up_mid
            else:
                total = up_mid + down_mid
                if total <= 0.0:
                    continue
                up_probability = up_mid / total
            up_probability = min(max(up_probability, 0.0), 1.0)
            favorite: Direction = "Up" if up_probability >= 0.5 else "Down"
            favorite_probability = up_probability if favorite == "Up" else 1.0 - up_probability
            favorite_snapshot = up if favorite == "Up" else down
            underdog_snapshot = down if favorite == "Up" else up
            timestamp = (
                favorite_snapshot.timestamp_ms
                if favorite_snapshot is not None
                else (up or down).timestamp_ms  # type: ignore[union-attr]
            )
            seconds_to_close = (
                favorite_snapshot.seconds_to_close
                if favorite_snapshot is not None
                else (up or down).seconds_to_close  # type: ignore[union-attr]
            )
            favorite_bid = _bid(favorite_snapshot)
            favorite_ask = _ask(favorite_snapshot)
            favorite_mid = _implied_mid(favorite_snapshot)
            if favorite_mid is None:
                favorite_mid = favorite_probability
            underdog_mid = _implied_mid(underdog_snapshot)
            observations.append(
                CalibrationObservation(
                    slug=market.slug,
                    target_seconds=target,
                    snapshot_timestamp_ms=timestamp,
                    seconds_to_close=seconds_to_close,
                    favorite=favorite,
                    implied_probability=favorite_probability,
                    favorite_bid=favorite_bid,
                    favorite_ask=favorite_ask,
                    favorite_mid=favorite_mid,
                    underdog_bid=_bid(underdog_snapshot),
                    underdog_ask=_ask(underdog_snapshot),
                    underdog_mid=underdog_mid,
                    favorite_won=market.settlement_direction == favorite,
                )
            )
    return tuple(observations)


def _bucket_rows(
    observations: Sequence[CalibrationObservation],
    config: RelativeValueConfig,
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for target in config.time_to_close_seconds:
        target_observations = [
            observation for observation in observations if observation.target_seconds == target
        ]
        for low, high in config.probability_bins:
            bucket = [
                observation
                for observation in target_observations
                if low <= observation.implied_probability < high
            ]
            n = len(bucket)
            wins = sum(1 for observation in bucket if observation.favorite_won)
            mean_implied = (
                statistics.fmean(observation.implied_probability for observation in bucket)
                if bucket
                else (low + high) / 2.0
            )
            mean_ask = _mean_available([observation.favorite_ask for observation in bucket])
            mean_bid = _mean_available([observation.favorite_bid for observation in bucket])
            mean_underdog_ask = _mean_available(
                [observation.underdog_ask for observation in bucket]
            )
            mean_underdog_bid = _mean_available(
                [observation.underdog_bid for observation in bucket]
            )
            actual = wins / n if n else 0.0
            underdog_actual = 1.0 - actual if n else 0.0
            favorite_edge_after_ask = (
                None if mean_ask is None or not n else actual - mean_ask
            )
            underdog_edge_after_ask = (
                None if mean_underdog_ask is None or not n else underdog_actual - mean_underdog_ask
            )
            max_edge_after_ask = (
                None
                if not n
                else max(
                    _float(favorite_edge_after_ask, default=-1.0),
                    _float(underdog_edge_after_ask, default=-1.0),
                )
            )
            ci_low, ci_high = _wilson_interval(wins, n) if n else (None, None)
            p_value = _binomial_two_sided_p_value(wins, n, mean_implied) if n else 1.0
            rows.append(
                {
                    "target_seconds": target,
                    "bucket": f"{low:.2f}-{high:.2f}",
                    "bucket_low": low,
                    "bucket_high": high,
                    "n": n,
                    "wins": wins,
                    "actual_win_rate": actual,
                    "mean_implied_probability": mean_implied,
                    "binomial_ci_low": ci_low,
                    "binomial_ci_high": ci_high,
                    "calibrated": (
                        True
                        if ci_low is None or ci_high is None
                        else ci_low <= mean_implied <= ci_high
                    ),
                    "p_value": p_value,
                    "mean_favorite_bid": mean_bid,
                    "mean_favorite_ask": mean_ask,
                    "mean_underdog_bid": mean_underdog_bid,
                    "mean_underdog_ask": mean_underdog_ask,
                    "mean_spread": (
                        None if mean_bid is None or mean_ask is None else mean_ask - mean_bid
                    ),
                    "edge_vs_mid": actual - mean_implied if n else 0.0,
                    "favorite_edge_after_ask": favorite_edge_after_ask,
                    "underdog_edge_after_ask": underdog_edge_after_ask,
                    "max_edge_after_ask": max_edge_after_ask,
                }
            )
    return rows


def _nearest_snapshot(
    market: ForwardMarket,
    outcome: Direction,
    target_seconds: int,
    tolerance_seconds: float,
) -> Any | None:
    candidates = [
        snapshot
        for snapshot in market.snapshots
        if snapshot.outcome == outcome
        and abs(snapshot.seconds_to_close - target_seconds) <= tolerance_seconds
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda snapshot: abs(snapshot.seconds_to_close - target_seconds))


def _implied_mid(snapshot: Any | None) -> float | None:
    if snapshot is None:
        return None
    bid = _bid(snapshot)
    ask = _ask(snapshot)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    return ask


def _bid(snapshot: Any | None) -> float | None:
    if snapshot is None:
        return None
    value = snapshot.best_bid
    return value if isinstance(value, float) and math.isfinite(value) else None


def _ask(snapshot: Any | None) -> float | None:
    if snapshot is None:
        return None
    value = snapshot.best_ask
    return value if isinstance(value, float) and math.isfinite(value) else None


def _mean_available(values: Sequence[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def _win_rate(observations: Sequence[CalibrationObservation]) -> float:
    return (
        sum(1 for observation in observations if observation.favorite_won) / len(observations)
        if observations
        else 0.0
    )


def _mean_spread(observations: Sequence[CalibrationObservation]) -> float | None:
    spreads = [
        observation.favorite_ask - observation.favorite_bid
        for observation in observations
        if observation.favorite_ask is not None and observation.favorite_bid is not None
    ]
    return statistics.fmean(spreads) if spreads else None


def _wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _binomial_two_sided_p_value(wins: int, n: int, probability: float) -> float:
    if n <= 0:
        return 1.0
    p = min(max(probability, 1e-9), 1.0 - 1e-9)
    observed = _binomial_pmf(wins, n, p)
    total = 0.0
    for k in range(n + 1):
        value = _binomial_pmf(k, n, p)
        if value <= observed + 1e-15:
            total += value
    return min(1.0, total)


def _binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def _coverage(markets: Sequence[ForwardMarket]) -> Mapping[str, int]:
    settled = [market for market in markets if market.settlement_direction is not None]
    return {
        "markets": len(markets),
        "settled_markets": len(settled),
        "snapshots": sum(len(market.snapshots) for market in markets),
    }


def _insufficient(
    reason: str, coverage: Mapping[str, object], config: RelativeValueConfig
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": TRIAL_COUNT_N,
        "coverage": dict(coverage),
        "standard_metrics": {},
        "benchmark_metrics": {"benchmark": "well_calibrated_market"},
        "multiple_testing": {
            "method": "binomial calibration tests + BH-FDR",
            "candidate_count_n": TRIAL_COUNT_N,
            "fdr_after": 0,
        },
        "calibration_buckets": [],
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "execution_or_fill_modeling": False,
            "settlement_only_for_labels": True,
            "survivor_light_ceiling_required": config.survivor_light,
        },
    }


def _float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return default


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0
