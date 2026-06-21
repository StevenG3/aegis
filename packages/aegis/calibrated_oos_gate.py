from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aegis.backtest_core import deflated_sharpe_threshold, sign_test_p_value

OosGateVerdict = Literal["ROBUST_OOS_EDGE", "NO_ROBUST_EDGE", "OOS_DATA_INSUFFICIENT"]


@dataclass(frozen=True)
class OosWindow:
    candidate_score: float
    static_score: float
    buy_hold_score: float
    candidate_return: float
    static_return: float
    buy_hold_return: float


@dataclass(frozen=True)
class CalibratedOosGateConfig:
    min_windows: int = 3
    alpha: float = 0.05
    bootstrap_iterations: int = 1_000
    bootstrap_seed: int = 39
    trial_count: int = 1
    annualization_periods: int = 365
    min_deflated_sharpe: float = 0.0


DEFAULT_OOS_GATE_CONFIG = CalibratedOosGateConfig()


def evaluate_calibrated_oos_gate(
    windows: Sequence[OosWindow],
    candidate_returns: Sequence[float],
    *,
    config: CalibratedOosGateConfig = DEFAULT_OOS_GATE_CONFIG,
) -> dict[str, Any]:
    if len(windows) < config.min_windows:
        return {
            "verdict": "OOS_DATA_INSUFFICIENT",
            "reason": f"windows {len(windows)} < min_windows {config.min_windows}",
            "windows": len(windows),
            "config": _config_dict(config),
            "checks": [],
        }

    excess_static = tuple(window.candidate_score - window.static_score for window in windows)
    excess_buy_hold = tuple(
        window.candidate_score - window.buy_hold_score for window in windows
    )
    static_test = _paired_excess_report(excess_static, config=config)
    buy_hold_test = _paired_excess_report(excess_buy_hold, config=config)
    sharpe_report = _deflated_sharpe_report(candidate_returns, config=config)
    checks = [
        _check(
            "static_sign_test",
            float(static_test["sign_test_p_value"]) < config.alpha,
            f"candidate risk-adjusted OOS score vs static RSI sign p >= {config.alpha}",
        ),
        _check(
            "static_signed_rank",
            float(static_test["signed_rank_p_value"]) < config.alpha,
            f"candidate risk-adjusted OOS score vs static RSI signed-rank p >= {config.alpha}",
        ),
        _check(
            "static_mean_ci",
            _ci_lower(static_test, "mean_ci") > 0.0,
            "candidate risk-adjusted OOS score vs static RSI mean CI lower <= 0",
        ),
        _check(
            "static_median_ci",
            _ci_lower(static_test, "median_ci") > 0.0,
            "candidate risk-adjusted OOS score vs static RSI median CI lower <= 0",
        ),
        _check(
            "buy_hold_sign_test",
            float(buy_hold_test["sign_test_p_value"]) < config.alpha,
            f"candidate risk-adjusted OOS score vs buy-and-hold sign p >= {config.alpha}",
        ),
        _check(
            "buy_hold_signed_rank",
            float(buy_hold_test["signed_rank_p_value"]) < config.alpha,
            f"candidate risk-adjusted OOS score vs buy-and-hold signed-rank p >= {config.alpha}",
        ),
        _check(
            "buy_hold_mean_ci",
            _ci_lower(buy_hold_test, "mean_ci") > 0.0,
            "candidate risk-adjusted OOS score vs buy-and-hold mean CI lower <= 0",
        ),
        _check(
            "buy_hold_median_ci",
            _ci_lower(buy_hold_test, "median_ci") > 0.0,
            "candidate risk-adjusted OOS score vs buy-and-hold median CI lower <= 0",
        ),
        _check(
            "deflated_sharpe",
            bool(sharpe_report["passed"]),
            "candidate OOS deflated Sharpe did not clear the predeclared threshold",
        ),
    ]
    failures = [str(check["reason"]) for check in checks if not bool(check["passed"])]
    verdict: OosGateVerdict = "ROBUST_OOS_EDGE" if not failures else "NO_ROBUST_EDGE"
    return {
        "verdict": verdict,
        "reason": "all calibrated OOS gates passed" if not failures else "; ".join(failures),
        "windows": len(windows),
        "config": _config_dict(config),
        "risk_adjusted_excess_vs_static": static_test,
        "risk_adjusted_excess_vs_buy_hold": buy_hold_test,
        "deflated_sharpe": sharpe_report,
        "checks": checks,
        "legacy_all_windows_note": (
            "The calibrated gate does not require every OOS window to beat both benchmarks; "
            "it evaluates paired risk-adjusted excess distributions instead."
        ),
        "raw_return_audit": {
            "median_candidate_return": _median(window.candidate_return for window in windows),
            "median_static_return": _median(window.static_return for window in windows),
            "median_buy_hold_return": _median(window.buy_hold_return for window in windows),
        },
    }


def annualization_periods_for_instrument(
    instrument_type: str,
    timeframe: str,
) -> int:
    normalized_instrument = instrument_type.lower().strip()
    normalized_timeframe = timeframe.lower().strip()
    if normalized_timeframe.endswith("h"):
        hours = int(normalized_timeframe[:-1])
        return int(24 / hours * 365)
    if normalized_timeframe.endswith("m"):
        minutes = int(normalized_timeframe[:-1])
        return int(24 * 60 / minutes * 365)
    if normalized_timeframe in {"1d", "d", "1day", "day"}:
        if normalized_instrument in {"crypto", "spot", "perp", "crypto_spot", "crypto_perp"}:
            return 365
        return 252
    if normalized_timeframe in {"1w", "w", "week"}:
        return 52
    return 365 if normalized_instrument in {"crypto", "spot", "perp"} else 252


def _paired_excess_report(
    values: Sequence[float],
    *,
    config: CalibratedOosGateConfig,
) -> dict[str, Any]:
    finite = tuple(float(value) for value in values if math.isfinite(value))
    if not finite:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "positive_share": 0.0,
            "sign_test_p_value": 1.0,
            "signed_rank_p_value": 1.0,
            "mean_ci": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
            "median_ci": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
        }
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "positive_share": sum(1 for value in finite if value > 0) / len(finite),
        "sign_test_p_value": sign_test_p_value(finite, alternative="greater"),
        "signed_rank_p_value": signed_rank_p_value_greater(finite),
        "mean_ci": _bootstrap_ci(finite, statistics.fmean, config=config),
        "median_ci": _bootstrap_ci(finite, statistics.median, config=config),
    }


def signed_rank_p_value_greater(values: Sequence[float]) -> float:
    non_zero = tuple(float(value) for value in values if value != 0.0 and math.isfinite(value))
    n = len(non_zero)
    if n == 0:
        return 1.0
    ranks = _absolute_ranks(non_zero)
    positive_rank_sum = sum(rank for rank, value in zip(ranks, non_zero, strict=True) if value > 0)
    total = n * (n + 1) / 2.0
    if n <= 20:
        count = 0
        favorable = 0
        scale = 1 << n
        for mask in range(scale):
            rank_sum = 0.0
            for index, rank in enumerate(ranks):
                if mask & (1 << index):
                    rank_sum += rank
            count += 1
            if rank_sum >= positive_rank_sum - 1e-12:
                favorable += 1
        return min(1.0, favorable / count)
    mean = total / 2.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    z_value = (positive_rank_sum - mean) / math.sqrt(variance)
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def _absolute_ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted((abs(value), index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for _value, index in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return tuple(ranks)


def _deflated_sharpe_report(
    returns: Sequence[float],
    *,
    config: CalibratedOosGateConfig,
) -> dict[str, float | int | bool]:
    values = tuple(float(value) for value in returns if math.isfinite(value))
    if len(values) < 2:
        return {
            "observations": len(values),
            "annualized_sharpe": 0.0,
            "threshold": config.min_deflated_sharpe,
            "passed": False,
        }
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    annualized = mean / stdev * math.sqrt(config.annualization_periods) if stdev > 0 else 0.0
    threshold = max(
        config.min_deflated_sharpe,
        deflated_sharpe_threshold(
            trial_count=max(config.trial_count, 1),
            observations=len(values),
            base_threshold=config.min_deflated_sharpe,
        ),
    )
    return {
        "observations": len(values),
        "annualized_sharpe": annualized,
        "threshold": threshold,
        "trial_count": config.trial_count,
        "annualization_periods": config.annualization_periods,
        "passed": annualized > threshold and annualized > 0.0,
    }


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Any,
    *,
    config: CalibratedOosGateConfig,
) -> dict[str, float | int]:
    rng = random.Random(config.bootstrap_seed)
    iterations = max(1, config.bootstrap_iterations)
    samples = sorted(
        float(statistic([rng.choice(values) for _ in values])) for _ in range(iterations)
    )
    return {
        "p05": samples[int(iterations * 0.05)],
        "p50": samples[int(iterations * 0.50)],
        "p95": samples[int(iterations * 0.95)],
        "iterations": iterations,
    }


def _check(identifier: str, passed: bool, reason: str) -> dict[str, str | bool]:
    return {"id": identifier, "passed": passed, "reason": "" if passed else reason}


def _ci_lower(payload: dict[str, Any], key: str) -> float:
    raw = payload.get(key)
    if isinstance(raw, dict):
        value = raw.get("p05")
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _median(values: Any) -> float:
    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    return statistics.median(finite) if finite else 0.0


def _config_dict(config: CalibratedOosGateConfig) -> dict[str, float | int]:
    return {
        "min_windows": config.min_windows,
        "alpha": config.alpha,
        "bootstrap_iterations": config.bootstrap_iterations,
        "bootstrap_seed": config.bootstrap_seed,
        "trial_count": config.trial_count,
        "annualization_periods": config.annualization_periods,
        "min_deflated_sharpe": config.min_deflated_sharpe,
    }
