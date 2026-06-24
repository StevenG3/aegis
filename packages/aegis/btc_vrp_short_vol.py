from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from aegis.backtest_core import benjamini_hochberg, pbo, sign_test_p_value

ShortVolState = Literal["EDGE", "NO_EDGE", "INSUFFICIENT"]
ShortVolVerdict = Literal[
    "SUGGESTIVE_SHORT_VOL_EDGE",
    "NO_EDGE",
    "PREMIUM_EXISTS_BUT_TAIL_UNSAFE",
    "INSUFFICIENT",
]


@dataclass(frozen=True)
class ShortVolVrpConfig:
    max_drawdown_limit: float = -0.30
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    survivor_light: bool = True


@dataclass(frozen=True)
class ShortVolObservation:
    variant: str
    iv_ts: int
    expiry_ts: int
    implied_vol: float
    realized_vol: float
    variance_year_fraction: float
    option_spread_cost: float
    hedge_fee_cost: float
    hedge_slippage_cost: float
    funding_cost: float
    tail_loss: float

    @property
    def net_return(self) -> float:
        variance_premium = (
            (self.implied_vol**2 - self.realized_vol**2) * self.variance_year_fraction
        )
        costs = (
            self.option_spread_cost
            + self.hedge_fee_cost
            + self.hedge_slippage_cost
            + self.funding_cost
            + self.tail_loss
        )
        return variance_premium - costs


def run_btc_short_vol_vrp(
    rows: Sequence[Mapping[str, object]],
    *,
    config: ShortVolVrpConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = ShortVolVrpConfig()
    observations, excluded = _observations(rows)
    variants = sorted({row.variant for row in observations})
    coverage = {
        "input_rows": len(rows),
        "observations": len(observations),
        "excluded_rows": len(excluded),
        "excluded_reasons": _reason_counts(excluded),
        "variants": variants,
    }
    if not observations:
        return _insufficient("no valid short-vol VRP observations", coverage, config)
    evaluations = [_evaluate_variant(variant, observations) for variant in variants]
    p_values = [float(evaluation["p_value"]) for evaluation in evaluations]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(evaluations, observations, config)
    pbo_valid = bool(pbo_report.get("valid", False))
    pbo_value = _float(pbo_report.get("pbo"), default=1.0)
    best = max(evaluations, key=lambda evaluation: float(evaluation["mean_net_return"]))
    best_observations = [row for row in observations if row.variant == best["variant"]]
    best_metrics = _tail_metrics(best_observations)
    fdr_pass = any(
        flag and evaluation["variant"] == best["variant"]
        for flag, evaluation in zip(fdr_flags, evaluations, strict=True)
    )
    premium_exists = float(best["mean_net_return"]) > 0.0
    max_drawdown = _float(best_metrics["max_drawdown"], default=-math.inf)
    tail_safe = max_drawdown >= config.max_drawdown_limit
    if premium_exists and not tail_safe:
        state: ShortVolState = "NO_EDGE"
        verdict: ShortVolVerdict = "PREMIUM_EXISTS_BUT_TAIL_UNSAFE"
        reason = (
            "short-vol premium is positive in sample but max drawdown breaches the "
            "predeclared survivability limit"
        )
    elif premium_exists and fdr_pass and pbo_valid and pbo_value <= 0.20 and tail_safe:
        state = "EDGE"
        verdict = "SUGGESTIVE_SHORT_VOL_EDGE"
        reason = "short-vol VRP survived net-EV, FDR, PBO, and tail survivability gates"
    else:
        state = "NO_EDGE"
        verdict = "NO_EDGE"
        reason = "no predeclared short-vol variant passed net-EV, FDR, PBO, and tail gates"
    return {
        "status": "OK",
        "state": state,
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "full PIT option chain bid/ask IV including 2020-03 and 2022 crash regimes "
            "plus hedge funding history"
        ),
        "candidate_count_n": len(evaluations),
        "coverage": coverage,
        "standard_metrics": {
            **best_metrics,
            "mean_net_return": best["mean_net_return"],
            "total_net_return": best["total_net_return"],
            "positive_period_win_rate": best["win_rate"],
            "variant": best["variant"],
        },
        "benchmark_metrics": {"cash": 0.0},
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": len(evaluations),
            "fdr_after": sum(1 for flag in fdr_flags if flag),
            "min_p": min(p_values) if p_values else None,
            "pbo": pbo_report,
        },
        "best_candidate": {key: value for key, value in best.items() if key != "returns"},
        "safety": _safety(config),
    }


def btc_vrp_data_blocked_report(
    *,
    reason: str,
    coverage: Mapping[str, object],
    config: ShortVolVrpConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = ShortVolVrpConfig()
    return _insufficient(reason, coverage, config)


def cast_returns(values: object) -> tuple[float, ...]:
    if not isinstance(values, Sequence):
        return ()
    clean: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            continue
        clean.append(float(value))
    return tuple(clean)


def _observations(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[ShortVolObservation], list[tuple[str, str]]]:
    observations: list[ShortVolObservation] = []
    excluded: list[tuple[str, str]] = []
    for row in rows:
        reason = _row_exclusion_reason(row)
        if reason is not None:
            excluded.append((_str(row.get("variant")) or "", reason))
            continue
        observations.append(
            ShortVolObservation(
                variant=str(row["variant"]),
                iv_ts=_required_int(row["iv_ts"]),
                expiry_ts=_required_int(row["expiry_ts"]),
                implied_vol=_required_float(row["implied_vol"]),
                realized_vol=_required_float(row["realized_vol"]),
                variance_year_fraction=_required_float(row["variance_year_fraction"]),
                option_spread_cost=_required_float(row["option_spread_cost"]),
                hedge_fee_cost=_required_float(row["hedge_fee_cost"]),
                hedge_slippage_cost=_required_float(row["hedge_slippage_cost"]),
                funding_cost=_required_float(row["funding_cost"]),
                tail_loss=_required_float(row["tail_loss"]),
            )
        )
    return observations, excluded


def _row_exclusion_reason(row: Mapping[str, object]) -> str | None:
    required = (
        "variant",
        "iv_ts",
        "expiry_ts",
        "implied_vol",
        "realized_vol",
        "variance_year_fraction",
        "option_spread_cost",
        "hedge_fee_cost",
        "hedge_slippage_cost",
        "funding_cost",
        "tail_loss",
    )
    for key in required:
        if key not in row:
            return f"missing_{key}"
    iv_ts = _optional_int(row["iv_ts"])
    expiry_ts = _optional_int(row["expiry_ts"])
    iv = _optional_float(row["implied_vol"])
    rv = _optional_float(row["realized_vol"])
    year_fraction = _optional_float(row["variance_year_fraction"])
    costs = (
        _optional_float(row["option_spread_cost"]),
        _optional_float(row["hedge_fee_cost"]),
        _optional_float(row["hedge_slippage_cost"]),
        _optional_float(row["funding_cost"]),
        _optional_float(row["tail_loss"]),
    )
    if (
        iv_ts is None
        or expiry_ts is None
        or iv is None
        or rv is None
        or year_fraction is None
        or any(c is None for c in costs)
    ):
        return "parse_error"
    if iv_ts >= expiry_ts:
        return "iv_timestamp_not_before_expiry"
    if iv < 0.0 or rv < 0.0:
        return "negative_volatility"
    if year_fraction <= 0.0:
        return "non_positive_year_fraction"
    if any(c < 0.0 for c in costs if c is not None):
        return "negative_cost"
    return None


def _evaluate_variant(
    variant: str,
    observations: Sequence[ShortVolObservation],
) -> Mapping[str, Any]:
    returns = tuple(row.net_return for row in observations if row.variant == variant)
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in returns]
    wins = sum(1 for value in returns if value > 0)
    return {
        "variant": variant,
        "trade_count": len(returns),
        "wins": wins,
        "losses": len(returns) - wins,
        "win_rate": wins / len(returns) if returns else 0.0,
        "mean_net_return": statistics.fmean(returns) if returns else 0.0,
        "total_net_return": sum(returns),
        "p_value": sign_test_p_value(signs) if returns else 1.0,
        "returns": returns,
    }


def _pbo_report(
    evaluations: Sequence[Mapping[str, Any]],
    observations: Sequence[ShortVolObservation],
    config: ShortVolVrpConfig,
) -> Mapping[str, Any]:
    ordered_times = sorted({row.iv_ts for row in observations})
    trial_returns: list[list[float]] = []
    for evaluation in evaluations:
        variant = str(evaluation["variant"])
        by_time = {row.iv_ts: row.net_return for row in observations if row.variant == variant}
        trial_returns.append([by_time.get(ts, 0.0) for ts in ordered_times])
    if len(trial_returns) < 2 or len(ordered_times) < config.pbo_splits:
        return {
            "method": "CSCV_PBO",
            "valid": False,
            "reason": "insufficient trials or observations for PBO",
            "trial_count": len(trial_returns),
            "observation_count": len(ordered_times),
            "n_splits": config.pbo_splits,
        }
    report = pbo(trial_returns, n_splits=config.pbo_splits)
    return {**report, "valid": True}


def _tail_metrics(observations: Sequence[ShortVolObservation]) -> Mapping[str, float | int | None]:
    returns = tuple(row.net_return for row in observations)
    if not returns:
        return {
            "trades": 0,
            "max_drawdown": None,
            "cvar_95": None,
            "cvar_99": None,
            "worst_trade": None,
            "sortino": None,
            "calmar": None,
            "return_to_cvar_95": None,
        }
    total = sum(returns)
    max_dd = _max_drawdown(returns)
    cvar95 = _cvar(returns, 0.95)
    cvar99 = _cvar(returns, 0.99)
    downside = [value for value in returns if value < 0.0]
    downside_stdev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    mean = statistics.fmean(returns)
    sortino = mean / downside_stdev if downside_stdev > 0.0 else None
    calmar = total / abs(max_dd) if max_dd < 0.0 else None
    return {
        "trades": len(returns),
        "max_drawdown": max_dd,
        "cvar_95": cvar95,
        "cvar_99": cvar99,
        "worst_trade": min(returns),
        "worst_month": _worst_month(observations),
        "sortino": sortino,
        "calmar": calmar,
        "return_to_cvar_95": total / abs(cvar95) if cvar95 < 0.0 else None,
    }


def _worst_month(observations: Sequence[ShortVolObservation]) -> float | None:
    if not observations:
        return None
    by_month: dict[str, float] = {}
    for row in observations:
        month = datetime.fromtimestamp(row.iv_ts / 1000, UTC).strftime("%Y-%m")
        by_month[month] = by_month.get(month, 0.0) + row.net_return
    return min(by_month.values()) if by_month else None


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0.0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _cvar(returns: Sequence[float], confidence: float) -> float:
    sorted_returns = sorted(returns)
    tail_count = max(1, math.ceil(len(sorted_returns) * (1.0 - confidence)))
    return statistics.fmean(sorted_returns[:tail_count])


def _insufficient(
    reason: str,
    coverage: Mapping[str, object],
    config: ShortVolVrpConfig,
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "state": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": 0,
        "coverage": dict(coverage),
        "standard_metrics": {},
        "benchmark_metrics": {"cash": 0.0},
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": 0,
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": "not run under INSUFFICIENT gate"},
        },
        "safety": _safety(config),
    }


def _safety(config: ShortVolVrpConfig) -> Mapping[str, object]:
    return {
        "read_only": True,
        "wallet_or_order_access": False,
        "live_trading": False,
        "account_access": False,
        "max_drawdown_limit": config.max_drawdown_limit,
    }


def _reason_counts(excluded: Sequence[tuple[str, str]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for _, reason in excluded:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else default


def _required_float(value: object) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"expected finite float-compatible value, got {value!r}")
    return parsed


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"expected int-compatible value, got {value!r}")
    return parsed
