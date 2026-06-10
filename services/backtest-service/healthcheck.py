from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict
from typing import Any, Literal, cast

Status = Literal["PASS", "WARN", "FAIL"]
CheckType = Literal["hard", "soft"]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "BEAT_BH_MIN": 0.5,
    "MAX_DD_LIMIT": 0.25,
    "EXIT_REASON_MIN": 0.99,
    "FEE_BPS": 10.0,
    "SLIPPAGE_BPS": 5.0,
    "FUNDING_BPS": 0.0,
    "FACTOR_ICIR_MIN": 0.3,
    "FACTOR_MONOTONIC_REQUIRED": 1.0,
    "WALK_FORWARD_MIN_WINDOWS": 2.0,
    "WALK_FORWARD_OOS_POSITIVE_SHARE_MIN": 0.5,
    "WALK_FORWARD_OOS_IS_RETURN_RATIO_MIN": 0.5,
    "WALK_FORWARD_OOS_IS_SHARPE_RATIO_MIN": 0.5,
}


def default_thresholds() -> dict[str, float]:
    values = dict(DEFAULT_THRESHOLDS)
    for key in values:
        raw = os.getenv(key)
        if raw is None or raw.strip() == "":
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            continue
    return values


def evaluate_strategy_health(
    runs: list[dict[str, Any]],
    *,
    edge_thesis: str,
    thresholds: dict[str, float] | None = None,
    cost_bps: float | None = None,
    factor_report: dict[str, Any] | None = None,
    walk_forward_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limits = {**default_thresholds(), **(thresholds or {})}
    round_trip_cost = _round_trip_cost(limits, cost_bps)
    checks = [
        _edge_check(edge_thesis),
        _factor_predictability_check(_factor_report(runs, factor_report), limits),
        _walk_forward_stability_check(
            _walk_forward_report(runs, walk_forward_report), limits
        ),
        _benchmark_check(runs, limits),
        _outlier_check(runs),
        _net_cost_check(runs, round_trip_cost),
        _sensitivity_check(runs),
        _drawdown_check(runs, limits),
        _attribution_check(runs, limits),
    ]
    verdict = _verdict(checks)
    return {
        "verdict": verdict,
        "checks": checks,
        "summary": {
            "runs": len(runs),
            "hard_failures": [
                check["id"]
                for check in checks
                if check["type"] == "hard" and check["status"] == "FAIL"
            ],
            "warnings": [
                check["id"]
                for check in checks
                if check["status"] == "WARN"
            ],
            "cost_model": {
                "round_trip_cost": round_trip_cost,
                "round_trip_cost_pct": round_trip_cost * 100,
                "note": (
                    "Estimated from bps inputs, not an engine-exact simulation. "
                    "net_return ~= gross_return - num_trades * round_trip_cost."
                ),
            },
        },
    }


def _edge_check(edge_thesis: str) -> dict[str, Any]:
    value = edge_thesis.strip()
    return _check(
        "edge_thesis",
        "Say the edge",
        "edge_thesis",
        bool(value),
        "non-empty",
        "PASS" if value else "FAIL",
        "hard",
    )


def _benchmark_check(runs: list[dict[str, Any]], limits: dict[str, float]) -> dict[str, Any]:
    comparisons: list[bool] = []
    for run in runs:
        strategy = _stat_float(run, "return_pct")
        benchmark = _first_stat_float(
            run,
            (
                "benchmark_return_pct",
                "risk_free_return_pct",
                "cash_return_pct",
                "buy_hold_return_pct",
            ),
        )
        if strategy is not None and benchmark is not None:
            comparisons.append(strategy > benchmark)
            continue
        sharpe = _stat_float(run, "sharpe")
        benchmark_sharpe = _stat_float(run, "benchmark_sharpe")
        if benchmark_sharpe is None:
            benchmark_sharpe = _stat_float(run, "buy_hold_sharpe")
        if sharpe is not None and benchmark_sharpe is not None:
            comparisons.append(sharpe > benchmark_sharpe)
    threshold = limits["BEAT_BH_MIN"]
    if not comparisons:
        return _insufficient(
            "beat_benchmark",
            "Beat benchmark",
            "beat_bh_share",
            threshold,
            "soft",
        )
    beat_share = sum(1 for value in comparisons if value) / len(comparisons)
    return _check(
        "beat_benchmark",
        "Beat benchmark",
        "beat_bh_share",
        round(beat_share, 6),
        threshold,
        "PASS" if beat_share >= threshold else "WARN",
        "soft",
    )


def _outlier_check(runs: list[dict[str, Any]]) -> dict[str, Any]:
    returns = _returns(runs)
    if not returns:
        return _insufficient(
            "not_one_big_winner",
            "Not dependent on one outlier",
            "median_return",
            "> 0 and top 10% trimmed sum > 0",
            "soft",
        )
    median_return = statistics.median(returns)
    trimmed = _trim_top_decile(returns)
    trimmed_sum = sum(trimmed)
    mean_return = statistics.fmean(returns)
    mean_median_ratio = None if median_return == 0 else mean_return / median_return
    return _check(
        "not_one_big_winner",
        "Not dependent on one outlier",
        "median_return",
        {
            "median_return": round(median_return, 6),
            "top10_trimmed_sum": round(trimmed_sum, 6),
            "mean_median_ratio": (
                None if mean_median_ratio is None else round(mean_median_ratio, 6)
            ),
        },
        "median_return > 0 and top10_trimmed_sum > 0",
        "PASS" if median_return > 0 and trimmed_sum > 0 else "WARN",
        "soft",
    )


def _net_cost_check(runs: list[dict[str, Any]], round_trip_cost: float) -> dict[str, Any]:
    values: list[float] = []
    for run in runs:
        net = _stat_float(run, "net_return_pct")
        if net is not None:
            values.append(net)
            continue
        gross = _stat_float(run, "return_pct")
        trades = _stat_float(run, "num_trades")
        if gross is None or trades is None:
            continue
        values.append(gross - trades * round_trip_cost)
    if not values:
        return _check(
            "net_cost_positive",
            "Positive after estimated costs",
            "net_median_return",
            "INSUFFICIENT_DATA",
            "> 0",
            "FAIL",
            "hard",
        )
    net_median = statistics.median(values)
    return _check(
        "net_cost_positive",
        "Positive after estimated costs",
        "net_median_return",
        round(net_median, 6),
        "> 0",
        "PASS" if net_median > 0 else "FAIL",
        "hard",
    )


def _sensitivity_check(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_params_key(run)].append(run)
    if len(grouped) <= 1:
        return _insufficient(
            "out_of_sample_stability",
            "Stable out of sample",
            "parameter_sensitivity",
            "perturbation groups supplied",
            "soft",
        )

    param_medians: dict[str, float] = {}
    period_medians: dict[str, float] = {}
    for key, group in grouped.items():
        returns = _returns(group)
        if returns:
            param_medians[key] = statistics.median(returns)
    periods = sorted({str(run.get("period", "")) for run in runs if run.get("period") is not None})
    for period in periods:
        returns = _returns([run for run in runs if str(run.get("period", "")) == period])
        if returns:
            period_medians[period] = statistics.median(returns)

    if not param_medians:
        return _insufficient(
            "out_of_sample_stability",
            "Stable out of sample",
            "parameter_sensitivity",
            "return_pct in perturbation groups",
            "soft",
        )

    medians_positive = all(value > 0 for value in param_medians.values())
    period_signs = {1 if value > 0 else -1 if value < 0 else 0 for value in period_medians.values()}
    sign_reversal = 1 in period_signs and -1 in period_signs
    return _check(
        "out_of_sample_stability",
        "Stable out of sample",
        "parameter_sensitivity",
        {
            "param_group_medians": {
                key: round(value, 6) for key, value in sorted(param_medians.items())
            },
            "period_medians": {
                key: round(value, 6) for key, value in sorted(period_medians.items())
            },
            "sign_reversal": sign_reversal,
        },
        "all parameter medians > 0 and no period sign reversal",
        "PASS" if medians_positive and not sign_reversal else "WARN",
        "soft",
    )


def _drawdown_check(runs: list[dict[str, Any]], limits: dict[str, float]) -> dict[str, Any]:
    drawdowns = [
        abs(value)
        for run in runs
        if (value := _stat_float(run, "max_drawdown_pct", normalize_percent=False)) is not None
    ]
    threshold = limits["MAX_DD_LIMIT"]
    if not drawdowns:
        return _check(
            "no_blowup",
            "No blow-up",
            "max_drawdown_pct",
            "INSUFFICIENT_DATA",
            f"< {threshold}",
            "FAIL",
            "hard",
        )
    max_drawdown = max(_to_fraction(value) for value in drawdowns)
    equity_touched_zero = any(_equity_touched_zero(run) for run in runs)
    return _check(
        "no_blowup",
        "No blow-up",
        "max_drawdown_pct",
        {"max_drawdown": round(max_drawdown, 6), "equity_touched_zero": equity_touched_zero},
        f"< {threshold}",
        "PASS" if max_drawdown < threshold and not equity_touched_zero else "FAIL",
        "hard",
    )


def _attribution_check(runs: list[dict[str, Any]], limits: dict[str, float]) -> dict[str, Any]:
    known = 0
    total = 0
    has_entry_basis = False
    for run in runs:
        breakdown = _stats(run).get("exit_breakdown")
        if isinstance(breakdown, dict):
            for reason, raw_count in breakdown.items():
                count = int(raw_count) if isinstance(raw_count, int | float) else 0
                total += count
                if reason != "unknown":
                    known += count
        for trade in _trades(run):
            has_basis = (
                trade.get("factors")
                or trade.get("rationale")
                or trade.get("entry_regime_up") is not None
            )
            if has_basis:
                has_entry_basis = True
    threshold = limits["EXIT_REASON_MIN"]
    if total == 0:
        return _insufficient(
            "attribution",
            "Attributable entries and exits",
            "known_exit_reason_share",
            threshold,
            "soft",
        )
    known_share = known / total
    return _check(
        "attribution",
        "Attributable entries and exits",
        "known_exit_reason_share",
        {"known_exit_reason_share": round(known_share, 6), "has_entry_basis": has_entry_basis},
        f">= {threshold} and entry basis present",
        "PASS" if known_share >= threshold and has_entry_basis else "WARN",
        "soft",
    )


def _factor_predictability_check(
    factor_report: dict[str, Any] | None,
    limits: dict[str, float],
) -> dict[str, Any]:
    threshold = limits["FACTOR_ICIR_MIN"]
    monotonic_required = limits["FACTOR_MONOTONIC_REQUIRED"] >= 1
    if factor_report is None:
        return _insufficient(
            "factor_predictability",
            "Factor predictive power",
            "rank_icir_and_monotonicity",
            f"|ICIR| >= {threshold} and monotonic groups",
            "soft",
        )
    raw_factors = factor_report.get("factors")
    if not isinstance(raw_factors, dict) or not raw_factors:
        return _insufficient(
            "factor_predictability",
            "Factor predictive power",
            "rank_icir_and_monotonicity",
            f"|ICIR| >= {threshold} and monotonic groups",
            "soft",
        )

    factor_rows: list[dict[str, Any]] = []
    predictive: list[str] = []
    insufficient = False
    for name, raw_factor in raw_factors.items():
        if not isinstance(raw_factor, dict):
            continue
        rank_ic = raw_factor.get("rank_ic")
        monotonicity = raw_factor.get("monotonicity")
        if not isinstance(rank_ic, dict) or not isinstance(monotonicity, dict):
            insufficient = True
            continue
        icir = _raw_float(rank_ic.get("icir"))
        monotonic = monotonicity.get("is_monotonic") is True
        has_power = icir is not None and abs(icir) >= threshold and (
            monotonic or not monotonic_required
        )
        factor_rows.append(
            {
                "factor": str(name),
                "rank_ic_mean": _raw_float(rank_ic.get("mean")),
                "rank_icir": icir,
                "t_value": _raw_float(rank_ic.get("t_value")),
                "ic_positive_share": _raw_float(rank_ic.get("positive_share")),
                "monotonic": monotonic,
                "top_bottom_return": _raw_float(monotonicity.get("top_bottom_return")),
                "has_predictive_power": has_power,
            }
        )
        if icir is None:
            insufficient = True
        if has_power:
            predictive.append(str(name))

    if predictive:
        return _check(
            "factor_predictability",
            "Factor predictive power",
            "rank_icir_and_monotonicity",
            {"predictive_factors": predictive, "factors": factor_rows},
            f"|ICIR| >= {threshold} and monotonic groups",
            "PASS",
            "soft",
        )
    reason = "INSUFFICIENT_DATA" if insufficient and not factor_rows else "NO_EDGE_SIGNAL"
    return _check(
        "factor_predictability",
        "Factor predictive power",
        "rank_icir_and_monotonicity",
        {"reason": reason, "factors": factor_rows},
        f"|ICIR| >= {threshold} and monotonic groups",
        "WARN",
        "soft",
    )


def _walk_forward_stability_check(
    walk_forward_report: dict[str, Any] | None,
    limits: dict[str, float],
) -> dict[str, Any]:
    min_windows = limits["WALK_FORWARD_MIN_WINDOWS"]
    min_positive_share = limits["WALK_FORWARD_OOS_POSITIVE_SHARE_MIN"]
    min_return_ratio = limits["WALK_FORWARD_OOS_IS_RETURN_RATIO_MIN"]
    min_sharpe_ratio = limits["WALK_FORWARD_OOS_IS_SHARPE_RATIO_MIN"]
    threshold = (
        f"windows >= {min_windows}, OOS positive share >= {min_positive_share}, "
        f"OOS/IS return >= {min_return_ratio}"
    )
    if walk_forward_report is None:
        return _insufficient(
            "walk_forward_oos_stability",
            "Walk-forward OOS stability",
            "oos_decay",
            threshold,
            "hard",
        )
    summary = walk_forward_report.get("summary")
    if not isinstance(summary, dict):
        return _insufficient(
            "walk_forward_oos_stability",
            "Walk-forward OOS stability",
            "oos_decay",
            threshold,
            "hard",
        )
    windows = _raw_float(summary.get("windows"))
    positive_share = _raw_float(summary.get("positive_oos_share"))
    median_oos_return = _raw_float(summary.get("median_oos_return_pct"))
    return_ratio = _raw_float(summary.get("median_oos_is_return_ratio"))
    sharpe_ratio = _raw_float(summary.get("median_oos_is_sharpe_ratio"))
    overfit = summary.get("overfit")
    overfit_flag = overfit.get("is_overfit") if isinstance(overfit, dict) else None
    value = {
        "windows": windows,
        "median_oos_return_pct": median_oos_return,
        "positive_oos_share": positive_share,
        "median_oos_is_return_ratio": return_ratio,
        "median_oos_is_sharpe_ratio": sharpe_ratio,
        "overfit": overfit,
    }
    if windows is None or windows < min_windows:
        return _check(
            "walk_forward_oos_stability",
            "Walk-forward OOS stability",
            "oos_decay",
            value | {"reason": "INSUFFICIENT_DATA"},
            threshold,
            "WARN",
            "hard",
        )
    failed = (
        median_oos_return is None
        or median_oos_return <= 0
        or positive_share is None
        or positive_share < min_positive_share
        or return_ratio is None
        or return_ratio < min_return_ratio
        or (
            sharpe_ratio is not None
            and sharpe_ratio < min_sharpe_ratio
        )
        or overfit_flag is True
    )
    return _check(
        "walk_forward_oos_stability",
        "Walk-forward OOS stability",
        "oos_decay",
        value,
        threshold,
        "FAIL" if failed else "PASS",
        "hard",
    )


def _verdict(checks: list[dict[str, Any]]) -> str:
    if any(check["type"] == "hard" and check["status"] == "FAIL" for check in checks):
        return "BLOCK"
    if any(check["status"] == "WARN" for check in checks):
        return "PASS_WITH_WARN"
    return "PASS"


def _round_trip_cost(limits: dict[str, float], cost_bps: float | None) -> float:
    if cost_bps is not None:
        return cost_bps / 10_000
    fee = limits["FEE_BPS"]
    slippage = limits["SLIPPAGE_BPS"]
    funding = limits["FUNDING_BPS"]
    return ((fee + slippage) * 2 + funding) / 10_000


def _stats(run: dict[str, Any]) -> dict[str, Any]:
    raw = run.get("stats", run)
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else {}


def _stat_float(
    run: dict[str, Any], key: str, *, normalize_percent: bool = True
) -> float | None:
    raw = _stats(run).get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if normalize_percent and key.endswith("_pct"):
        return value / 100
    return value


def _first_stat_float(run: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _stat_float(run, key)
        if value is not None:
            return value
    return None


def _returns(runs: list[dict[str, Any]]) -> list[float]:
    return [
        value
        for run in runs
        if (value := _stat_float(run, "return_pct")) is not None
    ]


def _trades(run: dict[str, Any]) -> list[dict[str, Any]]:
    raw = run.get("trades", [])
    if not isinstance(raw, list):
        return []
    return [cast(dict[str, Any], item) for item in raw if isinstance(item, dict)]


def _factor_report(
    runs: list[dict[str, Any]],
    factor_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if factor_report is not None:
        return factor_report
    for run in runs:
        raw = run.get("factor_report")
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
    return None


def _walk_forward_report(
    runs: list[dict[str, Any]],
    walk_forward_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if walk_forward_report is not None:
        return walk_forward_report
    for run in runs:
        raw = run.get("walk_forward_report")
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
    return None


def _raw_float(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _trim_top_decile(values: list[float]) -> list[float]:
    if len(values) < 10:
        return list(values)
    ordered = sorted(values)
    trim_count = max(int(len(ordered) * 0.1), 1)
    return ordered[:-trim_count]


def _to_fraction(value: float) -> float:
    return value / 100 if value > 1 else value


def _equity_touched_zero(run: dict[str, Any]) -> bool:
    raw = run.get("equity_curve", [])
    if not isinstance(raw, list):
        return False
    for point in raw:
        if not isinstance(point, dict):
            continue
        value = point.get("equity")
        try:
            if value is not None and float(value) <= 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _params_key(run: dict[str, Any]) -> str:
    params = run.get("params")
    if not isinstance(params, dict):
        request = run.get("request")
        params = request.get("params") if isinstance(request, dict) else None
    if not isinstance(params, dict):
        return "{}"
    return repr(sorted(params.items()))


def _insufficient(
    check_id: str,
    name: str,
    metric: str,
    threshold: object,
    check_type: CheckType,
) -> dict[str, Any]:
    return _check(
        check_id,
        name,
        metric,
        "INSUFFICIENT_DATA",
        threshold,
        "WARN",
        check_type,
    )


def _check(
    check_id: str,
    name: str,
    metric: str,
    value: object,
    threshold: object,
    status: Status,
    check_type: CheckType,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "status": status,
        "type": check_type,
    }
