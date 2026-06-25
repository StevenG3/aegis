"""EDGAR survivor-light full-universe fundamental IC diagnostics.

The runner consumes point-in-time, already-aligned observations. It does not fetch
SEC, price, broker, or live-trading data. Evidence scripts are responsible for
building observations from approved read-only sources and for keeping raw/private
outputs outside the public repository.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from aegis.backtest_core import CostModel, benjamini_hochberg, normal_two_sided_p, pbo

FACTOR_NAMES: tuple[str, ...] = (
    "earnings_yield_ep",
    "book_to_price_bp",
    "fcf_yield",
    "sales_to_price_sp",
    "roe",
    "gross_margin",
    "low_accruals",
    "asset_turnover",
)

HORIZONS: tuple[str, ...] = ("1m", "3m", "6m")
TRIAL_COUNT_N = len(FACTOR_NAMES) * len(HORIZONS)
SURVIVOR_LIGHT_UNLOCK_CONDITION = (
    "Sharadar/Norgate-grade PIT constituents, delisting-aware prices, and filing-date "
    "fundamentals are required to lift survivor-light limitations; configure a real "
    "AEGIS_SEC_USER_AGENT before any full free EDGAR rebuild."
)


@dataclass(frozen=True)
class EdgarIcObservation:
    symbol: str
    as_of: date
    available_on: date
    factors: Mapping[str, float]
    forward_returns: Mapping[str, float]
    in_universe: bool = True


@dataclass(frozen=True)
class EdgarIcConfig:
    min_symbols: int = 50
    min_periods: int = 12
    min_cross_section: int = 30
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    locked_oos_fraction: float = 0.30
    cost_model: CostModel = CostModel(fee_bps=1.0, slippage_bps=5.0)


def run_edgar_full_universe_ic(
    observations: Sequence[EdgarIcObservation],
    *,
    config: EdgarIcConfig | None = None,
    coverage: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    cfg = config or EdgarIcConfig()
    _validate_config(cfg)
    eligible, excluded = _eligible_observations(observations)
    periods = sorted({row.as_of for row in eligible})
    symbols = sorted({row.symbol.upper() for row in eligible})
    feasibility = _data_feasibility(eligible, cfg)
    if feasibility is not None:
        return _blocked_report(
            reason=feasibility,
            config=cfg,
            coverage=coverage or {},
            eligible_rows=len(eligible),
            excluded_rows=excluded,
            periods=len(periods),
            symbols=len(symbols),
        )

    oos_start = _locked_oos_start(periods, cfg.locked_oos_fraction)
    trial_reports: dict[str, dict[str, Any]] = {}
    p_values: list[float] = []
    trial_series: list[list[float]] = []
    for factor in FACTOR_NAMES:
        for horizon in HORIZONS:
            key = _trial_key(factor, horizon)
            report = _trial_report(
                eligible,
                factor=factor,
                horizon=horizon,
                oos_start=oos_start,
                cfg=cfg,
            )
            trial_reports[key] = report
            p_values.append(float(report["is_rank_ic"]["p_value"]))
            series = [float(item["ic"]) for item in report["monthly_ic"]]
            trial_series.append(series)

    discoveries = benjamini_hochberg(p_values, alpha=cfg.fdr_alpha)
    for discovered, report in zip(discoveries, trial_reports.values(), strict=True):
        report["fdr_discovery"] = discovered
    fdr_survivors = [
        key
        for key, report in trial_reports.items()
        if report["fdr_discovery"] and report["oos_rank_ic"]["mean"] > 0.0
    ]
    pbo_report = _pbo_report(trial_series, cfg.pbo_splits)
    verdict = (
        "SUGGESTIVE_NEEDS_PAID_CONFIRM"
        if (
            fdr_survivors
            and bool(pbo_report.get("valid"))
            and _float_value(pbo_report.get("pbo"), default=1.0) < 0.5
        )
        else "NO_EDGE"
    )
    reason = (
        "at least one predeclared factor-horizon survived BH-FDR, OOS IC sign, and valid PBO"
        if verdict == "SUGGESTIVE_NEEDS_PAID_CONFIRM"
        else "no predeclared factor-horizon survived full-scope BH-FDR plus OOS/PBO gates"
    )
    return {
        "status": "OK",
        "state": "EDGE" if verdict == "SUGGESTIVE_NEEDS_PAID_CONFIRM" else "NO_EDGE",
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": SURVIVOR_LIGHT_UNLOCK_CONDITION,
        "survivorship": "survivor_light",
        "verdict_ceiling": "SUGGESTIVE_NEEDS_PAID_CONFIRM",
        "predeclared": {
            "factors": list(FACTOR_NAMES),
            "horizons": list(HORIZONS),
            "trial_count_n": TRIAL_COUNT_N,
        },
        "coverage": {
            **dict(coverage or {}),
            "eligible_rows": len(eligible),
            "excluded_rows": excluded,
            "symbols": len(symbols),
            "periods": len(periods),
            "first_period": periods[0].isoformat(),
            "last_period": periods[-1].isoformat(),
        },
        "split": {
            "locked_oos_start": oos_start.isoformat(),
            "is_periods": sum(1 for value in periods if value < oos_start),
            "oos_periods": sum(1 for value in periods if value >= oos_start),
        },
        "multiple_testing": {
            "method": "BH-FDR over all factor x horizon trials",
            "alpha": cfg.fdr_alpha,
            "candidate_count_n": TRIAL_COUNT_N,
            "raw_survivors": sum(1 for p_value in p_values if p_value <= cfg.fdr_alpha),
            "fdr_survivors": len(fdr_survivors),
            "fdr_survivor_keys": fdr_survivors,
            "pbo": pbo_report,
        },
        "cost_model": {
            "fee_bps_one_way": cfg.cost_model.fee_bps,
            "slippage_bps_one_way": cfg.cost_model.slippage_bps,
            "one_way_cost": cfg.cost_model.one_way_cost,
            "funding": "N/A for equity factor IC / auxiliary long-short paper portfolio",
        },
        "trials": trial_reports,
        "standard_metrics": _aggregate_metrics(trial_reports),
        "benchmarks": {
            "cash": {"mean_return": 0.0},
            "note": (
                "IC diagnostics compare factor ranks to forward returns; auxiliary "
                "portfolios are long-short and cost-aware."
            ),
        },
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "data_adequacy": "limited",
            "unlock_condition": SURVIVOR_LIGHT_UNLOCK_CONDITION,
        },
    }


def _eligible_observations(
    observations: Sequence[EdgarIcObservation],
) -> tuple[list[EdgarIcObservation], int]:
    eligible: list[EdgarIcObservation] = []
    excluded = 0
    for row in observations:
        if not row.in_universe or row.available_on > row.as_of:
            excluded += 1
            continue
        clean_factors = {
            key: float(value)
            for key, value in row.factors.items()
            if key in FACTOR_NAMES and math.isfinite(float(value))
        }
        clean_returns = {
            key: float(value)
            for key, value in row.forward_returns.items()
            if key in HORIZONS and math.isfinite(float(value))
        }
        if not clean_factors or not clean_returns:
            excluded += 1
            continue
        eligible.append(
            EdgarIcObservation(
                symbol=row.symbol.upper(),
                as_of=row.as_of,
                available_on=row.available_on,
                factors=clean_factors,
                forward_returns=clean_returns,
                in_universe=True,
            )
        )
    return eligible, excluded


def _data_feasibility(rows: Sequence[EdgarIcObservation], cfg: EdgarIcConfig) -> str | None:
    if not rows:
        return "0 eligible PIT observations after universe, availability, and factor-label filters"
    symbols = {row.symbol for row in rows}
    periods = {row.as_of for row in rows}
    if len(symbols) < cfg.min_symbols:
        return f"eligible symbol count {len(symbols)} is below required {cfg.min_symbols}"
    if len(periods) < cfg.min_periods:
        return f"eligible monthly period count {len(periods)} is below required {cfg.min_periods}"
    max_cross_section = max(
        len({row.symbol for row in rows if row.as_of == period}) for period in periods
    )
    if max_cross_section < cfg.min_cross_section:
        return (
            f"maximum cross-section {max_cross_section} is below required "
            f"{cfg.min_cross_section}"
        )
    return None


def _blocked_report(
    *,
    reason: str,
    config: EdgarIcConfig,
    coverage: Mapping[str, object],
    eligible_rows: int,
    excluded_rows: int,
    periods: int,
    symbols: int,
) -> dict[str, Any]:
    return {
        "status": "INSUFFICIENT_DATA",
        "state": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": SURVIVOR_LIGHT_UNLOCK_CONDITION,
        "survivorship": "survivor_light",
        "predeclared": {
            "factors": list(FACTOR_NAMES),
            "horizons": list(HORIZONS),
            "trial_count_n": TRIAL_COUNT_N,
        },
        "coverage": {
            **dict(coverage),
            "eligible_rows": eligible_rows,
            "excluded_rows": excluded_rows,
            "symbols": symbols,
            "periods": periods,
        },
        "multiple_testing": {
            "method": "BH-FDR over all factor x horizon trials",
            "alpha": config.fdr_alpha,
            "candidate_count_n": TRIAL_COUNT_N,
            "raw_survivors": 0,
            "fdr_survivors": 0,
            "pbo": {"valid": False, "reason": "data gate blocked before PBO"},
        },
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "data_adequacy": "blocked",
            "unlock_condition": SURVIVOR_LIGHT_UNLOCK_CONDITION,
        },
    }


def _trial_report(
    rows: Sequence[EdgarIcObservation],
    *,
    factor: str,
    horizon: str,
    oos_start: date,
    cfg: EdgarIcConfig,
) -> dict[str, Any]:
    monthly_ic = _monthly_rank_ic(rows, factor=factor, horizon=horizon, min_n=cfg.min_cross_section)
    is_values = [item["ic"] for item in monthly_ic if item["as_of"] < oos_start]
    oos_values = [item["ic"] for item in monthly_ic if item["as_of"] >= oos_start]
    portfolio_returns = _auxiliary_top_bottom_returns(rows, factor=factor, horizon=horizon, cfg=cfg)
    return {
        "factor": factor,
        "horizon": horizon,
        "monthly_ic": [
            {"as_of": item["as_of"].isoformat(), "ic": item["ic"], "n": item["n"]}
            for item in monthly_ic
        ],
        "is_rank_ic": _series_summary(is_values),
        "oos_rank_ic": _series_summary(oos_values),
        "auxiliary_long_short": _return_summary(portfolio_returns),
        "fdr_discovery": False,
    }


def _monthly_rank_ic(
    rows: Sequence[EdgarIcObservation],
    *,
    factor: str,
    horizon: str,
    min_n: int,
) -> list[dict[str, Any]]:
    by_date: dict[date, list[EdgarIcObservation]] = defaultdict(list)
    for row in rows:
        by_date[row.as_of].append(row)
    result: list[dict[str, Any]] = []
    for as_of, period_rows in sorted(by_date.items()):
        clean = [
            row
            for row in period_rows
            if factor in row.factors and horizon in row.forward_returns
        ]
        if len(clean) < min_n:
            continue
        corr = _spearman(
            [row.factors[factor] for row in clean],
            [row.forward_returns[horizon] for row in clean],
        )
        if corr is not None:
            result.append({"as_of": as_of, "ic": corr, "n": len(clean)})
    return result


def _auxiliary_top_bottom_returns(
    rows: Sequence[EdgarIcObservation],
    *,
    factor: str,
    horizon: str,
    cfg: EdgarIcConfig,
) -> list[float]:
    by_date: dict[date, list[EdgarIcObservation]] = defaultdict(list)
    for row in rows:
        by_date[row.as_of].append(row)
    returns: list[float] = []
    previous_long: set[str] = set()
    previous_short: set[str] = set()
    for _as_of, period_rows in sorted(by_date.items()):
        clean = sorted(
            [
                row
                for row in period_rows
                if factor in row.factors and horizon in row.forward_returns
            ],
            key=lambda row: row.factors[factor],
        )
        if len(clean) < cfg.min_cross_section:
            continue
        bucket = max(1, len(clean) // 5)
        short_rows = clean[:bucket]
        long_rows = clean[-bucket:]
        long_symbols = {row.symbol for row in long_rows}
        short_symbols = {row.symbol for row in short_rows}
        turnover = _turnover(previous_long, long_symbols, previous_short, short_symbols)
        long_return = statistics.fmean(row.forward_returns[horizon] for row in long_rows)
        short_return = statistics.fmean(row.forward_returns[horizon] for row in short_rows)
        gross = long_return - short_return
        returns.append(gross - turnover * cfg.cost_model.one_way_cost)
        previous_long = long_symbols
        previous_short = short_symbols
    return returns


def _turnover(
    previous_long: set[str],
    long_symbols: set[str],
    previous_short: set[str],
    short_symbols: set[str],
) -> float:
    if not previous_long and not previous_short:
        return 2.0
    long_weight = 1.0 / max(1, len(long_symbols))
    short_weight = 1.0 / max(1, len(short_symbols))
    long_change = len(previous_long.symmetric_difference(long_symbols)) * long_weight
    short_change = len(previous_short.symmetric_difference(short_symbols)) * short_weight
    return long_change + short_change


def _series_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "status": "INSUFFICIENT_DATA",
            "n": 0,
            "mean": 0.0,
            "ic_ir": 0.0,
            "t_value": 0.0,
            "p_value": 1.0,
            "positive_share": None,
        }
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    t_value = mean / (std / math.sqrt(len(clean))) if std > 0 else 0.0
    return {
        "status": "OK",
        "n": len(clean),
        "mean": mean,
        "ic_ir": mean / std if std > 0 else 0.0,
        "t_value": t_value,
        "p_value": normal_two_sided_p(t_value),
        "positive_share": sum(1 for value in clean if value > 0.0) / len(clean),
    }


def _return_summary(returns: Sequence[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in returns]
    if not values:
        return {"periods": 0, "mean_return": None, "sharpe": None, "max_drawdown": None}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "periods": len(values),
        "mean_return": mean,
        "sharpe": mean / std * math.sqrt(12.0) if std > 0 else 0.0,
        "max_drawdown": _max_drawdown(values),
    }


def _aggregate_metrics(trial_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    oos_means = [float(report["oos_rank_ic"]["mean"]) for report in trial_reports.values()]
    return {
        "mean_oos_rank_ic_across_trials": statistics.fmean(oos_means) if oos_means else 0.0,
        "trial_count": len(trial_reports),
    }


def _pbo_report(trial_series: Sequence[Sequence[float]], splits: int) -> dict[str, object]:
    usable = [list(series) for series in trial_series if len(series) >= splits]
    if len(usable) < 2:
        return {
            "valid": False,
            "reason": "fewer than two trials have enough monthly IC observations for PBO",
            "n_splits": splits,
        }
    min_len = min(len(series) for series in usable)
    try:
        report = pbo([series[-min_len:] for series in usable], n_splits=splits)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "n_splits": splits}
    return {"valid": True, **report}


def _locked_oos_start(periods: Sequence[date], fraction: float) -> date:
    index = int(len(periods) * (1.0 - fraction))
    bounded = min(max(index, 1), len(periods) - 1)
    return periods[bounded]


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return None if denominator == 0.0 else numerator / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _value in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def _trial_key(factor: str, horizon: str) -> str:
    return f"{factor}__{horizon}"


def _validate_config(config: EdgarIcConfig) -> None:
    if config.min_symbols < 1:
        raise ValueError("min_symbols must be positive")
    if config.min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    if config.min_cross_section < 3:
        raise ValueError("min_cross_section must be at least 3")
    if not 0.0 < config.fdr_alpha < 1.0:
        raise ValueError("fdr_alpha must be in (0, 1)")
    if config.pbo_splits < 4 or config.pbo_splits % 2:
        raise ValueError("pbo_splits must be an even integer >= 4")
    if not 0.0 < config.locked_oos_fraction < 1.0:
        raise ValueError("locked_oos_fraction must be in (0, 1)")


def _float_value(value: object, *, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
