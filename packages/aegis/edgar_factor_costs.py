"""Cost-aware EDGAR factor portfolio diagnostics.

This module consumes already-built point-in-time EDGAR observations. It does not
fetch SEC, price, broker, or live-trading data.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from aegis.backtest_core import (
    benjamini_hochberg,
    metrics_from_returns,
    pbo,
    sign_test_p_value,
)
from aegis.edgar_full_universe_ic import EdgarIcObservation

LOCKED_SURVIVING_TRIALS: tuple[tuple[str, str], ...] = (
    ("earnings_yield_ep", "1m"),
    ("earnings_yield_ep", "3m"),
    ("earnings_yield_ep", "6m"),
    ("fcf_yield", "3m"),
    ("fcf_yield", "6m"),
    ("sales_to_price_sp", "6m"),
)

PortfolioKind = Literal["long_short", "long_only"]

SHARADAR_UNLOCK_CONDITION = (
    "Sharadar/Norgate-grade PIT constituents, delisting-aware prices including bankrupt "
    "names, executable bid/ask, and borrow-fee history are required for clean deployment."
)


@dataclass(frozen=True)
class FactorCostProfile:
    name: str
    one_way_bps: float
    short_borrow_bps_per_year: float = 50.0

    @property
    def one_way_cost(self) -> float:
        return self.one_way_bps / 10_000.0

    @property
    def annual_borrow_cost(self) -> float:
        return self.short_borrow_bps_per_year / 10_000.0


@dataclass(frozen=True)
class EdgarFactorCostConfig:
    quantile: float = 0.20
    min_cross_section: int = 30
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    base_profile: FactorCostProfile = FactorCostProfile("base_6bps", 6.0)
    sensitivity_profiles: tuple[FactorCostProfile, ...] = (
        FactorCostProfile("base_6bps", 6.0),
        FactorCostProfile("stress_15bps", 15.0),
        FactorCostProfile("stress_30bps", 30.0),
    )


DEFAULT_EDGAR_FACTOR_COST_CONFIG = EdgarFactorCostConfig()


@dataclass(frozen=True)
class PeriodPortfolioReturn:
    as_of: date
    gross_return: float
    net_return: float
    turnover: float
    trading_cost: float
    borrow_cost: float
    long_count: int
    short_count: int


def run_edgar_factor_cost_diagnostic(
    observations: Sequence[EdgarIcObservation],
    *,
    config: EdgarFactorCostConfig = DEFAULT_EDGAR_FACTOR_COST_CONFIG,
    coverage: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Run locked #82/#83 factor survivors through explicit cost diagnostics."""

    _validate_config(config)
    eligible = _eligible_observations(observations)
    periods = sorted({row.as_of for row in eligible})
    symbols = sorted({row.symbol.upper() for row in eligible})
    if not eligible:
        return _blocked_report(
            "0 eligible PIT observations after as-of availability filters",
            coverage,
        )
    if len(periods) < config.pbo_splits:
        return _blocked_report(
            f"eligible monthly period count {len(periods)} is below PBO splits {config.pbo_splits}",
            coverage,
        )

    base_rows: list[dict[str, Any]] = []
    base_p_values: list[float] = []
    base_trial_returns: list[tuple[float, ...]] = []
    profiles: dict[str, Any] = {}
    long_only_rows: list[dict[str, Any]] = []
    for factor, horizon in LOCKED_SURVIVING_TRIALS:
        profile_rows: dict[str, Any] = {}
        for profile in config.sensitivity_profiles:
            simulated = simulate_factor_portfolio(
                eligible,
                factor=factor,
                horizon=horizon,
                kind="long_short",
                cost_profile=profile,
                config=config,
            )
            profile_rows[profile.name] = _simulation_summary(simulated)
            if profile == config.base_profile:
                base_summary = _simulation_summary(simulated)
                p_value = sign_test_p_value(
                    [row.net_return for row in simulated],
                    alternative="greater",
                )
                base_rows.append(
                    {
                        "factor": factor,
                        "horizon": horizon,
                        "p_value": p_value,
                        **base_summary,
                    }
                )
                base_p_values.append(p_value)
                base_trial_returns.append(tuple(row.net_return for row in simulated))
        profiles[_trial_key(factor, horizon)] = profile_rows

        long_only = simulate_factor_portfolio(
            eligible,
            factor=factor,
            horizon=horizon,
            kind="long_only",
            cost_profile=config.base_profile,
            config=config,
        )
        long_only_rows.append(
            {
                "factor": factor,
                "horizon": horizon,
                **_simulation_summary(long_only),
            }
        )

    fdr_flags = benjamini_hochberg(base_p_values, alpha=config.fdr_alpha)
    pbo_report = _pbo_report(base_trial_returns, config.pbo_splits)
    pbo_pass = bool(pbo_report.get("valid")) and _object_float(
        pbo_report.get("pbo", 1.0)
    ) < 0.5
    for row, flag in zip(base_rows, fdr_flags, strict=True):
        row["fdr_pass"] = bool(flag)
        row["net_verdict"] = (
            "SUGGESTIVE_NET_EDGE"
            if flag and pbo_pass and _object_float(row["net_annualized_return"]) > 0.0
            else "NO_EDGE"
        )
    survivors = [row for row in base_rows if row["net_verdict"] == "SUGGESTIVE_NET_EDGE"]
    state = "EDGE" if survivors else "NO_EDGE"
    verdict = "SUGGESTIVE_NET_EDGE" if survivors else "NO_EDGE"
    reason = (
        "at least one locked #83 factor retained positive net cost-adjusted returns after FDR/PBO"
        if survivors
        else "locked #83 factors did not retain a net cost-adjusted survivor after FDR/PBO"
    )
    return {
        "status": "OK",
        "state": state,
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": SHARADAR_UNLOCK_CONDITION,
        "survivorship": "survivor_light_free_asof",
        "verdict_ceiling": "SUGGESTIVE",
        "predeclared": {
            "trials": [_trial_key(factor, horizon) for factor, horizon in LOCKED_SURVIVING_TRIALS],
            "quantile": config.quantile,
            "base_profile": _profile_dict(config.base_profile),
            "sensitivity_profiles": [
                _profile_dict(profile) for profile in config.sensitivity_profiles
            ],
            "trial_count_n": len(LOCKED_SURVIVING_TRIALS),
        },
        "coverage": {
            **dict(coverage or {}),
            "eligible_rows": len(eligible),
            "symbols": len(symbols),
            "periods": len(periods),
            "first_period": periods[0].isoformat(),
            "last_period": periods[-1].isoformat(),
        },
        "multiple_testing": {
            "method": "BH-FDR over locked #83 surviving factor-horizon cost trials",
            "alpha": config.fdr_alpha,
            "candidate_count_n": len(LOCKED_SURVIVING_TRIALS),
            "raw_survivors": sum(1 for value in base_p_values if value <= config.fdr_alpha),
            "fdr_survivors": sum(1 for flag in fdr_flags if flag),
            "net_survivors": len(survivors),
            "pbo": pbo_report,
        },
        "base_long_short": base_rows,
        "sensitivity": profiles,
        "personal_long_only": long_only_rows,
        "gross_to_net_decay": [_decay_row(row) for row in base_rows],
        "sharadar_decision": _sharadar_decision(survivors),
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "available_on_filter": True,
            "asof_universe_required": True,
            "monthly_rebalance": True,
            "costs_applied": True,
            "survivor_light_ceiling_required": True,
        },
    }


def simulate_factor_portfolio(
    observations: Sequence[EdgarIcObservation],
    *,
    factor: str,
    horizon: str,
    kind: PortfolioKind,
    cost_profile: FactorCostProfile,
    config: EdgarFactorCostConfig = DEFAULT_EDGAR_FACTOR_COST_CONFIG,
) -> tuple[PeriodPortfolioReturn, ...]:
    """Simulate monthly top/bottom quintile portfolios from PIT observations."""

    previous_long: dict[str, float] = {}
    previous_short: dict[str, float] = {}
    rows: list[PeriodPortfolioReturn] = []
    for as_of, period_rows in sorted(_by_period(_eligible_observations(observations)).items()):
        clean = sorted(
            (
                row
                for row in period_rows
                if factor in row.factors and horizon in row.forward_returns
            ),
            key=lambda row: row.factors[factor],
        )
        if len(clean) < config.min_cross_section:
            continue
        bucket = max(1, int(len(clean) * config.quantile))
        short_bucket = clean[:bucket] if kind == "long_short" else []
        long_bucket = clean[-bucket:]
        long_weights = {row.symbol.upper(): 1.0 / len(long_bucket) for row in long_bucket}
        short_weights = (
            {row.symbol.upper(): 1.0 / len(short_bucket) for row in short_bucket}
            if short_bucket
            else {}
        )
        long_return = statistics.fmean(row.forward_returns[horizon] for row in long_bucket)
        short_return = (
            statistics.fmean(row.forward_returns[horizon] for row in short_bucket)
            if short_bucket
            else 0.0
        )
        gross = long_return - short_return if kind == "long_short" else long_return
        long_turnover = _weight_turnover(previous_long, long_weights)
        short_turnover = _weight_turnover(previous_short, short_weights)
        trading_cost = (long_turnover + short_turnover) * cost_profile.one_way_cost
        borrow_cost = (
            cost_profile.annual_borrow_cost * _horizon_months(horizon) / 12.0
            if kind == "long_short"
            else 0.0
        )
        rows.append(
            PeriodPortfolioReturn(
                as_of=as_of,
                gross_return=gross,
                net_return=gross - trading_cost - borrow_cost,
                turnover=long_turnover + short_turnover,
                trading_cost=trading_cost,
                borrow_cost=borrow_cost,
                long_count=len(long_bucket),
                short_count=len(short_bucket),
            )
        )
        previous_long = long_weights
        previous_short = short_weights
    return tuple(rows)


def _eligible_observations(
    observations: Sequence[EdgarIcObservation],
) -> tuple[EdgarIcObservation, ...]:
    return tuple(
        EdgarIcObservation(
            symbol=row.symbol.upper(),
            as_of=row.as_of,
            available_on=row.available_on,
            factors=dict(row.factors),
            forward_returns=dict(row.forward_returns),
            in_universe=True,
        )
        for row in observations
        if row.in_universe and row.available_on <= row.as_of
    )


def _by_period(
    observations: Sequence[EdgarIcObservation],
) -> dict[date, list[EdgarIcObservation]]:
    by_date: dict[date, list[EdgarIcObservation]] = defaultdict(list)
    for row in observations:
        by_date[row.as_of].append(row)
    return by_date


def _weight_turnover(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    symbols = set(previous) | set(current)
    return sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)


def _simulation_summary(rows: Sequence[PeriodPortfolioReturn]) -> dict[str, Any]:
    gross_returns = [row.gross_return for row in rows]
    net_returns = [row.net_return for row in rows]
    turnover = sum(row.turnover for row in rows)
    net_cost = sum(row.trading_cost + row.borrow_cost for row in rows)
    gross = metrics_from_returns(
        gross_returns,
        annualization_periods=12,
        turnover=turnover,
        net_cost=0.0,
        nonpositive_annualized_return=0.0,
    )
    net = metrics_from_returns(
        net_returns,
        annualization_periods=12,
        turnover=turnover,
        net_cost=net_cost,
        nonpositive_annualized_return=0.0,
    )
    return {
        "periods": len(rows),
        "gross_annualized_return": gross.annualized_return,
        "net_annualized_return": net.annualized_return,
        "gross_sharpe": gross.sharpe,
        "net_sharpe": net.sharpe,
        "net_max_drawdown": net.max_drawdown,
        "net_sortino": net.sortino,
        "net_calmar": net.calmar,
        "monthly_win_rate": net.positive_period_win_rate,
        "turnover": turnover,
        "annualized_turnover": net.annualized_turnover,
        "net_cost": net_cost,
        "mean_trading_cost": statistics.fmean(row.trading_cost for row in rows) if rows else 0.0,
        "mean_borrow_cost": statistics.fmean(row.borrow_cost for row in rows) if rows else 0.0,
        "mean_long_count": statistics.fmean(row.long_count for row in rows) if rows else 0.0,
        "mean_short_count": statistics.fmean(row.short_count for row in rows) if rows else 0.0,
    }


def _pbo_report(trial_returns: Sequence[Sequence[float]], splits: int) -> dict[str, object]:
    usable = [tuple(series) for series in trial_returns if len(series) >= splits]
    if len(usable) < 2:
        return {
            "valid": False,
            "reason": "fewer than two cost trials have enough observations for PBO",
            "n_splits": splits,
        }
    min_len = min(len(series) for series in usable)
    try:
        report = pbo([series[-min_len:] for series in usable], n_splits=splits)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "n_splits": splits}
    return {"valid": True, **report}


def _blocked_report(reason: str, coverage: Mapping[str, object] | None) -> dict[str, Any]:
    return {
        "status": "INSUFFICIENT_DATA",
        "state": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": SHARADAR_UNLOCK_CONDITION,
        "coverage": dict(coverage or {}),
        "multiple_testing": {
            "method": "BH-FDR over locked #83 surviving factor-horizon cost trials",
            "candidate_count_n": len(LOCKED_SURVIVING_TRIALS),
            "raw_survivors": 0,
            "fdr_survivors": 0,
            "pbo": {"valid": False, "reason": "data gate blocked before PBO"},
        },
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
        },
    }


def _decay_row(row: Mapping[str, object]) -> dict[str, object]:
    gross = _object_float(row["gross_annualized_return"])
    net = _object_float(row["net_annualized_return"])
    return {
        "factor": row["factor"],
        "horizon": row["horizon"],
        "gross_annualized_return": gross,
        "net_annualized_return": net,
        "decay": gross - net,
        "decay_ratio": (gross - net) / abs(gross) if gross else None,
        "turnover": row["turnover"],
    }


def _sharadar_decision(survivors: Sequence[Mapping[str, object]]) -> dict[str, str]:
    if survivors:
        return {
            "decision": "SHARADAR_WORTH_PAYING_TO_CONFIRM_NET_COSTS",
            "reason": (
                "at least one free as-of factor retained a net cost-adjusted long-short "
                "survivor; paid PIT data should confirm delisted names, bid/ask, and borrow"
            ),
        }
    return {
        "decision": "DO_NOT_PAY_SHARADAR_FOR_THIS_FACTOR_SET_NOW",
        "reason": "the free as-of survivors did not retain a net cost-adjusted portfolio survivor",
    }


def _profile_dict(profile: FactorCostProfile) -> dict[str, float | str]:
    return {
        "name": profile.name,
        "one_way_bps": profile.one_way_bps,
        "short_borrow_bps_per_year": profile.short_borrow_bps_per_year,
    }


def _object_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _trial_key(factor: str, horizon: str) -> str:
    return f"{factor}__{horizon}"


def _horizon_months(horizon: str) -> int:
    if not horizon.endswith("m"):
        raise ValueError(f"unsupported horizon label: {horizon}")
    months = int(horizon[:-1])
    if months < 1:
        raise ValueError("horizon months must be positive")
    return months


def _validate_config(config: EdgarFactorCostConfig) -> None:
    if not 0.0 < config.quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    if config.min_cross_section < 5:
        raise ValueError("min_cross_section must be at least 5")
    if not 0.0 < config.fdr_alpha < 1.0:
        raise ValueError("fdr_alpha must be in (0, 1)")
    if config.pbo_splits < 4 or config.pbo_splits % 2:
        raise ValueError("pbo_splits must be an even integer >= 4")
    for profile in config.sensitivity_profiles:
        if profile.one_way_bps < 0.0 or profile.short_borrow_bps_per_year < 0.0:
            raise ValueError("cost profile values must be nonnegative")
