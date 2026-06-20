"""Disciplined survivor-light equity factor search harness.

The module evaluates predeclared cross-sectional equity factors and composites.
It consumes point-in-time fundamental snapshots and price observations supplied
by the caller; it does not fetch paid data, connect to live trading, or register
signals.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from aegis.edgar_pit import EdgarFact, derive_fcf, derive_net_debt

Verdict = Literal["SUGGESTIVE_NEEDS_PAID_CONFIRM", "NO_EDGE", "INSUFFICIENT"]
FactorFamily = Literal["value", "quality", "momentum", "risk_size", "composite"]

SURVIVOR_LIGHT_VERDICT_CEILING = "SURVIVOR_LIGHT_ONLY_NO_ROBUST_VERDICT"
DEFAULT_FDR_ALPHA = 0.10
DEFAULT_LOCKED_OOS_FRACTION = 0.30
DEFAULT_COMMISSION_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_FORWARD_PERIODS = 1
DEFAULT_GROUPS = 5
DEFAULT_WINSORIZE_PCT = 0.01
MIN_CROSS_SECTION = 3
MIN_OOS_PERIODS = 2


@dataclass(frozen=True)
class PriceObservation:
    symbol: str
    as_of: date
    close: float
    forward_return: float
    market_cap: float | None = None
    volatility_252: float | None = None
    beta_252: float | None = None
    momentum_12_1: float | None = None


@dataclass(frozen=True)
class FactorDeclaration:
    name: str
    family: FactorFamily
    description: str
    direction: int = 1


@dataclass(frozen=True)
class CompositeDeclaration:
    name: str
    family: FactorFamily
    description: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class SearchConfig:
    universe_name: str = "survivor_light_us_equity"
    locked_oos_fraction: float = DEFAULT_LOCKED_OOS_FRACTION
    fdr_alpha: float = DEFAULT_FDR_ALPHA
    groups: int = DEFAULT_GROUPS
    winsorize_pct: float = DEFAULT_WINSORIZE_PCT
    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    forward_periods: int = DEFAULT_FORWARD_PERIODS


PREDECLARED_FACTORS: tuple[FactorDeclaration, ...] = (
    FactorDeclaration("earnings_yield_ep", "value", "E/P: net income divided by market cap"),
    FactorDeclaration("book_to_price_bp", "value", "B/P: book equity divided by market cap"),
    FactorDeclaration("fcf_yield", "value", "Free cash flow yield"),
    FactorDeclaration("ebit_ev", "value", "EBIT/EV using operating income as EBIT"),
    FactorDeclaration("roe", "quality", "Return on equity"),
    FactorDeclaration("gp_assets", "quality", "Gross profit divided by assets"),
    FactorDeclaration("low_accruals", "quality", "Negative accruals over assets"),
    FactorDeclaration("low_leverage", "quality", "Negative liabilities over assets"),
    FactorDeclaration("low_asset_growth", "quality", "Negative asset growth"),
    FactorDeclaration("momentum_12_1", "momentum", "12-1 price momentum"),
    FactorDeclaration("low_volatility", "risk_size", "Negative realized volatility"),
    FactorDeclaration("small_size", "risk_size", "Negative market capitalization"),
)

PREDECLARED_COMPOSITES: tuple[CompositeDeclaration, ...] = (
    CompositeDeclaration(
        "qarp_quality_value",
        "composite",
        "QARP: equal-weight value and quality z-scores",
        (
            "earnings_yield_ep",
            "book_to_price_bp",
            "fcf_yield",
            "ebit_ev",
            "roe",
            "gp_assets",
            "low_accruals",
            "low_leverage",
            "low_asset_growth",
        ),
    ),
    CompositeDeclaration(
        "value_momentum",
        "composite",
        "Equal-weight value plus 12-1 momentum",
        ("earnings_yield_ep", "book_to_price_bp", "fcf_yield", "ebit_ev", "momentum_12_1"),
    ),
    CompositeDeclaration(
        "quality_momentum",
        "composite",
        "Equal-weight quality plus 12-1 momentum",
        (
            "roe",
            "gp_assets",
            "low_accruals",
            "low_leverage",
            "low_asset_growth",
            "momentum_12_1",
        ),
    ),
    CompositeDeclaration(
        "multifactor_equal_z",
        "composite",
        "Equal-weight value, quality, momentum, low-risk, and small-size z-scores",
        tuple(factor.name for factor in PREDECLARED_FACTORS),
    ),
)


def run_equity_factor_search(
    price_observations: Sequence[PriceObservation],
    fundamentals_by_symbol: Mapping[str, Mapping[date, Mapping[str, EdgarFact]]],
    *,
    config: SearchConfig | None = None,
) -> dict[str, Any]:
    config = config or SearchConfig()
    _validate_config(config)
    if not price_observations:
        return _insufficient_report(config, "price_observations is empty")
    if not fundamentals_by_symbol:
        return _insufficient_report(config, "fundamentals_by_symbol is empty")

    rows = _build_factor_rows(price_observations, fundamentals_by_symbol, config)
    if not rows:
        return _insufficient_report(config, "no PIT factor rows could be built")

    dates = sorted({row["as_of"] for row in rows})
    locked_oos_start_index = int(len(dates) * (1.0 - config.locked_oos_fraction))
    if locked_oos_start_index <= 0 or locked_oos_start_index >= len(dates):
        return _insufficient_report(config, "not enough dates for locked OOS split")
    locked_oos_start_date = dates[locked_oos_start_index]
    is_rows = [row for row in rows if row["as_of"] < locked_oos_start_date]
    oos_rows = [row for row in rows if row["as_of"] >= locked_oos_start_date]
    if not is_rows or not oos_rows:
        return _insufficient_report(config, "IS/OOS split produced empty side")

    trial_names = [factor.name for factor in PREDECLARED_FACTORS] + [
        composite.name for composite in PREDECLARED_COMPOSITES
    ]
    n = len(trial_names)
    factor_results = {
        name: _evaluate_trial(name, is_rows, oos_rows, config) for name in trial_names
    }
    p_values = [float(result["is"]["rank_ic"]["p_value"]) for result in factor_results.values()]
    discoveries = benjamini_hochberg(p_values, alpha=config.fdr_alpha)
    for discovered, result in zip(discoveries, factor_results.values(), strict=True):
        result["fdr_discovery"] = discovered
    fdr_survivors = [name for name, result in factor_results.items() if result["fdr_discovery"]]
    oos_survivors = [
        name
        for name in fdr_survivors
        if factor_results[name]["oos"]["top_bottom"]["net_mean_return"] > 0
        and factor_results[name]["oos"]["top_bottom"]["sharpe"] > 0
        and factor_results[name]["oos"]["ic"]["rank_ic_mean"] > 0
    ]
    verdict = _verdict(
        total_rows=len(rows),
        oos_periods=len({row["as_of"] for row in oos_rows}),
        fdr_survivors=len(fdr_survivors),
        oos_survivors=len(oos_survivors),
    )
    return {
        "status": "OK",
        "verdict": verdict,
        "verdict_ceiling": SURVIVOR_LIGHT_VERDICT_CEILING,
        "survivorship": "survivor_light",
        "point_in_time": {
            "fundamentals": (
                "EDGAR facts supplied as-of; row builder only reads facts visible on row date"
            ),
            "prices": "caller-supplied #48/free-data price observations",
            "restatement_note": (
                "EDGAR companyfacts latest-history limitations remain; verdict capped"
            ),
        },
        "predeclared": {
            "factors": [factor.__dict__ for factor in PREDECLARED_FACTORS],
            "composites": [composite.__dict__ for composite in PREDECLARED_COMPOSITES],
            "trial_count_n": n,
            "parameter_grid": {
                "groups": config.groups,
                "rebalance_frequency": "caller observation frequency",
                "winsorize_pct": config.winsorize_pct,
                "locked_oos_fraction": config.locked_oos_fraction,
                "forward_periods": config.forward_periods,
            },
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR over all predeclared factors and composites",
            "alpha": config.fdr_alpha,
            "trial_count_n": n,
            "raw_survivors": sum(
                1
                for result in factor_results.values()
                if result["is"]["rank_ic"]["p_value"] <= config.fdr_alpha
            ),
            "fdr_survivors": len(fdr_survivors),
            "oos_survivors_after_fdr": len(oos_survivors),
        },
        "split": {
            "locked_oos_start_date": locked_oos_start_date.isoformat(),
            "is_dates": len({row["as_of"] for row in is_rows}),
            "oos_dates": len({row["as_of"] for row in oos_rows}),
            "first_oos_signal_date": locked_oos_start_date.isoformat(),
        },
        "cost_model": {
            "commission_bps": config.commission_bps,
            "slippage_bps": config.slippage_bps,
            "round_trip_bps": round(_round_trip_cost(config) * 10_000.0, 8),
            "funding_or_borrow": "N/A spot long-short factor paper portfolios",
        },
        "benchmarks": _benchmarks(rows, config),
        "factors": factor_results,
        "counts": {
            "rows": len(rows),
            "symbols": len({row["symbol"] for row in rows}),
            "dates": len(dates),
            "n": n,
            "fdr_before": sum(
                1
                for result in factor_results.values()
                if result["is"]["rank_ic"]["p_value"] <= config.fdr_alpha
            ),
            "fdr_after": len(fdr_survivors),
        },
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "public_edge_values_sanitized": True,
        },
    }


def build_factor_values(
    facts: Mapping[str, EdgarFact],
    price: PriceObservation,
    previous_facts: Mapping[str, EdgarFact] | None = None,
) -> dict[str, float]:
    shares = _fact_value(facts, "CommonStockSharesOutstanding")
    if shares is None:
        shares = _fact_value(facts, "EntityCommonStockSharesOutstanding")
    if shares is None or shares <= 0 or price.close <= 0:
        return {}
    market_cap = price.market_cap if price.market_cap is not None else shares * price.close
    if market_cap <= 0:
        return {}
    assets = _fact_value(facts, "Assets")
    liabilities = _fact_value(facts, "Liabilities")
    equity = _fact_value(facts, "StockholdersEquity")
    net_income = _fact_value(facts, "NetIncomeLoss")
    gross_profit = _fact_value(facts, "GrossProfit")
    operating_income = _fact_value(facts, "OperatingIncomeLoss")
    operating_cash_flow = _fact_value(facts, "NetCashProvidedByUsedInOperatingActivities")
    fcf = derive_fcf(dict(facts))
    net_debt = derive_net_debt(dict(facts))
    enterprise_value = market_cap + (net_debt.value if net_debt is not None else 0.0)
    previous_assets = _fact_value(previous_facts or {}, "Assets")
    accruals = (
        None
        if net_income is None or operating_cash_flow is None
        else net_income - operating_cash_flow
    )
    accrual_ratio = _safe_div(accruals, assets)
    leverage_ratio = _safe_div(liabilities, assets)
    asset_growth_ratio = _safe_div(
        None if assets is None or previous_assets is None else assets - previous_assets,
        previous_assets,
    )
    values = {
        "earnings_yield_ep": _safe_div(net_income, market_cap),
        "book_to_price_bp": _safe_div(equity, market_cap),
        "fcf_yield": _safe_div(fcf.value if fcf is not None else None, market_cap),
        "ebit_ev": _safe_div(operating_income, enterprise_value),
        "roe": _safe_div(net_income, equity),
        "gp_assets": _safe_div(gross_profit, assets),
        "low_accruals": -accrual_ratio if accrual_ratio is not None else None,
        "low_leverage": -leverage_ratio if leverage_ratio is not None else None,
        "low_asset_growth": -asset_growth_ratio if asset_growth_ratio is not None else None,
        "momentum_12_1": price.momentum_12_1,
        "low_volatility": -price.volatility_252 if price.volatility_252 is not None else None,
        "small_size": -market_cap,
    }
    return {
        key: float(value)
        for key, value in values.items()
        if value is not None and math.isfinite(value)
    }


def benjamini_hochberg(p_values: Sequence[float], *, alpha: float) -> list[bool]:
    if not p_values:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    threshold_rank = -1
    m = len(p_values)
    for rank, (_index, p_value) in enumerate(ordered, start=1):
        if p_value <= (rank / m) * alpha:
            threshold_rank = rank
    discoveries = [False] * len(p_values)
    if threshold_rank > 0:
        cutoff = ordered[threshold_rank - 1][1]
        discoveries = [p_value <= cutoff for p_value in p_values]
    return discoveries


def sanitized_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "verdict": report.get("verdict"),
        "verdict_ceiling": report.get("verdict_ceiling"),
        "survivorship": report.get("survivorship"),
        "predeclared": report.get("predeclared"),
        "multiple_testing": report.get("multiple_testing"),
        "split": report.get("split"),
        "cost_model": report.get("cost_model"),
        "counts": report.get("counts"),
        "safety": report.get("safety"),
        "note": "edge metrics intentionally omitted from public sanitized report",
    }


def _build_factor_rows(
    price_observations: Sequence[PriceObservation],
    fundamentals_by_symbol: Mapping[str, Mapping[date, Mapping[str, EdgarFact]]],
    config: SearchConfig,
) -> list[dict[str, Any]]:
    by_symbol_dates = {
        symbol.upper(): sorted(snapshots)
        for symbol, snapshots in fundamentals_by_symbol.items()
    }
    rows: list[dict[str, Any]] = []
    for price in sorted(price_observations, key=lambda item: (item.as_of, item.symbol)):
        symbol = price.symbol.upper()
        snapshots = fundamentals_by_symbol.get(symbol, {})
        visible_date = _latest_visible_date(by_symbol_dates.get(symbol, []), price.as_of)
        if visible_date is None:
            continue
        previous_date = _previous_visible_date(by_symbol_dates.get(symbol, []), visible_date)
        values = build_factor_values(
            snapshots[visible_date],
            price,
            snapshots.get(previous_date) if previous_date is not None else None,
        )
        if not values:
            continue
        row: dict[str, Any] = {
            "symbol": symbol,
            "as_of": price.as_of,
            "forward_return": price.forward_return,
            "market_cap": price.market_cap,
        }
        row.update(values)
        rows.append(row)
    _attach_composites(rows, config)
    return rows


def _attach_composites(rows: list[dict[str, Any]], config: SearchConfig) -> None:
    factor_names = [factor.name for factor in PREDECLARED_FACTORS]
    for as_of in sorted({row["as_of"] for row in rows}):
        period_rows = [row for row in rows if row["as_of"] == as_of]
        zscores_by_factor = {
            factor: _cross_sectional_zscores(period_rows, factor, config)
            for factor in factor_names
        }
        for row in period_rows:
            for composite in PREDECLARED_COMPOSITES:
                values = [
                    zscores_by_factor[component].get(row["symbol"])
                    for component in composite.components
                ]
                clean = [value for value in values if value is not None]
                if clean:
                    row[composite.name] = statistics.fmean(clean)


def _evaluate_trial(
    name: str,
    is_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    config: SearchConfig,
) -> dict[str, Any]:
    return {
        "is": {
            "rank_ic": _rank_ic_summary(is_rows, name),
            "top_bottom": _top_bottom_summary(is_rows, name, config),
        },
        "oos": {
            "ic": _rank_ic_summary(oos_rows, name),
            "top_bottom": _top_bottom_summary(oos_rows, name, config),
        },
        "fdr_discovery": False,
    }


def _rank_ic_summary(rows: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    values: list[float] = []
    for period_rows in _rows_by_date(rows).values():
        clean = [row for row in period_rows if factor in row]
        if len(clean) < MIN_CROSS_SECTION:
            continue
        corr = _spearman(
            [float(row[factor]) for row in clean],
            [float(row["forward_return"]) for row in clean],
        )
        if corr is not None:
            values.append(corr)
    return _series_summary(values)


def _top_bottom_summary(
    rows: list[dict[str, Any]],
    factor: str,
    config: SearchConfig,
) -> dict[str, Any]:
    returns: list[float] = []
    turnovers: list[float] = []
    previous_long: set[str] = set()
    previous_short: set[str] = set()
    for period_rows in _rows_by_date(rows).values():
        clean = sorted(
            [row for row in period_rows if factor in row],
            key=lambda row: float(row[factor]),
        )
        if len(clean) < config.groups:
            continue
        bucket = max(1, len(clean) // config.groups)
        short = clean[:bucket]
        long = clean[-bucket:]
        long_symbols = {str(row["symbol"]) for row in long}
        short_symbols = {str(row["symbol"]) for row in short}
        turnover = _portfolio_turnover(previous_long, long_symbols, previous_short, short_symbols)
        long_return = statistics.fmean(float(row["forward_return"]) for row in long)
        short_return = statistics.fmean(float(row["forward_return"]) for row in short)
        gross_return = long_return - short_return
        net_return = gross_return - turnover * _round_trip_cost(config)
        returns.append(net_return)
        turnovers.append(turnover)
        previous_long = long_symbols
        previous_short = short_symbols
    summary = _return_metrics(returns)
    summary.update(
        {
            "periods": len(returns),
            "net_mean_return": statistics.fmean(returns) if returns else 0.0,
            "avg_turnover": statistics.fmean(turnovers) if turnovers else 0.0,
            "net_cost": (
                statistics.fmean(turnovers) * _round_trip_cost(config) if turnovers else 0.0
            ),
        }
    )
    return summary


def _benchmarks(rows: list[dict[str, Any]], config: SearchConfig) -> dict[str, Any]:
    equal_weight: list[float] = []
    cap_weight: list[float] = []
    for period_rows in _rows_by_date(rows).values():
        if not period_rows:
            continue
        equal_weight.append(statistics.fmean(float(row["forward_return"]) for row in period_rows))
        weighted = _cap_weighted_return(period_rows)
        if weighted is not None:
            cap_weight.append(weighted)
    return {
        "equal_weight": _return_metrics(equal_weight),
        "market_cap_weight": _return_metrics(cap_weight),
        "cost_note": (
            "benchmarks are observation-frequency buy-and-hold proxies; factor long-short "
            "portfolios pay turnover cost"
        ),
        "commission_bps": config.commission_bps,
        "slippage_bps": config.slippage_bps,
    }


def _cap_weighted_return(rows: list[dict[str, Any]]) -> float | None:
    weights = [
        (float(row["market_cap"]), float(row["forward_return"]))
        for row in rows
        if row.get("market_cap")
    ]
    total = sum(weight for weight, _return in weights)
    if total <= 0:
        return None
    return sum(weight / total * ret for weight, ret in weights)


def _return_metrics(returns: Sequence[float]) -> dict[str, float | int | None]:
    if not returns:
        return {
            "periods": 0,
            "mean_return": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "positive_share": None,
        }
    mean = statistics.fmean(returns)
    std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    downside = [min(0.0, value) for value in returns]
    downside_std = statistics.stdev(downside) if len(downside) > 1 else 0.0
    return {
        "periods": len(returns),
        "mean_return": mean,
        "sharpe": None if std == 0 else mean / std * math.sqrt(len(returns)),
        "sortino": None if downside_std == 0 else mean / downside_std * math.sqrt(len(returns)),
        "max_drawdown": _max_drawdown(returns),
        "positive_share": sum(1 for value in returns if value > 0) / len(returns),
    }


def _series_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "status": "INSUFFICIENT_DATA",
            "n": 0,
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "t_value": 0.0,
            "p_value": 1.0,
            "positive_share": None,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    t_value = 0.0 if std == 0 else mean / (std / math.sqrt(len(values)))
    return {
        "status": "OK",
        "n": len(values),
        "rank_ic_mean": mean,
        "rank_ic_ir": 0.0 if std == 0 else mean / std,
        "t_value": t_value,
        "p_value": _normal_two_sided_p(t_value),
        "positive_share": sum(1 for value in values if value > 0) / len(values),
    }


def _verdict(
    *,
    total_rows: int,
    oos_periods: int,
    fdr_survivors: int,
    oos_survivors: int,
) -> Verdict:
    if total_rows == 0 or oos_periods < MIN_OOS_PERIODS:
        return "INSUFFICIENT"
    if fdr_survivors > 0 and oos_survivors > 0:
        return "SUGGESTIVE_NEEDS_PAID_CONFIRM"
    return "NO_EDGE"


def _insufficient_report(config: SearchConfig, reason: str) -> dict[str, Any]:
    n = len(PREDECLARED_FACTORS) + len(PREDECLARED_COMPOSITES)
    return {
        "status": "INSUFFICIENT_DATA",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "verdict_ceiling": SURVIVOR_LIGHT_VERDICT_CEILING,
        "survivorship": "survivor_light",
        "predeclared": {
            "factors": [factor.__dict__ for factor in PREDECLARED_FACTORS],
            "composites": [composite.__dict__ for composite in PREDECLARED_COMPOSITES],
            "trial_count_n": n,
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR over all predeclared factors and composites",
            "alpha": config.fdr_alpha,
            "trial_count_n": n,
            "raw_survivors": 0,
            "fdr_survivors": 0,
            "oos_survivors_after_fdr": 0,
        },
    }


def _cross_sectional_zscores(
    rows: list[dict[str, Any]],
    factor: str,
    config: SearchConfig,
) -> dict[str, float]:
    values = [(str(row["symbol"]), float(row[factor])) for row in rows if factor in row]
    if len(values) < MIN_CROSS_SECTION:
        return {}
    raw = [value for _symbol, value in values]
    clipped = _winsorized(raw, config.winsorize_pct)
    mean = statistics.fmean(clipped)
    std = statistics.stdev(clipped) if len(clipped) > 1 else 0.0
    if std == 0:
        return {}
    return {
        symbol: (value - mean) / std
        for (symbol, _), value in zip(values, clipped, strict=True)
    }


def _winsorized(values: Sequence[float], pct: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    lower_index = int(len(ordered) * pct)
    upper_index = max(lower_index, int(len(ordered) * (1.0 - pct)) - 1)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return [min(max(value, lower), upper) for value in values]


def _rows_by_date(rows: Iterable[dict[str, Any]]) -> dict[date, list[dict[str, Any]]]:
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["as_of"]].append(row)
    return dict(sorted(grouped.items()))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < MIN_CROSS_SECTION:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < MIN_CROSS_SECTION:
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return None if denominator == 0 else numerator / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original, _value in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _portfolio_turnover(
    previous_long: set[str],
    long_symbols: set[str],
    previous_short: set[str],
    short_symbols: set[str],
) -> float:
    if not previous_long and not previous_short:
        return 1.0
    changed_long = len(previous_long.symmetric_difference(long_symbols))
    changed_short = len(previous_short.symmetric_difference(short_symbols))
    denominator = max(1, len(long_symbols) + len(short_symbols))
    return min(1.0, (changed_long + changed_short) / denominator)


def _round_trip_cost(config: SearchConfig) -> float:
    return (config.commission_bps + config.slippage_bps) * 2.0 / 10_000.0


def _fact_value(facts: Mapping[str, EdgarFact], concept: str) -> float | None:
    fact = facts.get(concept)
    return None if fact is None else fact.value


def _safe_div(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right == 0:
        return None
    value = left / right
    return value if math.isfinite(value) else None


def _latest_visible_date(dates: Sequence[date], as_of: date) -> date | None:
    visible = [candidate for candidate in dates if candidate <= as_of]
    return visible[-1] if visible else None


def _previous_visible_date(dates: Sequence[date], current: date) -> date | None:
    previous = [candidate for candidate in dates if candidate < current]
    return previous[-1] if previous else None


def _normal_two_sided_p(t_value: float) -> float:
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def _validate_config(config: SearchConfig) -> None:
    if not 0 < config.locked_oos_fraction < 1:
        raise ValueError("locked_oos_fraction must be in (0, 1)")
    if not 0 < config.fdr_alpha < 1:
        raise ValueError("fdr_alpha must be in (0, 1)")
    if config.groups < 2:
        raise ValueError("groups must be >= 2")
    if not 0 <= config.winsorize_pct < 0.5:
        raise ValueError("winsorize_pct must be in [0, 0.5)")
