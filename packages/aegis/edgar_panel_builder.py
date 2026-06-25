"""Build point-in-time EDGAR factor IC observations from cached fundamentals and prices."""

from __future__ import annotations

import calendar
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from aegis.edgar_full_universe_ic import EdgarIcObservation
from aegis.edgar_pit import EdgarFact, PitFundamentalStore, derive_fcf
from aegis.olympus_survivor_light import HistoricalConstituentStore, PriceBar


@dataclass(frozen=True)
class PanelBuildConfig:
    start: date
    end: date
    horizons_months: tuple[int, ...] = (1, 3, 6)


def month_end_dates(start: date, end: date) -> list[date]:
    cursor = date(start.year, start.month, 1)
    result: list[date] = []
    while cursor <= end:
        day = calendar.monthrange(cursor.year, cursor.month)[1]
        candidate = date(cursor.year, cursor.month, day)
        if start <= candidate <= end:
            result.append(candidate)
        cursor = _add_months(cursor, 1)
    return result


def historical_universe_symbols(
    constituent_store: HistoricalConstituentStore,
    dates: Sequence[date],
) -> set[str]:
    """Return the union of as-of members across rebalance dates.

    This is deliberately not the current constituent set. It is used by
    survivor-bias audits that need names which were removed before today.
    """

    symbols: set[str] = set()
    for as_of in dates:
        symbols.update(constituent_store.as_of(as_of))
    return symbols


def build_edgar_ic_panel(
    *,
    fundamentals: Mapping[str, PitFundamentalStore],
    prices: Mapping[str, Sequence[PriceBar]],
    constituent_store: HistoricalConstituentStore,
    config: PanelBuildConfig,
) -> tuple[list[EdgarIcObservation], dict[str, object]]:
    observations: list[EdgarIcObservation] = []
    sorted_prices = {
        symbol: tuple(sorted(symbol_bars, key=lambda bar: bar.date))
        for symbol, symbol_bars in prices.items()
    }
    rebalance_dates = month_end_dates(config.start, config.end)
    fact_snapshots = {
        symbol: _asof_snapshots(store, symbol=symbol, dates=rebalance_dates)
        for symbol, store in fundamentals.items()
    }
    skipped: dict[str, int] = {
        "not_in_price_cache": 0,
        "not_in_fundamentals": 0,
        "missing_entry_price": 0,
        "missing_forward_return": 0,
        "missing_factors": 0,
    }
    for as_of in rebalance_dates:
        for symbol in sorted(constituent_store.as_of(as_of)):
            price_bars = sorted_prices.get(symbol, ())
            if not price_bars:
                skipped["not_in_price_cache"] += 1
                continue
            store = fundamentals.get(symbol)
            if store is None:
                skipped["not_in_fundamentals"] += 1
                continue
            entry = first_price_after(price_bars, as_of)
            if entry is None:
                skipped["missing_entry_price"] += 1
                continue
            facts = fact_snapshots.get(symbol, {}).get(as_of, {})
            factors = factor_values_from_facts(facts, entry.close)
            if not factors:
                skipped["missing_factors"] += 1
                continue
            forward_returns = _forward_returns(
                price_bars,
                entry.close,
                as_of,
                config.horizons_months,
            )
            if not forward_returns:
                skipped["missing_forward_return"] += 1
                continue
            observations.append(
                EdgarIcObservation(
                    symbol=symbol,
                    as_of=as_of,
                    available_on=max(fact.available_on for fact in facts.values()),
                    factors=factors,
                    forward_returns=forward_returns,
                    in_universe=True,
                )
            )
    coverage: dict[str, object] = {
        "rebalance_periods_requested": len(rebalance_dates),
        "rebalance_start": rebalance_dates[0].isoformat() if rebalance_dates else None,
        "rebalance_end": rebalance_dates[-1].isoformat() if rebalance_dates else None,
        "observations": len(observations),
        "symbols_with_observations": len({row.symbol for row in observations}),
        "periods_with_observations": len({row.as_of for row in observations}),
        "skipped": skipped,
    }
    return observations, coverage


def factor_values_from_facts(facts: Mapping[str, EdgarFact], close: float) -> dict[str, float]:
    shares = _fact_value(facts, "CommonStockSharesOutstanding")
    if shares is None:
        shares = _fact_value(facts, "EntityCommonStockSharesOutstanding")
    if shares is None or shares <= 0.0 or close <= 0.0:
        return {}
    market_cap = shares * close
    net_income = _fact_value(facts, "NetIncomeLoss")
    equity = _fact_value(facts, "StockholdersEquity")
    revenue = _fact_value(facts, "Revenues")
    if revenue is None:
        revenue = _fact_value(facts, "SalesRevenueNet")
    gross_profit = _fact_value(facts, "GrossProfit")
    assets = _fact_value(facts, "Assets")
    operating_cash_flow = _fact_value(facts, "NetCashProvidedByUsedInOperatingActivities")
    fcf = derive_fcf(dict(facts))
    accruals = (
        None
        if net_income is None or operating_cash_flow is None
        else net_income - operating_cash_flow
    )
    accrual_ratio = _safe_div(accruals, assets)
    values = {
        "earnings_yield_ep": _safe_div(net_income, market_cap),
        "book_to_price_bp": _safe_div(equity, market_cap),
        "fcf_yield": _safe_div(fcf.value if fcf is not None else None, market_cap),
        "sales_to_price_sp": _safe_div(revenue, market_cap),
        "roe": _safe_div(net_income, equity),
        "gross_margin": _safe_div(gross_profit, revenue),
        "low_accruals": -accrual_ratio if accrual_ratio is not None else None,
        "asset_turnover": _safe_div(revenue, assets),
    }
    return {
        key: float(value)
        for key, value in values.items()
        if value is not None and math.isfinite(float(value))
    }


def _asof_snapshots(
    store: PitFundamentalStore,
    *,
    symbol: str,
    dates: Sequence[date],
) -> dict[date, dict[str, EdgarFact]]:
    symbol_upper = symbol.upper()
    facts = sorted(
        (fact for fact in store.facts if fact.ticker.upper() == symbol_upper),
        key=lambda fact: (fact.available_on, fact.filed, fact.period_end or date.min),
    )
    snapshots: dict[date, dict[str, EdgarFact]] = {}
    latest: dict[str, EdgarFact] = {}
    index = 0
    for as_of in sorted(dates):
        while index < len(facts) and facts[index].available_on <= as_of:
            fact = facts[index]
            current = latest.get(fact.concept)
            if current is None or _fact_recency_key(fact) > _fact_recency_key(current):
                latest[fact.concept] = fact
            index += 1
        snapshots[as_of] = dict(latest)
    return snapshots


def _fact_recency_key(fact: EdgarFact) -> tuple[date, date, str]:
    return (fact.filed, fact.period_end or date.min, fact.accession or "")


def first_price_after(bars: Sequence[PriceBar], query_date: date) -> PriceBar | None:
    eligible = [bar for bar in bars if bar.date > query_date]
    return min(eligible, key=lambda bar: bar.date) if eligible else None


def first_price_on_or_after(bars: Sequence[PriceBar], query_date: date) -> PriceBar | None:
    eligible = [bar for bar in bars if bar.date >= query_date]
    return min(eligible, key=lambda bar: bar.date) if eligible else None


def _forward_returns(
    bars: Sequence[PriceBar],
    entry_close: float,
    as_of: date,
    horizons_months: Sequence[int],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for months in horizons_months:
        exit_bar = first_price_on_or_after(bars, _add_months(as_of, months))
        if exit_bar is not None and entry_close > 0.0:
            result[f"{months}m"] = exit_bar.close / entry_close - 1.0
    return result


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _fact_value(facts: Mapping[str, EdgarFact], concept: str) -> float | None:
    fact = facts.get(concept)
    return None if fact is None else fact.value


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None
