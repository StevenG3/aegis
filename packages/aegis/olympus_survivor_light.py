from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, cast
from urllib.request import Request, urlopen

from aegis.edgar_pit import (
    DelistingAwarePriceSource,
    EdgarFact,
    PitFundamentalStore,
    derive_ebitda,
    derive_fcf,
    derive_net_debt,
)

SURVIVORSHIP_WARNING = (
    "survivor-light free prices may omit delisted securities; factor performance can be "
    "systematically optimistic and must not be used as edge evidence"
)
SURVIVORSHIP_LIGHT = "light"
ALLOWED_SURVIVOR_LIGHT_VERDICTS = {
    "SURVIVOR_LIGHT_PIPELINE_VALIDATED",
    "NO_EDGE",
    "INSUFFICIENT",
}
LOCKED_VALUE_QUALITY_FACTORS = (
    "earnings_yield_ep",
    "book_to_price_bp",
    "sales_to_price_sp",
    "ev_to_ebitda_inverse",
    "free_cash_flow_to_price",
    "roe",
    "gross_profitability",
    "accruals",
)
FULL_COST_DEFAULTS = {"commission_bps": 1.0, "slippage_bps": 5.0, "funding": "N/A"}
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class ConstituentChange:
    effective_date: date
    ticker: str
    action: Literal["add", "remove"]
    source: str
    note: str | None = None


@dataclass(frozen=True)
class CurrentConstituent:
    ticker: str
    name: str
    cik: str | None
    date_added: date | None


@dataclass(frozen=True)
class WikipediaConstituentSnapshot:
    as_of_date: date
    source_url: str
    current: dict[str, CurrentConstituent]
    changes: list[ConstituentChange]
    store: HistoricalConstituentStore
    caveat: str


@dataclass(frozen=True)
class PriceBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None
    volume: float


@dataclass(frozen=True)
class FactorObservation:
    ticker: str
    rebalance_date: date
    price_date: date
    trade_date: date
    close: float
    factors: dict[str, float]
    forward_return: float | None
    survivorship_status: str


class FreePriceSource:
    """Free survivor-light price adapter.

    CSV files are preferred for deterministic/private runs. If no CSV is present,
    yfinance is attempted lazily. Delisted tickers should be supplied by the caller
    so they are marked as missing instead of being mistaken for covered securities.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str = Path("data") / "olympus-free-prices",
        delisted_tickers: Iterable[str] = (),
        source_name: str = "free_csv_or_yfinance",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.delisted_tickers = {ticker.upper() for ticker in delisted_tickers}
        self.source_name = source_name

    def get_prices(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        ticker_upper = ticker.upper()
        status = self.survivorship_status(ticker_upper)
        if status["status"] == "delisted_price_missing_free_source":
            return {
                "ticker": ticker_upper,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "source": self.source_name,
                "survivorship": SURVIVORSHIP_LIGHT,
                "warning": SURVIVORSHIP_WARNING,
                "survivorship_status": status,
                "bars": [],
            }

        bars = self._load_csv(ticker_upper, start, end)
        if not bars:
            bars = self._load_yfinance(ticker_upper, start, end)
        return {
            "ticker": ticker_upper,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": self.source_name,
            "survivorship": SURVIVORSHIP_LIGHT,
            "warning": SURVIVORSHIP_WARNING,
            "survivorship_status": status,
            "bars": [bar_to_dict(bar) for bar in bars],
        }

    def survivorship_status(self, ticker: str) -> dict[str, Any]:
        ticker_upper = ticker.upper()
        if ticker_upper in self.delisted_tickers:
            return {
                "ticker": ticker_upper,
                "status": "delisted_price_missing_free_source",
                "reason": "free survivor-light source does not guarantee delisted price history",
            }
        return {
            "ticker": ticker_upper,
            "status": "active_or_unknown_free_source",
            "reason": "free source coverage is survivor-light and not delisting complete",
        }

    def _load_csv(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        path = self.cache_dir / f"{ticker}.csv"
        if not path.exists():
            return []
        bars: list[PriceBar] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed = _bar_from_row(row)
                if parsed is not None and start <= parsed.date <= end:
                    bars.append(parsed)
        return sorted(bars, key=lambda bar: bar.date)

    def _load_yfinance(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError:
            return []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if frame.empty:
            return []
        bars: list[PriceBar] = []
        for raw_index, row in frame.iterrows():
            bar_date = _coerce_date(raw_index)
            bars.append(
                PriceBar(
                    date=bar_date,
                    open=_numeric_cell(row["Open"]),
                    high=_numeric_cell(row["High"]),
                    low=_numeric_cell(row["Low"]),
                    close=_numeric_cell(row["Close"]),
                    adj_close=_numeric_cell(row["Adj Close"]) if "Adj Close" in row else None,
                    volume=_numeric_cell(row["Volume"]),
                )
            )
        _write_price_csv(self.cache_dir / f"{ticker}.csv", bars)
        return bars


SHARADAR_BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
SHARADAR_API_KEY_ENVS = (
    "NASDAQ_DATA_LINK_API_KEY",
    "SHARADAR_API_KEY",
    "QUANDL_API_KEY",
)


def sharadar_api_key() -> str | None:
    """Read a Sharadar/Nasdaq Data Link key from env; None when unset (free mode)."""
    for env in SHARADAR_API_KEY_ENVS:
        value = os.getenv(env, "").strip()
        if value:
            return value
    return None


def _datatable_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a Nasdaq Data Link datatable JSON into a list of column->value dicts."""
    table = payload.get("datatable")
    if not isinstance(table, dict):
        return []
    columns = [
        str(col.get("name"))
        for col in table.get("columns", [])
        if isinstance(col, dict) and col.get("name")
    ]
    rows: list[dict[str, Any]] = []
    for raw in table.get("data", []):
        if isinstance(raw, list) and len(raw) == len(columns):
            rows.append(dict(zip(columns, raw, strict=True)))
    return rows


class SharadarPriceSource:
    """Delisting-aware price source via Sharadar (Nasdaq Data Link REST).

    Survivorship-bias-free by design: the SEP table includes delisted tickers and the
    TICKERS table exposes ``isdelisted``. Requires a paid key in
    ``NASDAQ_DATA_LINK_API_KEY`` / ``SHARADAR_API_KEY``. Without a key it degrades
    gracefully (``status=paid_source_not_configured``, empty bars) and never raises,
    so callers can prefer it when configured and fall back to the free source
    otherwise. Only when it actually returns Sharadar data does it report
    ``survivorship=full`` (which can lift the survivor-light verdict ceiling in #41B-B).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: Path | str = Path("data") / "olympus-sharadar-prices",
        timeout_seconds: float = 20.0,
        http_get: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else sharadar_api_key()
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self._http_get = http_get  # injectable (url->dict) for deterministic tests

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _not_configured_payload(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        return {
            "ticker": ticker.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": "sharadar_sep",
            "survivorship": "unknown",
            "survivorship_status": self.survivorship_status(ticker),
            "status": "paid_source_not_configured",
            "reason": "set NASDAQ_DATA_LINK_API_KEY/SHARADAR_API_KEY to enable Sharadar",
            "bars": [],
        }

    def _get_json(self, url: str) -> dict[str, Any]:
        if self._http_get is not None:
            result = self._http_get(url)
            return result if isinstance(result, dict) else {}
        request = Request(url, headers={"User-Agent": "aegis-olympus-research/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            loaded = json.loads(response.read().decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    def get_prices(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        ticker_upper = ticker.upper()
        if not self.configured:
            return self._not_configured_payload(ticker_upper, start, end)
        status = self.survivorship_status(ticker_upper)
        url = (
            f"{SHARADAR_BASE_URL}/SEP.json?ticker={ticker_upper}"
            f"&date.gte={start.isoformat()}&date.lte={end.isoformat()}"
            f"&qopts.export=false&api_key={self.api_key}"
        )
        bars = sorted(
            (self._row_to_bar(row) for row in _datatable_rows(self._get_json(url))),
            key=lambda bar: bar.date,
        )
        return {
            "ticker": ticker_upper,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": "sharadar_sep",
            "survivorship": "full",
            "survivorship_status": status,
            "bars": [bar_to_dict(bar) for bar in bars],
        }

    def survivorship_status(self, ticker: str) -> dict[str, Any]:
        ticker_upper = ticker.upper()
        if not self.configured:
            return {
                "ticker": ticker_upper,
                "status": "paid_source_not_configured",
                "reason": "Sharadar key not set; using free survivor-light source instead",
            }
        url = f"{SHARADAR_BASE_URL}/TICKERS.json?ticker={ticker_upper}&api_key={self.api_key}"
        rows = _datatable_rows(self._get_json(url))
        row = rows[0] if rows else {}
        isdelisted = str(row.get("isdelisted", "")).upper()
        status = "delisted" if isdelisted == "Y" else "active" if isdelisted == "N" else "unknown"
        return {
            "ticker": ticker_upper,
            "status": status,
            "isdelisted": isdelisted or None,
            "first_price_date": row.get("firstpricedate"),
            "last_price_date": row.get("lastpricedate"),
            "source": "sharadar_tickers",
        }

    def _row_to_bar(self, row: dict[str, Any]) -> PriceBar:
        return PriceBar(
            date=_coerce_date(row.get("date")),
            open=_numeric_cell(row.get("open")),
            high=_numeric_cell(row.get("high")),
            low=_numeric_cell(row.get("low")),
            close=_numeric_cell(row.get("close")),
            adj_close=_numeric_cell(row.get("closeadj")) if "closeadj" in row else None,
            volume=_numeric_cell(row.get("volume")),
        )


def select_price_source(
    *,
    delisted_tickers: Iterable[str] = (),
    prefer_paid: bool = True,
) -> DelistingAwarePriceSource:
    """Return Sharadar when a paid key is configured, else the free survivor-light source.

    Lets the pipeline transparently upgrade to delisting-aware prices once a key is
    set, without code changes; until then it stays free/survivor-light.
    """
    if prefer_paid and sharadar_api_key():
        return SharadarPriceSource()
    return FreePriceSource(delisted_tickers=delisted_tickers)


class NorgatePriceSource:
    """Paid #41B slot: consume Windows NDU-exported CSV/parquet with delisting metadata."""

    def get_prices(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        raise NotImplementedError("Norgate paid price source is reserved for #41B")

    def survivorship_status(self, ticker: str) -> dict[str, Any]:
        raise NotImplementedError("Norgate paid survivorship status is reserved for #41B")


class HistoricalConstituentStore:
    def __init__(
        self,
        initial_members: Iterable[str],
        changes: Iterable[ConstituentChange],
    ) -> None:
        self.initial_members = {ticker.upper() for ticker in initial_members}
        self.changes = sorted(changes, key=lambda change: (change.effective_date, change.ticker))

    @classmethod
    def from_rows(
        cls,
        initial_members: Iterable[str],
        rows: Iterable[dict[str, Any]],
        *,
        source: str,
    ) -> HistoricalConstituentStore:
        changes: list[ConstituentChange] = []
        for row in rows:
            raw_action = str(row.get("action", "")).lower()
            if raw_action not in {"add", "remove"}:
                raise ValueError(f"unsupported constituent action: {raw_action}")
            changes.append(
                ConstituentChange(
                    effective_date=_parse_date(row["effective_date"]),
                    ticker=str(row["ticker"]).upper(),
                    action=cast(Literal["add", "remove"], raw_action),
                    source=source,
                    note=str(row["note"]) if row.get("note") is not None else None,
                )
            )
        return cls(initial_members, changes)

    @classmethod
    def from_current_members_and_changes(
        cls,
        current_members: Iterable[str],
        changes: Iterable[ConstituentChange],
    ) -> HistoricalConstituentStore:
        ordered = sorted(changes, key=lambda change: (change.effective_date, change.ticker))
        initial_members = {ticker.upper() for ticker in current_members}
        for change in reversed(ordered):
            if change.action == "add":
                initial_members.discard(change.ticker)
            else:
                initial_members.add(change.ticker)
        return cls(initial_members, ordered)

    def filtered(self, tickers: Iterable[str]) -> HistoricalConstituentStore:
        allowed = {ticker.upper() for ticker in tickers}
        return HistoricalConstituentStore(
            self.initial_members & allowed,
            [change for change in self.changes if change.ticker in allowed],
        )

    def as_of(self, query_date: date | str) -> set[str]:
        parsed = _parse_date(query_date)
        members = set(self.initial_members)
        for change in self.changes:
            if change.effective_date > parsed:
                break
            if change.action == "add":
                members.add(change.ticker)
            else:
                members.discard(change.ticker)
        return members

    def caveat(self) -> str:
        return (
            "free historical constituents are point-in-time best effort; community/Wikipedia "
            "change dates can be incomplete or imprecise and require paid-source validation"
        )


def download_wikipedia_sp500_html(
    *,
    cache_path: Path | str | None = None,
    url: str = WIKIPEDIA_SP500_URL,
    user_agent: str = "AegisOlympusResearch/0.1",
    timeout_seconds: float = 20.0,
) -> str:
    output = Path(cache_path) if cache_path is not None else None
    if output is not None and output.exists():
        return output.read_text(encoding="utf-8")
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        html: str = response.read().decode("utf-8", errors="replace")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
    return html


def parse_wikipedia_sp500_snapshot(
    html: str,
    *,
    as_of_date: date | None = None,
    source_url: str = WIKIPEDIA_SP500_URL,
) -> WikipediaConstituentSnapshot:
    parsed_as_of = as_of_date or date.today()
    tables = _extract_html_tables(html)
    current = _parse_wikipedia_current_constituents(tables)
    changes = [
        change
        for change in _parse_wikipedia_constituent_changes(tables)
        if change.effective_date <= parsed_as_of
    ]
    store = HistoricalConstituentStore.from_current_members_and_changes(
        current.keys(),
        changes,
    )
    return WikipediaConstituentSnapshot(
        as_of_date=parsed_as_of,
        source_url=source_url,
        current=current,
        changes=changes,
        store=store,
        caveat=(
            "Wikipedia S&P 500 constituents and changes are free community data; "
            "effective dates, corporate action mapping, and delisted coverage require "
            "paid-source validation before any robust edge conclusion"
        ),
    )


def align_pit_fundamentals_with_prices(
    *,
    fundamentals: PitFundamentalStore,
    price_source: FreePriceSource,
    constituent_store: HistoricalConstituentStore,
    rebalance_dates: list[date],
) -> list[FactorObservation]:
    prices_by_ticker: dict[str, list[PriceBar]] = {}
    observations_without_labels: list[FactorObservation] = []
    for rebalance_date in sorted(rebalance_dates):
        universe = constituent_store.as_of(rebalance_date)
        for ticker in sorted(universe):
            prices = prices_by_ticker.setdefault(
                ticker,
                _bars_from_payload(
                    price_source.get_prices(
                        ticker,
                        min(rebalance_dates) - timedelta(days=7),
                        max(rebalance_dates) + timedelta(days=14),
                    )
                ),
            )
            price = latest_price_on_or_before(prices, rebalance_date)
            trade_price = first_price_after(prices, rebalance_date)
            if price is None or trade_price is None:
                continue
            facts = fundamentals.as_of(ticker, rebalance_date)
            factors = compute_locked_factors(facts, price.close)
            if not factors:
                continue
            status = price_source.survivorship_status(ticker)["status"]
            observations_without_labels.append(
                FactorObservation(
                    ticker=ticker,
                    rebalance_date=rebalance_date,
                    price_date=price.date,
                    trade_date=trade_price.date,
                    close=price.close,
                    factors=factors,
                    forward_return=None,
                    survivorship_status=status,
                )
            )
    return with_forward_returns(observations_without_labels)


def compute_locked_factors(facts: dict[str, EdgarFact], close: float) -> dict[str, float]:
    shares = _fact_value(facts, "CommonStockSharesOutstanding")
    if shares is None:
        shares = _fact_value(facts, "EntityCommonStockSharesOutstanding")
    if shares is None or shares <= 0 or close <= 0:
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
    accrual_numerator = (
        None
        if net_income is None or operating_cash_flow is None
        else net_income - operating_cash_flow
    )
    ebitda = derive_ebitda(facts)
    free_cash_flow = derive_fcf(facts)
    net_debt = derive_net_debt(facts)
    enterprise_value = market_cap + (net_debt.value if net_debt is not None else 0.0)
    return _drop_none(
        {
            "earnings_yield_ep": _safe_div(net_income, market_cap),
            "book_to_price_bp": _safe_div(equity, market_cap),
            "sales_to_price_sp": _safe_div(revenue, market_cap),
            "ev_to_ebitda_inverse": _safe_div(
                ebitda.value if ebitda is not None else None, enterprise_value
            ),
            "free_cash_flow_to_price": _safe_div(
                free_cash_flow.value if free_cash_flow is not None else None, market_cap
            ),
            "roe": _safe_div(net_income, equity),
            "gross_profitability": _safe_div(gross_profit, assets),
            "accruals": _safe_div(accrual_numerator, assets),
        }
    )


def evaluate_survivor_light_ic(
    observations: list[FactorObservation],
    *,
    benchmark_symbol: str | None = None,
    commission_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> dict[str, Any]:
    cost = (commission_bps + slippage_bps) * 2.0 / 10_000.0
    factor_reports: dict[str, Any] = {}
    for factor in LOCKED_VALUE_QUALITY_FACTORS:
        rows = [
            row
            for row in observations
            if row.forward_return is not None and factor in row.factors
        ]
        rank_ics = _cross_sectional_rank_ic(rows, factor, cost)
        factor_reports[factor] = {
            "survivorship": SURVIVORSHIP_LIGHT,
            "warning": SURVIVORSHIP_WARNING,
            "rank_ic": _series_summary(rank_ics),
            "fdr_bh": None,
            "monotonicity": _quantile_monotonicity(rows, factor, cost),
            "walk_forward": _walk_forward_ic(rank_ics),
            "full_cost": {
                "commission_bps": commission_bps,
                "slippage_bps": slippage_bps,
                "round_trip_cost_return": round(cost, 8),
                "funding": "N/A",
            },
        }
    _attach_bh_fdr(factor_reports)
    metrics = _standard_metrics(observations, cost, benchmark_symbol)
    verdict = _survivor_light_verdict(factor_reports, observations)
    return {
        "status": "OK" if observations else "INSUFFICIENT_DATA",
        "verdict": verdict,
        "survivorship": SURVIVORSHIP_LIGHT,
        "warning": SURVIVORSHIP_WARNING,
        "factors": factor_reports,
        "benchmarks": metrics["benchmarks"],
        "standard_metrics": metrics["portfolio"],
        "allowed_verdicts": sorted(ALLOWED_SURVIVOR_LIGHT_VERDICTS),
        "disclaimer": (
            "pipeline validation only; no live trading, no plugin registration, no alpha claim"
        ),
    }


def write_sanitized_pipeline_status(path: Path | str, report: dict[str, Any]) -> None:
    sanitized = {
        "status": report.get("status"),
        "verdict": report.get("verdict"),
        "survivorship": SURVIVORSHIP_LIGHT,
        "warning": SURVIVORSHIP_WARNING,
        "private_results": "redacted; stored outside public aegis repository",
        "allowed_verdicts": sorted(ALLOWED_SURVIVOR_LIGHT_VERDICTS),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")


def bar_to_dict(bar: PriceBar) -> dict[str, Any]:
    return {
        "date": bar.date.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "adj_close": bar.adj_close,
        "volume": bar.volume,
    }


def latest_price_on_or_before(bars: list[PriceBar], query_date: date) -> PriceBar | None:
    eligible = [bar for bar in bars if bar.date <= query_date]
    return max(eligible, key=lambda bar: bar.date) if eligible else None


def first_price_after(bars: list[PriceBar], query_date: date) -> PriceBar | None:
    eligible = [bar for bar in bars if bar.date > query_date]
    return min(eligible, key=lambda bar: bar.date) if eligible else None


def with_forward_returns(observations: list[FactorObservation]) -> list[FactorObservation]:
    by_ticker: dict[str, list[FactorObservation]] = {}
    for row in observations:
        by_ticker.setdefault(row.ticker, []).append(row)
    result: list[FactorObservation] = []
    for rows in by_ticker.values():
        ordered = sorted(rows, key=lambda row: row.rebalance_date)
        for index, row in enumerate(ordered):
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            result.append(
                FactorObservation(
                    ticker=row.ticker,
                    rebalance_date=row.rebalance_date,
                    price_date=row.price_date,
                    trade_date=row.trade_date,
                    close=row.close,
                    factors=row.factors,
                    forward_return=(
                        None if next_row is None else next_row.close / row.close - 1.0
                    ),
                    survivorship_status=row.survivorship_status,
                )
            )
    return sorted(result, key=lambda row: (row.rebalance_date, row.ticker))


def _bar_from_row(row: dict[str, str]) -> PriceBar | None:
    raw_date = row.get("Date") or row.get("date")
    if raw_date is None:
        return None
    return PriceBar(
        date=_parse_date(raw_date),
        open=float(row.get("Open") or row.get("open") or 0),
        high=float(row.get("High") or row.get("high") or 0),
        low=float(row.get("Low") or row.get("low") or 0),
        close=float(row.get("Close") or row.get("close") or 0),
        adj_close=_optional_float(row.get("Adj Close") or row.get("adj_close")),
        volume=float(row.get("Volume") or row.get("volume") or 0),
    )


def _write_price_csv(path: Path, bars: list[PriceBar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "Date": bar.date.isoformat(),
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Adj Close": bar.adj_close if bar.adj_close is not None else "",
                    "Volume": bar.volume,
                }
            )


def _bars_from_payload(payload: dict[str, Any]) -> list[PriceBar]:
    bars = payload.get("bars")
    if not isinstance(bars, list):
        return []
    result: list[PriceBar] = []
    for item in bars:
        if isinstance(item, dict):
            result.append(
                PriceBar(
                    date=_parse_date(item["date"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    adj_close=_optional_float(item.get("adj_close")),
                    volume=float(item["volume"]),
                )
            )
    return result


def _cross_sectional_rank_ic(
    rows: list[FactorObservation],
    factor: str,
    cost: float,
) -> list[float]:
    by_date: dict[date, list[FactorObservation]] = {}
    for row in rows:
        by_date.setdefault(row.rebalance_date, []).append(row)
    values: list[float] = []
    for date_rows in by_date.values():
        if len(date_rows) < 3:
            continue
        factor_values = [row.factors[factor] for row in date_rows]
        returns = [cast(float, row.forward_return) - cost for row in date_rows]
        corr = _pearson(_ranks(factor_values), _ranks(returns))
        if corr is not None:
            values.append(corr)
    return values


def _quantile_monotonicity(
    rows: list[FactorObservation],
    factor: str,
    cost: float,
    groups: int = 5,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.forward_return is not None and factor in row.factors]
    if len(eligible) < groups:
        return {"status": "INSUFFICIENT_DATA", "is_monotonic": None, "groups": groups}
    ordered = sorted(eligible, key=lambda row: row.factors[factor])
    buckets: list[float] = []
    for index in range(groups):
        start = math.floor(index * len(ordered) / groups)
        end = math.floor((index + 1) * len(ordered) / groups)
        bucket = ordered[start:end]
        returns = [cast(float, row.forward_return) - cost for row in bucket]
        buckets.append(round(statistics.fmean(returns), 8))
    increasing = all(left <= right for left, right in zip(buckets, buckets[1:], strict=False))
    decreasing = all(left >= right for left, right in zip(buckets, buckets[1:], strict=False))
    return {
        "status": "OK",
        "groups": groups,
        "bucket_mean_returns": buckets,
        "is_monotonic": increasing or decreasing,
        "top_bottom_return": round(buckets[-1] - buckets[0], 8),
    }


def _walk_forward_ic(values: list[float]) -> dict[str, Any]:
    if len(values) < 4:
        return {"status": "INSUFFICIENT_DATA", "windows": 0}
    midpoint = len(values) // 2
    train = values[:midpoint]
    test = values[midpoint:]
    return {
        "status": "OK",
        "windows": 2,
        "in_sample_mean_rank_ic": round(statistics.fmean(train), 8),
        "out_of_sample_mean_rank_ic": round(statistics.fmean(test), 8),
    }


def _series_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "status": "INSUFFICIENT_DATA",
            "n": 0,
            "mean": None,
            "std": None,
            "icir": None,
            "t_value": None,
            "positive_share": None,
            "p_value_approx": None,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    icir = None if std == 0 else mean / std
    t_value = None if std == 0 else mean / (std / math.sqrt(len(values)))
    return {
        "status": "OK",
        "n": len(values),
        "mean": round(mean, 8),
        "std": round(std, 8),
        "icir": None if icir is None else round(icir, 8),
        "t_value": None if t_value is None else round(t_value, 8),
        "positive_share": round(sum(1 for value in values if value > 0) / len(values), 8),
        "p_value_approx": None if t_value is None else round(_normal_two_sided_p(abs(t_value)), 8),
    }


def _attach_bh_fdr(factor_reports: dict[str, Any]) -> None:
    p_values: list[tuple[str, float]] = []
    for name, report in factor_reports.items():
        p_value = report["rank_ic"].get("p_value_approx")
        if isinstance(p_value, float):
            p_values.append((name, p_value))
    total = len(p_values)
    for rank, (name, p_value) in enumerate(sorted(p_values, key=lambda item: item[1]), start=1):
        factor_reports[name]["fdr_bh"] = {
            "rank": rank,
            "q_value_approx": round(min(p_value * total / rank, 1.0), 8),
            "method": "Benjamini-Hochberg approximate two-sided normal p-value",
        }


def _standard_metrics(
    observations: list[FactorObservation],
    cost: float,
    benchmark_symbol: str | None,
) -> dict[str, Any]:
    net_returns = [
        row.forward_return - cost for row in observations if row.forward_return is not None
    ]
    by_date: dict[date, list[float]] = {}
    benchmark: list[float] = []
    for row in observations:
        if row.forward_return is None:
            continue
        by_date.setdefault(row.rebalance_date, []).append(row.forward_return - cost)
        if benchmark_symbol is not None and row.ticker == benchmark_symbol:
            benchmark.append(row.forward_return)
    equal_weight = [statistics.fmean(values) for values in by_date.values()]
    return {
        "portfolio": _metric_block(net_returns),
        "benchmarks": {
            "equal_weight": _metric_block(equal_weight),
            "index": None if not benchmark else _metric_block(benchmark),
        },
    }


def _metric_block(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "status": "INSUFFICIENT_DATA",
            "total_return": None,
            "max_drawdown": None,
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "positive_period_win_rate": None,
            "oos_window_win_rate_vs_status_quo": None,
            "annualized_turnover": None,
            "net_cost": None,
        }
    compounded = 1.0
    curve: list[float] = []
    for value in values:
        compounded *= 1.0 + value
        curve.append(compounded)
    max_drawdown = _max_drawdown(curve)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    downside = [min(value, 0.0) for value in values]
    downside_std = statistics.stdev(downside) if len(downside) > 1 else 0.0
    annualized_return = (compounded ** (252 / max(len(values), 1))) - 1.0
    return {
        "status": "OK",
        "total_return": round(compounded - 1.0, 8),
        "max_drawdown": round(max_drawdown, 8),
        "sharpe": None if std == 0 else round(mean / std * math.sqrt(252), 8),
        "sortino": None if downside_std == 0 else round(mean / downside_std * math.sqrt(252), 8),
        "calmar": None if max_drawdown == 0 else round(annualized_return / abs(max_drawdown), 8),
        "positive_period_win_rate": round(sum(1 for value in values if value > 0) / len(values), 8),
        "oos_window_win_rate_vs_status_quo": None,
        "annualized_turnover": 12.0,
        "net_cost": None,
    }


def _survivor_light_verdict(
    factor_reports: dict[str, Any],
    observations: list[FactorObservation],
) -> str:
    if not observations:
        return "INSUFFICIENT"
    if any(report["rank_ic"]["status"] == "OK" for report in factor_reports.values()):
        return "SURVIVOR_LIGHT_PIPELINE_VALIDATED"
    return "INSUFFICIENT"


def _fact_value(facts: dict[str, EdgarFact], concept: str) -> float | None:
    fact = facts.get(concept)
    return None if fact is None else fact.value


def _drop_none(values: dict[str, float | None]) -> dict[str, float]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and math.isfinite(value)
    }


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in indexed[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0:
        return None
    return numerator / denominator


def _max_drawdown(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _normal_two_sided_p(z_value: float) -> float:
    return math.erfc(z_value / math.sqrt(2.0))


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_stack = 0
        self._current_table: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._table_stack += 1
            if self._table_stack == 1:
                self._current_table = []
        elif self._table_stack and tag == "tr":
            self._current_row = []
        elif self._table_stack and tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_stack and tag in {"td", "th"} and self._current_cell is not None:
            text = _clean_wiki_text("".join(self._current_cell))
            if self._current_row is not None:
                self._current_row.append(text)
            self._current_cell = None
        elif self._table_stack and tag == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_stack:
            self._table_stack -= 1
            if self._table_stack == 0:
                self.tables.append(self._current_table)
                self._current_table = []


def _extract_html_tables(html: str) -> list[list[list[str]]]:
    parser = _HtmlTableParser()
    parser.feed(html)
    return parser.tables


def _parse_wikipedia_current_constituents(
    tables: list[list[list[str]]],
) -> dict[str, CurrentConstituent]:
    for table in tables:
        if not table:
            continue
        header = table[0]
        if {"Symbol", "Security", "CIK"}.issubset(set(header)):
            symbol_index = header.index("Symbol")
            security_index = header.index("Security")
            cik_index = header.index("CIK")
            date_added_index = header.index("Date added") if "Date added" in header else None
            current: dict[str, CurrentConstituent] = {}
            for row in table[1:]:
                if len(row) <= max(symbol_index, security_index, cik_index):
                    continue
                ticker = _normalize_ticker(row[symbol_index])
                if not ticker:
                    continue
                raw_date = row[date_added_index] if date_added_index is not None else ""
                current[ticker] = CurrentConstituent(
                    ticker=ticker,
                    name=row[security_index],
                    cik=_normalize_cik(row[cik_index]),
                    date_added=_parse_optional_date(raw_date),
                )
            if current:
                return current
    raise ValueError("could not locate Wikipedia S&P 500 current constituents table")


def _parse_wikipedia_constituent_changes(
    tables: list[list[list[str]]],
) -> list[ConstituentChange]:
    for table in tables:
        if not table:
            continue
        if table[0][:4] != ["Effective Date", "Added", "Removed", "Reason"]:
            continue
        changes: list[ConstituentChange] = []
        for row in table[1:]:
            if len(row) < 6 or row[:4] == ["Ticker", "Security", "Ticker", "Security"]:
                continue
            effective_date = _parse_constituent_date(row[0])
            added = _normalize_ticker(row[1])
            removed = _normalize_ticker(row[3])
            reason = row[5] if len(row) > 5 and row[5] else None
            if added:
                changes.append(
                    ConstituentChange(
                        effective_date=effective_date,
                        ticker=added,
                        action="add",
                        source="wikipedia_sp500_changes",
                        note=reason,
                    )
                )
            if removed:
                changes.append(
                    ConstituentChange(
                        effective_date=effective_date,
                        ticker=removed,
                        action="remove",
                        source="wikipedia_sp500_changes",
                        note=reason,
                    )
                )
        if changes:
            return sorted(changes, key=lambda change: (change.effective_date, change.ticker))
    raise ValueError("could not locate Wikipedia S&P 500 constituent changes table")


def _clean_wiki_text(value: str) -> str:
    return " ".join(unescape(value).replace("\xa0", " ").split())


def _normalize_ticker(value: str) -> str:
    cleaned = _clean_wiki_text(value).upper()
    if cleaned in {"", "—", "-", "N/A"}:
        return ""
    return cleaned.replace(".", "-")


def _normalize_cik(value: str) -> str | None:
    digits = "".join(character for character in value if character.isdigit())
    return digits.lstrip("0") or None


def _parse_optional_date(value: str) -> date | None:
    cleaned = _clean_wiki_text(value)
    if not cleaned:
        return None
    try:
        return _parse_constituent_date(cleaned)
    except ValueError:
        return None


def _parse_constituent_date(value: str) -> date:
    cleaned = _clean_wiki_text(value)
    for pattern in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned[:32], pattern).date()
        except ValueError:
            continue
    return _parse_date(cleaned)


def _parse_date(value: date | str | Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _coerce_date(value: Any) -> date:
    if hasattr(value, "date"):
        parsed = value.date()
        if isinstance(parsed, date):
            return parsed
    return _parse_date(value)


def _numeric_cell(value: Any) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"cannot convert value to float: {value!r}")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
