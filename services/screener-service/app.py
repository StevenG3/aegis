from __future__ import annotations

from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import data as data_module
from data import VALUATION_FIELDS, ValuationError, cache_stats, validate_universe

DISCLAIMER = "valuation screen, candidates only, not a buy signal"
SortField = Literal[
    "symbol",
    "trailing_pe",
    "forward_pe",
    "peg",
    "price_to_book",
    "dividend_yield",
    "market_cap",
    "price",
]

SECTOR_UNIVERSES: dict[str, list[str]] = {
    "mega_cap_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "QCOM", "MU", "TSM", "ASML"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV"],
    "consumer": ["WMT", "COST", "HD", "MCD", "NKE", "SBUX"],
}

app = FastAPI(title="screener-service", version="0.1.0")


class ScreenFilters(BaseModel):
    max_pe: Decimal | None = Field(default=None, ge=0)
    min_pe: Decimal | None = Field(default=None, ge=0)
    max_forward_pe: Decimal | None = Field(default=None, ge=0)
    max_peg: Decimal | None = Field(default=None, ge=0)
    max_price_to_book: Decimal | None = Field(default=None, ge=0)
    min_div_yield: Decimal | None = Field(default=None, ge=0)
    min_market_cap: Decimal | None = Field(default=None, ge=0)


class ScreenRequest(BaseModel):
    universe: list[str] = Field(default_factory=list)
    filters: ScreenFilters = Field(default_factory=ScreenFilters)
    sort_by: SortField = "trailing_pe"
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("universe")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        try:
            return validate_universe(value)
        except ValuationError as exc:
            raise ValueError(str(exc)) from exc


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _passes_filters(row: dict[str, Any], filters: ScreenFilters) -> bool:
    if row.get("error"):
        return True
    trailing_pe = _decimal(row.get("trailing_pe"))
    forward_pe = _decimal(row.get("forward_pe"))
    peg = _decimal(row.get("peg"))
    price_to_book = _decimal(row.get("price_to_book"))
    dividend_yield = _decimal(row.get("dividend_yield"))
    market_cap = _decimal(row.get("market_cap"))
    checks = [
        filters.max_pe is None or (trailing_pe is not None and trailing_pe <= filters.max_pe),
        filters.min_pe is None or (trailing_pe is not None and trailing_pe >= filters.min_pe),
        filters.max_forward_pe is None
        or (forward_pe is not None and forward_pe <= filters.max_forward_pe),
        filters.max_peg is None or (peg is not None and peg <= filters.max_peg),
        filters.max_price_to_book is None
        or (price_to_book is not None and price_to_book <= filters.max_price_to_book),
        filters.min_div_yield is None
        or (dividend_yield is not None and dividend_yield >= filters.min_div_yield),
        filters.min_market_cap is None
        or (market_cap is not None and market_cap >= filters.min_market_cap),
    ]
    return all(checks)


def _sort_key(sort_by: str, row: dict[str, Any]) -> tuple[int, Decimal | str]:
    value = row.get(sort_by)
    if sort_by == "symbol":
        return (0, str(row.get("symbol") or ""))
    parsed = _decimal(value)
    if parsed is None:
        return (1, Decimal("0"))
    return (0, parsed)


def _plain_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _median_string(values: list[Decimal]) -> str | None:
    if not values:
        return None
    return _plain_decimal(median(values))


def aggregate_sectors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sector = str(row.get("sector") or "Unknown")
        buckets.setdefault(sector, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for sector, sector_rows in buckets.items():
        trailing_values = [
            value for row in sector_rows if (value := _decimal(row.get("trailing_pe"))) is not None
        ]
        forward_values = [
            value for row in sector_rows if (value := _decimal(row.get("forward_pe"))) is not None
        ]
        pb_values = [
            value
            for row in sector_rows
            if (value := _decimal(row.get("price_to_book"))) is not None
        ]
        aggregates.append(
            {
                "sector": sector,
                "count": len(sector_rows),
                "valid_trailing_pe_count": len(trailing_values),
                "median_trailing_pe": _median_string(trailing_values),
                "median_forward_pe": _median_string(forward_values),
                "median_price_to_book": _median_string(pb_values),
            }
        )
    return sorted(
        aggregates,
        key=lambda item: (
            item["median_trailing_pe"] is None,
            _decimal(item["median_trailing_pe"]) or Decimal("0"),
            str(item["sector"]),
        ),
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    return {"status": "ok", "provider": "yfinance", "cache": cache_stats()}


@app.get("/sectors")
def sectors() -> dict[str, Any]:
    return {"disclaimer": DISCLAIMER, "universes": SECTOR_UNIVERSES}


@app.post("/screen")
def screen(request: ScreenRequest) -> dict[str, Any]:
    try:
        rows = data_module.fetch_valuation(request.universe)
    except ValuationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_UNIVERSE", "message": str(exc)},
        ) from exc
    filtered = [row for row in rows if _passes_filters(row, request.filters)]
    filtered.sort(key=lambda row: _sort_key(request.sort_by, row))
    limited = filtered[: request.limit]
    return {
        "disclaimer": DISCLAIMER,
        "universe": request.universe,
        "filters": request.filters.model_dump(mode="json"),
        "sort_by": request.sort_by,
        "limit": request.limit,
        "fields": VALUATION_FIELDS,
        "valuations": limited,
        "sectors": aggregate_sectors(filtered),
        "errors": [row for row in rows if row.get("error")],
    }
