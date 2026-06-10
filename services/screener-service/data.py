from __future__ import annotations

import importlib
import math
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, cast

CACHE_TTL_SEC = int(os.getenv("SCREENER_CACHE_TTL_SEC", "21600"))
MAX_SYMBOLS = int(os.getenv("SCREENER_MAX_SYMBOLS", "50"))
RETRY_COUNT = int(os.getenv("SCREENER_RETRY_COUNT", "2"))
REQUEST_TIMEOUT_SEC = float(os.getenv("SCREENER_REQUEST_TIMEOUT_SEC", "8"))

VALUATION_FIELDS = [
    "trailing_pe",
    "forward_pe",
    "peg",
    "price_to_book",
    "dividend_yield",
    "market_cap",
    "price",
]

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class ValuationError(RuntimeError):
    pass


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValuationError("symbol must be non-empty")
    if len(normalized) > 24:
        raise ValuationError(f"symbol too long: {normalized[:24]}")
    return normalized


def validate_universe(symbols: list[str]) -> list[str]:
    if not symbols:
        raise ValuationError("universe must contain at least one symbol")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = normalize_symbol(raw)
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    if len(normalized) > MAX_SYMBOLS:
        raise ValuationError(f"universe is limited to {MAX_SYMBOLS} symbols per request")
    return normalized


def _decimal_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(Decimal(str(value)).normalize())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = Decimal(stripped)
        except InvalidOperation:
            return None
        return str(parsed.normalize())
    return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _from_fast_info(fast_info: object, *names: str) -> object:
    for name in names:
        try:
            if hasattr(fast_info, "get"):
                value = fast_info.get(name)
            else:
                value = getattr(fast_info, name)
        except Exception:  # noqa: BLE001 - yfinance exposes multiple lazy shapes
            continue
        if value is not None:
            return value
    return None


def _request_with_retry(func: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - returned per symbol
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.35 * (attempt + 1))
    raise ValuationError(str(last_error) if last_error else "valuation request failed")


def _empty_row(symbol: str, *, error: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "name": None,
        "sector": None,
        "industry": None,
        "trailing_pe": None,
        "forward_pe": None,
        "peg": None,
        "price_to_book": None,
        "dividend_yield": None,
        "market_cap": None,
        "price": None,
        "error": error,
    }
    return row


def _load_symbol(symbol: str) -> dict[str, Any]:
    yf = importlib.import_module("yfinance")

    def call() -> dict[str, Any]:
        ticker = yf.Ticker(symbol)
        session = getattr(ticker, "session", None)
        if session is not None and hasattr(session, "timeout"):
            session.timeout = REQUEST_TIMEOUT_SEC
        info_raw = getattr(ticker, "info", {}) or {}
        info = info_raw if isinstance(info_raw, dict) else {}
        fast_info = getattr(ticker, "fast_info", {})
        row = _empty_row(symbol)
        row.update(
            {
                "name": _text(info.get("shortName") or info.get("longName")),
                "sector": _text(info.get("sector")),
                "industry": _text(info.get("industry")),
                "trailing_pe": _decimal_string(info.get("trailingPE")),
                "forward_pe": _decimal_string(info.get("forwardPE")),
                "peg": _decimal_string(info.get("pegRatio") or info.get("trailingPegRatio")),
                "price_to_book": _decimal_string(info.get("priceToBook")),
                "dividend_yield": _decimal_string(info.get("dividendYield")),
                "market_cap": _decimal_string(
                    info.get("marketCap") or _from_fast_info(fast_info, "market_cap", "marketCap")
                ),
                "price": _decimal_string(
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or _from_fast_info(fast_info, "last_price", "lastPrice")
                ),
            }
        )
        return row

    return cast(dict[str, Any], _request_with_retry(call))


def fetch_valuation(symbols: list[str]) -> list[dict[str, Any]]:
    normalized = validate_universe(symbols)
    now = time.time()
    rows: list[dict[str, Any]] = []
    for symbol in normalized:
        cached = _CACHE.get(symbol)
        if cached is not None and now - cached[0] <= CACHE_TTL_SEC:
            rows.append(dict(cached[1]))
            continue
        try:
            row = _load_symbol(symbol)
        except Exception as exc:  # noqa: BLE001 - one symbol must not fail the batch
            row = _empty_row(symbol, error=str(exc)[:240])
        _CACHE[symbol] = (now, dict(row))
        rows.append(row)
    return rows


def cache_stats() -> dict[str, Any]:
    return {"entries": len(_CACHE), "ttl_sec": CACHE_TTL_SEC, "max_symbols": MAX_SYMBOLS}
