from __future__ import annotations

import os
import time
from decimal import Decimal, InvalidOperation

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title="market-data", version="0.1.0")

FIXTURES = {
    "BTCUSDT": {"symbol": "BTCUSDT", "price": "100000.00"},
    "ETHUSDT": {"symbol": "ETHUSDT", "price": "5000.00"},
}
CACHE_TTL_SECONDS = 5.0
_CACHE: dict[str, tuple[float, dict[str, str]]] = {}
STOCK_QUOTE_CACHE_TTL_SEC = float(os.getenv("STOCK_QUOTE_CACHE_TTL_SEC", "60"))
_STOCK_QUOTE_CACHE: dict[str, tuple[float, dict[str, str]]] = {}
_STOCK_FIXTURES = {
    "NVDA": "450.00",
    "MSFT": "420.00",
    "AAPL": "190.00",
    "GOOGL": "175.00",
    "AMZN": "185.00",
    "META": "510.00",
    "TSLA": "240.00",
    "SPY": "550.00",
    "QQQ": "470.00",
    "IWM": "215.00",
}


def _fixtures_only() -> bool:
    return os.getenv("FIXTURES_ONLY", "false").lower() == "true"


def _stock_provider_chain() -> list[str]:
    return [
        provider.strip()
        for provider in os.getenv(
            "STOCK_QUOTE_PROVIDER_CHAIN", "ibkr,polygon,yahoo,fixture"
        ).split(",")
        if provider.strip()
    ]


def _fixture(symbol: str, source: str) -> dict[str, str]:
    if symbol not in FIXTURES:
        raise HTTPException(status_code=404, detail={"code": "SYMBOL_NOT_FOUND"})
    fixture = FIXTURES[symbol]
    return {"symbol": fixture["symbol"], "price": fixture["price"], "source": source}


def _ticker(symbol: str) -> dict[str, str]:
    normalized = symbol.upper()
    if _fixtures_only():
        return _fixture(normalized, "fixture")

    now = time.time()
    cached = _CACHE.get(normalized)
    if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        response = httpx.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": normalized},
            timeout=3.0,
        )
        response.raise_for_status()
        body = response.json()
        result = {"symbol": normalized, "price": str(body["price"]), "source": "binance"}
        _CACHE[normalized] = (now, result)
        return result
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        if normalized in FIXTURES:
            return _fixture(normalized, "fixture_fallback")
        raise HTTPException(
            status_code=503, detail={"code": "MARKET_DATA_UNAVAILABLE"}
        ) from exc


def _stock_fixture(symbol: str) -> dict[str, str]:
    return {
        "symbol": symbol,
        "price": _STOCK_FIXTURES.get(symbol, "100.00"),
        "source": "fixture",
        "asset_type": "stock",
    }


def _polygon_fetch(symbol: str) -> dict[str, str] | None:
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return None
    try:
        response = httpx.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
            params={"adjusted": "true", "apiKey": api_key},
            timeout=5.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    results = body.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    close = first.get("c")
    if close is None:
        return None
    try:
        price = Decimal(str(close))
    except (InvalidOperation, ValueError):
        return None
    if price <= 0:
        return None
    return {"symbol": symbol, "price": str(price), "source": "polygon", "asset_type": "stock"}


def _ibkr_fetch(symbol: str) -> dict[str, str] | None:
    bridge_url = os.getenv("IBKR_BRIDGE_URL", "http://ibkr-bridge:8086").rstrip("/")
    try:
        response = httpx.get(f"{bridge_url}/tickers/{symbol}", timeout=5.0)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    price_raw = body.get("price")
    if price_raw is None:
        return None
    try:
        price = Decimal(str(price_raw))
    except (InvalidOperation, ValueError):
        return None
    if price <= 0:
        return None
    return {"symbol": symbol, "price": str(price), "source": "ibkr", "asset_type": "stock"}


def _yahoo_fetch(symbol: str) -> dict[str, str] | None:
    try:
        response = httpx.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"interval": "1d", "range": "1d"},
            timeout=5.0,
            headers={"user-agent": "aegis/1.0"},
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    chart = body.get("chart")
    if not isinstance(chart, dict):
        return None
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    meta = first.get("meta") if isinstance(first, dict) else None
    if not isinstance(meta, dict):
        return None
    price_raw = meta.get("regularMarketPrice")
    if price_raw is None:
        return None
    try:
        price = Decimal(str(price_raw))
    except (InvalidOperation, ValueError):
        return None
    if price <= 0:
        return None
    return {"symbol": symbol, "price": str(price), "source": "yahoo", "asset_type": "stock"}


def _fetch_stock_provider(provider_name: str, symbol: str) -> dict[str, str] | None:
    if provider_name == "ibkr":
        return _ibkr_fetch(symbol)
    if provider_name == "polygon":
        return _polygon_fetch(symbol)
    if provider_name == "yahoo":
        return _yahoo_fetch(symbol)
    if provider_name == "fixture":
        return _stock_fixture(symbol)
    return None


def _stock_ticker(symbol: str) -> dict[str, str]:
    normalized = symbol.upper().strip()
    if not normalized:
        return _stock_fixture(normalized)
    now = time.monotonic()
    cached = _STOCK_QUOTE_CACHE.get(normalized)
    if cached is not None and now - cached[0] < STOCK_QUOTE_CACHE_TTL_SEC:
        return cached[1]
    if _fixtures_only():
        result = _stock_fixture(normalized)
        _STOCK_QUOTE_CACHE[normalized] = (now, result)
        return result
    for provider in _stock_provider_chain():
        candidate = _fetch_stock_provider(provider, normalized)
        if candidate is not None:
            _STOCK_QUOTE_CACHE[normalized] = (now, candidate)
            return candidate
    fallback = _stock_fixture(normalized)
    _STOCK_QUOTE_CACHE[normalized] = (now, fallback)
    return fallback


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/ticker")
def ticker(symbol: str, asset_type: str = "crypto") -> dict[str, str]:
    if asset_type == "stock":
        return _stock_ticker(symbol)
    return _ticker(symbol)


@app.get("/quotes")
def quotes(symbols: str) -> dict[str, dict[str, str]]:
    requested = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    if len(requested) > 10:
        raise HTTPException(status_code=400, detail={"code": "TOO_MANY_SYMBOLS"})
    return {
        symbol: {k: v for k, v in _ticker(symbol).items() if k != "symbol"}
        for symbol in requested
    }
