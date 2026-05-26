from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title="market-data", version="0.1.0")

FIXTURES = {
    "BTCUSDT": {"symbol": "BTCUSDT", "price": "100000.00"},
    "ETHUSDT": {"symbol": "ETHUSDT", "price": "5000.00"},
}
CACHE_TTL_SECONDS = 5.0
_CACHE: dict[str, tuple[float, dict[str, str]]] = {}


def _fixtures_only() -> bool:
    return os.getenv("FIXTURES_ONLY", "false").lower() == "true"


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


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/ticker")
def ticker(symbol: str) -> dict[str, str]:
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
