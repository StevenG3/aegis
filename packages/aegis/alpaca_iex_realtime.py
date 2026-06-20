"""Read-only Alpaca IEX latest quote adapter.

This module intentionally uses only Alpaca's Market Data API host. It does not
touch Alpaca Trading API endpoints, accounts, orders, wallets, or balances.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, cast

import httpx

ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
ALPACA_IEX_FEED = "iex"
ALPACA_KEY_NOT_CONFIGURED = "alpaca_key_not_configured"


class _HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any: ...


class _HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str],
        timeout: float,
    ) -> _HttpResponse: ...


class AlpacaIexRealtimeQuoteSource:
    """Read-only latest US equity quote source backed by Alpaca's IEX feed."""

    source = "alpaca"
    feed = ALPACA_IEX_FEED
    market = "us_equity"
    read_only = True

    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        http_client: _HttpClient | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.api_key_id = (
            api_key_id
            if api_key_id is not None
            else os.getenv("ALPACA_API_KEY_ID", "").strip()
        )
        self.api_secret_key = (
            api_secret_key
            if api_secret_key is not None
            else os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        )
        self._http_client = http_client
        self.timeout_sec = timeout_sec

    @property
    def configured(self) -> bool:
        return bool(self.api_key_id and self.api_secret_key)

    def capability_matrix(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "host": ALPACA_DATA_BASE_URL,
            "feed": self.feed,
            "market": self.market,
            "coverage": "US stocks and ETFs available on Alpaca's IEX feed",
            "fields": ["latest_trade", "latest_quote", "price", "ts"],
            "price_priority": ["latest_trade.p", "latest_quote_mid"],
            "requires_key": True,
            "key_env": ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"],
            "read_only": True,
            "trading_api_used": False,
            "known_limits": [
                "IEX is not SIP and does not represent full US market volume",
                "symbols without a recent IEX trade may use quote mid or return unavailable",
                "halts, closed sessions, and vendor entitlement changes can affect freshness",
                "rate limits are surfaced as alpaca_rate_limited",
            ],
        }

    def get_latest_quote(self, symbol: str) -> dict[str, Any]:
        symbol_key = _normalize_symbol(symbol)
        if not symbol_key:
            return self._empty_quote("", status="unavailable", reason="invalid_symbol")
        return self.get_latest_quotes([symbol_key])[symbol_key]

    def get_latest_quotes(self, symbols: Iterable[str]) -> dict[str, dict[str, Any]]:
        symbol_keys = _unique_symbols(symbols)
        if not symbol_keys:
            return {}
        if not self.configured:
            return {
                symbol: self._empty_quote(
                    symbol,
                    status="unavailable",
                    reason=ALPACA_KEY_NOT_CONFIGURED,
                )
                for symbol in symbol_keys
            }

        trade_payload = self._get_json(
            "/v2/stocks/trades/latest",
            {"symbols": ",".join(symbol_keys), "feed": self.feed},
        )
        if trade_payload.get("status") == "rate_limited":
            return {
                symbol: self._empty_quote(
                    symbol,
                    status="unavailable",
                    reason="alpaca_rate_limited",
                    retry_after=trade_payload.get("retry_after"),
                )
                for symbol in symbol_keys
            }
        if trade_payload.get("status") == "http_error":
            return {
                symbol: self._empty_quote(
                    symbol,
                    status="unavailable",
                    reason="alpaca_http_error",
                    http_status=trade_payload.get("http_status"),
                )
                for symbol in symbol_keys
            }

        quote_payload = self._get_json(
            "/v2/stocks/quotes/latest",
            {"symbols": ",".join(symbol_keys), "feed": self.feed},
        )
        if quote_payload.get("status") == "rate_limited":
            quote_payload = {"quotes": {}}
        if quote_payload.get("status") == "http_error":
            quote_payload = {"quotes": {}}

        trades = _mapping_payload(trade_payload.get("trades"))
        quotes = _mapping_payload(quote_payload.get("quotes"))
        return {
            symbol: self._quote_from_payload(
                symbol,
                _mapping_payload(trades.get(symbol)),
                _mapping_payload(quotes.get(symbol)),
            )
            for symbol in symbol_keys
        }

    def _get_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        if not path.startswith("/v2/stocks/"):
            raise ValueError(f"unsupported Alpaca data path: {path}")
        url = f"{ALPACA_DATA_BASE_URL}{path}"
        real_client: httpx.Client | None = None
        if self._http_client is None:
            real_client = httpx.Client()
            client = cast(_HttpClient, real_client)
        else:
            client = self._http_client
        try:
            response = client.get(
                url,
                headers={
                    "APCA-API-KEY-ID": self.api_key_id,
                    "APCA-API-SECRET-KEY": self.api_secret_key,
                },
                params=params,
                timeout=self.timeout_sec,
            )
            if response.status_code == 429:
                return {
                    "status": "rate_limited",
                    "retry_after": response.headers.get("Retry-After"),
                }
            if response.status_code >= 400:
                return {"status": "http_error", "http_status": response.status_code}
            payload = response.json()
            return payload if isinstance(payload, dict) else {"status": "invalid_json"}
        finally:
            if real_client is not None:
                real_client.close()

    def _quote_from_payload(
        self,
        symbol: str,
        trade: Mapping[str, Any],
        quote: Mapping[str, Any],
    ) -> dict[str, Any]:
        trade_price = _decimal_or_none(trade.get("p"))
        bid_price = _decimal_or_none(quote.get("bp"))
        ask_price = _decimal_or_none(quote.get("ap"))
        quote_mid = _quote_mid(bid_price, ask_price)
        if trade_price is not None:
            price = trade_price
            ts = _string_or_none(trade.get("t"))
            price_source = "latest_trade"
        elif quote_mid is not None:
            price = quote_mid
            ts = _string_or_none(quote.get("t"))
            price_source = "latest_quote_mid"
        else:
            return self._empty_quote(
                symbol,
                status="unavailable",
                reason="alpaca_quote_unavailable",
                latest_trade_ts=_string_or_none(trade.get("t")),
                quote_ts=_string_or_none(quote.get("t")),
            )
        payload = self._empty_quote(symbol, status="ok", reason=None)
        payload.update(
            {
                "price": str(price),
                "ts": ts,
                "price_source": price_source,
                "latest_trade_price": str(trade_price) if trade_price is not None else None,
                "latest_trade_ts": _string_or_none(trade.get("t")),
                "bid_price": str(bid_price) if bid_price is not None else None,
                "ask_price": str(ask_price) if ask_price is not None else None,
                "bid_size": _string_or_none(quote.get("bs")),
                "ask_size": _string_or_none(quote.get("as")),
                "quote_ts": _string_or_none(quote.get("t")),
            }
        )
        return payload

    def _empty_quote(
        self,
        symbol: str,
        *,
        status: str,
        reason: str | None,
        retry_after: object | None = None,
        http_status: object | None = None,
        latest_trade_ts: str | None = None,
        quote_ts: str | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "status": status,
            "reason": reason,
            "price": None,
            "ts": None,
            "source": self.source,
            "feed": self.feed,
            "market": self.market,
            "currency": "USD",
            "read_only": self.read_only,
            "coverage": "iex_only_not_sip",
            "price_source": None,
            "latest_trade_price": None,
            "latest_trade_ts": latest_trade_ts,
            "bid_price": None,
            "ask_price": None,
            "bid_size": None,
            "ask_size": None,
            "quote_ts": quote_ts,
            "retry_after": retry_after,
            "http_status": http_status,
        }


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def _unique_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _mapping_payload(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed > 0 else None


def _quote_mid(bid_price: Decimal | None, ask_price: Decimal | None) -> Decimal | None:
    if bid_price is None or ask_price is None:
        return None
    return (bid_price + ask_price) / Decimal("2")


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)
