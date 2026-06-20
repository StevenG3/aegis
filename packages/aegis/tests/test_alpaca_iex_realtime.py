from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from aegis.alpaca_iex_realtime import (
    ALPACA_DATA_BASE_URL,
    ALPACA_KEY_NOT_CONFIGURED,
    AlpacaIexRealtimeQuoteSource,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = dict(headers or {})

    def json(self) -> Any:
        return self._payload


class FakeHttpClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str],
        timeout: float,
    ) -> Any:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "params": dict(params),
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("unexpected HTTP request")
        return self._responses.pop(0)


def test_alpaca_without_keys_degrades_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    http = FakeHttpClient([])
    source = AlpacaIexRealtimeQuoteSource(http_client=http)

    quote = source.get_latest_quote("spy")

    assert quote["symbol"] == "SPY"
    assert quote["status"] == "unavailable"
    assert quote["reason"] == ALPACA_KEY_NOT_CONFIGURED
    assert quote["source"] == "alpaca"
    assert quote["feed"] == "iex"
    assert quote["read_only"] is True
    assert quote["price"] is None
    assert http.requests == []


def test_alpaca_parses_latest_trade_and_marks_iex_feed() -> None:
    http = FakeHttpClient(
        [
            FakeResponse(
                200,
                {
                    "trades": {
                        "SPY": {
                            "t": "2026-06-19T19:59:59Z",
                            "p": 550.12,
                            "s": 100,
                        }
                    }
                },
            ),
            FakeResponse(
                200,
                {
                    "quotes": {
                        "SPY": {
                            "t": "2026-06-19T19:59:58Z",
                            "bp": 550.10,
                            "ap": 550.14,
                            "bs": 2,
                            "as": 3,
                        }
                    }
                },
            ),
        ]
    )
    source = AlpacaIexRealtimeQuoteSource(
        api_key_id="key-id",
        api_secret_key="secret",
        http_client=http,
        timeout_sec=4.0,
    )

    quote = source.get_latest_quote("spy")

    assert quote["status"] == "ok"
    assert quote["symbol"] == "SPY"
    assert quote["price"] == "550.12"
    assert quote["ts"] == "2026-06-19T19:59:59Z"
    assert quote["price_source"] == "latest_trade"
    assert quote["latest_trade_price"] == "550.12"
    assert quote["bid_price"] == "550.1"
    assert quote["ask_price"] == "550.14"
    assert quote["feed"] == "iex"
    assert quote["coverage"] == "iex_only_not_sip"
    assert len(http.requests) == 2
    assert http.requests[0]["url"] == f"{ALPACA_DATA_BASE_URL}/v2/stocks/trades/latest"
    assert http.requests[1]["url"] == f"{ALPACA_DATA_BASE_URL}/v2/stocks/quotes/latest"
    assert all(request["params"]["feed"] == "iex" for request in http.requests)
    assert all("api.alpaca.markets" not in request["url"] for request in http.requests)


def test_alpaca_falls_back_to_quote_mid_when_trade_missing() -> None:
    http = FakeHttpClient(
        [
            FakeResponse(200, {"trades": {}}),
            FakeResponse(
                200,
                {
                    "quotes": {
                        "QQQ": {
                            "t": "2026-06-19T20:00:01Z",
                            "bp": "480.10",
                            "ap": "480.30",
                        }
                    }
                },
            ),
        ]
    )
    source = AlpacaIexRealtimeQuoteSource(
        api_key_id="key-id",
        api_secret_key="secret",
        http_client=http,
    )

    quote = source.get_latest_quote("QQQ")

    assert quote["status"] == "ok"
    assert quote["price"] == "480.20"
    assert quote["ts"] == "2026-06-19T20:00:01Z"
    assert quote["price_source"] == "latest_quote_mid"
    assert quote["latest_trade_price"] is None


def test_alpaca_batch_quotes_are_deduplicated_and_rate_limit_is_explicit() -> None:
    http = FakeHttpClient(
        [
            FakeResponse(
                429,
                {"message": "too many requests"},
                headers={"Retry-After": "60"},
            )
        ]
    )
    source = AlpacaIexRealtimeQuoteSource(
        api_key_id="key-id",
        api_secret_key="secret",
        http_client=http,
    )

    quotes = source.get_latest_quotes(["spy", "SPY", "qqq"])

    assert list(quotes) == ["SPY", "QQQ"]
    assert quotes["SPY"]["status"] == "unavailable"
    assert quotes["SPY"]["reason"] == "alpaca_rate_limited"
    assert quotes["SPY"]["retry_after"] == "60"
    assert quotes["QQQ"]["reason"] == "alpaca_rate_limited"
    assert len(http.requests) == 1
    assert http.requests[0]["params"] == {"symbols": "SPY,QQQ", "feed": "iex"}


def test_alpaca_capability_matrix_discloses_limits() -> None:
    source = AlpacaIexRealtimeQuoteSource(api_key_id="", api_secret_key="")

    matrix = source.capability_matrix()

    assert matrix["host"] == "https://data.alpaca.markets"
    assert matrix["feed"] == "iex"
    assert matrix["read_only"] is True
    assert matrix["trading_api_used"] is False
    assert "ALPACA_API_KEY_ID" in matrix["key_env"]
    assert any("not SIP" in limit for limit in matrix["known_limits"])
