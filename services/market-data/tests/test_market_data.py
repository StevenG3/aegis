import importlib.util
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


def load_service_app(name: str):
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


market_app = load_service_app("market_data_app")
app = market_app.app


class FakeResponse:
    def __init__(self, price: str, symbol: str = "BTCUSDT") -> None:
        self.price = price
        self.symbol = symbol

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"symbol": self.symbol, "price": self.price}


class FakeJSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def test_ticker_returns_fixture(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "true")
    response = TestClient(app).get("/ticker", params={"symbol": "BTCUSDT"})
    assert response.status_code == 200
    assert response.json() == {"symbol": "BTCUSDT", "price": "100000.00", "source": "fixture"}


def test_ticker_fetches_live_price(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    market_app._CACHE.clear()
    monkeypatch.setattr(market_app.httpx, "get", lambda *args, **kwargs: FakeResponse("105000.12"))
    response = TestClient(app).get("/ticker", params={"symbol": "BTCUSDT"})
    assert response.status_code == 200
    assert response.json() == {"symbol": "BTCUSDT", "price": "105000.12", "source": "binance"}


def test_ticker_falls_back_on_error(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    market_app._CACHE.clear()

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(market_app.httpx, "get", fail)
    response = TestClient(app).get("/ticker", params={"symbol": "BTCUSDT"})
    assert response.status_code == 200
    assert response.json() == {
        "symbol": "BTCUSDT",
        "price": "100000.00",
        "source": "fixture_fallback",
    }


def test_quotes_returns_multiple_symbols(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "true")
    response = TestClient(app).get("/quotes", params={"symbols": "BTCUSDT,ETHUSDT"})
    assert response.status_code == 200
    assert response.json() == {
        "BTCUSDT": {"price": "100000.00", "source": "fixture"},
        "ETHUSDT": {"price": "5000.00", "source": "fixture"},
    }


def test_quotes_rejects_too_many_symbols(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "true")
    symbols = ",".join(f"BTC{i}" for i in range(11))
    response = TestClient(app).get("/quotes", params={"symbols": symbols})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "TOO_MANY_SYMBOLS"


def test_unknown_symbol_is_404(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "true")
    assert TestClient(app).get("/ticker", params={"symbol": "NOPE"}).status_code == 404


def test_ticker_live_unknown_symbol_no_fallback(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    market_app._CACHE.clear()

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(market_app.httpx, "get", fail)
    response = TestClient(app).get("/ticker", params={"symbol": "SOLUSDT"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MARKET_DATA_UNAVAILABLE"


def test_ticker_live_unknown_symbol_success(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    market_app._CACHE.clear()
    monkeypatch.setattr(
        market_app.httpx, "get", lambda *args, **kwargs: FakeResponse("150.25", "SOLUSDT")
    )
    response = TestClient(app).get("/ticker", params={"symbol": "SOLUSDT"})
    assert response.status_code == 200
    assert response.json() == {"symbol": "SOLUSDT", "price": "150.25", "source": "binance"}


def test_ticker_routes_stock_for_asset_type_stock(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURES_ONLY", "true")
    market_app._STOCK_QUOTE_CACHE.clear()
    response = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert response.status_code == 200
    assert response.json() == {
        "symbol": "NVDA",
        "price": "450.00",
        "source": "fixture",
        "asset_type": "stock",
    }


def test_stock_polygon_provider_returns_close(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def fake_get(url: str, **kwargs: object) -> FakeJSONResponse:
        calls.append((url, kwargs))
        return FakeJSONResponse({"results": [{"T": "NVDA", "c": 451.78}]})

    monkeypatch.setattr(market_app.httpx, "get", fake_get)
    assert market_app._polygon_fetch("NVDA") == {
        "symbol": "NVDA",
        "price": "451.78",
        "source": "polygon",
        "asset_type": "stock",
    }
    assert calls[0][0] == "https://api.polygon.io/v2/aggs/ticker/NVDA/prev"


def test_stock_polygon_skipped_when_key_unset(monkeypatch) -> None:
    calls = {"n": 0}
    monkeypatch.setenv("POLYGON_API_KEY", "")

    def fake_get(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("polygon should not be called without a key")

    monkeypatch.setattr(market_app.httpx, "get", fake_get)
    assert market_app._polygon_fetch("NVDA") is None
    assert calls["n"] == 0


def test_stock_yahoo_provider_returns_price(monkeypatch) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeJSONResponse:
        assert url == "https://query1.finance.yahoo.com/v8/finance/chart/NVDA"
        return FakeJSONResponse(
            {"chart": {"result": [{"meta": {"regularMarketPrice": 452.25}}]}}
        )

    monkeypatch.setattr(market_app.httpx, "get", fake_get)
    assert market_app._yahoo_fetch("NVDA") == {
        "symbol": "NVDA",
        "price": "452.25",
        "source": "yahoo",
        "asset_type": "stock",
    }


def test_stock_chain_falls_through_polygon_failure_to_yahoo(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "polygon,yahoo,fixture")
    monkeypatch.setattr(market_app, "_polygon_fetch", lambda symbol: None)
    monkeypatch.setattr(
        market_app,
        "_yahoo_fetch",
        lambda symbol: {
            "symbol": symbol,
            "price": "452.25",
            "source": "yahoo",
            "asset_type": "stock",
        },
    )
    response = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert response.status_code == 200
    assert response.json()["source"] == "yahoo"


def test_stock_chain_falls_through_all_real_providers_to_fixture(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "polygon,yahoo,fixture")
    monkeypatch.setattr(market_app, "_polygon_fetch", lambda symbol: None)
    monkeypatch.setattr(market_app, "_yahoo_fetch", lambda symbol: None)
    response = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert response.status_code == 200
    assert response.json()["source"] == "fixture"


def test_stock_cache_returns_cached_within_ttl(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    calls = {"n": 0}
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "yahoo")
    monkeypatch.setattr(market_app.time, "monotonic", lambda: 10.0)

    def fake_yahoo(symbol: str) -> dict[str, str]:
        calls["n"] += 1
        return {"symbol": symbol, "price": "452.25", "source": "yahoo", "asset_type": "stock"}

    monkeypatch.setattr(market_app, "_yahoo_fetch", fake_yahoo)
    first = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    second = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert first.json() == second.json()
    assert calls["n"] == 1


def test_stock_cache_expires_after_ttl(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    calls = {"n": 0}
    now = {"value": 10.0}
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "yahoo")
    monkeypatch.setattr(market_app.time, "monotonic", lambda: now["value"])

    def fake_yahoo(symbol: str) -> dict[str, str]:
        calls["n"] += 1
        return {
            "symbol": symbol,
            "price": str(450 + calls["n"]),
            "source": "yahoo",
            "asset_type": "stock",
        }

    monkeypatch.setattr(market_app, "_yahoo_fetch", fake_yahoo)
    first = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    now["value"] = 71.0
    second = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert first.json()["price"] == "451"
    assert second.json()["price"] == "452"
    assert calls["n"] == 2


def test_stock_chain_custom_order(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    order: list[str] = []
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "yahoo,polygon")
    monkeypatch.setattr(
        market_app,
        "_yahoo_fetch",
        lambda symbol: order.append("yahoo")
        or {"symbol": symbol, "price": "452.25", "source": "yahoo", "asset_type": "stock"},
    )
    monkeypatch.setattr(
        market_app,
        "_polygon_fetch",
        lambda symbol: order.append("polygon")
        or {"symbol": symbol, "price": "451.78", "source": "polygon", "asset_type": "stock"},
    )
    response = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert response.json()["source"] == "yahoo"
    assert order == ["yahoo"]


def test_stock_chain_unknown_provider_silently_skipped(monkeypatch) -> None:
    market_app._STOCK_QUOTE_CACHE.clear()
    monkeypatch.setenv("FIXTURES_ONLY", "false")
    monkeypatch.setenv("STOCK_QUOTE_PROVIDER_CHAIN", "iex,yahoo")
    monkeypatch.setattr(
        market_app,
        "_yahoo_fetch",
        lambda symbol: {
            "symbol": symbol,
            "price": "452.25",
            "source": "yahoo",
            "asset_type": "stock",
        },
    )
    response = TestClient(app).get("/ticker", params={"symbol": "NVDA", "asset_type": "stock"})
    assert response.status_code == 200
    assert response.json()["source"] == "yahoo"
