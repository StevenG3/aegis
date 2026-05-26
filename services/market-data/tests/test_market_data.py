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
