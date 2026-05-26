import importlib.util
import re
import sys
from pathlib import Path
from uuid import uuid4

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


execution_app = load_service_app("execution_service_app")


def request_payload(idempotency_key: str = "demo-paper-1") -> dict[str, str | bool | None]:
    return {
        "execution_id": str(uuid4()),
        "intent_id": "11111111-1111-4111-8111-111111111111",
        "decision_id": "33333333-3333-4333-8333-333333333333",
        "idempotency_key": idempotency_key,
        "confirmation_token": None,
        "dry_run": False,
        "submitted_at": "2026-05-25T00:00:00Z",
    }


def execution_headers(**overrides: str) -> dict[str, str]:
    headers = {
        "x-decision-approved": "true",
        "x-mode": "paper",
        "x-symbol": "BTCUSDT",
        "x-quantity": "0.001",
        "x-side": "buy",
        "x-quantity-kind": "base",
        "x-quote-qty": "",
        "x-order-type": "market",
        "x-limit-price": "",
        "x-time-in-force": "GTC",
    }
    headers.update(overrides)
    return headers


class FakeTickerResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"symbol": "BTCUSDT", "price": "100000.00"}


class FakeBinanceResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def filled_order() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "orderId": 999,
        "status": "FILLED",
        "transactTime": 1748000000000,
        "origQty": "0.001",
        "executedQty": "0.001",
        "cummulativeQuoteQty": "100.00",
        "fills": [
            {
                "price": "100000.00",
                "qty": "0.001",
                "commission": "0.0000001",
                "commissionAsset": "BTC",
                "tradeId": 1,
            }
        ],
    }


def enable_live(monkeypatch) -> None:
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(execution_app, "EXCHANGE_API_KEY", "testkey")
    monkeypatch.setattr(execution_app, "EXCHANGE_API_SECRET", "testsecret")


def test_execute_requires_approved_decision() -> None:
    response = TestClient(execution_app.app).post("/execute", json=request_payload())
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "RISK_DECISION_NOT_APPROVED"


def test_live_mode_rejected_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", False)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "LIVE_TRADING_DISABLED"


def test_paper_execute_returns_simulated_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(execution_app.httpx, "get", lambda *args, **kwargs: FakeTickerResponse())
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "simulated"
    assert body["avg_price"] == "100000.00"
    assert (tmp_path / "trading.sqlite").exists()
    assert not (tmp_path / "phase1.sqlite").exists()


def test_live_mode_missing_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(execution_app, "EXCHANGE_API_KEY", "")
    monkeypatch.setattr(execution_app, "EXCHANGE_API_SECRET", "")
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error"] == "exchange credentials not configured"


def test_live_mode_filled_order(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    monkeypatch.setattr(
        execution_app.httpx,
        "post",
        lambda *args, **kwargs: FakeBinanceResponse(filled_order()),
    )
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "filled"
    assert body["venue_order_id"] == "999"
    assert len(body["fills"]) == 1
    assert body["avg_price"] == "100000.00"


def test_live_mode_signing_uses_hmac(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(filled_order())

    monkeypatch.setattr(execution_app.httpx, "post", fake_post)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 200
    params = captured["params"]
    headers = captured["headers"]
    assert isinstance(params, dict)
    assert "signature" in params
    assert "timestamp" in params
    assert isinstance(headers, dict)
    assert headers["X-MBX-APIKEY"] == "testkey"


def test_live_mode_new_status_polls_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    sleep_calls = {"n": 0}
    get_calls = {"n": 0}

    def fake_sleep(seconds: int | float) -> None:
        assert seconds == 2
        sleep_calls["n"] += 1

    def fake_get(*args: object, **kwargs: object) -> FakeBinanceResponse:
        get_calls["n"] += 1
        return FakeBinanceResponse(filled_order())

    new_order = dict(filled_order(), status="NEW")
    monkeypatch.setattr(execution_app.time_lib, "sleep", fake_sleep)
    monkeypatch.setattr(execution_app.httpx, "post", lambda *a, **k: FakeBinanceResponse(new_order))
    monkeypatch.setattr(execution_app.httpx, "get", fake_get)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "filled"
    assert sleep_calls["n"] == 1
    assert get_calls["n"] == 1


def test_live_mode_exception_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("connection failed")

    monkeypatch.setattr(execution_app.httpx, "post", fail)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "connection failed" in body["error"]


def test_sanitize_client_order_id() -> None:
    sanitized = execution_app._sanitize_client_order_id("bad key!*" * 10)
    assert re.fullmatch(r"[.A-Za-z0-9_-]{1,36}", sanitized)
    assert len(sanitized) <= 36


def test_quote_qty_used_for_quote_kind(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(filled_order())

    monkeypatch.setattr(execution_app.httpx, "post", fake_post)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(
            **{"x-mode": "live", "x-quantity-kind": "quote", "x-quote-qty": "100"}
        ),
    )
    assert response.status_code == 200
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["quoteOrderQty"] == "100"
    assert "quantity" not in params


def test_base_qty_used_for_base_kind(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(filled_order())

    monkeypatch.setattr(execution_app.httpx, "post", fake_post)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live", "x-quantity-kind": "base"}),
    )
    assert response.status_code == 200
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["quantity"] == "0.001"
    assert "quoteOrderQty" not in params



def test_live_limit_order_sends_correct_binance_params(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(dict(filled_order(), status="NEW", fills=[]))

    monkeypatch.setattr(execution_app.httpx, "post", fake_post)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(
            **{
                "x-mode": "live",
                "x-order-type": "limit",
                "x-limit-price": "95000.00",
                "x-time-in-force": "GTC",
                "x-quantity-kind": "base",
                "x-quantity": "0.001",
            }
        ),
    )
    assert response.status_code == 200
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["type"] == "LIMIT"
    assert params["price"] == "95000.00"
    assert params["timeInForce"] == "GTC"
    assert params["quantity"] == "0.001"
    assert "quoteOrderQty" not in params


def test_live_limit_order_new_status_returns_open_without_polling(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    sleep_calls = {"n": 0}
    get_calls = {"n": 0}

    def fake_sleep(seconds: int | float) -> None:
        sleep_calls["n"] += 1

    def fake_get(*args: object, **kwargs: object) -> FakeBinanceResponse:
        get_calls["n"] += 1
        return FakeBinanceResponse(filled_order())

    new_order = dict(filled_order(), status="NEW", fills=[])
    monkeypatch.setattr(execution_app.time_lib, "sleep", fake_sleep)
    monkeypatch.setattr(execution_app.httpx, "post", lambda *a, **k: FakeBinanceResponse(new_order))
    monkeypatch.setattr(execution_app.httpx, "get", fake_get)
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(
            **{"x-mode": "live", "x-order-type": "limit", "x-limit-price": "95000.00"}
        ),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "open"
    assert sleep_calls["n"] == 0
    assert get_calls["n"] == 0


def test_live_market_order_new_status_still_polls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    sleep_calls = {"n": 0}

    def fake_sleep(seconds: int | float) -> None:
        assert seconds == 2
        sleep_calls["n"] += 1

    new_order = dict(filled_order(), status="NEW")
    monkeypatch.setattr(execution_app.time_lib, "sleep", fake_sleep)
    monkeypatch.setattr(execution_app.httpx, "post", lambda *a, **k: FakeBinanceResponse(new_order))
    monkeypatch.setattr(
        execution_app.httpx, "get", lambda *a, **k: FakeBinanceResponse(filled_order())
    )
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-mode": "live", "x-order-type": "market"}),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "filled"
    assert sleep_calls["n"] == 1


def test_paper_limit_order_fills_at_limit_price(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(
            **{"x-order-type": "limit", "x-limit-price": "92000.00", "x-quantity": "0.001"}
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "simulated"
    assert body["avg_price"] == "92000.00"


def test_paper_market_order_still_fetches_market_price(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(execution_app.httpx, "get", lambda *args, **kwargs: FakeTickerResponse())
    response = TestClient(execution_app.app).post(
        "/execute",
        json=request_payload(),
        headers=execution_headers(**{"x-order-type": "market"}),
    )
    assert response.status_code == 200
    assert response.json()["avg_price"] == "100000.00"


def test_cancel_returns_403_when_live_disabled(monkeypatch) -> None:
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", False)
    response = TestClient(execution_app.app).post(
        "/cancel",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "LIVE_TRADING_DISABLED"


def test_cancel_returns_403_for_paper_mode(monkeypatch) -> None:
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", True)
    response = TestClient(execution_app.app).post(
        "/cancel",
        json=request_payload(),
        headers={"x-mode": "paper", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 403


def test_cancel_missing_credentials_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(execution_app, "EXCHANGE_API_KEY", "")
    monkeypatch.setattr(execution_app, "EXCHANGE_API_SECRET", "")
    response = TestClient(execution_app.app).post(
        "/cancel",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert "credentials" in response.json()["error"]


def test_cancel_calls_binance_delete_with_signed_params(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}
    canceled_order = dict(filled_order(), status="CANCELED", executedQty="0", fills=[])

    def fake_delete(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(canceled_order)

    monkeypatch.setattr(execution_app.httpx, "delete", fake_delete)
    response = TestClient(execution_app.app).post(
        "/cancel",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"
    params = captured["params"]
    assert isinstance(params, dict)
    assert "signature" in params
    assert params["orderId"] == "999"


def test_cancel_falls_back_to_query_when_binance_returns_400(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)

    def fake_delete(*args: object, **kwargs: object) -> FakeBinanceResponse:
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("DELETE", "https://api.binance.com/api/v3/order"),
            response=httpx.Response(400, json={"code": -2011, "msg": "Unknown order sent."}),
        )

    def fake_get(*args: object, **kwargs: object) -> FakeBinanceResponse:
        return FakeBinanceResponse(filled_order())

    monkeypatch.setattr(execution_app.httpx, "delete", fake_delete)
    monkeypatch.setattr(execution_app.httpx, "get", fake_get)
    response = TestClient(execution_app.app).post(
        "/cancel",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "filled"


def test_refresh_returns_403_when_live_disabled(monkeypatch) -> None:
    monkeypatch.setattr(execution_app, "LIVE_TRADING_ENABLED", False)
    response = TestClient(execution_app.app).post(
        "/refresh",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 403


def test_refresh_calls_binance_get_and_returns_current_status(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    captured: dict[str, object] = {}

    def fake_get(*args: object, **kwargs: object) -> FakeBinanceResponse:
        captured.update(kwargs)
        return FakeBinanceResponse(filled_order())

    monkeypatch.setattr(execution_app.httpx, "get", fake_get)
    response = TestClient(execution_app.app).post(
        "/refresh",
        json=request_payload(),
        headers={
            "x-mode": "live",
            "x-symbol": "BTCUSDT",
            "x-venue-order-id": "999",
            "x-order-type": "limit",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "filled"
    params = captured["params"]
    assert isinstance(params, dict)
    assert "signature" in params
    assert params["orderId"] == "999"


def test_refresh_returns_open_when_still_new(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)
    still_new = dict(filled_order(), status="NEW", executedQty="0", fills=[])
    monkeypatch.setattr(
        execution_app.httpx, "get", lambda *a, **k: FakeBinanceResponse(still_new)
    )
    response = TestClient(execution_app.app).post(
        "/refresh",
        json=request_payload(),
        headers={
            "x-mode": "live",
            "x-symbol": "BTCUSDT",
            "x-venue-order-id": "999",
            "x-order-type": "limit",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "open"


def test_refresh_exception_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    enable_live(monkeypatch)

    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("fail")

    monkeypatch.setattr(execution_app.httpx, "get", fail)
    response = TestClient(execution_app.app).post(
        "/refresh",
        json=request_payload(),
        headers={"x-mode": "live", "x-symbol": "BTCUSDT", "x-venue-order-id": "999"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"
