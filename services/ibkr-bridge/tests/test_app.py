from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path

import pytest
from starlette.testclient import TestClient


def load_service_app(name: str):
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge_app = load_service_app("ibkr_bridge_app")


class FakeClient:
    def __init__(self) -> None:
        self.connected = False
        self.connect_count = 0
        self.orders: dict[str, dict[str, object]] = {}
        self.position_cache_ready = True
        self.position_last_update = "2026-05-30T00:00:00+00:00"

    def connect(self) -> None:
        self.connect_count += 1
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_ready(self) -> bool:
        return self.connected

    def place_order(self, request: object) -> dict[str, object]:
        _ = request
        payload = {
            "id": "ibkr-123",
            "status": "filled",
            "fills": [],
            "avg_price": "451.25",
            "filled_qty": "2",
            "remaining_qty": "0",
            "error": None,
            "raw_order_ref": "123",
        }
        self.orders["ibkr-123"] = payload
        return payload

    def get_order(self, order_id: str) -> dict[str, object] | None:
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> dict[str, object] | None:
        payload = self.orders.get(order_id)
        if payload is None:
            return None
        payload = dict(payload, status="canceled")
        self.orders[order_id] = payload
        return payload

    def ticker(self, symbol: str) -> dict[str, str]:
        return {"symbol": symbol, "price": "452.50", "source": "ibkr"}

    def positions(self) -> list[dict[str, str]]:
        return [{"symbol": "NVDA", "qty": "10.00000000", "avg_cost": "450.25000000"}]

    def positions_ready(self) -> bool:
        return self.position_cache_ready

    def positions_last_update(self) -> str | None:
        return self.position_last_update

    def snapshot(self) -> dict[str, object]:
        return {
            "ok": True,
            "source": "ibkr-bridge",
            "ready": True,
            "account_summary": {
                "DU123": {
                    "NetLiquidation": {"value": "1000", "currency": "USD", "numeric": "1000"},
                    "TotalCashValue": {"value": "100", "currency": "USD", "numeric": "100"},
                    "GrossPositionValue": {"value": "900", "currency": "USD", "numeric": "900"},
                }
            },
            "positions": [
                {
                    "symbol": "NVDA",
                    "qty": "10.00000000",
                    "avg_cost": "450.25000000",
                    "market_price": "452.50",
                    "market_value": "4525.0000000000",
                }
            ],
            "positions_count": 1,
            "gross_position_value_computed": "4525.0000000000",
            "market_data_type": 3,
            "market_data_fallback_used": True,
            "market_data": [
                {"symbol": "NVDA", "market_price": "452.50", "price_source": "marketPrice"}
            ],
            "last_update": self.position_last_update,
            "ts_utc": "2026-05-30T00:00:01+00:00",
        }


class RuntimeErrorClient(FakeClient):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def connect(self) -> None:
        self.connect_count += 1
        raise RuntimeError(self.message)


@pytest.fixture(autouse=True)
def reset_reconnect_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_app, "_last_reconnect_monotonic", 0.0)
    monkeypatch.setattr(bridge_app, "_last_connect_error", None)
    monkeypatch.setenv("IBKR_RECONNECT_COOLDOWN_SEC", "0")


def test_lifespan_keeps_healthz_up_for_ibkr_connection_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(bridge_app, "client", RuntimeErrorClient("Cannot run event loop"))
    with TestClient(bridge_app.app) as test_client:
        response = test_client.get("/healthz")
    assert response.status_code == 200


def test_lifespan_runs_blocking_ibkr_connect_outside_async_loop(monkeypatch) -> None:
    class BlockingIBClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.connect_thread_name: str | None = None

        def connect(self) -> None:
            self.connect_thread_name = threading.current_thread().name
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            super().connect()

    fake = BlockingIBClient()
    monkeypatch.setattr(bridge_app, "client", fake)
    with TestClient(bridge_app.app) as test_client:
        response = test_client.get("/readyz")

    assert response.status_code == 200
    assert fake.connect_thread_name is not None


def test_lifespan_fails_closed_for_unauthorized_live_port(monkeypatch) -> None:
    monkeypatch.setattr(
        bridge_app,
        "client",
        RuntimeErrorClient("LIVE_PORT_NOT_AUTHORIZED: set IBKR_ALLOW_LIVE_PORT=true"),
    )
    try:
        with TestClient(bridge_app.app):
            raise AssertionError("startup should fail")
    except RuntimeError as exc:
        assert "LIVE_PORT_NOT_AUTHORIZED" in str(exc)


def test_healthz_ok_even_when_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(bridge_app, "client", FakeClient())
    response = TestClient(bridge_app.app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reflects_connection(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/readyz")
    assert response.status_code == 200
    assert fake.connect_count == 1
    response = TestClient(bridge_app.app).get("/readyz")
    assert response.status_code == 200
    assert fake.connect_count == 1


def test_readyz_reports_reconnect_failure(monkeypatch) -> None:
    fake = RuntimeErrorClient("Gateway not ready")
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "error": "Gateway not ready"}


def test_readyz_reconnect_uses_cooldown(monkeypatch) -> None:
    fake = RuntimeErrorClient("Gateway not ready")
    monkeypatch.setattr(bridge_app, "client", fake)
    monkeypatch.setenv("IBKR_RECONNECT_COOLDOWN_SEC", "60")
    test_client = TestClient(bridge_app.app)

    response = test_client.get("/readyz")
    assert response.status_code == 503
    assert fake.connect_count == 1

    response = test_client.get("/readyz")
    assert response.status_code == 503
    assert fake.connect_count == 1


def test_readyz_recovers_after_gateway_appears(monkeypatch) -> None:
    class LateGatewayClient(FakeClient):
        def connect(self) -> None:
            self.connect_count += 1
            if self.connect_count == 1:
                raise RuntimeError("Gateway not ready")
            self.connected = True

    fake = LateGatewayClient()
    monkeypatch.setattr(bridge_app, "client", fake)
    test_client = TestClient(bridge_app.app)

    response = test_client.get("/readyz")
    assert response.status_code == 503
    response = test_client.get("/readyz")
    assert response.status_code == 200
    assert fake.connect_count == 2


def test_place_get_cancel_order(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    monkeypatch.setattr(bridge_app, "client", fake)
    test_client = TestClient(bridge_app.app)
    response = test_client.post(
        "/orders",
        json={
            "idempotency_key": "abc",
            "symbol": "NVDA",
            "side": "buy",
            "order_type": "market",
            "quantity": "2",
            "limit_price": None,
            "time_in_force": "GTC",
        },
    )
    assert response.status_code == 200
    assert response.json()["id"] == "ibkr-123"

    response = test_client.get("/orders/ibkr-123")
    assert response.status_code == 200
    assert response.json()["status"] == "filled"

    response = test_client.delete("/orders/ibkr-123")
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"


def test_place_order_returns_503_when_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(bridge_app, "client", RuntimeErrorClient("Gateway not ready"))
    response = TestClient(bridge_app.app).post(
        "/orders",
        json={
            "idempotency_key": "abc",
            "symbol": "NVDA",
            "side": "buy",
            "order_type": "market",
            "quantity": "2",
            "limit_price": None,
            "time_in_force": "GTC",
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IBKR_NOT_READY"


def test_ticker(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/tickers/NVDA")
    assert response.status_code == 200
    assert response.json() == {"symbol": "NVDA", "price": "452.50", "source": "ibkr"}


def test_config_reads_live_port_authorization(monkeypatch) -> None:
    monkeypatch.setenv("IBKR_GATEWAY_PORT", "7496")
    monkeypatch.setenv("IBKR_ALLOW_LIVE_PORT", "true")
    monkeypatch.setenv("IBKR_ACCOUNT_CODE", "DU123")
    cfg = bridge_app._config_from_env()
    assert cfg.port == 7496
    assert cfg.allow_live_port is True
    assert cfg.account_code == "DU123"


def test_get_positions_returns_freshness_when_primed(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/positions")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "ibkr"
    assert body["ready"] is True
    assert body["last_update"] == "2026-05-30T00:00:00+00:00"
    assert len(body["positions"]) == 1
    assert body["positions"][0]["symbol"] == "NVDA"
    assert body["positions"][0]["qty"] == "10.00000000"


def test_get_positions_when_not_connected(monkeypatch) -> None:
    monkeypatch.setattr(bridge_app, "client", RuntimeErrorClient("Gateway not ready"))
    response = TestClient(bridge_app.app).get("/positions")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IBKR_NOT_READY"


def test_get_positions_503_when_not_primed(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    fake.position_cache_ready = False
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/positions")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IBKR_POSITIONS_NOT_READY"


def test_get_snapshot_returns_account_positions_prices(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/snapshot")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["source"] == "ibkr-bridge"
    assert body["account_summary"]["DU123"]["NetLiquidation"]["numeric"] == "1000"
    assert body["positions_count"] == 1
    assert body["positions"][0]["symbol"] == "NVDA"
    assert body["positions"][0]["market_value"] == "4525.0000000000"
    assert body["gross_position_value_computed"] == "4525.0000000000"
    assert body["market_data_type"] == 3
    assert body["market_data_fallback_used"] is True


def test_get_snapshot_when_not_connected(monkeypatch) -> None:
    monkeypatch.setattr(bridge_app, "client", RuntimeErrorClient("Gateway not ready"))
    response = TestClient(bridge_app.app).get("/snapshot")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IBKR_NOT_READY"


def test_get_snapshot_503_when_not_primed(monkeypatch) -> None:
    fake = FakeClient()
    fake.connect()
    fake.position_cache_ready = False
    monkeypatch.setattr(bridge_app, "client", fake)
    response = TestClient(bridge_app.app).get("/snapshot")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IBKR_POSITIONS_NOT_READY"
