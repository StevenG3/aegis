from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ibkr_client import IBKRClient, IBKRConfig, PlaceOrderRequest

logger = logging.getLogger(__name__)
_reconnect_lock = threading.Lock()
_last_reconnect_monotonic = 0.0
_last_connect_error: str | None = None


class BridgeOrderResponse(BaseModel):
    id: str
    status: Literal["pending", "submitted", "filled", "partial", "canceled", "rejected", "error"]
    fills: list[dict[str, str]]
    avg_price: str | None
    filled_qty: str
    remaining_qty: str
    error: str | None
    raw_order_ref: str | None


class TickerResponse(BaseModel):
    symbol: str
    price: str
    source: str = "ibkr"


class BridgePositionItem(BaseModel):
    symbol: str
    qty: str
    avg_cost: str


class BridgePositionsResponse(BaseModel):
    positions: list[BridgePositionItem]
    source: str = "ibkr"
    ready: bool = True
    last_update: str | None = None


def _config_from_env() -> IBKRConfig:
    return IBKRConfig(
        host=os.getenv("IBKR_GATEWAY_HOST", "host.docker.internal"),
        port=int(os.getenv("IBKR_GATEWAY_PORT", "4002")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        timeout_sec=float(os.getenv("IBKR_CONNECT_TIMEOUT_SEC", "10")),
        allow_live_port=os.getenv("IBKR_ALLOW_LIVE_PORT", "false").lower() == "true",
        account_code=os.getenv("IBKR_ACCOUNT_CODE", ""),
    )


client = IBKRClient(_config_from_env())


def _reconnect_cooldown_sec() -> float:
    return float(os.getenv("IBKR_RECONNECT_COOLDOWN_SEC", "15"))


def _try_connect(reason: str, *, force: bool = False) -> bool:
    global _last_connect_error, _last_reconnect_monotonic
    if client.is_ready():
        _last_connect_error = None
        return True
    with _reconnect_lock:
        if client.is_ready():
            _last_connect_error = None
            return True
        now = time.monotonic()
        if (
            not force
            and _last_reconnect_monotonic > 0
            and now - _last_reconnect_monotonic < _reconnect_cooldown_sec()
        ):
            return False
        _last_reconnect_monotonic = now
        try:
            client.disconnect()
            client.connect()
        except RuntimeError as exc:
            if str(exc).startswith("LIVE_PORT_NOT_AUTHORIZED"):
                logger.error("IBKR bridge connect failed: %s", exc)
                raise
            _last_connect_error = str(exc)
            logger.warning("IBKR bridge connect failed reason=%s error=%s", reason, exc)
            return False
        except Exception as exc:
            _last_connect_error = str(exc)
            logger.warning("IBKR bridge connect failed reason=%s error=%s", reason, exc)
            return False
        _last_connect_error = None
        logger.info("IBKR bridge connected reason=%s", reason)
        return True


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        await asyncio.to_thread(_try_connect, "startup", force=True)
    except RuntimeError as exc:
        if str(exc).startswith("LIVE_PORT_NOT_AUTHORIZED"):
            logger.error("IBKR bridge startup failed: %s", exc)
            raise
        # healthz stays up; readyz exposes disconnected state until Gateway/TWS appears.
        pass
    except Exception:
        # healthz stays up; readyz exposes disconnected state until Gateway/TWS appears.
        pass
    try:
        yield
    finally:
        await asyncio.to_thread(client.disconnect)


app = FastAPI(title="ibkr-bridge", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", response_model=None)
def readyz() -> JSONResponse | dict[str, str]:
    if not _try_connect("readyz"):
        content = {"status": "not_ready"}
        if _last_connect_error:
            content["error"] = _last_connect_error
        return JSONResponse(status_code=503, content=content)
    return {"status": "ready"}


def _ensure_ready() -> None:
    if not _try_connect("request"):
        raise HTTPException(status_code=503, detail={"code": "IBKR_NOT_READY"})


@app.post("/orders", response_model=BridgeOrderResponse)
def place_order(request: PlaceOrderRequest) -> dict[str, object]:
    _ensure_ready()
    try:
        return client.place_order(request)
    except ValueError as exc:
        code = str(exc)
        if code == "INVALID_SYMBOL":
            raise HTTPException(status_code=400, detail={"code": code}) from exc
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_ORDER", "message": code},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "IBKR_ORDER_FAILED"}) from exc


@app.get("/orders/{order_id}", response_model=BridgeOrderResponse)
def get_order(order_id: str) -> dict[str, object]:
    _ensure_ready()
    payload = client.get_order(order_id)
    if payload is None:
        raise HTTPException(status_code=404, detail={"code": "ORDER_NOT_FOUND"})
    return payload


@app.delete("/orders/{order_id}", response_model=BridgeOrderResponse)
def cancel_order(order_id: str) -> dict[str, object]:
    _ensure_ready()
    payload = client.cancel_order(order_id)
    if payload is None:
        raise HTTPException(status_code=404, detail={"code": "ORDER_NOT_FOUND"})
    return payload


@app.get("/positions", response_model=BridgePositionsResponse)
def get_positions() -> dict[str, object]:
    _ensure_ready()
    if not client.positions_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "IBKR_POSITIONS_NOT_READY",
                "message": "position subscription not primed yet",
            },
        )
    try:
        return {
            "positions": client.positions(),
            "source": "ibkr",
            "ready": True,
            "last_update": client.positions_last_update(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "IBKR_POSITIONS_FAILED", "error": str(exc)},
        ) from exc


@app.get("/tickers/{symbol}", response_model=TickerResponse)
def ticker(symbol: str = Path(min_length=1)) -> dict[str, str]:
    _ensure_ready()
    try:
        return client.ticker(symbol)
    except ValueError as exc:
        code = str(exc)
        if code == "INVALID_SYMBOL":
            raise HTTPException(status_code=400, detail={"code": code}) from exc
        raise HTTPException(status_code=503, detail={"code": "MARKET_DATA_UNAVAILABLE"}) from exc
