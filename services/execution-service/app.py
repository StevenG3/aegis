from __future__ import annotations

import hashlib
import hmac as hmac_lib
import os
import time as time_lib
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast
from urllib.parse import urlencode

import httpx
from db import connect
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import ExecutionRequest, ExecutionResult, Fill

app = FastAPI(title="execution-service", version="0.1.0")
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")
BINANCE_BASE_URL = "https://api.binance.com"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


def _now() -> datetime:
    return datetime.now(UTC)


def _binance_sign(params: dict[str, str]) -> dict[str, str]:
    params = dict(params)
    params["timestamp"] = str(int(time_lib.time() * 1000))
    query = urlencode(params)
    sig = hmac_lib.new(
        EXCHANGE_API_SECRET.encode(),
        query.encode(),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = sig
    return params


def _sanitize_client_order_id(key: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.")
    sanitized = "".join(c for c in key if c in allowed)
    return sanitized[:36] or "order"


def _cancel_binance_order(symbol: str, venue_order_id: str) -> dict[str, object]:
    params = _binance_sign({"symbol": symbol.upper(), "orderId": venue_order_id})
    response = httpx.delete(
        f"{BINANCE_BASE_URL}/api/v3/order",
        params=params,
        headers={"X-MBX-APIKEY": EXCHANGE_API_KEY},
        timeout=10.0,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, dict):
        raise ValueError("unexpected Binance cancel response")
    return raw


def _query_binance_order(symbol: str, venue_order_id: str) -> dict[str, object]:
    params = _binance_sign({"symbol": symbol.upper(), "orderId": venue_order_id})
    response = httpx.get(
        f"{BINANCE_BASE_URL}/api/v3/order",
        params=params,
        headers={"X-MBX-APIKEY": EXCHANGE_API_KEY},
        timeout=5.0,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, dict):
        raise ValueError("unexpected Binance order query response")
    return raw


def _persist_execution(request: ExecutionRequest, result: ExecutionResult) -> None:
    with connect() as conn:
        conn.execute(
            (
                "insert or replace into executions"
                "(execution_id,payload_json,result_json,created_at) values(?,?,?,?)"
            ),
            (
                str(request.execution_id),
                request.model_dump_json(),
                result.model_dump_json(),
                result.finalized_at.isoformat(),
            ),
        )
        conn.commit()


def _place_binance_order(
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    client_order_id: str,
    order_type: str = "market",
    limit_price: str = "",
    time_in_force: str = "GTC",
) -> dict[str, object]:
    params: dict[str, str] = {
        "symbol": symbol.upper(),
        "side": side.upper(),
        "type": order_type.upper(),
        "newClientOrderId": _sanitize_client_order_id(client_order_id),
    }
    if order_type.lower() == "limit":
        params["price"] = limit_price
        params["timeInForce"] = time_in_force.upper()
        params["quantity"] = base_qty
    elif quantity_kind == "quote":
        params["quoteOrderQty"] = quote_qty
    else:
        params["quantity"] = base_qty

    signed = _binance_sign(params)
    response = httpx.post(
        f"{BINANCE_BASE_URL}/api/v3/order",
        params=signed,
        headers={"X-MBX-APIKEY": EXCHANGE_API_KEY},
        timeout=10.0,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, dict):
        raise ValueError("unexpected Binance response")
    return raw


def _result_from_binance(
    request: ExecutionRequest,
    raw: dict[str, object],
    idempotency_key: str,
    order_type: str = "market",
) -> ExecutionResult:
    binance_status = str(raw.get("status", ""))
    status_map: dict[str, str] = {
        "FILLED": "filled",
        "PARTIALLY_FILLED": "partial",
        "REJECTED": "rejected",
        "CANCELED": "canceled",
        "EXPIRED": "canceled",
    }
    if order_type.lower() == "limit":
        status_map["NEW"] = "open"
    our_status = cast(
        Literal["filled", "partial", "rejected", "canceled", "error", "simulated", "open"],
        status_map.get(binance_status, "error"),
    )

    transact_time = int(str(raw.get("transactTime", 0)))
    finalized_at = (
        datetime.fromtimestamp(transact_time / 1000, UTC) if transact_time else _now()
    )

    fills: list[Fill] = []
    raw_fills = raw.get("fills", [])
    if isinstance(raw_fills, list):
        for fill in raw_fills:
            if not isinstance(fill, dict):
                continue
            fills.append(
                Fill(
                    price=Decimal(str(fill["price"])),
                    qty=Decimal(str(fill["qty"])),
                    fee=Decimal(str(fill["commission"])),
                    fee_asset=str(fill["commissionAsset"]),
                    ts=finalized_at,
                )
            )

    executed_qty = Decimal(str(raw.get("executedQty", "0")))
    orig_qty = Decimal(str(raw.get("origQty", "0")))
    cum_quote = Decimal(str(raw.get("cummulativeQuoteQty", "0")))
    avg_price = (cum_quote / executed_qty).quantize(Decimal("0.01")) if executed_qty else None

    return ExecutionResult(
        execution_id=request.execution_id,
        intent_id=request.intent_id,
        decision_id=request.decision_id,
        idempotency_key=idempotency_key,
        status=our_status,
        venue_order_id=str(raw.get("orderId", "")),
        fills=fills,
        avg_price=avg_price,
        filled_qty=executed_qty,
        remaining_qty=max(Decimal("0"), orig_qty - executed_qty),
        error=None,
        raw_venue_response_ref=str(raw.get("orderId", "")),
        finalized_at=finalized_at,
    )


def _error_result(request: ExecutionRequest, error: str) -> ExecutionResult:
    return ExecutionResult(
        execution_id=request.execution_id,
        intent_id=request.intent_id,
        decision_id=request.decision_id,
        idempotency_key=request.idempotency_key,
        status="error",
        venue_order_id=None,
        fills=[],
        avg_price=None,
        filled_qty=Decimal("0"),
        remaining_qty=Decimal("0"),
        error=error,
        raw_venue_response_ref=None,
        finalized_at=_now(),
    )


def _execute_live(
    request: ExecutionRequest,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    order_type: str = "market",
    limit_price: str = "",
    time_in_force: str = "GTC",
) -> ExecutionResult:
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        result = _error_result(request, "exchange credentials not configured")
        _persist_execution(request, result)
        return result

    try:
        raw = _place_binance_order(
            symbol=symbol,
            side=side,
            quantity_kind=quantity_kind,
            base_qty=base_qty,
            quote_qty=quote_qty,
            client_order_id=request.idempotency_key,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        if str(raw.get("status")) == "NEW" and order_type.lower() != "limit":
            time_lib.sleep(2)
            signed = _binance_sign({"symbol": symbol.upper(), "orderId": str(raw["orderId"])})
            poll = httpx.get(
                f"{BINANCE_BASE_URL}/api/v3/order",
                params=signed,
                headers={"X-MBX-APIKEY": EXCHANGE_API_KEY},
                timeout=5.0,
            )
            poll.raise_for_status()
            polled_raw = poll.json()
            if not isinstance(polled_raw, dict):
                raise ValueError("unexpected Binance poll response")
            raw = polled_raw
        result = _result_from_binance(request, raw, request.idempotency_key, order_type)
    except Exception as exc:  # noqa: BLE001
        result = _error_result(request, str(exc)[:500])

    _persist_execution(request, result)
    return result


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/cancel", response_model=ExecutionResult)
def cancel_order(
    request: ExecutionRequest,
    x_mode: str = Header(default="paper"),
    x_symbol: str = Header(default="BTCUSDT"),
    x_venue_order_id: str = Header(default=""),
    x_order_type: str = Header(default="limit"),
) -> ExecutionResult:
    if x_mode != "live" or not LIVE_TRADING_ENABLED:
        raise HTTPException(status_code=403, detail={"code": "LIVE_TRADING_DISABLED"})
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        result = _error_result(request, "exchange credentials not configured")
        _persist_execution(request, result)
        return result

    try:
        raw = _cancel_binance_order(x_symbol, x_venue_order_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400:
            try:
                raw = _query_binance_order(x_symbol, x_venue_order_id)
            except Exception as inner:  # noqa: BLE001
                result = _error_result(request, str(inner)[:500])
                _persist_execution(request, result)
                return result
        else:
            result = _error_result(request, str(exc)[:500])
            _persist_execution(request, result)
            return result
    except Exception as exc:  # noqa: BLE001
        result = _error_result(request, str(exc)[:500])
        _persist_execution(request, result)
        return result

    result = _result_from_binance(request, raw, request.idempotency_key, x_order_type)
    _persist_execution(request, result)
    return result


@app.post("/refresh", response_model=ExecutionResult)
def refresh_order(
    request: ExecutionRequest,
    x_mode: str = Header(default="paper"),
    x_symbol: str = Header(default="BTCUSDT"),
    x_venue_order_id: str = Header(default=""),
    x_order_type: str = Header(default="limit"),
) -> ExecutionResult:
    if x_mode != "live" or not LIVE_TRADING_ENABLED:
        raise HTTPException(status_code=403, detail={"code": "LIVE_TRADING_DISABLED"})
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        result = _error_result(request, "exchange credentials not configured")
        _persist_execution(request, result)
        return result

    try:
        raw = _query_binance_order(x_symbol, x_venue_order_id)
        result = _result_from_binance(request, raw, request.idempotency_key, x_order_type)
    except Exception as exc:  # noqa: BLE001
        result = _error_result(request, str(exc)[:500])

    _persist_execution(request, result)
    return result


@app.post("/execute", response_model=ExecutionResult)
def execute(
    request: ExecutionRequest,
    x_decision_approved: str = Header(default="false"),
    x_mode: str = Header(default="paper"),
    x_symbol: str = Header(default="BTCUSDT"),
    x_quantity: str = Header(default="0"),
    x_side: str = Header(default="buy"),
    x_quantity_kind: str = Header(default="base"),
    x_quote_qty: str = Header(default=""),
    x_order_type: str = Header(default="market"),
    x_limit_price: str = Header(default=""),
    x_time_in_force: str = Header(default="GTC"),
) -> ExecutionResult:
    if x_decision_approved.lower() != "true":
        raise HTTPException(status_code=403, detail={"code": "RISK_DECISION_NOT_APPROVED"})

    if x_mode == "live":
        if not LIVE_TRADING_ENABLED:
            raise HTTPException(status_code=403, detail={"code": "LIVE_TRADING_DISABLED"})
        return _execute_live(
            request,
            x_symbol,
            x_side,
            x_quantity_kind,
            x_quantity,
            x_quote_qty,
            x_order_type,
            x_limit_price,
            x_time_in_force,
        )

    if x_order_type.lower() == "limit" and x_limit_price:
        price = Decimal(x_limit_price)
    else:
        market_url = os.getenv("MARKET_DATA_URL", "http://market-data:8083")
        response = httpx.get(f"{market_url}/ticker", params={"symbol": x_symbol}, timeout=5.0)
        response.raise_for_status()
        price = Decimal(response.json()["price"])
    qty = Decimal(x_quantity)
    result = ExecutionResult(
        execution_id=request.execution_id,
        intent_id=request.intent_id,
        decision_id=request.decision_id,
        idempotency_key=request.idempotency_key,
        status="simulated",
        venue_order_id=None,
        fills=[Fill(price=price, qty=qty, fee=Decimal("0"), fee_asset="USDT", ts=_now())],
        avg_price=price,
        filled_qty=qty,
        remaining_qty=Decimal("0"),
        error=None,
        raw_venue_response_ref=None,
        finalized_at=_now(),
    )
    _persist_execution(request, result)
    return result
