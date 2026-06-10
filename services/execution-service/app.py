from __future__ import annotations

import base64
import binascii
import hashlib
import hmac as hmac_lib
import json
import os
import sqlite3
import time as time_lib
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal, cast
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from db import connect
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import ExecutionRequest, ExecutionResult, Fill

app = FastAPI(title="execution-service", version="0.1.0")
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
IBKR_LIVE_TRADING_ENABLED = (
    os.getenv("IBKR_LIVE_TRADING_ENABLED", "false").lower() == "true"
)
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")
BINANCE_BASE_URL = "https://api.binance.com"
IBKR_MODE = os.getenv("IBKR_MODE", "stub")
IBKR_BRIDGE_URL = os.getenv("IBKR_BRIDGE_URL", "http://ibkr-bridge:8086").rstrip("/")
IBKR_POLL_TIMEOUT_SEC = float(os.getenv("IBKR_POLL_TIMEOUT_SEC", "60"))
IBKR_POLL_INTERVAL_SEC = float(os.getenv("IBKR_POLL_INTERVAL_SEC", "1"))
CONFIRMATION_TOKEN_SECRET = os.getenv(
    "CONFIRMATION_TOKEN_SECRET", "aegis-local-confirmation-token-secret"
)
DEFAULT_CONFIRMATION_TOKEN_SECRET = "aegis-local-confirmation-token-secret"


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


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode())


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _token_error(code: str) -> HTTPException:
    return HTTPException(status_code=403, detail={"code": code})


def validate_confirmation_token_secret(secret: str | None = None) -> None:
    resolved = CONFIRMATION_TOKEN_SECRET if secret is None else secret
    if not resolved or resolved == DEFAULT_CONFIRMATION_TOKEN_SECRET:
        raise _token_error("CONFIRMATION_SECRET_NOT_SET")


def _decode_confirmation_token(token: str) -> dict[str, object]:
    try:
        encoded_payload, encoded_signature = token.split(".", maxsplit=1)
        expected_signature = hmac_lib.new(
            CONFIRMATION_TOKEN_SECRET.encode(),
            encoded_payload.encode(),
            hashlib.sha256,
        ).digest()
        if not hmac_lib.compare_digest(_b64url_decode(encoded_signature), expected_signature):
            raise ValueError("bad signature")
        payload = json.loads(_b64url_decode(encoded_payload))
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
        if _canonical_json(payload) != _b64url_decode(encoded_payload):
            raise ValueError("payload is not canonical")
    except (ValueError, json.JSONDecodeError, binascii.Error):
        raise _token_error("CONFIRMATION_TOKEN_INVALID") from None
    return payload


def _parse_token_datetime(payload: dict[str, object], key: str) -> datetime:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _token_error("CONFIRMATION_TOKEN_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise _token_error("CONFIRMATION_TOKEN_INVALID") from None
    if parsed.tzinfo is None:
        raise _token_error("CONFIRMATION_TOKEN_INVALID")
    return parsed.astimezone(UTC)


def _expect_token_value(
    payload: dict[str, object],
    key: str,
    expected: str | None,
    *,
    normalize_upper: bool = False,
    normalize_lower: bool = False,
) -> None:
    actual = payload.get(key)
    actual_value = "" if actual is None else str(actual)
    expected_value = "" if expected is None else expected
    if normalize_upper:
        actual_value = actual_value.upper()
        expected_value = expected_value.upper()
    if normalize_lower:
        actual_value = actual_value.lower()
        expected_value = expected_value.lower()
    if actual_value != expected_value:
        raise _token_error("CONFIRMATION_TOKEN_MISMATCH")


def _consume_confirmation_token(token: str, request: ExecutionRequest) -> None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    try:
        with connect() as conn:
            conn.execute(
                (
                    "insert into confirmation_token_consumptions"
                    "(token_hash,intent_id,execution_id,consumed_at) values(?,?,?,?)"
                ),
                (
                    token_hash,
                    str(request.intent_id),
                    str(request.execution_id),
                    _now().isoformat(),
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise _token_error("CONFIRMATION_TOKEN_REPLAYED") from None


def _validate_confirmation_token(
    request: ExecutionRequest,
    *,
    venue: str,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    order_type: str,
    limit_price: str,
    time_in_force: str,
) -> None:
    validate_confirmation_token_secret()
    if not request.confirmation_token:
        raise _token_error("CONFIRMATION_TOKEN_REQUIRED")
    payload = _decode_confirmation_token(request.confirmation_token)
    expires_at = _parse_token_datetime(payload, "expires_at")
    if _now() > expires_at:
        raise _token_error("CONFIRMATION_TOKEN_EXPIRED")

    _expect_token_value(payload, "intent_id", str(request.intent_id))
    _expect_token_value(payload, "venue", venue, normalize_lower=True)
    _expect_token_value(payload, "symbol", symbol, normalize_upper=True)
    _expect_token_value(payload, "side", side, normalize_lower=True)
    _expect_token_value(payload, "quantity_kind", quantity_kind, normalize_lower=True)
    _expect_token_value(payload, "order_type", order_type, normalize_lower=True)
    _expect_token_value(payload, "time_in_force", time_in_force, normalize_upper=True)
    expected_qty = quote_qty if quantity_kind.lower() == "quote" else base_qty
    _expect_token_value(payload, "quantity_value", expected_qty)
    expected_limit = limit_price or None
    _expect_token_value(payload, "limit_price", expected_limit)
    _consume_confirmation_token(request.confirmation_token, request)


def _execution_notional(
    result: ExecutionResult,
    *,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    limit_price: str,
) -> Decimal | None:
    if quantity_kind == "quote":
        return Decimal(quote_qty)
    if limit_price:
        return Decimal(base_qty) * Decimal(limit_price)
    if result.avg_price is not None and result.filled_qty > 0:
        return result.avg_price * result.filled_qty
    if result.fills:
        return sum((fill.price * fill.qty for fill in result.fills), Decimal("0"))
    return None


def _record_accepted_order(
    request: ExecutionRequest,
    result: ExecutionResult,
    *,
    mode: str,
    venue: str,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    limit_price: str,
) -> None:
    if result.status not in {"filled", "partial", "open", "simulated"}:
        return
    notional = _execution_notional(
        result,
        quantity_kind=quantity_kind,
        base_qty=base_qty,
        quote_qty=quote_qty,
        limit_price=limit_price,
    )
    if notional is None or notional <= 0:
        return
    with connect() as conn:
        conn.execute(
            """
            insert or ignore into accepted_order_ledger
            (intent_id,execution_id,mode,venue,symbol,side,notional,accepted_at)
            values(?,?,?,?,?,?,?,?)
            """,
            (
                str(request.intent_id),
                str(request.execution_id),
                mode,
                venue,
                symbol.upper(),
                side.lower(),
                str(notional),
                result.finalized_at.astimezone(UTC).isoformat(),
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


def _binance_paper_execute(
    request: ExecutionRequest,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    order_type: str = "market",
    limit_price: str = "",
) -> ExecutionResult:
    _ = side, quantity_kind, quote_qty
    if order_type.lower() == "limit" and limit_price:
        price = Decimal(limit_price)
    else:
        market_url = os.getenv("MARKET_DATA_URL", "http://market-data:8083")
        response = httpx.get(f"{market_url}/ticker", params={"symbol": symbol}, timeout=5.0)
        response.raise_for_status()
        price = Decimal(response.json()["price"])
    qty = Decimal(base_qty)
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


def _ibkr_paper_stub_execute(
    request: ExecutionRequest,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    mode: str,
    market_url: str,
) -> ExecutionResult:
    if mode == "live":
        result = _error_result(request, "ibkr live trading not yet available")
        _persist_execution(request, result)
        return result
    try:
        response = httpx.get(
            f"{market_url}/ticker",
            params={"symbol": symbol, "asset_type": "stock"},
            timeout=5.0,
        )
        response.raise_for_status()
        price = Decimal(str(response.json().get("price") or ""))
    except (httpx.HTTPError, ValueError, KeyError, InvalidOperation):
        result = _error_result(request, f"market data unavailable for {symbol}")
        _persist_execution(request, result)
        return result
    if price <= 0:
        result = _error_result(request, f"invalid price for {symbol}")
        _persist_execution(request, result)
        return result
    if quantity_kind == "base":
        qty = Decimal(base_qty)
    else:
        qty = (Decimal(quote_qty) / price).quantize(Decimal("0.0001"))
    if qty <= 0:
        result = _error_result(request, "computed quantity is zero")
        _persist_execution(request, result)
        return result
    result = ExecutionResult(
        execution_id=request.execution_id,
        intent_id=request.intent_id,
        decision_id=request.decision_id,
        idempotency_key=request.idempotency_key,
        status="simulated",
        venue_order_id=f"ibkr-paper-{uuid4().hex[:12]}",
        fills=[Fill(price=price, qty=qty, fee=Decimal("0"), fee_asset="USD", ts=_now())],
        avg_price=price,
        filled_qty=qty,
        remaining_qty=Decimal("0"),
        error=None,
        raw_venue_response_ref=None,
        finalized_at=_now(),
    )
    _persist_execution(request, result)
    return result


def _bridge_status(status: str) -> Literal[
    "filled", "partial", "rejected", "canceled", "error", "open"
]:
    status_map: dict[str, Literal["filled", "partial", "rejected", "canceled", "error", "open"]] = {
        "filled": "filled",
        "partial": "partial",
        "rejected": "rejected",
        "canceled": "canceled",
        "cancelled": "canceled",
        "error": "error",
        "pending": "open",
        "submitted": "open",
    }
    return status_map.get(status.lower(), "open")


def _bridge_result(
    request: ExecutionRequest, raw: dict[str, object], fallback_error: str | None = None
) -> ExecutionResult:
    fills: list[Fill] = []
    raw_fills = raw.get("fills", [])
    if not isinstance(raw_fills, list):
        raw_fills = []
    for fill in raw_fills:
        if not isinstance(fill, dict):
            continue
        fills.append(
            Fill(
                price=Decimal(str(fill["price"])),
                qty=Decimal(str(fill["qty"])),
                fee=Decimal(str(fill.get("fee", "0"))),
                fee_asset=str(fill.get("fee_asset", "USD")),
                ts=datetime.fromisoformat(str(fill["ts"]).replace("Z", "+00:00")),
            )
        )
    avg_price_raw = raw.get("avg_price")
    return ExecutionResult(
        execution_id=request.execution_id,
        intent_id=request.intent_id,
        decision_id=request.decision_id,
        idempotency_key=request.idempotency_key,
        status=_bridge_status(str(raw.get("status", "open"))),
        venue_order_id=str(raw["id"]) if raw.get("id") is not None else None,
        fills=fills,
        avg_price=Decimal(str(avg_price_raw)) if avg_price_raw is not None else None,
        filled_qty=Decimal(str(raw.get("filled_qty", "0"))),
        remaining_qty=Decimal(str(raw.get("remaining_qty", "0"))),
        error=str(raw["error"]) if raw.get("error") is not None else fallback_error,
        raw_venue_response_ref=str(raw["raw_order_ref"])
        if raw.get("raw_order_ref") is not None
        else None,
        finalized_at=_now(),
    )


def _ibkr_bridge_quantity(
    symbol: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
) -> Decimal:
    if quantity_kind == "base":
        return Decimal(base_qty)
    response = httpx.get(f"{IBKR_BRIDGE_URL}/tickers/{symbol.upper()}", timeout=5.0)
    response.raise_for_status()
    price = Decimal(str(response.json()["price"]))
    if price <= 0:
        raise ValueError("invalid price")
    return (Decimal(quote_qty) / price).quantize(Decimal("0.0001"))


def _poll_ibkr_order(order_id: str) -> dict[str, object]:
    deadline = time_lib.monotonic() + IBKR_POLL_TIMEOUT_SEC
    latest: dict[str, object] = {"id": order_id, "status": "submitted"}
    while time_lib.monotonic() < deadline:
        response = httpx.get(f"{IBKR_BRIDGE_URL}/orders/{order_id}", timeout=5.0)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("unexpected IBKR bridge response")
        latest = body
        terminal = {"filled", "partial", "rejected", "canceled", "error"}
        if str(body.get("status", "")).lower() in terminal:
            return body
        time_lib.sleep(IBKR_POLL_INTERVAL_SEC)
    return latest


def _ibkr_bridge_execute(
    request: ExecutionRequest,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    order_type: str,
    limit_price: str,
    time_in_force: str,
    mode: str,
) -> ExecutionResult:
    if mode == "live" and not IBKR_LIVE_TRADING_ENABLED:
        result = _error_result(request, "ibkr live trading not enabled in execution service")
        _persist_execution(request, result)
        return result
    try:
        quantity = _ibkr_bridge_quantity(symbol, quantity_kind, base_qty, quote_qty)
        if quantity <= 0:
            raise ValueError("computed quantity is zero")
        payload: dict[str, object] = {
            "idempotency_key": request.idempotency_key,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "order_type": order_type.lower(),
            "quantity": str(quantity),
            "limit_price": limit_price or None,
            "time_in_force": time_in_force.upper(),
        }
        response = httpx.post(f"{IBKR_BRIDGE_URL}/orders", json=payload, timeout=10.0)
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("unexpected IBKR bridge response")
        if str(raw.get("status", "")).lower() in {"pending", "submitted"} and raw.get("id"):
            raw = _poll_ibkr_order(str(raw["id"]))
        result = _bridge_result(request, raw)
    except (httpx.HTTPError, KeyError, ValueError, InvalidOperation) as exc:
        result = _error_result(request, f"ibkr bridge unavailable: {exc}"[:500])
    _persist_execution(request, result)
    return result


def _ibkr_execute(
    request: ExecutionRequest,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    mode: str,
    market_url: str,
    order_type: str,
    limit_price: str,
    time_in_force: str,
) -> ExecutionResult:
    if IBKR_MODE == "bridge":
        return _ibkr_bridge_execute(
            request,
            symbol,
            side,
            quantity_kind,
            base_qty,
            quote_qty,
            order_type,
            limit_price,
            time_in_force,
            mode,
        )
    if mode == "live":
        result = _error_result(request, "ibkr live trading requires bridge mode")
        _persist_execution(request, result)
        return result
    return _ibkr_paper_stub_execute(
        request, symbol, side, quantity_kind, base_qty, quote_qty, mode, market_url
    )


def _route_execution(
    request: ExecutionRequest,
    venue: str,
    mode: str,
    symbol: str,
    side: str,
    quantity_kind: str,
    base_qty: str,
    quote_qty: str,
    order_type: str,
    limit_price: str,
    time_in_force: str,
) -> ExecutionResult:
    if venue == "ibkr_us_equity":
        return _ibkr_execute(
            request,
            symbol,
            side,
            quantity_kind,
            base_qty,
            quote_qty,
            mode,
            os.getenv("MARKET_DATA_URL", "http://market-data:8083"),
            order_type,
            limit_price,
            time_in_force,
        )
    if venue != "binance_spot":
        result = _error_result(request, f"unsupported venue: {venue}")
        _persist_execution(request, result)
        return result
    if mode == "live":
        if not LIVE_TRADING_ENABLED:
            raise HTTPException(status_code=403, detail={"code": "LIVE_TRADING_DISABLED"})
        return _execute_live(
            request,
            symbol,
            side,
            quantity_kind,
            base_qty,
            quote_qty,
            order_type,
            limit_price,
            time_in_force,
        )
    return _binance_paper_execute(
        request,
        symbol,
        side,
        quantity_kind,
        base_qty,
        quote_qty,
        order_type,
        limit_price,
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
    x_venue: str = Header(default="binance_spot"),
    x_symbol: str = Header(default="BTCUSDT"),
    x_venue_order_id: str = Header(default=""),
    x_order_type: str = Header(default="limit"),
) -> ExecutionResult:
    if x_venue == "ibkr_us_equity" and IBKR_MODE == "bridge":
        try:
            response = httpx.delete(f"{IBKR_BRIDGE_URL}/orders/{x_venue_order_id}", timeout=5.0)
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, dict):
                raise ValueError("unexpected IBKR bridge response")
            result = _bridge_result(request, raw)
        except Exception as exc:  # noqa: BLE001
            result = _error_result(request, str(exc)[:500])
        _persist_execution(request, result)
        return result
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
    x_venue: str = Header(default="binance_spot"),
    x_symbol: str = Header(default="BTCUSDT"),
    x_venue_order_id: str = Header(default=""),
    x_order_type: str = Header(default="limit"),
) -> ExecutionResult:
    if x_venue == "ibkr_us_equity" and IBKR_MODE == "bridge":
        try:
            response = httpx.get(f"{IBKR_BRIDGE_URL}/orders/{x_venue_order_id}", timeout=5.0)
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, dict):
                raise ValueError("unexpected IBKR bridge response")
            result = _bridge_result(request, raw)
        except Exception as exc:  # noqa: BLE001
            result = _error_result(request, str(exc)[:500])
        _persist_execution(request, result)
        return result
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
    x_venue: str = Header(default="binance_spot"),
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
    if x_mode == "live" and LIVE_TRADING_ENABLED:
        _validate_confirmation_token(
            request,
            venue=x_venue,
            symbol=x_symbol,
            side=x_side,
            quantity_kind=x_quantity_kind,
            base_qty=x_quantity,
            quote_qty=x_quote_qty,
            order_type=x_order_type,
            limit_price=x_limit_price,
            time_in_force=x_time_in_force,
        )

    result = _route_execution(
        request,
        x_venue,
        x_mode,
        x_symbol,
        x_side,
        x_quantity_kind,
        x_quantity,
        x_quote_qty,
        x_order_type,
        x_limit_price,
        x_time_in_force,
    )
    _record_accepted_order(
        request,
        result,
        mode=x_mode,
        venue=x_venue,
        symbol=x_symbol,
        side=x_side,
        quantity_kind=x_quantity_kind,
        base_qty=x_quantity,
        quote_qty=x_quote_qty,
        limit_price=x_limit_price,
    )
    return result
