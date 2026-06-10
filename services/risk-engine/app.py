from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import HardCapsApplied, OrderIntent, RiskDecision, RiskReason

app = FastAPI(title="risk-engine", version="0.1.0")

MARKET_DATA_URL = os.getenv("MARKET_DATA_URL", "http://market-data:8083")
MAX_NOTIONAL_USDT = Decimal(os.getenv("MAX_NOTIONAL_USDT", "10000"))
CONFIRMATION_THRESHOLD_USDT = Decimal(os.getenv("CONFIRMATION_THRESHOLD_USDT", "500"))
PER_SYMBOL_DAILY_LIMIT_USDT = Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", "50000"))
DEFAULT_CONFIRMATION_TOKEN_SECRET = "aegis-local-confirmation-token-secret"
CONFIRMATION_TOKEN_SECRET = os.getenv(
    "CONFIRMATION_TOKEN_SECRET", DEFAULT_CONFIRMATION_TOKEN_SECRET
)
CONFIRMATION_TOKEN_TTL = timedelta(minutes=5)
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
IBKR_LIVE_TRADING_ENABLED = (
    os.getenv("IBKR_LIVE_TRADING_ENABLED", "false").lower() == "true"
)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
DAILY_DRAWDOWN_HARD_STOP_USDT = Decimal(
    os.getenv("DAILY_DRAWDOWN_HARD_STOP_USDT", "1000")
)
SUPPORTED_VENUES = {"binance_spot", "ibkr_us_equity"}
LIVE_AVAILABLE_VENUES = {"binance_spot", "ibkr_us_equity"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


def _hard_caps() -> HardCapsApplied:
    return HardCapsApplied(
        max_notional=MAX_NOTIONAL_USDT,
        max_leverage=None,
        max_drawdown_today=DAILY_DRAWDOWN_HARD_STOP_USDT,
        per_symbol_exposure=PER_SYMBOL_DAILY_LIMIT_USDT,
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def validate_confirmation_token_secret(secret: str | None = None) -> None:
    resolved = CONFIRMATION_TOKEN_SECRET if secret is None else secret
    if not resolved or resolved == DEFAULT_CONFIRMATION_TOKEN_SECRET:
        raise ValueError("CONFIRMATION_SECRET_NOT_SET")


def _db_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/aegis-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    new_path = data_dir / "trading.sqlite"
    old_path = data_dir / "phase1.sqlite"
    if not new_path.exists() and old_path.exists():
        old_path.replace(new_path)
    return new_path


def _init_accepted_order_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists accepted_order_ledger (
            intent_id text primary key,
            execution_id text not null,
            mode text not null,
            venue text not null,
            symbol text not null,
            side text not null,
            notional text not null,
            accepted_at text not null
        )
        """
    )
    conn.commit()


def _accepted_symbol_notional_today(
    *,
    symbol: str,
    mode: str,
    intent_id: str,
    as_of: datetime,
) -> Decimal:
    day_start = as_of.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    with sqlite3.connect(_db_path()) as conn:
        _init_accepted_order_ledger(conn)
        rows = conn.execute(
            """
            select notional from accepted_order_ledger
            where upper(symbol) = ?
              and mode = ?
              and intent_id != ?
              and accepted_at >= ?
              and accepted_at < ?
            """,
            (symbol.upper(), mode, intent_id, day_start.isoformat(), day_end.isoformat()),
        ).fetchall()
    total = Decimal("0")
    for row in rows:
        total += Decimal(str(row[0]))
    return total


def _per_symbol_daily_limit_check(
    intent: OrderIntent,
    *,
    evaluated_at: datetime,
    notional: Decimal,
) -> RiskReason | None:
    try:
        existing = _accepted_symbol_notional_today(
            symbol=intent.symbol,
            mode=intent.mode,
            intent_id=str(intent.intent_id),
            as_of=evaluated_at,
        )
    except (sqlite3.Error, OSError, ValueError):
        if intent.mode == "live":
            return RiskReason(
                code="PER_SYMBOL_LEDGER_UNAVAILABLE",
                detail="could not read accepted order ledger; live blocked for safety",
            )
        return None
    projected = existing + notional
    if projected > PER_SYMBOL_DAILY_LIMIT_USDT:
        return RiskReason(
            code="PER_SYMBOL_DAILY_LIMIT_BREACHED",
            detail=(
                f"{intent.symbol.upper()} {intent.mode} accepted notional today "
                f"{existing} + {notional} exceeds per-symbol daily limit "
                f"{PER_SYMBOL_DAILY_LIMIT_USDT}"
            ),
        )
    return None


def _confirmation_token(
    intent: OrderIntent,
    evaluated_at: datetime,
    expires_at: datetime,
    notional: Decimal,
) -> str:
    payload: dict[str, object] = {
        "version": 1,
        "intent_id": str(intent.intent_id),
        "venue": intent.venue,
        "symbol": intent.symbol.upper(),
        "side": intent.side,
        "order_type": intent.order_type,
        "quantity_kind": intent.quantity.kind,
        "quantity_value": str(intent.quantity.value),
        "limit_price": str(intent.limit_price) if intent.limit_price is not None else None,
        "time_in_force": intent.time_in_force,
        "notional": str(notional),
        "issued_at": evaluated_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }
    encoded_payload = _b64url(_canonical_json(payload))
    signature = hmac.new(
        CONFIRMATION_TOKEN_SECRET.encode(),
        encoded_payload.encode(),
        hashlib.sha256,
    ).digest()
    return f"{encoded_payload}.{_b64url(signature)}"


def _decision(
    intent: OrderIntent,
    evaluated_at: datetime,
    approved: bool,
    reasons: list[RiskReason],
    requires_confirmation: bool = False,
    confirmation_token: str | None = None,
    confirmation_expires_at: datetime | None = None,
) -> RiskDecision:
    return RiskDecision(
        decision_id=uuid4(),
        intent_id=intent.intent_id,
        evaluated_at=evaluated_at,
        approved=approved,
        reasons=reasons,
        requires_confirmation=requires_confirmation,
        confirmation_token=confirmation_token,
        confirmation_expires_at=confirmation_expires_at,
        hard_caps_applied=_hard_caps(),
    )


def _notional(intent: OrderIntent) -> Decimal | RiskReason:
    if intent.quantity.kind == "quote":
        return intent.quantity.value
    try:
        params = {"symbol": intent.symbol}
        if intent.venue == "ibkr_us_equity":
            params["asset_type"] = "stock"
        response = httpx.get(
            f"{MARKET_DATA_URL}/ticker", params=params, timeout=3.0
        )
        response.raise_for_status()
        price = Decimal(str(response.json()["price"]))
    except (httpx.HTTPError, KeyError, ValueError):
        return RiskReason(code="MARKET_DATA_UNAVAILABLE", detail="market data unavailable")
    return intent.quantity.value * price


def _drawdown_check(intent: OrderIntent) -> RiskReason | None:
    """Return a RiskReason when today's actor PnL has breached the hard stop."""
    try:
        response = httpx.get(
            f"{ORCHESTRATOR_URL}/pnl/today",
            params={"actor": intent.actor},
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        total_pnl = Decimal(str(payload["total_pnl"]))
    except (httpx.HTTPError, KeyError, ValueError):
        if intent.mode == "live":
            return RiskReason(
                code="DRAWDOWN_CHECK_UNAVAILABLE",
                detail="could not fetch today's PnL; live blocked for safety",
            )
        return None
    if total_pnl <= -DAILY_DRAWDOWN_HARD_STOP_USDT:
        return RiskReason(
            code="DAILY_DRAWDOWN_BREACHED",
            detail=(
                f"actor PnL today {total_pnl} <= -{DAILY_DRAWDOWN_HARD_STOP_USDT}; "
                "trading paused for the day"
            ),
        )
    return None


def evaluate(intent: OrderIntent) -> RiskDecision:
    evaluated_at = _now()
    reasons: list[RiskReason] = []
    if intent.mode == "live":
        if not LIVE_TRADING_ENABLED:
            reasons.append(
                RiskReason(
                    code="LIVE_TRADING_DISABLED",
                    detail="set LIVE_TRADING_ENABLED=true to enable live trading",
                )
            )
            return _decision(intent, evaluated_at, False, reasons)
        try:
            validate_confirmation_token_secret()
        except ValueError:
            reasons.append(
                RiskReason(
                    code="CONFIRMATION_SECRET_NOT_SET",
                    detail=(
                        "set CONFIRMATION_TOKEN_SECRET to a non-default value "
                        "before live trading"
                    ),
                )
            )
            return _decision(intent, evaluated_at, False, reasons)
        if intent.venue == "ibkr_us_equity" and not IBKR_LIVE_TRADING_ENABLED:
            reasons.append(
                RiskReason(
                    code="IBKR_LIVE_TRADING_DISABLED",
                    detail="set IBKR_LIVE_TRADING_ENABLED=true to enable IBKR live trading",
                )
            )
            return _decision(intent, evaluated_at, False, reasons)
    if intent.order_type == "limit" and intent.limit_price is None:
        reasons.append(
            RiskReason(
                code="LIMIT_PRICE_REQUIRED",
                detail="limit orders require limit_price to be set",
            )
        )
        return _decision(intent, evaluated_at, False, reasons)
    if intent.venue not in SUPPORTED_VENUES:
        reasons.append(
            RiskReason(code="UNSUPPORTED_VENUE", detail=intent.venue)
        )
        return _decision(intent, evaluated_at, False, reasons)

    drawdown = _drawdown_check(intent)
    if drawdown is not None:
        reasons.append(drawdown)
        return _decision(intent, evaluated_at, False, reasons)

    notional = _notional(intent)
    if isinstance(notional, RiskReason):
        return _decision(intent, evaluated_at, False, [notional])
    if notional > MAX_NOTIONAL_USDT:
        reasons.append(
            RiskReason(
                code="NOTIONAL_EXCEEDS_HARD_CAP",
                detail=f"notional {notional} exceeds max {MAX_NOTIONAL_USDT}",
            )
        )
    per_symbol = _per_symbol_daily_limit_check(
        intent,
        evaluated_at=evaluated_at,
        notional=notional,
    )
    if per_symbol is not None:
        reasons.append(per_symbol)
    if reasons:
        return _decision(intent, evaluated_at, False, reasons)

    requires_confirmation = (
        intent.mode == "live" or notional >= CONFIRMATION_THRESHOLD_USDT
    )
    expires_at = evaluated_at + CONFIRMATION_TOKEN_TTL if requires_confirmation else None
    token = (
        _confirmation_token(intent, evaluated_at, expires_at, notional)
        if expires_at is not None
        else None
    )
    return _decision(
        intent,
        evaluated_at,
        True,
        [],
        requires_confirmation=requires_confirmation,
        confirmation_token=token,
        confirmation_expires_at=expires_at,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/validate", response_model=RiskDecision)
def validate(intent: OrderIntent) -> RiskDecision:
    return evaluate(intent)
