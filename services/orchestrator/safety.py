from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid4

from fastapi.responses import JSONResponse

ConnectFn = Callable[[], sqlite3.Connection]
NowFn = Callable[[], datetime]
PnlTodayFn = Callable[..., JSONResponse | dict[str, object]]
QuantizeFn = Callable[[Decimal], Decimal]


def consume_live_unlock_or_error(
    *,
    token: str,
    actor: str,
    dry: bool,
    intent_id: UUID | None,
    connect: ConnectFn,
    now: NowFn,
) -> JSONResponse | None:
    if not token:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_REQUIRED"})
    now_iso = now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select actor, expires_at, consumed_at, bound_intent_id "
            "from live_unlock_tokens where token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=403, content={"code": "INVALID_LIVE_UNLOCK"})
    if row["actor"] != actor:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ACTOR_MISMATCH"})
    if row["consumed_at"] is not None:
        return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ALREADY_USED"})
    if row["expires_at"] < now_iso:
        return JSONResponse(status_code=410, content={"code": "LIVE_UNLOCK_EXPIRED"})
    if row["bound_intent_id"] is not None:
        if intent_id is None or row["bound_intent_id"] != str(intent_id):
            return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_INTENT_MISMATCH"})
    if not dry:
        with connect() as conn:
            cursor = conn.execute(
                "update live_unlock_tokens set consumed_at = ? "
                "where token = ? and consumed_at is NULL",
                (now_iso, token),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse(status_code=403, content={"code": "LIVE_UNLOCK_ALREADY_USED"})
    return None


def scorecard_should_mark_consumed(response: JSONResponse | dict[str, object]) -> bool:
    if isinstance(response, dict):
        return True
    code = getattr(response, "status_code", 500)
    if code == 202:
        return True
    if code == 422:
        body = json.loads(bytes(response.body).decode())
        if body.get("code") == "RISK_REJECTED":
            return True
    return False


def live_kill_switch_active(*, connect: ConnectFn) -> bool:
    with connect() as conn:
        row = conn.execute("select killed from live_autonomy_kill where id = 1").fetchone()
    return bool(row and row["killed"])


def check_drawdown_for_live_auto(
    *, actor: str, get_pnl_today: PnlTodayFn, daily_drawdown_hard_stop: Decimal
) -> str | None:
    try:
        body = get_pnl_today(actor=actor)
    except Exception:
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    if isinstance(body, JSONResponse):
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    try:
        total = Decimal(str(body.get("total_pnl", "0")))
    except (InvalidOperation, ValueError):
        return "DRAWDOWN_CHECK_UNAVAILABLE"
    if total <= -daily_drawdown_hard_stop:
        return "DAILY_DRAWDOWN_BREACHED"
    return None


def check_live_exposure_cap(
    *,
    actor: str,
    proposed_notional: Decimal,
    max_cap: Decimal,
    venue: str,
    connect: ConnectFn,
    quantize: QuantizeFn,
) -> tuple[bool, Decimal]:
    with connect() as conn:
        rows = conn.execute(
            """
            select p.live_qty, p.live_avg_cost
            from paper_positions p
            where p.actor = ?
              and p.venue = ?
              and cast(p.live_qty as real) > 0
            """,
            (actor, venue),
        ).fetchall()
    current = Decimal("0")
    for row in rows:
        current += Decimal(str(row["live_qty"])) * Decimal(str(row["live_avg_cost"]))
    current = quantize(current)
    return current + proposed_notional <= max_cap, current


def mint_user_live_unlock_token(
    *, actor: str, connect: ConnectFn, now: NowFn, ttl_min: int
) -> str:
    token = str(uuid4())
    created = now()
    expires = created + timedelta(minutes=ttl_min)
    with connect() as conn:
        conn.execute(
            "insert into live_unlock_tokens "
            "(token, actor, created_at, expires_at, consumed_at, bound_intent_id) "
            "values (?, ?, ?, ?, NULL, NULL)",
            (token, actor, created.isoformat(), expires.isoformat()),
        )
        conn.commit()
    return token


def mint_auto_unlock_bound_token(
    *, actor: str, intent_id: UUID | str, connect: ConnectFn, now: NowFn
) -> str:
    token = str(uuid4())
    created = now()
    expires = created + timedelta(minutes=2)
    with connect() as conn:
        conn.execute(
            "insert into live_unlock_tokens "
            "(token, actor, created_at, expires_at, consumed_at, bound_intent_id) "
            "values (?, ?, ?, ?, NULL, ?)",
            (token, actor, created.isoformat(), expires.isoformat(), str(intent_id)),
        )
        conn.commit()
    return token
