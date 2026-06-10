from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac as hmac_lib
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from schemas import (
    ConfirmationRequest,
    ExecutionRequest,
    ExecutionResult,
    FactorSignal,
    OrderIntent,
    Quantity,
    RiskDecision,
    Scorecard,
    Source,
)

try:
    from . import config as _orchestrator_config
    from . import db as _orchestrator_db
    from . import safety as _orchestrator_safety
except ImportError:  # pragma: no cover - legacy direct file loading path
    import config as _orchestrator_config  # type: ignore[import-not-found,no-redef]
    import db as _orchestrator_db  # type: ignore[import-not-found,no-redef]
    import safety as _orchestrator_safety  # type: ignore[import-not-found,no-redef]

_env_bool = _orchestrator_config.env_bool
_env_decimal = _orchestrator_config.env_decimal
_env_int = _orchestrator_config.env_int
connect = _orchestrator_db.connect
write_apply_log = _orchestrator_db.write_apply_log


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    bootstrap_paper_feedback_loop()
    await _start_scheduler()
    await _startup_ibkr_reconcile()
    await _warn_reconcile_apply_without_token()
    try:
        yield
    finally:
        await _stop_scheduler()


logger = logging.getLogger(__name__)
PER_SYMBOL_DAILY_LIMIT_USDT = Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", "50000"))
DECIMAL_8 = Decimal("0.00000001")
SUPPORTED_NOTIFICATION_EVENTS = frozenset({"fill", "alert"})
CALIBRATION_BUCKETS = [
    (Decimal("0.30"), Decimal("0.40")),
    (Decimal("0.40"), Decimal("0.50")),
    (Decimal("0.50"), Decimal("0.60")),
    (Decimal("0.60"), Decimal("0.70")),
    (Decimal("0.70"), Decimal("0.80")),
    (Decimal("0.80"), Decimal("0.90")),
    (Decimal("0.90"), Decimal("1.01")),
]
STANDARD_SIGNAL_GATE_CONVICTION = "0.70"
STANDARD_DATA_ORIGIN = "standard"
PAPER_FEEDBACK_BOOTSTRAP_ORIGIN = "paper_feedback_bootstrap"


class EvShadowSample(TypedDict):
    ev: Decimal
    pnl: Decimal
    data_origin: str
    gate_conviction: str | None


class _OutcomeSummaryBucket:
    def __init__(self) -> None:
        self.closed_count = 0
        self.hits = 0
        self.losses = 0
        self.open_count = 0
        self.total_pnl = Decimal("0")


CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
HERMES_MODEL = os.getenv("HERMES_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
SCORECARD_DEFAULT_TTL_MIN = int(os.getenv("SCORECARD_DEFAULT_TTL_MIN", "60"))
OPS_TOKEN: str = os.getenv("OPS_TOKEN", "")
RECONCILE_APPLY_ENABLED: bool = os.getenv("RECONCILE_APPLY_ENABLED", "false").lower() == "true"
RECONCILE_MAX_AGE_SECONDS: int = int(os.getenv("RECONCILE_MAX_AGE_SECONDS", "300"))
RECONCILE_APPLY_DEFAULT_ACTOR: str = "ibkr_reconcile"
LIVE_UNLOCK_TTL_MIN = int(os.getenv("LIVE_UNLOCK_TTL_MIN", "15"))
ANALYSIS_ADAPTER_URL = os.getenv("ANALYSIS_ADAPTER_URL", "http://analysis-adapter:8085")
REFLECT_TIMEOUT_SEC = float(os.getenv("REFLECT_TIMEOUT_SEC", "60"))
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
SCHEDULER_TICK_SEC = float(os.getenv("SCHEDULER_TICK_SEC", "60"))
SCHEDULER_BATCH_LIMIT = int(os.getenv("SCHEDULER_BATCH_LIMIT", "5"))
ORCHESTRATOR_SELF_URL = os.getenv("ORCHESTRATOR_SELF_URL", "http://localhost:8080")
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "true").lower() == "true"
AUTO_TRADE_BATCH_LIMIT = int(os.getenv("AUTO_TRADE_BATCH_LIMIT", "5"))
LIVE_AUTONOMY_GLOBAL_ENABLED = os.getenv("LIVE_AUTONOMY_GLOBAL_ENABLED", "false").lower() == "true"
IBKR_MODE = os.getenv("IBKR_MODE", "stub")
IBKR_LIVE_TRADING_ENABLED = os.getenv("IBKR_LIVE_TRADING_ENABLED", "false").lower() == "true"
IBKR_BRIDGE_URL = os.getenv("IBKR_BRIDGE_URL", "http://ibkr-bridge:8086").rstrip("/")
RECONCILE_AVG_COST_TOLERANCE = Decimal(os.getenv("RECONCILE_AVG_COST_TOLERANCE", "0.01"))
LIVE_AUTO_TICK_SEC = float(os.getenv("LIVE_AUTO_TICK_SEC", "300"))
LIVE_AUTO_BATCH_LIMIT = int(os.getenv("LIVE_AUTO_BATCH_LIMIT", "1"))
STOP_LOSS_WATCHDOG_ENABLED = os.getenv("STOP_LOSS_WATCHDOG_ENABLED", "true").lower() == "true"
STOP_LOSS_TICK_SEC = float(os.getenv("STOP_LOSS_TICK_SEC", "60"))
STOP_LOSS_BATCH_LIMIT = int(os.getenv("STOP_LOSS_BATCH_LIMIT", "20"))
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", "5"))
CALIBRATION_SHRINKAGE_K = Decimal(os.getenv("CALIBRATION_SHRINKAGE_K", "10"))
NOTIFICATION_HOST_ALLOWLIST = frozenset(
    item.strip()
    for item in os.getenv(
        "NOTIFICATION_HOST_ALLOWLIST", "hermes-agent,hermes,localhost,127.0.0.1"
    ).split(",")
    if item.strip()
)
NOTIFICATION_TIMEOUT_SEC = float(os.getenv("NOTIFICATION_TIMEOUT_SEC", "5"))
NOTIFICATION_HISTORY_LIMIT = int(os.getenv("NOTIFICATION_HISTORY_LIMIT", "100"))
EV_GATE_MODE = os.getenv("EV_GATE_MODE", "shadow").strip().lower()
MIN_EV = Decimal(os.getenv("MIN_EV", "0"))
EV_SHADOW_MIN_SAMPLES = int(os.getenv("EV_SHADOW_MIN_SAMPLES", "20"))
EV_SHADOW_REPORT_DIR = os.getenv("EV_SHADOW_REPORT_DIR", "aegis-strategies/incubating")
MEMORY_PNL_THRESHOLD_ENV = "MEMORY_PNL_THRESHOLD"
MEMORY_RETURN_THRESHOLD_ENV = "MEMORY_RETURN_THRESHOLD"
PAPER_FEEDBACK_BOOTSTRAP_ENABLED_ENV = "PAPER_FEEDBACK_BOOTSTRAP_ENABLED"
PAPER_FEEDBACK_ACTOR_ENV = "PAPER_FEEDBACK_ACTOR"
PAPER_FEEDBACK_SYMBOLS_ENV = "PAPER_FEEDBACK_SYMBOLS"
PAPER_FEEDBACK_DEFAULT_ASSET_TYPE_ENV = "PAPER_FEEDBACK_DEFAULT_ASSET_TYPE"
PAPER_FEEDBACK_CADENCE_MINUTES_ENV = "PAPER_FEEDBACK_CADENCE_MINUTES"
PAPER_FEEDBACK_DAILY_BUDGET_USDT_ENV = "PAPER_FEEDBACK_DAILY_BUDGET_USDT"
PAPER_FEEDBACK_PER_TRADE_USDT_ENV = "PAPER_FEEDBACK_PER_TRADE_USDT"
PAPER_FEEDBACK_MIN_CONVICTION_ENV = "PAPER_FEEDBACK_MIN_CONVICTION"
PAPER_FEEDBACK_ALLOWED_SOURCES_ENV = "PAPER_FEEDBACK_ALLOWED_SOURCES"
FEE_BPS = Decimal(os.getenv("FEE_BPS", "8"))
SLIPPAGE_BPS = Decimal(os.getenv("SLIPPAGE_BPS", "5"))
FUNDING_BPS = Decimal(os.getenv("FUNDING_BPS", "3"))
AUTONOMY_DRAWDOWN_ALERT_USDT = Decimal(os.getenv("AUTONOMY_DRAWDOWN_ALERT_USDT", "100"))
AUTONOMY_CONSECUTIVE_LOSS_ALERT_N = int(os.getenv("AUTONOMY_CONSECUTIVE_LOSS_ALERT_N", "3"))
AUTONOMY_DRAWDOWN_HALT_USDT = Decimal(os.getenv("AUTONOMY_DRAWDOWN_HALT_USDT", "25"))
AUTONOMY_CONSECUTIVE_LOSS_HALT_N = int(os.getenv("AUTONOMY_CONSECUTIVE_LOSS_HALT_N", "3"))
AUTONOMY_MARK_FAILURE_HALT_N = int(os.getenv("AUTONOMY_MARK_FAILURE_HALT_N", "3"))
AUTONOMY_DECISION_ERROR_RATE_HALT_PCT = Decimal(
    os.getenv("AUTONOMY_DECISION_ERROR_RATE_HALT_PCT", "0.50")
)
AUTONOMY_DECISION_ERROR_RATE_MIN_N = int(os.getenv("AUTONOMY_DECISION_ERROR_RATE_MIN_N", "3"))
AUTONOMY_DECISION_ERROR_LOOKBACK_MINUTES = int(
    os.getenv("AUTONOMY_DECISION_ERROR_LOOKBACK_MINUTES", "120")
)
FEEDBACK_EV_READY_N = int(os.getenv("FEEDBACK_EV_READY_N", "20"))
FEEDBACK_CLOSED_OUTCOME_READY_N = int(os.getenv("FEEDBACK_CLOSED_OUTCOME_READY_N", "20"))
FEEDBACK_CALIBRATION_READY_BUCKETS = int(os.getenv("FEEDBACK_CALIBRATION_READY_BUCKETS", "2"))
FEEDBACK_MEMORY_AUTOMATIC_READY_N = int(os.getenv("FEEDBACK_MEMORY_AUTOMATIC_READY_N", "5"))
_scheduler_task: asyncio.Task[None] | None = None

_HERMES_SYSTEM_PROMPT = """\
You are an order intent parser for a cryptocurrency spot trading platform (Binance Spot only).

Given a natural language trading instruction, extract the following fields and respond with
ONLY a valid JSON object -- no markdown, no explanation, no code fences:

{
  "symbol":        "<COIN>USDT uppercase, e.g. BTCUSDT or ETHUSDT",
  "side":          "buy" | "sell",
  "order_type":    "market" | "limit",
  "quantity_kind": "quote" | "base",
  "quantity_value": "<positive decimal string, e.g. '100' or '0.001'>",
  "limit_price":   "<decimal string>" | null
}

Rules:
- quantity_kind="quote" means the user specified a USDT amount (e.g. "100 USDT of BTC")
- quantity_kind="base"  means the user specified a coin amount (e.g. "0.001 BTC")
- limit_price must be null for market orders and a decimal string for limit orders
- Only support spot pairs quoted in USDT (append USDT if the user omits it)
- If the instruction is not a valid, unambiguous trading order, respond with exactly:
  {"error": "<one-sentence reason>"}
"""


class NLIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    message: str = PydanticField(min_length=1)
    idempotency_key: str = PydanticField(min_length=1)
    hermes_message_id: str | None = None
    mode: Literal["paper", "live"] = "paper"
    request_id: UUID | None = None


NLIntentRequest.model_rebuild(_types_namespace={"Literal": Literal, "UUID": UUID})


class ScorecardCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    symbol: str = PydanticField(min_length=1)
    action: Literal["buy", "sell", "hold"]
    source: Literal["manual", "tradingagents", "hermes_chat"]
    conviction: str = PydanticField(min_length=1)
    thesis: str = PydanticField(min_length=1, max_length=4000)
    entry_low: str | None = None
    entry_high: str | None = None
    stop_loss: str | None = None
    take_profit: str | None = None
    time_horizon: Literal["intraday", "swing", "position"]
    ttl_minutes: int | None = None
    metadata: dict[str, str] | None = None
    factors: list[FactorSignal] | None = None


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    subject: str
    content: str
    source_ref: dict[str, str]
    created_at: str
    tags: list[str]
    confidence: str | None = None
    superseded_by: str | None = None


class RecordMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["decision", "outcome", "lesson", "convention", "observation"]
    subject: str = PydanticField(min_length=1)
    content: str = PydanticField(min_length=1, max_length=4000)
    source_ref: dict[str, str] = PydanticField(min_length=1)
    tags: list[str] = PydanticField(default_factory=list)
    confidence: str | None = None
    superseded_by: str | None = None
    trigger: str | None = None
    created_by: str | None = None


class ScorecardIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scorecard_id: UUID
    actor: str = PydanticField(min_length=1)
    idempotency_key: str = PydanticField(min_length=1)
    mode: Literal["paper", "live"] = "paper"
    usdt_budget: str = PydanticField(min_length=1)
    position_fraction: str = "1.0"
    order_type: Literal["market", "limit"] = "market"
    request_id: UUID | None = None
    intent_id: UUID | None = None


class LiveAutonomyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    enabled: bool | None = None
    daily_live_budget_usdt: str | None = None
    per_live_trade_max_usdt: str | None = None
    max_live_exposure_usdt: str | None = None
    max_us_equity_exposure_usd: str | None = None
    daily_live_trade_count_max: int | None = None
    min_calibrated_conviction: str | None = None
    min_closed_outcomes: int | None = None
    allowed_sources: str | None = None


class NotificationSubscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    webhook_url: str = PydanticField(min_length=1)
    secret: str | None = None
    events: list[str] = PydanticField(default_factory=lambda: ["fill"])


class TrailingStopUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trailing_pct: str = PydanticField(min_length=1)


class AutonomyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    enabled: bool | None = None
    daily_budget_usdt: str | None = None
    min_conviction: str | None = None
    per_trade_usdt: str | None = None
    allowed_sources: str | None = None


class WatchlistAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)
    symbol: str = PydanticField(min_length=1)
    asset_type: Literal["stock", "crypto"] = "crypto"
    cadence_minutes: int = PydanticField(ge=15, le=1440)


class LiveUnlockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = PydanticField(min_length=1)


class ReconcileApplyResponse(BaseModel):
    applied_at: str
    reconcile_run_at: str
    total_patched: int
    patches: list[dict[str, object]]


ScorecardCreateRequest.model_rebuild(
    _types_namespace={"Literal": Literal, "FactorSignal": FactorSignal}
)
RecordMemoryRequest.model_rebuild(_types_namespace={"Literal": Literal})
ScorecardIntentRequest.model_rebuild(_types_namespace={"Literal": Literal, "UUID": UUID})
LiveAutonomyUpdateRequest.model_rebuild()
NotificationSubscribeRequest.model_rebuild()
TrailingStopUpdateRequest.model_rebuild()
AutonomyUpdateRequest.model_rebuild()
WatchlistAddRequest.model_rebuild(_types_namespace={"Literal": Literal})
LiveUnlockRequest.model_rebuild()
ReconcileApplyResponse.model_rebuild()


class DuplicateIntentIdError(Exception):
    pass


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


def _bucket_label(heuristic: Decimal) -> str | None:
    for lo, hi in CALIBRATION_BUCKETS:
        if lo <= heuristic < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def _calibration_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "source": row["source"],
        "asset_type": row["asset_type"],
        "heuristic_bucket": row["heuristic_bucket"],
        "sample_count": row["sample_count"],
        "hit_count": row["hit_count"],
        "avg_alpha_return": row["avg_alpha_return"],
        "empirical_hit_rate": row["empirical_hit_rate"],
        "calibrated_conviction": row["calibrated_conviction"],
        "updated_at": row["updated_at"],
    }


def _calibration_origin_breakdown(
    source: str | None = None, asset_type: str | None = None
) -> dict[tuple[str, str, str], dict[str, object]]:
    clauses = ["o.status = 'closed'"]
    params: list[object] = []
    if source:
        clauses.append("o.source = ?")
        params.append(source)
    where = " where " + " and ".join(clauses)
    with connect() as conn:
        rows = conn.execute(
            f"""
            select o.source, s.payload_json
            from scorecard_outcomes o
            join scorecards s on s.scorecard_id = o.scorecard_id
            {where}
            """,
            params,
        ).fetchall()
    breakdown: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        metadata = _scorecard_metadata_from_payload(payload)
        item_asset_type = str(metadata.get("asset_type") or "crypto")
        if asset_type and item_asset_type != asset_type:
            continue
        heuristic_raw = metadata.get("heuristic_conviction") or payload.get("conviction")
        try:
            heuristic = Decimal(str(heuristic_raw))
        except (InvalidOperation, ValueError):
            continue
        bucket = _bucket_label(heuristic)
        if bucket is None:
            continue
        key = (str(row["source"]), item_asset_type, bucket)
        origin = _metadata_data_origin(metadata)
        gate = _metadata_gate_conviction(metadata) or "standard"
        origins = breakdown.setdefault(key, {})
        origin_bucket = cast(dict[str, object], origins.setdefault(origin, {"n": 0, "gates": {}}))
        origin_bucket["n"] = cast(int, origin_bucket["n"]) + 1
        gates = cast(dict[str, int], origin_bucket["gates"])
        gates[gate] = gates.get(gate, 0) + 1
    return breakdown


def _serializable_calibration_breakdown(
    breakdown: dict[tuple[str, str, str], dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        f"{source}:{asset_type}:{bucket}": origins
        for (source, asset_type, bucket), origins in sorted(breakdown.items())
    }


def _memory_scorecard_payload(row: sqlite3.Row) -> dict[str, object]:
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _scorecard_metadata_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw_metadata = payload.get("metadata")
    return raw_metadata if isinstance(raw_metadata, dict) else {}


def _metadata_data_origin(metadata: Mapping[str, object]) -> str:
    origin = str(metadata.get("origin") or "").strip()
    return origin or STANDARD_DATA_ORIGIN


def _metadata_gate_conviction(metadata: Mapping[str, object]) -> str | None:
    raw = metadata.get("gate_conviction")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _scorecard_origin_tags(metadata: Mapping[str, object]) -> list[str]:
    origin = _metadata_data_origin(metadata)
    gate_conviction = _metadata_gate_conviction(metadata)
    return _memory_tags(
        f"origin:{origin}",
        f"gate_conviction:{gate_conviction}" if gate_conviction else None,
    )


def _memory_tags(*items: object) -> list[str]:
    tags: list[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, list):
            tags.extend(_memory_tags(*item))
            continue
        value = str(item).strip()
        if value and value not in tags:
            tags.append(value)
    return tags


def _memory_factor_tags(payload: dict[str, object]) -> list[str]:
    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, list):
        return []
    tags: list[str] = []
    for raw in raw_factors:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        direction = raw.get("direction")
        if isinstance(name, str) and name:
            tags.append(f"factor:{name}")
        if isinstance(direction, str) and direction:
            tags.append(f"direction:{direction}")
    return tags


def _memory_threshold() -> Decimal:
    try:
        return abs(Decimal(os.getenv(MEMORY_PNL_THRESHOLD_ENV, "10")))
    except (InvalidOperation, ValueError):
        return Decimal("10")


def _memory_return_threshold() -> Decimal:
    try:
        return abs(Decimal(os.getenv(MEMORY_RETURN_THRESHOLD_ENV, "0.02")))
    except (InvalidOperation, ValueError):
        return Decimal("0.02")


def _paper_feedback_symbol_pool() -> list[tuple[str, str]]:
    raw = os.getenv(PAPER_FEEDBACK_SYMBOLS_ENV, "BTCUSDT,ETHUSDT,SOLUSDT")
    default_asset_type = os.getenv(PAPER_FEEDBACK_DEFAULT_ASSET_TYPE_ENV, "crypto").strip().lower()
    if default_asset_type not in {"crypto", "stock"}:
        default_asset_type = "crypto"
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if ":" in value:
            symbol, asset_type = [part.strip() for part in value.split(":", 1)]
            asset_type = asset_type.lower()
        else:
            symbol = value
            asset_type = default_asset_type
        symbol = symbol.upper()
        if asset_type not in {"crypto", "stock"} or not symbol or symbol in seen:
            continue
        seen.add(symbol)
        entries.append((symbol, asset_type))
    return entries


def _paper_feedback_runtime_settings() -> tuple[int, Decimal, Decimal, Decimal, str]:
    cadence = _env_int(PAPER_FEEDBACK_CADENCE_MINUTES_ENV, 240, minimum=15, maximum=1440)
    daily_budget = max(Decimal("0"), _env_decimal(PAPER_FEEDBACK_DAILY_BUDGET_USDT_ENV, "50"))
    per_trade = max(Decimal("0.00000001"), _env_decimal(PAPER_FEEDBACK_PER_TRADE_USDT_ENV, "10"))
    min_conviction = _env_decimal(PAPER_FEEDBACK_MIN_CONVICTION_ENV, "0.65")
    min_conviction = max(Decimal("0"), min(Decimal("1"), min_conviction))
    allowed_sources = os.getenv(PAPER_FEEDBACK_ALLOWED_SOURCES_ENV, "tradingagents").strip()
    if not allowed_sources:
        allowed_sources = "tradingagents"
    return cadence, daily_budget, per_trade, min_conviction, allowed_sources


def _disable_paper_feedback_bootstrap(actor: str, now: datetime) -> dict[str, object]:
    _, daily_budget, per_trade, min_conviction, allowed_sources = _paper_feedback_runtime_settings()
    with connect() as conn:
        watchlist_cursor = conn.execute(
            "update watchlist_entries set enabled = 0 "
            "where actor = ? and source_origin = ? and enabled = 1",
            (actor, PAPER_FEEDBACK_BOOTSTRAP_ORIGIN),
        )
        autonomy_cursor = conn.execute(
            """
            update autonomy_settings
            set enabled = 0, updated_at = ?
            where actor = ?
              and enabled = 1
              and daily_budget_usdt = ?
              and min_conviction = ?
              and per_trade_usdt = ?
              and allowed_sources = ?
            """,
            (
                now.isoformat(),
                actor,
                str(daily_budget),
                str(min_conviction),
                str(per_trade),
                allowed_sources,
            ),
        )
        conn.commit()
    return {
        "disabled_watchlist_entries": watchlist_cursor.rowcount,
        "disabled_autonomy_settings": autonomy_cursor.rowcount,
    }


def _paper_bootstrap_halt_row(actor: str) -> sqlite3.Row | None:
    with connect() as conn:
        row = conn.execute(
            "select actor, halted, reason, observed_json, halted_at, resumed_at, "
            "resumed_by, updated_at from paper_bootstrap_halts where actor = ?",
            (actor,),
        ).fetchone()
    return cast(sqlite3.Row | None, row)


def _paper_bootstrap_halt_status(actor: str) -> dict[str, object]:
    row = _paper_bootstrap_halt_row(actor)
    if row is None:
        return {
            "actor": actor,
            "halted": False,
            "reason": None,
            "observed": {},
            "halted_at": None,
            "resumed_at": None,
            "resumed_by": None,
        }
    try:
        observed = json.loads(str(row["observed_json"]))
    except json.JSONDecodeError:
        observed = {}
    return {
        "actor": actor,
        "halted": bool(row["halted"]),
        "reason": row["reason"],
        "observed": observed if isinstance(observed, dict) else {},
        "halted_at": row["halted_at"],
        "resumed_at": row["resumed_at"],
        "resumed_by": row["resumed_by"],
    }


def _paper_bootstrap_guardrail_state(actor: str) -> dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select mark_failure_streak, decision_error_streak, last_error_rate, "
            "last_evaluated_at from paper_bootstrap_guardrail_state where actor = ?",
            (actor,),
        ).fetchone()
    if row is None:
        return {
            "mark_failure_streak": 0,
            "decision_error_streak": 0,
            "last_error_rate": None,
            "last_evaluated_at": None,
        }
    return {
        "mark_failure_streak": int(row["mark_failure_streak"]),
        "decision_error_streak": int(row["decision_error_streak"]),
        "last_error_rate": row["last_error_rate"],
        "last_evaluated_at": row["last_evaluated_at"],
    }


def _write_paper_bootstrap_guardrail_state(
    actor: str,
    *,
    mark_failure_streak: int,
    decision_error_streak: int,
    last_error_rate: str | None,
    evaluated_at: datetime,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into paper_bootstrap_guardrail_state
              (actor, mark_failure_streak, decision_error_streak, last_error_rate,
               last_evaluated_at)
            values (?,?,?,?,?)
            on conflict(actor) do update set
              mark_failure_streak = excluded.mark_failure_streak,
              decision_error_streak = excluded.decision_error_streak,
              last_error_rate = excluded.last_error_rate,
              last_evaluated_at = excluded.last_evaluated_at
            """,
            (
                actor,
                mark_failure_streak,
                decision_error_streak,
                last_error_rate,
                evaluated_at.isoformat(),
            ),
        )
        conn.commit()


def _analysis_adapter_db_path() -> Path:
    configured = os.getenv("ANALYSIS_ADAPTER_DB_PATH")
    if configured:
        return Path(configured)
    return Path(os.getenv("DATA_DIR", "/tmp/aegis-data")) / "analysis_adapter.sqlite"


def _recent_analysis_error_rate(actor: str, now: datetime) -> dict[str, object]:
    db_path = _analysis_adapter_db_path()
    if not db_path.exists():
        return {"available": False, "total": 0, "failed": 0, "failure_rate": None}
    since = (now - timedelta(minutes=AUTONOMY_DECISION_ERROR_LOOKBACK_MINUTES)).isoformat()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select status from analysis_jobs where actor = ? and requested_at >= ?",
            (actor, since),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return {"available": False, "total": 0, "failed": 0, "failure_rate": None}
    total = len(rows)
    failed = sum(1 for row in rows if str(row["status"]).lower() == "failed")
    failure_rate = (Decimal(failed) / Decimal(total)) if total else None
    return {
        "available": True,
        "total": total,
        "failed": failed,
        "failure_rate": _q8s(failure_rate) if failure_rate is not None else None,
    }


def _paper_bootstrap_actor_candidates() -> set[str]:
    actors: set[str] = set()
    env_actor = os.getenv(PAPER_FEEDBACK_ACTOR_ENV, "").strip()
    if env_actor:
        actors.add(env_actor)
    with connect() as conn:
        rows = conn.execute(
            "select distinct actor from watchlist_entries where source_origin = ? "
            "union select actor from paper_bootstrap_halts",
            (PAPER_FEEDBACK_BOOTSTRAP_ORIGIN,),
        ).fetchall()
    actors.update(str(row["actor"]) for row in rows)
    return actors


def _halt_paper_feedback_bootstrap(
    actor: str, reason: str, observed: dict[str, object], now: datetime
) -> dict[str, object]:
    was_halted = bool(_paper_bootstrap_halt_status(actor)["halted"])
    disabled = _disable_paper_feedback_bootstrap(actor, now)
    with connect() as conn:
        conn.execute(
            """
            insert into paper_bootstrap_halts
              (actor, halted, reason, observed_json, halted_at, resumed_at, resumed_by,
               updated_at)
            values (?,?,?,?,?,NULL,NULL,?)
            on conflict(actor) do update set
              halted = 1,
              reason = excluded.reason,
              observed_json = excluded.observed_json,
              halted_at = coalesce(paper_bootstrap_halts.halted_at, excluded.halted_at),
              updated_at = excluded.updated_at
            """,
            (
                actor,
                1,
                reason,
                json.dumps(observed, sort_keys=True),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        conn.commit()
    alert = {
        "type": "paper_bootstrap_halted",
        "severity": "critical",
        "message": "paper feedback bootstrap halted; manual resume required",
        "reason": reason,
        "observed": observed,
        "paper_only": True,
        "live_trading_enabled": False,
    }
    sent = 0
    if not was_halted and _notify_actor_event(actor, "alert", {"alert": alert}):
        sent = 1
    return {
        "halted": True,
        "new_halt": not was_halted,
        "reason": reason,
        "observed": observed,
        "alert_sent": sent,
        **disabled,
    }


def _resume_paper_feedback_bootstrap(
    actor: str, *, resumed_by: str, now: datetime | None = None
) -> dict[str, object]:
    current = now or _now()
    with connect() as conn:
        conn.execute(
            """
            insert into paper_bootstrap_halts
              (actor, halted, reason, observed_json, halted_at, resumed_at, resumed_by,
               updated_at)
            values (?,0,NULL,'{}',NULL,?,?,?)
            on conflict(actor) do update set
              halted = 0,
              reason = NULL,
              observed_json = '{}',
              resumed_at = excluded.resumed_at,
              resumed_by = excluded.resumed_by,
              updated_at = excluded.updated_at
            """,
            (actor, current.isoformat(), resumed_by, current.isoformat()),
        )
        conn.execute(
            "insert into paper_bootstrap_guardrail_state "
            "(actor, mark_failure_streak, decision_error_streak, last_error_rate, "
            "last_evaluated_at) values (?,0,0,NULL,?) "
            "on conflict(actor) do update set mark_failure_streak = 0, "
            "decision_error_streak = 0, last_error_rate = NULL, "
            "last_evaluated_at = excluded.last_evaluated_at",
            (actor, current.isoformat()),
        )
        conn.commit()
    restored: dict[str, object] = {
        "enabled": False,
        "watchlist_upserted": 0,
        "autonomy_upserted": False,
    }
    if (
        _env_bool(PAPER_FEEDBACK_BOOTSTRAP_ENABLED_ENV, default=False)
        and os.getenv(PAPER_FEEDBACK_ACTOR_ENV, "").strip() == actor
    ):
        restored = bootstrap_paper_feedback_loop(now=current)
    return {
        "actor": actor,
        "halted": False,
        "resumed_by": resumed_by,
        "resumed_at": current.isoformat(),
        "restored": restored,
    }


def _paper_bootstrap_guardrail_thresholds() -> dict[str, object]:
    return {
        "drawdown_halt_usdt": _q8s(AUTONOMY_DRAWDOWN_HALT_USDT),
        "consecutive_loss_halt_n": AUTONOMY_CONSECUTIVE_LOSS_HALT_N,
        "mark_failure_halt_n": AUTONOMY_MARK_FAILURE_HALT_N,
        "decision_error_rate_halt_pct": _q8s(AUTONOMY_DECISION_ERROR_RATE_HALT_PCT),
        "decision_error_rate_min_n": AUTONOMY_DECISION_ERROR_RATE_MIN_N,
        "decision_error_lookback_minutes": AUTONOMY_DECISION_ERROR_LOOKBACK_MINUTES,
    }


def evaluate_paper_bootstrap_guardrails(
    actor: str, now: datetime | None = None
) -> dict[str, object]:
    current = now or _now()
    if not actor:
        return {"actor": actor, "halted": False, "skipped": "ACTOR_REQUIRED"}
    if bool(_paper_bootstrap_halt_status(actor)["halted"]):
        disabled = _disable_paper_feedback_bootstrap(actor, current)
        return {
            "actor": actor,
            "halted": True,
            "reason": "already_halted",
            "thresholds": _paper_bootstrap_guardrail_thresholds(),
            **disabled,
        }
    overview = build_paper_autonomy_observability(
        actor, include_guardrails=False, date=current.strftime("%Y-%m-%d")
    )
    max_drawdown = Decimal(str(overview["max_drawdown_today_usdt"]))
    consecutive_losses = int(str(overview["consecutive_closed_losses"]))
    mark_failures_raw = overview.get("mark_failures", [])
    mark_failures = (
        [str(item) for item in mark_failures_raw] if isinstance(mark_failures_raw, list) else []
    )
    previous_state = _paper_bootstrap_guardrail_state(actor)
    previous_mark_streak = int(str(previous_state["mark_failure_streak"]))
    mark_streak = previous_mark_streak + 1 if mark_failures else 0
    analysis_errors = _recent_analysis_error_rate(actor, current)
    failure_rate_raw = analysis_errors.get("failure_rate")
    failure_rate = Decimal(str(failure_rate_raw)) if failure_rate_raw is not None else None
    previous_error_streak = int(str(previous_state["decision_error_streak"]))
    analysis_total = int(str(analysis_errors["total"]))
    decision_error_streak = (
        previous_error_streak + 1
        if failure_rate is not None
        and analysis_total >= AUTONOMY_DECISION_ERROR_RATE_MIN_N
        and failure_rate >= AUTONOMY_DECISION_ERROR_RATE_HALT_PCT
        else 0
    )
    _write_paper_bootstrap_guardrail_state(
        actor,
        mark_failure_streak=mark_streak,
        decision_error_streak=decision_error_streak,
        last_error_rate=_q8s(failure_rate) if failure_rate is not None else None,
        evaluated_at=current,
    )
    triggers: list[str] = []
    if max_drawdown > AUTONOMY_DRAWDOWN_HALT_USDT:
        triggers.append("drawdown")
    if consecutive_losses >= AUTONOMY_CONSECUTIVE_LOSS_HALT_N:
        triggers.append("consecutive_losses")
    if mark_streak >= AUTONOMY_MARK_FAILURE_HALT_N:
        triggers.append("mark_price_unavailable")
    if decision_error_streak > 0:
        triggers.append("decision_error_rate")
    observed = {
        "max_drawdown_today_usdt": _q8s(max_drawdown),
        "consecutive_closed_losses": consecutive_losses,
        "mark_failures": mark_failures,
        "mark_failure_streak": mark_streak,
        "analysis_error_rate": analysis_errors,
        "decision_error_streak": decision_error_streak,
        "thresholds": _paper_bootstrap_guardrail_thresholds(),
    }
    if triggers:
        return {
            "actor": actor,
            "triggers": triggers,
            **_halt_paper_feedback_bootstrap(actor, ",".join(triggers), observed, current),
        }
    return {
        "actor": actor,
        "halted": False,
        "triggers": [],
        "observed": observed,
    }


def evaluate_all_paper_bootstrap_guardrails(now: datetime | None = None) -> dict[str, object]:
    current = now or _now()
    results = [
        evaluate_paper_bootstrap_guardrails(actor, now=current)
        for actor in sorted(_paper_bootstrap_actor_candidates())
    ]
    return {
        "evaluated": len(results),
        "halted": sum(1 for result in results if bool(result.get("halted"))),
        "results": results,
    }


def bootstrap_paper_feedback_loop(now: datetime | None = None) -> dict[str, object]:
    now = now or _now()
    actor = os.getenv(PAPER_FEEDBACK_ACTOR_ENV, "").strip()
    if not _env_bool(PAPER_FEEDBACK_BOOTSTRAP_ENABLED_ENV, default=False):
        disabled = _disable_paper_feedback_bootstrap(actor, now) if actor else {}
        return {
            "enabled": False,
            "watchlist_upserted": 0,
            "autonomy_upserted": False,
            **disabled,
        }
    if not actor:
        return {
            "enabled": True,
            "skipped": "PAPER_FEEDBACK_ACTOR_REQUIRED",
            "watchlist_upserted": 0,
            "autonomy_upserted": False,
        }
    halt_status = _paper_bootstrap_halt_status(actor)
    if bool(halt_status["halted"]):
        disabled = _disable_paper_feedback_bootstrap(actor, now)
        return {
            "enabled": True,
            "paused": True,
            "actor": actor,
            "skipped": "PAPER_BOOTSTRAP_HALTED_MANUAL_RESUME_REQUIRED",
            "watchlist_upserted": 0,
            "autonomy_upserted": False,
            "halt": halt_status,
            **disabled,
        }
    symbols = _paper_feedback_symbol_pool()
    if not symbols:
        return {
            "enabled": True,
            "actor": actor,
            "skipped": "PAPER_FEEDBACK_SYMBOLS_REQUIRED",
            "watchlist_upserted": 0,
            "autonomy_upserted": False,
        }
    cadence, daily_budget, per_trade, min_conviction, allowed_sources = (
        _paper_feedback_runtime_settings()
    )
    next_run = now.isoformat()
    with connect() as conn:
        for symbol, asset_type in symbols:
            conn.execute(
                "insert into watchlist_entries "
                "(actor, symbol, asset_type, cadence_minutes, last_run_at, "
                "next_run_at, enabled, source_origin, gate_conviction, created_at) "
                "values (?,?,?,?,NULL,?,?,?,?,?) "
                "on conflict(actor, symbol) do update set "
                "asset_type = excluded.asset_type, cadence_minutes = excluded.cadence_minutes, "
                "next_run_at = excluded.next_run_at, enabled = 1, "
                "source_origin = excluded.source_origin, "
                "gate_conviction = excluded.gate_conviction",
                (
                    actor,
                    symbol,
                    asset_type,
                    cadence,
                    next_run,
                    1,
                    PAPER_FEEDBACK_BOOTSTRAP_ORIGIN,
                    str(min_conviction),
                    now.isoformat(),
                ),
            )
        conn.execute(
            """
            insert into autonomy_settings
              (actor, enabled, daily_budget_usdt, min_conviction, per_trade_usdt,
               allowed_sources, updated_at)
            values (?,?,?,?,?,?,?)
            on conflict(actor) do update set
              enabled = excluded.enabled,
              daily_budget_usdt = excluded.daily_budget_usdt,
              min_conviction = excluded.min_conviction,
              per_trade_usdt = excluded.per_trade_usdt,
              allowed_sources = excluded.allowed_sources,
              updated_at = excluded.updated_at
            """,
            (
                actor,
                1,
                str(daily_budget),
                str(min_conviction),
                str(per_trade),
                allowed_sources,
                now.isoformat(),
            ),
        )
        conn.commit()
    return {
        "enabled": True,
        "actor": actor,
        "watchlist_upserted": len(symbols),
        "autonomy_upserted": True,
        "cadence_minutes": cadence,
        "daily_budget_usdt": str(daily_budget),
        "per_trade_usdt": str(per_trade),
        "min_conviction": str(min_conviction),
        "allowed_sources": allowed_sources,
        "origin": PAPER_FEEDBACK_BOOTSTRAP_ORIGIN,
        "gate_conviction": str(min_conviction),
    }


def _canonical_source_ref(source_ref: Mapping[str, object]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in sorted(source_ref.items(), key=lambda item: str(item[0]))
        if value is not None
    }


def _source_ref_json(source_ref: Mapping[str, object]) -> str:
    return json.dumps(_canonical_source_ref(source_ref), sort_keys=True, separators=(",", ":"))


def _memory_id(memory_type: str, source_ref: Mapping[str, object]) -> str:
    digest = hashlib.sha256(f"{memory_type}:{_source_ref_json(source_ref)}".encode()).hexdigest()
    return f"memory_entry:{digest[:24]}"


def _written_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    try:
        source_ref = json.loads(str(row["source_ref_json"]))
    except json.JSONDecodeError:
        source_ref = {"table": "memory_entries", "memory_id": str(row["memory_id"])}
    try:
        tags = json.loads(str(row["tags_json"]))
    except json.JSONDecodeError:
        tags = []
    return MemoryEntry(
        id=str(row["memory_id"]),
        type=str(row["type"]),
        subject=str(row["subject"]),
        content=str(row["content"]),
        source_ref=source_ref if isinstance(source_ref, dict) else {},
        created_at=str(row["created_at"]),
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
        confidence=str(row["confidence"]) if row["confidence"] is not None else None,
        superseded_by=str(row["superseded_by"]) if row["superseded_by"] else None,
    )


def record_memory(
    memory_type: Literal["decision", "outcome", "lesson", "convention", "observation"],
    subject: str,
    content: str,
    source_ref: Mapping[str, object],
    *,
    tags: list[str] | None = None,
    confidence: str | None = None,
    superseded_by: str | None = None,
    trigger: str | None = None,
    created_by: str | None = None,
) -> tuple[MemoryEntry, bool]:
    canonical_source_ref = _canonical_source_ref(source_ref)
    if not canonical_source_ref:
        canonical_source_ref = {"table": "manual_memory", "manual_id": str(uuid4())}
    memory_id = _memory_id(memory_type, canonical_source_ref)
    tags_json = json.dumps(_memory_tags(tags or []), sort_keys=True, separators=(",", ":"))
    created_at = _now().isoformat()
    source_json = _source_ref_json(canonical_source_ref)
    with connect() as conn:
        existing = conn.execute(
            "select memory_id, type, subject, content, source_ref_json, tags_json, "
            "confidence, superseded_by, trigger, created_by, created_at "
            "from memory_entries where type = ? and source_ref_json = ?",
            (memory_type, source_json),
        ).fetchone()
        if existing is not None:
            return _written_memory_entry(existing), False
        conn.execute(
            """
            insert into memory_entries
              (memory_id, type, subject, content, source_ref_json, tags_json,
               confidence, superseded_by, trigger, created_by, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                memory_type,
                subject.strip(),
                content.strip(),
                source_json,
                tags_json,
                confidence,
                superseded_by,
                trigger,
                created_by,
                created_at,
            ),
        )
        conn.commit()
        row = conn.execute(
            "select memory_id, type, subject, content, source_ref_json, tags_json, "
            "confidence, superseded_by, trigger, created_by, created_at "
            "from memory_entries where memory_id = ?",
            (memory_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("memory insert did not return a row")
    return _written_memory_entry(row), True


def _scorecard_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    payload = _memory_scorecard_payload(row)
    metadata = _scorecard_metadata_from_payload(payload)
    action = str(payload.get("action") or row["action"])
    conviction = payload.get("conviction")
    horizon = payload.get("time_horizon")
    thesis = str(payload.get("thesis") or "").strip()
    origin = _metadata_data_origin(metadata)
    gate_conviction = _metadata_gate_conviction(metadata)
    content = (
        f"{row['source']} {action} decision for {row['symbol']}; "
        f"conviction={conviction or 'unknown'}; origin={origin}; "
        f"gate_conviction={gate_conviction or 'standard'}; thesis={thesis or 'not recorded'}"
    )
    if row["consumed_by_intent_id"]:
        content += f"; consumed_by_intent_id={row['consumed_by_intent_id']}"
    return MemoryEntry(
        id=f"decision:scorecard:{row['scorecard_id']}",
        type="decision",
        subject=str(row["symbol"]),
        content=content,
        source_ref={"table": "scorecards", "scorecard_id": str(row["scorecard_id"])},
        created_at=str(row["created_at"]),
        tags=_memory_tags(
            str(row["actor"]),
            str(row["source"]),
            action,
            horizon,
            f"asset:{metadata.get('asset_type')}" if metadata.get("asset_type") else None,
            _scorecard_origin_tags(metadata),
            _memory_factor_tags(payload),
        ),
        confidence=str(conviction) if conviction is not None else None,
    )


def _outcome_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    closed = row["closed_at"] or row["opened_at"]
    content = (
        f"{row['status']} {row['action']} outcome for {row['symbol']}; "
        f"opened_cost_basis={row['opened_cost_basis']}"
    )
    if row["status"] == "closed":
        content += (
            f"; closed_realized_pnl={row['closed_realized_pnl']}; "
            f"closed_return_pct={row['closed_return_pct']}"
        )
    if row["notes"]:
        content += f"; notes={row['notes']}"
    return MemoryEntry(
        id=f"outcome:scorecard_outcome:{row['outcome_id']}",
        type="outcome",
        subject=str(row["symbol"]),
        content=content,
        source_ref={
            "table": "scorecard_outcomes",
            "outcome_id": str(row["outcome_id"]),
            "scorecard_id": str(row["scorecard_id"]),
        },
        created_at=str(closed),
        tags=_memory_tags(
            str(row["actor"]), str(row["source"]), str(row["action"]), str(row["status"])
        ),
        confidence=None,
    )


def _lesson_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    pnl = _decimal_or_none(row["closed_realized_pnl"])
    if pnl is not None and pnl > 0:
        result = "profitable"
    elif pnl is not None and pnl < 0:
        result = "loss"
    else:
        result = "flat"
    reflected = "reflected" if row["reflected_at"] else "pending_reflection"
    return MemoryEntry(
        id=f"lesson:scorecard_outcome:{row['outcome_id']}",
        type="lesson",
        subject=str(row["symbol"]),
        content=(
            f"{result} {row['action']} lesson for {row['symbol']}; "
            f"return_pct={row['closed_return_pct']}; source={row['source']}; {reflected}"
        ),
        source_ref={
            "table": "scorecard_outcomes",
            "outcome_id": str(row["outcome_id"]),
            "scorecard_id": str(row["scorecard_id"]),
        },
        created_at=str(row["closed_at"] or row["opened_at"]),
        tags=_memory_tags(
            str(row["actor"]), str(row["source"]), str(row["action"]), "closed", result, reflected
        ),
        confidence=None,
    )


def _ev_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    content = (
        f"EV gate {row['gate_result']} for {row['symbol']} in {row['mode']} mode; "
        f"ev={row['ev']}; min_ev={row['min_ev']}; reason={row['reason'] or 'none'}"
    )
    return MemoryEntry(
        id=f"observation:ev_estimate:{row['scorecard_id']}",
        type="observation",
        subject=str(row["symbol"]),
        content=content,
        source_ref={"table": "ev_estimates", "scorecard_id": str(row["scorecard_id"])},
        created_at=str(row["updated_at"] or row["created_at"]),
        tags=_memory_tags(
            str(row["actor"]), "ev_estimate", str(row["mode"]), str(row["gate_result"])
        ),
        confidence=str(row["p"]) if row["p"] is not None else None,
    )


def _calibration_memory_entry(row: sqlite3.Row) -> MemoryEntry:
    source = str(row["source"])
    asset_type = str(row["asset_type"])
    bucket = str(row["heuristic_bucket"])
    return MemoryEntry(
        id=f"observation:conviction_calibration:{source}:{asset_type}:{bucket}",
        type="observation",
        subject=f"{source}:{asset_type}:{bucket}",
        content=(
            f"Calibration {source}/{asset_type}/{bucket}: "
            f"samples={row['sample_count']}; hit_rate={row['empirical_hit_rate']}; "
            f"calibrated_conviction={row['calibrated_conviction']}; "
            f"avg_alpha_return={row['avg_alpha_return']}"
        ),
        source_ref={
            "table": "conviction_calibration",
            "source": source,
            "asset_type": asset_type,
            "heuristic_bucket": bucket,
        },
        created_at=str(row["updated_at"]),
        tags=_memory_tags(source, asset_type, "calibration", bucket),
        confidence=str(row["calibrated_conviction"]),
    )


def _load_memory_entries() -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    with connect() as conn:
        written_rows = conn.execute(
            "select memory_id, type, subject, content, source_ref_json, tags_json, "
            "confidence, superseded_by, trigger, created_by, created_at "
            "from memory_entries"
        ).fetchall()
        scorecards = conn.execute(
            "select scorecard_id, actor, symbol, action, source, payload_json, created_at, "
            "expires_at, consumed_by_intent_id from scorecards"
        ).fetchall()
        outcomes = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at, trailing_pct, peak_mark "
            "from scorecard_outcomes"
        ).fetchall()
        ev_rows = conn.execute(
            "select scorecard_id, outcome_id, actor, symbol, mode, gate_result, reason, "
            "p, tp_pct, sl_pct, fee_bps, slippage_bps, funding_bps, min_ev, ev, "
            "created_at, updated_at from ev_estimates"
        ).fetchall()
        calibration_rows = conn.execute(
            "select source, asset_type, heuristic_bucket, sample_count, hit_count, "
            "avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at "
            "from conviction_calibration"
        ).fetchall()
    entries.extend(_written_memory_entry(row) for row in written_rows)
    entries.extend(_scorecard_memory_entry(row) for row in scorecards)
    entries.extend(_outcome_memory_entry(row) for row in outcomes)
    entries.extend(_lesson_memory_entry(row) for row in outcomes if row["status"] == "closed")
    entries.extend(_ev_memory_entry(row) for row in ev_rows)
    entries.extend(_calibration_memory_entry(row) for row in calibration_rows)
    deduped: dict[str, MemoryEntry] = {}
    for entry in entries:
        deduped.setdefault(entry.id, entry)
    return _mark_superseded_memory(list(deduped.values()))


def _mark_superseded_memory(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    latest_decision: dict[tuple[str, str], MemoryEntry] = {}
    for entry in sorted(entries, key=lambda item: item.created_at, reverse=True):
        if entry.type != "decision":
            continue
        key = (entry.subject.upper(), ",".join(sorted(entry.tags)))
        newer = latest_decision.get(key)
        if newer is None:
            latest_decision[key] = entry
        else:
            entry.superseded_by = newer.id
    return entries


def _memory_matches(
    entry: MemoryEntry,
    *,
    subject: str | None,
    memory_type: str | None,
    since: str | None,
    tags: set[str],
    keyword: str | None,
    include_superseded: bool,
) -> bool:
    if memory_type and entry.type != memory_type:
        return False
    if subject:
        subject_l = subject.lower()
        haystack = " ".join([entry.subject, entry.content, *entry.tags]).lower()
        if subject_l not in haystack:
            return False
    if since and entry.created_at < since:
        return False
    entry_tags = {tag.lower() for tag in entry.tags}
    if tags and not {tag.lower() for tag in tags}.issubset(entry_tags):
        return False
    if keyword:
        keyword_l = keyword.lower()
        haystack = " ".join(
            [
                entry.id,
                entry.subject,
                entry.content,
                json.dumps(entry.source_ref, sort_keys=True),
                *entry.tags,
            ]
        ).lower()
        if keyword_l not in haystack:
            return False
    if entry.superseded_by and not include_superseded:
        return False
    return True


def _parse_tags(tags: str | None) -> set[str]:
    if not tags:
        return set()
    return {item.strip() for item in tags.split(",") if item.strip()}


def _find_memory_entry(memory_id: str) -> MemoryEntry | None:
    for entry in _load_memory_entries():
        if entry.id == memory_id:
            return entry
    return None


def _watchlist_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "actor": row["actor"],
        "symbol": row["symbol"],
        "asset_type": row["asset_type"],
        "cadence_minutes": row["cadence_minutes"],
        "last_run_at": row["last_run_at"],
        "next_run_at": row["next_run_at"],
        "enabled": bool(row["enabled"]),
        "source_origin": row["source_origin"],
        "gate_conviction": row["gate_conviction"],
        "created_at": row["created_at"],
    }


def _fire_scheduled_analysis(
    actor: str,
    symbol: str,
    asset_type: str,
    *,
    source_origin: str = STANDARD_DATA_ORIGIN,
    gate_conviction: str | None = None,
) -> bool:
    payload: dict[str, object] = {"actor": actor, "symbol": symbol, "asset_type": asset_type}
    if source_origin != STANDARD_DATA_ORIGIN:
        payload["origin"] = source_origin
    if gate_conviction:
        payload["gate_conviction"] = gate_conviction
    try:
        response = httpx.post(
            f"{ANALYSIS_ADAPTER_URL}/analyze",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def scheduler_tick(now: datetime | None = None) -> dict[str, int]:
    now = now or _now()
    evaluate_all_paper_bootstrap_guardrails(now=now)
    now_iso = now.isoformat()
    with connect() as conn:
        rows = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes, source_origin, gate_conviction "
            "from watchlist_entries "
            "where enabled = 1 and next_run_at <= ? order by next_run_at asc limit ?",
            (now_iso, SCHEDULER_BATCH_LIMIT),
        ).fetchall()
    fired = 0
    failed = 0
    for row in rows:
        cadence = int(row["cadence_minutes"])
        ok = _fire_scheduled_analysis(
            str(row["actor"]),
            str(row["symbol"]),
            str(row["asset_type"]),
            source_origin=str(row["source_origin"] or STANDARD_DATA_ORIGIN),
            gate_conviction=(
                str(row["gate_conviction"]) if row["gate_conviction"] is not None else None
            ),
        )
        next_run = now + timedelta(minutes=cadence if ok else max(cadence, 60))
        with connect() as conn:
            conn.execute(
                "update watchlist_entries set last_run_at = ?, next_run_at = ? "
                "where actor = ? and symbol = ?",
                (now_iso, next_run.isoformat(), row["actor"], row["symbol"]),
            )
            conn.commit()
        if ok:
            fired += 1
        else:
            failed += 1
    return {"due": len(rows), "fired": fired, "failed": failed}


async def _scheduler_loop() -> None:
    last_live_auto_tick = 0.0
    last_stop_loss_tick = 0.0
    while True:
        try:
            scheduler_tick()
        except Exception as log_exc:
            logger.error("RECONCILE_APPLY_ERROR_LOG_FAILED error=%s", log_exc)
        try:
            auto_trade_tick()
        except Exception:
            pass
        now_mono = time.monotonic()
        if now_mono - last_live_auto_tick >= LIVE_AUTO_TICK_SEC:
            try:
                live_auto_trade_tick()
            except Exception:
                pass
            last_live_auto_tick = now_mono
        if now_mono - last_stop_loss_tick >= STOP_LOSS_TICK_SEC:
            try:
                stop_loss_watchdog_tick()
            except Exception:
                pass
            last_stop_loss_tick = now_mono
        await asyncio.sleep(SCHEDULER_TICK_SEC)


async def _start_scheduler() -> None:
    global _scheduler_task
    if SCHEDULER_ENABLED and _scheduler_task is None:
        _scheduler_task = asyncio.create_task(_scheduler_loop())


async def _stop_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None:
        return
    _scheduler_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _scheduler_task
    _scheduler_task = None


async def _startup_ibkr_reconcile() -> None:
    if IBKR_MODE != "bridge" or not IBKR_LIVE_TRADING_ENABLED:
        return
    await asyncio.sleep(2.0)
    try:
        run_ibkr_reconciliation()
    except Exception as exc:
        logging.getLogger(__name__).warning("RECONCILE_IBKR startup_hook_exception error=%s", exc)


async def _warn_reconcile_apply_without_token() -> None:
    if RECONCILE_APPLY_ENABLED and not OPS_TOKEN:
        logger.warning(
            "RECONCILE_APPLY_ENABLED is true but OPS_TOKEN is empty; "
            "POST /reconcile/ibkr/apply will reject all calls"
        )


def _now() -> datetime:
    return datetime.now(UTC)


def _check_ops_auth(request: Request) -> None:
    """Double-lock check: env flag + token. Raises HTTPException on failure."""
    if not RECONCILE_APPLY_ENABLED:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "RECONCILE_APPLY_DISABLED",
                "message": "set RECONCILE_APPLY_ENABLED=true to use this endpoint",
            },
        )
    token = request.headers.get("X-Ops-Token", "")
    if not OPS_TOKEN or token != OPS_TOKEN:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_OPS_TOKEN",
                "message": "missing or invalid X-Ops-Token header",
            },
        )


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def _daily_limit() -> Decimal:
    return Decimal(os.getenv("PER_SYMBOL_DAILY_LIMIT_USDT", str(PER_SYMBOL_DAILY_LIMIT_USDT)))


def _execution_url() -> str:
    return os.getenv("EXECUTION_SERVICE_URL", "http://execution-service:8082")


def _market_url() -> str:
    return os.getenv("MARKET_DATA_URL", "http://market-data:8083")


def _resolve_qty(intent: OrderIntent, market_url: str) -> tuple[str, str]:
    try:
        params = {"symbol": intent.symbol}
        if intent.venue == "ibkr_us_equity":
            params["asset_type"] = "stock"
        response = httpx.get(f"{market_url}/ticker", params=params, timeout=3.0)
        response.raise_for_status()
        price_str = str(response.json()["price"])
        price = Decimal(price_str)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"code": "MARKET_DATA_UNAVAILABLE"}) from exc

    if intent.quantity.kind == "base":
        return str(intent.quantity.value), price_str
    base_qty = (intent.quantity.value / price).quantize(Decimal("0.00000001"))
    return str(base_qty), price_str


def _build_execution_request(intent: OrderIntent, decision: RiskDecision) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id=uuid4(),
        intent_id=intent.intent_id,
        decision_id=decision.decision_id,
        idempotency_key=intent.idempotency_key,
        confirmation_token=decision.confirmation_token,
        dry_run=False,
        submitted_at=_now(),
    )


def _call_execution(intent: OrderIntent, decision: RiskDecision, base_qty: str) -> ExecutionResult:
    request = _build_execution_request(intent, decision)
    execution_response = httpx.post(
        f"{_execution_url()}/execute",
        content=request.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-decision-approved": str(decision.approved).lower(),
            "x-mode": intent.mode,
            "x-venue": intent.venue,
            "x-symbol": intent.symbol,
            "x-quantity": base_qty,
            "x-side": intent.side,
            "x-quantity-kind": intent.quantity.kind,
            "x-quote-qty": str(intent.quantity.value) if intent.quantity.kind == "quote" else "",
            "x-order-type": intent.order_type,
            "x-limit-price": str(intent.limit_price) if intent.limit_price is not None else "",
            "x-time-in-force": intent.time_in_force,
        },
        timeout=5.0,
    )
    execution_response.raise_for_status()
    return ExecutionResult.model_validate(execution_response.json())


def _row_to_item(row: sqlite3.Row) -> dict[str, object]:
    execution_json = row["execution_json"]
    return {
        "status": row["status"],
        "intent": OrderIntent.model_validate_json(row["payload_json"]),
        "decision": RiskDecision.model_validate_json(row["decision_json"]),
        "execution": (
            ExecutionResult.model_validate_json(execution_json) if execution_json else None
        ),
    }


def _pending_response(intent: OrderIntent, decision: RiskDecision) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content=jsonable_encoder(
            {
                "status": "pending_confirmation",
                "intent_id": str(intent.intent_id),
                "confirmation_token": decision.confirmation_token,
                "confirmation_expires_at": decision.confirmation_expires_at,
            }
        ),
    )


def _rejected_response(decision: RiskDecision) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"code": "RISK_REJECTED", "reasons": decision.reasons}),
    )


def _idempotent_response(row: sqlite3.Row) -> JSONResponse | dict[str, object]:
    item = _row_to_item(row)
    decision = item["decision"]
    intent = item["intent"]
    if not isinstance(decision, RiskDecision) or not isinstance(intent, OrderIntent):
        raise RuntimeError("invalid persisted intent")
    status = row["status"]
    if status == "executed":
        return item
    if status == "pending_confirmation":
        return _pending_response(intent, decision)
    if status == "rejected":
        return _rejected_response(decision)
    if status == "canceled":
        return JSONResponse(status_code=410, content={"code": "INTENT_CANCELED"})
    return item


def _current_exposure(symbol: str, date: str) -> Decimal:
    with connect() as conn:
        row = conn.execute(
            "select coalesce(sum(cast(notional_usdt as real)), 0.0) "
            "from daily_fills where date = ? and symbol = ?",
            (date, symbol),
        ).fetchone()
    return Decimal(str(row[0]))


def _exposure_limit_response(
    intent: OrderIntent, current: Decimal, requested: Decimal
) -> JSONResponse | None:
    limit = _daily_limit()
    if current + requested <= limit:
        return None
    return JSONResponse(
        status_code=422,
        content={
            "code": "PER_SYMBOL_DAILY_LIMIT_EXCEEDED",
            "symbol": intent.symbol,
            "limit": str(limit),
            "current": str(current),
            "requested": str(requested),
        },
    )


def _record_fill(execution: ExecutionResult, symbol: str, side: str) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    notional = execution.filled_qty * execution.avg_price
    with connect() as conn:
        conn.execute(
            "insert or ignore into daily_fills "
            "(fill_id, date, symbol, side, notional_usdt, created_at) "
            "values (?, ?, ?, ?, ?, ?)",
            (
                str(execution.execution_id),
                _today(),
                symbol,
                side,
                str(notional),
                _now().isoformat(),
            ),
        )
        conn.commit()


def _is_allowed_webhook_host(webhook_url: str) -> bool:
    parsed = urlparse(webhook_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    return parsed.hostname in NOTIFICATION_HOST_ALLOWLIST


def _record_notification_delivery(
    actor: str,
    event_type: str,
    webhook_url: str,
    status_code: int | None,
    ok: bool,
    error_class: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into notification_deliveries
              (actor, event_type, webhook_url, status_code, ok, error_class, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor,
                event_type,
                webhook_url,
                status_code,
                1 if ok else 0,
                error_class,
                _now().isoformat(),
            ),
        )
        conn.execute(
            """
            delete from notification_deliveries
            where actor = ?
              and id not in (
                select id from notification_deliveries
                where actor = ?
                order by created_at desc, id desc
                limit ?
              )
            """,
            (actor, actor, NOTIFICATION_HISTORY_LIMIT),
        )
        conn.commit()


def _deliver_webhook(
    actor: str, webhook_url: str, secret: str, event_type: str, payload: dict[str, object]
) -> None:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    signature = hmac_lib.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    try:
        response = httpx.post(
            webhook_url,
            content=body,
            headers={
                "content-type": "application/json",
                "X-GitHub-Event": event_type,
                "X-Hub-Signature-256": f"sha256={signature}",
                "x-trading-agent-event": event_type,
                "x-trading-agent-signature": signature,
            },
            timeout=NOTIFICATION_TIMEOUT_SEC,
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        _record_notification_delivery(
            actor, event_type, webhook_url, status_code, 200 <= status_code < 300, None
        )
    except Exception as exc:  # noqa: BLE001
        _record_notification_delivery(
            actor, event_type, webhook_url, None, False, exc.__class__.__name__
        )


def _notify_fill(intent: OrderIntent, execution: ExecutionResult) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    with connect() as conn:
        row = conn.execute(
            "select webhook_url, secret, events_json from notification_subscriptions "
            "where actor = ? and enabled = 1",
            (intent.actor,),
        ).fetchone()
    if row is None:
        return
    try:
        events = json.loads(str(row["events_json"]))
    except json.JSONDecodeError:
        return
    if "fill" not in events:
        return
    payload: dict[str, object] = {
        "event_type": "fill",
        "actor": intent.actor,
        "symbol": intent.symbol,
        "side": intent.side,
        "qty_str": _q8s(execution.filled_qty),
        "avg_price_str": _q8s(execution.avg_price),
        "mode": intent.mode,
        "status": execution.status,
        "intent_id": str(intent.intent_id),
    }
    _deliver_webhook(intent.actor, str(row["webhook_url"]), str(row["secret"]), "fill", payload)


def _notify_actor_event(actor: str, event_type: str, payload: dict[str, object]) -> bool:
    if event_type not in SUPPORTED_NOTIFICATION_EVENTS:
        return False
    with connect() as conn:
        row = conn.execute(
            "select webhook_url, secret, events_json from notification_subscriptions "
            "where actor = ? and enabled = 1",
            (actor,),
        ).fetchone()
    if row is None:
        return False
    try:
        events = json.loads(str(row["events_json"]))
    except json.JSONDecodeError:
        return False
    if event_type not in events:
        return False
    private_payload = {"event_type": event_type, "actor": actor, **payload}
    _deliver_webhook(
        actor, str(row["webhook_url"]), str(row["secret"]), event_type, private_payload
    )
    return True


def _after_fill_side_effects(execution: ExecutionResult, intent: OrderIntent) -> None:
    _record_fill(execution, intent.symbol, intent.side)
    _update_position(execution, intent)
    with contextlib.suppress(Exception):
        _notify_fill(intent, execution)


def _q8(value: Decimal) -> Decimal:
    return value.quantize(DECIMAL_8)


def _q8s(value: Decimal) -> str:
    return f"{_q8(value):.8f}"


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return _q8(sum(values) / Decimal(len(values)))


def _pearson(xs: list[Decimal], ys: list[Decimal]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    xvals = [float(x) for x in xs]
    yvals = [float(y) for y in ys]
    xmean = sum(xvals) / len(xvals)
    ymean = sum(yvals) / len(yvals)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xvals, yvals, strict=True))
    xden = sum((x - xmean) ** 2 for x in xvals)
    yden = sum((y - ymean) ** 2 for y in yvals)
    if xden <= 0 or yden <= 0:
        return None
    return numerator / math.sqrt(xden * yden)


def _ranks(values: list[Decimal]) -> list[Decimal]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [Decimal("0")] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = Decimal(index + end + 1) / Decimal("2")
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def _spearman(xs: list[Decimal], ys: list[Decimal]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _corr_to_report(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.6f}"


def _sign_matches(ev: Decimal, pnl: Decimal) -> bool:
    return (ev > 0 and pnl > 0) or (ev < 0 and pnl < 0) or (ev == 0 and pnl == 0)


def _scorecard_reference_price(scorecard: dict[str, object]) -> Decimal | None:
    for key in ("entry_low", "entry_high"):
        raw = scorecard.get(key)
        if raw is None:
            continue
        try:
            value = Decimal(str(raw))
            if value > 0:
                return value
        except (InvalidOperation, ValueError):
            continue
    symbol = scorecard.get("symbol")
    if not symbol:
        return None
    mark, _source = _mark_for_symbol_str(str(symbol))
    return mark if mark is not None and mark > 0 else None


def _ev_cost_pct() -> Decimal:
    return (FEE_BPS + SLIPPAGE_BPS + FUNDING_BPS) / Decimal("10000")


def _estimate_ev(
    scorecard: dict[str, object],
    calibrated_conviction: Decimal,
) -> dict[str, str | None]:
    reference_price = _scorecard_reference_price(scorecard)
    take_profit_raw = scorecard.get("take_profit")
    stop_loss_raw = scorecard.get("stop_loss")
    base: dict[str, str | None] = {
        "p": _q8s(calibrated_conviction),
        "tp_pct": None,
        "sl_pct": None,
        "fee_bps": str(FEE_BPS),
        "slippage_bps": str(SLIPPAGE_BPS),
        "funding_bps": str(FUNDING_BPS),
        "min_ev": _q8s(MIN_EV),
        "ev": None,
    }
    if reference_price is None or take_profit_raw is None or stop_loss_raw is None:
        return base
    try:
        take_profit = Decimal(str(take_profit_raw))
        stop_loss = Decimal(str(stop_loss_raw))
    except (InvalidOperation, ValueError):
        return base
    action = str(scorecard.get("action", "buy"))
    if action == "sell":
        tp_pct = (reference_price - take_profit) / reference_price
        sl_pct = (stop_loss - reference_price) / reference_price
    else:
        tp_pct = (take_profit - reference_price) / reference_price
        sl_pct = (reference_price - stop_loss) / reference_price
    if tp_pct <= 0 or sl_pct <= 0:
        return base
    ev = calibrated_conviction * tp_pct - (Decimal("1") - calibrated_conviction) * sl_pct
    ev -= _ev_cost_pct()
    base["tp_pct"] = _q8s(tp_pct)
    base["sl_pct"] = _q8s(sl_pct)
    base["ev"] = _q8s(ev)
    return base


def _record_ev_estimate(
    actor: str,
    scorecard: dict[str, object],
    estimate: dict[str, str | None],
    *,
    mode: str,
    gate_result: str,
    reason: str | None,
) -> None:
    scorecard_id = scorecard.get("scorecard_id")
    symbol = str(scorecard.get("symbol") or "")
    if not scorecard_id or not symbol:
        return
    now_iso = _now().isoformat()
    with connect() as conn:
        conn.execute(
            """
            insert into ev_estimates
              (scorecard_id, outcome_id, actor, symbol, mode, gate_result, reason,
               p, tp_pct, sl_pct, fee_bps, slippage_bps, funding_bps, min_ev, ev,
               created_at, updated_at)
            values (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(scorecard_id) do update set
              actor = excluded.actor,
              symbol = excluded.symbol,
              mode = excluded.mode,
              gate_result = excluded.gate_result,
              reason = excluded.reason,
              p = excluded.p,
              tp_pct = excluded.tp_pct,
              sl_pct = excluded.sl_pct,
              fee_bps = excluded.fee_bps,
              slippage_bps = excluded.slippage_bps,
              funding_bps = excluded.funding_bps,
              min_ev = excluded.min_ev,
              ev = excluded.ev,
              updated_at = excluded.updated_at
            """,
            (
                str(scorecard_id),
                actor,
                symbol,
                mode,
                gate_result,
                reason,
                estimate["p"],
                estimate["tp_pct"],
                estimate["sl_pct"],
                estimate["fee_bps"],
                estimate["slippage_bps"],
                estimate["funding_bps"],
                estimate["min_ev"],
                estimate["ev"],
                now_iso,
                now_iso,
            ),
        )
        conn.commit()


def _ev_gate(
    actor: str,
    scorecard: dict[str, object],
    calibrated_conviction: Decimal,
) -> tuple[bool, str]:
    mode = EV_GATE_MODE if EV_GATE_MODE in {"shadow", "enforce"} else "shadow"
    estimate = _estimate_ev(scorecard, calibrated_conviction)
    if estimate["ev"] is None:
        _record_ev_estimate(
            actor,
            scorecard,
            estimate,
            mode=mode,
            gate_result="pass",
            reason="EV_INPUTS_MISSING",
        )
        logger.info(
            "ev_gate scorecard=%s result=pass reason=EV_INPUTS_MISSING",
            scorecard.get("scorecard_id"),
        )
        return True, "OK"
    ev = Decimal(str(estimate["ev"]))
    if mode == "enforce" and ev < MIN_EV:
        _record_ev_estimate(
            actor,
            scorecard,
            estimate,
            mode=mode,
            gate_result="block",
            reason="BELOW_MIN_EV",
        )
        logger.info(
            "ev_gate scorecard=%s result=block ev=%s min_ev=%s",
            scorecard.get("scorecard_id"),
            estimate["ev"],
            _q8s(MIN_EV),
        )
        return False, "BELOW_MIN_EV"
    _record_ev_estimate(
        actor,
        scorecard,
        estimate,
        mode=mode,
        gate_result="pass",
        reason=None,
    )
    logger.info(
        "ev_gate scorecard=%s result=pass ev=%s mode=%s",
        scorecard.get("scorecard_id"),
        estimate["ev"],
        mode,
    )
    return True, "OK"


def _scorecard_trade_ev_gate(actor: str, scorecard: Scorecard) -> tuple[bool, str]:
    metadata = scorecard.metadata or {}
    calibrated_raw = metadata.get("calibrated_conviction") if isinstance(metadata, dict) else None
    if calibrated_raw is None:
        calibrated = scorecard.conviction
    else:
        try:
            calibrated = Decimal(str(calibrated_raw))
        except (InvalidOperation, ValueError):
            calibrated = scorecard.conviction
    return _ev_gate(actor, scorecard.model_dump(mode="json"), calibrated)


def _ev_group_stats(rows: list[EvShadowSample]) -> dict[str, object]:
    pnls = [row["pnl"] for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0]
    return {
        "n": len(rows),
        "win_rate": _q8s(Decimal(len(wins)) / Decimal(len(rows))) if rows else None,
        "avg_realized_pnl": _q8s(_mean(pnls)) if rows else None,
        "total_realized_pnl": _q8s(sum(pnls, Decimal("0"))),
    }


def _ev_counterfactual(rows: list[EvShadowSample], threshold: Decimal) -> dict[str, object]:
    rejected = [row for row in rows if row["ev"] < threshold]
    avoided_losses = [-row["pnl"] for row in rejected if row["pnl"] < 0]
    false_kills = [row["pnl"] for row in rejected if row["pnl"] > 0]
    neutral = [row for row in rejected if row["pnl"] == 0]
    net = sum(avoided_losses, Decimal("0")) - sum(false_kills, Decimal("0"))
    return {
        "threshold_min_ev": _q8s(threshold),
        "rejected_count": len(rejected),
        "avoided_loss_count": len(avoided_losses),
        "avoided_loss_usdt": _q8s(sum(avoided_losses, Decimal("0"))),
        "false_kill_profit_count": len(false_kills),
        "false_kill_profit_usdt": _q8s(sum(false_kills, Decimal("0"))),
        "neutral_reject_count": len(neutral),
        "net_impact_usdt": _q8s(net),
    }


def _best_ev_threshold(rows: list[EvShadowSample]) -> tuple[Decimal, Decimal]:
    candidates = {Decimal("0"), MIN_EV}
    candidates.update(row["ev"] for row in rows)
    best_threshold = MIN_EV
    best_net = Decimal("-Infinity")
    for threshold in sorted(candidates):
        cf = _ev_counterfactual(rows, threshold)
        net = Decimal(str(cf["net_impact_usdt"]))
        if net > best_net:
            best_net = net
            best_threshold = threshold
    return best_threshold, best_net


def _origin_breakdown(samples: list[EvShadowSample]) -> dict[str, dict[str, object]]:
    origins: dict[str, dict[str, object]] = {}
    for sample in samples:
        origin = sample["data_origin"]
        gate = sample["gate_conviction"] or "standard"
        bucket = origins.setdefault(origin, {"n": 0, "gate_convictions": {}})
        bucket["n"] = cast(int, bucket["n"]) + 1
        gate_counts = cast(dict[str, int], bucket["gate_convictions"])
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    return origins


def _payload_origin_and_gate(payload_json: object) -> tuple[str, str | None]:
    try:
        payload = json.loads(str(payload_json))
    except json.JSONDecodeError:
        return STANDARD_DATA_ORIGIN, None
    if not isinstance(payload, dict):
        return STANDARD_DATA_ORIGIN, None
    metadata = _scorecard_metadata_from_payload(payload)
    return _metadata_data_origin(metadata), _metadata_gate_conviction(metadata)


def _load_ev_shadow_samples(actor: str | None) -> tuple[list[EvShadowSample], int]:
    clauses = ["e.outcome_id is not null", "o.status = 'closed'"]
    params: list[object] = []
    if actor:
        clauses.append("e.actor = ?")
        params.append(actor)
    where = " where " + " and ".join(clauses)
    with connect() as conn:
        rows = conn.execute(
            f"""
            select e.ev, o.closed_realized_pnl, s.payload_json
            from ev_estimates e
            join scorecard_outcomes o on o.outcome_id = e.outcome_id
            left join scorecards s on s.scorecard_id = e.scorecard_id
            {where}
            """,
            params,
        ).fetchall()
    samples: list[EvShadowSample] = []
    excluded = 0
    for row in rows:
        ev = _decimal_or_none(row["ev"])
        pnl = _decimal_or_none(row["closed_realized_pnl"])
        if ev is None or pnl is None:
            excluded += 1
            continue
        origin, gate_conviction = _payload_origin_and_gate(row["payload_json"])
        samples.append(
            {
                "ev": ev,
                "pnl": pnl,
                "data_origin": origin,
                "gate_conviction": gate_conviction,
            }
        )
    return samples, excluded


def build_ev_shadow_report(
    actor: str | None = None, min_ev: Decimal | None = None
) -> dict[str, object]:
    threshold = min_ev if min_ev is not None else MIN_EV
    samples, excluded = _load_ev_shadow_samples(actor)
    evs = [row["ev"] for row in samples]
    pnls = [row["pnl"] for row in samples]
    n = len(samples)
    positive_rows = [row for row in samples if row["ev"] > 0]
    negative_rows = [row for row in samples if row["ev"] < 0]
    direction_matches = sum(1 for row in samples if _sign_matches(row["ev"], row["pnl"]))
    pearson = _pearson(evs, pnls)
    spearman = _spearman(evs, pnls)
    counterfactual = _ev_counterfactual(samples, threshold)
    best_threshold, best_net = _best_ev_threshold(samples) if samples else (threshold, Decimal("0"))
    sufficient = n >= EV_SHADOW_MIN_SAMPLES
    positive_corr = (pearson is not None and pearson > 0) or (spearman is not None and spearman > 0)
    net = Decimal(str(counterfactual["net_impact_usdt"]))
    recommend_enforce = sufficient and positive_corr and net > 0
    if not sufficient:
        recommendation = "不建议 enforce: 数据不足,继续 shadow 积累"
    elif not positive_corr:
        recommendation = "不建议 enforce: EV 预估与实际盈亏无正相关"
    elif net <= 0:
        recommendation = "不建议 enforce: 当前 MIN_EV 反事实净影响不为正"
    else:
        recommendation = "可以考虑由人确认切 enforce; 不自动切换"
    report: dict[str, object] = {
        "sample": {
            "n": n,
            "excluded_unusable_rows": excluded,
            "min_required_n": EV_SHADOW_MIN_SAMPLES,
            "data_sufficiency": "ok" if sufficient else "insufficient",
            "data_origin_breakdown": _origin_breakdown(samples),
        },
        "groups": {
            "ev_gt_0": _ev_group_stats(positive_rows),
            "ev_lt_0": _ev_group_stats(negative_rows),
        },
        "directionality": {
            "direction_match_rate": (_q8s(Decimal(direction_matches) / Decimal(n)) if n else None),
            "pearson_ev_vs_realized_pnl": _corr_to_report(pearson),
            "spearman_ev_rank_vs_realized_pnl_rank": _corr_to_report(spearman),
            "positive_correlation": positive_corr,
        },
        "counterfactual_enforce": counterfactual,
        "recommendation": {
            "recommend_enforce": recommend_enforce,
            "message": recommendation,
            "recommended_min_ev": _q8s(best_threshold) if recommend_enforce else None,
            "best_threshold_net_impact_usdt": _q8s(best_net),
            "human_gate_required": True,
            "env_change_required_by_human": "EV_GATE_MODE=enforce plus MIN_EV",
        },
    }
    report["human_readable"] = render_ev_shadow_report(report, actor)
    return report


def render_ev_shadow_report(report: dict[str, object], actor: str | None = None) -> str:
    sample = report["sample"]
    groups = report["groups"]
    direction = report["directionality"]
    counterfactual = report["counterfactual_enforce"]
    recommendation = report["recommendation"]
    if not isinstance(sample, dict) or not isinstance(groups, dict):
        return ""
    ev_gt = groups.get("ev_gt_0", {})
    ev_lt = groups.get("ev_lt_0", {})
    match_rate = direction.get("direction_match_rate") if isinstance(direction, dict) else None
    pearson = direction.get("pearson_ev_vs_realized_pnl") if isinstance(direction, dict) else None
    spearman = (
        direction.get("spearman_ev_rank_vs_realized_pnl_rank")
        if isinstance(direction, dict)
        else None
    )
    threshold = counterfactual.get("threshold_min_ev") if isinstance(counterfactual, dict) else None
    avoided_count = (
        counterfactual.get("avoided_loss_count") if isinstance(counterfactual, dict) else None
    )
    avoided_usdt = (
        counterfactual.get("avoided_loss_usdt") if isinstance(counterfactual, dict) else None
    )
    false_kill_count = (
        counterfactual.get("false_kill_profit_count") if isinstance(counterfactual, dict) else None
    )
    false_kill_usdt = (
        counterfactual.get("false_kill_profit_usdt") if isinstance(counterfactual, dict) else None
    )
    net_impact = counterfactual.get("net_impact_usdt") if isinstance(counterfactual, dict) else None
    message = recommendation.get("message") if isinstance(recommendation, dict) else None
    lines = [
        "EV shadow replay report",
        f"actor: {actor or 'all'}",
        f"n: {sample.get('n')} (min_required={sample.get('min_required_n')}, "
        f"excluded={sample.get('excluded_unusable_rows')})",
        f"data_origin_breakdown: {sample.get('data_origin_breakdown')}",
        f"EV>0: n={ev_gt.get('n') if isinstance(ev_gt, dict) else None}, "
        f"win_rate={ev_gt.get('win_rate') if isinstance(ev_gt, dict) else None}, "
        f"avg_realized_pnl={ev_gt.get('avg_realized_pnl') if isinstance(ev_gt, dict) else None}",
        f"EV<0: n={ev_lt.get('n') if isinstance(ev_lt, dict) else None}, "
        f"win_rate={ev_lt.get('win_rate') if isinstance(ev_lt, dict) else None}, "
        f"avg_realized_pnl={ev_lt.get('avg_realized_pnl') if isinstance(ev_lt, dict) else None}",
        f"directionality: match_rate={match_rate}, pearson={pearson}, spearman={spearman}",
        "counterfactual enforce: "
        f"threshold={threshold}, avoid_losses={avoided_count} ({avoided_usdt}), "
        f"false_kill_winners={false_kill_count} ({false_kill_usdt}), net={net_impact}",
        f"recommendation: {message}",
        "human gate: do not change EV_GATE_MODE automatically",
    ]
    return "\n".join(lines)


def write_ev_shadow_report(report: dict[str, object]) -> dict[str, str]:
    directory = Path(EV_SHADOW_REPORT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    json_path = directory / f"ev-shadow-report-{stamp}.json"
    text_path = directory / f"ev-shadow-report-{stamp}.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(str(report.get("human_readable", "")) + "\n", encoding="utf-8")
    return {"json": str(json_path), "text": str(text_path)}


def _maybe_open_scorecard_outcome(
    execution: ExecutionResult,
    intent: OrderIntent,
    new_qty_after_buy: Decimal,
) -> None:
    """Record an opened outcome when a scorecard-sourced BUY produces fill."""
    _ = new_qty_after_buy
    if intent.side != "buy":
        return
    if intent.source.origin != "scorecard" or not intent.source.scorecard_id:
        return
    if execution.avg_price is None or execution.filled_qty <= 0:
        return
    with connect() as conn:
        row = conn.execute(
            "select source, payload_json from scorecards where scorecard_id = ?",
            (intent.source.scorecard_id,),
        ).fetchone()
    source = row["source"] if row else "unknown"
    trailing_pct = "0"
    if row is not None:
        try:
            payload = json.loads(str(row["payload_json"]))
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if isinstance(metadata, dict) and metadata.get("trailing_pct") is not None:
                tp = Decimal(str(metadata["trailing_pct"]))
                if Decimal("0") < tp < Decimal("1"):
                    trailing_pct = str(tp)
        except (json.JSONDecodeError, InvalidOperation, ValueError):
            pass
    opened_qty = _q8(execution.filled_qty)
    opened_avg_cost = _q8(execution.avg_price)
    opened_cost_basis = _q8(opened_qty * opened_avg_cost)
    outcome_id = str(uuid4())
    with connect() as conn:
        conn.execute(
            """
            insert into scorecard_outcomes
              (outcome_id, scorecard_id, actor, symbol, source, action,
               opened_intent_id, opened_at, opened_qty, opened_avg_cost,
               opened_cost_basis, status, closed_at, closed_realized_pnl,
               closed_return_pct, notes, trailing_pct, peak_mark)
            values (?,?,?,?,?,?,?,?,?,?,?,'open',NULL,NULL,NULL,NULL,?,?)
            """,
            (
                outcome_id,
                intent.source.scorecard_id,
                intent.actor,
                intent.symbol,
                source,
                intent.side,
                str(intent.intent_id),
                _now().isoformat(),
                _q8s(opened_qty),
                _q8s(opened_avg_cost),
                _q8s(opened_cost_basis),
                trailing_pct,
                "0",
            ),
        )
        conn.execute(
            "update ev_estimates set outcome_id = ?, updated_at = ? where scorecard_id = ?",
            (outcome_id, _now().isoformat(), intent.source.scorecard_id),
        )
        conn.commit()


def _maybe_close_scorecard_outcomes(
    actor: str,
    symbol: str,
    realized_delta: Decimal,
    new_qty: Decimal,
) -> None:
    if new_qty != Decimal("0"):
        return
    with connect() as conn:
        rows = conn.execute(
            "select outcome_id, opened_cost_basis from scorecard_outcomes "
            "where actor = ? and symbol = ? and status = 'open'",
            (actor, symbol),
        ).fetchall()
    if not rows:
        return
    total_basis = sum(Decimal(row["opened_cost_basis"]) for row in rows) or Decimal("1")
    closed_at = _now().isoformat()
    notes = "split-attribution" if len(rows) > 1 else None
    closed_ids: list[str] = []
    with connect() as conn:
        for row in rows:
            basis = Decimal(row["opened_cost_basis"])
            share = (basis / total_basis) if total_basis else Decimal("0")
            attributed = _q8(realized_delta * share)
            return_pct = _q8(attributed / basis) if basis > 0 else Decimal("0")
            conn.execute(
                "update scorecard_outcomes set status = 'closed', "
                "closed_at = ?, closed_realized_pnl = ?, closed_return_pct = ?, "
                "notes = ? where outcome_id = ?",
                (closed_at, _q8s(attributed), _q8s(return_pct), notes, row["outcome_id"]),
            )
            closed_ids.append(str(row["outcome_id"]))
        conn.commit()

    for outcome_id in closed_ids:
        outcome = _load_outcome_for_reflection(outcome_id)
        if outcome:
            _update_factor_attribution(outcome)
            try:
                _record_closed_outcome_memory(outcome)
            except Exception:
                logger.exception("failed to record closed outcome memory")
            if _push_outcome_reflection(outcome):
                _mark_outcome_reflected(outcome_id)
                try:
                    _record_reflection_memory(outcome)
                except Exception:
                    logger.exception("failed to record reflection memory")


def _load_outcome_for_reflection(outcome_id: str) -> dict[str, object] | None:
    with connect() as conn:
        row = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at, trailing_pct, peak_mark "
            "from scorecard_outcomes where outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    if row is None:
        return None
    return _outcome_row_to_dict(row)


def _paper_autonomy_positions(actor: str) -> tuple[list[dict[str, object]], Decimal, list[str]]:
    with connect() as conn:
        rows = conn.execute(
            "select symbol, paper_qty, paper_avg_cost, realized_pnl, venue, last_updated "
            "from paper_positions where actor = ? and cast(paper_qty as real) > 0 "
            "order by symbol",
            (actor,),
        ).fetchall()
    positions: list[dict[str, object]] = []
    unrealized_total = Decimal("0")
    mark_failures: list[str] = []
    for row in rows:
        qty = Decimal(str(row["paper_qty"]))
        avg_cost = Decimal(str(row["paper_avg_cost"]))
        mark, source = _mark_for_symbol(str(row["symbol"]))
        unrealized: Decimal | None = None
        mark_value: Decimal | None = None
        if mark is None:
            mark_failures.append(str(row["symbol"]))
        else:
            unrealized = _q8(qty * (mark - avg_cost))
            mark_value = _q8(qty * mark)
            unrealized_total += unrealized
        positions.append(
            {
                "symbol": row["symbol"],
                "venue": row["venue"],
                "qty": _q8s(qty),
                "avg_cost": _q8s(avg_cost),
                "realized_pnl": _q8s(Decimal(str(row["realized_pnl"]))),
                "mark_price": _q8s(mark) if mark is not None else None,
                "mark_value": _q8s(mark_value) if mark_value is not None else None,
                "unrealized_pnl": _q8s(unrealized) if unrealized is not None else None,
                "mark_source": source,
                "last_updated": row["last_updated"],
            }
        )
    return positions, _q8(unrealized_total), mark_failures


def _daily_realized_pnl_curve(actor: str, date: str) -> tuple[Decimal, Decimal]:
    with connect() as conn:
        rows = conn.execute(
            "select realized_delta from daily_pnl where actor = ? and date = ? "
            "order by created_at asc",
            (actor, date),
        ).fetchall()
    equity = Decimal("0")
    peak = Decimal("0")
    max_drawdown = Decimal("0")
    for row in rows:
        equity += Decimal(str(row["realized_delta"]))
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return _q8(equity), _q8(max_drawdown)


def _cumulative_realized_pnl(actor: str) -> Decimal:
    with connect() as conn:
        row = conn.execute(
            "select coalesce(sum(cast(realized_pnl as real)), 0.0) "
            "from paper_positions where actor = ?",
            (actor,),
        ).fetchone()
    return _q8(Decimal(str(row[0] if row is not None else "0")))


def _paper_decision_count(actor: str, date: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "select count(*) from intents where json_extract(payload_json, '$.actor') = ? "
            "and json_extract(payload_json, '$.mode') = 'paper' and substr(created_at, 1, 10) = ?",
            (actor, date),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def _consecutive_closed_losses(actor: str) -> int:
    with connect() as conn:
        rows = conn.execute(
            "select closed_realized_pnl from scorecard_outcomes "
            "where actor = ? and status = 'closed' and closed_realized_pnl is not null "
            "order by closed_at desc limit 50",
            (actor,),
        ).fetchall()
    losses = 0
    for row in rows:
        pnl = _decimal_or_none(row["closed_realized_pnl"])
        if pnl is None or pnl >= 0:
            break
        losses += 1
    return losses


def _live_autonomy_switch_status(actor: str) -> dict[str, object]:
    settings = get_live_autonomy_settings(actor=actor)
    if isinstance(settings, JSONResponse):
        actor_enabled = False
    else:
        actor_enabled = bool(settings.get("enabled"))
    return {
        "global_enabled": LIVE_AUTONOMY_GLOBAL_ENABLED,
        "actor_enabled": actor_enabled,
        "kill_switch_active": _live_kill_switch_active(),
    }


def _automatic_memory_count(actor: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "select count(*) from memory_entries "
            "where created_by = 'orchestrator' "
            "and trigger in ('closed_outcome','reflection') "
            "and tags_json like ?",
            (f"%{actor}%",),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def _closed_outcome_count(actor: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "select count(*) from scorecard_outcomes where actor = ? and status = 'closed'",
            (actor,),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def _calibration_bucket_coverage() -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select source, asset_type, heuristic_bucket, sample_count "
            "from conviction_calibration where sample_count > 0 "
            "order by source, asset_type, heuristic_bucket"
        ).fetchall()
    buckets = [
        {
            "source": row["source"],
            "asset_type": row["asset_type"],
            "heuristic_bucket": row["heuristic_bucket"],
            "sample_count": int(row["sample_count"]),
        }
        for row in rows
    ]
    return {"bucket_count": len(buckets), "buckets": buckets}


def _feedback_loop_progress(actor: str) -> dict[str, object]:
    ev_samples, excluded = _load_ev_shadow_samples(actor)
    calibration = _calibration_bucket_coverage()
    automatic_memories = _automatic_memory_count(actor)
    closed_outcomes = _closed_outcome_count(actor)
    calibration_bucket_count = int(str(calibration["bucket_count"]))
    readiness = {
        "ev_ready": len(ev_samples) >= FEEDBACK_EV_READY_N,
        "closed_outcomes_ready": closed_outcomes >= FEEDBACK_CLOSED_OUTCOME_READY_N,
        "calibration_ready": calibration_bucket_count >= FEEDBACK_CALIBRATION_READY_BUCKETS,
        "automatic_memory_ready": automatic_memories >= FEEDBACK_MEMORY_AUTOMATIC_READY_N,
    }
    ready_for_checkup = all(bool(value) for value in readiness.values())
    return {
        "ev_shadow": {
            "n": len(ev_samples),
            "excluded_unusable_rows": excluded,
            "ready_n": FEEDBACK_EV_READY_N,
        },
        "closed_outcomes": {
            "n": closed_outcomes,
            "ready_n": FEEDBACK_CLOSED_OUTCOME_READY_N,
        },
        "automatic_memories": {
            "n": automatic_memories,
            "ready_n": FEEDBACK_MEMORY_AUTOMATIC_READY_N,
        },
        "calibration": {
            **calibration,
            "ready_bucket_count": FEEDBACK_CALIBRATION_READY_BUCKETS,
        },
        "readiness": readiness,
        "ready_for_feedback_loop_checkup": ready_for_checkup,
        "next_action": (
            "ready_for_feedback_loop_checkup_and_ev_enforce_review"
            if ready_for_checkup
            else "keep_accumulating_paper_feedback"
        ),
        "human_gate_required": True,
        "auto_enforce": False,
    }


def _paper_bootstrap_runtime_status(actor: str) -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select symbol, enabled, gate_conviction, last_run_at, next_run_at "
            "from watchlist_entries where actor = ? and source_origin = ? order by symbol",
            (actor, PAPER_FEEDBACK_BOOTSTRAP_ORIGIN),
        ).fetchall()
    watchlist = [
        {
            "symbol": row["symbol"],
            "enabled": bool(row["enabled"]),
            "gate_conviction": row["gate_conviction"],
            "last_run_at": row["last_run_at"],
            "next_run_at": row["next_run_at"],
        }
        for row in rows
    ]
    return {
        "env_enabled": _env_bool(PAPER_FEEDBACK_BOOTSTRAP_ENABLED_ENV, default=False),
        "env_actor_matches": os.getenv(PAPER_FEEDBACK_ACTOR_ENV, "").strip() == actor,
        "origin": PAPER_FEEDBACK_BOOTSTRAP_ORIGIN,
        "halt": _paper_bootstrap_halt_status(actor),
        "guardrail_state": _paper_bootstrap_guardrail_state(actor),
        "guardrail_thresholds": _paper_bootstrap_guardrail_thresholds(),
        "watchlist": watchlist,
        "watchlist_enabled_count": sum(1 for row in watchlist if bool(row["enabled"])),
        "manual_resume_required": bool(_paper_bootstrap_halt_status(actor)["halted"]),
    }


def build_paper_autonomy_observability(
    actor: str, *, include_guardrails: bool = True, date: str | None = None
) -> dict[str, object]:
    today = date or _today()
    positions, unrealized, mark_failures = _paper_autonomy_positions(actor)
    today_realized, max_drawdown = _daily_realized_pnl_curve(actor, today)
    consecutive_losses = _consecutive_closed_losses(actor)
    alerts = _paper_autonomy_alerts(max_drawdown, consecutive_losses, mark_failures)
    return {
        "actor": actor,
        "date": today,
        "paper": True,
        "positions": positions,
        "today_decision_count": _paper_decision_count(actor, today),
        "today_realized_pnl": _q8s(today_realized),
        "current_unrealized_pnl": _q8s(unrealized),
        "current_total_pnl": _q8s(today_realized + unrealized),
        "cumulative_realized_pnl": _q8s(_cumulative_realized_pnl(actor)),
        "max_drawdown_today_usdt": _q8s(max_drawdown),
        "consecutive_closed_losses": consecutive_losses,
        "mark_failures": mark_failures,
        "ev_gate": {"mode": EV_GATE_MODE if EV_GATE_MODE in {"shadow", "enforce"} else "shadow"},
        "live_autonomy": _live_autonomy_switch_status(actor),
        "bootstrap": _paper_bootstrap_runtime_status(actor) if include_guardrails else None,
        "feedback_loop_progress": _feedback_loop_progress(actor),
        "alerts": alerts,
        "privacy": {
            "scope": "private_actor_channel",
            "public_persistence": False,
        },
    }


def _paper_autonomy_alerts(
    max_drawdown: Decimal, consecutive_losses: int, mark_failures: list[str]
) -> list[dict[str, object]]:
    alerts: list[dict[str, object]] = []
    if max_drawdown >= AUTONOMY_DRAWDOWN_ALERT_USDT:
        alerts.append(
            {
                "type": "drawdown",
                "severity": "warning",
                "message": "daily drawdown threshold breached",
                "threshold_usdt": _q8s(AUTONOMY_DRAWDOWN_ALERT_USDT),
                "observed_usdt": _q8s(max_drawdown),
            }
        )
    if consecutive_losses >= AUTONOMY_CONSECUTIVE_LOSS_ALERT_N:
        alerts.append(
            {
                "type": "consecutive_losses",
                "severity": "warning",
                "message": "consecutive closed losses threshold breached",
                "threshold_count": AUTONOMY_CONSECUTIVE_LOSS_ALERT_N,
                "observed_count": consecutive_losses,
            }
        )
    if mark_failures:
        alerts.append(
            {
                "type": "mark_price_unavailable",
                "severity": "warning",
                "message": "mark price unavailable for open paper positions",
                "symbols": mark_failures,
            }
        )
    return alerts


def notify_paper_autonomy_alerts(actor: str) -> dict[str, object]:
    overview = build_paper_autonomy_observability(actor)
    alerts = overview["alerts"]
    sent = 0
    if isinstance(alerts, list):
        for alert in alerts:
            if isinstance(alert, dict) and _notify_actor_event(actor, "alert", {"alert": alert}):
                sent += 1
    alert_count = len(alerts) if isinstance(alerts, list) else 0
    return {"actor": actor, "alert_count": alert_count, "sent": sent}


def _scorecard_metadata(scorecard_id: str) -> dict[str, str]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?", (scorecard_id,)
        ).fetchone()
    if row is None:
        return {}
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _scorecard_factors(scorecard_id: str) -> list[dict[str, object]]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?", (scorecard_id,)
        ).fetchone()
    if row is None:
        return []
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return []
    raw_factors = payload.get("factors") if isinstance(payload, dict) else None
    if not isinstance(raw_factors, list):
        return []
    factors: list[dict[str, object]] = []
    for raw in raw_factors:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        direction = raw.get("direction")
        if not isinstance(name, str) or direction not in {"support", "oppose", "neutral"}:
            continue
        factors.append({"name": name, "direction": str(direction), "score": raw.get("score")})
    return factors


def _update_factor_attribution(outcome: dict[str, object]) -> None:
    factors = _scorecard_factors(str(outcome.get("scorecard_id", "")))
    if not factors:
        return
    try:
        pnl = Decimal(str(outcome.get("closed_realized_pnl") or "0"))
    except (InvalidOperation, ValueError):
        pnl = Decimal("0")
    win_delta = 1 if pnl > 0 else 0
    loss_delta = abs(pnl) if pnl < 0 else Decimal("0")
    now_iso = _now().isoformat()
    with connect() as conn:
        for factor in factors:
            conn.execute(
                """
                insert into factor_attribution
                  (actor, factor, direction, support_count, win_count,
                   total_pnl, loss_contribution, updated_at)
                values (?, ?, ?, 1, ?, ?, ?, ?)
                on conflict(actor, factor, direction) do update set
                  support_count = support_count + 1,
                  win_count = win_count + excluded.win_count,
                  total_pnl = printf(
                    '%.8f',
                    cast(total_pnl as real) + cast(excluded.total_pnl as real)
                  ),
                  loss_contribution = printf(
                    '%.8f',
                    cast(loss_contribution as real) + cast(excluded.loss_contribution as real)
                  ),
                  updated_at = excluded.updated_at
                """,
                (
                    str(outcome.get("actor")),
                    str(factor["name"]),
                    str(factor["direction"]),
                    win_delta,
                    _q8s(pnl),
                    _q8s(loss_delta),
                    now_iso,
                ),
            )
        conn.commit()


def _scorecard_payload_by_id(scorecard_id: str) -> dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?", (scorecard_id,)
        ).fetchone()
    if row is None:
        return {}
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ev_estimate_for_scorecard(scorecard_id: str) -> dict[str, object] | None:
    with connect() as conn:
        row = conn.execute(
            "select scorecard_id, outcome_id, actor, symbol, mode, gate_result, reason, "
            "p, tp_pct, sl_pct, fee_bps, slippage_bps, funding_bps, min_ev, ev, "
            "created_at, updated_at from ev_estimates where scorecard_id = ?",
            (scorecard_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _ev_direction_mismatched(outcome: dict[str, object]) -> bool:
    ev_row = _ev_estimate_for_scorecard(str(outcome.get("scorecard_id", "")))
    if ev_row is None or ev_row.get("ev") is None:
        return False
    pnl = _decimal_or_none(outcome.get("closed_realized_pnl"))
    ev = _decimal_or_none(ev_row.get("ev"))
    if pnl is None or ev is None:
        return False
    return not _sign_matches(ev, pnl)


def _factor_summary(factors: list[dict[str, object]]) -> str:
    if not factors:
        return "none"
    parts = []
    for factor in factors[:8]:
        score = factor.get("score")
        suffix = f"/{score}" if score is not None else ""
        parts.append(f"{factor['name']}:{factor['direction']}{suffix}")
    return ", ".join(parts)


def _record_closed_outcome_memory(outcome: dict[str, object]) -> None:
    pnl = _decimal_or_none(outcome.get("closed_realized_pnl")) or Decimal("0")
    return_pct = _decimal_or_none(outcome.get("closed_return_pct")) or Decimal("0")
    ev_mismatch = _ev_direction_mismatched(outcome)
    pnl_threshold = _memory_threshold()
    return_threshold = _memory_return_threshold()
    significant_pnl = abs(pnl) >= pnl_threshold
    significant_return = abs(return_pct) >= return_threshold
    if not significant_pnl and not significant_return and not ev_mismatch:
        return
    scorecard_id = str(outcome.get("scorecard_id") or "")
    payload = _scorecard_payload_by_id(scorecard_id)
    metadata = _scorecard_metadata_from_payload(payload)
    origin = _metadata_data_origin(metadata)
    gate_conviction = _metadata_gate_conviction(metadata)
    ev_row = _ev_estimate_for_scorecard(scorecard_id) or {}
    factors = _scorecard_factors(scorecard_id)
    tags = _memory_tags(
        outcome.get("actor"),
        outcome.get("source"),
        outcome.get("action"),
        "closed_outcome",
        "significant_pnl" if significant_pnl else None,
        "significant_return" if significant_return else None,
        "ev_mismatch" if ev_mismatch else None,
        _scorecard_origin_tags(metadata),
        _memory_factor_tags(payload),
    )
    record_memory(
        "lesson",
        str(outcome.get("symbol") or ""),
        (
            f"Closed outcome lesson for {outcome.get('symbol')}; "
            f"realized_pnl={outcome.get('closed_realized_pnl')}; "
            f"return_pct={outcome.get('closed_return_pct')}; "
            f"memory_pnl_threshold={pnl_threshold}; "
            f"memory_return_threshold={return_threshold}; "
            f"origin={origin}; gate_conviction={gate_conviction or 'standard'}; "
            f"action={outcome.get('action')}; conviction={payload.get('conviction')}; "
            f"ev={ev_row.get('ev')}; ev_gate={ev_row.get('gate_result')}; "
            f"ev_direction_mismatch={ev_mismatch}; factors={_factor_summary(factors)}"
        ),
        {
            "table": "scorecard_outcomes",
            "outcome_id": str(outcome.get("outcome_id") or ""),
            "scorecard_id": scorecard_id,
            "trigger": "closed_outcome",
        },
        tags=tags,
        confidence=(
            str(payload.get("conviction")) if payload.get("conviction") is not None else None
        ),
        trigger="closed_outcome",
        created_by="orchestrator",
    )


def _record_reflection_memory(outcome: dict[str, object]) -> None:
    scorecard_id = str(outcome.get("scorecard_id") or "")
    payload = _scorecard_payload_by_id(scorecard_id)
    record_memory(
        "lesson",
        str(outcome.get("symbol") or ""),
        (
            f"Reflection accepted for closed outcome {outcome.get('symbol')}; "
            f"realized_pnl={outcome.get('closed_realized_pnl')}; "
            f"return_pct={outcome.get('closed_return_pct')}; "
            f"action={outcome.get('action')}; conviction={payload.get('conviction')}; "
            f"factors={_factor_summary(_scorecard_factors(scorecard_id))}"
        ),
        {
            "table": "scorecard_outcomes",
            "outcome_id": str(outcome.get("outcome_id") or ""),
            "scorecard_id": scorecard_id,
            "trigger": "reflection",
        },
        tags=_memory_tags(
            outcome.get("actor"), outcome.get("source"), outcome.get("action"), "reflection"
        ),
        confidence=(
            str(payload.get("conviction")) if payload.get("conviction") is not None else None
        ),
        trigger="reflection",
        created_by="orchestrator",
    )


def _record_risk_reject_memory(intent: OrderIntent, decision: RiskDecision) -> None:
    reason_codes = [reason.code for reason in decision.reasons]
    primary_reason = reason_codes[0] if reason_codes else "UNKNOWN_RISK_REJECT"
    record_memory(
        "observation",
        intent.symbol,
        (
            f"Risk engine rejected {intent.mode} {intent.side} intent for {intent.symbol}; "
            f"reason_codes={','.join(reason_codes) or primary_reason}; "
            f"quantity={intent.quantity.kind}:{intent.quantity.value}; venue={intent.venue}"
        ),
        {
            "table": "intents",
            "intent_id": str(intent.intent_id),
            "decision_id": str(decision.decision_id),
            "trigger": "risk_reject",
        },
        tags=_memory_tags(intent.actor, intent.mode, intent.venue, intent.side, primary_reason),
        trigger="risk_reject",
        created_by="orchestrator",
    )


def _compute_alpha_return(
    raw_return: Decimal, benchmark_symbol: str | None, benchmark_open: str | None
) -> tuple[Decimal, str | None]:
    if not benchmark_symbol or benchmark_symbol == "self":
        return _q8(raw_return), "benchmark unavailable; using raw return as alpha"
    if not benchmark_open:
        return _q8(raw_return), "benchmark open price unavailable; using raw return as alpha"
    try:
        open_price = Decimal(str(benchmark_open))
        if open_price <= 0:
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return _q8(raw_return), "benchmark open price invalid; using raw return as alpha"
    close_price, _source = _mark_for_symbol_str(benchmark_symbol)
    if close_price is None:
        return _q8(raw_return), "benchmark close price unavailable; using raw return as alpha"
    benchmark_return = (close_price - open_price) / open_price
    return _q8(raw_return - benchmark_return), None


def _holding_days(opened_at: str | None, closed_at: str | None) -> int:
    try:
        opened = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        closed = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(0, (closed.date() - opened.date()).days)


def _push_outcome_reflection(outcome: dict[str, object]) -> bool:
    metadata = _scorecard_metadata(str(outcome.get("scorecard_id", "")))
    closed_return = str(outcome.get("closed_return_pct") or "0")
    try:
        raw_return = _q8(Decimal(closed_return))
    except (InvalidOperation, ValueError):
        raw_return = Decimal("0")
    benchmark_symbol = metadata.get("benchmark_symbol")
    alpha_return, alpha_note = _compute_alpha_return(
        raw_return, benchmark_symbol, metadata.get("benchmark_open_price")
    )
    payload = {
        "ticker": metadata.get("ta_ticker") or outcome.get("symbol"),
        "trade_date": metadata.get("ta_date") or str(outcome.get("opened_at", ""))[:10],
        "raw_return": _q8s(raw_return),
        "alpha_return": _q8s(alpha_return),
        "holding_days": _holding_days(
            str(outcome.get("opened_at") or ""), str(outcome.get("closed_at") or "")
        ),
        "provider": metadata.get("provider"),
        "benchmark_name": benchmark_symbol or "paper-position-baseline",
    }
    if alpha_note:
        payload["alpha_note"] = alpha_note
    try:
        response = httpx.post(
            f"{ANALYSIS_ADAPTER_URL}/reflect/outcome",
            json=payload,
            timeout=REFLECT_TIMEOUT_SEC,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return bool(isinstance(body, dict) and body.get("ok") and body.get("reflected"))


def _mark_outcome_reflected(outcome_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "update scorecard_outcomes set reflected_at = ? where outcome_id = ?",
            (_now().isoformat(), outcome_id),
        )
        conn.commit()


def _update_position(execution: ExecutionResult, intent: OrderIntent) -> None:
    if execution.avg_price is None or execution.filled_qty == Decimal("0"):
        return
    fill_qty = _q8(execution.filled_qty)
    fill_price = execution.avg_price
    qty_col = "live_qty" if intent.mode == "live" else "paper_qty"
    avg_col = "live_avg_cost" if intent.mode == "live" else "paper_avg_cost"
    with connect() as conn:
        row = conn.execute(
            "select qty, avg_cost, total_cost, realized_pnl, "
            "paper_qty, paper_avg_cost, live_qty, live_avg_cost, venue from paper_positions "
            "where actor = ? and symbol = ?",
            (intent.actor, intent.symbol),
        ).fetchone()
        old_bucket_qty = Decimal(row[qty_col]) if row else Decimal("0")
        old_bucket_avg = Decimal(row[avg_col]) if row else Decimal("0")
        old_realized = Decimal(row["realized_pnl"]) if row else Decimal("0")
        paper_qty = Decimal(row["paper_qty"]) if row else Decimal("0")
        paper_avg = Decimal(row["paper_avg_cost"]) if row else Decimal("0")
        live_qty = Decimal(row["live_qty"]) if row else Decimal("0")
        live_avg = Decimal(row["live_avg_cost"]) if row else Decimal("0")

        if intent.side == "buy":
            new_bucket_qty = _q8(old_bucket_qty + fill_qty)
            bucket_total = _q8((old_bucket_qty * old_bucket_avg) + (fill_qty * fill_price))
            new_bucket_avg = _q8(bucket_total / new_bucket_qty) if new_bucket_qty else Decimal("0")
            realized = old_realized
            realized_delta = Decimal("0")
        else:
            sell_qty = min(fill_qty, old_bucket_qty)
            new_bucket_qty = _q8(old_bucket_qty - sell_qty)
            realized_delta = _q8(sell_qty * (fill_price - old_bucket_avg))
            realized = _q8(old_realized + realized_delta)
            new_bucket_avg = _q8(old_bucket_avg if new_bucket_qty else Decimal("0"))

        if intent.mode == "live":
            live_qty = new_bucket_qty
            live_avg = new_bucket_avg
        else:
            paper_qty = new_bucket_qty
            paper_avg = new_bucket_avg

        new_qty = _q8(paper_qty + live_qty)
        paper_cost = _q8(paper_qty * paper_avg)
        live_cost = _q8(live_qty * live_avg)
        total_cost = _q8(paper_cost + live_cost)
        avg_cost = _q8(total_cost / new_qty) if new_qty else Decimal("0")

        conn.execute(
            """
            insert into paper_positions
            (actor, symbol, qty, avg_cost, total_cost, realized_pnl,
             paper_qty, paper_avg_cost, live_qty, live_avg_cost, venue, last_updated)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(actor, symbol) do update set
                qty = excluded.qty,
                avg_cost = excluded.avg_cost,
                total_cost = excluded.total_cost,
                realized_pnl = excluded.realized_pnl,
                paper_qty = excluded.paper_qty,
                paper_avg_cost = excluded.paper_avg_cost,
                live_qty = excluded.live_qty,
                live_avg_cost = excluded.live_avg_cost,
                venue = excluded.venue,
                last_updated = excluded.last_updated
            """,
            (
                intent.actor,
                intent.symbol,
                _q8s(new_qty),
                _q8s(avg_cost),
                _q8s(total_cost),
                _q8s(realized),
                _q8s(paper_qty),
                _q8s(paper_avg),
                _q8s(live_qty),
                _q8s(live_avg),
                intent.venue,
                _now().isoformat(),
            ),
        )
        if realized_delta != Decimal("0"):
            conn.execute(
                "insert into daily_pnl "
                "(actor, date, realized_delta, symbol, venue, created_at) values (?,?,?,?,?,?)",
                (
                    intent.actor,
                    _today(),
                    _q8s(realized_delta),
                    intent.symbol,
                    intent.venue,
                    _now().isoformat(),
                ),
            )
        conn.commit()
        try:
            if intent.side == "buy":
                _maybe_open_scorecard_outcome(execution, intent, new_qty)
            else:
                _maybe_close_scorecard_outcomes(
                    intent.actor, intent.symbol, realized_delta, new_qty
                )
        except Exception:
            pass


def _mark_for_symbol_str(symbol: str) -> tuple[Decimal | None, str | None]:
    return _mark_for_symbol(symbol)


def _mark_for_symbol(symbol: str) -> tuple[Decimal | None, str | None]:
    try:
        params = {"symbol": symbol}
        if not symbol.upper().endswith("USDT"):
            params["asset_type"] = "stock"
        response = httpx.get(f"{_market_url()}/ticker", params=params, timeout=3.0)
        response.raise_for_status()
        payload = response.json()
        return Decimal(str(payload["price"])), str(payload.get("source", "binance"))
    except (httpx.HTTPError, KeyError, ValueError):
        return None, None


def _parse_nl_to_intent_fields(message: str) -> dict[str, object]:
    """Call the Claude API and return the parsed JSON dict."""
    response = httpx.post(
        CLAUDE_API_URL,
        headers={
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": HERMES_MODEL,
            "max_tokens": 256,
            "system": _HERMES_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": message}],
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    text = str(payload["content"][0]["text"]).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Claude response was not a JSON object")
    return {str(key): value for key, value in parsed.items()}


def _build_intent_from_nl(nl: NLIntentRequest, fields: dict[str, object]) -> OrderIntent:
    """Construct an OrderIntent from extracted NL fields."""
    side_raw = str(fields["side"])
    if side_raw not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    side = cast(Literal["buy", "sell"], side_raw)

    order_type_raw = str(fields["order_type"])
    if order_type_raw not in {"market", "limit"}:
        raise ValueError("order_type must be market or limit")
    order_type = cast(Literal["market", "limit"], order_type_raw)

    quantity_kind_raw = str(fields["quantity_kind"])
    if quantity_kind_raw not in {"base", "quote"}:
        raise ValueError("quantity_kind must be base or quote")
    quantity_kind = cast(Literal["base", "quote"], quantity_kind_raw)

    limit_price = (
        Decimal(str(fields["limit_price"])) if fields.get("limit_price") is not None else None
    )

    return OrderIntent(
        intent_id=uuid4(),
        request_id=nl.request_id or uuid4(),
        idempotency_key=nl.idempotency_key,
        actor=nl.actor,
        created_at=_now(),
        mode=nl.mode,
        venue="binance_spot",
        symbol=str(fields["symbol"]),
        side=side,
        order_type=order_type,
        quantity=Quantity(
            kind=quantity_kind,
            value=Decimal(str(fields["quantity_value"])),
        ),
        limit_price=limit_price,
        time_in_force="GTC",
        reduce_only=False,
        leverage=None,
        stop_loss=None,
        take_profit=None,
        source=Source(
            origin="user_nl",
            scorecard_id=None,
            hermes_message_id=nl.hermes_message_id,
        ),
        client_confirmation_required=False,
    )


def _consume_live_unlock_or_error(
    token: str, actor: str, dry: bool, intent_id: UUID | None = None
) -> JSONResponse | None:
    """Validate or consume a single-use live-unlock token."""
    return _orchestrator_safety.consume_live_unlock_or_error(
        token=token,
        actor=actor,
        dry=dry,
        intent_id=intent_id,
        connect=connect,
        now=_now,
    )


def _scorecard_should_mark_consumed(
    response: JSONResponse | dict[str, object],
) -> bool:
    return _orchestrator_safety.scorecard_should_mark_consumed(response)


def _cancel_refresh_request(intent: OrderIntent, execution: ExecutionResult) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id=uuid4(),
        intent_id=intent.intent_id,
        decision_id=execution.decision_id,
        idempotency_key=intent.idempotency_key,
        confirmation_token=None,
        dry_run=False,
        submitted_at=_now(),
    )


def _call_cancel(intent: OrderIntent, execution: ExecutionResult) -> ExecutionResult:
    req = _cancel_refresh_request(intent, execution)
    response = httpx.post(
        f"{_execution_url()}/cancel",
        content=req.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-mode": intent.mode,
            "x-symbol": intent.symbol,
            "x-venue-order-id": execution.venue_order_id or "",
            "x-order-type": intent.order_type,
        },
        timeout=15.0,
    )
    response.raise_for_status()
    return ExecutionResult.model_validate(response.json())


def _call_refresh(intent: OrderIntent, execution: ExecutionResult) -> ExecutionResult:
    req = _cancel_refresh_request(intent, execution)
    response = httpx.post(
        f"{_execution_url()}/refresh",
        content=req.model_dump_json(),
        headers={
            "content-type": "application/json",
            "x-mode": intent.mode,
            "x-symbol": intent.symbol,
            "x-venue-order-id": execution.venue_order_id or "",
            "x-order-type": intent.order_type,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return ExecutionResult.model_validate(response.json())


def healthz() -> dict[str, str]:
    return {"status": "ok"}


def readyz() -> dict[str, str]:
    return {"status": "ready"}


def subscribe_notifications(req: NotificationSubscribeRequest) -> JSONResponse | dict[str, object]:
    if not _is_allowed_webhook_host(req.webhook_url):
        return JSONResponse(status_code=400, content={"code": "WEBHOOK_HOST_NOT_ALLOWED"})
    events = sorted({event for event in req.events if event in SUPPORTED_NOTIFICATION_EVENTS})
    if not events:
        return JSONResponse(status_code=400, content={"code": "NO_SUPPORTED_EVENTS"})
    secret = req.secret or str(uuid4())
    now_iso = _now().isoformat()
    with connect() as conn:
        conn.execute(
            """
            insert into notification_subscriptions
              (actor, webhook_url, secret, events_json, enabled, created_at, updated_at)
            values (?, ?, ?, ?, 1, ?, ?)
            on conflict(actor) do update set
              webhook_url = excluded.webhook_url,
              secret = excluded.secret,
              events_json = excluded.events_json,
              enabled = 1,
              updated_at = excluded.updated_at
            """,
            (req.actor, req.webhook_url, secret, json.dumps(events), now_iso, now_iso),
        )
        conn.commit()
    return {
        "actor": req.actor,
        "webhook_url": req.webhook_url,
        "events": events,
        "secret": secret,
    }


def unsubscribe_notifications(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        cursor = conn.execute(
            "update notification_subscriptions set enabled = 0, updated_at = ? where actor = ?",
            (_now().isoformat(), actor),
        )
        conn.commit()
    return {"actor": actor, "enabled": False, "changed": cursor.rowcount > 0}


def list_notification_deliveries(
    actor: str | None = None, limit: int = Query(default=20, ge=1)
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    clamped = min(limit, NOTIFICATION_HISTORY_LIMIT)
    with connect() as conn:
        rows = conn.execute(
            """
            select event_type, webhook_url, status_code, ok, error_class, created_at
            from notification_deliveries
            where actor = ?
            order by created_at desc, id desc
            limit ?
            """,
            (actor, clamped),
        ).fetchall()
    return {
        "actor": actor,
        "deliveries": [
            {
                "event_type": row["event_type"],
                "webhook_url": row["webhook_url"],
                "status_code": row["status_code"],
                "ok": bool(row["ok"]),
                "error_class": row["error_class"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
    }


def get_ev_shadow_report(
    actor: str | None = None,
    min_ev: str | None = None,
    write_report: bool = True,
    output: Literal["json", "text"] = "json",
) -> JSONResponse | dict[str, object] | Response:
    threshold: Decimal | None = None
    if min_ev is not None:
        threshold = _decimal_or_none(min_ev)
        if threshold is None:
            return JSONResponse(status_code=400, content={"code": "INVALID_MIN_EV"})
    report = build_ev_shadow_report(actor=actor, min_ev=threshold)
    if write_report:
        report["written_files"] = write_ev_shadow_report(report)
    if output == "text":
        return Response(
            content=str(report.get("human_readable", "")) + "\n",
            media_type="text/plain",
        )
    return report


def _live_kill_switch_active() -> bool:
    return _orchestrator_safety.live_kill_switch_active(connect=connect)


def _default_live_autonomy(actor: str) -> dict[str, object]:
    return {
        "actor": actor,
        "enabled": False,
        "daily_live_budget_usdt": "0",
        "per_live_trade_max_usdt": "50",
        "max_live_exposure_usdt": "0",
        "max_us_equity_exposure_usd": "0",
        "current_live_exposure_usdt": "0",
        "current_us_equity_exposure_usd": "0",
        "daily_live_trade_count_max": 3,
        "min_calibrated_conviction": "0.70",
        "min_closed_outcomes": 20,
        "allowed_sources": "tradingagents",
        "created_at": None,
        "updated_at": None,
    }


def _live_autonomy_row_to_dict(actor: str, row: sqlite3.Row | None) -> dict[str, object]:
    if row is None:
        return _default_live_autonomy(actor)
    max_exposure = str(row["max_live_exposure_usdt"])
    max_stock_exposure = str(row["max_us_equity_exposure_usd"])
    current_exposure = _check_live_exposure_cap(
        actor, Decimal("0"), Decimal(max_exposure), "binance_spot"
    )[1]
    current_stock_exposure = _check_live_exposure_cap(
        actor, Decimal("0"), Decimal(max_stock_exposure), "ibkr_us_equity"
    )[1]
    return {
        "actor": actor,
        "enabled": bool(row["enabled"]),
        "daily_live_budget_usdt": str(row["daily_live_budget_usdt"]),
        "per_live_trade_max_usdt": str(row["per_live_trade_max_usdt"]),
        "max_live_exposure_usdt": max_exposure,
        "max_us_equity_exposure_usd": max_stock_exposure,
        "current_live_exposure_usdt": _q8s(current_exposure) if current_exposure else "0",
        "current_us_equity_exposure_usd": (
            _q8s(current_stock_exposure) if current_stock_exposure else "0"
        ),
        "daily_live_trade_count_max": int(row["daily_live_trade_count_max"]),
        "min_calibrated_conviction": str(row["min_calibrated_conviction"]),
        "min_closed_outcomes": int(row["min_closed_outcomes"]),
        "allowed_sources": str(row["allowed_sources"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _check_drawdown_for_live_auto(actor: str) -> str | None:
    return _orchestrator_safety.check_drawdown_for_live_auto(
        actor=actor,
        get_pnl_today=get_pnl_today,
        daily_drawdown_hard_stop=Decimal(os.getenv("DAILY_DRAWDOWN_HARD_STOP_USDT", "1000")),
    )


def _check_live_exposure_cap(
    actor: str, proposed_notional: Decimal, max_cap: Decimal, venue: str
) -> tuple[bool, Decimal]:
    return _orchestrator_safety.check_live_exposure_cap(
        actor=actor,
        proposed_notional=proposed_notional,
        max_cap=max_cap,
        venue=venue,
        connect=connect,
        quantize=_q8,
    )


def _venue_for_scorecard(scorecard: dict[str, object]) -> str:
    metadata = scorecard.get("metadata") or {}
    if isinstance(metadata, dict) and str(metadata.get("asset_type") or "crypto") == "stock":
        return "ibkr_us_equity"
    return "binance_spot"


def stop_loss_watchdog_tick(now: datetime | None = None) -> dict[str, int]:
    if not STOP_LOSS_WATCHDOG_ENABLED:
        return {"checked": 0, "fired": 0, "skipped": 0, "failed": 0}
    _ = now
    with connect() as conn:
        rows = conn.execute(
            """
            select o.outcome_id, o.scorecard_id, o.actor, o.symbol, o.opened_intent_id,
                   o.opened_qty, o.trailing_pct, o.peak_mark, s.payload_json
            from scorecard_outcomes o
            join scorecards s on s.scorecard_id = o.scorecard_id
            where o.status = 'open'
            order by o.opened_at asc
            limit ?
            """,
            (STOP_LOSS_BATCH_LIMIT,),
        ).fetchall()

    checked = len(rows)
    fired = 0
    skipped = 0
    failed = 0
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                skipped += 1
                continue
            stop_loss = _optional_decimal(payload.get("stop_loss"))
            take_profit = _optional_decimal(payload.get("take_profit"))
            trailing_pct = _optional_decimal(row["trailing_pct"])
            trailing_enabled = trailing_pct is not None and trailing_pct > 0
            if stop_loss is None and take_profit is None and not trailing_enabled:
                skipped += 1
                continue
            mark, _source = _mark_for_symbol(str(row["symbol"]))
            if mark is None:
                skipped += 1
                continue
            peak = _peak_for_row(row)
            if mark > peak:
                peak = mark
                _update_peak_mark(str(row["outcome_id"]), peak)
            should_sell_static = (stop_loss is not None and mark <= stop_loss) or (
                take_profit is not None and mark >= take_profit
            )
            should_sell_trailing = False
            if trailing_enabled and peak > 0 and trailing_pct is not None:
                trailing_floor = peak * (Decimal("1") - trailing_pct)
                should_sell_trailing = mark < trailing_floor
            if not (should_sell_static or should_sell_trailing):
                skipped += 1
                continue
            reason = "trailing" if should_sell_trailing and not should_sell_static else "static"
            if _fire_protective_sell(row, reason=reason):
                fired += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    return {"checked": checked, "fired": fired, "skipped": skipped, "failed": failed}


def _optional_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _peak_for_row(row: sqlite3.Row) -> Decimal:
    try:
        return Decimal(str(row["peak_mark"] or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _update_peak_mark(outcome_id: str, peak: Decimal) -> None:
    with connect() as conn:
        conn.execute(
            "update scorecard_outcomes set peak_mark = ? where outcome_id = ?",
            (str(peak), outcome_id),
        )
        conn.commit()


def _fire_protective_sell(row: sqlite3.Row, *, reason: str = "static") -> bool:
    actor = str(row["actor"])
    symbol = str(row["symbol"])
    opened_intent_id = str(row["opened_intent_id"])
    mode = _opened_intent_mode(opened_intent_id)
    venue = _opened_intent_venue(opened_intent_id)
    qty = _protective_sell_qty(actor, symbol, mode, str(row["opened_qty"]))
    if qty <= 0:
        return False
    intent_id = uuid4()
    payload: dict[str, object] = {
        "intent_id": str(intent_id),
        "request_id": str(uuid4()),
        "idempotency_key": f"protective-{row['outcome_id']}-{reason}-{int(time.time())}",
        "actor": actor,
        "created_at": _now().isoformat(),
        "mode": mode,
        "venue": venue,
        "symbol": symbol,
        "side": "sell",
        "order_type": "market",
        "quantity": {"kind": "base", "value": _q8s(qty)},
        "limit_price": None,
        "time_in_force": "GTC",
        "reduce_only": True,
        "leverage": None,
        "stop_loss": None,
        "take_profit": None,
        "source": {
            "origin": "scorecard",
            "scorecard_id": str(row["scorecard_id"]),
            "hermes_message_id": None,
        },
        "client_confirmation_required": False,
    }
    headers: dict[str, str] = {}
    if mode == "live":
        headers["x-live-unlock"] = _mint_auto_unlock_bound_token(actor, intent_id)
    try:
        response = httpx.post(
            f"{ORCHESTRATOR_SELF_URL}/intents",
            json=payload,
            headers=headers,
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return _execution_filled(body)


def _execution_filled(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    execution = body.get("execution")
    if not isinstance(execution, dict):
        return False
    return execution.get("status") in {"simulated", "filled", "partial"}


def _opened_intent_mode(intent_id: str) -> Literal["paper", "live"]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from intents where intent_id = ?",
            (intent_id,),
        ).fetchone()
    if row is None:
        return "paper"
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return "paper"
    mode = payload.get("mode") if isinstance(payload, dict) else None
    return "live" if mode == "live" else "paper"


def _opened_intent_venue(intent_id: str) -> str:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from intents where intent_id = ?",
            (intent_id,),
        ).fetchone()
    if row is None:
        return "binance_spot"
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return "binance_spot"
    venue = payload.get("venue") if isinstance(payload, dict) else None
    return str(venue) if venue in {"binance_spot", "ibkr_us_equity"} else "binance_spot"


def _protective_sell_qty(actor: str, symbol: str, mode: str, opened_qty: str) -> Decimal:
    with connect() as conn:
        row = conn.execute(
            "select paper_qty, live_qty from paper_positions where actor = ? and symbol = ?",
            (actor, symbol),
        ).fetchone()
    if row is None:
        return Decimal("0")
    bucket_qty = Decimal(str(row["live_qty"] if mode == "live" else row["paper_qty"]))
    opened = Decimal(opened_qty)
    return _q8(min(bucket_qty, opened))


def _eligible_for_live_auto(
    actor: str, scorecard: dict[str, object], settings: dict[str, object]
) -> tuple[bool, str]:
    if not LIVE_AUTONOMY_GLOBAL_ENABLED:
        return False, "LIVE_AUTONOMY_GLOBAL_DISABLED"
    if _live_kill_switch_active():
        return False, "LIVE_AUTONOMY_KILL_SWITCH"
    if not settings.get("enabled"):
        return False, "ACTOR_NOT_OPTED_IN"
    source = str(scorecard.get("source", ""))
    allowed = {s.strip() for s in str(settings.get("allowed_sources", "")).split(",") if s.strip()}
    if source not in allowed:
        return False, "SOURCE_NOT_ALLOWED"
    try:
        conviction = Decimal(str(scorecard.get("conviction", "0")))
        min_conv = Decimal(str(settings.get("min_calibrated_conviction", "0.70")))
    except (InvalidOperation, ValueError):
        return False, "INVALID_CONVICTION"
    if conviction < min_conv:
        return False, "BELOW_MIN_CONVICTION"
    metadata = scorecard.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False, "MISSING_METADATA"
    asset_type = str(metadata.get("asset_type") or "crypto")
    heuristic_raw = metadata.get("heuristic_conviction")
    if heuristic_raw is None:
        return False, "MISSING_HEURISTIC"
    try:
        heuristic = Decimal(str(heuristic_raw))
    except (InvalidOperation, ValueError):
        return False, "INVALID_HEURISTIC"
    bucket = _bucket_label(heuristic)
    if bucket is None:
        return False, "HEURISTIC_OUT_OF_RANGE"
    with connect() as conn:
        cal = conn.execute(
            "select sample_count, calibrated_conviction from conviction_calibration "
            "where source = ? and asset_type = ? and heuristic_bucket = ?",
            (source, asset_type, bucket),
        ).fetchone()
    if cal is None:
        return False, "NO_CALIBRATION_DATA"
    if int(cal["sample_count"]) < int(str(settings.get("min_closed_outcomes", 20))):
        return False, "INSUFFICIENT_SAMPLES"
    try:
        calibrated_conviction = Decimal(str(cal["calibrated_conviction"]))
    except (InvalidOperation, ValueError):
        return False, "INVALID_CALIBRATION_DATA"
    if calibrated_conviction < min_conv:
        return False, "BELOW_MIN_CONVICTION"
    ev_allowed, ev_reason = _ev_gate(actor, scorecard, calibrated_conviction)
    if not ev_allowed:
        return False, ev_reason
    today = _today()
    with connect() as conn:
        spend = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    spent = Decimal(str(spend["spent_usdt"])) if spend else Decimal("0")
    count = int(spend["trade_count"]) if spend else 0
    venue = _venue_for_scorecard(scorecard)
    try:
        if venue == "ibkr_us_equity":
            max_cap = Decimal(str(settings.get("max_us_equity_exposure_usd", "0")))
            cap_not_set_reason = "MAX_US_EQUITY_EXPOSURE_NOT_SET"
            cap_breached_reason = "US_EQUITY_EXPOSURE_CAP_BREACHED"
        else:
            max_cap = Decimal(str(settings.get("max_live_exposure_usdt", "0")))
            cap_not_set_reason = "MAX_LIVE_EXPOSURE_NOT_SET"
            cap_breached_reason = "LIVE_EXPOSURE_CAP_BREACHED"
        proposed = Decimal(str(settings.get("per_live_trade_max_usdt", "0")))
    except (InvalidOperation, ValueError):
        return False, "INVALID_LIVE_EXPOSURE_CAP"
    if max_cap <= 0:
        return False, cap_not_set_reason
    allowed_by_cap, current_exposure = _check_live_exposure_cap(actor, proposed, max_cap, venue)
    if not allowed_by_cap:
        return False, f"{cap_breached_reason}:{_q8s(current_exposure)}/{max_cap}"
    budget = Decimal(str(settings.get("daily_live_budget_usdt", "0")))
    if spent + proposed > budget:
        return False, "DAILY_BUDGET_EXHAUSTED"
    if count >= int(str(settings.get("daily_live_trade_count_max", 3))):
        return False, "DAILY_TRADE_COUNT_EXHAUSTED"
    pnl_block = _check_drawdown_for_live_auto(actor)
    if pnl_block is not None:
        return False, pnl_block
    return True, "OK"


def _mint_user_live_unlock_token(actor: str) -> str:
    return _orchestrator_safety.mint_user_live_unlock_token(
        actor=actor,
        connect=connect,
        now=_now,
        ttl_min=LIVE_UNLOCK_TTL_MIN,
    )


def _mint_auto_unlock_bound_token(actor: str, intent_id: UUID | str) -> str:
    return _orchestrator_safety.mint_auto_unlock_bound_token(
        actor=actor,
        intent_id=intent_id,
        connect=connect,
        now=_now,
    )


def _record_live_autonomy_spend(actor: str, date: str, amount: Decimal) -> None:
    now_iso = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, date),
        ).fetchone()
        spent = _q8((Decimal(str(row["spent_usdt"])) if row else Decimal("0")) + amount)
        count = (int(row["trade_count"]) if row else 0) + 1
        conn.execute(
            "insert into live_autonomy_spend "
            "(actor, date, spent_usdt, trade_count, last_updated) values (?, ?, ?, ?, ?) "
            "on conflict(actor, date) do update set spent_usdt = excluded.spent_usdt, "
            "trade_count = excluded.trade_count, last_updated = excluded.last_updated",
            (actor, date, _q8s(spent), count, now_iso),
        )
        conn.commit()


def _fire_live_autonomous_trade(actor: str, scorecard_id: str, usdt_budget: Decimal) -> bool:
    intent_id = uuid4()
    token = _mint_auto_unlock_bound_token(actor, intent_id)
    payload = {
        "scorecard_id": scorecard_id,
        "actor": actor,
        "idempotency_key": f"live-auto-{actor}-{scorecard_id[:8]}-{int(time.time())}",
        "usdt_budget": str(usdt_budget),
        "position_fraction": "1.0",
        "mode": "live",
        "intent_id": str(intent_id),
    }
    try:
        response = httpx.post(
            f"{ORCHESTRATOR_SELF_URL}/intents/from_scorecard",
            json=payload,
            headers={"x-live-unlock": token},
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(body, dict) and body.get("status") == "executed"


def live_auto_trade_tick(now: datetime | None = None) -> dict[str, object]:
    if not LIVE_AUTONOMY_GLOBAL_ENABLED:
        return {"placed": 0, "skipped": 0, "reason": "GLOBAL_DISABLED"}
    if _live_kill_switch_active():
        return {"placed": 0, "skipped": 0, "reason": "KILL_SWITCH"}
    current = now or _now()
    placed = 0
    skipped = 0
    with connect() as conn:
        actors = conn.execute(
            "select actor, enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, max_us_equity_exposure_usd, "
            "daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where enabled = 1"
        ).fetchall()
    for actor_row in actors:
        actor = str(actor_row["actor"])
        settings = _live_autonomy_row_to_dict(actor, actor_row)
        with connect() as conn:
            candidates = conn.execute(
                """
                select scorecard_id, payload_json, source from scorecards
                where actor = ? and consumed_by_intent_id is NULL
                  and expires_at > ?
                  and action in ('buy','sell')
                order by created_at asc limit ?
                """,
                (actor, current.isoformat(), LIVE_AUTO_BATCH_LIMIT),
            ).fetchall()
        for row in candidates:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(payload, dict):
                skipped += 1
                continue
            eligible, _reason = _eligible_for_live_auto(actor, payload, settings)
            if not eligible:
                skipped += 1
                continue
            today = current.strftime("%Y-%m-%d")
            with connect() as conn:
                spend_row = conn.execute(
                    "select spent_usdt from live_autonomy_spend where actor = ? and date = ?",
                    (actor, today),
                ).fetchone()
            spent = Decimal(str(spend_row["spent_usdt"])) if spend_row else Decimal("0")
            budget = Decimal(str(settings["daily_live_budget_usdt"]))
            per_trade = Decimal(str(settings["per_live_trade_max_usdt"]))
            spend_amount = min(per_trade, budget - spent)
            if spend_amount < Decimal("10"):
                skipped += 1
                continue
            if _fire_live_autonomous_trade(actor, str(row["scorecard_id"]), spend_amount):
                placed += 1
                _record_live_autonomy_spend(actor, today, spend_amount)
                break
            skipped += 1
    return {"placed": placed, "skipped": skipped}


def _record_autonomy_spend(actor: str, date: str, amount: Decimal) -> None:
    now_iso = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from autonomy_spend where actor = ? and date = ?",
            (actor, date),
        ).fetchone()
        if row is None:
            spent = _q8(amount)
            count = 1
        else:
            spent = _q8(Decimal(str(row["spent_usdt"])) + amount)
            count = int(row["trade_count"]) + 1
        conn.execute(
            "insert into autonomy_spend "
            "(actor, date, spent_usdt, trade_count, last_updated) "
            "values (?, ?, ?, ?, ?) "
            "on conflict(actor, date) do update set "
            "spent_usdt = excluded.spent_usdt, "
            "trade_count = excluded.trade_count, "
            "last_updated = excluded.last_updated",
            (actor, date, _q8s(spent), count, now_iso),
        )
        conn.commit()


def _fire_autonomous_trade(actor: str, scorecard_id: str, usdt_budget: Decimal) -> bool:
    payload = {
        "scorecard_id": scorecard_id,
        "actor": actor,
        "idempotency_key": f"auto-{actor}-{scorecard_id[:8]}-{int(time.time())}",
        "usdt_budget": str(usdt_budget),
        "position_fraction": "1.0",
        "mode": "paper",
    }
    try:
        response = httpx.post(
            f"{ORCHESTRATOR_SELF_URL}/intents/from_scorecard",
            json=payload,
            headers={"x-live-unlock": ""},
            timeout=10.0,
        )
        return getattr(response, "status_code", 200) in {200, 202}
    except httpx.HTTPError:
        return False


def auto_trade_tick(now: datetime | None = None) -> dict[str, object]:
    if not AUTO_TRADE_ENABLED:
        return {"placed": 0, "skipped_budget": 0, "skipped_other": 0, "reason": "disabled"}
    current = now or _now()
    evaluate_all_paper_bootstrap_guardrails(now=current)
    today = current.strftime("%Y-%m-%d")
    placed = 0
    skipped_budget = 0
    skipped_other = 0
    with connect() as conn:
        actors = conn.execute(
            "select actor, daily_budget_usdt, min_conviction, per_trade_usdt, allowed_sources "
            "from autonomy_settings where enabled = 1"
        ).fetchall()
    for actor_row in actors:
        try:
            actor = str(actor_row["actor"])
            budget = Decimal(str(actor_row["daily_budget_usdt"]))
            min_conv = Decimal(str(actor_row["min_conviction"]))
            per_trade = Decimal(str(actor_row["per_trade_usdt"]))
        except (InvalidOperation, ValueError):
            skipped_other += 1
            continue
        if bool(_paper_bootstrap_halt_status(actor)["halted"]):
            skipped_other += 1
            continue
        allowed = {
            item.strip() for item in str(actor_row["allowed_sources"]).split(",") if item.strip()
        }
        with connect() as conn:
            spend_row = conn.execute(
                "select spent_usdt from autonomy_spend where actor = ? and date = ?",
                (actor, today),
            ).fetchone()
        spent = Decimal(str(spend_row["spent_usdt"])) if spend_row else Decimal("0")
        if spent >= budget:
            continue
        with connect() as conn:
            candidates = conn.execute(
                """
                select scorecard_id, payload_json, source from scorecards
                where actor = ? and consumed_by_intent_id is NULL
                  and expires_at > ?
                  and action in ('buy','sell')
                  and cast(json_extract(payload_json, '$.conviction') as real) >= ?
                order by created_at asc limit ?
                """,
                (actor, current.isoformat(), float(min_conv), AUTO_TRADE_BATCH_LIMIT),
            ).fetchall()
        for row in candidates:
            if str(row["source"]) not in allowed:
                skipped_other += 1
                continue
            remaining = budget - spent
            spend = min(per_trade, remaining)
            if spend < Decimal("1"):
                skipped_budget += 1
                continue
            if _fire_autonomous_trade(actor, str(row["scorecard_id"]), spend):
                spent += spend
                placed += 1
                _record_autonomy_spend(actor, today, spend)
            else:
                skipped_other += 1
            if spent >= budget:
                break
    return {"placed": placed, "skipped_budget": skipped_budget, "skipped_other": skipped_other}


def recompute_calibration() -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            """
            select o.source, s.payload_json, o.closed_return_pct
            from scorecard_outcomes o
            join scorecards s on s.scorecard_id = o.scorecard_id
            where o.status = 'closed'
            """
        ).fetchall()
    buckets: dict[tuple[str, str, str], list[tuple[Decimal, Decimal]]] = {}
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
            metadata = payload.get("metadata") or {}
            heuristic_raw = metadata.get("heuristic_conviction") or payload.get("conviction")
            asset_type = str(metadata.get("asset_type") or "crypto")
            heuristic = Decimal(str(heuristic_raw))
            alpha = Decimal(str(row["closed_return_pct"]))
        except (json.JSONDecodeError, InvalidOperation, KeyError, TypeError):
            continue
        bucket = _bucket_label(heuristic)
        if bucket is None:
            continue
        buckets.setdefault((str(row["source"]), asset_type, bucket), []).append((heuristic, alpha))
    now_iso = _now().isoformat()
    written = 0
    with connect() as conn:
        conn.execute("delete from conviction_calibration")
        for (source, asset_type, bucket), samples in buckets.items():
            sample_count = len(samples)
            hit_count = sum(1 for _, alpha in samples if alpha > 0)
            avg_alpha = sum((alpha for _, alpha in samples), Decimal("0")) / Decimal(sample_count)
            empirical = Decimal(hit_count) / Decimal(sample_count)
            k = CALIBRATION_SHRINKAGE_K
            calibrated = (Decimal(hit_count) + k * Decimal("0.5")) / (Decimal(sample_count) + k)
            conn.execute(
                """
                insert into conviction_calibration
                  (source, asset_type, heuristic_bucket, sample_count, hit_count,
                   avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at)
                values (?,?,?,?,?,?,?,?,?)
                """,
                (
                    source,
                    asset_type,
                    bucket,
                    sample_count,
                    hit_count,
                    _q8s(avg_alpha),
                    _q8s(empirical),
                    _q8s(calibrated),
                    now_iso,
                ),
            )
            written += 1
        conn.commit()
    return {
        "buckets_written": written,
        "rows_considered": len(rows),
        "data_origin_breakdown": _serializable_calibration_breakdown(
            _calibration_origin_breakdown()
        ),
    }


def get_calibration(source: str | None = None, asset_type: str | None = None) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if asset_type:
        clauses.append("asset_type = ?")
        params.append(asset_type)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            "select source, asset_type, heuristic_bucket, sample_count, hit_count, "
            "avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at "
            f"from conviction_calibration{where} "
            "order by source, asset_type, heuristic_bucket",
            params,
        ).fetchall()
    breakdown = _calibration_origin_breakdown(source=source, asset_type=asset_type)
    items: list[dict[str, object]] = []
    for row in rows:
        item = _calibration_row_to_dict(row)
        key = (str(row["source"]), str(row["asset_type"]), str(row["heuristic_bucket"]))
        item["data_origin_breakdown"] = breakdown.get(key, {})
        items.append(item)
    return {
        "shrinkage_k": str(CALIBRATION_SHRINKAGE_K),
        "min_samples": CALIBRATION_MIN_SAMPLES,
        "items": items,
    }


def record_memory_endpoint(req: RecordMemoryRequest) -> dict[str, object]:
    entry, created = record_memory(
        req.type,
        req.subject,
        req.content,
        req.source_ref,
        tags=req.tags,
        confidence=req.confidence,
        superseded_by=req.superseded_by,
        trigger=req.trigger,
        created_by=req.created_by,
    )
    return {
        "memory_id": entry.id,
        "created": created,
        "entry": entry.model_dump(),
        "policy": {
            "append_only": True,
            "source_ref_deduped": True,
            "not_used_for_trading_or_risk_decisions": True,
            "no_vector_store": True,
            "no_graph_store": True,
        },
    }


def recall_memory(
    subject: str | None = None,
    type_: Literal["decision", "outcome", "lesson", "convention", "observation"] | None = Query(
        default=None, alias="type"
    ),
    since: str | None = None,
    tags: str | None = None,
    q: str | None = None,
    include_superseded: bool = False,
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, object]:
    tag_set = _parse_tags(tags)
    items = [
        entry
        for entry in _load_memory_entries()
        if _memory_matches(
            entry,
            subject=subject,
            memory_type=type_,
            since=since,
            tags=tag_set,
            keyword=q,
            include_superseded=include_superseded,
        )
    ]
    items.sort(key=lambda entry: entry.created_at, reverse=True)
    limited = items[:limit]
    return {
        "items": [entry.model_dump() for entry in limited],
        "total": len(limited),
        "matched": len(items),
        "filters": {
            "subject": subject,
            "type": type_,
            "since": since,
            "tags": sorted(tag_set),
            "q": q,
            "include_superseded": include_superseded,
            "limit": limit,
        },
        "policy": {
            "readonly": True,
            "storage": "sqlite memory_entries plus read-time adapter over existing domain tables",
            "no_vector_store": True,
            "no_graph_store": True,
            "default_excludes_superseded": True,
            "not_used_for_trading_or_risk_decisions": True,
        },
    }


def memory_lineage(memory_id: str) -> JSONResponse | dict[str, object]:
    entry = _find_memory_entry(memory_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"code": "MEMORY_NOT_FOUND"})
    source = entry.source_ref
    chain: list[dict[str, object]] = [{"kind": "memory_entry", "data": entry.model_dump()}]
    table = source.get("table")
    with connect() as conn:
        if table == "scorecards":
            row = conn.execute(
                "select scorecard_id, actor, symbol, action, source, payload_json, "
                "created_at, expires_at, consumed_by_intent_id "
                "from scorecards where scorecard_id = ?",
                (source.get("scorecard_id"),),
            ).fetchone()
            if row is not None:
                chain.append({"kind": "scorecard", "data": _scorecard_memory_entry(row).source_ref})
                chain.append({"kind": "scorecard_payload", "data": _memory_scorecard_payload(row)})
        elif table == "scorecard_outcomes":
            outcome = conn.execute(
                "select outcome_id, scorecard_id, actor, symbol, source, action, "
                "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
                "opened_cost_basis, status, closed_at, closed_realized_pnl, "
                "closed_return_pct, notes, reflected_at, trailing_pct, peak_mark "
                "from scorecard_outcomes where outcome_id = ?",
                (source.get("outcome_id"),),
            ).fetchone()
            if outcome is not None:
                chain.append({"kind": "scorecard_outcome", "data": _outcome_row_to_dict(outcome)})
                scorecard = conn.execute(
                    "select scorecard_id, actor, symbol, action, source, payload_json, "
                    "created_at, expires_at, consumed_by_intent_id "
                    "from scorecards where scorecard_id = ?",
                    (outcome["scorecard_id"],),
                ).fetchone()
                if scorecard is not None:
                    chain.append(
                        {"kind": "scorecard_payload", "data": _memory_scorecard_payload(scorecard)}
                    )
        elif table == "ev_estimates":
            ev = conn.execute(
                "select scorecard_id, outcome_id, actor, symbol, mode, gate_result, reason, "
                "p, tp_pct, sl_pct, fee_bps, slippage_bps, funding_bps, min_ev, ev, "
                "created_at, updated_at from ev_estimates where scorecard_id = ?",
                (source.get("scorecard_id"),),
            ).fetchone()
            if ev is not None:
                chain.append({"kind": "ev_estimate", "data": dict(ev)})
                scorecard = conn.execute(
                    "select scorecard_id, actor, symbol, action, source, payload_json, "
                    "created_at, expires_at, consumed_by_intent_id "
                    "from scorecards where scorecard_id = ?",
                    (ev["scorecard_id"],),
                ).fetchone()
                if scorecard is not None:
                    chain.append(
                        {"kind": "scorecard_payload", "data": _memory_scorecard_payload(scorecard)}
                    )
        elif table == "intents":
            intent_row = conn.execute(
                "select payload_json, decision_json, execution_json, status "
                "from intents where intent_id = ?",
                (source.get("intent_id"),),
            ).fetchone()
            if intent_row is not None:
                chain.append({"kind": "intent", "data": _row_to_item(intent_row)})
        elif table == "conviction_calibration":
            calibration = conn.execute(
                "select source, asset_type, heuristic_bucket, sample_count, hit_count, "
                "avg_alpha_return, empirical_hit_rate, calibrated_conviction, updated_at "
                "from conviction_calibration "
                "where source = ? and asset_type = ? and heuristic_bucket = ?",
                (
                    source.get("source"),
                    source.get("asset_type"),
                    source.get("heuristic_bucket"),
                ),
            ).fetchone()
            if calibration is not None:
                chain.append(
                    {
                        "kind": "conviction_calibration",
                        "data": _calibration_row_to_dict(calibration),
                    }
                )
    return {"memory_id": memory_id, "lineage": chain}


def update_autonomy(req: AutonomyUpdateRequest) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        existing = conn.execute(
            "select enabled, daily_budget_usdt, min_conviction, per_trade_usdt, "
            "allowed_sources from autonomy_settings where actor = ?",
            (req.actor,),
        ).fetchone()
    current: dict[str, object] = {
        "enabled": int(existing["enabled"]) if existing else 0,
        "daily_budget_usdt": str(existing["daily_budget_usdt"]) if existing else "0",
        "min_conviction": str(existing["min_conviction"]) if existing else "0.65",
        "per_trade_usdt": str(existing["per_trade_usdt"]) if existing else "50",
        "allowed_sources": str(existing["allowed_sources"]) if existing else "tradingagents",
    }
    if req.enabled is not None:
        current["enabled"] = 1 if req.enabled else 0
    try:
        if req.daily_budget_usdt is not None:
            if Decimal(req.daily_budget_usdt) < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_BUDGET"})
            current["daily_budget_usdt"] = req.daily_budget_usdt
        if req.min_conviction is not None:
            min_conviction = Decimal(req.min_conviction)
            if not (Decimal("0") <= min_conviction <= Decimal("1")):
                return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CONVICTION"})
            current["min_conviction"] = req.min_conviction
        if req.per_trade_usdt is not None:
            if Decimal(req.per_trade_usdt) <= 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_PER_TRADE"})
            current["per_trade_usdt"] = req.per_trade_usdt
    except InvalidOperation:
        return JSONResponse(status_code=400, content={"code": "INVALID_DECIMAL"})
    if req.allowed_sources is not None:
        current["allowed_sources"] = req.allowed_sources
    with connect() as conn:
        conn.execute(
            """
            insert into autonomy_settings
              (actor, enabled, daily_budget_usdt, min_conviction, per_trade_usdt,
               allowed_sources, updated_at)
            values (?,?,?,?,?,?,?)
            on conflict(actor) do update set
              enabled = excluded.enabled,
              daily_budget_usdt = excluded.daily_budget_usdt,
              min_conviction = excluded.min_conviction,
              per_trade_usdt = excluded.per_trade_usdt,
              allowed_sources = excluded.allowed_sources,
              updated_at = excluded.updated_at
            """,
            (
                req.actor,
                current["enabled"],
                current["daily_budget_usdt"],
                current["min_conviction"],
                current["per_trade_usdt"],
                current["allowed_sources"],
                _now().isoformat(),
            ),
        )
        conn.commit()
    return {"actor": req.actor, **current, "enabled": bool(current["enabled"])}


def get_autonomy_settings(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_budget_usdt, min_conviction, per_trade_usdt, "
            "allowed_sources, updated_at from autonomy_settings where actor = ?",
            (actor,),
        ).fetchone()
    if row is None:
        return {
            "actor": actor,
            "enabled": False,
            "daily_budget_usdt": "0",
            "min_conviction": "0.65",
            "per_trade_usdt": "50",
            "allowed_sources": "tradingagents",
            "updated_at": None,
        }
    return {
        "actor": actor,
        "enabled": bool(row["enabled"]),
        "daily_budget_usdt": str(row["daily_budget_usdt"]),
        "min_conviction": str(row["min_conviction"]),
        "per_trade_usdt": str(row["per_trade_usdt"]),
        "allowed_sources": str(row["allowed_sources"]),
        "updated_at": row["updated_at"],
    }


def get_autonomy_today(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    today = _today()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    return {
        "actor": actor,
        "date": today,
        "spent_usdt": str(row["spent_usdt"]) if row else "0",
        "trade_count": int(row["trade_count"]) if row else 0,
    }


def disable_live_autonomy(
    x_ops_token: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN or x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    global LIVE_AUTONOMY_GLOBAL_ENABLED
    LIVE_AUTONOMY_GLOBAL_ENABLED = False
    killed_at = _now().isoformat()
    with connect() as conn:
        conn.execute(
            "update live_autonomy_kill set killed = 1, killed_at = ?, killed_by = ? where id = 1",
            (killed_at, "ops_token"),
        )
        conn.commit()
    return {"killed": True, "killed_at": killed_at}


def reenable_live_autonomy(
    x_ops_token: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN or x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    with connect() as conn:
        conn.execute(
            "update live_autonomy_kill set killed = 0, killed_at = ?, "
            "killed_by = NULL where id = 1",
            (_now().isoformat(),),
        )
        conn.commit()
    return {"killed": False, "note": "kill flag cleared; env var still gates"}


def update_live_autonomy(req: LiveAutonomyUpdateRequest) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, max_us_equity_exposure_usd, "
            "daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where actor = ?",
            (req.actor,),
        ).fetchone()
    current = _live_autonomy_row_to_dict(req.actor, row)
    now_iso = _now().isoformat()
    if current["created_at"] is None:
        current["created_at"] = now_iso
    if req.enabled is not None:
        current["enabled"] = req.enabled
    try:
        if req.daily_live_budget_usdt is not None:
            if Decimal(req.daily_live_budget_usdt) < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_BUDGET"})
            current["daily_live_budget_usdt"] = req.daily_live_budget_usdt
        if req.per_live_trade_max_usdt is not None:
            per_trade = Decimal(req.per_live_trade_max_usdt)
            if per_trade <= 0 or per_trade > Decimal("500"):
                return JSONResponse(status_code=400, content={"code": "INVALID_PER_TRADE"})
            current["per_live_trade_max_usdt"] = req.per_live_trade_max_usdt
        if req.max_live_exposure_usdt is not None:
            max_exposure = Decimal(req.max_live_exposure_usdt)
            if max_exposure < 0:
                return JSONResponse(status_code=400, content={"code": "INVALID_MAX_EXPOSURE"})
            if max_exposure > Decimal("100000"):
                return JSONResponse(status_code=400, content={"code": "MAX_EXPOSURE_TOO_HIGH"})
            current["max_live_exposure_usdt"] = req.max_live_exposure_usdt
        if req.max_us_equity_exposure_usd is not None:
            max_stock_exposure = Decimal(req.max_us_equity_exposure_usd)
            if max_stock_exposure < 0:
                return JSONResponse(
                    status_code=400, content={"code": "INVALID_MAX_US_EQUITY_EXPOSURE"}
                )
            if max_stock_exposure > Decimal("100000"):
                return JSONResponse(
                    status_code=400, content={"code": "MAX_US_EQUITY_EXPOSURE_TOO_HIGH"}
                )
            current["max_us_equity_exposure_usd"] = req.max_us_equity_exposure_usd
        if req.min_calibrated_conviction is not None:
            min_conv = Decimal(req.min_calibrated_conviction)
            if not (Decimal("0.5") <= min_conv <= Decimal("1.0")):
                return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CONVICTION"})
            current["min_calibrated_conviction"] = req.min_calibrated_conviction
    except (InvalidOperation, ValueError):
        return JSONResponse(status_code=400, content={"code": "INVALID_DECIMAL"})
    if req.daily_live_trade_count_max is not None:
        if req.daily_live_trade_count_max < 1 or req.daily_live_trade_count_max > 10:
            return JSONResponse(status_code=400, content={"code": "INVALID_TRADE_COUNT"})
        current["daily_live_trade_count_max"] = req.daily_live_trade_count_max
    if req.min_closed_outcomes is not None:
        if req.min_closed_outcomes < 5:
            return JSONResponse(status_code=400, content={"code": "INVALID_MIN_CLOSED_OUTCOMES"})
        current["min_closed_outcomes"] = req.min_closed_outcomes
    if req.allowed_sources is not None:
        current["allowed_sources"] = req.allowed_sources
    current["updated_at"] = now_iso
    with connect() as conn:
        conn.execute(
            """
            insert into live_autonomy_settings
              (actor, enabled, daily_live_budget_usdt, per_live_trade_max_usdt,
               max_live_exposure_usdt, max_us_equity_exposure_usd,
               daily_live_trade_count_max, min_calibrated_conviction,
               min_closed_outcomes, allowed_sources, created_at, updated_at)
            values (?,?,?,?,?,?,?,?,?,?,?,?)
            on conflict(actor) do update set
              enabled = excluded.enabled,
              daily_live_budget_usdt = excluded.daily_live_budget_usdt,
              per_live_trade_max_usdt = excluded.per_live_trade_max_usdt,
              max_live_exposure_usdt = excluded.max_live_exposure_usdt,
              max_us_equity_exposure_usd = excluded.max_us_equity_exposure_usd,
              daily_live_trade_count_max = excluded.daily_live_trade_count_max,
              min_calibrated_conviction = excluded.min_calibrated_conviction,
              min_closed_outcomes = excluded.min_closed_outcomes,
              allowed_sources = excluded.allowed_sources,
              updated_at = excluded.updated_at
            """,
            (
                req.actor,
                1 if current["enabled"] else 0,
                current["daily_live_budget_usdt"],
                current["per_live_trade_max_usdt"],
                current["max_live_exposure_usdt"],
                current["max_us_equity_exposure_usd"],
                current["daily_live_trade_count_max"],
                current["min_calibrated_conviction"],
                current["min_closed_outcomes"],
                current["allowed_sources"],
                current["created_at"],
                current["updated_at"],
            ),
        )
        conn.commit()
    return current


def get_live_autonomy_settings(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        row = conn.execute(
            "select enabled, daily_live_budget_usdt, per_live_trade_max_usdt, "
            "max_live_exposure_usdt, max_us_equity_exposure_usd, "
            "daily_live_trade_count_max, min_calibrated_conviction, "
            "min_closed_outcomes, allowed_sources, created_at, updated_at "
            "from live_autonomy_settings where actor = ?",
            (actor,),
        ).fetchone()
    return _live_autonomy_row_to_dict(actor, row)


def get_live_autonomy_today(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    today = _today()
    with connect() as conn:
        row = conn.execute(
            "select spent_usdt, trade_count from live_autonomy_spend where actor = ? and date = ?",
            (actor, today),
        ).fetchone()
    return {
        "actor": actor,
        "date": today,
        "spent_usdt": str(row["spent_usdt"]) if row else "0",
        "trade_count": int(row["trade_count"]) if row else 0,
    }


def get_paper_autonomy_observability(
    actor: str | None = None,
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    return build_paper_autonomy_observability(actor)


def post_paper_autonomy_alerts(
    actor: str | None = None, notify: bool = Query(default=False)
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    overview = build_paper_autonomy_observability(actor)
    result: dict[str, object] = {
        "actor": actor,
        "alerts": overview["alerts"],
        "notified": False,
        "sent": 0,
    }
    if notify:
        delivered = notify_paper_autonomy_alerts(actor)
        result["notified"] = True
        result["sent"] = delivered["sent"]
    return result


def get_paper_bootstrap_status(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    overview = build_paper_autonomy_observability(actor)
    return {
        "actor": actor,
        "bootstrap": overview["bootstrap"],
        "feedback_loop_progress": overview["feedback_loop_progress"],
        "paper": {
            "today_decision_count": overview["today_decision_count"],
            "today_realized_pnl": overview["today_realized_pnl"],
            "current_unrealized_pnl": overview["current_unrealized_pnl"],
            "current_total_pnl": overview["current_total_pnl"],
            "cumulative_realized_pnl": overview["cumulative_realized_pnl"],
            "max_drawdown_today_usdt": overview["max_drawdown_today_usdt"],
            "consecutive_closed_losses": overview["consecutive_closed_losses"],
            "alerts": overview["alerts"],
        },
    }


def post_paper_bootstrap_evaluate_guardrails(
    actor: str | None = None,
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    return evaluate_paper_bootstrap_guardrails(actor)


def post_paper_bootstrap_resume(
    actor: str | None = None, resumed_by: str | None = None
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    by = resumed_by or "manual"
    return _resume_paper_feedback_bootstrap(actor, resumed_by=by)


def add_watchlist(req: WatchlistAddRequest) -> dict[str, object]:
    now = _now()
    next_run = now + timedelta(minutes=req.cadence_minutes)
    symbol = req.symbol.upper()
    with connect() as conn:
        conn.execute(
            "insert into watchlist_entries "
            "(actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, source_origin, gate_conviction, created_at) "
            "values (?,?,?,?,NULL,?,?,?,?,?) "
            "on conflict(actor, symbol) do update set "
            "asset_type = excluded.asset_type, cadence_minutes = excluded.cadence_minutes, "
            "next_run_at = excluded.next_run_at, enabled = 1, "
            "source_origin = excluded.source_origin, "
            "gate_conviction = excluded.gate_conviction",
            (
                req.actor,
                symbol,
                req.asset_type,
                req.cadence_minutes,
                next_run.isoformat(),
                1,
                STANDARD_DATA_ORIGIN,
                None,
                now.isoformat(),
            ),
        )
        row = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, source_origin, gate_conviction, created_at "
            "from watchlist_entries where actor = ? and symbol = ?",
            (req.actor, symbol),
        ).fetchone()
        conn.commit()
    return {"item": _watchlist_row_to_dict(row)}


def list_watchlist(actor: str = Query(..., min_length=1)) -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select actor, symbol, asset_type, cadence_minutes, last_run_at, "
            "next_run_at, enabled, source_origin, gate_conviction, created_at "
            "from watchlist_entries where actor = ? and enabled = 1 order by symbol",
            (actor,),
        ).fetchall()
    return {"items": [_watchlist_row_to_dict(row) for row in rows]}


def delete_watchlist(symbol: str, actor: str = Query(..., min_length=1)) -> dict[str, bool]:
    with connect() as conn:
        cursor = conn.execute(
            "update watchlist_entries set enabled = 0 where actor = ? and symbol = ?",
            (actor, symbol.upper()),
        )
        conn.commit()
    return {"deleted": cursor.rowcount > 0}


def issue_live_unlock(
    req: LiveUnlockRequest, x_ops_token: str = Header(default="")
) -> JSONResponse | dict[str, object]:
    if not OPS_TOKEN:
        return JSONResponse(
            status_code=503,
            content={"code": "LIVE_UNLOCK_DISABLED", "detail": "OPS_TOKEN not configured"},
        )
    if x_ops_token != OPS_TOKEN:
        return JSONResponse(status_code=403, content={"code": "INVALID_OPS_TOKEN"})
    token = _mint_user_live_unlock_token(req.actor)
    expires = _now() + timedelta(minutes=LIVE_UNLOCK_TTL_MIN)
    return {"token": token, "actor": req.actor, "expires_at": expires}


def create_scorecard(req: ScorecardCreateRequest) -> JSONResponse | dict[str, object]:
    ttl_min = req.ttl_minutes if req.ttl_minutes is not None else SCORECARD_DEFAULT_TTL_MIN
    if ttl_min <= 0 or ttl_min > 1440:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_TTL", "detail": "ttl_minutes must be 1..1440"},
        )
    now = _now()
    try:
        scorecard = Scorecard(
            scorecard_id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_min),
            source=req.source,
            actor=req.actor,
            symbol=req.symbol,
            action=req.action,
            conviction=Decimal(req.conviction),
            thesis=req.thesis,
            entry_low=Decimal(req.entry_low) if req.entry_low else None,
            entry_high=Decimal(req.entry_high) if req.entry_high else None,
            stop_loss=Decimal(req.stop_loss) if req.stop_loss else None,
            take_profit=Decimal(req.take_profit) if req.take_profit else None,
            time_horizon=req.time_horizon,
            metadata=req.metadata,
            factors=req.factors,
        )
    except (InvalidOperation, ValueError) as exc:
        return JSONResponse(
            status_code=400, content={"code": "INVALID_SCORECARD", "detail": str(exc)}
        )
    with connect() as conn:
        conn.execute(
            "insert into scorecards "
            "(scorecard_id, actor, symbol, action, source, payload_json, "
            "created_at, expires_at, consumed_by_intent_id) "
            "values (?,?,?,?,?,?,?,?,NULL)",
            (
                str(scorecard.scorecard_id),
                scorecard.actor,
                scorecard.symbol,
                scorecard.action,
                scorecard.source,
                scorecard.model_dump_json(),
                scorecard.created_at.isoformat(),
                scorecard.expires_at.isoformat(),
            ),
        )
        conn.commit()
    return scorecard.model_dump()


def get_scorecard(scorecard_id: UUID) -> JSONResponse | Scorecard:
    with connect() as conn:
        row = conn.execute(
            "select payload_json from scorecards where scorecard_id = ?",
            (str(scorecard_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "SCORECARD_NOT_FOUND"})
    return Scorecard.model_validate_json(row["payload_json"])


def list_scorecards(
    actor: str | None = None,
    symbol: str | None = None,
    active_only: bool = False,
    limit: int = Query(default=50, ge=1),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if active_only:
        clauses.append("expires_at > ?")
        clauses.append("consumed_by_intent_id is NULL")
        params.append(_now().isoformat())
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"select payload_json, consumed_by_intent_id from scorecards{where} "
            "order by created_at desc limit ?",
            [*params, min(limit, 200)],
        ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        sc = Scorecard.model_validate_json(row["payload_json"])
        items.append(
            {
                "scorecard": sc,
                "consumed_by_intent_id": row["consumed_by_intent_id"],
                "is_expired": sc.expires_at < _now(),
            }
        )
    return {"items": items, "total": len(items)}


def list_outcomes(
    actor: str | None = None,
    symbol: str | None = None,
    source: str | None = None,
    status: Literal["open", "closed"] | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    for col, val in (
        ("actor", actor),
        ("symbol", symbol),
        ("source", source),
        ("status", status),
    ):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"""
            select outcome_id, scorecard_id, actor, symbol, source, action,
                   opened_intent_id, opened_at, opened_qty, opened_avg_cost,
                   opened_cost_basis, status, closed_at, closed_realized_pnl,
                   closed_return_pct, notes, reflected_at, trailing_pct, peak_mark
            from scorecard_outcomes{where}
            order by opened_at desc
            limit ?
            """,
            [*params, limit],
        ).fetchall()
    return {
        "items": [_outcome_row_to_dict(row) for row in rows],
        "attribution_rule": (
            "When the aggregate position closes with multiple open outcomes, "
            "realized PnL is split proportionally by opened_cost_basis. "
            "Rows with notes='split-attribution' indicate this case."
        ),
    }


def outcomes_summary(actor: str | None = None, since: str | None = None) -> dict[str, object]:
    clauses_closed: list[str] = ["status = 'closed'"]
    params_closed: list[object] = []
    if actor:
        clauses_closed.append("actor = ?")
        params_closed.append(actor)
    if since:
        clauses_closed.append("closed_at >= ?")
        params_closed.append(since)
    where_closed = " where " + " and ".join(clauses_closed)
    with connect() as conn:
        closed_rows = conn.execute(
            f"select source, closed_realized_pnl from scorecard_outcomes{where_closed}",
            params_closed,
        ).fetchall()
        open_clauses = ["status = 'open'"]
        open_params: list[object] = []
        if actor:
            open_clauses.append("actor = ?")
            open_params.append(actor)
        open_rows = conn.execute(
            f"select source, count(*) as n from scorecard_outcomes "
            f"where {' and '.join(open_clauses)} group by source",
            open_params,
        ).fetchall()
        pending_reflection_clauses = ["status = 'closed'", "reflected_at is null"]
        pending_reflection_params: list[object] = []
        if actor:
            pending_reflection_clauses.append("actor = ?")
            pending_reflection_params.append(actor)
        if since:
            pending_reflection_clauses.append("closed_at >= ?")
            pending_reflection_params.append(since)
        pending_reflection_rows = conn.execute(
            f"select source, count(*) as n from scorecard_outcomes "
            f"where {' and '.join(pending_reflection_clauses)} group by source",
            pending_reflection_params,
        ).fetchall()

    by_source: dict[str, _OutcomeSummaryBucket] = {}
    for row in closed_rows:
        src = str(row["source"])
        pnl = Decimal(str(row["closed_realized_pnl"]))
        bucket = by_source.setdefault(src, _OutcomeSummaryBucket())
        bucket.closed_count += 1
        bucket.total_pnl += pnl
        if pnl > 0:
            bucket.hits += 1
        elif pnl < 0:
            bucket.losses += 1

    pending_reflections: dict[str, int] = {}
    for row in pending_reflection_rows:
        pending_reflections[str(row["source"])] = int(str(row["n"]))
        by_source.setdefault(str(row["source"]), _OutcomeSummaryBucket())

    for row in open_rows:
        bucket = by_source.setdefault(str(row["source"]), _OutcomeSummaryBucket())
        bucket.open_count = int(str(row["n"]))

    summary: dict[str, object] = {}
    for src, bucket in by_source.items():
        summary[src] = {
            "closed_count": bucket.closed_count,
            "open_count": bucket.open_count,
            "hits": bucket.hits,
            "losses": bucket.losses,
            "hit_rate": (
                f"{bucket.hits / bucket.closed_count:.4f}" if bucket.closed_count else "0.0000"
            ),
            "realized_pnl": _q8s(bucket.total_pnl),
            "total_pnl": _q8s(bucket.total_pnl),
            "pending_reflection_count": pending_reflections.get(src, 0),
        }
    return {"actor": actor, "since": since, "by_source": summary}


def factor_attribution(actor: str | None = None) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            "select actor, factor, direction, support_count, win_count, "
            "total_pnl, loss_contribution, updated_at "
            f"from factor_attribution{where} "
            "order by cast(loss_contribution as real) desc, factor, direction",
            params,
        ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        support_count = int(row["support_count"])
        win_count = int(row["win_count"])
        total_pnl = Decimal(str(row["total_pnl"]))
        avg_pnl = _q8(total_pnl / Decimal(support_count)) if support_count else Decimal("0")
        items.append(
            {
                "actor": row["actor"],
                "factor": row["factor"],
                "direction": row["direction"],
                "support_count": support_count,
                "win_count": win_count,
                "win_rate": f"{win_count / support_count:.4f}" if support_count else "0.0000",
                "avg_pnl": _q8s(avg_pnl),
                "total_pnl": _q8s(total_pnl),
                "loss_contribution": row["loss_contribution"],
                "updated_at": row["updated_at"],
            }
        )
    return {"actor": actor, "items": items}


def reflect_pending(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, object]:
    with connect() as conn:
        rows = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at, trailing_pct, peak_mark "
            "from scorecard_outcomes "
            "where status = 'closed' and reflected_at is null "
            "order by closed_at asc limit ?",
            (limit,),
        ).fetchall()
    attempted = 0
    reflected = 0
    failed = 0
    for row in rows:
        attempted += 1
        outcome = _outcome_row_to_dict(row)
        if _push_outcome_reflection(outcome):
            _mark_outcome_reflected(str(outcome["outcome_id"]))
            try:
                _record_reflection_memory(outcome)
            except Exception:
                logger.exception("failed to record reflection memory")
            reflected += 1
        else:
            failed += 1
    return {"attempted": attempted, "reflected": reflected, "failed": failed}


def get_outcome(outcome_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select outcome_id, scorecard_id, actor, symbol, source, action, "
            "opened_intent_id, opened_at, opened_qty, opened_avg_cost, "
            "opened_cost_basis, status, closed_at, closed_realized_pnl, "
            "closed_return_pct, notes, reflected_at, trailing_pct, peak_mark "
            "from scorecard_outcomes where outcome_id = ?",
            (str(outcome_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "OUTCOME_NOT_FOUND"})
    return _outcome_row_to_dict(row)


def update_outcome_trailing(
    outcome_id: UUID, req: TrailingStopUpdateRequest
) -> JSONResponse | dict[str, object]:
    try:
        trailing_pct = Decimal(req.trailing_pct)
    except (InvalidOperation, ValueError):
        return JSONResponse(status_code=400, content={"code": "INVALID_TRAILING_PCT"})
    if trailing_pct < 0 or trailing_pct >= 1:
        return JSONResponse(status_code=400, content={"code": "TRAILING_PCT_OUT_OF_RANGE"})
    with connect() as conn:
        cursor = conn.execute(
            "update scorecard_outcomes set trailing_pct = ? "
            "where outcome_id = ? and status = 'open'",
            (str(trailing_pct), str(outcome_id)),
        )
        conn.commit()
    if cursor.rowcount == 0:
        return JSONResponse(status_code=404, content={"code": "OUTCOME_NOT_FOUND_OR_NOT_OPEN"})
    return {"outcome_id": str(outcome_id), "trailing_pct": str(trailing_pct)}


def _outcome_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "outcome_id": row["outcome_id"],
        "scorecard_id": row["scorecard_id"],
        "actor": row["actor"],
        "symbol": row["symbol"],
        "source": row["source"],
        "action": row["action"],
        "opened_intent_id": row["opened_intent_id"],
        "opened_at": row["opened_at"],
        "opened_qty": row["opened_qty"],
        "opened_avg_cost": row["opened_avg_cost"],
        "opened_cost_basis": row["opened_cost_basis"],
        "status": row["status"],
        "closed_at": row["closed_at"],
        "closed_realized_pnl": row["closed_realized_pnl"],
        "closed_return_pct": row["closed_return_pct"],
        "notes": row["notes"],
        "reflected_at": row["reflected_at"],
        "trailing_pct": row["trailing_pct"],
        "peak_mark": row["peak_mark"],
    }


def get_pnl_today(
    actor: str | None = None, venue: str | None = None
) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    if venue is not None and venue not in {"binance_spot", "ibkr_us_equity"}:
        return JSONResponse(status_code=400, content={"code": "INVALID_VENUE"})
    today = _today()
    with connect() as conn:
        if venue is None:
            rows = conn.execute(
                "select symbol, realized_delta from daily_pnl where actor = ? and date = ?",
                (actor, today),
            ).fetchall()
            position_rows = conn.execute(
                "select symbol, qty, avg_cost from paper_positions where actor = ?",
                (actor,),
            ).fetchall()
        else:
            rows = conn.execute(
                "select symbol, realized_delta from daily_pnl "
                "where actor = ? and date = ? and venue = ?",
                (actor, today, venue),
            ).fetchall()
            position_rows = conn.execute(
                "select symbol, qty, avg_cost from paper_positions where actor = ? and venue = ?",
                (actor, venue),
            ).fetchall()

    realized = Decimal("0")
    by_symbol_realized: dict[str, Decimal] = {}
    for row in rows:
        delta = Decimal(row["realized_delta"])
        realized += delta
        by_symbol_realized[row["symbol"]] = (
            by_symbol_realized.get(row["symbol"], Decimal("0")) + delta
        )

    unrealized = Decimal("0")
    by_symbol_unrealized: dict[str, Decimal] = {}
    for row in position_rows:
        qty = Decimal(row["qty"])
        if qty <= 0:
            continue
        mark, _ = _mark_for_symbol(row["symbol"])
        if mark is None:
            continue
        delta = _q8(qty * (mark - Decimal(row["avg_cost"])))
        unrealized += delta
        by_symbol_unrealized[row["symbol"]] = delta

    return {
        "actor": actor,
        "date": today,
        "realized_pnl": _q8s(realized),
        "unrealized_pnl": _q8s(unrealized),
        "total_pnl": _q8s(realized + unrealized),
        "by_symbol": {
            "realized": {k: _q8s(v) for k, v in by_symbol_realized.items()},
            "unrealized": {k: _q8s(v) for k, v in by_symbol_unrealized.items()},
        },
    }


def create_intent_from_nl(
    nl: NLIntentRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if not CLAUDE_API_KEY:
        return JSONResponse(
            status_code=503,
            content={
                "code": "HERMES_UNAVAILABLE",
                "detail": "CLAUDE_API_KEY not configured",
            },
        )

    try:
        fields = _parse_nl_to_intent_fields(nl.message)
    except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            status_code=503,
            content={"code": "HERMES_UNAVAILABLE", "detail": str(exc)[:300]},
        )

    if "error" in fields:
        return JSONResponse(
            status_code=400,
            content={"code": "HERMES_PARSE_ERROR", "detail": str(fields["error"])},
        )

    try:
        intent = _build_intent_from_nl(nl, fields)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"code": "HERMES_PARSE_ERROR", "detail": str(exc)[:300]},
        )

    return create_intent(intent, x_live_unlock=x_live_unlock)


def create_intent_from_scorecard(
    req: ScorecardIntentRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, consumed_by_intent_id, expires_at from scorecards "
            "where scorecard_id = ?",
            (str(req.scorecard_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "SCORECARD_NOT_FOUND"})
    if row["consumed_by_intent_id"] is not None:
        return JSONResponse(
            status_code=409,
            content={
                "code": "SCORECARD_ALREADY_CONSUMED",
                "intent_id": row["consumed_by_intent_id"],
            },
        )

    if row["expires_at"] < _now().isoformat():
        return JSONResponse(status_code=410, content={"code": "SCORECARD_EXPIRED"})
    scorecard = Scorecard.model_validate_json(row["payload_json"])
    if scorecard.expires_at < _now():
        return JSONResponse(status_code=410, content={"code": "SCORECARD_EXPIRED"})
    if scorecard.action == "hold":
        return JSONResponse(
            status_code=400,
            content={
                "code": "SCORECARD_ACTION_HOLD",
                "detail": "hold scorecards are informational only",
            },
        )
    if scorecard.actor != req.actor:
        return JSONResponse(status_code=403, content={"code": "SCORECARD_ACTOR_MISMATCH"})

    ev_allowed, ev_reason = _scorecard_trade_ev_gate(req.actor, scorecard)
    if not ev_allowed:
        return JSONResponse(
            status_code=409,
            content={"code": ev_reason, "detail": "scorecard EV gate blocked this intent"},
        )

    try:
        budget = Decimal(req.usdt_budget)
        fraction = Decimal(req.position_fraction)
    except (InvalidOperation, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_AMOUNT",
                "detail": "usdt_budget / position_fraction must be decimal strings",
            },
        )
    if budget <= 0 or fraction <= 0 or fraction > 1:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_AMOUNT",
                "detail": "0 < position_fraction <= 1 and budget > 0",
            },
        )

    notional = (budget * scorecard.conviction * fraction).quantize(Decimal("0.01"))
    if notional <= 0:
        return JSONResponse(
            status_code=400,
            content={
                "code": "ZERO_SIZED_INTENT",
                "detail": "conviction * budget * fraction rounded to zero",
            },
        )

    limit_price: Decimal | None = None
    if req.order_type == "limit":
        if scorecard.action == "buy":
            limit_price = scorecard.entry_low or scorecard.entry_high
        else:
            limit_price = scorecard.entry_high or scorecard.entry_low
        if limit_price is None:
            return JSONResponse(status_code=400, content={"code": "SCORECARD_MISSING_ENTRY_PRICE"})

    side: Literal["buy", "sell"] = scorecard.action
    intent_id = req.intent_id or uuid4()
    metadata = scorecard.metadata or {}
    asset_type = metadata.get("asset_type", "crypto")
    venue: Literal["binance_spot", "ibkr_us_equity"] = (
        "ibkr_us_equity" if asset_type == "stock" else "binance_spot"
    )
    intent = OrderIntent(
        intent_id=intent_id,
        request_id=req.request_id or uuid4(),
        idempotency_key=req.idempotency_key,
        actor=req.actor,
        created_at=_now(),
        mode=req.mode,
        venue=venue,
        symbol=scorecard.symbol,
        side=side,
        order_type=req.order_type,
        quantity=Quantity(kind="quote", value=notional),
        limit_price=limit_price,
        time_in_force="GTC",
        reduce_only=False,
        leverage=None,
        stop_loss=scorecard.stop_loss,
        take_profit=scorecard.take_profit,
        source=Source(
            origin="scorecard",
            scorecard_id=str(scorecard.scorecard_id),
            hermes_message_id=None,
        ),
        client_confirmation_required=False,
    )

    response = create_intent(intent, x_live_unlock=x_live_unlock)
    if _scorecard_should_mark_consumed(response):
        with connect() as conn:
            cursor = conn.execute(
                "update scorecards set consumed_by_intent_id = ? "
                "where scorecard_id = ? and consumed_by_intent_id is NULL",
                (str(intent.intent_id), str(scorecard.scorecard_id)),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse(
                status_code=409,
                content={
                    "code": "SCORECARD_RACED",
                    "detail": "scorecard was consumed by a concurrent request",
                    "your_intent_id": str(intent.intent_id),
                },
            )
    return response


def create_intent(
    intent: OrderIntent, x_live_unlock: str = Header(default="")
) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        existing = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where idempotency_key = ?",
            (intent.idempotency_key,),
        ).fetchone()
    if existing is not None:
        return _idempotent_response(existing)

    if intent.mode == "live":
        unlock_error = _consume_live_unlock_or_error(
            str(x_live_unlock), intent.actor, dry=True, intent_id=intent.intent_id
        )
        if unlock_error is not None:
            return unlock_error

    base_qty: str | None = None
    price_str: str | None = None
    if intent.quantity.kind == "quote":
        requested_notional = intent.quantity.value
    else:
        base_qty, price_str = _resolve_qty(intent, _market_url())
        requested_notional = intent.quantity.value * Decimal(price_str)

    exposure_response = _exposure_limit_response(
        intent, _current_exposure(intent.symbol, _today()), requested_notional
    )
    if exposure_response is not None:
        return exposure_response

    risk_url = os.getenv("RISK_ENGINE_URL", "http://risk-engine:8081")
    risk_response = httpx.post(
        f"{risk_url}/validate",
        content=intent.model_dump_json(),
        headers={"content-type": "application/json"},
        timeout=5.0,
    )
    risk_response.raise_for_status()
    decision = RiskDecision.model_validate(risk_response.json())

    try:
        if not decision.approved:
            _persist(intent, decision, None, "rejected")
            try:
                _record_risk_reject_memory(intent, decision)
            except Exception:
                logger.exception("failed to record risk reject memory")
            return _rejected_response(decision)
        if decision.requires_confirmation:
            _persist(intent, decision, None, "pending_confirmation")
            return _pending_response(intent, decision)

        if base_qty is None:
            base_qty, _ = _resolve_qty(intent, _market_url())
        if intent.mode == "live":
            unlock_error = _consume_live_unlock_or_error(
                str(x_live_unlock), intent.actor, dry=False, intent_id=intent.intent_id
            )
            if unlock_error is not None:
                return unlock_error
        execution = _call_execution(intent, decision, base_qty)
        _persist(intent, decision, execution, "executed")
        _after_fill_side_effects(execution, intent)
        return {
            "status": "executed",
            "intent": intent,
            "decision": decision,
            "execution": execution,
        }
    except DuplicateIntentIdError:
        return JSONResponse(status_code=409, content={"code": "DUPLICATE_INTENT_ID"})


def list_intents(
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
    mode: str | None = None,
) -> dict[str, object]:
    clamped_limit = min(limit, 100)
    params: list[object] = []
    where = ""
    if mode is not None:
        where = " where json_extract(payload_json, '$.mode') = ?"
        params.append(mode)
    with connect() as conn:
        total = conn.execute(f"select count(*) from intents{where}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            select payload_json, decision_json, execution_json, status from intents{where}
            order by created_at desc limit ? offset ?
            """,
            [*params, clamped_limit, offset],
        ).fetchall()
    return {"items": [_row_to_item(row) for row in rows], "total": total}


def get_exposure(date: str | None = None) -> dict[str, object]:
    target_date = date or _today()
    symbols: dict[str, dict[str, str]] = {}
    with connect() as conn:
        rows = conn.execute(
            "select symbol, side, sum(cast(notional_usdt as real)) as total "
            "from daily_fills where date = ? group by symbol, side",
            (target_date,),
        ).fetchall()
    totals: dict[str, dict[str, Decimal]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        side = str(row["side"])
        total = Decimal(str(row["total"]))
        bucket = totals.setdefault(
            symbol,
            {"side_buy": Decimal("0"), "side_sell": Decimal("0"), "total": Decimal("0")},
        )
        if side == "buy":
            bucket["side_buy"] += total
        elif side == "sell":
            bucket["side_sell"] += total
        bucket["total"] += total
    for symbol, values in totals.items():
        symbols[symbol] = {key: str(value) for key, value in values.items()}
    return {"date": target_date, "limit_usdt": str(_daily_limit()), "symbols": symbols}


def confirm_intent(
    intent_id: UUID,
    request: ConfirmationRequest,
    x_live_unlock: str = Header(default=""),
) -> JSONResponse | dict[str, object]:
    if request.intent_id != intent_id:
        return JSONResponse(status_code=400, content={"code": "INTENT_ID_MISMATCH"})
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    if row["status"] == "canceled":
        return JSONResponse(status_code=410, content={"code": "INTENT_CANCELED"})
    intent = OrderIntent.model_validate_json(row["payload_json"])
    decision = RiskDecision.model_validate_json(row["decision_json"])
    if not decision.requires_confirmation:
        return JSONResponse(status_code=409, content={"code": "CONFIRMATION_NOT_REQUIRED"})
    if row["execution_json"] is not None:
        return JSONResponse(status_code=409, content={"code": "ALREADY_EXECUTED"})
    if request.confirmation_token != decision.confirmation_token:
        return JSONResponse(status_code=403, content={"code": "INVALID_CONFIRMATION_TOKEN"})
    if decision.confirmation_expires_at is not None and decision.confirmation_expires_at < _now():
        return JSONResponse(status_code=410, content={"code": "CONFIRMATION_EXPIRED"})

    if intent.mode == "live":
        unlock_error = _consume_live_unlock_or_error(
            str(x_live_unlock), intent.actor, dry=False, intent_id=intent.intent_id
        )
        if unlock_error is not None:
            return unlock_error
    base_qty, _ = _resolve_qty(intent, _market_url())
    execution = _call_execution(intent, decision, base_qty)
    with connect() as conn:
        conn.execute(
            "update intents set execution_json = ?, status = ? where intent_id = ?",
            (execution.model_dump_json(), "executed", str(intent_id)),
        )
        conn.commit()
    _after_fill_side_effects(execution, intent)
    return {"intent": intent, "decision": decision, "execution": execution}


def cancel_intent(intent_id: UUID) -> Response | JSONResponse:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id = ?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    status = str(row["status"])
    if status == "pending_confirmation":
        with connect() as conn:
            conn.execute(
                "update intents set status = ? where intent_id = ?",
                ("canceled", str(intent_id)),
            )
            conn.commit()
        return Response(status_code=204)

    if status == "executed" and row["execution_json"] is not None:
        execution = ExecutionResult.model_validate_json(row["execution_json"])
        intent = OrderIntent.model_validate_json(row["payload_json"])
        if execution.status == "open" and intent.mode == "live" and execution.venue_order_id:
            canceled = _call_cancel(intent, execution)
            with connect() as conn:
                conn.execute(
                    "update intents set execution_json = ? where intent_id = ?",
                    (canceled.model_dump_json(), str(intent_id)),
                )
                conn.commit()
            _after_fill_side_effects(canceled, intent)
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(
                    {
                        "status": status,
                        "intent": intent,
                        "decision": RiskDecision.model_validate_json(row["decision_json"]),
                        "execution": canceled,
                    }
                ),
            )

    return JSONResponse(
        status_code=409, content={"code": "CANNOT_CANCEL", "current_status": status}
    )


def get_paper_positions(actor: str | None = None) -> JSONResponse | dict[str, object]:
    if not actor:
        return JSONResponse(status_code=400, content={"code": "ACTOR_REQUIRED"})
    with connect() as conn:
        rows = conn.execute(
            "select symbol, qty, avg_cost, total_cost, realized_pnl, venue from paper_positions "
            "where actor = ? order by symbol",
            (actor,),
        ).fetchall()

    mark_cache: dict[str, tuple[Decimal | None, str | None]] = {}
    positions: list[dict[str, object]] = []
    for row in rows:
        qty = Decimal(row["qty"])
        avg_cost = Decimal(row["avg_cost"])
        total_cost = Decimal(row["total_cost"])
        realized_pnl = Decimal(row["realized_pnl"])
        mark_price: Decimal | None = None
        mark_source: str | None = None
        mark_value: Decimal | None = None
        unrealized_pnl: Decimal | None = None
        if qty > 0:
            symbol = str(row["symbol"])
            mark_cache.setdefault(symbol, _mark_for_symbol(symbol))
            mark_price, mark_source = mark_cache[symbol]
            if mark_price is not None:
                mark_value = _q8(qty * mark_price)
                unrealized_pnl = _q8(qty * (mark_price - avg_cost))
        positions.append(
            {
                "symbol": row["symbol"],
                "venue": row["venue"],
                "qty": _q8s(qty),
                "avg_cost": _q8s(avg_cost),
                "total_cost": _q8s(total_cost),
                "realized_pnl": _q8s(realized_pnl),
                "mark_price": _q8s(mark_price) if mark_price is not None else None,
                "mark_value": _q8s(mark_value) if mark_value is not None else None,
                "unrealized_pnl": _q8s(unrealized_pnl) if unrealized_pnl is not None else None,
                "mark_source": mark_source,
            }
        )
    return {"actor": actor, "positions": positions}


def refresh_intent(intent_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})

    item = _row_to_item(row)
    intent = item["intent"]
    execution = item["execution"]

    if (
        row["status"] == "executed"
        and isinstance(intent, OrderIntent)
        and isinstance(execution, ExecutionResult)
        and execution.status == "open"
        and intent.mode == "live"
        and execution.venue_order_id
    ):
        refreshed = _call_refresh(intent, execution)
        if refreshed.status != "open":
            with connect() as conn:
                conn.execute(
                    "update intents set execution_json = ? where intent_id = ?",
                    (refreshed.model_dump_json(), str(intent_id)),
                )
                conn.commit()
            _after_fill_side_effects(refreshed, intent)
            item["execution"] = refreshed

    return item


def get_intent(intent_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select payload_json, decision_json, execution_json, status "
            "from intents where intent_id=?",
            (str(intent_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "INTENT_NOT_FOUND"})
    return _row_to_item(row)


def _persist(
    intent: OrderIntent, decision: RiskDecision, execution: ExecutionResult | None, status: str
) -> None:
    try:
        with connect() as conn:
            conn.execute(
                """
                insert into intents
                (intent_id,idempotency_key,payload_json,decision_json,execution_json,created_at,status)
                values(?,?,?,?,?,?,?)
                """,
                (
                    str(intent.intent_id),
                    intent.idempotency_key,
                    intent.model_dump_json(),
                    decision.model_dump_json(),
                    execution.model_dump_json() if execution else None,
                    _now().isoformat(),
                    status,
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        if "intent_id" in str(exc):
            raise DuplicateIntentIdError from exc
        raise


# ---------------------------------------------------------------------------
# IBKR Portfolio Reconciliation (Phase 23)
# ---------------------------------------------------------------------------


def _write_reconcile_log(
    run_at: str,
    ibkr_positions: list[dict[str, str]],
    aegis_positions: list[dict[str, str]],
    drift: list[dict[str, object]],
    status: str,
    error: str | None,
    bridge_last_update: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into reconcile_log
              (run_at, ibkr_positions_json, aegis_positions_json,
               drift_json, status, error, bridge_last_update)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_at,
                json.dumps(ibkr_positions),
                json.dumps(aegis_positions),
                json.dumps(drift),
                status,
                error,
                bridge_last_update,
            ),
        )
        conn.commit()


def _ibkr_apply_total_cost(qty: object, avg_cost: object) -> str:
    return _q8s(_q8(Decimal(str(qty)) * Decimal(str(avg_cost))))


def _build_reconcile_apply_patches(drift_json: str) -> list[dict[str, object]]:
    # TODO(STOP-AND-ASK): Phase 25 spec names split JSON columns on reconcile_log,
    # but the existing Phase 23 schema stores all entries in drift_json.
    raw_drift = json.loads(drift_json or "[]")
    patches: list[dict[str, object]] = []
    if not isinstance(raw_drift, list):
        return patches

    for raw_item in raw_drift:
        if not isinstance(raw_item, dict):
            continue
        item = cast(dict[str, object], raw_item)
        kind = item.get("kind") or item.get("type")
        actor_value = item.get("actor")
        actor = str(actor_value) if actor_value is not None else None
        if kind == "qty_mismatch":
            patch: dict[str, object] = {
                "type": "qty_mismatch",
                "symbol": item["symbol"],
                "ibkr_qty": item["ibkr_qty"],
                "ibkr_avg_cost": item["ibkr_avg_cost"],
            }
            if actor:
                patch["actor"] = actor
            patches.append(patch)
        elif kind == "unknown_ibkr_position":
            patches.append(
                {
                    "type": "unknown_ibkr_position",
                    "symbol": item["symbol"],
                    "actor": actor or RECONCILE_APPLY_DEFAULT_ACTOR,
                    "ibkr_qty": item["ibkr_qty"],
                    "ibkr_avg_cost": item["ibkr_avg_cost"],
                }
            )
        elif kind == "phantom_aegis_position":
            patch = {
                "type": "phantom_aegis_position",
                "symbol": item["symbol"],
            }
            if actor:
                patch["actor"] = actor
            patches.append(patch)
    return patches


def _run_reconcile_apply_patches(
    conn: sqlite3.Connection,
    patches: list[dict[str, object]],
    applied_at: str,
) -> None:
    for patch in patches:
        sym = str(patch["symbol"])
        patch_type = str(patch["type"])
        actor_value = patch.get("actor")
        actor = str(actor_value) if actor_value is not None else None
        if patch_type == "qty_mismatch":
            if actor:
                conn.execute(
                    """
                    update paper_positions
                    set live_qty = ?, live_avg_cost = ?, last_updated = ?
                    where actor = ? and symbol = ? and venue = 'ibkr_us_equity'
                    """,
                    (patch["ibkr_qty"], patch["ibkr_avg_cost"], applied_at, actor, sym),
                )
            else:
                conn.execute(
                    """
                    update paper_positions
                    set live_qty = ?, live_avg_cost = ?, last_updated = ?
                    where symbol = ? and venue = 'ibkr_us_equity'
                    """,
                    (patch["ibkr_qty"], patch["ibkr_avg_cost"], applied_at, sym),
                )
        elif patch_type == "unknown_ibkr_position":
            # TODO(STOP-AND-ASK): paper_positions is keyed by (actor, symbol), not
            # UNIQUE(symbol, venue). Unknown IBKR rows have no natural actor, so
            # they are stored under a deterministic reconcile-owned actor.
            patch_actor = actor or RECONCILE_APPLY_DEFAULT_ACTOR
            live_qty = patch["ibkr_qty"]
            live_avg_cost = patch["ibkr_avg_cost"]
            conn.execute(
                """
                insert into paper_positions
                  (actor, symbol, qty, avg_cost, total_cost, realized_pnl,
                   paper_qty, paper_avg_cost, live_qty, live_avg_cost, venue, last_updated)
                values (?, ?, ?, ?, ?, '0', '0', '0', ?, ?, 'ibkr_us_equity', ?)
                on conflict(actor, symbol) do update set
                    qty = excluded.qty,
                    avg_cost = excluded.avg_cost,
                    total_cost = excluded.total_cost,
                    live_qty = excluded.live_qty,
                    live_avg_cost = excluded.live_avg_cost,
                    venue = excluded.venue,
                    last_updated = excluded.last_updated
                """,
                (
                    patch_actor,
                    sym,
                    live_qty,
                    live_avg_cost,
                    _ibkr_apply_total_cost(live_qty, live_avg_cost),
                    live_qty,
                    live_avg_cost,
                    applied_at,
                ),
            )
        elif patch_type == "phantom_aegis_position":
            if actor:
                conn.execute(
                    """
                    update paper_positions
                    set live_qty = '0', last_updated = ?
                    where actor = ? and symbol = ? and venue = 'ibkr_us_equity'
                    """,
                    (applied_at, actor, sym),
                )
            else:
                conn.execute(
                    """
                    update paper_positions
                    set live_qty = '0', last_updated = ?
                    where symbol = ? and venue = 'ibkr_us_equity'
                    """,
                    (applied_at, sym),
                )


def post_reconcile_apply(request: Request) -> dict[str, object]:
    _check_ops_auth(request)

    applied_at = _now().isoformat()
    with connect() as conn:
        row = conn.execute(
            """
            select run_at, status, drift_json
            from reconcile_log
            order by run_at desc, id desc
            limit 1
            """
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RECONCILE_LOG_EMPTY",
                "message": "no reconciliation has been run yet",
            },
        )

    run_at = str(row["run_at"])
    log_status = str(row["status"])
    if log_status == "error":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RECONCILE_LOG_ERROR_STATUS",
                "message": (
                    "latest reconcile log has status='error'; re-run POST /reconcile/ibkr first"
                ),
            },
        )

    run_at_dt = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    if run_at_dt.tzinfo is None:
        run_at_dt = run_at_dt.replace(tzinfo=UTC)
    age_seconds = (_now() - run_at_dt.astimezone(UTC)).total_seconds()
    if age_seconds > RECONCILE_MAX_AGE_SECONDS:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RECONCILE_LOG_STALE",
                "message": (
                    f"reconcile log is {int(age_seconds)}s old "
                    f"(max {RECONCILE_MAX_AGE_SECONDS}s); "
                    "re-run POST /reconcile/ibkr first"
                ),
            },
        )

    patches = _build_reconcile_apply_patches(str(row["drift_json"]))

    try:
        with connect() as conn:
            _run_reconcile_apply_patches(conn, patches, applied_at)
            write_apply_log(conn, applied_at, run_at, patches, "ok")
            conn.commit()
    except Exception as exc:
        logger.error("RECONCILE_APPLY_FAILED error=%s", exc)
        try:
            with connect() as conn:
                write_apply_log(conn, applied_at, run_at, patches, "error", str(exc)[:500])
                conn.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail={"code": "RECONCILE_APPLY_FAILED", "message": str(exc)[:500]},
        ) from exc

    return {
        "applied_at": applied_at,
        "reconcile_run_at": run_at,
        "total_patched": len(patches),
        "patches": patches,
    }


def run_ibkr_reconciliation(
    bridge_url: str | None = None,
    tolerance: Decimal | None = None,
) -> dict[str, object]:
    """
    Fetch IBKR positions from the bridge and compare against paper_positions
    where venue='ibkr_us_equity'.  Read-only: never modifies paper_positions.

    Returns a result dict and writes one row to reconcile_log.
    """
    url = (bridge_url or IBKR_BRIDGE_URL).rstrip("/")
    tol = tolerance if tolerance is not None else RECONCILE_AVG_COST_TOLERANCE
    run_at = _now().isoformat()
    bridge_last_update: str | None = None

    try:
        resp = httpx.get(f"{url}/positions", timeout=10.0)
        if getattr(resp, "status_code", 200) == 503:
            try:
                detail = resp.json().get("detail", {})
            except Exception:
                detail = {}
            if isinstance(detail, dict) and detail.get("code") == "IBKR_POSITIONS_NOT_READY":
                msg = "bridge position cache not ready"
                _write_reconcile_log(run_at, [], [], [], "error", msg)
                logger.warning("RECONCILE_IBKR bridge_not_ready")
                return {
                    "run_at": run_at,
                    "ibkr_positions": [],
                    "aegis_positions": [],
                    "drift": [],
                    "status": "error",
                    "error": msg,
                    "bridge_last_update": None,
                }
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        _write_reconcile_log(run_at, [], [], [], "error", str(exc)[:500])
        logger.warning("RECONCILE_IBKR bridge_unreachable error=%s", exc)
        return {
            "run_at": run_at,
            "ibkr_positions": [],
            "aegis_positions": [],
            "drift": [],
            "status": "error",
            "error": str(exc)[:500],
            "bridge_last_update": None,
        }

    bridge_last_update = cast(str | None, payload.get("last_update"))
    if not payload.get("ready", True):
        msg = "bridge position cache not ready"
        _write_reconcile_log(run_at, [], [], [], "error", msg, bridge_last_update)
        logger.warning("RECONCILE_IBKR bridge_not_ready")
        return {
            "run_at": run_at,
            "ibkr_positions": [],
            "aegis_positions": [],
            "drift": [],
            "status": "error",
            "error": msg,
            "bridge_last_update": bridge_last_update,
        }

    ibkr_raw = cast(list[dict[str, str]], payload.get("positions", []))

    with connect() as conn:
        rows = conn.execute(
            "select actor, symbol, live_qty, live_avg_cost "
            "from paper_positions "
            "where venue = 'ibkr_us_equity' "
            "  and cast(live_qty as real) != 0"
        ).fetchall()

    aegis_raw: list[dict[str, str]] = [
        {
            "actor": row["actor"],
            "symbol": row["symbol"],
            "live_qty": row["live_qty"],
            "live_avg_cost": row["live_avg_cost"],
        }
        for row in rows
    ]

    ibkr_map: dict[str, dict[str, str]] = {p["symbol"].upper(): p for p in ibkr_raw}
    aegis_map: dict[str, list[dict[str, str]]] = {}
    for pos in aegis_raw:
        aegis_map.setdefault(pos["symbol"].upper(), []).append(pos)

    all_symbols = sorted(set(ibkr_map) | set(aegis_map))
    drift: list[dict[str, object]] = []

    for symbol in all_symbols:
        ibkr_pos = ibkr_map.get(symbol)
        aegis_list = aegis_map.get(symbol, [])

        if ibkr_pos and not aegis_list:
            drift.append(
                {
                    "kind": "unknown_ibkr_position",
                    "symbol": symbol,
                    "actor": None,
                    "ibkr_qty": ibkr_pos["qty"],
                    "ibkr_avg_cost": ibkr_pos["avg_cost"],
                    "aegis_qty": "0.00000000",
                    "aegis_avg_cost": "0.00000000",
                    "qty_delta": ibkr_pos["qty"],
                    "avg_cost_delta": ibkr_pos["avg_cost"],
                }
            )
            continue

        if not ibkr_pos and aegis_list:
            for ap in aegis_list:
                drift.append(
                    {
                        "kind": "phantom_aegis_position",
                        "symbol": symbol,
                        "actor": ap["actor"],
                        "ibkr_qty": "0.00000000",
                        "ibkr_avg_cost": "0.00000000",
                        "aegis_qty": ap["live_qty"],
                        "aegis_avg_cost": ap["live_avg_cost"],
                        "qty_delta": _q8s(-_q8(Decimal(ap["live_qty"]))),
                        "avg_cost_delta": _q8s(-_q8(Decimal(ap["live_avg_cost"]))),
                    }
                )
            continue

        for ap in aegis_list:
            ibkr_qty = _q8(Decimal(ibkr_pos["qty"]))  # type: ignore[index]
            ibkr_avg = _q8(Decimal(ibkr_pos["avg_cost"]))  # type: ignore[index]
            aegis_qty = _q8(Decimal(ap["live_qty"]))
            aegis_avg = _q8(Decimal(ap["live_avg_cost"]))
            qty_delta = ibkr_qty - aegis_qty
            avg_delta = ibkr_avg - aegis_avg
            if qty_delta != Decimal("0") or abs(avg_delta) > tol:
                drift.append(
                    {
                        "kind": "qty_mismatch",
                        "symbol": symbol,
                        "actor": ap["actor"],
                        "ibkr_qty": _q8s(ibkr_qty),
                        "ibkr_avg_cost": _q8s(ibkr_avg),
                        "aegis_qty": _q8s(aegis_qty),
                        "aegis_avg_cost": _q8s(aegis_avg),
                        "qty_delta": _q8s(qty_delta),
                        "avg_cost_delta": _q8s(avg_delta),
                    }
                )

    status = "ok" if not drift else "drift"
    _write_reconcile_log(run_at, ibkr_raw, aegis_raw, drift, status, None, bridge_last_update)

    log_level = logging.INFO if status == "ok" else logging.WARNING
    logger.log(log_level, "RECONCILE_IBKR status=%s drift_count=%d", status, len(drift))

    return {
        "run_at": run_at,
        "ibkr_positions": ibkr_raw,
        "aegis_positions": aegis_raw,
        "drift": drift,
        "status": status,
        "error": None,
        "bridge_last_update": bridge_last_update,
    }


def trigger_ibkr_reconcile() -> JSONResponse | dict[str, object]:
    result = run_ibkr_reconciliation()
    if result["status"] == "error":
        return JSONResponse(status_code=503, content=result)
    return result


def get_latest_ibkr_reconcile() -> dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select run_at, ibkr_positions_json, aegis_positions_json, "
            "drift_json, status, error, bridge_last_update "
            "from reconcile_log "
            "order by run_at desc, id desc "
            "limit 1"
        ).fetchone()
    if row is None:
        return {"status": "never_run"}
    return {
        "run_at": row["run_at"],
        "ibkr_positions": json.loads(str(row["ibkr_positions_json"])),
        "aegis_positions": json.loads(str(row["aegis_positions_json"])),
        "drift": json.loads(str(row["drift_json"])),
        "status": row["status"],
        "error": row["error"],
        "bridge_last_update": row["bridge_last_update"],
    }
