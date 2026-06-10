from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator


def _parse_decimal_string(value: object) -> Decimal:
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("invalid decimal string") from exc
    else:
        raise ValueError("monetary values must be decimal strings")
    if not parsed.is_finite():
        raise ValueError("decimal string must be finite")
    return parsed


DecimalString = Annotated[Decimal, PlainSerializer(lambda value: str(value), return_type=str)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Quantity(StrictModel):
    kind: Literal["base", "quote"]
    value: DecimalString

    @field_validator("value", mode="before")
    @classmethod
    def validate_decimal_string(cls, value: object) -> Decimal:
        parsed = _parse_decimal_string(value)
        if parsed <= 0:
            raise ValueError("quantity.value must be greater than 0")
        return parsed


class Source(StrictModel):
    origin: Literal["user_nl", "scorecard", "manual_api"]
    scorecard_id: str | None
    hermes_message_id: str | None


class OrderIntent(StrictModel):
    intent_id: UUID
    request_id: UUID
    idempotency_key: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    created_at: datetime
    mode: Literal["paper", "live"]
    venue: Literal["binance_spot", "binance_futures", "ibkr_us_equity"]
    symbol: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: Quantity
    limit_price: DecimalString | None
    time_in_force: Literal["GTC", "IOC", "FOK"]
    reduce_only: bool
    leverage: int | None
    stop_loss: DecimalString | None
    take_profit: DecimalString | None
    source: Source
    client_confirmation_required: bool

    @field_validator("limit_price", "stop_loss", "take_profit", mode="before")
    @classmethod
    def validate_optional_decimal_string(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _parse_decimal_string(value)


class RiskReason(StrictModel):
    code: str
    detail: str


class HardCapsApplied(StrictModel):
    max_notional: DecimalString | None
    max_leverage: int | None
    max_drawdown_today: DecimalString | None
    per_symbol_exposure: DecimalString | None

    @field_validator("max_notional", "max_drawdown_today", "per_symbol_exposure", mode="before")
    @classmethod
    def validate_optional_decimal_string(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _parse_decimal_string(value)


class RiskDecision(StrictModel):
    decision_id: UUID
    intent_id: UUID
    evaluated_at: datetime
    approved: bool
    reasons: list[RiskReason]
    requires_confirmation: bool
    confirmation_token: str | None
    confirmation_expires_at: datetime | None
    hard_caps_applied: HardCapsApplied
    evaluator_version: str = "risk-engine@0.1.0"


class ConfirmationRequest(StrictModel):
    intent_id: UUID
    confirmation_token: str = Field(min_length=1)


class FactorSignal(StrictModel):
    name: str = Field(min_length=1)
    direction: Literal["support", "oppose", "neutral"]
    score: DecimalString | None = None


class Scorecard(StrictModel):
    scorecard_id: UUID
    created_at: datetime
    expires_at: datetime
    source: Literal["manual", "tradingagents", "hermes_chat"]
    actor: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action: Literal["buy", "sell", "hold"]
    conviction: DecimalString
    thesis: str = Field(min_length=1, max_length=4000)
    entry_low: DecimalString | None
    entry_high: DecimalString | None
    stop_loss: DecimalString | None
    take_profit: DecimalString | None
    time_horizon: Literal["intraday", "swing", "position"]
    metadata: dict[str, str] | None = None
    factors: list[FactorSignal] | None = None

    @field_validator("conviction", mode="before")
    @classmethod
    def validate_conviction(cls, value: object) -> Decimal:
        parsed = _parse_decimal_string(value)
        if parsed < 0 or parsed > 1:
            raise ValueError("conviction must be between 0 and 1 inclusive")
        return parsed

    @field_validator(
        "entry_low", "entry_high", "stop_loss", "take_profit", mode="before"
    )
    @classmethod
    def validate_optional_decimal_string(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _parse_decimal_string(value)


class ExecutionRequest(StrictModel):
    execution_id: UUID
    intent_id: UUID
    decision_id: UUID
    idempotency_key: str
    confirmation_token: str | None
    dry_run: bool
    submitted_at: datetime


class Fill(StrictModel):
    price: DecimalString
    qty: DecimalString
    fee: DecimalString
    fee_asset: str
    ts: datetime

    @field_validator("price", "qty", "fee", mode="before")
    @classmethod
    def validate_decimal_string(cls, value: object) -> Decimal:
        return _parse_decimal_string(value)


class ExecutionResult(StrictModel):
    execution_id: UUID
    intent_id: UUID
    decision_id: UUID
    idempotency_key: str
    status: Literal["filled", "partial", "rejected", "canceled", "error", "simulated", "open"]
    venue_order_id: str | None
    fills: list[Fill]
    avg_price: DecimalString | None
    filled_qty: DecimalString
    remaining_qty: DecimalString
    error: str | None
    raw_venue_response_ref: str | None
    finalized_at: datetime

    @field_validator("avg_price", mode="before")
    @classmethod
    def validate_optional_decimal_string(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _parse_decimal_string(value)

    @field_validator("filled_qty", "remaining_qty", mode="before")
    @classmethod
    def validate_decimal_string(cls, value: object) -> Decimal:
        return _parse_decimal_string(value)
