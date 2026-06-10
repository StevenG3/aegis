from typing import Any

import pytest
from pydantic import ValidationError

from schemas import ExecutionResult, OrderIntent, RiskDecision, Scorecard

VALID_INTENT: dict[str, Any] = {
    "intent_id": "11111111-1111-4111-8111-111111111111",
    "request_id": "22222222-2222-4222-8222-222222222222",
    "idempotency_key": "demo-paper-1",
    "actor": "user_1",
    "created_at": "2026-05-25T00:00:00Z",
    "mode": "paper",
    "venue": "binance_spot",
    "symbol": "BTCUSDT",
    "side": "buy",
    "order_type": "market",
    "quantity": {"kind": "quote", "value": "100"},
    "limit_price": None,
    "time_in_force": "GTC",
    "reduce_only": False,
    "leverage": None,
    "stop_loss": None,
    "take_profit": None,
    "source": {"origin": "manual_api", "scorecard_id": None, "hermes_message_id": None},
    "client_confirmation_required": False,
}

VALID_DECISION: dict[str, Any] = {
    "decision_id": "33333333-3333-4333-8333-333333333333",
    "intent_id": VALID_INTENT["intent_id"],
    "evaluated_at": "2026-05-25T00:00:01Z",
    "approved": True,
    "reasons": [],
    "requires_confirmation": False,
    "confirmation_token": None,
    "confirmation_expires_at": None,
    "hard_caps_applied": {
        "max_notional": "10000",
        "max_leverage": None,
        "max_drawdown_today": None,
        "per_symbol_exposure": None,
    },
    "evaluator_version": "risk-engine@0.1.0",
}

VALID_EXECUTION: dict[str, Any] = {
    "execution_id": "44444444-4444-4444-8444-444444444444",
    "intent_id": VALID_INTENT["intent_id"],
    "decision_id": VALID_DECISION["decision_id"],
    "idempotency_key": VALID_INTENT["idempotency_key"],
    "status": "simulated",
    "venue_order_id": None,
    "fills": [
        {
            "price": "100000.00",
            "qty": "0.001",
            "fee": "0",
            "fee_asset": "USDT",
            "ts": "2026-05-25T00:00:02Z",
        }
    ],
    "avg_price": "100000.00",
    "filled_qty": "0.001",
    "remaining_qty": "0",
    "error": None,
    "raw_venue_response_ref": None,
    "finalized_at": "2026-05-25T00:00:03Z",
}


def test_order_intent_round_trip() -> None:
    intent = OrderIntent.model_validate(VALID_INTENT)
    again = OrderIntent.model_validate(intent.model_dump())
    assert again == intent


def test_order_intent_accepts_ibkr_us_equity_venue() -> None:
    intent = OrderIntent.model_validate(
        {**VALID_INTENT, "venue": "ibkr_us_equity", "symbol": "NVDA"}
    )
    assert intent.venue == "ibkr_us_equity"
    assert intent.symbol == "NVDA"


def test_order_intent_rejects_unknown_venue() -> None:
    with pytest.raises(ValidationError):
        OrderIntent.model_validate({**VALID_INTENT, "venue": "nasdaq_arca"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        OrderIntent.model_validate(dict(VALID_INTENT, extra="nope"))


def test_quantity_must_be_positive() -> None:
    payload = dict(VALID_INTENT)
    payload["quantity"] = {"kind": "quote", "value": "0"}
    with pytest.raises(ValidationError):
        OrderIntent.model_validate(payload)


def test_monetary_fields_reject_floats() -> None:
    with pytest.raises(ValidationError):
        OrderIntent.model_validate(dict(VALID_INTENT, limit_price=1.5))


def test_risk_decision_round_trip() -> None:
    decision = RiskDecision.model_validate(VALID_DECISION)
    again = RiskDecision.model_validate(decision.model_dump())
    assert again == decision


def test_execution_result_round_trip() -> None:
    execution = ExecutionResult.model_validate(VALID_EXECUTION)
    again = ExecutionResult.model_validate(execution.model_dump())
    assert again == execution


VALID_SCORECARD: dict[str, Any] = {
    "scorecard_id": "55555555-5555-4555-8555-555555555555",
    "created_at": "2026-05-25T00:00:00Z",
    "expires_at": "2026-05-25T01:00:00Z",
    "source": "manual",
    "actor": "user_1",
    "symbol": "BTCUSDT",
    "action": "buy",
    "conviction": "0.8",
    "thesis": "Breakout above 100k support retest",
    "entry_low": "95000.00",
    "entry_high": "100000.00",
    "stop_loss": "90000.00",
    "take_profit": "110000.00",
    "time_horizon": "swing",
    "metadata": {"source_note": "manual"},
}


def test_scorecard_roundtrip() -> None:
    scorecard = Scorecard.model_validate(VALID_SCORECARD)
    again = Scorecard.model_validate_json(scorecard.model_dump_json())
    assert again == scorecard


def test_scorecard_conviction_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Scorecard.model_validate(dict(VALID_SCORECARD, conviction="1.5"))


def test_scorecard_conviction_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        Scorecard.model_validate(dict(VALID_SCORECARD, conviction="-0.1"))


def test_scorecard_action_hold_allowed_in_schema() -> None:
    scorecard = Scorecard.model_validate(dict(VALID_SCORECARD, action="hold"))
    assert scorecard.action == "hold"


def test_scorecard_thesis_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        Scorecard.model_validate(dict(VALID_SCORECARD, thesis=""))


def test_scorecard_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Scorecard.model_validate(dict(VALID_SCORECARD, extra="nope"))
