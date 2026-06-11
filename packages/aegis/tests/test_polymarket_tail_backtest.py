from __future__ import annotations

from decimal import Decimal

from aegis.polymarket_onchain import PolymarketClosedMarket, PolymarketTrade
from aegis.polymarket_tail_backtest import (
    PolymarketTailCostConfig,
    build_tail_positions,
    summarize_tail_backtest,
)


def test_tail_positions_use_last_trade_at_or_before_decision_and_ignore_future() -> None:
    market = PolymarketClosedMarket(
        condition_id="0xcondition",
        slug="synthetic-flip",
        title="Synthetic flip?",
        outcomes=("Yes", "No"),
        outcome_prices=(Decimal("0"), Decimal("1")),
        closed_time="2026-06-11 00:00:00+00",
    )
    trades = {
        "0xcondition": [
            PolymarketTrade("0xcondition", 0, Decimal("0.96"), Decimal("1"), 100, "BUY"),
            PolymarketTrade("0xcondition", 0, Decimal("0.10"), Decimal("1"), 110, "SELL"),
            PolymarketTrade("0xcondition", 1, Decimal("0.97"), Decimal("1"), 100, "BUY"),
        ]
    }

    positions = build_tail_positions(
        [market],
        trades,
        cost_config=PolymarketTailCostConfig(
            fee_coefficient=Decimal("0"),
            slippage_bps=Decimal("0"),
            gas_usd_per_entry=Decimal("0"),
        ),
    )

    loser = next(position for position in positions if position.outcome == "Yes")
    assert loser.decision_timestamp == 100
    assert loser.decision_price == Decimal("0.96")
    assert loser.settlement_price == Decimal("0")
    assert loser.gross_return == Decimal("-1")


def test_tail_summary_caps_positive_carry_with_thin_loss_events() -> None:
    markets = [
        PolymarketClosedMarket(
            condition_id=f"0xwin{index}",
            slug=f"nba-win-{index}",
            title="NBA synthetic?",
            outcomes=("Yes", "No"),
            outcome_prices=(Decimal("1"), Decimal("0")),
            closed_time=f"2026-06-{index + 1:02d} 00:00:00+00",
        )
        for index in range(60)
    ]
    markets.extend(
        [
            PolymarketClosedMarket(
                condition_id="0xloss1",
                slug="nba-loss-1",
                title="NBA synthetic loss?",
                outcomes=("Yes", "No"),
                outcome_prices=(Decimal("0"), Decimal("1")),
                closed_time="2026-06-20 00:00:00+00",
            ),
            PolymarketClosedMarket(
                condition_id="0xloss2",
                slug="nba-loss-2",
                title="NBA synthetic loss?",
                outcomes=("Yes", "No"),
                outcome_prices=(Decimal("0"), Decimal("1")),
                closed_time="2026-06-21 00:00:00+00",
            ),
        ]
    )
    trades = {
        market.condition_id: [
            PolymarketTrade(market.condition_id, 0, Decimal("0.95"), Decimal("1"), index, "BUY")
        ]
        for index, market in enumerate(markets, start=1)
    }

    positions = build_tail_positions(
        markets,
        trades,
        cost_config=PolymarketTailCostConfig(
            fee_coefficient=Decimal("0"),
            slippage_bps=Decimal("0"),
            gas_usd_per_entry=Decimal("0"),
        ),
    )
    summary = summarize_tail_backtest(
        positions,
        risk_free_return_per_trade=Decimal("0"),
        bootstrap_iterations=200,
    )

    assert summary["sample"]["positions"] == 62
    assert summary["sample"]["losses"] == 2
    assert summary["returns"]["mean_net_return"] > 0
    assert summary["verdict"] == "CARRY_POSITIVE_TAIL_UNDER_SAMPLED"
    assert summary["tail_metrics"]["wins_needed_reference_by_entry_price"]["0.99"] == 99
    assert summary["zero_risk_claim"] == "explicitly_rejected"


def test_tail_summary_marks_no_robust_edge_when_bootstrap_low_tail_is_not_positive() -> None:
    markets = [
        PolymarketClosedMarket(
            condition_id="0xwin",
            slug="synthetic-win",
            title="Synthetic win?",
            outcomes=("Yes", "No"),
            outcome_prices=(Decimal("1"), Decimal("0")),
        ),
        PolymarketClosedMarket(
            condition_id="0xloss",
            slug="synthetic-loss",
            title="Synthetic loss?",
            outcomes=("Yes", "No"),
            outcome_prices=(Decimal("0"), Decimal("1")),
        ),
    ]
    trades = {
        "0xwin": [PolymarketTrade("0xwin", 0, Decimal("0.98"), Decimal("1"), 100, "BUY")],
        "0xloss": [PolymarketTrade("0xloss", 0, Decimal("0.98"), Decimal("1"), 101, "BUY")],
    }

    positions = build_tail_positions(
        markets,
        trades,
        cost_config=PolymarketTailCostConfig(
            fee_coefficient=Decimal("0"),
            slippage_bps=Decimal("0"),
            gas_usd_per_entry=Decimal("0"),
        ),
    )
    summary = summarize_tail_backtest(
        positions,
        risk_free_return_per_trade=Decimal("0"),
        bootstrap_iterations=200,
    )

    assert summary["verdict"] == "NO_ROBUST_EDGE"
    assert summary["tail_metrics"]["max_single_loss"] == -1.0
