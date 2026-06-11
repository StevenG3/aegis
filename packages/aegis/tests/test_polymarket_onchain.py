from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from aegis.polymarket_onchain import (
    PolymarketClosedMarket,
    PolymarketDataApiClient,
    PolymarketTrade,
    SurvivorPowerThreshold,
    analyze_survivor_power_coverage,
    find_losing_high_price_samples,
    last_trade_at_or_before,
    losing_outcome_indices,
    parse_closed_market,
    parse_trade,
    survivor_power_coverage_to_dict,
)


def test_client_builds_read_only_data_api_trade_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["user_agent"] = request.headers["User-agent"]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("aegis.polymarket_onchain.urlopen", fake_urlopen)
    client = PolymarketDataApiClient(timeout_seconds=3.0)

    assert client.get_trades("0xabc", limit=500, offset=100, taker_only=False) == []

    assert captured["timeout"] == 3.0
    assert captured["user_agent"] == "aegis-polymarket-research/0.1 read-only"
    url = str(captured["url"])
    assert url.startswith("https://data-api.polymarket.com/trades?")
    assert "market=0xabc" in url
    assert "limit=500" in url
    assert "offset=100" in url
    assert "takerOnly=false" in url


def test_parse_closed_market_and_losing_indices_from_gamma_shape() -> None:
    market = parse_closed_market(
        {
            "conditionId": "0xcondition",
            "slug": "synthetic-market",
            "question": "Synthetic market?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0", "1"]',
            "endDate": "2026-06-10T00:00:00Z",
            "closedTime": "2026-06-10 01:00:00+00",
        }
    )

    assert market == PolymarketClosedMarket(
        condition_id="0xcondition",
        slug="synthetic-market",
        title="Synthetic market?",
        outcomes=("Yes", "No"),
        outcome_prices=(Decimal("0"), Decimal("1")),
        end_time="2026-06-10T00:00:00Z",
        closed_time="2026-06-10 01:00:00+00",
    )
    assert losing_outcome_indices(market) == (0,)


def test_parse_trade_from_data_api_shape() -> None:
    trade = parse_trade(
        {
            "conditionId": "0xcondition",
            "outcomeIndex": 0,
            "price": 0.97,
            "size": 12.5,
            "timestamp": 1_780_000_000,
            "side": "BUY",
            "transactionHash": "0xhash",
        }
    )

    assert trade == PolymarketTrade(
        condition_id="0xcondition",
        outcome_index=0,
        price=Decimal("0.97"),
        size=Decimal("12.5"),
        timestamp=1_780_000_000,
        side="BUY",
        transaction_hash="0xhash",
    )


def test_last_trade_at_or_before_never_uses_future_trade() -> None:
    trades = [
        PolymarketTrade("0xcondition", 0, Decimal("0.91"), Decimal("1"), 90, "BUY"),
        PolymarketTrade("0xcondition", 0, Decimal("0.96"), Decimal("1"), 100, "BUY"),
        PolymarketTrade("0xcondition", 0, Decimal("0.20"), Decimal("1"), 110, "SELL"),
        PolymarketTrade("0xcondition", 1, Decimal("0.99"), Decimal("1"), 100, "BUY"),
    ]

    decision = last_trade_at_or_before(trades, outcome_index=0, decision_timestamp=105)

    assert decision is not None
    assert decision.timestamp == 100
    assert decision.price == Decimal("0.96")


def test_find_losing_high_price_samples_requires_settled_loser() -> None:
    market = PolymarketClosedMarket(
        condition_id="0xcondition",
        slug="synthetic-flip",
        title="Synthetic flip?",
        outcomes=("Yes", "No"),
        outcome_prices=(Decimal("0"), Decimal("1")),
    )
    trades = {
        "0xcondition": [
            PolymarketTrade("0xcondition", 0, Decimal("0.97"), Decimal("2"), 100, "BUY"),
            PolymarketTrade("0xcondition", 1, Decimal("0.98"), Decimal("2"), 100, "BUY"),
        ]
    }

    samples = find_losing_high_price_samples([market], trades)

    assert len(samples) == 1
    assert samples[0].losing_outcome == "Yes"
    assert samples[0].decision_price == Decimal("0.97")


def test_survivor_power_coverage_counts_winners_losers_and_verdicts() -> None:
    raw_markets = [
        {
            "conditionId": "0xloser",
            "slug": "losing-high-price",
            "question": "Losing high price?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0", "1"]',
        },
        {
            "conditionId": "0xwinner",
            "slug": "winning-high-price",
            "question": "Winning high price?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
        },
    ]
    trades = {
        "0xloser": [
            PolymarketTrade("0xloser", 0, Decimal("0.96"), Decimal("2"), 100, "BUY"),
            PolymarketTrade("0xloser", 0, Decimal("0.10"), Decimal("2"), 110, "SELL"),
        ],
        "0xwinner": [
            PolymarketTrade("0xwinner", 0, Decimal("0.98"), Decimal("2"), 100, "BUY"),
        ],
    }

    coverage = analyze_survivor_power_coverage(
        raw_markets,
        trades,
        threshold=SurvivorPowerThreshold(min_closed_markets=2, min_markets_with_trades=2),
    )
    payload = survivor_power_coverage_to_dict(coverage)

    assert coverage.threshold_met is True
    assert payload["verdict"] == "SURVIVOR_GATE_SATISFIED"
    assert payload["closed_markets_scanned"] == 2
    assert payload["markets_with_trades"] == 2
    assert payload["high_price_markets"] == 2
    assert payload["high_price_losing_outcomes"] == 1
    assert payload["high_price_winning_outcomes"] == 1
    assert payload["losing_samples"][0]["condition_id"] == "0xloser"


def test_survivor_power_coverage_can_make_tail_rare_verdict_after_threshold() -> None:
    raw_markets = [
        {
            "conditionId": "0xwinner",
            "slug": "winning-high-price",
            "question": "Winning high price?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
        },
    ]
    trades = {
        "0xwinner": [
            PolymarketTrade("0xwinner", 0, Decimal("0.98"), Decimal("2"), 100, "BUY"),
        ],
    }

    coverage = analyze_survivor_power_coverage(
        raw_markets,
        trades,
        threshold=SurvivorPowerThreshold(min_closed_markets=1, min_markets_with_trades=1),
    )

    assert coverage.verdict == "TAIL_SAMPLE_RARE_OR_UNREACHABLE"
    assert coverage.high_price_losing_outcomes == 0


def test_survivor_power_coverage_stops_when_threshold_not_met() -> None:
    coverage = analyze_survivor_power_coverage(
        [],
        {},
        threshold=SurvivorPowerThreshold(min_closed_markets=1, min_markets_with_trades=1),
    )

    assert coverage.verdict == "STOP_INSUFFICIENT_COVERAGE"
    assert survivor_power_coverage_to_dict(coverage)["threshold"]["met"] is False
