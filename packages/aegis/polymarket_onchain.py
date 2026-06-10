from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATA_API_BASE_URL = "https://data-api.polymarket.com"
GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_USER_AGENT = "aegis-polymarket-research/0.1 read-only"


@dataclass(frozen=True)
class PolymarketTrade:
    condition_id: str
    outcome_index: int
    price: Decimal
    size: Decimal
    timestamp: int
    side: str
    transaction_hash: str | None = None


@dataclass(frozen=True)
class PolymarketClosedMarket:
    condition_id: str
    slug: str
    title: str
    outcomes: tuple[str, ...]
    outcome_prices: tuple[Decimal, ...]
    end_time: str | None = None
    closed_time: str | None = None


@dataclass(frozen=True)
class LosingHighPriceSample:
    condition_id: str
    slug: str
    title: str
    losing_outcome: str
    losing_outcome_index: int
    decision_timestamp: int
    decision_price: Decimal
    transaction_hash: str | None


class PolymarketDataApiClient:
    def __init__(
        self,
        *,
        data_api_base_url: str = DATA_API_BASE_URL,
        gamma_api_base_url: str = GAMMA_API_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.data_api_base_url = data_api_base_url.rstrip("/")
        self.gamma_api_base_url = gamma_api_base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def get_closed_markets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order: str = "closedTime",
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        params = {
            "closed": "true",
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        data = self._get_json(f"{self.gamma_api_base_url}/markets?{urlencode(params)}")
        if not isinstance(data, list):
            raise ValueError("unexpected Gamma closed markets response")
        return [row for row in data if isinstance(row, dict)]

    def get_trades(
        self,
        condition_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
        taker_only: bool = False,
    ) -> list[dict[str, Any]]:
        params = {
            "market": condition_id,
            "limit": str(limit),
            "offset": str(offset),
            "takerOnly": str(taker_only).lower(),
        }
        data = self._get_json(f"{self.data_api_base_url}/trades?{urlencode(params)}")
        if not isinstance(data, list):
            raise ValueError("unexpected Data API trades response")
        return [row for row in data if isinstance(row, dict)]

    def _get_json(self, url: str) -> Any:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode())


def parse_closed_market(raw: Mapping[str, Any]) -> PolymarketClosedMarket | None:
    condition_id = raw.get("conditionId")
    if not isinstance(condition_id, str) or not condition_id:
        return None
    outcomes = _json_list(raw.get("outcomes"))
    prices = _json_list(raw.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    try:
        outcome_prices = tuple(Decimal(str(price)) for price in prices)
    except Exception:
        return None
    return PolymarketClosedMarket(
        condition_id=condition_id,
        slug=str(raw.get("slug") or ""),
        title=str(raw.get("question") or raw.get("title") or ""),
        outcomes=tuple(str(outcome) for outcome in outcomes),
        outcome_prices=outcome_prices,
        end_time=_optional_str(raw.get("endDate")),
        closed_time=_optional_str(raw.get("closedTime")),
    )


def parse_trade(raw: Mapping[str, Any]) -> PolymarketTrade | None:
    condition_id = raw.get("conditionId")
    if not isinstance(condition_id, str) or not condition_id:
        return None
    try:
        return PolymarketTrade(
            condition_id=condition_id,
            outcome_index=int(raw["outcomeIndex"]),
            price=Decimal(str(raw["price"])),
            size=Decimal(str(raw.get("size", "0"))),
            timestamp=int(raw["timestamp"]),
            side=str(raw.get("side") or ""),
            transaction_hash=_optional_str(raw.get("transactionHash")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def losing_outcome_indices(
    market: PolymarketClosedMarket,
    *,
    losing_price_threshold: Decimal = Decimal("0.001"),
) -> tuple[int, ...]:
    return tuple(
        index
        for index, price in enumerate(market.outcome_prices)
        if price <= losing_price_threshold
    )


def last_trade_at_or_before(
    trades: Iterable[PolymarketTrade],
    *,
    outcome_index: int,
    decision_timestamp: int,
) -> PolymarketTrade | None:
    eligible = (
        trade
        for trade in trades
        if trade.outcome_index == outcome_index and trade.timestamp <= decision_timestamp
    )
    return max(eligible, key=lambda trade: trade.timestamp, default=None)


def find_losing_high_price_samples(
    markets: Iterable[PolymarketClosedMarket],
    trades_by_condition: Mapping[str, Iterable[PolymarketTrade]],
    *,
    lower: Decimal = Decimal("0.95"),
    upper: Decimal = Decimal("0.99"),
) -> list[LosingHighPriceSample]:
    samples: list[LosingHighPriceSample] = []
    for market in markets:
        trades = list(trades_by_condition.get(market.condition_id, ()))
        for outcome_index in losing_outcome_indices(market):
            for trade in trades:
                if trade.outcome_index != outcome_index:
                    continue
                if lower <= trade.price <= upper:
                    samples.append(
                        LosingHighPriceSample(
                            condition_id=market.condition_id,
                            slug=market.slug,
                            title=market.title,
                            losing_outcome=market.outcomes[outcome_index],
                            losing_outcome_index=outcome_index,
                            decision_timestamp=trade.timestamp,
                            decision_price=trade.price,
                            transaction_hash=trade.transaction_hash,
                        )
                    )
    return samples


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    return []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
