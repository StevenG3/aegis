from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATA_API_BASE_URL = "https://data-api.polymarket.com"
GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_API_BASE_URL = "https://clob.polymarket.com"
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


@dataclass(frozen=True)
class SurvivorPowerThreshold:
    min_closed_markets: int
    min_markets_with_trades: int
    target_closed_window_days: int | None = None


@dataclass(frozen=True)
class SurvivorPowerCoverage:
    closed_markets_scanned: int
    closed_markets_parsed: int
    markets_with_trades: int
    high_price_markets: int
    high_price_outcomes: int
    high_price_winning_outcomes: int
    high_price_losing_outcomes: int
    high_price_unresolved_outcomes: int
    losing_samples: tuple[LosingHighPriceSample, ...]
    threshold: SurvivorPowerThreshold

    @property
    def threshold_met(self) -> bool:
        return (
            self.closed_markets_parsed >= self.threshold.min_closed_markets
            and self.markets_with_trades >= self.threshold.min_markets_with_trades
        )

    @property
    def verdict(self) -> str:
        if self.losing_samples:
            return "SURVIVOR_GATE_SATISFIED"
        if self.threshold_met:
            return "TAIL_SAMPLE_RARE_OR_UNREACHABLE"
        return "STOP_INSUFFICIENT_COVERAGE"


class PolymarketDataApiClient:
    def __init__(
        self,
        *,
        data_api_base_url: str = DATA_API_BASE_URL,
        gamma_api_base_url: str = GAMMA_API_BASE_URL,
        clob_api_base_url: str = CLOB_API_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.data_api_base_url = data_api_base_url.rstrip("/")
        self.gamma_api_base_url = gamma_api_base_url.rstrip("/")
        self.clob_api_base_url = clob_api_base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def get_closed_markets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order: str = "closedTime",
        ascending: bool = False,
        closed: bool = True,
    ) -> list[dict[str, Any]]:
        params = {
            "closed": str(closed).lower(),
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        data = self._get_json(f"{self.gamma_api_base_url}/markets?{urlencode(params)}")
        if not isinstance(data, list):
            raise ValueError("unexpected Gamma closed markets response")
        return [row for row in data if isinstance(row, dict)]

    def iter_closed_markets(
        self,
        *,
        limit: int = 500,
        max_markets: int = 5_000,
        sleep_seconds: float = 0.0,
        order: str = "closedTime",
        ascending: bool = False,
        closed: bool = True,
    ) -> Iterable[dict[str, Any]]:
        offset = 0
        yielded = 0
        while yielded < max_markets:
            page_limit = min(limit, max_markets - yielded)
            page = self.get_closed_markets(
                limit=page_limit,
                offset=offset,
                order=order,
                ascending=ascending,
                closed=closed,
            )
            if not page:
                break
            for row in page:
                yield row
                yielded += 1
                if yielded >= max_markets:
                    break
            if len(page) < page_limit:
                break
            offset += page_limit
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    def get_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order: str = "endDate",
        ascending: bool = True,
        closed: bool = False,
        active: bool = True,
        tag_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag_slug is not None:
            params["tag_slug"] = tag_slug
        data = self._get_json(f"{self.gamma_api_base_url}/events?{urlencode(params)}")
        if not isinstance(data, list):
            raise ValueError("unexpected Gamma events response")
        return [row for row in data if isinstance(row, dict)]

    def iter_events(
        self,
        *,
        limit: int = 100,
        max_events: int = 500,
        sleep_seconds: float = 0.0,
        order: str = "endDate",
        ascending: bool = True,
        closed: bool = False,
        active: bool = True,
        tag_slug: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        offset = 0
        yielded = 0
        while yielded < max_events:
            page_limit = min(limit, max_events - yielded)
            page = self.get_events(
                limit=page_limit,
                offset=offset,
                order=order,
                ascending=ascending,
                closed=closed,
                active=active,
                tag_slug=tag_slug,
            )
            if not page:
                break
            for row in page:
                yield row
                yielded += 1
                if yielded >= max_events:
                    break
            if len(page) < page_limit:
                break
            offset += page_limit
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

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

    def get_order_book(self, token_id: str) -> dict[str, Any]:
        params = {"token_id": token_id}
        data = self._get_json(f"{self.clob_api_base_url}/book?{urlencode(params)}")
        if not isinstance(data, dict):
            raise ValueError("unexpected CLOB order book response")
        return data

    def iter_trades(
        self,
        condition_id: str,
        *,
        limit: int = 500,
        max_trades: int = 10_000,
        sleep_seconds: float = 0.0,
        taker_only: bool = False,
    ) -> Iterable[dict[str, Any]]:
        offset = 0
        yielded = 0
        while yielded < max_trades:
            page_limit = min(limit, max_trades - yielded)
            page = self.get_trades(
                condition_id,
                limit=page_limit,
                offset=offset,
                taker_only=taker_only,
            )
            if not page:
                break
            for row in page:
                yield row
                yielded += 1
                if yielded >= max_trades:
                    break
            if len(page) < page_limit:
                break
            offset += page_limit
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

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


def analyze_survivor_power_coverage(
    raw_markets: Iterable[Mapping[str, Any]],
    trades_by_condition: Mapping[str, Iterable[PolymarketTrade]],
    *,
    threshold: SurvivorPowerThreshold,
    lower: Decimal = Decimal("0.95"),
    upper: Decimal = Decimal("0.99"),
    losing_price_threshold: Decimal = Decimal("0.001"),
    winning_price_threshold: Decimal = Decimal("0.999"),
) -> SurvivorPowerCoverage:
    scanned = 0
    parsed_markets: list[PolymarketClosedMarket] = []
    markets_with_trades = 0
    high_price_market_ids: set[str] = set()
    high_price_outcomes: set[tuple[str, int]] = set()
    winning_outcomes: set[tuple[str, int]] = set()
    losing_outcomes: set[tuple[str, int]] = set()
    unresolved_outcomes: set[tuple[str, int]] = set()
    for raw_market in raw_markets:
        scanned += 1
        market = parse_closed_market(raw_market)
        if market is None:
            continue
        parsed_markets.append(market)
        trades = list(trades_by_condition.get(market.condition_id, ()))
        if trades:
            markets_with_trades += 1
        for trade in trades:
            if trade.outcome_index < 0 or trade.outcome_index >= len(market.outcome_prices):
                continue
            if lower <= trade.price <= upper:
                key = (market.condition_id, trade.outcome_index)
                high_price_market_ids.add(market.condition_id)
                high_price_outcomes.add(key)
                final_price = market.outcome_prices[trade.outcome_index]
                if final_price <= losing_price_threshold:
                    losing_outcomes.add(key)
                elif final_price >= winning_price_threshold:
                    winning_outcomes.add(key)
                else:
                    unresolved_outcomes.add(key)
    losing_samples = find_losing_high_price_samples(
        parsed_markets,
        trades_by_condition,
        lower=lower,
        upper=upper,
    )
    return SurvivorPowerCoverage(
        closed_markets_scanned=scanned,
        closed_markets_parsed=len(parsed_markets),
        markets_with_trades=markets_with_trades,
        high_price_markets=len(high_price_market_ids),
        high_price_outcomes=len(high_price_outcomes),
        high_price_winning_outcomes=len(winning_outcomes),
        high_price_losing_outcomes=len(losing_outcomes),
        high_price_unresolved_outcomes=len(unresolved_outcomes),
        losing_samples=tuple(losing_samples),
        threshold=threshold,
    )


def survivor_power_coverage_to_dict(coverage: SurvivorPowerCoverage) -> dict[str, Any]:
    return {
        "closed_markets_scanned": coverage.closed_markets_scanned,
        "closed_markets_parsed": coverage.closed_markets_parsed,
        "markets_with_trades": coverage.markets_with_trades,
        "high_price_markets": coverage.high_price_markets,
        "high_price_outcomes": coverage.high_price_outcomes,
        "high_price_winning_outcomes": coverage.high_price_winning_outcomes,
        "high_price_losing_outcomes": coverage.high_price_losing_outcomes,
        "high_price_unresolved_outcomes": coverage.high_price_unresolved_outcomes,
        "losing_samples": [
            {
                "condition_id": sample.condition_id,
                "slug": sample.slug,
                "title": sample.title,
                "losing_outcome": sample.losing_outcome,
                "losing_outcome_index": sample.losing_outcome_index,
                "decision_timestamp": sample.decision_timestamp,
                "decision_price": str(sample.decision_price),
                "transaction_hash": sample.transaction_hash,
            }
            for sample in coverage.losing_samples
        ],
        "threshold": {
            "min_closed_markets": coverage.threshold.min_closed_markets,
            "min_markets_with_trades": coverage.threshold.min_markets_with_trades,
            "target_closed_window_days": coverage.threshold.target_closed_window_days,
            "met": coverage.threshold_met,
        },
        "verdict": coverage.verdict,
        "zero_risk_claim": "explicitly_rejected",
        "risk_classification": "tail_risk_selling_not_arbitrage",
    }


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
