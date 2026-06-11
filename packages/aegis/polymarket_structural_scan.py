from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from aegis.polymarket_onchain import PolymarketClosedMarket, parse_closed_market


@dataclass(frozen=True)
class StructuralCostConfig:
    fee_rate: Decimal = Decimal("0")
    gas_usdc: Decimal = Decimal("0.02")
    min_trade_size: Decimal = Decimal("5")
    min_net_edge: Decimal = Decimal("0.001")


@dataclass(frozen=True)
class OrderBookLevel:
    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class ExecutablePrice:
    average_price: Decimal
    size: Decimal
    levels_consumed: int


@dataclass(frozen=True)
class StructuralCandidate:
    kind: Literal["negrisk_yes", "negrisk_no", "logic_subset"]
    group_id: str
    market_slugs: tuple[str, ...]
    legs: tuple[dict[str, str], ...]
    target_size: Decimal
    payout_floor: Decimal
    gross_cost: Decimal
    fee_cost: Decimal
    gas_cost: Decimal
    net_edge: Decimal
    executable: bool
    assumptions_verified: bool
    caveat: str


@dataclass(frozen=True)
class StructuralScanResult:
    neg_risk_groups_scanned: int
    neg_risk_groups_with_books: int
    neg_risk_candidates: tuple[StructuralCandidate, ...]
    logic_pairs_evaluated: int
    logic_candidates: tuple[StructuralCandidate, ...]
    orderbook_errors: tuple[str, ...]

    @property
    def executable_positive_candidates(self) -> tuple[StructuralCandidate, ...]:
        return tuple(
            candidate
            for candidate in (*self.neg_risk_candidates, *self.logic_candidates)
            if candidate.executable and candidate.net_edge > Decimal("0")
        )

    @property
    def verdict(self) -> str:
        if self.executable_positive_candidates:
            return "STRUCTURAL_EDGE_FOUND"
        if self.neg_risk_groups_scanned == 0 and self.logic_pairs_evaluated == 0:
            return "STRUCTURAL_EDGE_RARE"
        if self.orderbook_errors and self.neg_risk_groups_with_books == 0:
            return "STRUCTURAL_EDGE_RARE"
        return "NO_STRUCTURAL_EDGE"


def parse_order_book_levels(raw_levels: object) -> tuple[OrderBookLevel, ...]:
    if not isinstance(raw_levels, list):
        return ()
    levels: list[OrderBookLevel] = []
    for raw in raw_levels:
        if not isinstance(raw, Mapping):
            continue
        try:
            price = Decimal(str(raw["price"]))
            size = Decimal(str(raw["size"]))
        except Exception:
            continue
        if price >= 0 and size > 0:
            levels.append(OrderBookLevel(price=price, size=size))
    return tuple(sorted(levels, key=lambda level: level.price))


def executable_buy_price(
    asks: Iterable[OrderBookLevel], target_size: Decimal
) -> ExecutablePrice | None:
    if target_size <= 0:
        return None
    remaining = target_size
    notional = Decimal("0")
    consumed = 0
    for level in asks:
        take = min(remaining, level.size)
        if take <= 0:
            continue
        notional += take * level.price
        remaining -= take
        consumed += 1
        if remaining <= 0:
            break
    if remaining > 0:
        return None
    return ExecutablePrice(
        average_price=notional / target_size,
        size=target_size,
        levels_consumed=consumed,
    )


def clob_token_ids(raw_market: Mapping[str, Any]) -> tuple[str, ...]:
    raw = raw_market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
    else:
        parsed = raw
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if str(item))


def scan_neg_risk_groups(
    raw_markets: Iterable[Mapping[str, Any]],
    books_by_token: Mapping[str, Mapping[str, Any]],
    *,
    costs: StructuralCostConfig,
    target_size: Decimal,
    known_complete_group_ids: set[str] | None = None,
) -> tuple[int, int, tuple[StructuralCandidate, ...]]:
    complete_group_ids = known_complete_group_ids or set()
    groups: dict[str, list[tuple[Mapping[str, Any], PolymarketClosedMarket, tuple[str, ...]]]] = (
        defaultdict(list)
    )
    for raw in raw_markets:
        if raw.get("closed") is True or raw.get("active") is False:
            continue
        if raw.get("negRisk") is not True:
            continue
        group_id = str(raw.get("negRiskMarketID") or "")
        market = parse_closed_market(raw)
        token_ids = clob_token_ids(raw)
        if not group_id or market is None or len(token_ids) != len(market.outcomes):
            continue
        groups[group_id].append((raw, market, token_ids))

    candidates: list[StructuralCandidate] = []
    groups_with_books = 0
    for group_id, rows in groups.items():
        if len(rows) < 2:
            continue
        yes_legs: list[tuple[PolymarketClosedMarket, str, ExecutablePrice]] = []
        no_legs: list[tuple[PolymarketClosedMarket, str, ExecutablePrice]] = []
        for _raw, market, token_ids in rows:
            outcome_to_token = dict(zip(market.outcomes, token_ids, strict=False))
            for outcome_name, collector in (("Yes", yes_legs), ("No", no_legs)):
                token_id = outcome_to_token.get(outcome_name)
                if token_id is None:
                    continue
                book = books_by_token.get(token_id)
                executable = executable_buy_price(
                    parse_order_book_levels(book.get("asks") if book else None),
                    target_size,
                )
                if executable is not None:
                    collector.append((market, token_id, executable))
        if yes_legs or no_legs:
            groups_with_books += 1
        if len(yes_legs) == len(rows):
            candidates.append(
                _build_candidate(
                    kind="negrisk_yes",
                    group_id=group_id,
                    legs=yes_legs,
                    payout_floor=Decimal("1"),
                    costs=costs,
                    assumptions_verified=group_id in complete_group_ids,
                    caveat=(
                        "Assumes the negRisk group is exhaustive and exactly one outcome "
                        "resolves Yes."
                    ),
                )
            )
        if len(no_legs) == len(rows):
            payout_floor = Decimal(max(len(rows) - 1, 0))
            candidates.append(
                _build_candidate(
                    kind="negrisk_no",
                    group_id=group_id,
                    legs=no_legs,
                    payout_floor=payout_floor,
                    costs=costs,
                    assumptions_verified=group_id in complete_group_ids,
                    caveat=(
                        "Assumes the negRisk group is mutually exclusive and exactly one "
                        "outcome resolves Yes."
                    ),
                )
            )
    return len([rows for rows in groups.values() if len(rows) >= 2]), groups_with_books, tuple(
        candidates
    )


def scan_logic_subset_pairs(
    raw_markets: Iterable[Mapping[str, Any]],
    books_by_token: Mapping[str, Mapping[str, Any]],
    *,
    costs: StructuralCostConfig,
    target_size: Decimal,
    known_subset_pairs: set[tuple[str, str]] | None = None,
) -> tuple[int, tuple[StructuralCandidate, ...]]:
    verified_subset_pairs = known_subset_pairs or set()
    parsed: list[tuple[Mapping[str, Any], PolymarketClosedMarket, tuple[str, ...], str]] = []
    for raw in raw_markets:
        if raw.get("closed") is True or raw.get("active") is False:
            continue
        market = parse_closed_market(raw)
        token_ids = clob_token_ids(raw)
        if market is None or tuple(market.outcomes) != ("Yes", "No") or len(token_ids) != 2:
            continue
        parsed.append((raw, market, token_ids, _normalize_question(market.title)))

    evaluated = 0
    candidates: list[StructuralCandidate] = []
    for superset_raw, superset, superset_tokens, superset_norm in parsed:
        for subset_raw, subset, subset_tokens, subset_norm in parsed:
            if superset.condition_id == subset.condition_id:
                continue
            if not _looks_like_subset_pair(superset_norm, subset_norm):
                continue
            evaluated += 1
            superset_yes = executable_buy_price(
                parse_order_book_levels(books_by_token.get(superset_tokens[0], {}).get("asks")),
                target_size,
            )
            subset_no = executable_buy_price(
                parse_order_book_levels(books_by_token.get(subset_tokens[1], {}).get("asks")),
                target_size,
            )
            if superset_yes is None or subset_no is None:
                continue
            candidates.append(
                _build_candidate(
                    kind="logic_subset",
                    group_id=f"{superset.condition_id}:{subset.condition_id}",
                    legs=[
                        (superset, str(superset_tokens[0]), superset_yes),
                        (subset, str(subset_tokens[1]), subset_no),
                    ],
                    payout_floor=Decimal("1"),
                    costs=costs,
                    assumptions_verified=(
                        superset.condition_id,
                        subset.condition_id,
                    )
                    in verified_subset_pairs,
                    caveat=(
                        "Assumes the second market is a strict logical subset of the first; "
                        "heuristic title matching must be reviewed before any action."
                    ),
                )
            )
            _ = superset_raw, subset_raw
    return evaluated, tuple(candidates)


def structural_scan_to_dict(result: StructuralScanResult) -> dict[str, Any]:
    return {
        "verdict": result.verdict,
        "neg_risk_groups_scanned": result.neg_risk_groups_scanned,
        "neg_risk_groups_with_books": result.neg_risk_groups_with_books,
        "neg_risk_candidates": [
            _candidate_to_dict(candidate) for candidate in result.neg_risk_candidates
        ],
        "logic_pairs_evaluated": result.logic_pairs_evaluated,
        "logic_candidates": [
            _candidate_to_dict(candidate) for candidate in result.logic_candidates
        ],
        "executable_positive_candidates": [
            _candidate_to_dict(candidate) for candidate in result.executable_positive_candidates
        ],
        "orderbook_errors": list(result.orderbook_errors),
        "zero_risk_claim": "explicitly_rejected",
        "wallet_order_funds_connected": False,
    }


def _build_candidate(
    *,
    kind: Literal["negrisk_yes", "negrisk_no", "logic_subset"],
    group_id: str,
    legs: Iterable[tuple[PolymarketClosedMarket, str, ExecutablePrice]],
    payout_floor: Decimal,
    costs: StructuralCostConfig,
    assumptions_verified: bool,
    caveat: str,
) -> StructuralCandidate:
    leg_rows = tuple(legs)
    gross_cost = sum((leg[2].average_price for leg in leg_rows), Decimal("0"))
    fee_cost = gross_cost * costs.fee_rate
    gas_cost = costs.gas_usdc / leg_rows[0][2].size if leg_rows else Decimal("0")
    net_edge = payout_floor - gross_cost - fee_cost - gas_cost
    executable = bool(leg_rows) and assumptions_verified and net_edge >= costs.min_net_edge
    return StructuralCandidate(
        kind=kind,
        group_id=group_id,
        market_slugs=tuple(leg[0].slug for leg in leg_rows),
        legs=tuple(
            {
                "condition_id": leg[0].condition_id,
                "slug": leg[0].slug,
                "title": leg[0].title,
                "token_id": leg[1],
                "average_price": str(leg[2].average_price),
                "target_size": str(leg[2].size),
                "levels_consumed": str(leg[2].levels_consumed),
            }
            for leg in leg_rows
        ),
        target_size=leg_rows[0][2].size if leg_rows else Decimal("0"),
        payout_floor=payout_floor,
        gross_cost=gross_cost,
        fee_cost=fee_cost,
        gas_cost=gas_cost,
        net_edge=net_edge,
        executable=executable,
        assumptions_verified=assumptions_verified,
        caveat=caveat,
    )


def _candidate_to_dict(candidate: StructuralCandidate) -> dict[str, Any]:
    return {
        "kind": candidate.kind,
        "group_id": candidate.group_id,
        "market_slugs": list(candidate.market_slugs),
        "legs": list(candidate.legs),
        "target_size": str(candidate.target_size),
        "payout_floor": str(candidate.payout_floor),
        "gross_cost": str(candidate.gross_cost),
        "fee_cost": str(candidate.fee_cost),
        "gas_cost": str(candidate.gas_cost),
        "net_edge": str(candidate.net_edge),
        "executable": candidate.executable,
        "assumptions_verified": candidate.assumptions_verified,
        "caveat": candidate.caveat,
    }


def _normalize_question(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(
        r"\b(will|the|a|an|by|on|in|before|after|during|202[0-9])\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _looks_like_subset_pair(superset: str, subset: str) -> bool:
    if not superset or not subset or superset == subset:
        return False
    superset_terms = set(superset.split())
    subset_terms = set(subset.split())
    if len(superset_terms) < 3 or len(subset_terms) < 4:
        return False
    overlap = len(superset_terms & subset_terms)
    return overlap >= max(3, int(len(superset_terms) * 0.8)) and len(subset_terms) > len(
        superset_terms
    )
