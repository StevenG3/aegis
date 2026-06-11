from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean
from typing import Any

from aegis.polymarket_onchain import (
    PolymarketClosedMarket,
    PolymarketTrade,
    last_trade_at_or_before,
)


@dataclass(frozen=True)
class PolymarketTailCostConfig:
    """Execution assumptions for a read-only research backtest."""

    notional_usd: Decimal = Decimal("100")
    fee_coefficient: Decimal = Decimal("0.05")
    slippage_bps: Decimal = Decimal("10")
    gas_usd_per_entry: Decimal = Decimal("0.02")


@dataclass(frozen=True)
class PolymarketTailPosition:
    condition_id: str
    slug: str
    title: str
    outcome: str
    outcome_index: int
    decision_timestamp: int
    decision_price: Decimal
    settlement_price: Decimal
    gross_return: Decimal
    net_return: Decimal
    cost_return: Decimal
    binary_hold: bool
    cluster_key: str
    settlement_day: str | None
    transaction_hash: str | None

    @property
    def is_loss(self) -> bool:
        return self.settlement_price <= Decimal("0.001")


def build_tail_positions(
    markets: Iterable[PolymarketClosedMarket],
    trades_by_condition: Mapping[str, Iterable[PolymarketTrade]],
    *,
    lower: Decimal = Decimal("0.95"),
    upper: Decimal = Decimal("0.985"),
    cost_config: PolymarketTailCostConfig | None = None,
) -> list[PolymarketTailPosition]:
    """Build one no-lookahead entry per high-price outcome.

    The selector uses only trades at or before the decision timestamp. Final
    settlement prices are used only after selection to compute PnL.
    """

    costs = cost_config or PolymarketTailCostConfig()
    positions: list[PolymarketTailPosition] = []
    for market in markets:
        trades = sorted(
            list(trades_by_condition.get(market.condition_id, ())),
            key=lambda trade: trade.timestamp,
        )
        if not trades:
            continue
        for outcome_index, settlement_price in enumerate(market.outcome_prices):
            first_high_price = next(
                (
                    trade
                    for trade in trades
                    if trade.outcome_index == outcome_index and lower <= trade.price <= upper
                ),
                None,
            )
            if first_high_price is None:
                continue
            decision = last_trade_at_or_before(
                trades,
                outcome_index=outcome_index,
                decision_timestamp=first_high_price.timestamp,
            )
            if decision is None or not lower <= decision.price <= upper:
                continue
            gross_return = (settlement_price - decision.price) / decision.price
            cost_return = estimate_cost_return(decision.price, costs)
            net_return = gross_return - cost_return
            positions.append(
                PolymarketTailPosition(
                    condition_id=market.condition_id,
                    slug=market.slug,
                    title=market.title,
                    outcome=market.outcomes[outcome_index],
                    outcome_index=outcome_index,
                    decision_timestamp=decision.timestamp,
                    decision_price=decision.price,
                    settlement_price=settlement_price,
                    gross_return=gross_return,
                    net_return=net_return,
                    cost_return=cost_return,
                    binary_hold=True,
                    cluster_key=derive_cluster_key(market),
                    settlement_day=_settlement_day(market),
                    transaction_hash=decision.transaction_hash,
                )
            )
    return positions


def estimate_cost_return(price: Decimal, config: PolymarketTailCostConfig) -> Decimal:
    contracts = config.notional_usd / price
    fee_usd = config.fee_coefficient * contracts * price * (Decimal("1") - price)
    slippage_usd = config.notional_usd * (config.slippage_bps / Decimal("10000"))
    total_cost_usd = fee_usd + slippage_usd + config.gas_usd_per_entry
    return total_cost_usd / config.notional_usd


def summarize_tail_backtest(
    positions: Sequence[PolymarketTailPosition],
    *,
    risk_free_return_per_trade: Decimal = Decimal("0.0002"),
    bootstrap_iterations: int = 2_000,
    random_seed: int = 42,
    min_independent_loss_events_for_edge: int = 50,
) -> dict[str, Any]:
    ordered = sorted(positions, key=lambda position: position.decision_timestamp)
    returns = [float(position.net_return) for position in ordered]
    gross_returns = [float(position.gross_return) for position in ordered]
    cost_returns = [float(position.cost_return) for position in ordered]
    losses = [position for position in ordered if position.is_loss]
    wins = [position for position in ordered if not position.is_loss]
    independent_loss_events = _independent_loss_events(losses)
    net_mean = mean(returns) if returns else 0.0
    risk_free = float(risk_free_return_per_trade)
    ci = bootstrap_mean_ci(
        returns,
        iterations=bootstrap_iterations,
        seed=random_seed,
    )
    cluster_payload = cluster_loss_events(losses)
    verdict = _tail_verdict(
        returns=returns,
        loss_count=len(losses),
        independent_loss_events=independent_loss_events,
        net_mean=net_mean,
        risk_free=risk_free,
        ci_low=float(ci["p05"]),
        min_independent_loss_events_for_edge=min_independent_loss_events_for_edge,
    )
    return {
        "verdict": verdict,
        "zero_risk_claim": "explicitly_rejected",
        "risk_classification": "short-duration_tail_risk_selling_not_arbitrage",
        "sample": {
            "positions": len(ordered),
            "wins": len(wins),
            "losses": len(losses),
            "loss_rate": len(losses) / len(ordered) if ordered else 0.0,
            "independent_loss_events": independent_loss_events,
        },
        "returns": {
            "mean_net_return": net_mean,
            "mean_gross_return": mean(gross_returns) if gross_returns else 0.0,
            "mean_cost_return": mean(cost_returns) if cost_returns else 0.0,
            "cumulative_net_return_sum": sum(returns),
            "risk_free_return_per_trade": risk_free,
            "excess_vs_risk_free_per_trade": net_mean - risk_free,
            "bootstrap_mean_net_return_ci": ci,
        },
        "tail_metrics": {
            "win_rate": len(wins) / len(ordered) if ordered else 0.0,
            "average_win": _average([float(position.net_return) for position in wins]),
            "average_loss": _average([float(position.net_return) for position in losses]),
            "payoff_ratio": _payoff_ratio(wins, losses),
            "max_single_loss": min(returns) if returns else 0.0,
            "wins_needed_to_offset_average_loss": _wins_needed_to_offset_average_loss(
                wins,
                losses,
            ),
            "wins_needed_reference_by_entry_price": {
                "0.95": 19,
                "0.9697": 32,
                "0.98": 49,
                "0.99": 99,
            },
            "left_tail": {
                "p01": percentile(returns, 0.01),
                "p05": percentile(returns, 0.05),
                "p10": percentile(returns, 0.10),
            },
            "max_drawdown": max_drawdown(returns),
            "cvar_95": cvar_left_tail(returns, 0.05),
            "cvar_99": cvar_left_tail(returns, 0.01),
            "kelly_fraction_gaussian": kelly_fraction(returns),
            "risk_of_ruin_proxy": risk_of_ruin_proxy(returns),
        },
        "costs": {
            "funding": "not_applicable",
            "fee_model": "fee_coefficient_x_contracts_x_price_x_1_minus_price",
            "gas_and_slippage_included": True,
            "binary_hold_to_settlement": True,
        },
        "clustering": cluster_payload,
        "benchmarks": {
            "risk_free_carry": {
                "return_per_trade": risk_free,
                "sample_sum": risk_free * len(ordered),
            },
            "unfiltered_all_0_95_to_0_985": {
                "positions": len(ordered),
                "mean_net_return": net_mean,
                "note": (
                    "Primary sample is the unfiltered high-price survivor-safe sample; "
                    "no subjective qualitative filter was applied in this layer."
                ),
            },
        },
        "walk_forward": walk_forward_summary(ordered, risk_free_return_per_trade),
        "ready_conditions": {
            "full_trade_archive_needed": True,
            "bounded_trade_limit_per_market_is_not_enough_for_robust_edge": True,
            "minimum_independent_loss_events_for_edge": min_independent_loss_events_for_edge,
        },
    }


def positions_to_dict(positions: Sequence[PolymarketTailPosition]) -> list[dict[str, Any]]:
    return [
        {
            "condition_id": position.condition_id,
            "slug": position.slug,
            "title": position.title,
            "outcome": position.outcome,
            "outcome_index": position.outcome_index,
            "decision_timestamp": position.decision_timestamp,
            "decision_time_utc": _utc_from_timestamp(position.decision_timestamp),
            "decision_price": str(position.decision_price),
            "settlement_price": str(position.settlement_price),
            "gross_return": str(position.gross_return),
            "net_return": str(position.net_return),
            "cost_return": str(position.cost_return),
            "binary_hold": position.binary_hold,
            "cluster_key": position.cluster_key,
            "settlement_day": position.settlement_day,
            "transaction_hash": position.transaction_hash,
            "is_loss": position.is_loss,
        }
        for position in positions
    ]


def cluster_loss_events(positions: Sequence[PolymarketTailPosition]) -> dict[str, Any]:
    by_cluster: dict[str, int] = defaultdict(int)
    by_day: dict[str, int] = defaultdict(int)
    by_cluster_day: dict[str, int] = defaultdict(int)
    for position in positions:
        by_cluster[position.cluster_key] += 1
        day = position.settlement_day or "unknown"
        by_day[day] += 1
        by_cluster_day[f"{position.cluster_key}|{day}"] += 1
    return {
        "losses_by_cluster": dict(sorted(by_cluster.items())),
        "losses_by_settlement_day": dict(sorted(by_day.items())),
        "losses_by_cluster_day": dict(sorted(by_cluster_day.items())),
        "effective_independent_loss_events": len(by_cluster_day),
        "note": "Heuristic clustering by slug/title theme plus settlement day.",
    }


def walk_forward_summary(
    positions: Sequence[PolymarketTailPosition],
    risk_free_return_per_trade: Decimal,
) -> dict[str, Any]:
    if len(positions) < 30:
        return {"status": "INSUFFICIENT", "reason": "fewer than 30 positions"}
    midpoint = len(positions) // 2
    insample = positions[:midpoint]
    outsample = positions[midpoint:]
    return {
        "status": "COMPUTED",
        "split": "chronological_50_50_by_decision_timestamp",
        "insample": _slice_stats(insample, risk_free_return_per_trade),
        "outsample": _slice_stats(outsample, risk_free_return_per_trade),
        "note": "Small loss count means this is a stability diagnostic, not proof of edge.",
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    if not values:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "iterations": iterations}
    rng = random.Random(seed)
    sample_size = len(values)
    boot = [
        mean(rng.choice(values) for _ in range(sample_size))
        for _ in range(max(1, iterations))
    ]
    boot.sort()
    return {
        "p05": percentile(boot, 0.05),
        "p50": percentile(boot, 0.50),
        "p95": percentile(boot, 0.95),
        "iterations": iterations,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(q * (len(ordered) - 1)))))
    return ordered[index]


def cvar_left_tail(values: Sequence[float], tail_probability: float) -> float:
    if not values:
        return 0.0
    threshold = percentile(values, tail_probability)
    tail = [value for value in values if value <= threshold]
    return mean(tail) if tail else threshold


def max_drawdown(returns: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def kelly_fraction(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = mean((value - avg) ** 2 for value in values)
    if variance <= 0:
        return 0.0
    return max(0.0, min(1.0, avg / variance))


def risk_of_ruin_proxy(values: Sequence[float], ruin_drawdown: float = -1.0) -> float:
    if not values:
        return 0.0
    paths = 250
    rng = random.Random(7)
    ruined = 0
    for _ in range(paths):
        equity = 0.0
        peak = 0.0
        for _ in range(len(values)):
            equity += rng.choice(values)
            peak = max(peak, equity)
            if equity - peak <= ruin_drawdown:
                ruined += 1
                break
    return ruined / paths


def derive_cluster_key(market: PolymarketClosedMarket) -> str:
    text = f"{market.slug} {market.title}".lower()
    if any(token in text for token in ("nba", "nfl", "nhl", "mlb", "tennis", "itf", "ufc")):
        return "sports"
    if any(token in text for token in ("bitcoin", "btc", "ethereum", "eth", "solana", "crypto")):
        return "crypto"
    if any(token in text for token in ("election", "trump", "biden", "president", "senate")):
        return "politics"
    if any(token in text for token in ("fed", "cpi", "inflation", "rate", "gdp")):
        return "macro"
    if any(token in text for token in ("weather", "temperature", "hurricane")):
        return "weather"
    return "other"


def _slice_stats(
    positions: Sequence[PolymarketTailPosition],
    risk_free_return_per_trade: Decimal,
) -> dict[str, Any]:
    values = [float(position.net_return) for position in positions]
    losses = [position for position in positions if position.is_loss]
    avg = mean(values) if values else 0.0
    risk_free = float(risk_free_return_per_trade)
    return {
        "positions": len(positions),
        "losses": len(losses),
        "mean_net_return": avg,
        "excess_vs_risk_free_per_trade": avg - risk_free,
    }


def _tail_verdict(
    *,
    returns: Sequence[float],
    loss_count: int,
    independent_loss_events: int,
    net_mean: float,
    risk_free: float,
    ci_low: float,
    min_independent_loss_events_for_edge: int,
) -> str:
    if not returns or loss_count == 0:
        return "INSUFFICIENT"
    if net_mean <= risk_free:
        return "NO_ROBUST_EDGE"
    if independent_loss_events < min_independent_loss_events_for_edge:
        return "CARRY_POSITIVE_TAIL_UNDER_SAMPLED"
    if ci_low <= risk_free:
        return "CARRY_POSITIVE_TAIL_UNDER_SAMPLED"
    return "CARRY_POSITIVE_TAIL_UNDER_SAMPLED"


def _independent_loss_events(positions: Sequence[PolymarketTailPosition]) -> int:
    return len({(position.cluster_key, position.settlement_day) for position in positions})


def _average(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _payoff_ratio(
    wins: Sequence[PolymarketTailPosition],
    losses: Sequence[PolymarketTailPosition],
) -> float:
    avg_win = _average([float(position.net_return) for position in wins])
    avg_loss = abs(_average([float(position.net_return) for position in losses]))
    if avg_loss == 0:
        return 0.0
    return avg_win / avg_loss


def _wins_needed_to_offset_average_loss(
    wins: Sequence[PolymarketTailPosition],
    losses: Sequence[PolymarketTailPosition],
) -> int | None:
    avg_win = _average([float(position.net_return) for position in wins])
    avg_loss = abs(_average([float(position.net_return) for position in losses]))
    if avg_win <= 0 or avg_loss <= 0:
        return None
    return math.ceil(avg_loss / avg_win)


def _settlement_day(market: PolymarketClosedMarket) -> str | None:
    text = market.closed_time or market.end_time
    if not text:
        return None
    return text[:10]


def _utc_from_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()  # noqa: UP017
