from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aegis.backtest_core import benjamini_hochberg, pbo, sign_test_p_value

Direction = Literal["Up", "Down"]
ExitMode = Literal["preclose_30s", "settlement"]

MOVE_THRESHOLDS_USD = (50.0, 70.0, 100.0, 150.0)
ENTRY_WINDOWS_SECONDS = ((90, 60), (60, 30))
ASK_BANDS = ((0.80, 0.95), (0.85, 0.99))
LATENCIES_MS = (200, 500)
EXIT_MODES: tuple[ExitMode, ...] = ("preclose_30s", "settlement")
CHAINLINK_BTC_DATA_STREAM_URL = "https://data.chain.link/streams/btc-usd"
CHAINLINK_DATA_STREAM_SOURCE_TYPE = "chainlink_data_streams_not_aggregator_v3"
CHAINLINK_RTDS_SOURCE = "polymarket_rtds_crypto_prices_chainlink"
TRIAL_COUNT_N = (
    len(MOVE_THRESHOLDS_USD)
    * len(ENTRY_WINDOWS_SECONDS)
    * len(ASK_BANDS)
    * len(LATENCIES_MS)
    * len(EXIT_MODES)
)


@dataclass(frozen=True)
class ForwardExecutionConfig:
    notional_usd: float = 25.0
    min_markets: int = 100
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    venue_geoblocked: bool = True


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class ForwardSnapshot:
    slug: str
    condition_id: str
    outcome: Direction
    timestamp_ms: int
    seconds_to_close: float
    start_ts: int
    end_ts: int
    chainlink_start_price: float | None
    chainlink_reference_price: float | None
    best_ask: float | None
    best_bid: float | None
    ask_levels: tuple[BookLevel, ...]
    bid_levels: tuple[BookLevel, ...]


@dataclass(frozen=True)
class ForwardMarket:
    slug: str
    condition_id: str
    start_ts: int
    end_ts: int
    settlement_direction: Direction | None
    snapshots: tuple[ForwardSnapshot, ...]


@dataclass(frozen=True)
class ForwardCandidate:
    name: str
    move_threshold_usd: float
    window_start_seconds: int
    window_end_seconds: int
    min_ask: float
    max_ask: float
    latency_ms: int
    exit_mode: ExitMode


@dataclass(frozen=True)
class Fill:
    cost: float
    shares: float
    average_price: float
    fill_ratio: float
    levels_consumed: int


@dataclass(frozen=True)
class ForwardTrade:
    slug: str
    candidate: str
    direction: Direction
    settlement_direction: Direction | None
    signal_timestamp_ms: int
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    entry_seconds_to_close: float
    exit_seconds_to_close: float
    chainlink_move_usd: float
    entry_average_price: float
    exit_average_price: float
    entry_fill_ratio: float
    exit_fill_ratio: float
    net_pnl_usd: float
    net_return: float
    exit_mode: ExitMode


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: ForwardCandidate
    trades: tuple[ForwardTrade, ...]
    p_value: float
    oos_mean_return: float


def run_forward_execution_backtest(
    rows: Sequence[Mapping[str, object]],
    *,
    config: ForwardExecutionConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = ForwardExecutionConfig()
    source_coverage = _settlement_source_coverage(rows)
    markets = parse_forward_markets(rows)
    coverage = {**_coverage(markets), **source_coverage}
    if not markets:
        return _insufficient(
            "no forward Polymarket rows with verified Chainlink Data Streams settlement source",
            coverage,
        )
    if coverage["chainlink_ready_markets"] == 0:
        return _insufficient("no Chainlink BTC reference prices in forward rows", coverage)
    if int(coverage["settled_markets"]) < config.min_markets:
        return _insufficient(
            f"settled forward markets {coverage['settled_markets']} < min_markets "
            f"{config.min_markets}",
            coverage,
        )

    candidates = _candidate_grid()
    evaluations = tuple(
        _evaluate_candidate(candidate, markets, notional_usd=config.notional_usd)
        for candidate in candidates
    )
    valid = tuple(evaluation for evaluation in evaluations if evaluation.trades)
    if not valid:
        return _insufficient("no executable forward entries passed the predeclared grid", coverage)

    p_values = [evaluation.p_value for evaluation in valid]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(valid, config.pbo_splits, markets)
    pbo_valid = bool(pbo_report.get("valid", False))
    pbo_value = _float_value(pbo_report.get("pbo"))
    if pbo_value is None:
        pbo_value = 1.0
    survivors = [
        evaluation
        for evaluation, fdr_pass in zip(valid, fdr_flags, strict=True)
        if evaluation.candidate.exit_mode == "preclose_30s"
        and fdr_pass
        and pbo_valid
        and pbo_value <= 0.20
        and _mean_return(evaluation.trades) > 0.0
        and evaluation.oos_mean_return > 0.0
    ]
    best = max(
        valid,
        key=lambda evaluation: (_mean_return(evaluation.trades), len(evaluation.trades)),
    )
    preclose_valid = tuple(
        evaluation for evaluation in valid if evaluation.candidate.exit_mode == "preclose_30s"
    )
    best_preclose = (
        max(
            preclose_valid,
            key=lambda evaluation: (_mean_return(evaluation.trades), len(evaluation.trades)),
        )
        if preclose_valid
        else None
    )

    if survivors and config.venue_geoblocked:
        verdict = "EXECUTION_VALID_PENDING_VENUE"
        reason = "preclose executable candidates survived gates, but venue geoblock is a hard stop"
    elif survivors:
        verdict = "EXECUTION_VALID_PENDING_VENUE"
        reason = "preclose executable candidates survived gates; venue/legal review still required"
    else:
        verdict = "NO_EDGE"
        reason = "no preclose executable candidate passed FDR, valid PBO, and OOS stability gates"

    return {
        "status": "OK",
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "multi-week cross-regime forward executable order-book capture with venue "
            "access resolved and non-survivor-limited coverage"
        ),
        "candidate_count_n": len(candidates),
        "raw_is_survivors": sum(
            1
            for evaluation in valid
            if evaluation.candidate.exit_mode == "preclose_30s"
            and _mean_return(evaluation.trades) > 0.0
        ),
        "fdr_is_survivors": sum(1 for flag in fdr_flags if flag),
        "coverage": coverage,
        "settlement_source": {
            "source": CHAINLINK_BTC_DATA_STREAM_URL,
            "source_type": CHAINLINK_DATA_STREAM_SOURCE_TYPE,
            "signal_source": CHAINLINK_RTDS_SOURCE,
            "fail_closed_on_missing_or_mismatched_source": True,
        },
        "standard_metrics": _candidate_metrics(best_preclose or best),
        "benchmark_metrics": {
            "benchmark": "no_trade_cash",
            "mean_return": 0.0,
            "total_pnl_usd": 0.0,
        },
        "multiple_testing": {
            "method": "BH-FDR + valid CSCV_PBO",
            "candidate_count_n": len(candidates),
            "tested_candidates": len(valid),
            "fdr_alpha": config.fdr_alpha,
            "fdr_after": sum(1 for flag in fdr_flags if flag),
            "pbo": pbo_report,
            "preclose_survivors": len(survivors),
            "settlement_is_control_only": True,
        },
        "best_candidate": _candidate_summary(best),
        "best_preclose_candidate": None
        if best_preclose is None
        else _candidate_summary(best_preclose),
        "fill_ratio_distribution": _fill_ratio_distribution(valid),
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "venue_geoblocked_hard_gate": config.venue_geoblocked,
        },
    }


def parse_forward_markets(rows: Sequence[Mapping[str, object]]) -> tuple[ForwardMarket, ...]:
    chainlink_ticks = _chainlink_ticks(rows)
    snapshots_by_slug: dict[str, list[ForwardSnapshot]] = defaultdict(list)
    settlements: dict[str, Direction] = {}
    conditions: dict[str, str] = {}
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for row in rows:
        record_type = str(row.get("record_type", "snapshot"))
        slug = _str_value(row.get("slug"))
        if slug is None:
            continue
        condition_id = _str_value(row.get("condition_id")) or ""
        if condition_id:
            conditions[slug] = condition_id
        if record_type == "settlement":
            if not _has_verified_settlement_source(row):
                continue
            direction = _direction(row.get("settlement_direction"))
            if direction is not None:
                settlements[slug] = direction
            start_ts = _int_value(row.get("start_ts"))
            end_ts = _int_value(row.get("end_ts"))
            if start_ts is not None:
                starts[slug] = start_ts
            if end_ts is not None:
                ends[slug] = end_ts
            continue
        snapshot = _snapshot_from_row(row, chainlink_ticks=chainlink_ticks)
        if snapshot is None:
            continue
        snapshots_by_slug[slug].append(snapshot)
        starts[slug] = snapshot.start_ts
        ends[slug] = snapshot.end_ts
        if snapshot.condition_id:
            conditions[slug] = snapshot.condition_id
        row_settlement = _direction(row.get("settlement_direction"))
        if row_settlement is not None:
            settlements[slug] = row_settlement

    markets: list[ForwardMarket] = []
    for slug, snapshots in snapshots_by_slug.items():
        if slug not in starts or slug not in ends:
            continue
        markets.append(
            ForwardMarket(
                slug=slug,
                condition_id=conditions.get(slug, ""),
                start_ts=starts[slug],
                end_ts=ends[slug],
                settlement_direction=settlements.get(slug),
                snapshots=tuple(sorted(snapshots, key=lambda snapshot: snapshot.timestamp_ms)),
            )
        )
    return tuple(sorted(markets, key=lambda market: (market.end_ts, market.slug)))


def _snapshot_from_row(
    row: Mapping[str, object],
    *,
    chainlink_ticks: Sequence[tuple[int, float]],
) -> ForwardSnapshot | None:
    slug = _str_value(row.get("slug"))
    condition_id = _str_value(row.get("condition_id")) or ""
    outcome = _direction(row.get("outcome"))
    timestamp_ms = _int_value(row.get("captured_ts_ms"))
    end_ts = _int_value(row.get("end_ts"))
    start_ts = _int_value(row.get("start_ts"))
    if start_ts is None and end_ts is not None:
        start_ts = _start_ts_from_slug(slug, end_ts)
    seconds_to_close = _float_value(row.get("seconds_to_close"))
    if (
        slug is None
        or outcome is None
        or timestamp_ms is None
        or start_ts is None
        or end_ts is None
        or seconds_to_close is None
    ):
        return None
    if not _has_verified_settlement_source(row):
        return None
    chainlink_start = _float_value(
        row.get("chainlink_start_price"), row.get("btc_start_price_chainlink")
    )
    chainlink_ref = _float_value(
        row.get("chainlink_reference_price"),
        row.get("btc_reference_price_chainlink"),
        row.get("chainlink_price"),
    )
    if chainlink_start is None:
        chainlink_start = _nearest_chainlink_price(
            chainlink_ticks,
            target_ts_ms=start_ts * 1000,
            tolerance_ms=10_000,
        )
    if chainlink_ref is None:
        chainlink_ref = _nearest_chainlink_price(
            chainlink_ticks,
            target_ts_ms=timestamp_ms,
            tolerance_ms=10_000,
        )
    return ForwardSnapshot(
        slug=slug,
        condition_id=condition_id,
        outcome=outcome,
        timestamp_ms=timestamp_ms,
        seconds_to_close=seconds_to_close,
        start_ts=start_ts,
        end_ts=end_ts,
        chainlink_start_price=chainlink_start,
        chainlink_reference_price=chainlink_ref,
        best_ask=_float_value(row.get("best_ask")),
        best_bid=_float_value(row.get("best_bid")),
        ask_levels=_levels(row.get("ask_levels")),
        bid_levels=_levels(row.get("bid_levels")),
    )


def _candidate_grid() -> tuple[ForwardCandidate, ...]:
    candidates: list[ForwardCandidate] = []
    for move in MOVE_THRESHOLDS_USD:
        for window_start, window_end in ENTRY_WINDOWS_SECONDS:
            for min_ask, max_ask in ASK_BANDS:
                for latency_ms in LATENCIES_MS:
                    for exit_mode in EXIT_MODES:
                        candidates.append(
                            ForwardCandidate(
                                name=(
                                    f"move{move:g}_w{window_start}-{window_end}_"
                                    f"ask{min_ask:g}-{max_ask:g}_lat{latency_ms}ms_{exit_mode}"
                                ),
                                move_threshold_usd=move,
                                window_start_seconds=window_start,
                                window_end_seconds=window_end,
                                min_ask=min_ask,
                                max_ask=max_ask,
                                latency_ms=latency_ms,
                                exit_mode=exit_mode,
                            )
                        )
    return tuple(candidates)


def _evaluate_candidate(
    candidate: ForwardCandidate,
    markets: Sequence[ForwardMarket],
    *,
    notional_usd: float,
) -> CandidateEvaluation:
    trades = tuple(
        trade
        for market in markets
        if (trade := _trade_for_market(candidate, market, notional_usd=notional_usd)) is not None
    )
    returns = tuple(trade.net_return for trade in trades)
    p_value = sign_test_p_value(returns, alternative="greater") if returns else 1.0
    oos = returns[len(returns) * 60 // 100 :]
    return CandidateEvaluation(
        candidate=candidate,
        trades=trades,
        p_value=p_value,
        oos_mean_return=statistics.fmean(oos) if oos else 0.0,
    )


def _trade_for_market(
    candidate: ForwardCandidate,
    market: ForwardMarket,
    *,
    notional_usd: float,
) -> ForwardTrade | None:
    signal = _signal_snapshot(candidate, market)
    if signal is None:
        return None
    entry = _entry_snapshot(candidate, market, signal)
    if entry is None or entry.best_ask is None:
        return None
    entry_fill = _buy_at_ask(entry.ask_levels, notional_usd)
    if entry_fill is None or entry_fill.cost <= 0.0:
        return None
    if candidate.exit_mode == "settlement":
        if market.settlement_direction is None:
            return None
        exit_timestamp_ms = market.end_ts * 1000
        exit_seconds_to_close = 0.0
        exit_average_price = 1.0 if market.settlement_direction == signal.outcome else 0.0
        exit_fill_ratio = 1.0
        proceeds = entry_fill.shares * exit_average_price
    else:
        exit_snapshot = _preclose_exit_snapshot(market, signal.outcome, entry.timestamp_ms)
        if exit_snapshot is None:
            return None
        exit_fill = _sell_at_bid(exit_snapshot.bid_levels, entry_fill.shares)
        if exit_fill is None or exit_fill.cost <= 0.0:
            return None
        exit_timestamp_ms = exit_snapshot.timestamp_ms
        exit_seconds_to_close = exit_snapshot.seconds_to_close
        exit_average_price = exit_fill.average_price
        exit_fill_ratio = exit_fill.fill_ratio
        proceeds = exit_fill.cost
    net_pnl = proceeds - entry_fill.cost
    return ForwardTrade(
        slug=market.slug,
        candidate=candidate.name,
        direction=signal.outcome,
        settlement_direction=market.settlement_direction,
        signal_timestamp_ms=signal.timestamp_ms,
        entry_timestamp_ms=entry.timestamp_ms,
        exit_timestamp_ms=exit_timestamp_ms,
        entry_seconds_to_close=entry.seconds_to_close,
        exit_seconds_to_close=exit_seconds_to_close,
        chainlink_move_usd=_chainlink_move(signal),
        entry_average_price=entry_fill.average_price,
        exit_average_price=exit_average_price,
        entry_fill_ratio=entry_fill.fill_ratio,
        exit_fill_ratio=exit_fill_ratio,
        net_pnl_usd=net_pnl,
        net_return=net_pnl / entry_fill.cost,
        exit_mode=candidate.exit_mode,
    )


def _signal_snapshot(
    candidate: ForwardCandidate, market: ForwardMarket
) -> ForwardSnapshot | None:
    for snapshot in market.snapshots:
        if snapshot.chainlink_start_price is None or snapshot.chainlink_reference_price is None:
            continue
        move = _chainlink_move(snapshot)
        direction: Direction = "Up" if move >= 0.0 else "Down"
        if snapshot.outcome != direction:
            continue
        if abs(move) < candidate.move_threshold_usd:
            continue
        if not (
            candidate.window_end_seconds
            <= snapshot.seconds_to_close
            <= candidate.window_start_seconds
        ):
            continue
        if snapshot.best_ask is None:
            continue
        if not candidate.min_ask <= snapshot.best_ask <= candidate.max_ask:
            continue
        return snapshot
    return None


def _entry_snapshot(
    candidate: ForwardCandidate,
    market: ForwardMarket,
    signal: ForwardSnapshot,
) -> ForwardSnapshot | None:
    entry_not_before = signal.timestamp_ms + candidate.latency_ms
    for snapshot in market.snapshots:
        if snapshot.outcome == signal.outcome and snapshot.timestamp_ms >= entry_not_before:
            return snapshot
    return None


def _preclose_exit_snapshot(
    market: ForwardMarket,
    outcome: Direction,
    entry_timestamp_ms: int,
) -> ForwardSnapshot | None:
    for snapshot in market.snapshots:
        if snapshot.outcome != outcome or snapshot.timestamp_ms <= entry_timestamp_ms:
            continue
        if 0.0 <= snapshot.seconds_to_close <= 30.0 and snapshot.best_bid is not None:
            return snapshot
    return None


def _buy_at_ask(levels: Sequence[BookLevel], notional_usd: float) -> Fill | None:
    return _walk_book(
        tuple(sorted(levels, key=lambda level: level.price)),
        target_cost=notional_usd,
    )


def _sell_at_bid(levels: Sequence[BookLevel], shares: float) -> Fill | None:
    sorted_levels = tuple(sorted(levels, key=lambda level: level.price, reverse=True))
    remaining = shares
    proceeds = 0.0
    filled = 0.0
    consumed = 0
    for level in sorted_levels:
        if remaining <= 1e-12:
            break
        take = min(remaining, level.size)
        if take <= 0.0:
            continue
        proceeds += take * level.price
        filled += take
        remaining -= take
        consumed += 1
    if filled <= 0.0:
        return None
    return Fill(
        cost=proceeds,
        shares=filled,
        average_price=proceeds / filled,
        fill_ratio=min(1.0, filled / shares) if shares > 0.0 else 0.0,
        levels_consumed=consumed,
    )


def _walk_book(levels: Sequence[BookLevel], *, target_cost: float) -> Fill | None:
    remaining_cost = target_cost
    cost = 0.0
    shares = 0.0
    consumed = 0
    for level in levels:
        if remaining_cost <= 1e-12:
            break
        level_cost = level.price * level.size
        take_cost = min(remaining_cost, level_cost)
        if take_cost <= 0.0:
            continue
        cost += take_cost
        shares += take_cost / level.price
        remaining_cost -= take_cost
        consumed += 1
    if shares <= 0.0:
        return None
    return Fill(
        cost=cost,
        shares=shares,
        average_price=cost / shares,
        fill_ratio=min(1.0, cost / target_cost) if target_cost > 0.0 else 0.0,
        levels_consumed=consumed,
    )


def _pbo_report(
    evaluations: Sequence[CandidateEvaluation],
    pbo_splits: int,
    markets: Sequence[ForwardMarket],
) -> Mapping[str, Any]:
    preclose = tuple(
        evaluation for evaluation in evaluations if evaluation.candidate.exit_mode == "preclose_30s"
    )
    if len(preclose) < 2:
        return {"valid": False, "reason": "PBO requires at least two preclose trials"}
    if len(markets) < pbo_splits:
        return {
            "valid": False,
            "reason": f"forward markets {len(markets)} < pbo_splits {pbo_splits}",
            "n_splits": pbo_splits,
            "observation_count": len(markets),
        }
    market_order = [market.slug for market in markets]
    matrix = []
    for evaluation in preclose:
        returns_by_slug = {trade.slug: trade.net_return for trade in evaluation.trades}
        matrix.append([returns_by_slug.get(slug, 0.0) for slug in market_order])
    try:
        report = dict(pbo(matrix, n_splits=pbo_splits))
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "n_splits": pbo_splits}
    report["valid"] = True
    return report


def _coverage(markets: Sequence[ForwardMarket]) -> Mapping[str, int]:
    settled = [market for market in markets if market.settlement_direction is not None]
    chainlink_ready = [
        market
        for market in settled
        if any(
            snapshot.chainlink_start_price is not None
            and snapshot.chainlink_reference_price is not None
            for snapshot in market.snapshots
        )
    ]
    return {
        "markets": len(markets),
        "settled_markets": len(settled),
        "chainlink_ready_markets": len(chainlink_ready),
        "snapshots": sum(len(market.snapshots) for market in markets),
    }


def _settlement_source_coverage(rows: Sequence[Mapping[str, object]]) -> Mapping[str, int]:
    verified_slugs: set[str] = set()
    missing_slugs: set[str] = set()
    mismatched_slugs: set[str] = set()
    for row in rows:
        record_type = str(row.get("record_type", "snapshot"))
        if record_type not in {"snapshot", "settlement"}:
            continue
        slug = _str_value(row.get("slug"))
        if slug is None:
            continue
        source = _str_value(row.get("actual_settlement_source"))
        if source == CHAINLINK_BTC_DATA_STREAM_URL:
            verified_slugs.add(slug)
        elif source is None:
            missing_slugs.add(slug)
        else:
            mismatched_slugs.add(slug)
    return {
        "verified_settlement_source_markets": len(verified_slugs),
        "missing_settlement_source_markets": len(missing_slugs - verified_slugs),
        "mismatched_settlement_source_markets": len(mismatched_slugs),
    }


def _candidate_metrics(evaluation: CandidateEvaluation) -> Mapping[str, Any]:
    trades = evaluation.trades
    returns = [trade.net_return for trade in trades]
    pnls = [trade.net_pnl_usd for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl < 0.0)
    return {
        "candidate": evaluation.candidate.name,
        "exit_mode": evaluation.candidate.exit_mode,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "flats": len(trades) - wins - losses,
        "win_rate": wins / len(trades) if trades else 0.0,
        "total_pnl_usd": sum(pnls),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "mean_entry_fill_ratio": statistics.fmean(
            trade.entry_fill_ratio for trade in trades
        )
        if trades
        else 0.0,
        "mean_exit_fill_ratio": statistics.fmean(trade.exit_fill_ratio for trade in trades)
        if trades
        else 0.0,
        "p_value": evaluation.p_value,
        "oos_mean_return": evaluation.oos_mean_return,
    }


def _candidate_summary(evaluation: CandidateEvaluation) -> Mapping[str, Any]:
    return {
        **_candidate_metrics(evaluation),
        "sample_trades": [_trade_to_dict(trade) for trade in evaluation.trades[:10]],
    }


def _fill_ratio_distribution(evaluations: Sequence[CandidateEvaluation]) -> Mapping[str, Any]:
    entry = [trade.entry_fill_ratio for evaluation in evaluations for trade in evaluation.trades]
    exit_ = [trade.exit_fill_ratio for evaluation in evaluations for trade in evaluation.trades]
    return {"entry": _distribution(entry), "exit": _distribution(exit_)}


def _distribution(values: Sequence[float]) -> Mapping[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p10": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p10": ordered[int((len(ordered) - 1) * 0.10)],
        "median": statistics.median(ordered),
    }


def _trade_to_dict(trade: ForwardTrade) -> Mapping[str, Any]:
    return {
        "slug": trade.slug,
        "candidate": trade.candidate,
        "direction": trade.direction,
        "settlement_direction": trade.settlement_direction,
        "signal_timestamp_ms": trade.signal_timestamp_ms,
        "entry_timestamp_ms": trade.entry_timestamp_ms,
        "exit_timestamp_ms": trade.exit_timestamp_ms,
        "entry_seconds_to_close": trade.entry_seconds_to_close,
        "exit_seconds_to_close": trade.exit_seconds_to_close,
        "chainlink_move_usd": trade.chainlink_move_usd,
        "entry_average_price": trade.entry_average_price,
        "exit_average_price": trade.exit_average_price,
        "entry_fill_ratio": trade.entry_fill_ratio,
        "exit_fill_ratio": trade.exit_fill_ratio,
        "net_pnl_usd": trade.net_pnl_usd,
        "net_return": trade.net_return,
        "exit_mode": trade.exit_mode,
    }


def _insufficient(reason: str, coverage: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": TRIAL_COUNT_N,
        "coverage": coverage,
        "multiple_testing": {
            "candidate_count_n": TRIAL_COUNT_N,
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": reason},
        },
        "standard_metrics": {},
        "benchmark_metrics": {"benchmark": "no_trade_cash"},
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }


def _chainlink_move(snapshot: ForwardSnapshot) -> float:
    if snapshot.chainlink_start_price is None or snapshot.chainlink_reference_price is None:
        return 0.0
    return snapshot.chainlink_reference_price - snapshot.chainlink_start_price


def _has_verified_settlement_source(row: Mapping[str, object]) -> bool:
    return _str_value(row.get("actual_settlement_source")) == CHAINLINK_BTC_DATA_STREAM_URL


def _mean_return(trades: Sequence[ForwardTrade]) -> float:
    return statistics.fmean(trade.net_return for trade in trades) if trades else 0.0


def _levels(value: object) -> tuple[BookLevel, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    levels: list[BookLevel] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        price = _float_value(raw.get("price"))
        size = _float_value(raw.get("size"))
        if price is None or size is None or not (0.0 < price <= 1.0) or size <= 0.0:
            continue
        levels.append(BookLevel(price=price, size=size))
    return tuple(levels)


def _chainlink_ticks(rows: Sequence[Mapping[str, object]]) -> tuple[tuple[int, float], ...]:
    ticks: list[tuple[int, float]] = []
    for row in rows:
        if row.get("record_type") != "chainlink_price":
            continue
        source = _str_value(row.get("source"))
        if source is not None and source != CHAINLINK_RTDS_SOURCE:
            continue
        symbol = row.get("symbol")
        if symbol != "btc/usd":
            continue
        price = _float_value(row.get("price"))
        price_ts_ms = _int_value(row.get("price_ts_ms"))
        if price_ts_ms is None:
            price_ts_ms = _int_value(row.get("timestamp_ms"))
        if price is None or price_ts_ms is None:
            continue
        ticks.append((price_ts_ms, price))
    return tuple(sorted(ticks))


def _nearest_chainlink_price(
    ticks: Sequence[tuple[int, float]],
    *,
    target_ts_ms: int,
    tolerance_ms: int,
) -> float | None:
    best_distance: int | None = None
    best_price: float | None = None
    for ts_ms, price in ticks:
        distance = abs(ts_ms - target_ts_ms)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_price = price
    if best_distance is None or best_distance > tolerance_ms:
        return None
    return best_price


def _direction(value: object) -> Direction | None:
    if value == "Up":
        return "Up"
    if value == "Down":
        return "Down"
    return None


def _str_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _float_value(*values: object, default: float | None = None) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return default


def _start_ts_from_slug(slug: str | None, end_ts: int) -> int:
    if slug is not None:
        parts = slug.rsplit("-", 1)
        if len(parts) == 2:
            parsed = _int_value(parts[1])
            if parsed is not None:
                return parsed
    return end_ts - 300
