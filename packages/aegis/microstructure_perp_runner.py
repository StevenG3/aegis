from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from aegis.backtest_core import (
    benjamini_hochberg,
    metrics_from_returns,
    pbo,
    sign_test_p_value,
    trade_scorecard,
    trade_scorecard_to_dict,
)

MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR = 15_000.0


@dataclass(frozen=True)
class MicrostructureBar:
    symbol: str
    timestamp: int
    close: float
    btc_close: float | None
    open_interest: float
    funding_rate: float
    buy_volume: float
    sell_volume: float
    bid_ask_spread_bps: float | None
    top_depth_usd: float | None
    quote_volume_usd: float | None
    order_book_event_rate_per_hour: float | None = None
    survivor_status: str = "active"


@dataclass(frozen=True)
class MicrostructureCandidate:
    name: str
    funding_abs_bps: float
    imbalance_abs: float
    oi_drop_abs: float
    score_threshold: int


@dataclass(frozen=True)
class CandidateResult:
    symbol: str
    candidate: MicrostructureCandidate
    returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    trade_returns: tuple[float, ...]
    fold_excess: tuple[float, ...]
    p_value: float
    turnover: float
    net_cost: float
    event_log: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class BtcImpulseConfig:
    enabled: bool
    lookback_bars: int
    return_threshold: float
    zscore_threshold: float


@dataclass(frozen=True)
class LiquidityGuardConfig:
    enabled: bool
    max_spread_bps: float
    min_top_depth_usd: float
    min_quote_volume_usd: float


@dataclass(frozen=True)
class LiquidityDataAvailability:
    spread_data_blocked: bool
    top_depth_data_blocked: bool


@dataclass(frozen=True)
class EntryWindow:
    start_timestamp: int | None
    end_timestamp: int | None


def run_microstructure_perp_from_spec(spec: Any) -> Mapping[str, Any]:
    """Run the offline perp microstructure hypothesis against private spec params."""
    params = _mapping(spec.params, "params")
    bars = _bars_from_params(params)
    if not bars:
        return _insufficient_payload("params.observations is empty")

    filtered, excluded_symbols, data_blocked_log = _exclude_data_blocked_symbols(bars)
    if not filtered:
        return _insufficient_payload(
            "all symbols excluded by order_book_event_rate_per_hour data-block",
            excluded_symbols=excluded_symbols,
            event_log=data_blocked_log,
        )

    by_symbol = _group_bars(filtered)
    usable_symbols = tuple(sorted(by_symbol))
    if not usable_symbols:
        return _insufficient_payload("no usable symbol histories after filtering")

    grid = _candidate_grid(params)
    if not grid:
        return _insufficient_payload("predeclared microstructure grid is empty")

    costs = _costs(spec.cost_model)
    locked_oos_fraction = _float_param(params, "locked_oos_fraction", 0.40)
    fold_count = max(2, _int_param(params, "fold_count", 4))
    annualization = max(1, _int_param(params, "annualization_periods", 365 * 3))
    fdr_alpha = _float_param(params, "fdr_alpha", 0.10)
    pbo_splits = _int_param(params, "pbo_splits", 4)
    pbo_threshold = _float_param(params, "pbo_threshold", 0.20)
    btc_impulse = _btc_impulse_config(params)
    liquidity_guard = _liquidity_guard_config(params)
    liquidity_availability = _liquidity_data_availability(filtered)
    entry_window = _entry_window(params)

    results = tuple(
        _evaluate_candidate(
            candidate,
            symbol=symbol,
            bars=bars,
            costs=costs,
            locked_oos_fraction=locked_oos_fraction,
            fold_count=fold_count,
            btc_impulse=btc_impulse,
            liquidity_guard=liquidity_guard,
            liquidity_availability=liquidity_availability,
            entry_window=entry_window,
        )
        for symbol, bars in sorted(by_symbol.items())
        for candidate in grid
    )
    valid_results = tuple(result for result in results if result.returns)
    if not valid_results:
        event_log = tuple(data_blocked_log) + tuple(
            entry for result in results for entry in result.event_log
        )
        return _insufficient_payload(
            "insufficient locked-OOS t+1 samples after applying grid",
            excluded_symbols=excluded_symbols,
            candidate_count=max(_spec_trial_count(spec), len(grid) * len(by_symbol)),
            event_log=event_log,
        )

    p_values = [result.p_value for result in valid_results]
    fdr_flags = benjamini_hochberg(p_values, alpha=fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(valid_results, pbo_splits=pbo_splits)
    pbo_value = _number_from_mapping(pbo_report, "pbo", 1.0)
    pbo_pass = pbo_value <= pbo_threshold
    survivors = [
        result
        for result, fdr_pass in zip(valid_results, fdr_flags, strict=True)
        if fdr_pass and pbo_pass and _beats_benchmark(result, annualization)
    ]
    best = max(valid_results, key=_candidate_rank)
    strategy_metrics = metrics_from_returns(
        best.returns,
        annualization_periods=annualization,
        turnover=best.turnover,
        net_cost=best.net_cost,
        oos_vs_buy_hold_window_win_rate=_positive_share(best.fold_excess),
    )
    benchmark_metrics = metrics_from_returns(
        best.benchmark_returns,
        annualization_periods=annualization,
        turnover=0.0,
        net_cost=0.0,
    )
    scorecard = trade_scorecard(best.trade_returns)
    verdict = "EDGE" if survivors else "NO_EDGE"
    status = "OK"
    reason = (
            "predeclared microstructure symbol-grid survived BH-FDR and PBO"
        if survivors
        else "no predeclared microstructure symbol-grid survived BH-FDR, PBO, and buy-hold gates"
    )
    candidate_count_n = max(_spec_trial_count(spec), len(grid) * len(by_symbol))
    return {
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "strategy": "microstructure_perp_funding_oi_orderflow",
        "standard_metrics": _metrics_to_dict(strategy_metrics),
        "benchmark_metrics": {"buy_and_hold": _metrics_to_dict(benchmark_metrics)},
        "trade_scorecard": trade_scorecard_to_dict(scorecard),
        "candidate_count_n": candidate_count_n,
        "raw_is_survivors": sum(1 for result in valid_results if _candidate_rank(result) > 0.0),
        "fdr_is_survivors": sum(1 for value in fdr_flags if value),
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": candidate_count_n,
            "tested_candidates": len(valid_results),
            "pooled_grid_candidates": len(grid),
            "symbol_count": len(by_symbol),
            "fdr_alpha": fdr_alpha,
            "fdr_before": sum(1 for result in valid_results if result.p_value < fdr_alpha),
            "fdr_after": sum(1 for value in fdr_flags if value),
            "pbo_before_survivors": sum(1 for value in fdr_flags if value),
            "pbo_after_survivors": len(survivors),
            "pbo_threshold": pbo_threshold,
            "pbo": pbo_report,
        },
        "safety": {
            "local_file_only": True,
            "network": False,
            "live": False,
            "t_plus_1_execution": True,
            "locked_oos": True,
            "walk_forward": True,
            "full_costs": True,
            "perp_funding_counted": True,
            "survivor_light_ceiling_required": True,
            "order_book_event_rate_cap_per_hour": MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR,
            "excluded_data_blocked_symbols": excluded_symbols,
        },
        "best_candidate": _best_candidate_to_dict(best),
        "event_log": tuple(data_blocked_log)
        + tuple(entry for result in valid_results for entry in result.event_log),
        "universe": {
            "usable_symbols": usable_symbols,
            "excluded_data_blocked_symbols": excluded_symbols,
            "survivor_light": True,
        },
        "research_controls": {
            "btc_impulse": _btc_impulse_to_dict(btc_impulse),
            "entry_window": _entry_window_to_dict(entry_window),
            "liquidity_guard": _liquidity_guard_to_dict(liquidity_guard),
            "liquidity_data_availability": _liquidity_availability_to_dict(
                liquidity_availability
            ),
        },
    }


def _bars_from_params(params: Mapping[str, object]) -> tuple[MicrostructureBar, ...]:
    raw = params.get("observations", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("params.observations must be a list of bar objects")
    return tuple(_bar_from_mapping(_mapping(item, "observation")) for item in raw)


def _bar_from_mapping(item: Mapping[str, object]) -> MicrostructureBar:
    return MicrostructureBar(
        symbol=_required_str(item, "symbol"),
        timestamp=_required_int(item, "timestamp"),
        close=_positive_float(item, "close"),
        btc_close=_optional_positive_float(item, "btc_close"),
        open_interest=_nonnegative_float(item, "open_interest"),
        funding_rate=_float_value(item, "funding_rate"),
        buy_volume=_nonnegative_float(item, "buy_volume"),
        sell_volume=_nonnegative_float(item, "sell_volume"),
        bid_ask_spread_bps=_optional_nonnegative_float(item, "bid_ask_spread_bps"),
        top_depth_usd=_optional_nonnegative_float(item, "top_depth_usd"),
        quote_volume_usd=_optional_nonnegative_float(item, "quote_volume_usd"),
        order_book_event_rate_per_hour=_optional_nonnegative_float(
            item, "order_book_event_rate_per_hour"
        ),
        survivor_status=_str_value(item.get("survivor_status", "active")),
    )


def _exclude_data_blocked_symbols(
    bars: Sequence[MicrostructureBar],
) -> tuple[tuple[MicrostructureBar, ...], tuple[str, ...], tuple[Mapping[str, object], ...]]:
    blocked = {
        bar.symbol
        for bar in bars
        if bar.order_book_event_rate_per_hour is not None
        and bar.order_book_event_rate_per_hour > MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR
    }
    kept = tuple(bar for bar in bars if bar.symbol not in blocked)
    log = tuple(
        _data_blocked_log_entry(bar)
        for bar in bars
        if bar.symbol in blocked
        and bar.order_book_event_rate_per_hour is not None
        and bar.order_book_event_rate_per_hour > MAX_ORDER_BOOK_EVENT_RATE_PER_HOUR
    )
    return kept, tuple(sorted(blocked)), log


def _group_bars(
    bars: Sequence[MicrostructureBar],
) -> dict[str, tuple[MicrostructureBar, ...]]:
    grouped: dict[str, list[MicrostructureBar]] = {}
    for bar in bars:
        grouped.setdefault(bar.symbol, []).append(bar)
    return {
        symbol: tuple(sorted(items, key=lambda item: item.timestamp))
        for symbol, items in grouped.items()
    }


def _candidate_grid(params: Mapping[str, object]) -> tuple[MicrostructureCandidate, ...]:
    grid = _mapping(params.get("grid", {}), "params.grid")
    funding_values = _float_list(grid.get("funding_abs_bps", (1.0, 3.0)))
    imbalance_values = _float_list(grid.get("imbalance_abs", (0.10, 0.20)))
    oi_drop_values = _float_list(grid.get("oi_drop_abs", (0.02, 0.05)))
    score_values = _int_list(grid.get("score_threshold", (1, 2)))
    candidates: list[MicrostructureCandidate] = []
    for funding_bps, imbalance, oi_drop, score in product(
        funding_values, imbalance_values, oi_drop_values, score_values
    ):
        candidates.append(
            MicrostructureCandidate(
                name=(
                    f"fund{funding_bps:g}_imb{imbalance:g}_"
                    f"oidrop{oi_drop:g}_score{score}"
                ),
                funding_abs_bps=funding_bps,
                imbalance_abs=imbalance,
                oi_drop_abs=oi_drop,
                score_threshold=score,
            )
        )
    return tuple(candidates)


def _evaluate_candidate(
    candidate: MicrostructureCandidate,
    *,
    symbol: str,
    bars: Sequence[MicrostructureBar],
    costs: Mapping[str, float],
    locked_oos_fraction: float,
    fold_count: int,
    btc_impulse: BtcImpulseConfig,
    liquidity_guard: LiquidityGuardConfig,
    liquidity_availability: LiquidityDataAvailability,
    entry_window: EntryWindow,
) -> CandidateResult:
    (
        strategy_returns,
        benchmark_returns,
        trade_returns,
        total_turnover,
        total_cost,
        event_log,
    ) = _symbol_returns(
        candidate,
        bars=bars,
        costs=costs,
        locked_oos_fraction=locked_oos_fraction,
        btc_impulse=btc_impulse,
        liquidity_guard=liquidity_guard,
        liquidity_availability=liquidity_availability,
        entry_window=entry_window,
    )
    fold_excess = _fold_excess(strategy_returns, benchmark_returns, fold_count)
    return CandidateResult(
        symbol=symbol,
        candidate=candidate,
        returns=tuple(strategy_returns),
        benchmark_returns=tuple(benchmark_returns),
        trade_returns=tuple(trade_returns),
        fold_excess=fold_excess,
        p_value=sign_test_p_value(fold_excess, alternative="greater"),
        turnover=total_turnover,
        net_cost=total_cost,
        event_log=tuple(event_log),
    )


def _symbol_returns(
    candidate: MicrostructureCandidate,
    *,
    bars: Sequence[MicrostructureBar],
    costs: Mapping[str, float],
    locked_oos_fraction: float,
    btc_impulse: BtcImpulseConfig,
    liquidity_guard: LiquidityGuardConfig,
    liquidity_availability: LiquidityDataAvailability,
    entry_window: EntryWindow,
) -> tuple[list[float], list[float], list[float], float, float, list[Mapping[str, object]]]:
    if len(bars) < 6:
        return [], [], [], 0.0, 0.0, []
    start = max(btc_impulse.lookback_bars, int(len(bars) * (1.0 - locked_oos_fraction)))
    strategy: list[float] = []
    benchmark: list[float] = []
    trades: list[float] = []
    event_log: list[Mapping[str, object]] = []
    turnover = 0.0
    net_cost = 0.0
    previous_position = 0
    for index in range(start, len(bars) - 2):
        previous = bars[index - 1]
        current = bars[index]
        entry = bars[index + 1]
        exit_bar = bars[index + 2]
        signal = _signal(candidate, previous=previous, current=current)
        btc_pass, btc_reason = _btc_impulse_pass(bars, index=index, config=btc_impulse)
        liquidity_pass, liquidity_reason = _liquidity_guard_pass(
            current, liquidity_guard, liquidity_availability
        )
        entry_window_pass = _entry_window_pass(current.timestamp, entry_window)
        orderbook_missing = current.order_book_event_rate_per_hour is None
        excluded_reason = _excluded_reason(
            btc_reason=btc_reason,
            liquidity_reason=liquidity_reason,
            entry_window_pass=entry_window_pass,
        )
        if not btc_pass or not liquidity_pass or not entry_window_pass:
            signal = 0
        gross = signal * (exit_bar.close / entry.close - 1.0)
        position_change = abs(signal - previous_position)
        fee_cost = position_change * costs["fee_cost"]
        slippage_cost = position_change * costs["slippage_cost"]
        funding_cost = signal * entry.funding_rate
        net = gross - fee_cost - slippage_cost - funding_cost
        strategy.append(net)
        benchmark.append(exit_bar.close / entry.close - 1.0)
        turnover += float(position_change)
        net_cost += fee_cost + slippage_cost + abs(funding_cost)
        if signal != 0:
            trades.append(net)
        event_log.append(
            _event_log_entry(
                candidate=candidate,
                current=current,
                previous=previous,
                signal=signal,
                btc_impulse_pass=btc_pass,
                liquidity_guard_pass=liquidity_pass,
                liquidity_data_blocked=_liquidity_data_blocked_fields(liquidity_availability),
                excluded_reason=excluded_reason,
                entry_timestamp=entry.timestamp,
                exit_timestamp=exit_bar.timestamp,
                net_return_after_costs=net,
                funding_cost=funding_cost,
                fee_cost=fee_cost,
                slippage_cost=slippage_cost,
                orderbook_event_rate_missing=orderbook_missing,
            )
        )
        previous_position = signal
    return strategy, benchmark, trades, turnover, net_cost, event_log


def _signal(
    candidate: MicrostructureCandidate,
    *,
    previous: MicrostructureBar,
    current: MicrostructureBar,
) -> int:
    score = 0
    funding_bps = current.funding_rate * 10_000.0
    if funding_bps >= candidate.funding_abs_bps:
        score -= 1
    elif funding_bps <= -candidate.funding_abs_bps:
        score += 1

    price_change = current.close / previous.close - 1.0
    oi_change = (
        current.open_interest / previous.open_interest - 1.0
        if previous.open_interest > 0
        else 0.0
    )
    if price_change > 0.0 and oi_change <= -candidate.oi_drop_abs:
        score -= 1
    elif price_change < 0.0 and oi_change <= -candidate.oi_drop_abs:
        score += 1

    total_volume = current.buy_volume + current.sell_volume
    imbalance = (
        (current.buy_volume - current.sell_volume) / total_volume if total_volume > 0 else 0.0
    )
    if imbalance >= candidate.imbalance_abs:
        score += 1
    elif imbalance <= -candidate.imbalance_abs:
        score -= 1

    if score >= candidate.score_threshold:
        return 1
    if score <= -candidate.score_threshold:
        return -1
    return 0


def _btc_impulse_pass(
    bars: Sequence[MicrostructureBar],
    *,
    index: int,
    config: BtcImpulseConfig,
) -> tuple[bool, str]:
    if not config.enabled:
        return True, ""
    current = bars[index]
    if current.btc_close is None:
        return False, "btc_close_missing"
    lookback_index = index - config.lookback_bars
    if lookback_index < 0:
        return False, "btc_impulse_lookback_insufficient"
    reference = bars[lookback_index]
    if reference.btc_close is None:
        return False, "btc_close_missing"
    impulse_return = current.btc_close / reference.btc_close - 1.0
    abs_return_pass = abs(impulse_return) >= config.return_threshold
    zscore_pass = True
    if config.zscore_threshold > 0.0:
        btc_returns = _btc_returns(bars[max(0, lookback_index) : index + 1])
        if len(btc_returns) < 2:
            return False, "btc_impulse_zscore_insufficient"
        stdev = statistics.pstdev(btc_returns)
        zscore = 0.0 if stdev == 0.0 else (btc_returns[-1] - statistics.fmean(btc_returns)) / stdev
        zscore_pass = abs(zscore) >= config.zscore_threshold
    if abs_return_pass and zscore_pass:
        return True, ""
    return False, "btc_impulse_not_triggered"


def _btc_returns(bars: Sequence[MicrostructureBar]) -> tuple[float, ...]:
    values: list[float] = []
    for previous, current in zip(bars, bars[1:], strict=False):
        if previous.btc_close is None or current.btc_close is None:
            continue
        values.append(current.btc_close / previous.btc_close - 1.0)
    return tuple(values)


def _liquidity_guard_pass(
    current: MicrostructureBar,
    config: LiquidityGuardConfig,
    availability: LiquidityDataAvailability,
) -> tuple[bool, str]:
    if not config.enabled:
        return True, ""
    if current.bid_ask_spread_bps is None and not availability.spread_data_blocked:
        return False, "bid_ask_spread_bps_missing"
    if current.top_depth_usd is None and not availability.top_depth_data_blocked:
        return False, "top_depth_usd_missing"
    if current.quote_volume_usd is None:
        return False, "quote_volume_usd_missing"
    if (
        current.bid_ask_spread_bps is not None
        and current.bid_ask_spread_bps > config.max_spread_bps
    ):
        return False, "spread_guard_fail"
    if current.top_depth_usd is not None and current.top_depth_usd < config.min_top_depth_usd:
        return False, "top_depth_guard_fail"
    if current.quote_volume_usd < config.min_quote_volume_usd:
        return False, "quote_volume_guard_fail"
    return True, ""


def _liquidity_data_availability(
    bars: Sequence[MicrostructureBar],
) -> LiquidityDataAvailability:
    return LiquidityDataAvailability(
        spread_data_blocked=all(bar.bid_ask_spread_bps is None for bar in bars),
        top_depth_data_blocked=all(bar.top_depth_usd is None for bar in bars),
    )


def _liquidity_data_blocked_fields(
    availability: LiquidityDataAvailability,
) -> tuple[str, ...]:
    fields: list[str] = []
    if availability.spread_data_blocked:
        fields.append("bid_ask_spread_bps")
    if availability.top_depth_data_blocked:
        fields.append("top_depth_usd")
    return tuple(fields)


def _entry_window_pass(timestamp: int, window: EntryWindow) -> bool:
    if window.start_timestamp is not None and timestamp < window.start_timestamp:
        return False
    return not (window.end_timestamp is not None and timestamp > window.end_timestamp)


def _excluded_reason(
    *,
    btc_reason: str,
    liquidity_reason: str,
    entry_window_pass: bool,
) -> str:
    if not entry_window_pass:
        return "outside_predeclared_entry_window"
    if btc_reason:
        return btc_reason
    if liquidity_reason:
        return liquidity_reason
    return ""


def _event_log_entry(
    *,
    candidate: MicrostructureCandidate,
    current: MicrostructureBar,
    previous: MicrostructureBar,
    signal: int,
    btc_impulse_pass: bool,
    liquidity_guard_pass: bool,
    liquidity_data_blocked: Sequence[str],
    excluded_reason: str,
    entry_timestamp: int,
    exit_timestamp: int,
    net_return_after_costs: float,
    funding_cost: float,
    fee_cost: float,
    slippage_cost: float,
    orderbook_event_rate_missing: bool,
) -> Mapping[str, object]:
    return {
        "candidate": candidate.name,
        "symbol": current.symbol,
        "timestamp": current.timestamp,
        "btc_impulse_pass": btc_impulse_pass,
        "funding_rate": current.funding_rate,
        "funding_sign": _sign(current.funding_rate),
        "open_interest": current.open_interest,
        "oi_price_divergence": _oi_price_divergence(previous, current),
        "order_flow_imbalance": _order_flow_imbalance(current),
        "bid_ask_spread_bps": current.bid_ask_spread_bps,
        "top_depth_usd": current.top_depth_usd,
        "quote_volume_usd": current.quote_volume_usd,
        "liquidity_guard_pass": liquidity_guard_pass,
        "liquidity_data_blocked": tuple(liquidity_data_blocked),
        "excluded_reason": excluded_reason,
        "entry_timestamp": entry_timestamp,
        "exit_timestamp": exit_timestamp,
        "net_return_after_costs": net_return_after_costs,
        "funding_cost": funding_cost,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "orderbook_event_rate_missing": orderbook_event_rate_missing,
        "signal": signal,
    }


def _data_blocked_log_entry(bar: MicrostructureBar) -> Mapping[str, object]:
    return {
        "candidate": "",
        "symbol": bar.symbol,
        "timestamp": bar.timestamp,
        "btc_impulse_pass": False,
        "funding_rate": bar.funding_rate,
        "funding_sign": _sign(bar.funding_rate),
        "open_interest": bar.open_interest,
        "oi_price_divergence": False,
        "order_flow_imbalance": _order_flow_imbalance(bar),
        "bid_ask_spread_bps": bar.bid_ask_spread_bps,
        "top_depth_usd": bar.top_depth_usd,
        "quote_volume_usd": bar.quote_volume_usd,
        "liquidity_guard_pass": False,
        "liquidity_data_blocked": (),
        "excluded_reason": "orderbook_event_rate_data_blocked",
        "entry_timestamp": None,
        "exit_timestamp": None,
        "net_return_after_costs": 0.0,
        "funding_cost": 0.0,
        "fee_cost": 0.0,
        "slippage_cost": 0.0,
        "orderbook_event_rate_missing": False,
    }


def _oi_price_divergence(previous: MicrostructureBar, current: MicrostructureBar) -> bool:
    price_change = current.close / previous.close - 1.0
    oi_change = (
        current.open_interest / previous.open_interest - 1.0
        if previous.open_interest > 0
        else 0.0
    )
    return (price_change > 0.0 and oi_change < 0.0) or (
        price_change < 0.0 and oi_change < 0.0
    )


def _order_flow_imbalance(current: MicrostructureBar) -> float:
    total_volume = current.buy_volume + current.sell_volume
    return (current.buy_volume - current.sell_volume) / total_volume if total_volume > 0 else 0.0


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _fold_excess(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    fold_count: int,
) -> tuple[float, ...]:
    n = min(len(strategy_returns), len(benchmark_returns))
    if n == 0:
        return ()
    folds = min(fold_count, n)
    fold_size = max(1, n // folds)
    values: list[float] = []
    for start in range(0, n, fold_size):
        stop = min(n, start + fold_size)
        strategy = math.prod(1.0 + value for value in strategy_returns[start:stop]) - 1.0
        benchmark = math.prod(1.0 + value for value in benchmark_returns[start:stop]) - 1.0
        values.append(strategy - benchmark)
    return tuple(values)


def _pbo_report(
    results: Sequence[CandidateResult],
    *,
    pbo_splits: int,
) -> Mapping[str, object]:
    trials = [result.fold_excess for result in results if result.fold_excess]
    if len(trials) < 2:
        return {"valid": False, "reason": "PBO requires at least two candidates", "pbo": 1.0}
    min_len = min(len(trial) for trial in trials)
    effective_splits = min(pbo_splits, min_len)
    if effective_splits % 2 != 0:
        effective_splits -= 1
    if effective_splits < 4:
        return {
            "valid": False,
            "reason": "PBO requires at least four OOS fold observations",
            "pbo": 1.0,
            "trial_count": len(trials),
            "observation_count": min_len,
        }
    aligned = [tuple(trial[:min_len]) for trial in trials]
    report = dict(pbo(aligned, n_splits=effective_splits))
    report["valid"] = True
    return report


def _beats_benchmark(result: CandidateResult, annualization_periods: int) -> bool:
    strategy = metrics_from_returns(
        result.returns,
        annualization_periods=annualization_periods,
        turnover=result.turnover,
        net_cost=result.net_cost,
    )
    benchmark = metrics_from_returns(
        result.benchmark_returns,
        annualization_periods=annualization_periods,
        turnover=0.0,
        net_cost=0.0,
    )
    return strategy.total_return > benchmark.total_return and strategy.sharpe > benchmark.sharpe


def _candidate_rank(result: CandidateResult) -> float:
    return statistics.fmean(result.fold_excess) if result.fold_excess else -math.inf


def _costs(cost_model: object) -> Mapping[str, float]:
    model = _mapping(cost_model, "cost_model")
    fee_bps = _float_value(model, "fee_bps")
    slippage_bps = _float_value(model, "slippage_bps")
    return {
        "fee_cost": fee_bps / 10_000.0,
        "slippage_cost": slippage_bps / 10_000.0,
        "one_way_cost": (fee_bps + slippage_bps) / 10_000.0,
    }


def _btc_impulse_config(params: Mapping[str, object]) -> BtcImpulseConfig:
    raw = _mapping(params.get("btc_impulse", {}), "params.btc_impulse")
    return BtcImpulseConfig(
        enabled=_bool_param(raw, "enabled", True),
        lookback_bars=max(1, _int_param(raw, "lookback_bars", 3)),
        return_threshold=_float_param(raw, "return_threshold", 0.02),
        zscore_threshold=_float_param(raw, "zscore_threshold", 0.0),
    )


def _liquidity_guard_config(params: Mapping[str, object]) -> LiquidityGuardConfig:
    raw = _mapping(params.get("liquidity_guard", {}), "params.liquidity_guard")
    return LiquidityGuardConfig(
        enabled=_bool_param(raw, "enabled", True),
        max_spread_bps=_float_param(raw, "max_spread_bps", 25.0),
        min_top_depth_usd=_float_param(raw, "min_top_depth_usd", 50_000.0),
        min_quote_volume_usd=_float_param(raw, "min_quote_volume_usd", 1_000_000.0),
    )


def _entry_window(params: Mapping[str, object]) -> EntryWindow:
    raw = _mapping(params.get("entry_window", {}), "params.entry_window")
    return EntryWindow(
        start_timestamp=_optional_int(raw, "start_timestamp"),
        end_timestamp=_optional_int(raw, "end_timestamp"),
    )


def _btc_impulse_to_dict(config: BtcImpulseConfig) -> dict[str, float | int | bool]:
    return {
        "enabled": config.enabled,
        "lookback_bars": config.lookback_bars,
        "return_threshold": config.return_threshold,
        "zscore_threshold": config.zscore_threshold,
    }


def _liquidity_guard_to_dict(config: LiquidityGuardConfig) -> dict[str, float | bool]:
    return {
        "enabled": config.enabled,
        "max_spread_bps": config.max_spread_bps,
        "min_top_depth_usd": config.min_top_depth_usd,
        "min_quote_volume_usd": config.min_quote_volume_usd,
    }


def _liquidity_availability_to_dict(
    availability: LiquidityDataAvailability,
) -> dict[str, bool]:
    return {
        "spread_data_blocked": availability.spread_data_blocked,
        "top_depth_data_blocked": availability.top_depth_data_blocked,
    }


def _entry_window_to_dict(config: EntryWindow) -> dict[str, int | None]:
    return {
        "start_timestamp": config.start_timestamp,
        "end_timestamp": config.end_timestamp,
    }


def _metrics_to_dict(metrics: Any) -> dict[str, float]:
    return {
        "annualized_return": metrics.annualized_return,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "positive_period_win_rate": metrics.positive_period_win_rate,
        "oos_vs_buy_hold_window_win_rate": metrics.oos_vs_buy_hold_window_win_rate,
        "annualized_turnover": metrics.annualized_turnover,
        "net_cost": metrics.net_cost,
    }


def _candidate_to_dict(candidate: MicrostructureCandidate) -> dict[str, float | int | str]:
    return {
        "name": candidate.name,
        "funding_abs_bps": candidate.funding_abs_bps,
        "imbalance_abs": candidate.imbalance_abs,
        "oi_drop_abs": candidate.oi_drop_abs,
        "score_threshold": candidate.score_threshold,
    }


def _best_candidate_to_dict(result: CandidateResult) -> dict[str, float | int | str]:
    payload = _candidate_to_dict(result.candidate)
    payload["symbol"] = result.symbol
    return payload


def _spec_trial_count(spec: Any) -> int:
    value = getattr(spec, "trial_count_n", 0)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def _insufficient_payload(
    reason: str,
    *,
    excluded_symbols: Sequence[str] = (),
    candidate_count: int = 0,
    event_log: Sequence[Mapping[str, object]] = (),
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "candidate_count_n": candidate_count,
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": candidate_count,
            "fdr_after": 0,
            "pbo_after_survivors": 0,
        },
        "safety": {
            "local_file_only": True,
            "network": False,
            "live": False,
            "excluded_data_blocked_symbols": tuple(excluded_symbols),
        },
        "event_log": tuple(event_log),
    }


def _positive_share(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value > 0.0) / len(values)


def _number_from_mapping(raw: Mapping[str, object], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _mapping(raw: object, name: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _str_value(raw: object) -> str:
    return str(raw).strip() if raw is not None else ""


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_value(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _positive_float(raw: Mapping[str, object], key: str) -> float:
    value = _float_value(raw, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _nonnegative_float(
    raw: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    if key not in raw and default is not None:
        return default
    value = _float_value(raw, key)
    if value < 0.0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def _float_param(params: Mapping[str, object], key: str, default: float) -> float:
    if key not in params:
        return default
    return _float_value(params, key)


def _int_param(params: Mapping[str, object], key: str, default: int) -> int:
    if key not in params:
        return default
    return _required_int(params, key)


def _bool_param(params: Mapping[str, object], key: str, default: bool) -> bool:
    if key not in params:
        return default
    value = params[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    return _required_int(raw, key)


def _optional_positive_float(raw: Mapping[str, object], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return _positive_float(raw, key)


def _optional_nonnegative_float(raw: Mapping[str, object], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return _nonnegative_float(raw, key)


def _float_list(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("grid values must be lists")
    values = tuple(float(value) for value in raw)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("grid float values must be nonnegative finite numbers")
    return values


def _int_list(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("grid values must be lists")
    values = tuple(int(value) for value in raw)
    if not values or any(value < 1 for value in values):
        raise ValueError("grid integer values must be positive")
    return values
