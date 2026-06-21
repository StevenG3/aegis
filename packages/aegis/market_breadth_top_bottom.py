from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from aegis.backtest_core import (
    CostModel,
    benjamini_hochberg,
    deflated_sharpe_threshold,
    metrics_from_returns,
    paired_block_bootstrap_risk_difference_test,
    pbo,
    sign_test_p_value,
)

UniverseVerdict = Literal["SUGGESTIVE", "NO_EDGE", "INSUFFICIENT"]
HealthStatus = Literal[
    "NO_GO_DATA",
    "DATA_LIMITED",
    "RESEARCH_OK",
    "PAPER_CANDIDATE_ONLY",
    "NO_LIVE",
]


@dataclass(frozen=True)
class BreadthBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    tradable: bool = True


@dataclass(frozen=True)
class BreadthConfig:
    horizons: tuple[int, ...] = (5, 10, 20, 60)
    hot_thresholds: tuple[float, ...] = (0.70,)
    floor_thresholds: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05)
    panic_8d_thresholds: tuple[float, ...] = (0.03, 0.06)
    panic_21d_thresholds: tuple[float, ...] = (0.05,)
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    pbo_threshold: float = 0.50
    min_events_per_candidate: int = 5
    min_oos_years: int = 3
    overlap_correction: bool = False
    block_bootstrap_samples: int = 1_000
    block_bootstrap_ci_alpha: float = 0.05
    block_bootstrap_seed: int = 70
    annualization_periods: int = 252
    survivor_light: bool = True


@dataclass(frozen=True)
class BreadthFrame:
    timestamp: int
    benchmark_close: float
    breadth_ma8: float
    breadth_ma21: float
    breadth_ma60: float
    ma8_above_ma21_ratio: float
    ma21_above_ma60_ratio: float
    index_ret_8d: float | None
    index_ret_21d: float | None
    breadth_momentum: float | None
    top_divergence_20d: bool
    bottom_divergence_20d: bool
    regime: str
    constituent_count: int
    missing_count: int


@dataclass(frozen=True)
class EventRecord:
    key: str
    signal: str
    threshold: float
    horizon: int
    timestamp: int
    entry_timestamp: int
    exit_timestamp: int
    regime: str
    forward_return_after_costs: float
    baseline_return: float
    excess_return: float
    drawdown: float
    breadth_ma8: float
    breadth_ma21: float
    breadth_ma60: float


@dataclass(frozen=True)
class _BlockBootstrapConfig:
    annualization_periods: int
    risk_diff_bootstrap_samples: int
    risk_diff_bootstrap_block_bars: int
    risk_diff_ci_alpha: float
    risk_diff_random_seed: int


DEFAULT_COST_MODEL = CostModel(
    fee_bps=10.0,
    slippage_bps=5.0,
    funding_label="N/A for spot breadth event study; no perp funding modeled",
)
DEFAULT_BREADTH_CONFIG = BreadthConfig()


def run_market_breadth_study(
    *,
    universe_name: str,
    member_bars: Mapping[str, Sequence[BreadthBar]],
    benchmark_bars: Sequence[BreadthBar],
    config: BreadthConfig = DEFAULT_BREADTH_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    data_source: str,
    benchmark_name: str,
    survivor_light: bool = True,
) -> Mapping[str, Any]:
    frames = compute_breadth_frames(member_bars, benchmark_bars)
    trial_count = trial_count_for_config(config)
    if not frames:
        return _empty_report(
            universe_name,
            benchmark_name,
            data_source,
            verdict="INSUFFICIENT",
            status="NO_GO_DATA",
            reason="no aligned breadth frames; daily member and benchmark data are unavailable",
            trial_count=trial_count,
            survivor_light=survivor_light,
        )
    events = build_event_records(
        frames,
        config=config,
        cost_model=cost_model,
    )
    data_quality = _data_quality(
        member_bars,
        benchmark_bars,
        frames,
        survivor_light=survivor_light,
        data_source=data_source,
    )
    if not events:
        return {
            **_empty_report(
                universe_name,
                benchmark_name,
                data_source,
                verdict="INSUFFICIENT",
                status="DATA_LIMITED",
                reason="no predeclared breadth events triggered after t+1/horizon alignment",
                trial_count=trial_count,
                survivor_light=survivor_light,
            ),
            "breadth_tail": [_frame_dict(frame) for frame in frames[-5:]],
            "data_quality": data_quality,
        }
    scoring_events = disjoint_event_records(events) if config.overlap_correction else events
    candidates = _candidate_statistics(
        scoring_events,
        trial_count=trial_count,
        config=config,
        cost_model=cost_model,
        use_block_bootstrap_p=config.overlap_correction,
    )
    raw_candidates = _candidate_statistics(
        events,
        trial_count=trial_count,
        config=config,
        cost_model=cost_model,
        use_block_bootstrap_p=False,
    )
    raw_p_values = [_float_field(row["p_value"]) for row in raw_candidates]
    raw_fdr_flags = (
        benjamini_hochberg(raw_p_values, alpha=config.fdr_alpha) if raw_p_values else []
    )
    for row, passed in zip(raw_candidates, raw_fdr_flags, strict=True):
        row["bh_fdr_pass"] = bool(passed)
    raw_fdr_survivors = [
        row
        for row in raw_candidates
        if bool(row["bh_fdr_pass"])
        and _float_field(row["mean_excess_return"]) > 0.0
        and bool(row["deflated_sharpe_pass"])
    ]
    p_values = [_float_field(row["p_value"]) for row in candidates]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha) if p_values else []
    for row, passed in zip(candidates, fdr_flags, strict=True):
        row["bh_fdr_pass"] = bool(passed)
    pbo_report = _pbo_report(scoring_events, config=config)
    fdr_survivors = [
        row
        for row in candidates
        if bool(row["bh_fdr_pass"])
        and _float_field(row["mean_excess_return"]) > 0.0
        and bool(row["deflated_sharpe_pass"])
    ]
    oos_stability = _annual_oos_stability(events, fdr_survivors, config=config)
    pbo_value = pbo_report.get("pbo")
    positive_gate = (
        bool(fdr_survivors)
        and bool(pbo_report.get("valid"))
        and _float_field(pbo_value, default=1.0) < config.pbo_threshold
        and bool(oos_stability["passed"])
    )
    verdict: UniverseVerdict
    status: HealthStatus
    if positive_gate:
        verdict = "SUGGESTIVE"
        status = "PAPER_CANDIDATE_ONLY"
        reason = (
            "at least one breadth event passed BH-FDR, deflated Sharpe, PBO, and annual OOS "
            "stability; survivor-light data caps the result below ROBUST"
        )
    else:
        verdict = "NO_EDGE"
        status = "DATA_LIMITED" if survivor_light else "RESEARCH_OK"
        reason = "no breadth event survived full-scope BH-FDR + PBO + deflated Sharpe gates"
        if fdr_survivors and not bool(oos_stability["passed"]):
            reason = (
                "some events passed full-scope BH-FDR/PBO/deflated Sharpe, but failed annual "
                "locked-OOS stability"
            )
    return {
        "universe": universe_name,
        "benchmark": benchmark_name,
        "data_source": data_source,
        "verdict": verdict,
        "health_status": status,
        "reason": reason,
        "survivor_light_ceiling": survivor_light,
        "max_positive_verdict": "SUGGESTIVE" if survivor_light else "PAPER_CANDIDATE_ONLY",
        "trial_count_n": trial_count,
        "multiple_testing": {
            "candidate_count_n": trial_count,
            "tested_candidate_count": len(candidates),
            "fdr_alpha": config.fdr_alpha,
            "fdr_survivors": len(fdr_survivors),
            "raw_min_p": min(p_values) if p_values else None,
            "pbo": pbo_report,
            "annual_oos_stability": oos_stability,
            "overlap_correction": {
                "enabled": config.overlap_correction,
                "raw_event_count": len(events),
                "disjoint_event_count": len(scoring_events),
                "raw_fdr_survivors": len(raw_fdr_survivors),
                "corrected_fdr_survivors": len(fdr_survivors),
                "raw_min_p": min(raw_p_values) if raw_p_values else None,
                "event_count_by_candidate": _overlap_event_count_comparison(
                    events,
                    scoring_events,
                ),
            },
        },
        "costs": {
            "fee_bps": cost_model.fee_bps,
            "slippage_bps": cost_model.slippage_bps,
            "funding": cost_model.funding_label,
        },
        "data_quality": data_quality,
        "factor_tail": [_frame_dict(frame) for frame in frames[-5:]],
        "raw_event_summary": _event_summary(events),
        "event_summary": _event_summary(scoring_events),
        "raw_candidate_statistics": raw_candidates,
        "candidate_statistics": candidates,
        "sanitized_event_sample": [_event_dict(event) for event in scoring_events[:10]],
        "safety": {
            "research_only": True,
            "live_trading": False,
            "broker_or_wallet_access": False,
            "credentials_read": False,
        },
    }


def compute_breadth_frames(
    member_bars: Mapping[str, Sequence[BreadthBar]],
    benchmark_bars: Sequence[BreadthBar],
) -> tuple[BreadthFrame, ...]:
    members = {symbol: _sorted_bars(bars) for symbol, bars in member_bars.items() if bars}
    benchmark = _sorted_bars(benchmark_bars)
    if not members or len(benchmark) < 220:
        return ()
    benchmark_by_timestamp = {bar.timestamp: bar for bar in benchmark if bar.tradable}
    member_by_timestamp = {
        symbol: {bar.timestamp: bar for bar in bars if bar.tradable}
        for symbol, bars in members.items()
    }
    member_sma = {
        symbol: {
            8: _sma_by_timestamp(bars, 8),
            21: _sma_by_timestamp(bars, 21),
            60: _sma_by_timestamp(bars, 60),
        }
        for symbol, bars in members.items()
    }
    benchmark_sma60 = _sma_by_timestamp(benchmark, 60)
    benchmark_sma200 = _sma_by_timestamp(benchmark, 200)
    timestamps = sorted(benchmark_by_timestamp)
    frames: list[BreadthFrame] = []
    benchmark_closes = [bar.close for bar in benchmark]
    benchmark_timestamps = [bar.timestamp for bar in benchmark]
    benchmark_index_by_timestamp = {
        timestamp: index for index, timestamp in enumerate(benchmark_timestamps)
    }
    close_by_timestamp = dict(zip(benchmark_timestamps, benchmark_closes, strict=True))
    for timestamp in timestamps:
        if timestamp not in close_by_timestamp:
            continue
        close = close_by_timestamp[timestamp]
        above8 = 0
        above21 = 0
        above60 = 0
        ma8_gt21 = 0
        ma21_gt60 = 0
        valid8 = 0
        valid21 = 0
        valid60 = 0
        missing = 0
        for symbol in members:
            bar = member_by_timestamp[symbol].get(timestamp)
            ma8 = member_sma[symbol][8].get(timestamp)
            ma21 = member_sma[symbol][21].get(timestamp)
            ma60 = member_sma[symbol][60].get(timestamp)
            if bar is None:
                missing += 1
                continue
            if ma8 is not None:
                valid8 += 1
                above8 += int(bar.close > ma8)
            if ma21 is not None:
                valid21 += 1
                above21 += int(bar.close > ma21)
            if ma60 is not None:
                valid60 += 1
                above60 += int(bar.close > ma60)
            if ma8 is not None and ma21 is not None:
                ma8_gt21 += int(ma8 > ma21)
            if ma21 is not None and ma60 is not None:
                ma21_gt60 += int(ma21 > ma60)
        if valid60 == 0:
            continue
        index = benchmark_index_by_timestamp[timestamp]
        index_ret_8d = _ret_at(benchmark_closes, index, 8)
        index_ret_21d = _ret_at(benchmark_closes, index, 21)
        breadth_ma21 = above21 / valid21 if valid21 else 0.0
        prior_breadth = frames[-5].breadth_ma21 if len(frames) >= 5 else None
        benchmark_ma60 = benchmark_sma60.get(timestamp)
        benchmark_ma200 = benchmark_sma200.get(timestamp)
        regime = _regime(close, benchmark_ma60, benchmark_ma200)
        frames.append(
            BreadthFrame(
                timestamp=timestamp,
                benchmark_close=close,
                breadth_ma8=above8 / valid8 if valid8 else 0.0,
                breadth_ma21=breadth_ma21,
                breadth_ma60=above60 / valid60,
                ma8_above_ma21_ratio=ma8_gt21 / valid21 if valid21 else 0.0,
                ma21_above_ma60_ratio=ma21_gt60 / valid60,
                index_ret_8d=index_ret_8d,
                index_ret_21d=index_ret_21d,
                breadth_momentum=(
                    None if prior_breadth is None else breadth_ma21 - prior_breadth
                ),
                top_divergence_20d=_top_divergence(frames, close, breadth_ma21),
                bottom_divergence_20d=_bottom_divergence(frames, close, breadth_ma21),
                regime=regime,
                constituent_count=valid60,
                missing_count=missing,
            )
        )
    return tuple(frames)


def build_event_records(
    frames: Sequence[BreadthFrame],
    *,
    config: BreadthConfig,
    cost_model: CostModel,
) -> tuple[EventRecord, ...]:
    events: list[EventRecord] = []
    baseline_by_horizon = _baseline_returns(frames, config.horizons)
    for index, frame in enumerate(frames):
        for signal, threshold in _triggered_signals(frame, config):
            for horizon in config.horizons:
                exit_index = index + horizon
                entry_index = index + 1
                if entry_index >= len(frames) or exit_index >= len(frames):
                    continue
                entry = frames[entry_index]
                exit_frame = frames[exit_index]
                gross = exit_frame.benchmark_close / entry.benchmark_close - 1.0
                net = gross - cost_model.one_way_cost * 2.0
                baseline = baseline_by_horizon.get(horizon, 0.0)
                path = [item.benchmark_close for item in frames[entry_index : exit_index + 1]]
                key = f"{signal}|thr={threshold:.4f}|h={horizon}"
                events.append(
                    EventRecord(
                        key=key,
                        signal=signal,
                        threshold=threshold,
                        horizon=horizon,
                        timestamp=frame.timestamp,
                        entry_timestamp=entry.timestamp,
                        exit_timestamp=exit_frame.timestamp,
                        regime=frame.regime,
                        forward_return_after_costs=net,
                        baseline_return=baseline,
                        excess_return=net - baseline,
                        drawdown=_path_drawdown(path),
                        breadth_ma8=frame.breadth_ma8,
                        breadth_ma21=frame.breadth_ma21,
                        breadth_ma60=frame.breadth_ma60,
                    )
                )
    return tuple(events)


def disjoint_event_records(events: Sequence[EventRecord]) -> tuple[EventRecord, ...]:
    grouped: dict[str, list[EventRecord]] = {}
    for event in events:
        grouped.setdefault(event.key, []).append(event)
    kept: list[EventRecord] = []
    for group in grouped.values():
        next_allowed_timestamp: int | None = None
        for event in sorted(group, key=lambda item: item.timestamp):
            if next_allowed_timestamp is None or event.timestamp >= next_allowed_timestamp:
                kept.append(event)
                next_allowed_timestamp = event.exit_timestamp
    return tuple(sorted(kept, key=lambda event: (event.timestamp, event.key)))


def trial_count_for_config(config: BreadthConfig) -> int:
    signal_thresholds = (
        len(config.hot_thresholds)
        + len(config.floor_thresholds)
        + len(config.floor_thresholds)
        + len(config.panic_8d_thresholds) * len(config.floor_thresholds)
        + len(config.panic_21d_thresholds) * len(config.floor_thresholds)
        + len(config.floor_thresholds)
        + len(config.hot_thresholds)
    )
    return signal_thresholds * len(config.horizons)


def _triggered_signals(
    frame: BreadthFrame,
    config: BreadthConfig,
) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for threshold in config.hot_thresholds:
        if (
            frame.breadth_ma8 > threshold
            and frame.breadth_ma21 > threshold
            and frame.breadth_ma60 > threshold
        ):
            out.append(("A_overheat", threshold))
    for threshold in config.floor_thresholds:
        floor_hit = (
            frame.breadth_ma8 < threshold
            and frame.breadth_ma21 < threshold
            and frame.breadth_ma60 < threshold
        )
        if frame.regime == "bull" and floor_hit:
            out.append(("B_slow_bull_floor", threshold))
        if frame.regime == "bear" and frame.ma8_above_ma21_ratio < threshold:
            out.append(("C_bear_deep_bottom", threshold))
        if frame.bottom_divergence_20d and frame.breadth_ma21 < threshold:
            out.append(("E_bottom_divergence", threshold))
        for panic in config.panic_8d_thresholds:
            if frame.index_ret_8d is not None and frame.index_ret_8d < -panic and floor_hit:
                out.append(("D_panic_8d", panic))
        for panic in config.panic_21d_thresholds:
            if frame.index_ret_21d is not None and frame.index_ret_21d < -panic and floor_hit:
                out.append(("D_panic_21d", panic))
    if frame.top_divergence_20d:
        for threshold in config.hot_thresholds:
            if frame.breadth_ma21 > threshold:
                out.append(("A_top_divergence", threshold))
    return tuple(out)


def _candidate_statistics(
    events: Sequence[EventRecord],
    *,
    trial_count: int,
    config: BreadthConfig,
    cost_model: CostModel,
    use_block_bootstrap_p: bool,
) -> list[dict[str, object]]:
    grouped: dict[str, list[EventRecord]] = {}
    for event in events:
        grouped.setdefault(event.key, []).append(event)
    rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        returns = tuple(event.forward_return_after_costs for event in group)
        excess = tuple(event.excess_return for event in group)
        sign_p_value = (
            sign_test_p_value(excess, alternative="greater")
            if len(group) >= config.min_events_per_candidate
            else 1.0
        )
        block_report = _paired_block_bootstrap_excess_report(group, config=config)
        p_value = (
            _float_field(block_report["p_value"], default=1.0)
            if use_block_bootstrap_p
            else sign_p_value
        )
        metrics = metrics_from_returns(
            returns,
            annualization_periods=config.annualization_periods,
            turnover=len(group) * 2.0,
            net_cost=len(group) * cost_model.one_way_cost * 2.0,
        )
        dsr_threshold = deflated_sharpe_threshold(
            trial_count=trial_count,
            observations=max(len(returns), 1),
        )
        rows.append(
            {
                "key": key,
                "signal": group[0].signal,
                "threshold": group[0].threshold,
                "horizon": group[0].horizon,
                "events": len(group),
                "mean_return": statistics.fmean(returns),
                "median_return": statistics.median(returns),
                "hit_rate": sum(1 for value in returns if value > 0) / len(returns),
                "mean_excess_return": statistics.fmean(excess),
                "median_excess_return": statistics.median(excess),
                "max_drawdown": min(event.drawdown for event in group),
                "p_value": p_value,
                "sign_test_p_value": sign_p_value,
                "block_bootstrap_p_value": block_report["p_value"],
                "block_bootstrap_valid": block_report["valid"],
                "block_bootstrap_ci_lower_gt_0": block_report["ci_lower_gt_0"],
                "block_bootstrap_block_bars": block_report["block_bars"],
                "block_bootstrap_sample_count": block_report["sample_count"],
                "risk_difference_block_bootstrap": block_report["risk_difference_reference"],
                "sharpe": metrics.sharpe,
                "deflated_sharpe_threshold": dsr_threshold,
                "deflated_sharpe_pass": metrics.sharpe > dsr_threshold and metrics.sharpe > 0.0,
            }
        )
    missing_trials = max(0, trial_count - len(rows))
    for index in range(missing_trials):
        rows.append(
            {
                "key": f"untriggered_{index}",
                "signal": "untriggered",
                "threshold": 0.0,
                "horizon": 0,
                "events": 0,
                "mean_return": 0.0,
                "median_return": 0.0,
                "hit_rate": 0.0,
                "mean_excess_return": 0.0,
                "median_excess_return": 0.0,
                "max_drawdown": 0.0,
                "p_value": 1.0,
                "sign_test_p_value": 1.0,
                "block_bootstrap_p_value": 1.0,
                "block_bootstrap_valid": False,
                "block_bootstrap_ci_lower_gt_0": False,
                "block_bootstrap_block_bars": 0,
                "block_bootstrap_sample_count": 0,
                "risk_difference_block_bootstrap": {},
                "sharpe": 0.0,
                "deflated_sharpe_threshold": deflated_sharpe_threshold(
                    trial_count=trial_count,
                    observations=1,
                ),
                "deflated_sharpe_pass": False,
            }
        )
    return rows


def _paired_block_bootstrap_excess_report(
    group: Sequence[EventRecord],
    *,
    config: BreadthConfig,
) -> dict[str, object]:
    if not group:
        return _empty_block_report(reason="empty candidate")
    horizon = max(event.horizon for event in group)
    block_bars = max(1, horizon)
    excess = tuple(event.excess_return for event in group)
    if len(excess) < max(30, block_bars):
        return _empty_block_report(
            reason="insufficient disjoint events for block bootstrap",
            block_bars=block_bars,
        )
    rng = random.Random(config.block_bootstrap_seed + sum(ord(char) for char in group[0].key))
    sample_means: list[float] = []
    block = min(block_bars, len(excess))
    for _ in range(config.block_bootstrap_samples):
        indices: list[int] = []
        while len(indices) < len(excess):
            start = rng.randint(0, len(excess) - block)
            indices.extend(range(start, start + block))
        indices = indices[: len(excess)]
        sample_means.append(statistics.fmean(excess[index] for index in indices))
    p_value = sum(1 for value in sample_means if value <= 0.0) / len(sample_means)
    ci_low = _quantile(sample_means, config.block_bootstrap_ci_alpha)
    risk_report = paired_block_bootstrap_risk_difference_test(
        excess,
        tuple(0.0 for _ in excess),
        0.0,
        0.0,
        _BlockBootstrapConfig(
            annualization_periods=config.annualization_periods,
            risk_diff_bootstrap_samples=config.block_bootstrap_samples,
            risk_diff_bootstrap_block_bars=block_bars,
            risk_diff_ci_alpha=config.block_bootstrap_ci_alpha,
            risk_diff_random_seed=config.block_bootstrap_seed,
        ),
        group[0].key,
    )
    return {
        "valid": True,
        "method": "paired_block_bootstrap_mean_excess",
        "p_value": p_value,
        "ci_lower_gt_0": ci_low > 0.0,
        "mean_ci_low": ci_low,
        "block_bars": block,
        "sample_count": config.block_bootstrap_samples,
        "risk_difference_reference": risk_report,
    }


def _empty_block_report(*, reason: str, block_bars: int = 0) -> dict[str, object]:
    return {
        "valid": False,
        "method": "paired_block_bootstrap_mean_excess",
        "reason": reason,
        "p_value": 1.0,
        "ci_lower_gt_0": False,
        "mean_ci_low": 0.0,
        "block_bars": block_bars,
        "sample_count": 0,
        "risk_difference_reference": {},
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(math.floor(q * (len(ordered) - 1))), 0), len(ordered) - 1)
    return ordered[index]


def _pbo_report(events: Sequence[EventRecord], *, config: BreadthConfig) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        grouped.setdefault(event.key, []).append(event.forward_return_after_costs)
    series = [values for values in grouped.values() if len(values) >= config.pbo_splits]
    if len(series) < 2:
        return {
            "valid": False,
            "reason": "fewer than two event candidates have enough observations for PBO",
            "pbo": None,
        }
    min_len = min(len(values) for values in series)
    if min_len < config.pbo_splits:
        return {"valid": False, "reason": "candidate observations < pbo_splits", "pbo": None}
    trials = [tuple(values[:min_len]) for values in series]
    try:
        result = pbo(trials, n_splits=config.pbo_splits)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc), "pbo": None}
    return {"valid": True, **result}


def _overlap_event_count_comparison(
    raw_events: Sequence[EventRecord],
    disjoint_events: Sequence[EventRecord],
) -> list[dict[str, object]]:
    raw_counts: dict[str, int] = {}
    disjoint_counts: dict[str, int] = {}
    for event in raw_events:
        raw_counts[event.key] = raw_counts.get(event.key, 0) + 1
    for event in disjoint_events:
        disjoint_counts[event.key] = disjoint_counts.get(event.key, 0) + 1
    return [
        {
            "key": key,
            "raw_events": raw_counts.get(key, 0),
            "disjoint_events": disjoint_counts.get(key, 0),
            "removed_events": raw_counts.get(key, 0) - disjoint_counts.get(key, 0),
        }
        for key in sorted(raw_counts)
    ]


def _annual_oos_stability(
    events: Sequence[EventRecord],
    survivor_rows: Sequence[Mapping[str, object]],
    *,
    config: BreadthConfig,
) -> dict[str, object]:
    survivor_keys = {str(row["key"]) for row in survivor_rows}
    if not survivor_keys:
        return {
            "passed": False,
            "reason": "no FDR/DSR survivors to evaluate in annual OOS windows",
            "min_oos_years": config.min_oos_years,
            "candidates": [],
        }
    grouped: dict[str, list[EventRecord]] = {}
    for event in events:
        if event.key in survivor_keys:
            grouped.setdefault(event.key, []).append(event)
    reports: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        by_year: dict[str, list[EventRecord]] = {}
        for event in group:
            year = str(datetime.fromtimestamp(event.entry_timestamp, tz=UTC).year)
            by_year.setdefault(year, []).append(event)
        years: list[dict[str, object]] = []
        for year, year_events in sorted(by_year.items()):
            returns = [event.forward_return_after_costs for event in year_events]
            excess = [event.excess_return for event in year_events]
            years.append(
                {
                    "year": year,
                    "events": len(year_events),
                    "mean_return": statistics.fmean(returns),
                    "mean_excess_return": statistics.fmean(excess),
                    "hit_rate": sum(1 for value in returns if value > 0.0) / len(returns),
                }
            )
        positive_years = [
            item
            for item in years
            if _float_field(item["mean_return"]) > 0.0
            and _float_field(item["mean_excess_return"]) > 0.0
        ]
        valid_years = [item for item in years if _float_field(item["events"]) > 0.0]
        positive_share = len(positive_years) / len(valid_years) if valid_years else 0.0
        passed = len(valid_years) >= config.min_oos_years and positive_share >= 0.60
        reports.append(
            {
                "key": key,
                "passed": passed,
                "oos_years": len(valid_years),
                "positive_oos_year_share": positive_share,
                "years": years,
            }
        )
    passed_reports = [item for item in reports if bool(item["passed"])]
    return {
        "passed": bool(passed_reports),
        "reason": (
            "at least one survivor passed annual OOS stability"
            if passed_reports
            else "no survivor had enough positive annual OOS windows"
        ),
        "min_oos_years": config.min_oos_years,
        "candidates": reports,
    }


def _baseline_returns(
    frames: Sequence[BreadthFrame],
    horizons: Sequence[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for horizon in horizons:
        values: list[float] = []
        for index in range(0, len(frames) - horizon):
            entry_index = index + 1
            exit_index = index + horizon
            if entry_index >= len(frames):
                continue
            values.append(
                frames[exit_index].benchmark_close / frames[entry_index].benchmark_close - 1.0
            )
        out[horizon] = statistics.fmean(values) if values else 0.0
    return out


def _sma_by_timestamp(bars: Sequence[BreadthBar], window: int) -> dict[int, float]:
    ordered = _sorted_bars(bars)
    out: dict[int, float] = {}
    for index, bar in enumerate(ordered):
        if index + 1 >= window:
            sample = [item.close for item in ordered[index + 1 - window : index + 1]]
            out[bar.timestamp] = statistics.fmean(sample)
    return out


def _ret_at(values: Sequence[float], index: int, lookback: int) -> float | None:
    if index < lookback or values[index - lookback] == 0:
        return None
    return values[index] / values[index - lookback] - 1.0


def _regime(close: float, ma60: float | None, ma200: float | None) -> str:
    if ma200 is None:
        return "unknown"
    if close > ma200 or (ma60 is not None and ma60 > ma200):
        return "bull"
    return "bear"


def _top_divergence(frames: Sequence[BreadthFrame], close: float, breadth_ma21: float) -> bool:
    prior = frames[-20:]
    if len(prior) < 20:
        return False
    return close >= max(frame.benchmark_close for frame in prior) and breadth_ma21 < max(
        frame.breadth_ma21 for frame in prior
    )


def _bottom_divergence(frames: Sequence[BreadthFrame], close: float, breadth_ma21: float) -> bool:
    prior = frames[-20:]
    if len(prior) < 20:
        return False
    return close <= min(frame.benchmark_close for frame in prior) and breadth_ma21 > min(
        frame.breadth_ma21 for frame in prior
    )


def _path_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


def _data_quality(
    member_bars: Mapping[str, Sequence[BreadthBar]],
    benchmark_bars: Sequence[BreadthBar],
    frames: Sequence[BreadthFrame],
    *,
    survivor_light: bool,
    data_source: str,
) -> dict[str, object]:
    missing = sum(frame.missing_count for frame in frames)
    possible = max(len(member_bars) * len(frames), 1)
    return {
        "symbols": len(member_bars),
        "benchmark_bars": len(benchmark_bars),
        "breadth_frames": len(frames),
        "missing_member_observations": missing,
        "missing_member_observation_rate": missing / possible,
        "survivorship_bias_warning": survivor_light,
        "point_in_time_constituents": False,
        "data_source": data_source,
        "closed_daily_bars_only": True,
    }


def _event_summary(events: Sequence[EventRecord]) -> dict[str, object]:
    by_signal: dict[str, int] = {}
    by_horizon: dict[str, int] = {}
    for event in events:
        by_signal[event.signal] = by_signal.get(event.signal, 0) + 1
        key = str(event.horizon)
        by_horizon[key] = by_horizon.get(key, 0) + 1
    returns = [event.forward_return_after_costs for event in events]
    return {
        "events": len(events),
        "by_signal": by_signal,
        "by_horizon": by_horizon,
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "median_return": statistics.median(returns) if returns else 0.0,
        "hit_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0,
    }


def _frame_dict(frame: BreadthFrame) -> dict[str, object]:
    return {
        "timestamp": frame.timestamp,
        "benchmark_close": frame.benchmark_close,
        "breadth_ma8": frame.breadth_ma8,
        "breadth_ma21": frame.breadth_ma21,
        "breadth_ma60": frame.breadth_ma60,
        "ma8_above_ma21_ratio": frame.ma8_above_ma21_ratio,
        "ma21_above_ma60_ratio": frame.ma21_above_ma60_ratio,
        "index_ret_8d": frame.index_ret_8d,
        "index_ret_21d": frame.index_ret_21d,
        "breadth_momentum": frame.breadth_momentum,
        "top_divergence_20d": frame.top_divergence_20d,
        "bottom_divergence_20d": frame.bottom_divergence_20d,
        "regime": frame.regime,
        "constituent_count": frame.constituent_count,
        "missing_count": frame.missing_count,
    }


def _event_dict(event: EventRecord) -> dict[str, object]:
    return {
        "key": event.key,
        "signal": event.signal,
        "threshold": event.threshold,
        "horizon": event.horizon,
        "timestamp": event.timestamp,
        "entry_timestamp": event.entry_timestamp,
        "exit_timestamp": event.exit_timestamp,
        "regime": event.regime,
        "forward_return_after_costs": event.forward_return_after_costs,
        "baseline_return": event.baseline_return,
        "excess_return": event.excess_return,
        "drawdown": event.drawdown,
        "breadth_ma8": event.breadth_ma8,
        "breadth_ma21": event.breadth_ma21,
        "breadth_ma60": event.breadth_ma60,
    }


def _float_field(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _empty_report(
    universe_name: str,
    benchmark_name: str,
    data_source: str,
    *,
    verdict: UniverseVerdict,
    status: HealthStatus,
    reason: str,
    trial_count: int,
    survivor_light: bool,
) -> dict[str, object]:
    return {
        "universe": universe_name,
        "benchmark": benchmark_name,
        "data_source": data_source,
        "verdict": verdict,
        "health_status": status,
        "reason": reason,
        "survivor_light_ceiling": survivor_light,
        "trial_count_n": trial_count,
        "multiple_testing": {
            "candidate_count_n": trial_count,
            "tested_candidate_count": 0,
            "fdr_survivors": 0,
        },
    }


def _sorted_bars(bars: Sequence[BreadthBar]) -> tuple[BreadthBar, ...]:
    return tuple(
        sorted(
            (
                bar
                for bar in bars
                if all(
                    math.isfinite(value)
                    for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)
                )
                and bar.close > 0.0
            ),
            key=lambda bar: bar.timestamp,
        )
    )
