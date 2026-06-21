from __future__ import annotations

from dataclasses import replace

import pytest

from aegis.backtest_core import CostModel
from aegis.market_breadth_top_bottom import (
    BreadthBar,
    BreadthConfig,
    EventRecord,
    build_event_records,
    compute_breadth_frames,
    disjoint_event_records,
    run_market_breadth_study,
    trial_count_for_config,
)


def _bars(
    *,
    count: int = 280,
    start: float = 100.0,
    drift: float = 0.001,
    wave: float = 0.0,
    missing_at: set[int] | None = None,
    untradable_at: set[int] | None = None,
) -> list[BreadthBar]:
    missing = missing_at or set()
    untradable = untradable_at or set()
    price = start
    out: list[BreadthBar] = []
    for index in range(count):
        if index in missing:
            price *= 1.0 + drift
            continue
        cycle = wave if index % 2 == 0 else -wave
        open_price = price
        close = max(1.0, open_price * (1.0 + drift + cycle))
        out.append(
            BreadthBar(
                timestamp=index,
                open=open_price,
                high=max(open_price, close) * 1.01,
                low=min(open_price, close) * 0.99,
                close=close,
                volume=1_000_000.0 + index,
                tradable=index not in untradable,
            )
        )
        price = close
    return out


def test_breadth_factors_use_closed_current_and_past_bars_only() -> None:
    benchmark = _bars()
    members = {
        "AAA": _bars(drift=0.002),
        "BBB": _bars(drift=0.0015),
        "CCC": _bars(drift=-0.0005),
    }
    original = compute_breadth_frames(members, benchmark)
    mutated = dict(members)
    changed_symbol = list(mutated["AAA"])
    changed_symbol[240] = replace(changed_symbol[240], close=changed_symbol[240].close * 100.0)
    mutated["AAA"] = changed_symbol

    changed = compute_breadth_frames(mutated, benchmark)

    assert original[100].timestamp == changed[100].timestamp
    assert changed[100].breadth_ma8 == original[100].breadth_ma8
    assert changed[100].breadth_ma21 == original[100].breadth_ma21
    assert changed[100].breadth_ma60 == original[100].breadth_ma60


def test_missing_or_untradable_members_are_not_counted_as_passing() -> None:
    benchmark = _bars()
    members = {
        "AAA": _bars(drift=0.002),
        "BBB": _bars(drift=0.001, missing_at={120}),
        "CCC": _bars(drift=0.001, untradable_at={120}),
    }

    frames = compute_breadth_frames(members, benchmark)
    target = next(frame for frame in frames if frame.timestamp == 120)

    assert target.missing_count == 2
    assert target.constituent_count == 1


def test_events_execute_t_plus_one_after_signal_timestamp() -> None:
    benchmark = _bars(drift=0.002)
    members = {
        "AAA": _bars(drift=0.003),
        "BBB": _bars(drift=0.0025),
        "CCC": _bars(drift=0.002),
    }
    frames = compute_breadth_frames(members, benchmark)
    config = BreadthConfig(
        horizons=(5,),
        hot_thresholds=(0.50,),
        floor_thresholds=(0.10,),
        min_events_per_candidate=1,
    )

    events = build_event_records(
        frames,
        config=config,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert events
    assert events[0].entry_timestamp > events[0].timestamp
    assert events[0].exit_timestamp > events[0].entry_timestamp


def test_full_scope_trial_count_and_survivor_ceiling_are_reported() -> None:
    config = BreadthConfig(
        horizons=(5, 10),
        hot_thresholds=(0.70,),
        floor_thresholds=(0.30, 0.20),
        panic_8d_thresholds=(0.03,),
        panic_21d_thresholds=(0.05,),
        min_events_per_candidate=1,
    )
    result = run_market_breadth_study(
        universe_name="synthetic_us",
        member_bars={
            "AAA": _bars(drift=0.003),
            "BBB": _bars(drift=0.002),
            "CCC": _bars(drift=0.001),
        },
        benchmark_bars=_bars(drift=0.002),
        config=config,
        cost_model=CostModel(fee_bps=10.0, slippage_bps=5.0),
        data_source="synthetic",
        benchmark_name="SYNTH",
        survivor_light=True,
    )

    multiple = result["multiple_testing"]
    assert result["survivor_light_ceiling"] is True
    assert result["verdict"] != "ROBUST"
    assert result["trial_count_n"] == trial_count_for_config(config)
    assert multiple["candidate_count_n"] == trial_count_for_config(config)
    assert multiple["tested_candidate_count"] == trial_count_for_config(config)


def test_t_plus_one_return_does_not_use_signal_day_close_as_entry() -> None:
    frames = compute_breadth_frames(
        {
            "AAA": _bars(drift=0.003),
            "BBB": _bars(drift=0.003),
            "CCC": _bars(drift=0.003),
        },
        _bars(drift=0.002),
    )
    config = BreadthConfig(
        horizons=(5,),
        hot_thresholds=(0.50,),
        floor_thresholds=(0.10,),
        min_events_per_candidate=1,
    )
    events = build_event_records(
        frames,
        config=config,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    first = events[0]
    signal_index = next(
        index for index, frame in enumerate(frames) if frame.timestamp == first.timestamp
    )
    expected = (
        frames[signal_index + first.horizon].benchmark_close
        / frames[signal_index + 1].benchmark_close
        - 1.0
    )

    assert first.forward_return_after_costs == pytest.approx(expected)


def test_disjoint_events_are_deduped_per_trial_key_and_horizon() -> None:
    base = BreadthBar(timestamp=0, open=1, high=1, low=1, close=1, volume=1)
    events = []
    for timestamp in (10, 12, 15, 19, 20):
        events.append(
            _event(
                key="A|thr=0.7000|h=5",
                timestamp=timestamp,
                horizon=5,
            )
        )
    events.append(_event(key="B|thr=0.7000|h=5", timestamp=12, horizon=5))
    assert base.close == 1

    kept = disjoint_event_records(events)

    assert [
        event.timestamp for event in kept if event.key == "A|thr=0.7000|h=5"
    ] == [10, 15, 20]
    assert [event.timestamp for event in kept if event.key == "B|thr=0.7000|h=5"] == [12]


def test_overlap_correction_is_explicit_and_reports_raw_vs_disjoint_counts() -> None:
    member_bars = {
        "AAA": _bars(drift=0.003),
        "BBB": _bars(drift=0.003),
        "CCC": _bars(drift=0.003),
    }
    benchmark_bars = _bars(drift=0.002)
    cost_model = CostModel(fee_bps=0.0, slippage_bps=0.0)

    raw = run_market_breadth_study(
        universe_name="synthetic",
        member_bars=member_bars,
        benchmark_bars=benchmark_bars,
        config=BreadthConfig(
            horizons=(5,),
            hot_thresholds=(0.50,),
            floor_thresholds=(0.10,),
            min_events_per_candidate=1,
            overlap_correction=False,
        ),
        cost_model=cost_model,
        data_source="synthetic",
        benchmark_name="SYNTH",
        survivor_light=True,
    )
    corrected = run_market_breadth_study(
        universe_name="synthetic",
        member_bars=member_bars,
        benchmark_bars=benchmark_bars,
        config=BreadthConfig(
            horizons=(5,),
            hot_thresholds=(0.50,),
            floor_thresholds=(0.10,),
            min_events_per_candidate=1,
            overlap_correction=True,
        ),
        cost_model=cost_model,
        data_source="synthetic",
        benchmark_name="SYNTH",
        survivor_light=True,
    )

    raw_multiple = raw["multiple_testing"]
    corrected_multiple = corrected["multiple_testing"]
    raw_overlap = raw_multiple["overlap_correction"]
    corrected_overlap = corrected_multiple["overlap_correction"]
    assert raw_overlap["enabled"] is False
    assert corrected_overlap["enabled"] is True
    assert corrected_overlap["disjoint_event_count"] < corrected_overlap["raw_event_count"]
    assert raw["event_summary"]["events"] == raw_overlap["raw_event_count"]


def test_overlap_correction_uses_block_p_value_for_fdr() -> None:
    result = run_market_breadth_study(
        universe_name="small_sample",
        member_bars={
            "AAA": _bars(drift=0.003, count=230),
            "BBB": _bars(drift=0.003, count=230),
            "CCC": _bars(drift=0.003, count=230),
        },
        benchmark_bars=_bars(drift=0.002, count=230),
        config=BreadthConfig(
            horizons=(60,),
            hot_thresholds=(0.50,),
            floor_thresholds=(0.10,),
            min_events_per_candidate=1,
            overlap_correction=True,
        ),
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
        data_source="synthetic",
        benchmark_name="SYNTH",
        survivor_light=True,
    )

    rows = [
        row
        for row in result["candidate_statistics"]
        if row["signal"] != "untriggered" and row["events"] > 0
    ]
    assert rows
    assert all(row["p_value"] == row["block_bootstrap_p_value"] for row in rows)
    assert any(row["block_bootstrap_valid"] is False for row in rows)


def _event(*, key: str, timestamp: int, horizon: int) -> EventRecord:
    return EventRecord(
        key=key,
        signal=key.split("|", maxsplit=1)[0],
        threshold=0.7,
        horizon=horizon,
        timestamp=timestamp,
        entry_timestamp=timestamp + 1,
        exit_timestamp=timestamp + horizon,
        regime="bull",
        forward_return_after_costs=0.01,
        baseline_return=0.0,
        excess_return=0.01,
        drawdown=0.0,
        breadth_ma8=0.8,
        breadth_ma21=0.8,
        breadth_ma60=0.8,
    )
