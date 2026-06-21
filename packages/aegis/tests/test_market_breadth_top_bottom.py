from __future__ import annotations

from dataclasses import replace

import pytest

from aegis.backtest_core import CostModel
from aegis.market_breadth_top_bottom import (
    BreadthBar,
    BreadthConfig,
    build_event_records,
    compute_breadth_frames,
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
