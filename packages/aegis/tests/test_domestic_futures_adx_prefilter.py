from __future__ import annotations

from dataclasses import replace

import pytest

from aegis.backtest_core import CostModel
from aegis.domestic_futures_adx_prefilter import (
    CandidateKey,
    FuturesAdxConfig,
    FuturesBar,
    run_underlying_adx_prefilter,
    simulate_candidate,
    trial_count,
    validate_bars,
)


def _bars(count: int = 1_150, *, drift: float = 0.001, wave: float = 0.0) -> list[FuturesBar]:
    price = 100.0
    out: list[FuturesBar] = []
    for index in range(count):
        cycle = wave if index % 2 == 0 else -wave
        open_price = price
        close = max(1.0, open_price * (1.0 + drift + cycle))
        high = max(open_price, close) * 1.02
        low = min(open_price, close) * 0.98
        out.append(
            FuturesBar(
                timestamp=index,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1_000.0 + index,
                roll_marker=index in {252, 504, 756},
            )
        )
        price = close
    return out


def test_trial_count_keeps_grid_small() -> None:
    assert trial_count(tuple(f"S{i}" for i in range(7)), FuturesAdxConfig()) == 84


def test_validate_bars_fails_closed_for_missing_or_short_symbols() -> None:
    report = validate_bars(
        {"AU": _bars(100), "CU": _bars()},
        required_symbols=("AU", "CU", "TA"),
        config=FuturesAdxConfig(min_bars=900),
    )

    assert report["status"] == "INSUFFICIENT"
    assert report["missing_symbols"] == ["TA"]
    assert report["too_short_symbols"] == {"AU": 100}


def test_signal_uses_prior_bar_and_executes_next_open() -> None:
    bars = _bars(drift=0.003)
    key = CandidateKey("AU", 5.0, 13, 34, "long_flat")
    result = simulate_candidate(
        bars,
        key=key,
        start=120,
        end=220,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert result.entry_timestamps
    first_entry = result.entry_timestamps[0]
    assert first_entry > bars[120 - 1].timestamp
    assert len(result.returns) == len(result.benchmark_returns)


def test_future_bar_mutation_does_not_change_past_returns() -> None:
    bars = _bars(drift=0.002)
    key = CandidateKey("M", 5.0, 13, 34, "long_flat")
    original = simulate_candidate(
        bars,
        key=key,
        start=120,
        end=220,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )
    mutated = list(bars)
    mutated[800] = replace(mutated[800], close=mutated[800].close * 100.0)
    changed = simulate_candidate(
        mutated,
        key=key,
        start=120,
        end=220,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )

    assert changed.returns == original.returns
    assert changed.positions == original.positions


def test_costs_reduce_strategy_returns() -> None:
    bars = _bars(drift=0.003)
    key = CandidateKey("Y", 5.0, 13, 34, "long_short")
    free = simulate_candidate(
        bars,
        key=key,
        start=120,
        end=320,
        cost_model=CostModel(fee_bps=0.0, slippage_bps=0.0),
    )
    costly = simulate_candidate(
        bars,
        key=key,
        start=120,
        end=320,
        cost_model=CostModel(fee_bps=5.0, slippage_bps=5.0),
    )

    assert sum(costly.costs) > 0
    assert sum(costly.returns) < sum(free.returns)


def test_run_prefilter_reports_no_edge_without_fdr_survivors() -> None:
    symbols = tuple(f"S{i}" for i in range(7))
    bars = {symbol: _bars(drift=0.0, wave=0.01) for symbol in symbols}
    report = run_underlying_adx_prefilter(
        bars,
        required_symbols=symbols,
        config=FuturesAdxConfig(
            adx_thresholds=(18.0,),
            ema_pairs=((13, 34),),
            trade_modes=("long_flat",),
            train_bars=260,
            test_bars=120,
            step_bars=120,
            min_bars=900,
        ),
        cost_model=CostModel(fee_bps=5.0, slippage_bps=5.0),
        data_source="synthetic",
        roll_method="synthetic_back_adjusted",
    )

    assert report["status"] == "OK"
    assert report["standard_verdict"] in {"NO_EDGE", "INSUFFICIENT"}
    assert report["candidate_count_n"] == 7
    assert report["fdr_survivors"] == 0
    assert report["best_candidate"] is None
    assert report["diagnostic_best_candidate"] is not None


def test_invalid_same_or_reversed_ema_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="fast"):
        simulate_candidate(
            _bars(),
            key=CandidateKey("AU", 20.0, 34, 13, "long_flat"),
            start=100,
            end=200,
            cost_model=CostModel(),
        )
