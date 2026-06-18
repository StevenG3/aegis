from __future__ import annotations

import pytest

from aegis.combo_indicator_search import ComboBar, ComboCostModel
from aegis.combo_scorecard import (
    PredeclaredCombo,
    ScorecardCandidate,
    ScorecardConfig,
    _locked_oos_scorecard,
    composite_score,
    predeclared_scorecard_combos,
    run_combo_scorecard,
    simulate_combo,
    trade_scorecard,
)


def _bars(closes: list[float]) -> list[ComboBar]:
    return [
        ComboBar(
            timestamp=index,
            open=float(close),
            high=float(close) * 1.02,
            low=float(close) * 0.98,
            close=float(close),
            volume=1000.0 + index,
        )
        for index, close in enumerate(closes)
    ]


def _test_config() -> ScorecardConfig:
    return ScorecardConfig(
        train_bars=30,
        test_bars=10,
        step_bars=10,
        locked_oos_fraction=0.30,
        min_is_folds=3,
        min_trades=2,
    )


def test_predeclared_combo_set_has_thesis_and_expected_n() -> None:
    combos = predeclared_scorecard_combos()
    bars = _bars([100 + index * 0.1 for index in range(150)])
    report = run_combo_scorecard(
        {"BTC/USDT": bars, "ETH/USDT": bars},
        config=_test_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert len(combos) == 8
    assert all(combo.thesis and combo.params for combo in combos)
    assert report.candidate_count_n == 16
    assert report.multiple_testing["trial_count_n"] == 16


def test_trade_scorecard_reports_expectancy_profit_factor_and_loss_streak() -> None:
    scorecard = trade_scorecard([0.10, -0.05, -0.02, 0.04])

    assert scorecard.total_trades == 4
    assert scorecard.win_rate == pytest.approx(0.5)
    assert scorecard.win_loss_ratio == pytest.approx(0.07 / 0.035)
    assert scorecard.expectancy_per_trade == pytest.approx(0.5 * 0.07 - 0.5 * 0.035)
    assert scorecard.profit_factor == pytest.approx(0.14 / 0.07)
    assert scorecard.max_consecutive_losses == 2


def test_no_lookahead_uses_prior_close_signal_and_next_open_return() -> None:
    bars = _bars([100, 100, 110, 80, 60, 66])
    combo = PredeclaredCombo(
        key="prior_close",
        thesis="synthetic",
        params={},
        signal=lambda _bars, index: _bars[index].close > 100,
        warmup=1,
    )
    result = simulate_combo(
        bars,
        combo,
        start=3,
        end=5,
        config=_test_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert result.first_execution_index == 3
    assert result.positions == (1, 0)
    assert result.returns[0] == pytest.approx(60 / 80 - 1)


def test_high_win_rate_negative_expectancy_is_no_go() -> None:
    trade = trade_scorecard([0.01, 0.01, 0.01, -0.10])
    assert trade.win_rate == pytest.approx(0.75)
    assert trade.expectancy_per_trade < 0


def test_hard_gate_requires_expectancy_profit_factor_benchmark_fdr_and_trades() -> None:
    bars = _bars([100, 120, 100, 80, 100, 120, 100, 80, 100, 120, 100, 80, 100])
    combo = PredeclaredCombo(
        key="always_long",
        thesis="synthetic",
        params={},
        signal=lambda _bars, _index: True,
        warmup=1,
    )
    scorecard = _locked_oos_scorecard(
        ScorecardCandidate("BTC/USDT", combo),
        bars,
        None,
        locked_oos_start=2,
        config=ScorecardConfig(min_trades=1),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
        fdr_discovery=False,
    )

    assert not scorecard.gate_checks["fdr_discovery"]
    assert scorecard.verdict == "NO_GO"


def test_composite_score_is_bounded_and_recomputable() -> None:
    trade = trade_scorecard([0.05, -0.01, 0.02])
    score = composite_score(
        trade,
        metrics=type(
            "Metrics",
            (),
            {"sharpe": 1.0, "max_drawdown": -0.2},
        )(),
        excess_return=0.1,
    )

    assert 0 <= score <= 100
