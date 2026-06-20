from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from aegis.backtest_core import (
    BacktestDiscipline,
    HypothesisSpec,
    StandardVerdict,
    benjamini_hochberg,
    metrics_from_returns,
    paired_block_bootstrap_risk_difference_test,
    run_backtest,
    sign_test_p_value,
    trade_scorecard,
)


def test_benjamini_hochberg_supports_legacy_rank_and_cutoff_modes() -> None:
    p_values = [0.01, 0.01, 0.04]

    assert benjamini_hochberg(p_values, alpha=0.03, tie_policy="rank") == [
        True,
        True,
        False,
    ]
    assert benjamini_hochberg(p_values, alpha=0.03, tie_policy="p_value_cutoff") == [
        True,
        True,
        False,
    ]


def test_sign_test_supports_one_sided_and_two_sided_modes() -> None:
    excess = [1.0, 1.0, 1.0, -1.0]

    assert sign_test_p_value(excess, alternative="greater") == pytest.approx(0.3125)
    assert sign_test_p_value(excess, alternative="two-sided") == pytest.approx(0.625)


def test_metrics_from_returns_preserves_combo_metric_shape() -> None:
    metrics = metrics_from_returns(
        [0.10, -0.05, 0.02],
        annualization_periods=365,
        turnover=2.0,
        net_cost=0.01,
        oos_vs_buy_hold_window_win_rate=0.5,
    )

    assert metrics.total_return == pytest.approx((1.10 * 0.95 * 1.02) - 1.0)
    assert metrics.max_drawdown == pytest.approx(-0.05)
    assert metrics.oos_vs_buy_hold_window_win_rate == 0.5
    assert metrics.net_cost == 0.01


def test_trade_scorecard_matches_legacy_profit_factor_edge_cases() -> None:
    scorecard = trade_scorecard([0.1, -0.05, -0.02, 0.03])

    assert scorecard.total_trades == 4
    assert scorecard.win_rate == 0.5
    assert scorecard.profit_factor == pytest.approx(0.13 / 0.07)
    assert scorecard.max_consecutive_losses == 2
    assert math.isinf(trade_scorecard([0.1]).profit_factor)


@dataclass(frozen=True)
class _RiskDiffConfig:
    annualization_periods: int = 365
    risk_diff_bootstrap_samples: int = 20
    risk_diff_bootstrap_block_bars: int = 5
    risk_diff_ci_alpha: float = 0.05
    risk_diff_random_seed: int = 47


def test_paired_block_bootstrap_risk_difference_smoke() -> None:
    strategy = [0.002] * 40
    benchmark = [0.001] * 40

    result = paired_block_bootstrap_risk_difference_test(
        strategy,
        benchmark,
        0.3,
        0.3,
        _RiskDiffConfig(),
        "candidate",
    )

    assert result["valid"] is True
    assert result["method"] == "paired_block_bootstrap"
    assert result["sample_count"] == 20
    assert "p_value" in result


def test_run_backtest_standardizes_payload_and_injects_trial_count() -> None:
    spec = HypothesisSpec(
        key="unit_combo",
        hypothesis_type="combo",
        universe=("BTC/USDT",),
        predeclared_signals=("a", "b"),
        params={"lookback": 20},
        cost_model={"fee_bps": 1.0},
        benchmark="buy_and_hold",
        data_source="synthetic",
        trial_count_n=2,
        runner=lambda: {
            "status": "OK",
            "verdict": "NO_EDGE",
            "reason": "failed gates",
            "multiple_testing": {"method": "BH-FDR"},
            "safety": {"paper_only": True},
        },
    )

    result = run_backtest(spec)

    assert isinstance(result.payload, dict)
    assert result.payload["verdict"] == "NO_EDGE"
    assert result.verdict.state == "NO_EDGE"
    assert result.verdict.candidate_count_n == 2
    assert result.verdict.multiple_testing["hypothesis_trial_count_n"] == 2


def test_run_backtest_rejects_missing_discipline() -> None:
    spec = HypothesisSpec(
        key="bad",
        hypothesis_type="factor",
        universe=("AAPL",),
        predeclared_signals=("value",),
        params={},
        cost_model={},
        benchmark="equal_weight",
        data_source="synthetic",
        trial_count_n=1,
        discipline=BacktestDiscipline(t_plus_1_execution=False),
        runner=lambda: {"status": "OK", "verdict": "NO_EDGE"},
    )

    with pytest.raises(ValueError, match="t_plus_1_execution"):
        run_backtest(spec)


def test_run_backtest_applies_survivor_light_standard_ceiling() -> None:
    spec = HypothesisSpec(
        key="survivor_factor",
        hypothesis_type="factor",
        universe=("AAPL",),
        predeclared_signals=("value",),
        params={},
        cost_model={},
        benchmark="equal_weight",
        data_source="synthetic",
        trial_count_n=1,
        survivor_light=True,
        runner=lambda: {"status": "OK", "verdict": "EDGE", "reason": "positive"},
    )

    result = run_backtest(spec)

    assert result.verdict.verdict == "SUGGESTIVE_NEEDS_PAID_CONFIRM"
    assert result.verdict.survivor_ceiling_applied is True


def test_run_backtest_accepts_custom_verdict_adapter() -> None:
    spec = HypothesisSpec(
        key="custom",
        hypothesis_type="event",
        universe=("BTC/USDT",),
        predeclared_signals=("event",),
        params={},
        cost_model={},
        benchmark="cash",
        data_source="synthetic",
        trial_count_n=3,
        runner=lambda: object(),
        verdict_adapter=lambda _payload, _spec: StandardVerdict(
            state="INSUFFICIENT",
            verdict="INSUFFICIENT",
            reason="too few events",
        ),
    )

    result = run_backtest(spec)

    assert result.verdict.state == "INSUFFICIENT"
    assert result.verdict.candidate_count_n == 3
