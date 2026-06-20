from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from aegis.backtest_core import (
    benjamini_hochberg,
    metrics_from_returns,
    paired_block_bootstrap_risk_difference_test,
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
