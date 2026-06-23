from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import pytest

from aegis.backtest_core import (
    BacktestDiscipline,
    CostModel,
    HypothesisSpec,
    StandardVerdict,
    benjamini_hochberg,
    metrics_from_returns,
    paired_block_bootstrap_risk_difference_test,
    pbo,
    run_backtest,
    sign_test_p_value,
    trade_scorecard,
)


def test_cost_model_one_way_and_round_trip_semantics() -> None:
    cost_model = CostModel(fee_bps=10.0, slippage_bps=5.0)

    assert cost_model.one_way_cost == pytest.approx(0.0015)
    assert cost_model.round_trip_cost == pytest.approx(0.0030)
    assert cost_model.round_trip_bps == pytest.approx(30.0)
    assert cost_model.round_trip_cost == pytest.approx(2.0 * cost_model.one_way_cost)
    assert cost_model.round_trip_bps == pytest.approx(
        2.0 * (cost_model.fee_bps + cost_model.slippage_bps)
    )


def test_complete_open_close_cost_equals_round_trip() -> None:
    cost_model = CostModel(fee_bps=10.0, slippage_bps=5.0)
    position_changes = (1.0, 1.0)

    total_cost = sum(change * cost_model.one_way_cost for change in position_changes)

    assert total_cost == pytest.approx(cost_model.round_trip_cost)
    assert total_cost == pytest.approx(cost_model.round_trip_bps / 10_000.0)


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


def test_sign_test_large_sample_uses_stable_approximation() -> None:
    excess = [1.0] * 600 + [-1.0] * 400

    greater = sign_test_p_value(excess, alternative="greater")
    two_sided = sign_test_p_value(excess, alternative="two-sided")

    assert 0.0 < greater < 1e-9
    assert 0.0 < two_sided < 1e-8


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


def _slice_memorizing_trials(*, n_splits: int = 8, segment_len: int = 8) -> list[list[float]]:
    trials: list[list[float]] = []
    for winner_slice in range(n_splits):
        values: list[float] = []
        for split in range(n_splits):
            level = 0.04 if split == winner_slice else -0.01
            values.extend(
                level + (0.0001 if index % 2 else -0.0001) for index in range(segment_len)
            )
        trials.append(values)
    return trials


def _robust_trials(*, n_splits: int = 8, segment_len: int = 8) -> list[list[float]]:
    observations = n_splits * segment_len
    robust = [0.004 + (0.0005 if index % 2 else -0.0005) for index in range(observations)]
    trials = [robust]
    for weak_slice in range(1, n_splits):
        values: list[float] = []
        for split in range(n_splits):
            level = 0.003 if split == weak_slice else -0.002
            values.extend(
                level + (0.0002 if index % 2 else -0.0002) for index in range(segment_len)
            )
        trials.append(values)
    return trials


def test_pbo_detects_known_overfit_synthetic_trials() -> None:
    result = pbo(_slice_memorizing_trials(), n_splits=8)
    logits = cast(list[float], result["logits"])

    assert result["method"] == "CSCV_PBO"
    assert result["split_count"] == 70
    assert cast(float, result["pbo"]) > 0.75
    assert min(logits) < 0


def test_pbo_stays_low_for_robust_synthetic_trials() -> None:
    result = pbo(_robust_trials(), n_splits=8)

    assert cast(float, result["pbo"]) < 0.25
    assert result["trial_count"] == 8
    assert cast(float, result["dsr_sharpe_threshold"]) > 0


def test_pbo_validates_split_boundaries() -> None:
    with pytest.raises(ValueError, match="even"):
        pbo([[0.1] * 8, [0.2] * 8], n_splits=5)
    with pytest.raises(ValueError, match="at least n_splits"):
        pbo([[0.1] * 3, [0.2] * 3], n_splits=4)
    with pytest.raises(ValueError, match="same observation count"):
        pbo([[0.1] * 8, [0.2] * 7], n_splits=4)


def test_pbo_is_deterministic() -> None:
    trials = _slice_memorizing_trials()

    assert pbo(trials, n_splits=8) == pbo(trials, n_splits=8)
