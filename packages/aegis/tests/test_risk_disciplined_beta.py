from __future__ import annotations

import pytest

from aegis.combo_indicator_search import ComboBar, ComboCostModel
from aegis.risk_disciplined_beta import (
    RiskBetaConfig,
    RiskCandidate,
    _target_weights,
    buy_hold_simulation,
    equal_weight_buy_hold,
    predeclared_risk_candidates,
    risk_metrics,
    run_risk_disciplined_beta,
    simulate_candidate,
)


def _bars(closes: list[float]) -> list[ComboBar]:
    return [
        ComboBar(
            timestamp=index,
            open=float(close),
            high=float(close) * 1.01,
            low=float(close) * 0.99,
            close=float(close),
            volume=1000.0 + index,
        )
        for index, close in enumerate(closes)
    ]


def _config() -> RiskBetaConfig:
    return RiskBetaConfig(
        train_bars=30,
        test_bars=10,
        step_bars=10,
        locked_oos_fraction=0.30,
        min_is_folds=3,
        oos_folds=2,
    )


def test_predeclared_risk_configurations_count_n() -> None:
    candidates = predeclared_risk_candidates()
    bars = _bars([100 + index * 0.1 for index in range(150)])
    report = run_risk_disciplined_beta(
        {"BTC/USDT": bars, "ETH/USDT": bars, "SOL/USDT": bars},
        config=_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )

    assert len(candidates) == 62
    assert report.candidate_count_n == 62
    assert report.multiple_testing["trial_count_n"] == 62


def test_volatility_target_uses_lagged_data_only() -> None:
    stable_then_crash = _bars([100.0] * 35 + [50.0])
    crash_then_stable = _bars([100.0] * 34 + [50.0, 50.0])
    candidate = RiskCandidate(
        key="test",
        method="vol_target",
        thesis="test",
        symbols=("BTC/USDT",),
        target_vol=0.40,
        lookback=20,
        max_exposure=1.0,
        rebalance_days=1,
    )

    before_crash_weight = _target_weights(candidate, [stable_then_crash], 34, _config())[0]
    after_crash_weight = _target_weights(candidate, [crash_then_stable], 35, _config())[0]

    assert before_crash_weight == pytest.approx(1.0)
    assert after_crash_weight < before_crash_weight


def test_turnover_costs_reduce_dynamic_returns() -> None:
    bars = _bars([100, 110, 90, 120, 80, 130, 70, 140, 60, 150, 55, 160] * 20)
    candidate = RiskCandidate(
        key="test",
        method="vol_target",
        thesis="test",
        symbols=("BTC/USDT",),
        target_vol=0.30,
        lookback=3,
        max_exposure=1.0,
        rebalance_days=1,
    )
    no_cost = simulate_candidate(
        candidate,
        {"BTC/USDT": bars},
        start=10,
        end=80,
        config=_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )
    with_cost = simulate_candidate(
        candidate,
        {"BTC/USDT": bars},
        start=10,
        end=80,
        config=_config(),
        cost_model=ComboCostModel(fee_bps=10, slippage_bps=5),
    )

    assert with_cost.metrics.net_cost > 0
    assert with_cost.metrics.annualized_return < no_cost.metrics.annualized_return


def test_three_benchmarks_are_available() -> None:
    bars = _bars([100 + index for index in range(120)])
    cost_model = ComboCostModel(fee_bps=0, slippage_bps=0)

    single = buy_hold_simulation(bars, start=10, end=100, config=_config(), cost_model=cost_model)
    equal = equal_weight_buy_hold(
        {"BTC/USDT": bars, "ETH/USDT": bars, "SOL/USDT": bars},
        ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
        start=10,
        end=100,
        config=_config(),
        cost_model=cost_model,
    )
    static_60_40 = buy_hold_simulation(
        bars,
        start=10,
        end=100,
        config=_config(),
        cost_model=cost_model,
        allocation=0.60,
    )

    assert single.metrics.annualized_turnover == pytest.approx(365 / 45)
    assert equal.metrics.annualized_return == pytest.approx(single.metrics.annualized_return)
    assert static_60_40.metrics.realized_volatility < single.metrics.realized_volatility


def test_risk_metrics_include_worst_month_and_ulcer_index() -> None:
    metrics = risk_metrics(
        [0.01, -0.02, 0.01, -0.03] * 20,
        annualization_periods=365,
        target_volatility=0.30,
        turnover=2,
        net_cost=0.01,
    )

    assert metrics.worst_month < 0
    assert metrics.ulcer_index > 0
    assert metrics.target_volatility == pytest.approx(0.30)


def test_report_records_gate_checks_for_each_candidate() -> None:
    bars = _bars([100 + index * 0.1 for index in range(150)])
    report = run_risk_disciplined_beta(
        {"BTC/USDT": bars, "ETH/USDT": bars, "SOL/USDT": bars},
        config=_config(),
        cost_model=ComboCostModel(fee_bps=0, slippage_bps=0),
    )
    first = next(iter(report.results.values()))
    gate_checks = first["gate_checks"]

    assert isinstance(gate_checks, dict)
    assert {
        "drawdown_reduction_ge_20pct",
        "calmar_gt_buy_hold",
        "sortino_gt_buy_hold",
        "realized_vol_near_target",
        "net_cost_positive_and_counted",
        "oos_fold_pass_rate",
        "fdr_discovery",
    } <= set(gate_checks)
