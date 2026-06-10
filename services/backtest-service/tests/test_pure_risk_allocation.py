from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def load_candidate_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    path = root / "incubating" / "pure_risk_allocation.py"
    spec = importlib.util.spec_from_file_location("pure_risk_allocation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pure_risk_allocation"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_two_asset_frame(rows: int = 760) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    stable: list[float] = []
    volatile: list[float] = []
    stable_price = 100.0
    volatile_price = 100.0
    for item in range(rows):
        stable_price *= 1.0006
        if item % 17 == 0:
            volatile_price *= 0.97
        elif item % 11 == 0:
            volatile_price *= 1.04
        else:
            volatile_price *= 1.0008
        stable.append(stable_price)
        volatile.append(volatile_price)
    return pd.DataFrame({"stable": stable, "volatile": volatile}, index=index)


def test_pure_risk_candidate_has_no_directional_timing_and_dual_benchmarks() -> None:
    module = load_candidate_module()
    closes = synthetic_two_asset_frame()
    strategy_config = module.StrategyConfig(
        method="risk_parity_vol_target",
        vol_window=20,
        target_ann_vol=0.15,
        rebalance_bars=20,
        min_asset_weight=0.0,
        max_asset_weight=0.80,
        max_gross_exposure=1.00,
        drawdown_stop=0.12,
        drawdown_reduction=0.0,
        drawdown_cooldown_bars=10,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    evaluation_config = module.EvaluationConfig(
        start="2021-01-01",
        end="2023-01-31",
        timeframe="1d",
        train_bars=252,
        test_bars=126,
        step_bars=126,
        min_drawdown_reduction_ratio=0.20,
        max_sharpe_shortfall=0.0,
        max_calmar_shortfall=0.0,
        max_parameter_trials=12,
    )

    strategy = module.simulate_strategy(closes, strategy_config)
    buy_hold = module.simulate_buy_hold(closes)
    equal_weight = module.simulate_equal_weight_rebalanced(closes, strategy_config)
    report = module.dual_benchmark_report(strategy, buy_hold, equal_weight, evaluation_config)

    assert module.safety_statement()["directional_timing_added"] is False
    assert "trend" not in strategy["kind"]
    assert set(report["benchmarks"]) == {"buy_hold", "equal_weight"}
    assert strategy["max_gross_exposure"] <= 1.0
    assert strategy["total_cost_pct"] > 0


def test_walk_forward_reports_dual_benchmark_oos_summary() -> None:
    module = load_candidate_module()
    closes = synthetic_two_asset_frame()
    strategy_config = module.StrategyConfig(
        method="risk_parity_vol_target",
        vol_window=20,
        target_ann_vol=0.15,
        rebalance_bars=20,
        min_asset_weight=0.0,
        max_asset_weight=0.80,
        max_gross_exposure=1.00,
        drawdown_stop=0.12,
        drawdown_reduction=0.0,
        drawdown_cooldown_bars=10,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    evaluation_config = module.EvaluationConfig(
        start="2021-01-01",
        end="2023-01-31",
        timeframe="1d",
        train_bars=252,
        test_bars=126,
        step_bars=126,
        min_drawdown_reduction_ratio=0.20,
        max_sharpe_shortfall=0.0,
        max_calmar_shortfall=0.0,
        max_parameter_trials=12,
    )

    report = module.run_portfolio_walk_forward(closes, strategy_config, evaluation_config)

    assert report["status"] == "OK"
    assert report["summary"]["parameter_trials"] <= 12
    assert "buy_hold" in report["summary"]
    assert "equal_weight" in report["summary"]


def test_olympus29_defaults_are_diversified_and_grid_is_bounded() -> None:
    module = load_candidate_module()

    assert [item.asset_class for item in module.DEFAULT_UNIVERSE] == [
        "crypto",
        "equity",
        "gold",
        "bond",
    ]
    assert module.DEFAULT_STRATEGY_CONFIG.target_ann_vol == 0.30
    assert len(module.parameter_grid(module.DEFAULT_STRATEGY_CONFIG)) == 12


def test_universe_assessment_blocks_silent_single_asset_degradation() -> None:
    module = load_candidate_module()
    frames = {
        "binance:BTCUSDT": synthetic_two_asset_frame()[["stable"]].rename(
            columns={"stable": "Close"}
        )
    }
    failures = [
        {
            "symbol": "SPY",
            "source": "yfinance",
            "asset_class": "equity",
            "status": "DATA_UNAVAILABLE",
            "error": "probe failure",
        },
        {
            "symbol": "GLD",
            "source": "yfinance",
            "asset_class": "gold",
            "status": "DATA_UNAVAILABLE",
            "error": "probe failure",
        },
        {
            "symbol": "TLT",
            "source": "yfinance",
            "asset_class": "bond",
            "status": "DATA_UNAVAILABLE",
            "error": "probe failure",
        },
    ]

    assessment = module.universe_assessment(
        list(module.DEFAULT_UNIVERSE),
        frames,
        failures,
        module.align_closes(frames),
    )

    assert assessment["status"] == "DATA_INSUFFICIENT"
    assert assessment["loaded_asset_classes"] == ["crypto"]


def test_run_evaluation_is_candidate_only_and_marks_insufficient_data() -> None:
    module = load_candidate_module()
    report = module.insufficient_report(
        module.datetime.now(module.UTC),
        module.DEFAULT_STRATEGY_CONFIG,
        module.DEFAULT_EVALUATION_CONFIG,
        list(module.DEFAULT_UNIVERSE),
        {},
        [],
        "not enough aligned bars",
    )

    assert report["verdict"] == module.VERDICT_DATA
    assert report["safety"]["order_path_added"] is False
    assert report["safety"]["risk_gate_changes"] is False
