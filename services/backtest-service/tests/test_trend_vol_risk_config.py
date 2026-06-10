from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def load_candidate_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    path = root / "incubating" / "trend_vol_risk_config.py"
    spec = importlib.util.spec_from_file_location("trend_vol_risk_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["trend_vol_risk_config"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_crash_frame(rows: int = 760) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    close: list[float] = []
    price = 100.0
    for item in range(rows):
        if item < 360:
            price *= 1.0015
        elif item < 520:
            price *= 0.992
        else:
            price *= 1.0008
        close.append(price)
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Close": close,
            "Volume": [10_000.0 for _ in close],
        },
        index=index,
    )


def test_trend_vol_candidate_reports_drawdown_and_risk_adjusted_gates() -> None:
    module = load_candidate_module()
    closes = pd.DataFrame(
        {
            "asset_a": synthetic_crash_frame()["Close"],
            "asset_b": synthetic_crash_frame()["Close"] * 1.02,
        }
    )
    strategy_config = module.StrategyConfig(
        trend_window=80,
        vol_window=20,
        target_ann_vol=0.20,
        max_asset_weight=0.50,
        max_gross_exposure=1.00,
        drawdown_stop=0.10,
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
    benchmark = module.simulate_buy_hold(closes)
    report = module.comparison_report(strategy, benchmark, evaluation_config)

    assert report["comparison"]["drawdown_reduction_ratio"] > 0
    assert "sharpe_not_worse_pass" in report["comparison"]
    assert strategy["max_gross_exposure"] <= 1.0
    assert strategy["total_cost_pct"] > 0


def test_run_evaluation_is_candidate_only_and_marks_insufficient_data() -> None:
    module = load_candidate_module()
    config = module.DEFAULT_EVALUATION_CONFIG
    report = module.insufficient_report(
        module.datetime.now(module.UTC),
        module.DEFAULT_STRATEGY_CONFIG,
        config,
        {},
        [],
        "not enough aligned bars",
    )

    assert report["verdict"] == module.VERDICT_DATA
    assert report["safety"]["order_path_added"] is False
    assert report["safety"]["risk_gate_changes"] is False


def test_walk_forward_uses_train_history_to_warm_oos_indicators() -> None:
    module = load_candidate_module()
    index = pd.date_range("2021-01-01", periods=760, freq="D")
    trend = pd.Series([100.0 + item * 0.2 for item in range(760)], index=index)
    closes = pd.DataFrame({"asset_a": trend, "asset_b": trend * 1.01})
    strategy_config = module.StrategyConfig(
        trend_window=200,
        vol_window=30,
        target_ann_vol=0.15,
        max_asset_weight=0.50,
        max_gross_exposure=1.00,
        drawdown_stop=0.20,
        drawdown_reduction=0.0,
        drawdown_cooldown_bars=20,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    evaluation_config = module.EvaluationConfig(
        start="2021-01-01",
        end="2023-01-31",
        timeframe="1d",
        train_bars=504,
        test_bars=126,
        step_bars=126,
        min_drawdown_reduction_ratio=0.20,
        max_sharpe_shortfall=0.0,
        max_calmar_shortfall=0.0,
        max_parameter_trials=12,
    )

    report = module.run_portfolio_walk_forward(closes, strategy_config, evaluation_config)
    first_oos = report["windows"][0]["oos"]["strategy"]

    assert report["status"] == "OK"
    assert first_oos["max_gross_exposure"] > 0
