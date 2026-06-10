from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def load_candidate_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    pure_path = root / "incubating" / "pure_risk_allocation.py"
    pure_spec = importlib.util.spec_from_file_location("pure_risk_allocation", pure_path)
    assert pure_spec is not None and pure_spec.loader is not None
    pure_module = importlib.util.module_from_spec(pure_spec)
    sys.modules["pure_risk_allocation"] = pure_module
    pure_spec.loader.exec_module(pure_module)

    path = root / "incubating" / "equal_weight_allocation.py"
    spec = importlib.util.spec_from_file_location("equal_weight_allocation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["equal_weight_allocation"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_cross_asset_frame(rows: int = 760) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    prices = {
        "binance:BTCUSDT": 100.0,
        "yfinance:SPY": 100.0,
        "yfinance:GLD": 100.0,
        "yfinance:TLT": 100.0,
    }
    series: dict[str, list[float]] = {name: [] for name in prices}
    for offset in range(rows):
        prices["binance:BTCUSDT"] *= 0.98 if offset % 29 == 0 else 1.0018
        prices["yfinance:SPY"] *= 0.992 if offset % 37 == 0 else 1.0008
        prices["yfinance:GLD"] *= 1.0004 if offset % 13 else 1.0012
        prices["yfinance:TLT"] *= 1.0002 if offset % 17 else 0.998
        for name, price in prices.items():
            series[name].append(price)
    return pd.DataFrame(series, index=index)


def test_equal_weight_defaults_are_predeclared_and_candidate_only() -> None:
    module = load_candidate_module()

    assert [item.rebalance_bars for item in module.DEFAULT_REBALANCE_GRID] == [21, 63, 126]
    assert {item.fee_bps for item in module.DEFAULT_REBALANCE_GRID} == {10.0}
    assert {item.slippage_bps for item in module.DEFAULT_REBALANCE_GRID} == {5.0}
    assert module.EDGE_THESIS.count("不做择时") == 1


def test_equal_weight_rebalances_periodically_with_costs() -> None:
    module = load_candidate_module()
    closes = synthetic_cross_asset_frame(90)
    config = module.EqualWeightConfig(rebalance_bars=21, fee_bps=10.0, slippage_bps=5.0)

    result = module.simulate_equal_weight_periodic(closes, config)
    daily = result["daily"]

    assert result["kind"] == "equal_weight_1n_periodic_rebalanced"
    assert result["total_cost_pct"] > 0
    assert daily["turnover"].iloc[0] == 1.0
    assert daily["turnover"].iloc[21] > 0
    assert daily["turnover"].iloc[1] == 0


def test_equal_weight_report_sweeps_every_frequency_and_reports_required_gates() -> None:
    module = load_candidate_module()
    closes = synthetic_cross_asset_frame()
    evaluation_config = module.EvaluationConfig(
        start="2021-01-01",
        end="2023-01-31",
        timeframe="1d",
        train_bars=252,
        test_bars=126,
        step_bars=126,
        min_drawdown_reduction_ratio=0.0,
        max_sharpe_shortfall=0.0,
        max_calmar_shortfall=0.0,
        max_parameter_trials=3,
    )

    report = module.full_sample_report(
        closes,
        module.DEFAULT_REBALANCE_GRID,
        evaluation_config,
    )

    assert report["single_asset_required_benchmarks"] == [
        "binance:BTCUSDT",
        "yfinance:SPY",
    ]
    assert [item["rebalance_bars"] for item in report["rebalance_sweep"]] == [
        21,
        63,
        126,
    ]
    for item in report["rebalance_sweep"]:
        comparison = item["comparison"]
        assert set(comparison["required_single_assets"]) == {
            "binance:BTCUSDT",
            "yfinance:SPY",
        }
        assert "risk_parity" in comparison
        assert comparison["net_costs_included"] is True


def test_walk_forward_summary_uses_frequency_window_pass_share_gate() -> None:
    module = load_candidate_module()
    windows = [
        {
            "frequency_results": [
                {"rebalance_bars": 21, "passes_all_gates": True, "comparison": comparison()},
                {"rebalance_bars": 63, "passes_all_gates": False, "comparison": comparison()},
                {"rebalance_bars": 126, "passes_all_gates": True, "comparison": comparison()},
            ]
        }
    ]

    summary = module.walk_forward_summary(windows, 3)

    assert summary["frequency_window_trials"] == 3
    assert summary["pass_share"] == 0.666667
    assert summary["oos_stable"] is True


def comparison() -> dict[str, Any]:
    return {
        "required_single_assets": {
            "binance:BTCUSDT": {
                "drawdown_reduction_ratio": 0.1,
                "sharpe_delta": 0.2,
                "calmar_delta": 0.3,
            },
            "yfinance:SPY": {
                "drawdown_reduction_ratio": 0.1,
                "sharpe_delta": 0.2,
                "calmar_delta": 0.3,
            },
        },
        "risk_parity": {
            "drawdown_reduction_ratio": 0.1,
            "sharpe_delta": 0.2,
            "calmar_delta": 0.3,
        },
    }
