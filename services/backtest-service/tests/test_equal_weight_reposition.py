from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def load_candidate_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    for name in ("pure_risk_allocation", "equal_weight_allocation", "equal_weight_reposition"):
        sys.modules.pop(name, None)

    pure_path = root / "incubating" / "pure_risk_allocation.py"
    pure_spec = importlib.util.spec_from_file_location("pure_risk_allocation", pure_path)
    assert pure_spec is not None and pure_spec.loader is not None
    pure_module = importlib.util.module_from_spec(pure_spec)
    sys.modules["pure_risk_allocation"] = pure_module
    pure_spec.loader.exec_module(pure_module)

    equal_path = root / "incubating" / "equal_weight_allocation.py"
    equal_spec = importlib.util.spec_from_file_location("equal_weight_allocation", equal_path)
    assert equal_spec is not None and equal_spec.loader is not None
    equal_module = importlib.util.module_from_spec(equal_spec)
    sys.modules["equal_weight_allocation"] = equal_module
    equal_spec.loader.exec_module(equal_module)

    path = root / "incubating" / "equal_weight_reposition.py"
    spec = importlib.util.spec_from_file_location("equal_weight_reposition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["equal_weight_reposition"] = module
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
        prices["binance:BTCUSDT"] *= 0.94 if offset % 31 == 0 else 1.0025
        prices["yfinance:SPY"] *= 0.995 if offset % 43 == 0 else 1.0008
        prices["yfinance:GLD"] *= 1.0005 if offset % 11 else 1.001
        prices["yfinance:TLT"] *= 1.0002 if offset % 19 else 0.998
        for name, price in prices.items():
            series[name].append(price)
    return pd.DataFrame(series, index=index)


def test_same_vol_status_quo_cash_matches_candidate_vol_scale() -> None:
    module = load_candidate_module()
    closes = synthetic_cross_asset_frame()
    config = module.EqualWeightConfig(rebalance_bars=21, fee_bps=10.0, slippage_bps=5.0)
    candidate = module.simulate_equal_weight_periodic(closes, config)

    benchmark = module.simulate_same_vol_status_quo_cash(
        closes,
        "binance:BTCUSDT",
        candidate,
    )

    assert benchmark["kind"] == "status_quo_plus_cash_same_vol"
    assert 0 < benchmark["status_quo_weight"] <= 1
    assert benchmark["total_cost_pct"] == 0.0


def test_reposition_report_keeps_old_gate_and_compact_failed_matrix() -> None:
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

    full = module.full_sample_report(
        closes,
        module.DEFAULT_REBALANCE_GRID,
        evaluation_config,
        "binance:BTCUSDT",
    )
    walk_forward = module.walk_forward_report(
        closes,
        module.DEFAULT_REBALANCE_GRID,
        evaluation_config,
        "binance:BTCUSDT",
    )
    matrix = module.compact_failed_gate_matrix(full, walk_forward)

    assert full["old_olympus30_gate_summary"]["kept_for_integrity"] is True
    assert full["summary"]["frequency_count"] == 3
    first = full["rebalance_sweep"][0]
    assert first["candidate"]["standard_metrics"]["sortino"] != 0
    assert first["candidate"]["standard_metrics"]["funding_borrow"]["applicability"] == "N/A"
    assert "annualized_turnover" in first["benchmarks"]["status_quo"]["standard_metrics"]
    assert matrix["summary"]["old_olympus30_full_pass_share"] <= 1.0
    assert any(row["scope"] == "full_sample" for row in matrix["rows"])
    assert "oos_window_win_rate_vs_status_quo_pct" in walk_forward["summary"]


def test_verdict_distinguishes_trivial_derisk_from_robust_diversification() -> None:
    module = load_candidate_module()
    full = {
        "summary": {
            "reposition_all_frequencies_pass": True,
            "same_vol_cash_integrity_all_frequencies_pass": False,
            "reposition_pass_share": 1.0,
            "same_vol_cash_integrity_pass_share": 0.0,
        }
    }
    walk = {
        "summary": {
            "reposition_oos_stable": True,
            "same_vol_cash_integrity_oos_stable": False,
            "reposition_pass_share": 1.0,
            "same_vol_cash_integrity_pass_share": 0.0,
        }
    }

    verdict, reasons = module.verdict_from_reports(full, walk)

    assert verdict == module.VERDICT_TRIVIAL
    assert "same-vol BTC+cash" in reasons[0]


def test_benchmark_revision_documents_not_gate_relaxation() -> None:
    module = load_candidate_module()

    revision = module.benchmark_revision("binance:BTCUSDT")

    assert "old gate" in revision["not_a_gate_relaxation"]
    assert "status-quo" in revision["status_quo_rationale"]
