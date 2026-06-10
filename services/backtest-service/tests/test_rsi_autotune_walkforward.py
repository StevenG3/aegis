from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def load_candidate_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    service_dir = root / "services" / "backtest-service"
    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))
    path = root / "incubating" / "rsi_autotune_walkforward.py"
    spec = importlib.util.spec_from_file_location("rsi_autotune_walkforward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rsi_autotune_walkforward"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_rsi_frame(rows: int = 720) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    price = 100.0
    closes: list[float] = []
    for offset in range(rows):
        cycle = offset % 40
        if cycle < 10:
            price *= 0.985
        elif cycle < 22:
            price *= 1.014
        else:
            price *= 1.0005
        closes.append(price)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value * 1.025 for value in closes],
            "Low": [value * 0.975 for value in closes],
            "Close": closes,
            "Volume": [10_000.0 for _ in closes],
        },
        index=index,
    )


def test_strategy_has_explicit_repaired_exit_logic_and_costs() -> None:
    module = load_candidate_module()
    config = module.StrategyConfig(
        rsi_period=3,
        entry_threshold=45.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_bars=5,
        position_size_r=0.01,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = module.simulate_strategy(synthetic_rsi_frame(160), config)

    assert result.trades
    assert set(result.metrics) >= {
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "calmar",
        "positive_period_win_rate",
        "annualized_turnover",
        "net_cost_pct",
    }
    assert {trade.reason for trade in result.trades} & {
        "stop_loss",
        "take_profit",
        "max_holding_bars",
    }
    assert result.costs["total_cost_pct"] > 0
    assert result.costs["funding_or_borrow"] == "N/A"


def test_autotune_reflection_changes_at_most_one_variable_per_step() -> None:
    module = load_candidate_module()
    frame = synthetic_rsi_frame(260)
    config = module.StrategyConfig(
        rsi_period=3,
        entry_threshold=45.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.04,
        max_holding_bars=8,
    )
    goal = module.GoalConfig(reflection_every=2, max_reflections=3)

    tuned = module.autotune_on_is(frame, config, goal)

    assert tuned["reflections"]
    for reflection in tuned["reflections"]:
        assert reflection["one_variable_only"] is True
        previous = reflection["previous_params"]
        candidate = reflection["candidate_params"]
        changed = [key for key, value in previous.items() if candidate[key] != value]
        assert len(changed) == 1


def test_walk_forward_freezes_oos_and_does_not_use_oos_for_param_selection() -> None:
    module = load_candidate_module()
    frame = synthetic_rsi_frame(520)
    config = module.StrategyConfig(
        rsi_period=3,
        entry_threshold=45.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.04,
        max_holding_bars=8,
        train_bars=180,
        test_bars=80,
        step_bars=80,
    )
    goal = module.GoalConfig(reflection_every=2, max_reflections=2)

    report = module.run_walk_forward(frame, config, goal)

    assert report["status"] == "OK"
    first_window = report["windows"][0]
    assert first_window["is_autotune"]["max_selector_bar_seen"] == config.train_bars - 1
    assert first_window["is_autotune"]["selector_data_end"] == first_window["is_period"]["end"]
    assert first_window["oos_period"]["start"] > first_window["is_period"]["end"]


def test_run_evaluation_reports_dual_benchmarks_and_honest_verdict() -> None:
    module = load_candidate_module()
    config = module.StrategyConfig(
        rsi_period=3,
        entry_threshold=45.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.04,
        max_holding_bars=8,
        train_bars=180,
        test_bars=80,
        step_bars=80,
    )
    goal = module.GoalConfig(reflection_every=2, max_reflections=2)

    report = module.run_evaluation(
        strategy=config,
        goal=goal,
        start="2021-01-01",
        end="2022-12-31",
        frame=synthetic_rsi_frame(520),
    )

    assert report["safety"]["order_path_added"] is False
    assert set(report["benchmarks"]) == {"buy_and_hold", "static_rsi"}
    assert report["walk_forward"]["summary"]["windows"] >= 3
    assert report["verdict"] in {
        module.VERDICT_PASS,
        module.VERDICT_FAIL,
        module.VERDICT_DATA,
    }
