from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient


def load_healthcheck():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    sys.modules.pop("healthcheck", None)
    path = service_dir / "healthcheck.py"
    spec = importlib.util.spec_from_file_location("healthcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["healthcheck"] = module
    spec.loader.exec_module(module)
    return module


def load_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in (
        "app",
        "data",
        "factor_ic",
        "funding_arb",
        "healthcheck",
        "strategies",
        "walk_forward",
    ):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("backtest_service_app_health", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app_health"] = module
    spec.loader.exec_module(module)
    return module


def run(
    return_pct: float = 12.0,
    buy_hold_return_pct: float = 5.0,
    max_drawdown_pct: float = -10.0,
    sharpe: float = 1.2,
    num_trades: int = 2,
    unknown: int = 0,
    known: int = 2,
    params: dict[str, Any] | None = None,
    period: str = "full",
) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "period": period,
        "params": params or {"fast": 20, "slow": 50},
        "stats": {
            "return_pct": return_pct,
            "buy_hold_return_pct": buy_hold_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe": sharpe,
            "num_trades": num_trades,
            "exit_breakdown": {"signal": known, "unknown": unknown},
        },
        "trades": [{"exit_reason": "signal", "entry_regime_up": True}],
        "equity_curve": [{"equity": 10_000}, {"equity": 10_100}],
    }


def by_id(result: dict[str, Any], check_id: str) -> dict[str, Any]:
    return {check["id"]: check for check in result["checks"]}[check_id]


def predictive_factor_report() -> dict[str, Any]:
    return {
        "factors": {
            "momentum_20": {
                "rank_ic": {
                    "mean": 0.04,
                    "std": 0.1,
                    "icir": 0.4,
                    "t_value": 2.0,
                    "positive_share": 0.6,
                },
                "monotonicity": {
                    "is_monotonic": True,
                    "top_bottom_return": 0.03,
                },
            }
        }
    }


def stable_walk_forward_report() -> dict[str, Any]:
    return {
        "summary": {
            "windows": 3,
            "median_oos_return_pct": 4.0,
            "positive_oos_share": 1.0,
            "median_oos_is_return_ratio": 0.7,
            "median_oos_is_sharpe_ratio": 0.8,
            "overfit": {"is_overfit": False, "reason": "OOS performance is stable"},
        }
    }


def test_passing_healthcheck_returns_pass() -> None:
    module = load_healthcheck()
    runs = [
        run(params={"fast": 16, "slow": 40}, period="full"),
        run(return_pct=8, params={"fast": 24, "slow": 60}, period="recent"),
    ]

    result = module.evaluate_strategy_health(
        runs,
        edge_thesis="Captures persistent trend continuation after broad-market confirmation.",
        thresholds={"FEE_BPS": 1, "SLIPPAGE_BPS": 1, "FUNDING_BPS": 0},
        factor_report=predictive_factor_report(),
        walk_forward_report=stable_walk_forward_report(),
    )

    assert result["verdict"] == "PASS"
    assert all(check["status"] == "PASS" for check in result["checks"])
    assert by_id(result, "factor_predictability")["status"] == "PASS"
    assert by_id(result, "walk_forward_oos_stability")["status"] == "PASS"


def test_missing_factor_report_warns_no_edge_signal() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run()],
        edge_thesis="Trend edge.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
    )

    check = by_id(result, "factor_predictability")
    assert check["status"] == "WARN"
    assert check["value"] == "INSUFFICIENT_DATA"
    assert result["verdict"] == "PASS_WITH_WARN"


def test_low_icir_factor_report_warns_no_edge_signal() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run()],
        edge_thesis="Trend edge.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
        factor_report={
            "factors": {
                "weak_factor": {
                    "rank_ic": {"mean": 0.001, "icir": 0.02, "t_value": 0.1},
                    "monotonicity": {"is_monotonic": False, "top_bottom_return": -0.001},
                }
            }
        },
    )

    check = by_id(result, "factor_predictability")
    assert check["status"] == "WARN"
    assert check["value"]["reason"] == "NO_EDGE_SIGNAL"
    assert check["value"]["factors"][0]["has_predictive_power"] is False


def test_overfit_walk_forward_report_is_hard_fail() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run()],
        edge_thesis="Trend edge.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
        factor_report=predictive_factor_report(),
        walk_forward_report={
            "summary": {
                "windows": 3,
                "median_oos_return_pct": -1.2,
                "positive_oos_share": 0.333333,
                "median_oos_is_return_ratio": -0.2,
                "median_oos_is_sharpe_ratio": -0.1,
                "overfit": {"is_overfit": True, "reason": "OOS return turned negative"},
            }
        },
    )

    check = by_id(result, "walk_forward_oos_stability")
    assert check["type"] == "hard"
    assert check["status"] == "FAIL"
    assert result["verdict"] == "BLOCK"


def test_empty_edge_is_hard_fail_and_blocks() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health([run()], edge_thesis="")

    check = by_id(result, "edge_thesis")
    assert check["type"] == "hard"
    assert check["status"] == "FAIL"
    assert result["verdict"] == "BLOCK"


def test_negative_net_cost_is_hard_fail_and_blocks() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run(return_pct=0.1, num_trades=100)],
        edge_thesis="Small gross edge tested against costs.",
        cost_bps=5,
    )

    check = by_id(result, "net_cost_positive")
    assert check["type"] == "hard"
    assert check["status"] == "FAIL"
    assert result["verdict"] == "BLOCK"


def test_net_return_pct_is_not_charged_twice() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [
            run(
                return_pct=0.6,
                buy_hold_return_pct=0,
                num_trades=20,
            )
            | {"stats": run()["stats"] | {"net_return_pct": 0.6, "benchmark_return_pct": 0.0}}
        ],
        edge_thesis="Delta-neutral funding capture after modeled costs.",
        cost_bps=100,
    )

    check = by_id(result, "net_cost_positive")
    assert check["type"] == "hard"
    assert check["status"] == "PASS"


def test_benchmark_return_pct_supports_cash_baseline() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [
            run(return_pct=1, buy_hold_return_pct=100)
            | {"stats": run()["stats"] | {"return_pct": 1, "benchmark_return_pct": 0.0}},
            run(return_pct=-1, buy_hold_return_pct=-100)
            | {"stats": run()["stats"] | {"return_pct": -1, "benchmark_return_pct": 0.0}},
        ],
        edge_thesis="Funding arb benchmark is cash, not buy and hold.",
        thresholds={"BEAT_BH_MIN": 0.5, "FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
    )

    check = by_id(result, "beat_benchmark")
    assert check["status"] == "PASS"
    assert check["value"] == 0.5


def test_excessive_drawdown_is_hard_fail_and_blocks() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run(max_drawdown_pct=-31)],
        edge_thesis="Trend strategy with capped drawdown requirement.",
    )

    check = by_id(result, "no_blowup")
    assert check["type"] == "hard"
    assert check["status"] == "FAIL"
    assert result["verdict"] == "BLOCK"


def test_soft_warnings_do_not_block_graduation_verdict() -> None:
    module = load_healthcheck()
    runs = [
        run(
            return_pct=3,
            buy_hold_return_pct=4,
            unknown=1,
            known=1,
            params={"fast": 20, "slow": 50},
            period="full",
        ),
        run(
            return_pct=2,
            buy_hold_return_pct=4,
            unknown=1,
            known=1,
            params={"fast": 24, "slow": 60},
            period="full",
        ),
        run(
            return_pct=-1,
            buy_hold_return_pct=4,
            unknown=1,
            known=1,
            params={"fast": 28, "slow": 70},
            period="recent",
        ),
    ]

    result = module.evaluate_strategy_health(
        runs,
        edge_thesis="Mean reversion idea under review.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
    )

    assert by_id(result, "beat_benchmark")["status"] == "WARN"
    assert by_id(result, "out_of_sample_stability")["status"] == "WARN"
    assert by_id(result, "attribution")["status"] == "WARN"
    assert by_id(result, "net_cost_positive")["status"] == "PASS"
    assert by_id(result, "no_blowup")["status"] == "PASS"
    assert result["verdict"] == "PASS_WITH_WARN"


def test_negative_median_return_is_soft_warn() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [
            run(return_pct=-2, num_trades=1),
            run(return_pct=1, num_trades=1),
        ],
        edge_thesis="Candidate edge under review.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
    )

    assert by_id(result, "not_one_big_winner")["status"] == "WARN"


def test_missing_parameter_perturbations_are_insufficient_data_warn() -> None:
    module = load_healthcheck()

    result = module.evaluate_strategy_health(
        [run()],
        edge_thesis="Trend edge.",
        thresholds={"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
    )

    check = by_id(result, "out_of_sample_stability")
    assert check["status"] == "WARN"
    assert check["value"] == "INSUFFICIENT_DATA"
    assert result["verdict"] == "PASS_WITH_WARN"


def test_strategy_healthcheck_endpoint() -> None:
    module = load_app()

    response = TestClient(module.app).post(
        "/strategy-healthcheck",
        json={
            "runs": [run()],
            "edge_thesis": "Trend following edge.",
            "thresholds": {"FEE_BPS": 0, "SLIPPAGE_BPS": 0, "FUNDING_BPS": 0},
        },
    )

    assert response.status_code == 200
    assert response.json()["verdict"] == "PASS_WITH_WARN"
