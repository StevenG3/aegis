from __future__ import annotations

import math
import statistics
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from backtesting import Backtest, Strategy  # type: ignore[import-untyped]

StrategyParams = dict[str, int | float | bool | str]
Objective = Literal["return_pct", "sharpe"]


def run_walk_forward(
    frame: pd.DataFrame,
    strategy_cls: type[Strategy],
    param_grid: list[StrategyParams],
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    cash: float = 10_000,
    commission: float = 0.001,
    objective: Objective = "return_pct",
    min_oos_return_pct: float = 0.0,
    min_oos_is_return_ratio: float = 0.5,
    min_oos_is_sharpe_ratio: float = 0.5,
    max_parameter_trials: int = 20,
) -> dict[str, Any]:
    if not param_grid:
        raise ValueError("param_grid must not be empty")
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    resolved_step = step_bars or test_bars
    if resolved_step <= 0:
        raise ValueError("step_bars must be positive")
    if len(frame) < train_bars + test_bars:
        return _insufficient_report(
            "not enough bars for one train->test walk-forward split",
            frame,
            train_bars,
            test_bars,
            resolved_step,
            len(param_grid),
        )

    windows: list[dict[str, Any]] = []
    for index, start in enumerate(_window_starts(len(frame), train_bars, test_bars, resolved_step)):
        train = frame.iloc[start : start + train_bars]
        test = frame.iloc[start + train_bars : start + train_bars + test_bars]
        selected = _select_params(
            train,
            strategy_cls,
            param_grid,
            cash=cash,
            commission=commission,
            objective=objective,
        )
        oos_stats = _run_backtest(
            test,
            strategy_cls,
            cast(StrategyParams, selected["params"]),
            cash=cash,
            commission=commission,
        )
        window = _window_report(index, train, test, selected, oos_stats)
        windows.append(window)

    summary = _summary(
        windows,
        min_oos_return_pct=min_oos_return_pct,
        min_oos_is_return_ratio=min_oos_is_return_ratio,
        min_oos_is_sharpe_ratio=min_oos_is_sharpe_ratio,
        max_parameter_trials=max_parameter_trials,
        parameter_trials=len(param_grid),
    )
    report = {
        "status": "OK",
        "mode": "walk_forward",
        "data": {
            "bars": len(frame),
            "start": _date_value(frame.index[0]),
            "end": _date_value(frame.index[-1]),
        },
        "windows": windows,
        "summary": summary,
        "thresholds": {
            "min_oos_return_pct": min_oos_return_pct,
            "min_oos_is_return_ratio": min_oos_is_return_ratio,
            "min_oos_is_sharpe_ratio": min_oos_is_sharpe_ratio,
            "max_parameter_trials": max_parameter_trials,
        },
        "disclaimer": "candidates-only walk-forward evaluation; no trading signal or order path",
    }
    report["readable_report"] = render_walk_forward_report(report)
    return report


def render_walk_forward_report(report: dict[str, Any]) -> str:
    summary = cast(dict[str, Any], report.get("summary", {}))
    lines = [
        "Walk-forward evaluation",
        "candidates-only; no trading signal or order path",
        (
            f"windows={summary.get('windows')} "
            f"median_oos_return_pct={_fmt(summary.get('median_oos_return_pct'))} "
            f"positive_oos_share={_fmt(summary.get('positive_oos_share'))} "
            f"median_return_decay={_fmt(summary.get('median_oos_is_return_ratio'))} "
            f"overfit={cast(dict[str, Any], summary.get('overfit', {})).get('is_overfit')}"
        ),
    ]
    windows = report.get("windows")
    if isinstance(windows, list):
        for raw in windows:
            if not isinstance(raw, dict):
                continue
            lines.append(
                f"- window {raw.get('index')}: train={raw.get('train_period')} "
                f"test={raw.get('test_period')} params={raw.get('selected_params')} "
                f"is_return={_fmt(raw.get('is_return_pct'))} "
                f"oos_return={_fmt(raw.get('oos_return_pct'))} "
                f"return_decay={_fmt(raw.get('oos_is_return_ratio'))}"
            )
    return "\n".join(lines)


def _insufficient_report(
    reason: str,
    frame: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    parameter_trials: int,
) -> dict[str, Any]:
    report = {
        "status": "INSUFFICIENT_DATA",
        "reason": reason,
        "mode": "walk_forward",
        "data": {
            "bars": len(frame),
            "start": _date_value(frame.index[0]) if len(frame) else None,
            "end": _date_value(frame.index[-1]) if len(frame) else None,
        },
        "windows": [],
        "summary": {
            "windows": 0,
            "parameter_trials": parameter_trials,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "step_bars": step_bars,
            "overfit": {"is_overfit": None, "reason": reason},
        },
        "disclaimer": "candidates-only walk-forward evaluation; no trading signal or order path",
    }
    report["readable_report"] = render_walk_forward_report(report)
    return report


def _window_starts(
    total_bars: int, train_bars: int, test_bars: int, step_bars: int
) -> list[int]:
    result: list[int] = []
    start = 0
    while start + train_bars + test_bars <= total_bars:
        result.append(start)
        start += step_bars
    return result


def _select_params(
    train: pd.DataFrame,
    strategy_cls: type[Strategy],
    param_grid: list[StrategyParams],
    *,
    cash: float,
    commission: float,
    objective: Objective,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for params in param_grid:
        stats = _run_backtest(train, strategy_cls, params, cash=cash, commission=commission)
        candidates.append({"params": dict(params), "stats": stats})
    candidates.sort(key=lambda item: _objective_value(item["stats"], objective), reverse=True)
    selected = candidates[0]
    return {
        "params": selected["params"],
        "stats": selected["stats"],
        "objective": objective,
        "trials": candidates,
    }


def _run_backtest(
    frame: pd.DataFrame,
    strategy_cls: type[Strategy],
    params: StrategyParams,
    *,
    cash: float,
    commission: float,
) -> dict[str, float | int]:
    configured = cast(type[Strategy], type("WalkForwardStrategy", (strategy_cls,), dict(params)))
    backtest = Backtest(
        frame,
        configured,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = backtest.run()
    return {
        "return_pct": _stat_float(stats, "Return [%]"),
        "win_rate_pct": _stat_float(stats, "Win Rate [%]"),
        "max_drawdown_pct": _stat_float(stats, "Max. Drawdown [%]"),
        "sharpe": _stat_float(stats, "Sharpe Ratio"),
        "num_trades": _stat_int(stats, "# Trades"),
    }


def _window_report(
    index: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, Any],
    oos_stats: dict[str, float | int],
) -> dict[str, Any]:
    is_stats = cast(dict[str, float | int], selected["stats"])
    is_return = _float_value(is_stats["return_pct"])
    oos_return = _float_value(oos_stats["return_pct"])
    is_sharpe = _float_value(is_stats["sharpe"])
    oos_sharpe = _float_value(oos_stats["sharpe"])
    return {
        "index": index,
        "train_period": {"start": _date_value(train.index[0]), "end": _date_value(train.index[-1])},
        "test_period": {"start": _date_value(test.index[0]), "end": _date_value(test.index[-1])},
        "selected_params": selected["params"],
        "is_stats": is_stats,
        "oos_stats": oos_stats,
        "is_return_pct": is_return,
        "oos_return_pct": oos_return,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "oos_is_return_ratio": _ratio(oos_return, is_return),
        "oos_is_sharpe_ratio": _ratio(oos_sharpe, is_sharpe),
    }


def _summary(
    windows: list[dict[str, Any]],
    *,
    min_oos_return_pct: float,
    min_oos_is_return_ratio: float,
    min_oos_is_sharpe_ratio: float,
    max_parameter_trials: int,
    parameter_trials: int,
) -> dict[str, Any]:
    oos_returns = [_float_value(window["oos_return_pct"]) for window in windows]
    oos_sharpes = [_float_value(window["oos_sharpe"]) for window in windows]
    return_ratios = [
        value
        for window in windows
        if (value := _optional_float(window.get("oos_is_return_ratio"))) is not None
    ]
    sharpe_ratios = [
        value
        for window in windows
        if (value := _optional_float(window.get("oos_is_sharpe_ratio"))) is not None
    ]
    median_return = statistics.median(oos_returns)
    positive_count = sum(1 for value in oos_returns if value > min_oos_return_pct)
    positive_share = positive_count / len(oos_returns)
    median_return_ratio = statistics.median(return_ratios) if return_ratios else None
    median_sharpe_ratio = statistics.median(sharpe_ratios) if sharpe_ratios else None
    overfit_reasons: list[str] = []
    if median_return <= min_oos_return_pct:
        overfit_reasons.append("median OOS return is not positive after modeled costs")
    if positive_share < 0.5:
        overfit_reasons.append("fewer than half of OOS windows are positive")
    if median_return_ratio is None:
        overfit_reasons.append("IS return is not positive enough to compute return decay")
    elif median_return_ratio < min_oos_is_return_ratio:
        overfit_reasons.append("OOS/IS return ratio below threshold")
    if median_sharpe_ratio is not None and median_sharpe_ratio < min_oos_is_sharpe_ratio:
        overfit_reasons.append("OOS/IS Sharpe ratio below threshold")
    multiple_testing_warning = parameter_trials > max_parameter_trials
    if multiple_testing_warning:
        overfit_reasons.append("many parameter trials; best IS result may be data-mined")
    return {
        "windows": len(windows),
        "parameter_trials": parameter_trials,
        "median_oos_return_pct": round(median_return, 6),
        "total_oos_return_pct": round(sum(oos_returns), 6),
        "median_oos_sharpe": round(statistics.median(oos_sharpes), 6),
        "positive_oos_share": round(positive_share, 6),
        "median_oos_is_return_ratio": _rounded_optional(median_return_ratio),
        "median_oos_is_sharpe_ratio": _rounded_optional(median_sharpe_ratio),
        "multiple_testing_warning": multiple_testing_warning,
        "overfit": {
            "is_overfit": bool(overfit_reasons),
            "reason": (
                "; ".join(overfit_reasons)
                if overfit_reasons
                else "OOS performance is stable"
            ),
        },
    }


def _objective_value(stats: object, objective: Objective) -> float:
    if not isinstance(stats, dict):
        return -math.inf
    value = stats.get(objective)
    if isinstance(value, int | float):
        return float(value)
    return -math.inf


def _stat_float(stats: Any, key: str) -> float:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _stat_int(stats: Any, key: str) -> int:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _rounded_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _date_value(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return "NA"
