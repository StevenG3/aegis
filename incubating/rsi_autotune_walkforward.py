from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "backtest-service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from data import DataLoadError, load_ohlcv  # noqa: E402

VERDICT_PASS = "ROBUST_AUTOTUNE_EDGE"
VERDICT_FAIL = "NO_ROBUST_EDGE"
VERDICT_DATA = "RSI_AUTOTUNE_DATA_INSUFFICIENT"
THESIS = (
    "In-sample one-variable-at-a-time hill climbing on RSI long-only parameters should "
    "produce out-of-sample performance that is stable versus both static RSI and BTC "
    "buy-and-hold."
)
NULL_HYPOTHESIS = (
    "The one-variable hill climber fits IS noise; OOS performance does not beat static RSI "
    "or buy-and-hold after fees and slippage."
)
DEFAULT_OUTPUT_DIR = (
    Path(
        os.getenv(
            "OLYMPUS_EVIDENCE_DIR",
            str(Path(__file__).resolve().parents[2] / "aegis-strategies" / "incubating"),
        )
    )
    / "olympus38"
)


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "BTC/USDT"
    source: str = "binance"
    timeframe: str = "1d"
    instrument_type: str = "spot"
    rsi_period: int = 14
    entry_threshold: float = 30.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    max_holding_bars: int = 14
    position_size_r: float = 0.01
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    train_bars: int = 365
    test_bars: int = 90
    step_bars: int = 90


@dataclass(frozen=True)
class GoalConfig:
    target_return_30d: float = 0.03
    max_drawdown: float = 0.12
    min_sharpe: float = 0.8
    failure_below: float = -0.04
    reflection_every: int = 5
    one_variable_only: bool = True
    return_weight: float = 0.333333
    drawdown_weight: float = 0.333333
    sharpe_weight: float = 0.333334
    min_trades_for_sharpe: int = 10
    min_bars_for_sharpe: int = 30
    max_reflections: int = 6
    threshold_step: float = 2.0
    stop_loss_step: float = 0.002
    take_profit_step: float = 0.005
    rsi_period_step: int = 2


@dataclass(frozen=True)
class Trade:
    entry_bar: int
    exit_bar: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    fee_cost: float
    slippage_cost: float
    reason: str


@dataclass(frozen=True)
class SimulationResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict[str, float | int | str]
    costs: dict[str, float | str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Olympus #38 paper-only RSI autotune walk-forward falsification."
    )
    parser.add_argument(
        "--strategy",
        default=str(Path(__file__).with_name("rsi_autotune_strategy.yaml")),
    )
    parser.add_argument(
        "--goal",
        default=str(Path(__file__).with_name("rsi_autotune_goal.yaml")),
    )
    parser.add_argument("--start", default=os.getenv("OLYMPUS38_START", "2021-01-01"))
    parser.add_argument("--end", default=os.getenv("OLYMPUS38_END", "2026-06-01"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    strategy = load_strategy_config(Path(str(args.strategy)))
    goal = load_goal_config(Path(str(args.goal)))
    report = run_evaluation(strategy=strategy, goal=goal, start=str(args.start), end=str(args.end))
    if not args.no_write:
        report["written_files"] = write_report(report, Path(str(args.output_dir)))
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["verdict"] != VERDICT_DATA else 1


def run_evaluation(
    *,
    strategy: StrategyConfig,
    goal: GoalConfig,
    start: str,
    end: str,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    load_failure: str | None = None
    if frame is None:
        try:
            frame = load_ohlcv(
                strategy.symbol,
                cast(Any, strategy.source),
                strategy.timeframe,
                start,
                end,
            )
        except (DataLoadError, ModuleNotFoundError, ImportError) as exc:
            load_failure = str(exc)
            frame = pd.DataFrame()
    frame = normalize_frame(frame)
    if load_failure is not None or len(frame) < strategy.train_bars + strategy.test_bars:
        return insufficient_report(generated_at, strategy, goal, frame, load_failure)

    static = simulate_strategy(frame, strategy)
    buy_hold = simulate_buy_hold(frame, strategy)
    walk_forward = run_walk_forward(frame, strategy, goal)
    verdict, reasons = verdict_from_report(walk_forward)
    return {
        "generated_at": generated_at.isoformat(),
        "name": "rsi_autotune_walkforward",
        "status": "OK",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "thesis": THESIS,
        "null_hypothesis": NULL_HYPOTHESIS,
        "expected_verdict": VERDICT_FAIL,
        "original_script_gap_remediation": {
            "take_profit_added": True,
            "max_holding_bars_added": True,
            "note": (
                "The source demo had stop-loss only; this backtest repairs that gap "
                "explicitly."
            ),
        },
        "config": {
            "strategy": asdict(strategy),
            "goal": asdict(goal),
            "position_sizing": (
                "Each entry risks account_equity * position_size_r at the stop distance; "
                "spot notional is capped at current equity, so no leverage is introduced."
            ),
            "costs": "fee_bps and slippage_bps are charged on entry and exit notional.",
            "funding_or_borrow": "N/A for default spot long-only BTC/USDT.",
        },
        "data": data_summary(frame, start, end),
        "benchmarks": {
            "buy_and_hold": public_result(buy_hold),
            "static_rsi": public_result(static),
        },
        "walk_forward": walk_forward,
        "safety": safety_statement(),
        "disclaimer": "candidate-only paper research; no trading signal or order path",
    }


def run_walk_forward(
    frame: pd.DataFrame,
    strategy: StrategyConfig,
    goal: GoalConfig,
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    starts = window_starts(len(frame), strategy.train_bars, strategy.test_bars, strategy.step_bars)
    if not starts:
        return {"status": "INSUFFICIENT_DATA", "windows": [], "summary": {"windows": 0}}
    oos_equity_parts: list[pd.Series] = []
    for index, start in enumerate(starts):
        train = frame.iloc[start : start + strategy.train_bars]
        test = frame.iloc[
            start + strategy.train_bars : start + strategy.train_bars + strategy.test_bars
        ]
        tuned = autotune_on_is(train, strategy, goal, fold_index=index)
        tuned_strategy = cast(StrategyConfig, tuned["final_config"])
        candidate = simulate_strategy(test, tuned_strategy)
        static = simulate_strategy(test, strategy)
        buy_hold = simulate_buy_hold(test, strategy)
        if not candidate.equity.empty:
            oos_equity_parts.append(candidate.equity)
        windows.append(
            {
                "index": index,
                "is_period": period(train),
                "oos_period": period(test),
                "selected_params": public_strategy_params(tuned_strategy),
                "is_autotune": {
                    "initial_score": tuned["initial_score"],
                    "final_score": tuned["final_score"],
                    "reflection_count": len(cast(list[Any], tuned["reflections"])),
                    "reflections": tuned["reflections"],
                    "oscillation_detected": tuned["oscillation_detected"],
                    "max_selector_bar_seen": tuned["max_selector_bar_seen"],
                    "selector_data_end": period(train)["end"],
                },
                "oos": {
                    "autotune": public_result(candidate),
                    "static_rsi": public_result(static),
                    "buy_and_hold": public_result(buy_hold),
                    "beats_static_rsi": bool(
                        metric_float(candidate, "total_return_pct")
                        > metric_float(static, "total_return_pct")
                    ),
                    "beats_buy_and_hold": bool(
                        metric_float(candidate, "total_return_pct")
                        > metric_float(buy_hold, "total_return_pct")
                    ),
                },
            }
        )
    summary = walk_forward_summary(windows)
    return {
        "status": "OK",
        "mode": "rolling_walk_forward_is_autotune_oos_frozen",
        "windows": windows,
        "summary": summary,
        "combined_oos_equity": equity_summary(oos_equity_parts),
    }


def autotune_on_is(
    frame: pd.DataFrame,
    initial: StrategyConfig,
    goal: GoalConfig,
    *,
    fold_index: int = 0,
) -> dict[str, Any]:
    current = initial
    current_result = simulate_strategy(frame, current)
    current_score = score(current_result, goal)
    initial_score = current_score
    reflections: list[dict[str, Any]] = []
    trade_budget = len(current_result.trades) // max(goal.reflection_every, 1)
    steps = min(goal.max_reflections, trade_budget)
    for version in range(1, steps + 1):
        candidates = candidate_configs(current, goal)
        scored: list[tuple[float, str, StrategyConfig, SimulationResult]] = []
        for variable, config in candidates:
            result = simulate_strategy(frame, config)
            scored.append((score(result, goal), variable, config, result))
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, variable, best_config, best_result = scored[0]
        improved = best_score > current_score
        reflection = {
            "fold": fold_index,
            "version": version,
            "changed_variable": variable if improved else None,
            "one_variable_only": True,
            "previous_score": round(current_score, 6),
            "candidate_score": round(best_score, 6),
            "score_improved": improved,
            "previous_params": public_strategy_params(current),
            "candidate_params": public_strategy_params(best_config),
            "closed_trades_seen": len(current_result.trades),
            "is_bars_seen": len(frame),
        }
        reflections.append(reflection)
        if not improved:
            break
        current = best_config
        current_result = best_result
        current_score = best_score
    return {
        "initial_config": initial,
        "final_config": current,
        "initial_score": round(initial_score, 6),
        "final_score": round(current_score, 6),
        "reflections": reflections,
        "oscillation_detected": detect_oscillation(reflections),
        "max_selector_bar_seen": len(frame) - 1,
    }


def simulate_strategy(frame: pd.DataFrame, config: StrategyConfig) -> SimulationResult:
    normalized = normalize_frame(frame)
    if normalized.empty:
        return empty_simulation()
    closes = normalized["Close"]
    rsi = compute_rsi(closes, config.rsi_period)
    equity = 10_000.0
    equity_curve: list[float] = []
    trades: list[Trade] = []
    position: dict[str, float | int] | None = None
    traded_notional = 0.0
    fee_cost_total = 0.0
    slippage_cost_total = 0.0
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000

    for bar, (timestamp, row) in enumerate(normalized.iterrows()):
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        if position is not None:
            entry_price = float(position["entry_price"])
            entry_bar = int(position["entry_bar"])
            quantity = float(position["quantity"])
            notional = float(position["notional"])
            stop_price = entry_price * (1 - config.stop_loss_pct)
            take_profit_price = entry_price * (1 + config.take_profit_pct)
            exit_reason: str | None = None
            raw_exit_price: float | None = None
            if low <= stop_price:
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif high >= take_profit_price:
                exit_reason = "take_profit"
                raw_exit_price = take_profit_price
            elif bar - entry_bar >= config.max_holding_bars:
                exit_reason = "max_holding_bars"
                raw_exit_price = close
            if exit_reason is not None and raw_exit_price is not None:
                exit_price = raw_exit_price * (1 - slippage_rate)
                exit_notional = quantity * exit_price
                fee_cost = (notional + exit_notional) * fee_rate
                slippage_cost = notional * slippage_rate + quantity * raw_exit_price * slippage_rate
                pnl = exit_notional - notional - fee_cost
                equity += pnl
                traded_notional += notional + exit_notional
                fee_cost_total += fee_cost
                slippage_cost_total += slippage_cost
                trades.append(
                    Trade(
                        entry_bar=entry_bar,
                        exit_bar=bar,
                        entry_time=_date_value(normalized.index[entry_bar]),
                        exit_time=_date_value(timestamp),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        notional=notional,
                        pnl=pnl,
                        fee_cost=fee_cost,
                        slippage_cost=slippage_cost,
                        reason=exit_reason,
                    )
                )
                position = None
        if (
            position is None
            and not math.isnan(float(rsi.iloc[bar]))
            and rsi.iloc[bar] < config.entry_threshold
        ):
            risk_budget = max(equity, 0) * config.position_size_r
            notional = min(max(equity, 0), risk_budget / max(config.stop_loss_pct, 0.0001))
            if notional > 0:
                entry_price = close * (1 + slippage_rate)
                quantity = notional / entry_price
                position = {
                    "entry_bar": bar,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "notional": notional,
                }
        mark_to_market = equity
        if position is not None:
            mark_to_market += (close - float(position["entry_price"])) * float(position["quantity"])
        equity_curve.append(mark_to_market)

    if position is not None:
        bar = len(normalized) - 1
        close = float(normalized["Close"].iloc[-1])
        entry_price = float(position["entry_price"])
        quantity = float(position["quantity"])
        notional = float(position["notional"])
        exit_price = close * (1 - slippage_rate)
        exit_notional = quantity * exit_price
        fee_cost = (notional + exit_notional) * fee_rate
        slippage_cost = notional * slippage_rate + quantity * close * slippage_rate
        pnl = exit_notional - notional - fee_cost
        equity += pnl
        traded_notional += notional + exit_notional
        fee_cost_total += fee_cost
        slippage_cost_total += slippage_cost
        trades.append(
            Trade(
                entry_bar=int(position["entry_bar"]),
                exit_bar=bar,
                entry_time=_date_value(normalized.index[int(position["entry_bar"])]),
                exit_time=_date_value(normalized.index[-1]),
                entry_price=entry_price,
                exit_price=exit_price,
                notional=notional,
                pnl=pnl,
                fee_cost=fee_cost,
                slippage_cost=slippage_cost,
                reason="finalize",
            )
        )
        equity_curve[-1] = equity

    series = pd.Series(equity_curve, index=normalized.index, dtype="float64")
    costs: dict[str, float | str] = {
        "fee_cost": round(fee_cost_total, 6),
        "slippage_cost": round(slippage_cost_total, 6),
        "total_cost": round(fee_cost_total + slippage_cost_total, 6),
        "total_cost_pct": round((fee_cost_total + slippage_cost_total) / 10_000.0, 6),
        "funding_or_borrow": "N/A",
    }
    return SimulationResult(
        series, trades, metrics_from_equity(series, trades, traded_notional, costs), costs
    )


def simulate_buy_hold(frame: pd.DataFrame, config: StrategyConfig) -> SimulationResult:
    normalized = normalize_frame(frame)
    if normalized.empty:
        return empty_simulation()
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000
    start_price = float(normalized["Close"].iloc[0]) * (1 + slippage_rate)
    end_price = float(normalized["Close"].iloc[-1]) * (1 - slippage_rate)
    notional = 10_000.0
    quantity = notional / start_price
    entry_fee = notional * fee_rate
    exit_notional = quantity * end_price
    exit_fee = exit_notional * fee_rate
    equity = normalized["Close"].astype(float) * quantity - entry_fee
    equity.iloc[-1] = exit_notional - entry_fee - exit_fee
    trade = Trade(
        entry_bar=0,
        exit_bar=len(normalized) - 1,
        entry_time=_date_value(normalized.index[0]),
        exit_time=_date_value(normalized.index[-1]),
        entry_price=start_price,
        exit_price=end_price,
        notional=notional,
        pnl=exit_notional - notional - entry_fee - exit_fee,
        fee_cost=entry_fee + exit_fee,
        slippage_cost=notional * slippage_rate
        + quantity * float(normalized["Close"].iloc[-1]) * slippage_rate,
        reason="buy_and_hold",
    )
    costs: dict[str, float | str] = {
        "fee_cost": round(entry_fee + exit_fee, 6),
        "slippage_cost": round(trade.slippage_cost, 6),
        "total_cost": round(entry_fee + exit_fee + trade.slippage_cost, 6),
        "total_cost_pct": round((entry_fee + exit_fee + trade.slippage_cost) / 10_000.0, 6),
        "funding_or_borrow": "N/A",
    }
    return SimulationResult(
        equity,
        [trade],
        metrics_from_equity(equity, [trade], notional + exit_notional, costs),
        costs,
    )


def score(result: SimulationResult, goal: GoalConfig) -> float:
    metrics = result.metrics
    bars = max(int(metrics["bars"]), 1)
    total_return = float(metrics["total_return_pct"])
    return_30d = total_return * (30 / bars)
    drawdown = abs(float(metrics["max_drawdown_pct"]))
    sharpe = float(metrics["sharpe"])
    return_component = clamp(return_30d / max(goal.target_return_30d, 0.0001), -1.0, 1.0)
    drawdown_component = clamp(
        (goal.max_drawdown - drawdown) / max(goal.max_drawdown, 0.0001), -1.0, 1.0
    )
    sharpe_component = clamp(sharpe / max(goal.min_sharpe, 0.0001), -1.0, 1.0)
    if int(metrics["trades"]) < goal.min_trades_for_sharpe or bars < goal.min_bars_for_sharpe:
        sharpe_component *= 0.5
    weighted = (
        goal.return_weight * return_component
        + goal.drawdown_weight * drawdown_component
        + goal.sharpe_weight * sharpe_component
    )
    return clamp(weighted, -1.0, 1.0)


def candidate_configs(config: StrategyConfig, goal: GoalConfig) -> list[tuple[str, StrategyConfig]]:
    return [
        (
            "entry_threshold",
            replace(
                config, entry_threshold=min(50.0, config.entry_threshold + goal.threshold_step)
            ),
        ),
        (
            "stop_loss_pct",
            replace(config, stop_loss_pct=max(0.005, config.stop_loss_pct - goal.stop_loss_step)),
        ),
        (
            "take_profit_pct",
            replace(
                config, take_profit_pct=max(0.01, config.take_profit_pct + goal.take_profit_step)
            ),
        ),
        (
            "rsi_period",
            replace(config, rsi_period=max(2, config.rsi_period + goal.rsi_period_step)),
        ),
    ]


def detect_oscillation(reflections: list[dict[str, Any]]) -> bool:
    changed = [
        str(item.get("changed_variable")) for item in reflections if item.get("changed_variable")
    ]
    if len(changed) < 3:
        return False
    for index in range(len(changed) - 2):
        if changed[index] == changed[index + 2] and changed[index] != changed[index + 1]:
            return True
    return False


def metrics_from_equity(
    equity: pd.Series,
    trades: list[Trade],
    traded_notional: float,
    costs: dict[str, float | str],
) -> dict[str, float | int | str]:
    if equity.empty:
        return default_metrics()
    returns = equity.pct_change().replace([math.inf, -math.inf], math.nan).dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max(len(equity) / 252, 1 / 252)
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    max_dd = max_drawdown(equity)
    sharpe = annualized_sharpe(returns)
    sortino = annualized_sortino(returns)
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0
    positive_period_win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    return {
        "bars": int(len(equity)),
        "trades": int(len(trades)),
        "total_return_pct": round(total_return, 6),
        "annual_return_pct": round(annual_return, 6),
        "max_drawdown_pct": round(max_dd, 6),
        "sharpe": round(sharpe, 6),
        "sortino": round(sortino, 6),
        "calmar": round(calmar, 6),
        "positive_period_win_rate": round(positive_period_win_rate, 6),
        "trade_win_rate": round(sum(1 for trade in trades if trade.pnl > 0) / len(trades), 6)
        if trades
        else 0.0,
        "annualized_turnover": round((traded_notional / 10_000.0) / years, 6),
        "net_cost_pct": costs["total_cost_pct"],
    }


def walk_forward_summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if not windows:
        return {"windows": 0, "verdict_ready": False}
    beats_static = [bool(window["oos"]["beats_static_rsi"]) for window in windows]
    beats_hold = [bool(window["oos"]["beats_buy_and_hold"]) for window in windows]
    candidate_returns = [
        float(window["oos"]["autotune"]["metrics"]["total_return_pct"]) for window in windows
    ]
    static_returns = [
        float(window["oos"]["static_rsi"]["metrics"]["total_return_pct"]) for window in windows
    ]
    hold_returns = [
        float(window["oos"]["buy_and_hold"]["metrics"]["total_return_pct"]) for window in windows
    ]
    oscillations = [bool(window["is_autotune"]["oscillation_detected"]) for window in windows]
    return {
        "windows": len(windows),
        "autotune_oos_median_return": round(statistics.median(candidate_returns), 6),
        "static_rsi_oos_median_return": round(statistics.median(static_returns), 6),
        "buy_hold_oos_median_return": round(statistics.median(hold_returns), 6),
        "oos_window_win_rate_vs_static_rsi": round(sum(beats_static) / len(beats_static), 6),
        "oos_window_win_rate_vs_buy_hold": round(sum(beats_hold) / len(beats_hold), 6),
        "all_windows_beat_static_rsi": all(beats_static),
        "all_windows_beat_buy_hold": all(beats_hold),
        "oscillation_window_count": sum(oscillations),
    }


def verdict_from_report(walk_forward: dict[str, Any]) -> tuple[str, list[str]]:
    summary = cast(dict[str, Any], walk_forward.get("summary", {}))
    reasons: list[str] = []
    if summary.get("windows", 0) < 3:
        return VERDICT_DATA, ["fewer than three OOS windows; insufficient walk-forward sample"]
    if not summary.get("all_windows_beat_static_rsi"):
        reasons.append("autotune did not beat static RSI in every OOS window")
    if not summary.get("all_windows_beat_buy_hold"):
        reasons.append("autotune did not beat buy-and-hold in every OOS window")
    if summary.get("oscillation_window_count", 0):
        reasons.append("one-variable reflection showed oscillation in at least one IS window")
    if reasons:
        return VERDICT_FAIL, reasons
    return VERDICT_PASS, ["autotune beat both benchmarks in all OOS windows after costs"]


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, math.nan)
    return cast(pd.Series, 100 - (100 / (1 + rs)))


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized = normalized[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return cast(pd.DataFrame, normalized.dropna().sort_index())


def load_strategy_config(path: Path) -> StrategyConfig:
    raw = parse_simple_yaml(path)
    return StrategyConfig(
        symbol=str(raw.get("symbol", "BTC/USDT")),
        source=str(raw.get("source", "binance")),
        timeframe=str(raw.get("timeframe", "1d")),
        instrument_type=str(raw.get("instrument_type", "spot")),
        rsi_period=as_int(raw.get("rsi_period", 14)),
        entry_threshold=as_float(nested(raw, "entry", "threshold", 30)),
        stop_loss_pct=as_float(nested(raw, "exit", "stop_loss_pct", 0.03)),
        take_profit_pct=as_float(nested(raw, "exit", "take_profit_pct", 0.06)),
        max_holding_bars=as_int(nested(raw, "exit", "max_holding_bars", 14)),
        position_size_r=as_float(nested(raw, "position", "position_size_r", 0.01)),
        fee_bps=as_float(nested(raw, "costs", "fee_bps", 10)),
        slippage_bps=as_float(nested(raw, "costs", "slippage_bps", 5)),
        train_bars=as_int(nested(raw, "walk_forward", "train_bars", 365)),
        test_bars=as_int(nested(raw, "walk_forward", "test_bars", 90)),
        step_bars=as_int(nested(raw, "walk_forward", "step_bars", 90)),
    )


def load_goal_config(path: Path) -> GoalConfig:
    raw = parse_simple_yaml(path)
    return GoalConfig(
        target_return_30d=as_float(raw.get("target_return_30d", 0.03)),
        max_drawdown=as_float(raw.get("max_drawdown", 0.12)),
        min_sharpe=as_float(raw.get("min_sharpe", 0.8)),
        failure_below=as_float(raw.get("failure_below", -0.04)),
        reflection_every=as_int(raw.get("reflection_every", 5)),
        one_variable_only=bool(raw.get("one_variable_only", True)),
        return_weight=as_float(nested(raw, "score", "return_weight", 0.333333)),
        drawdown_weight=as_float(nested(raw, "score", "drawdown_weight", 0.333333)),
        sharpe_weight=as_float(nested(raw, "score", "sharpe_weight", 0.333334)),
        min_trades_for_sharpe=as_int(nested(raw, "score", "min_trades_for_sharpe", 10)),
        min_bars_for_sharpe=as_int(nested(raw, "score", "min_bars_for_sharpe", 30)),
        max_reflections=as_int(nested(raw, "autotune", "max_reflections", 6)),
        threshold_step=as_float(nested(raw, "autotune", "threshold_step", 2)),
        stop_loss_step=as_float(nested(raw, "autotune", "stop_loss_step", 0.002)),
        take_profit_step=as_float(nested(raw, "autotune", "take_profit_step", 0.005)),
        rsi_period_step=as_int(nested(raw, "autotune", "rsi_period_step", 2)),
    )


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        if not value:
            current_section = key
            result[current_section] = {}
            continue
        parsed = parse_scalar(value.strip())
        if indent and current_section is not None:
            cast(dict[str, Any], result[current_section])[key] = parsed
        else:
            current_section = None
            result[key] = parsed
    return result


def parse_scalar(value: str) -> str | int | float | bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def nested(raw: dict[str, Any], section: str, key: str, default: object) -> object:
    section_value = raw.get(section)
    if isinstance(section_value, dict):
        return section_value.get(key, default)
    return default


def as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"expected numeric value, got {type(value).__name__}")


def as_int(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"expected integer value, got {type(value).__name__}")


def window_starts(total: int, train_bars: int, test_bars: int, step_bars: int) -> list[int]:
    starts: list[int] = []
    start = 0
    while start + train_bars + test_bars <= total:
        starts.append(start)
        start += step_bars
    return starts


def public_strategy_params(config: StrategyConfig) -> dict[str, int | float]:
    return {
        "rsi_period": config.rsi_period,
        "entry_threshold": round(config.entry_threshold, 6),
        "stop_loss_pct": round(config.stop_loss_pct, 6),
        "take_profit_pct": round(config.take_profit_pct, 6),
        "max_holding_bars": config.max_holding_bars,
        "position_size_r": round(config.position_size_r, 6),
    }


def public_result(result: SimulationResult) -> dict[str, Any]:
    return {
        "metrics": result.metrics,
        "costs": result.costs,
        "trades": len(result.trades),
        "exit_reason_counts": exit_reason_counts(result.trades),
    }


def metric_float(result: SimulationResult, key: str) -> float:
    value = result.metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def exit_reason_counts(trades: list[Trade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        counts[trade.reason] = counts.get(trade.reason, 0) + 1
    return counts


def period(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    return {"start": _date_value(frame.index[0]), "end": _date_value(frame.index[-1])}


def data_summary(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    return {
        "requested_start": start,
        "requested_end": end,
        "bars": len(frame),
        "period": period(frame),
        "bulk_cache_committed": False,
    }


def equity_summary(parts: list[pd.Series]) -> dict[str, Any]:
    if not parts:
        return default_metrics()
    combined = pd.concat(parts)
    return metrics_from_equity(combined, [], 0.0, {"total_cost_pct": 0.0})


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"olympus38-rsi-autotune-walkforward-{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    history_dir = output_dir / "state" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    hypotheses_path = output_dir / "hypotheses.jsonl"
    json_path.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(markdown_report(report, json_path), encoding="utf-8")
    write_history(report, history_dir, hypotheses_path)
    return {"json": str(json_path), "markdown": str(md_path), "hypotheses": str(hypotheses_path)}


def write_history(report: dict[str, Any], history_dir: Path, hypotheses_path: Path) -> None:
    walk = cast(dict[str, Any], report.get("walk_forward", {}))
    windows = walk.get("windows", [])
    if not isinstance(windows, list):
        return
    lines: list[str] = []
    version = 1
    for window in windows:
        if not isinstance(window, dict):
            continue
        autotune = cast(dict[str, Any], window.get("is_autotune", {}))
        reflections = autotune.get("reflections", [])
        if not isinstance(reflections, list):
            continue
        for reflection in reflections:
            if not isinstance(reflection, dict) or not reflection.get("score_improved"):
                continue
            history_path = history_dir / f"v{version:04d}.yaml"
            history_path.write_text(
                "\n".join(
                    [
                        f"version: {version}",
                        f"fold: {reflection.get('fold')}",
                        f"changed_variable: {reflection.get('changed_variable')}",
                        f"previous_score: {reflection.get('previous_score')}",
                        f"candidate_score: {reflection.get('candidate_score')}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            lines.append(json.dumps(reflection, sort_keys=True))
            version += 1
    hypotheses_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def markdown_report(report: dict[str, Any], json_path: Path) -> str:
    summary = cast(
        dict[str, Any], cast(dict[str, Any], report.get("walk_forward", {})).get("summary", {})
    )
    return "\n".join(
        [
            "# Olympus #38 RSI Autotune Walk-Forward",
            "",
            f"- Verdict: {report.get('verdict')}",
            f"- JSON: {json_path}",
            f"- Thesis: {report.get('thesis')}",
            f"- Null hypothesis: {report.get('null_hypothesis')}",
            f"- OOS windows: {summary.get('windows')}",
            f"- Win rate vs static RSI: {summary.get('oos_window_win_rate_vs_static_rsi')}",
            f"- Win rate vs buy-and-hold: {summary.get('oos_window_win_rate_vs_buy_hold')}",
            f"- Oscillation windows: {summary.get('oscillation_window_count')}",
            "",
            "Paper research only; no trading signal or order path.",
        ]
    )


def insufficient_report(
    generated_at: datetime,
    strategy: StrategyConfig,
    goal: GoalConfig,
    frame: pd.DataFrame,
    load_failure: str | None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "name": "rsi_autotune_walkforward",
        "status": "DATA_INSUFFICIENT",
        "verdict": VERDICT_DATA,
        "verdict_reasons": [load_failure or "not enough bars for one IS/OOS walk-forward split"],
        "thesis": THESIS,
        "null_hypothesis": NULL_HYPOTHESIS,
        "config": {"strategy": asdict(strategy), "goal": asdict(goal)},
        "data": data_summary(frame, "", ""),
        "safety": safety_statement(),
        "disclaimer": "data limitation is not evidence of no edge",
    }


def safety_statement() -> dict[str, bool | str]:
    return {
        "paper_only": True,
        "order_path_added": False,
        "strategy_plugin_registered": False,
        "live_trading": False,
        "private_evidence_required": True,
    }


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std())
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean()) / std * math.sqrt(252)


def annualized_sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    std = float(downside.std())
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean()) / std * math.sqrt(252)


def empty_simulation() -> SimulationResult:
    return SimulationResult(
        pd.Series(dtype="float64"),
        [],
        default_metrics(),
        {"total_cost_pct": 0.0, "funding_or_borrow": "N/A"},
    )


def default_metrics() -> dict[str, float | int | str]:
    return {
        "bars": 0,
        "trades": 0,
        "total_return_pct": 0.0,
        "annual_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "positive_period_win_rate": 0.0,
        "trade_win_rate": 0.0,
        "annualized_turnover": 0.0,
        "net_cost_pct": 0.0,
    }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _date_value(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return {str(index): float(item) for index, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict()
    if isinstance(value, Trade):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
