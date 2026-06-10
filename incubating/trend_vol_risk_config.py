from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "backtest-service"
sys.path.insert(0, str(SERVICE_DIR))

from data import Source, load_ohlcv  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "OLYMPUS_EVIDENCE_DIR",
        str(Path(__file__).resolve().parents[2] / "aegis-strategies" / "incubating"),
    )
)
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-06-01"
DEFAULT_TIMEFRAME = "1d"
TRADING_DAYS = 252

EDGE_THESIS = (
    "不赌方向预测;靠趋势过滤避开下行、波动率目标控暴露、回撤止损保命,"
    "追求风险调整后稳健"
)
VERDICT_PASS = "RISK_REPOSITIONING_CANDIDATE_INCUBATING"
VERDICT_FAIL = "NO_ROBUST_RISK_REPOSITIONING_EDGE"
VERDICT_DATA = "RISK_REPOSITIONING_DATA_INSUFFICIENT"


@dataclass(frozen=True)
class UniverseItem:
    symbol: str
    source: Source
    asset_class: str


@dataclass(frozen=True)
class StrategyConfig:
    trend_window: int
    vol_window: int
    target_ann_vol: float
    max_asset_weight: float
    max_gross_exposure: float
    drawdown_stop: float
    drawdown_reduction: float
    drawdown_cooldown_bars: int
    fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class EvaluationConfig:
    start: str
    end: str
    timeframe: str
    train_bars: int
    test_bars: int
    step_bars: int
    min_drawdown_reduction_ratio: float
    max_sharpe_shortfall: float
    max_calmar_shortfall: float
    max_parameter_trials: int


DEFAULT_UNIVERSE: tuple[UniverseItem, ...] = (
    UniverseItem("BTCUSDT", "binance", "crypto"),
    UniverseItem("ETHUSDT", "binance", "crypto"),
)


DEFAULT_STRATEGY_CONFIG = StrategyConfig(
    trend_window=200,
    vol_window=30,
    target_ann_vol=0.15,
    max_asset_weight=0.50,
    max_gross_exposure=1.00,
    drawdown_stop=0.20,
    drawdown_reduction=0.00,
    drawdown_cooldown_bars=20,
    fee_bps=10.0,
    slippage_bps=5.0,
)


DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    start=DEFAULT_START,
    end=DEFAULT_END,
    timeframe=DEFAULT_TIMEFRAME,
    train_bars=504,
    test_bars=126,
    step_bars=126,
    min_drawdown_reduction_ratio=0.20,
    max_sharpe_shortfall=0.0,
    max_calmar_shortfall=0.0,
    max_parameter_trials=12,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Olympus #27 trend/vol/drawdown beta configuration."
    )
    parser.add_argument("--start", default=os.getenv("OLYMPUS27_START", DEFAULT_START))
    parser.add_argument("--end", default=os.getenv("OLYMPUS27_END", DEFAULT_END))
    parser.add_argument("--timeframe", default=os.getenv("OLYMPUS27_TIMEFRAME", DEFAULT_TIMEFRAME))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OLYMPUS27_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    evaluation_config = env_evaluation_config(
        start=str(args.start),
        end=str(args.end),
        timeframe=str(args.timeframe),
    )
    report = run_evaluation(
        strategy_config=env_strategy_config(),
        evaluation_config=evaluation_config,
        universe=parse_universe(os.getenv("OLYMPUS27_UNIVERSE", "")),
        verbose=True,
    )
    if not args.no_write:
        report["written_files"] = write_report(report, Path(str(args.output_dir)))
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["verdict"] != VERDICT_DATA else 1


def env_strategy_config() -> StrategyConfig:
    return StrategyConfig(
        trend_window=_env_int("OLYMPUS27_TREND_WINDOW", DEFAULT_STRATEGY_CONFIG.trend_window),
        vol_window=_env_int("OLYMPUS27_VOL_WINDOW", DEFAULT_STRATEGY_CONFIG.vol_window),
        target_ann_vol=_env_float(
            "OLYMPUS27_TARGET_ANN_VOL", DEFAULT_STRATEGY_CONFIG.target_ann_vol
        ),
        max_asset_weight=_env_float(
            "OLYMPUS27_MAX_ASSET_WEIGHT", DEFAULT_STRATEGY_CONFIG.max_asset_weight
        ),
        max_gross_exposure=_env_float(
            "OLYMPUS27_MAX_GROSS_EXPOSURE", DEFAULT_STRATEGY_CONFIG.max_gross_exposure
        ),
        drawdown_stop=_env_float("OLYMPUS27_DRAWDOWN_STOP", DEFAULT_STRATEGY_CONFIG.drawdown_stop),
        drawdown_reduction=_env_float(
            "OLYMPUS27_DRAWDOWN_REDUCTION", DEFAULT_STRATEGY_CONFIG.drawdown_reduction
        ),
        drawdown_cooldown_bars=_env_int(
            "OLYMPUS27_DRAWDOWN_COOLDOWN_BARS",
            DEFAULT_STRATEGY_CONFIG.drawdown_cooldown_bars,
        ),
        fee_bps=_env_float("OLYMPUS27_FEE_BPS", DEFAULT_STRATEGY_CONFIG.fee_bps),
        slippage_bps=_env_float("OLYMPUS27_SLIPPAGE_BPS", DEFAULT_STRATEGY_CONFIG.slippage_bps),
    )


def env_evaluation_config(*, start: str, end: str, timeframe: str) -> EvaluationConfig:
    return EvaluationConfig(
        start=start,
        end=end,
        timeframe=timeframe,
        train_bars=_env_int("OLYMPUS27_TRAIN_BARS", DEFAULT_EVALUATION_CONFIG.train_bars),
        test_bars=_env_int("OLYMPUS27_TEST_BARS", DEFAULT_EVALUATION_CONFIG.test_bars),
        step_bars=_env_int("OLYMPUS27_STEP_BARS", DEFAULT_EVALUATION_CONFIG.step_bars),
        min_drawdown_reduction_ratio=_env_float(
            "OLYMPUS27_MIN_DRAWDOWN_REDUCTION_RATIO",
            DEFAULT_EVALUATION_CONFIG.min_drawdown_reduction_ratio,
        ),
        max_sharpe_shortfall=_env_float(
            "OLYMPUS27_MAX_SHARPE_SHORTFALL",
            DEFAULT_EVALUATION_CONFIG.max_sharpe_shortfall,
        ),
        max_calmar_shortfall=_env_float(
            "OLYMPUS27_MAX_CALMAR_SHORTFALL",
            DEFAULT_EVALUATION_CONFIG.max_calmar_shortfall,
        ),
        max_parameter_trials=_env_int(
            "OLYMPUS27_MAX_PARAMETER_TRIALS",
            DEFAULT_EVALUATION_CONFIG.max_parameter_trials,
        ),
    )


def parse_universe(raw: str) -> tuple[UniverseItem, ...]:
    if not raw.strip():
        return DEFAULT_UNIVERSE
    items: list[UniverseItem] = []
    for chunk in raw.split(","):
        fields = [field.strip() for field in chunk.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError(
                "OLYMPUS27_UNIVERSE entries must be source:symbol[:asset_class]"
            )
        source = cast(Source, fields[0])
        if source not in ("binance", "okx", "bybit", "yfinance"):
            raise ValueError(f"unsupported source in OLYMPUS27_UNIVERSE: {source}")
        items.append(
            UniverseItem(
                symbol=fields[1],
                source=source,
                asset_class=fields[2] if len(fields) == 3 else "unknown",
            )
        )
    if not items:
        raise ValueError("OLYMPUS27_UNIVERSE must contain at least one asset")
    return tuple(items)


def run_evaluation(
    *,
    strategy_config: StrategyConfig,
    evaluation_config: EvaluationConfig,
    universe: Iterable[UniverseItem],
    verbose: bool = False,
) -> dict[str, Any]:
    validate_configs(strategy_config, evaluation_config)
    universe_items = parse_universe_items(universe)
    generated_at = datetime.now(UTC)
    frames, load_failures = load_frames(universe_items, evaluation_config, verbose=verbose)
    aligned = align_closes(frames)
    if aligned.empty or len(aligned) < max(
        strategy_config.trend_window + strategy_config.vol_window + 20,
        evaluation_config.train_bars + evaluation_config.test_bars,
    ):
        report = insufficient_report(
            generated_at,
            strategy_config,
            evaluation_config,
            frames,
            load_failures,
            "not enough aligned bars for trend warmup plus one walk-forward split",
        )
        return report

    strategy = simulate_strategy(aligned, strategy_config)
    benchmark = simulate_buy_hold(aligned)
    full_sample = comparison_report(strategy, benchmark, evaluation_config)
    walk_forward = run_portfolio_walk_forward(aligned, strategy_config, evaluation_config)
    verdict, reasons = verdict_from_reports(full_sample, walk_forward, evaluation_config)
    return {
        "generated_at": generated_at.isoformat(),
        "name": "trend_vol_risk_config",
        "status": "OK",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "edge_thesis": EDGE_THESIS,
        "config": {
            "strategy": asdict(strategy_config),
            "evaluation": asdict(evaluation_config),
        },
        "universe": [asdict(item) for item in universe_items],
        "data": data_summary(aligned, frames, load_failures),
        "full_sample": full_sample,
        "walk_forward": walk_forward,
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only research; no trading signal or order path",
    }


def load_frames(
    universe: Iterable[UniverseItem],
    config: EvaluationConfig,
    *,
    verbose: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    for item in parse_universe_items(universe):
        key = f"{item.source}:{item.symbol}"
        try:
            if verbose:
                print(f"loading {key}", file=sys.stderr, flush=True)
            frames[key] = load_ohlcv(
                item.symbol,
                item.source,
                config.timeframe,
                config.start,
                config.end,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "symbol": item.symbol,
                    "source": item.source,
                    "asset_class": item.asset_class,
                    "status": "DATA_UNAVAILABLE",
                    "error": str(exc),
                }
            )
    return frames, failures


def align_closes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if "Close" not in frame.columns:
            continue
        columns[symbol] = pd.to_numeric(frame["Close"], errors="coerce")
    if not columns:
        return pd.DataFrame()
    result = pd.DataFrame(columns).sort_index().dropna(how="any")
    return cast(pd.DataFrame, result)


def simulate_strategy(closes: pd.DataFrame, config: StrategyConfig) -> dict[str, Any]:
    returns = closes.pct_change().fillna(0.0)
    raw_weights = target_weights(closes, config)
    equity = 1.0
    peak = 1.0
    previous_weights = pd.Series(0.0, index=closes.columns)
    cooldown = 0
    rows: list[dict[str, Any]] = []
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0

    for timestamp in closes.index:
        desired = cast(pd.Series, raw_weights.loc[timestamp]).fillna(0.0)
        pre_day_drawdown = 1.0 - equity / peak if peak > 0 else 0.0
        stop_triggered = False
        if pre_day_drawdown > config.drawdown_stop and cooldown <= 0:
            cooldown = config.drawdown_cooldown_bars
            stop_triggered = True
        if cooldown > 0:
            desired = desired * config.drawdown_reduction
            cooldown -= 1
        turnover = float((desired - previous_weights).abs().sum())
        cost = turnover * cost_rate
        gross_return = float((desired * cast(pd.Series, returns.loc[timestamp])).sum())
        net_return = gross_return - cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        rows.append(
            {
                "timestamp": timestamp,
                "return": net_return,
                "gross_return": gross_return,
                "cost": cost,
                "equity": equity,
                "gross_exposure": float(desired.abs().sum()),
                "turnover": turnover,
                "drawdown": 1.0 - equity / peak if peak > 0 else 0.0,
                "drawdown_stop_active": cooldown > 0 or stop_triggered,
            }
        )
        previous_weights = desired

    frame = pd.DataFrame(rows).set_index("timestamp")
    return {
        "kind": "trend_vol_drawdown",
        "daily": frame,
        "metrics": metrics_from_returns(cast(pd.Series, frame["return"])),
        "total_cost_pct": round(float(cast(pd.Series, frame["cost"]).sum()) * 100.0, 6),
        "average_gross_exposure": round(
            float(cast(pd.Series, frame["gross_exposure"]).mean()), 6
        ),
        "max_gross_exposure": round(float(cast(pd.Series, frame["gross_exposure"]).max()), 6),
        "turnover": {
            "total": round(float(cast(pd.Series, frame["turnover"]).sum()), 6),
            "daily_average": round(float(cast(pd.Series, frame["turnover"]).mean()), 6),
        },
        "drawdown_stop_days": int(cast(pd.Series, frame["drawdown_stop_active"]).sum()),
    }


def strategy_result_for_period(
    closes_with_warmup: pd.DataFrame,
    period_index: pd.Index,
    config: StrategyConfig,
) -> dict[str, Any]:
    full_result = simulate_strategy(closes_with_warmup, config)
    daily = cast(pd.DataFrame, full_result["daily"]).loc[period_index]
    return strategy_result_from_daily(daily)


def strategy_result_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "kind": "trend_vol_drawdown",
        "daily": daily,
        "metrics": metrics_from_returns(cast(pd.Series, daily["return"])),
        "total_cost_pct": round(float(cast(pd.Series, daily["cost"]).sum()) * 100.0, 6),
        "average_gross_exposure": round(
            float(cast(pd.Series, daily["gross_exposure"]).mean()), 6
        ),
        "max_gross_exposure": round(float(cast(pd.Series, daily["gross_exposure"]).max()), 6),
        "turnover": {
            "total": round(float(cast(pd.Series, daily["turnover"]).sum()), 6),
            "daily_average": round(float(cast(pd.Series, daily["turnover"]).mean()), 6),
        },
        "drawdown_stop_days": int(cast(pd.Series, daily["drawdown_stop_active"]).sum()),
    }


def target_weights(closes: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    returns = closes.pct_change()
    trend = closes.shift(1) > closes.shift(1).rolling(config.trend_window).mean()
    realized_vol = returns.shift(1).rolling(config.vol_window).std() * math.sqrt(TRADING_DAYS)
    asset_count = max(len(closes.columns), 1)
    base_weight = 1.0 / asset_count
    safe_vol = realized_vol.where(realized_vol != 0)
    exposure = (config.target_ann_vol / safe_vol).clip(
        lower=0.0,
        upper=config.max_asset_weight / base_weight,
    )
    weights = trend.astype(float) * exposure * base_weight
    gross = weights.abs().sum(axis=1)
    safe_gross = gross.where(gross != 0)
    scale = (config.max_gross_exposure / safe_gross).clip(upper=1.0)
    weights = weights.mul(scale.fillna(0.0), axis=0).fillna(0.0)
    return cast(pd.DataFrame, weights)


def simulate_buy_hold(closes: pd.DataFrame) -> dict[str, Any]:
    normalized = closes / closes.iloc[0]
    equity = normalized.mean(axis=1)
    returns = equity.pct_change().fillna(0.0)
    return {
        "kind": "equal_weight_buy_hold",
        "daily": pd.DataFrame({"return": returns, "equity": equity}, index=closes.index),
        "metrics": metrics_from_returns(cast(pd.Series, returns)),
        "total_cost_pct": 0.0,
    }


def metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
        }
    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(clean) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    ann_vol = float(clean.std()) * math.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    drawdown = 1.0 - equity / equity.cummax()
    max_drawdown = float(drawdown.max())
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0
    return {
        "total_return_pct": round(total_return * 100.0, 6),
        "ann_return_pct": round(ann_return * 100.0, 6),
        "ann_vol_pct": round(ann_vol * 100.0, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown_pct": round(max_drawdown * 100.0, 6),
        "calmar": round(calmar, 6),
    }


def comparison_report(
    strategy: dict[str, Any],
    benchmark: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    strategy_metrics = cast(dict[str, float], strategy["metrics"])
    benchmark_metrics = cast(dict[str, float], benchmark["metrics"])
    benchmark_dd = benchmark_metrics["max_drawdown_pct"]
    strategy_dd = strategy_metrics["max_drawdown_pct"]
    drawdown_reduction = (
        (benchmark_dd - strategy_dd) / benchmark_dd if benchmark_dd > 0 else 0.0
    )
    sharpe_delta = strategy_metrics["sharpe"] - benchmark_metrics["sharpe"]
    calmar_delta = strategy_metrics["calmar"] - benchmark_metrics["calmar"]
    return {
        "strategy": public_strategy_report(strategy),
        "benchmark": public_strategy_report(benchmark),
        "comparison": {
            "drawdown_reduction_ratio": round(drawdown_reduction, 6),
            "drawdown_reduction_pass": drawdown_reduction >= config.min_drawdown_reduction_ratio,
            "sharpe_delta": round(sharpe_delta, 6),
            "sharpe_not_worse_pass": sharpe_delta >= -config.max_sharpe_shortfall,
            "calmar_delta": round(calmar_delta, 6),
            "calmar_not_worse_pass": calmar_delta >= -config.max_calmar_shortfall,
            "net_costs_included": True,
        },
    }


def public_strategy_report(result: dict[str, Any]) -> dict[str, Any]:
    report = {
        "kind": result["kind"],
        "metrics": result["metrics"],
        "total_cost_pct": result["total_cost_pct"],
    }
    for key in (
        "average_gross_exposure",
        "max_gross_exposure",
        "turnover",
        "drawdown_stop_days",
    ):
        if key in result:
            report[key] = result[key]
    return report


def run_portfolio_walk_forward(
    closes: pd.DataFrame,
    base_config: StrategyConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    param_grid = parameter_grid(base_config)
    if len(param_grid) > evaluation_config.max_parameter_trials:
        raise ValueError("parameter grid exceeds max_parameter_trials")
    starts = window_starts(
        len(closes),
        evaluation_config.train_bars,
        evaluation_config.test_bars,
        evaluation_config.step_bars,
    )
    if not starts:
        return {
            "status": "INSUFFICIENT_DATA",
            "windows": [],
            "summary": {
                "windows": 0,
                "parameter_trials": len(param_grid),
                "oos_stable": False,
                "reason": "not enough bars for one train->test walk-forward split",
            },
        }

    windows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        train = closes.iloc[start : start + evaluation_config.train_bars]
        test = closes.iloc[
            start + evaluation_config.train_bars : start
            + evaluation_config.train_bars
            + evaluation_config.test_bars
        ]
        selected = select_params(train, param_grid, evaluation_config)
        test_strategy = strategy_result_for_period(
            closes.iloc[start : start + evaluation_config.train_bars + evaluation_config.test_bars],
            test.index,
            cast(StrategyConfig, selected["params"]),
        )
        test_benchmark = simulate_buy_hold(test)
        comparison = comparison_report(test_strategy, test_benchmark, evaluation_config)
        windows.append(
            {
                "index": index,
                "train_period": period(train),
                "test_period": period(test),
                "selected_params": asdict(cast(StrategyConfig, selected["params"])),
                "is_score": selected["score"],
                "oos": comparison,
            }
        )
    return {
        "status": "OK",
        "windows": windows,
        "summary": walk_forward_summary(windows, len(param_grid)),
        "selection_objective": (
            "IS score = drawdown_reduction + max(sharpe_delta, -2) + "
            "max(calmar_delta, -2); OOS pass still requires all gates"
        ),
    }


def parameter_grid(base_config: StrategyConfig) -> list[StrategyConfig]:
    trend_values = sorted({150, base_config.trend_window, 250})
    target_values = sorted({base_config.target_ann_vol, 0.10})
    drawdown_values = sorted({base_config.drawdown_stop, 0.25})
    return [
        replace(
            base_config,
            trend_window=trend,
            target_ann_vol=target,
            drawdown_stop=drawdown_stop,
        )
        for trend in trend_values
        for target in target_values
        for drawdown_stop in drawdown_values
    ]


def select_params(
    train: pd.DataFrame,
    param_grid: list[StrategyConfig],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    benchmark = simulate_buy_hold(train)
    for params in param_grid:
        strategy = simulate_strategy(train, params)
        report = comparison_report(strategy, benchmark, evaluation_config)
        comparison = cast(dict[str, Any], report["comparison"])
        score = (
            float(comparison["drawdown_reduction_ratio"])
            + max(float(comparison["sharpe_delta"]), -2.0)
            + max(float(comparison["calmar_delta"]), -2.0)
        )
        candidates.append({"params": params, "score": round(score, 6), "report": report})
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[0]


def walk_forward_summary(windows: list[dict[str, Any]], parameter_trials: int) -> dict[str, Any]:
    comparisons = [
        cast(dict[str, Any], cast(dict[str, Any], window["oos"])["comparison"])
        for window in windows
    ]
    pass_count = sum(
        1
        for comparison in comparisons
        if comparison["drawdown_reduction_pass"]
        and comparison["sharpe_not_worse_pass"]
        and comparison["calmar_not_worse_pass"]
    )
    drawdown_reductions = [float(item["drawdown_reduction_ratio"]) for item in comparisons]
    sharpe_deltas = [float(item["sharpe_delta"]) for item in comparisons]
    calmar_deltas = [float(item["calmar_delta"]) for item in comparisons]
    pass_share = pass_count / len(windows) if windows else 0.0
    stable = bool(
        windows
        and pass_share >= 0.60
        and statistics.median(drawdown_reductions) > 0.0
        and statistics.median(sharpe_deltas) >= 0.0
        and statistics.median(calmar_deltas) >= 0.0
    )
    reasons: list[str] = []
    if pass_share < 0.60:
        reasons.append("fewer than 60% of OOS windows passed all risk-adjusted gates")
    if drawdown_reductions and statistics.median(drawdown_reductions) <= 0.0:
        reasons.append("median OOS drawdown reduction was not positive")
    if sharpe_deltas and statistics.median(sharpe_deltas) < 0.0:
        reasons.append("median OOS Sharpe was worse than buy-and-hold")
    if calmar_deltas and statistics.median(calmar_deltas) < 0.0:
        reasons.append("median OOS Calmar was worse than buy-and-hold")
    return {
        "windows": len(windows),
        "parameter_trials": parameter_trials,
        "pass_count": pass_count,
        "pass_share": round(pass_share, 6),
        "median_drawdown_reduction_ratio": round(statistics.median(drawdown_reductions), 6),
        "median_sharpe_delta": round(statistics.median(sharpe_deltas), 6),
        "median_calmar_delta": round(statistics.median(calmar_deltas), 6),
        "oos_stable": stable,
        "reason": "; ".join(reasons) if reasons else "OOS risk-adjusted behavior is stable",
    }


def verdict_from_reports(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
    config: EvaluationConfig,
) -> tuple[str, list[str]]:
    comparison = cast(dict[str, Any], full_sample["comparison"])
    summary = cast(dict[str, Any], walk_forward.get("summary", {}))
    reasons: list[str] = []
    if not comparison["drawdown_reduction_pass"]:
        reasons.append(
            "full-sample max drawdown reduction below "
            f"{config.min_drawdown_reduction_ratio:.0%}"
        )
    if not comparison["sharpe_not_worse_pass"]:
        reasons.append("full-sample Sharpe worse than buy-and-hold")
    if not comparison["calmar_not_worse_pass"]:
        reasons.append("full-sample Calmar worse than buy-and-hold")
    if walk_forward.get("status") != "OK":
        reasons.append("walk-forward data insufficient")
    elif summary.get("oos_stable") is not True:
        reasons.append(f"walk-forward OOS not stable: {summary.get('reason')}")
    if reasons:
        return VERDICT_FAIL, reasons
    return VERDICT_PASS, [
        "drawdown significantly lower, Sharpe/Calmar not worse, OOS stable, costs included"
    ]


def insufficient_report(
    generated_at: datetime,
    strategy_config: StrategyConfig,
    evaluation_config: EvaluationConfig,
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
    reason: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "name": "trend_vol_risk_config",
        "status": "INSUFFICIENT_DATA",
        "verdict": VERDICT_DATA,
        "verdict_reasons": [reason],
        "edge_thesis": EDGE_THESIS,
        "config": {
            "strategy": asdict(strategy_config),
            "evaluation": asdict(evaluation_config),
        },
        "data": data_summary(pd.DataFrame(), frames, load_failures),
        "full_sample": None,
        "walk_forward": {
            "status": "INSUFFICIENT_DATA",
            "summary": {"windows": 0, "oos_stable": False, "reason": reason},
            "windows": [],
        },
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only research; no trading signal or order path",
    }


def data_summary(
    aligned: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "loaded_symbols": len(frames),
        "aligned_bars": len(aligned),
        "start": _date_value(aligned.index[0]) if len(aligned) else None,
        "end": _date_value(aligned.index[-1]) if len(aligned) else None,
        "symbols": sorted(frames),
        "load_failures": load_failures,
    }


def safety_statement() -> dict[str, Any]:
    return {
        "candidate_only": True,
        "paper_research_only": True,
        "order_path_added": False,
        "strategy_plugin_registered": False,
        "risk_gate_changes": False,
        "live_trading_changes": False,
        "public_bind_changes": False,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"olympus27-trend-vol-risk-config-{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    written_files = {"json": str(json_path), "markdown": str(md_path)}
    persisted_report = {**report, "written_files": written_files}
    json_path.write_text(
        json.dumps(to_jsonable(persisted_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(markdown_report(persisted_report, json_path), encoding="utf-8")
    return written_files


def markdown_report(report: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Olympus #27 Trend/Vol/Risk Configuration",
        "",
        f"- Generated: {report['generated_at']}",
        f"- JSON evidence: `{json_path}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Edge thesis: {report['edge_thesis']}",
        "- Discipline: candidate-only, paper research, no order path.",
        "",
    ]
    full_sample = report.get("full_sample")
    if isinstance(full_sample, dict):
        strategy = cast(dict[str, Any], cast(dict[str, Any], full_sample["strategy"])["metrics"])
        benchmark = cast(dict[str, Any], cast(dict[str, Any], full_sample["benchmark"])["metrics"])
        comparison = cast(dict[str, Any], full_sample["comparison"])
        lines.extend(
            [
                "## Full Sample",
                "",
                "| metric | strategy | buy_hold | delta/pass |",
                "|---|---:|---:|---|",
                "| max_drawdown_pct | "
                f"{strategy['max_drawdown_pct']} | {benchmark['max_drawdown_pct']} | "
                f"reduction={comparison['drawdown_reduction_ratio']} "
                f"pass={comparison['drawdown_reduction_pass']} |",
                "| sharpe | "
                f"{strategy['sharpe']} | {benchmark['sharpe']} | "
                f"delta={comparison['sharpe_delta']} pass={comparison['sharpe_not_worse_pass']} |",
                "| calmar | "
                f"{strategy['calmar']} | {benchmark['calmar']} | "
                f"delta={comparison['calmar_delta']} pass={comparison['calmar_not_worse_pass']} |",
                "| total_return_pct | "
                f"{strategy['total_return_pct']} | {benchmark['total_return_pct']} | "
                "not an alpha gate |",
                "",
            ]
        )
    wf = cast(dict[str, Any], report.get("walk_forward", {}))
    summary = cast(dict[str, Any], wf.get("summary", {}))
    lines.extend(
        [
            "## Walk-Forward OOS",
            "",
            f"- Status: `{wf.get('status')}`",
            f"- Windows: {summary.get('windows')}",
            f"- Parameter trials: {summary.get('parameter_trials')}",
            f"- OOS stable: {summary.get('oos_stable')}",
            f"- Reason: {summary.get('reason')}",
            "",
            "## Safety",
            "",
            "- candidate-only research; no paper/live setting changed.",
            "- no strategy plugin, live order code, API key, or public bind added.",
        ]
    )
    return "\n".join(lines) + "\n"


def to_jsonable(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return "<omitted:dataframe>"
    if isinstance(value, pd.Series):
        return "<omitted:series>"
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def validate_configs(strategy: StrategyConfig, evaluation: EvaluationConfig) -> None:
    positive_ints = {
        "trend_window": strategy.trend_window,
        "vol_window": strategy.vol_window,
        "drawdown_cooldown_bars": strategy.drawdown_cooldown_bars,
        "train_bars": evaluation.train_bars,
        "test_bars": evaluation.test_bars,
        "step_bars": evaluation.step_bars,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    bounded = {
        "target_ann_vol": strategy.target_ann_vol,
        "max_asset_weight": strategy.max_asset_weight,
        "max_gross_exposure": strategy.max_gross_exposure,
        "drawdown_stop": strategy.drawdown_stop,
        "min_drawdown_reduction_ratio": evaluation.min_drawdown_reduction_ratio,
    }
    for name, value in bounded.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= strategy.drawdown_reduction <= 1.0:
        raise ValueError("drawdown_reduction must be between 0 and 1")
    if strategy.fee_bps < 0 or strategy.slippage_bps < 0:
        raise ValueError("cost assumptions must be non-negative")


def window_starts(total_bars: int, train_bars: int, test_bars: int, step_bars: int) -> list[int]:
    starts: list[int] = []
    start = 0
    while start + train_bars + test_bars <= total_bars:
        starts.append(start)
        start += step_bars
    return starts


def period(frame: pd.DataFrame) -> dict[str, str]:
    return {"start": _date_value(frame.index[0]), "end": _date_value(frame.index[-1])}


def parse_universe_items(items: Iterable[UniverseItem]) -> list[UniverseItem]:
    return list(items)


def _date_value(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


if __name__ == "__main__":
    raise SystemExit(main())
