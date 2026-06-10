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
from typing import Any, Literal, cast

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
AllocationMethod = Literal["vol_target", "risk_parity", "risk_parity_vol_target"]

EDGE_THESIS = (
    "不预测方向;按已实现风险分配和目标波动率控制总暴露,叠加回撤硬止损与定期再平衡,"
    "追求风险调整稳健和显著低于buy&hold的回撤"
)
VERDICT_PASS = "DIVERSIFIED_RISK_ALLOCATION_CANDIDATE_INCUBATING"
VERDICT_FAIL = "NO_ROBUST_DIVERSIFIED_RISK_ALLOCATION_EDGE"
VERDICT_DATA = "DIVERSIFIED_RISK_ALLOCATION_DATA_INSUFFICIENT"


@dataclass(frozen=True)
class UniverseItem:
    symbol: str
    source: Source
    asset_class: str


@dataclass(frozen=True)
class StrategyConfig:
    method: AllocationMethod
    vol_window: int
    target_ann_vol: float
    rebalance_bars: int
    min_asset_weight: float
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
    UniverseItem("SPY", "yfinance", "equity"),
    UniverseItem("GLD", "yfinance", "gold"),
    UniverseItem("TLT", "yfinance", "bond"),
)

DEFAULT_STRATEGY_CONFIG = StrategyConfig(
    method="risk_parity_vol_target",
    vol_window=30,
    target_ann_vol=0.30,
    rebalance_bars=20,
    min_asset_weight=0.0,
    max_asset_weight=0.60,
    max_gross_exposure=1.00,
    drawdown_stop=0.35,
    drawdown_reduction=0.50,
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
        description="Evaluate Olympus #29 diversified pure risk allocation candidate."
    )
    parser.add_argument("--start", default=_env_text("START", DEFAULT_START))
    parser.add_argument("--end", default=_env_text("END", DEFAULT_END))
    parser.add_argument("--timeframe", default=_env_text("TIMEFRAME", DEFAULT_TIMEFRAME))
    parser.add_argument(
        "--output-dir",
        default=_env_text("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
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
        universe=parse_universe(_env_text("UNIVERSE", "")),
        verbose=True,
    )
    if not args.no_write:
        report["written_files"] = write_report(report, Path(str(args.output_dir)))
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["verdict"] != VERDICT_DATA else 1


def env_strategy_config() -> StrategyConfig:
    return StrategyConfig(
        method=cast(
            AllocationMethod,
            _env_text("METHOD", DEFAULT_STRATEGY_CONFIG.method),
        ),
        vol_window=_env_int("VOL_WINDOW", DEFAULT_STRATEGY_CONFIG.vol_window),
        target_ann_vol=_env_float(
            "TARGET_ANN_VOL", DEFAULT_STRATEGY_CONFIG.target_ann_vol
        ),
        rebalance_bars=_env_int(
            "REBALANCE_BARS", DEFAULT_STRATEGY_CONFIG.rebalance_bars
        ),
        min_asset_weight=_env_float(
            "MIN_ASSET_WEIGHT", DEFAULT_STRATEGY_CONFIG.min_asset_weight
        ),
        max_asset_weight=_env_float(
            "MAX_ASSET_WEIGHT", DEFAULT_STRATEGY_CONFIG.max_asset_weight
        ),
        max_gross_exposure=_env_float(
            "MAX_GROSS_EXPOSURE", DEFAULT_STRATEGY_CONFIG.max_gross_exposure
        ),
        drawdown_stop=_env_float("DRAWDOWN_STOP", DEFAULT_STRATEGY_CONFIG.drawdown_stop),
        drawdown_reduction=_env_float(
            "DRAWDOWN_REDUCTION", DEFAULT_STRATEGY_CONFIG.drawdown_reduction
        ),
        drawdown_cooldown_bars=_env_int(
            "DRAWDOWN_COOLDOWN_BARS",
            DEFAULT_STRATEGY_CONFIG.drawdown_cooldown_bars,
        ),
        fee_bps=_env_float("FEE_BPS", DEFAULT_STRATEGY_CONFIG.fee_bps),
        slippage_bps=_env_float("SLIPPAGE_BPS", DEFAULT_STRATEGY_CONFIG.slippage_bps),
    )


def env_evaluation_config(*, start: str, end: str, timeframe: str) -> EvaluationConfig:
    return EvaluationConfig(
        start=start,
        end=end,
        timeframe=timeframe,
        train_bars=_env_int("TRAIN_BARS", DEFAULT_EVALUATION_CONFIG.train_bars),
        test_bars=_env_int("TEST_BARS", DEFAULT_EVALUATION_CONFIG.test_bars),
        step_bars=_env_int("STEP_BARS", DEFAULT_EVALUATION_CONFIG.step_bars),
        min_drawdown_reduction_ratio=_env_float(
            "MIN_DRAWDOWN_REDUCTION_RATIO",
            DEFAULT_EVALUATION_CONFIG.min_drawdown_reduction_ratio,
        ),
        max_sharpe_shortfall=_env_float(
            "MAX_SHARPE_SHORTFALL",
            DEFAULT_EVALUATION_CONFIG.max_sharpe_shortfall,
        ),
        max_calmar_shortfall=_env_float(
            "MAX_CALMAR_SHORTFALL",
            DEFAULT_EVALUATION_CONFIG.max_calmar_shortfall,
        ),
        max_parameter_trials=_env_int(
            "MAX_PARAMETER_TRIALS",
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
                "OLYMPUS29_UNIVERSE entries must be source:symbol[:asset_class]"
            )
        source = cast(Source, fields[0])
        if source not in ("binance", "okx", "bybit", "yfinance"):
            raise ValueError(f"unsupported source in OLYMPUS29_UNIVERSE: {source}")
        items.append(
            UniverseItem(
                symbol=fields[1],
                source=source,
                asset_class=fields[2] if len(fields) == 3 else "unknown",
            )
        )
    if not items:
        raise ValueError("OLYMPUS29_UNIVERSE must contain at least one asset")
    return tuple(items)


def run_evaluation(
    *,
    strategy_config: StrategyConfig,
    evaluation_config: EvaluationConfig,
    universe: Iterable[UniverseItem],
    verbose: bool = False,
) -> dict[str, Any]:
    validate_configs(strategy_config, evaluation_config)
    universe_items = list(universe)
    generated_at = datetime.now(UTC)
    frames, load_failures = load_frames(universe_items, evaluation_config, verbose=verbose)
    aligned = align_closes(frames)
    universe_status = universe_assessment(universe_items, frames, load_failures, aligned)
    if universe_status["status"] == "DATA_INSUFFICIENT":
        return insufficient_report(
            generated_at,
            strategy_config,
            evaluation_config,
            universe_items,
            frames,
            load_failures,
            str(universe_status["reason"]),
        )
    min_bars = max(
        strategy_config.vol_window + strategy_config.rebalance_bars + 20,
        evaluation_config.train_bars + evaluation_config.test_bars,
    )
    if aligned.empty or len(aligned) < min_bars:
        return insufficient_report(
            generated_at,
            strategy_config,
            evaluation_config,
            universe_items,
            frames,
            load_failures,
            "not enough aligned bars for volatility warmup plus one walk-forward split",
        )

    strategy = simulate_strategy(aligned, strategy_config)
    buy_hold = simulate_buy_hold(aligned)
    equal_weight = simulate_equal_weight_rebalanced(aligned, strategy_config)
    full_sample = dual_benchmark_report(strategy, buy_hold, equal_weight, evaluation_config)
    walk_forward = run_portfolio_walk_forward(aligned, strategy_config, evaluation_config)
    verdict, reasons = verdict_from_reports(full_sample, walk_forward, evaluation_config)
    return {
        "generated_at": generated_at.isoformat(),
        "name": "pure_risk_allocation",
        "status": "OK",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "edge_thesis": EDGE_THESIS,
        "config": {
            "strategy": asdict(strategy_config),
            "evaluation": asdict(evaluation_config),
            "predeclared_parameter_grid": [
                asdict(params) for params in parameter_grid(strategy_config)
            ],
        },
        "universe": [asdict(item) for item in universe_items],
        "universe_assessment": universe_status,
        "data": data_summary(aligned, frames, load_failures),
        "full_sample": full_sample,
        "walk_forward": walk_forward,
        "olympus28_contrast": olympus28_contrast(walk_forward),
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only research; no trading signal or order path",
    }


def universe_assessment(
    universe: list[UniverseItem],
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
    aligned: pd.DataFrame,
) -> dict[str, Any]:
    requested_classes = sorted({item.asset_class for item in universe})
    loaded_classes = sorted(
        {
            item.asset_class
            for item in universe
            if f"{item.source}:{item.symbol}" in frames
        }
    )
    pairwise_corr = correlation_matrix(aligned)
    average_abs_corr = average_abs_pairwise_corr(aligned)
    status = "OK"
    reason = (
        "predeclared diversified basket loaded; risk parity has a cross-asset structure to test"
    )
    if load_failures and len(loaded_classes) < 3:
        status = "DATA_INSUFFICIENT"
        reason = (
            "predeclared multi-asset history was unavailable and the loaded subset no longer "
            "contains enough distinct asset classes for a fair diversified risk-parity test"
        )
    elif load_failures:
        status = "PARTIAL_UNIVERSE"
        reason = (
            "some predeclared assets failed to load; evaluation uses only the explicitly reported "
            "loaded subset and must be treated as universe-limited"
        )
    elif average_abs_corr is not None and average_abs_corr >= 0.70:
        status = "HIGH_CORRELATION_UNIVERSE"
        reason = (
            "loaded assets are highly correlated on average, so the universe remains an unfair "
            "risk-parity test bed"
        )
    return {
        "status": status,
        "reason": reason,
        "requested_asset_classes": requested_classes,
        "loaded_asset_classes": loaded_classes,
        "requested_symbols": [f"{item.source}:{item.symbol}" for item in universe],
        "loaded_symbols": sorted(frames),
        "correlation_matrix": pairwise_corr,
        "average_abs_pairwise_correlation": average_abs_corr,
        "annualized_volatility_pct": annualized_volatility_summary(aligned),
        "data_failures_explicit": bool(load_failures),
    }


def load_frames(
    universe: Iterable[UniverseItem],
    config: EvaluationConfig,
    *,
    verbose: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    for item in universe:
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
        if "Close" in frame.columns:
            columns[symbol] = pd.to_numeric(frame["Close"], errors="coerce")
    if not columns:
        return pd.DataFrame()
    result = pd.DataFrame(columns).sort_index().dropna(how="any")
    return cast(pd.DataFrame, result)


def simulate_strategy(closes: pd.DataFrame, config: StrategyConfig) -> dict[str, Any]:
    returns = closes.pct_change().fillna(0.0)
    scheduled_weights = pure_risk_weights(closes, config)
    equity = 1.0
    peak = 1.0
    previous_weights = pd.Series(0.0, index=closes.columns)
    current_weights = pd.Series(0.0, index=closes.columns)
    cooldown = 0
    rows: list[dict[str, Any]] = []
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0

    for offset, timestamp in enumerate(closes.index):
        if offset % config.rebalance_bars == 0:
            current_weights = cast(pd.Series, scheduled_weights.loc[timestamp]).fillna(0.0)
        desired = current_weights.copy()
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
    return result_from_daily("pure_risk_allocation", frame)


def strategy_result_for_period(
    closes_with_warmup: pd.DataFrame,
    period_index: pd.Index,
    config: StrategyConfig,
) -> dict[str, Any]:
    full_result = simulate_strategy(closes_with_warmup, config)
    daily = cast(pd.DataFrame, full_result["daily"]).loc[period_index]
    return result_from_daily("pure_risk_allocation", daily)


def result_from_daily(kind: str, daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "kind": kind,
        "daily": daily,
        "metrics": metrics_from_returns(cast(pd.Series, daily["return"])),
        "total_cost_pct": round(float(cast(pd.Series, daily["cost"]).sum()) * 100.0, 6)
        if "cost" in daily
        else 0.0,
        "average_gross_exposure": round(
            float(cast(pd.Series, daily["gross_exposure"]).mean()), 6
        )
        if "gross_exposure" in daily
        else None,
        "max_gross_exposure": round(float(cast(pd.Series, daily["gross_exposure"]).max()), 6)
        if "gross_exposure" in daily
        else None,
        "turnover": {
            "total": round(float(cast(pd.Series, daily["turnover"]).sum()), 6),
            "daily_average": round(float(cast(pd.Series, daily["turnover"]).mean()), 6),
        }
        if "turnover" in daily
        else None,
        "drawdown_stop_days": int(cast(pd.Series, daily["drawdown_stop_active"]).sum())
        if "drawdown_stop_active" in daily
        else None,
    }


def pure_risk_weights(closes: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    returns = closes.pct_change()
    realized_vol = returns.shift(1).rolling(config.vol_window).std() * math.sqrt(TRADING_DAYS)
    safe_vol = realized_vol.where(realized_vol > 0)
    inv_vol = 1.0 / safe_vol
    risk_parity = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    asset_count = max(len(closes.columns), 1)
    equal = pd.DataFrame(1.0 / asset_count, index=closes.index, columns=closes.columns)

    if config.method == "vol_target":
        base = equal
    else:
        base = risk_parity

    if config.min_asset_weight > 0 or config.max_asset_weight < 1.0:
        base = base.clip(lower=config.min_asset_weight, upper=config.max_asset_weight)
        gross_after_clip = base.sum(axis=1).where(lambda item: item > 0)
        base = base.div(gross_after_clip, axis=0).fillna(0.0)

    if config.method in ("vol_target", "risk_parity_vol_target"):
        portfolio_vol = portfolio_realized_vol(returns, base, config.vol_window)
        exposure = (config.target_ann_vol / portfolio_vol.where(portfolio_vol > 0)).clip(
            lower=0.0,
            upper=config.max_gross_exposure,
        )
        weights = base.mul(exposure.fillna(0.0), axis=0)
    else:
        weights = base * config.max_gross_exposure

    gross = weights.abs().sum(axis=1)
    scale = (config.max_gross_exposure / gross.where(gross > 0)).clip(upper=1.0)
    weights = weights.mul(scale.fillna(0.0), axis=0).fillna(0.0)
    return cast(pd.DataFrame, weights)


def portfolio_realized_vol(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    vol_window: int,
) -> pd.Series:
    weighted_returns = (returns.shift(1).fillna(0.0) * weights.shift(1).fillna(0.0)).sum(axis=1)
    vol = weighted_returns.rolling(vol_window).std() * math.sqrt(TRADING_DAYS)
    return cast(pd.Series, vol)


def simulate_buy_hold(closes: pd.DataFrame) -> dict[str, Any]:
    normalized = closes / closes.iloc[0]
    equity = normalized.mean(axis=1)
    returns = equity.pct_change().fillna(0.0)
    daily = pd.DataFrame({"return": returns, "equity": equity}, index=closes.index)
    return {
        "kind": "initial_equal_weight_buy_hold",
        "daily": daily,
        "metrics": metrics_from_returns(cast(pd.Series, returns)),
        "total_cost_pct": 0.0,
    }


def simulate_equal_weight_rebalanced(
    closes: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, Any]:
    returns = closes.pct_change().fillna(0.0)
    weight = 1.0 / max(len(closes.columns), 1)
    target = pd.Series(weight, index=closes.columns)
    previous = pd.Series(0.0, index=closes.columns)
    rows: list[dict[str, Any]] = []
    equity = 1.0
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    for offset, timestamp in enumerate(closes.index):
        desired = target if offset % config.rebalance_bars == 0 else previous
        turnover = float((desired - previous).abs().sum())
        cost = turnover * cost_rate
        net_return = float((desired * cast(pd.Series, returns.loc[timestamp])).sum()) - cost
        equity *= 1.0 + net_return
        rows.append(
            {
                "timestamp": timestamp,
                "return": net_return,
                "cost": cost,
                "equity": equity,
                "gross_exposure": float(desired.abs().sum()),
                "turnover": turnover,
            }
        )
        previous = desired
    daily = pd.DataFrame(rows).set_index("timestamp")
    return result_from_daily("equal_weight_rebalanced", daily)


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


def dual_benchmark_report(
    strategy: dict[str, Any],
    buy_hold: dict[str, Any],
    equal_weight: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    return {
        "strategy": public_strategy_report(strategy),
        "benchmarks": {
            "buy_hold": public_strategy_report(buy_hold),
            "equal_weight": public_strategy_report(equal_weight),
        },
        "comparison": {
            "buy_hold": comparison_against(strategy, buy_hold, config),
            "equal_weight": comparison_against(strategy, equal_weight, config),
            "net_costs_included": True,
        },
    }


def comparison_against(
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
        "drawdown_reduction_ratio": round(drawdown_reduction, 6),
        "drawdown_reduction_pass": drawdown_reduction >= config.min_drawdown_reduction_ratio,
        "sharpe_delta": round(sharpe_delta, 6),
        "sharpe_not_worse_pass": sharpe_delta >= -config.max_sharpe_shortfall,
        "calmar_delta": round(calmar_delta, 6),
        "calmar_not_worse_pass": calmar_delta >= -config.max_calmar_shortfall,
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
        if result.get(key) is not None:
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
        warm = closes.iloc[
            start : start + evaluation_config.train_bars + evaluation_config.test_bars
        ]
        test_strategy = strategy_result_for_period(
            warm,
            test.index,
            cast(StrategyConfig, selected["params"]),
        )
        test_buy_hold = simulate_buy_hold(test)
        test_equal_weight = simulate_equal_weight_rebalanced(
            test,
            cast(StrategyConfig, selected["params"]),
        )
        comparison = dual_benchmark_report(
            test_strategy,
            test_buy_hold,
            test_equal_weight,
            evaluation_config,
        )
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
            "IS score averages dual-benchmark drawdown reduction, Sharpe delta, and Calmar delta; "
            "OOS pass still requires all gates versus both benchmarks"
        ),
    }


def parameter_grid(base_config: StrategyConfig) -> list[StrategyConfig]:
    vol_values = sorted({base_config.vol_window, 60})
    target_values = sorted({base_config.target_ann_vol, 0.25, 0.35})
    rebalance_values = sorted({base_config.rebalance_bars, 63})
    return [
        replace(
            base_config,
            vol_window=vol_window,
            target_ann_vol=target,
            rebalance_bars=rebalance_bars,
        )
        for vol_window in vol_values
        for target in target_values
        for rebalance_bars in rebalance_values
    ]


def select_params(
    train: pd.DataFrame,
    param_grid: list[StrategyConfig],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    buy_hold = simulate_buy_hold(train)
    for params in param_grid:
        strategy = simulate_strategy(train, params)
        equal_weight = simulate_equal_weight_rebalanced(train, params)
        report = dual_benchmark_report(strategy, buy_hold, equal_weight, evaluation_config)
        comparisons = cast(dict[str, Any], report["comparison"])
        score = statistics.mean(
            [
                risk_adjusted_score(cast(dict[str, Any], comparisons["buy_hold"])),
                risk_adjusted_score(cast(dict[str, Any], comparisons["equal_weight"])),
            ]
        )
        candidates.append({"params": params, "score": round(score, 6), "report": report})
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[0]


def risk_adjusted_score(comparison: dict[str, Any]) -> float:
    return (
        float(comparison["drawdown_reduction_ratio"])
        + max(float(comparison["sharpe_delta"]), -2.0)
        + max(float(comparison["calmar_delta"]), -2.0)
    )


def walk_forward_summary(windows: list[dict[str, Any]], parameter_trials: int) -> dict[str, Any]:
    buy_hold: list[dict[str, Any]] = []
    equal_weight: list[dict[str, Any]] = []
    for window in windows:
        comparison = cast(
            dict[str, Any],
            cast(dict[str, Any], window["oos"])["comparison"],
        )
        buy_hold.append(cast(dict[str, Any], comparison["buy_hold"]))
        equal_weight.append(cast(dict[str, Any], comparison["equal_weight"]))
    pass_count = sum(
        1
        for bh, ew in zip(buy_hold, equal_weight, strict=True)
        if benchmark_passed(bh) and benchmark_passed(ew)
    )
    bh_drawdowns = [float(item["drawdown_reduction_ratio"]) for item in buy_hold]
    ew_drawdowns = [float(item["drawdown_reduction_ratio"]) for item in equal_weight]
    bh_sharpes = [float(item["sharpe_delta"]) for item in buy_hold]
    ew_sharpes = [float(item["sharpe_delta"]) for item in equal_weight]
    bh_calmars = [float(item["calmar_delta"]) for item in buy_hold]
    ew_calmars = [float(item["calmar_delta"]) for item in equal_weight]
    pass_share = pass_count / len(windows) if windows else 0.0
    stable = bool(
        windows
        and pass_share >= 0.60
        and min(statistics.median(bh_drawdowns), statistics.median(ew_drawdowns)) > 0.0
        and min(statistics.median(bh_sharpes), statistics.median(ew_sharpes)) >= 0.0
        and min(statistics.median(bh_calmars), statistics.median(ew_calmars)) >= 0.0
    )
    reasons: list[str] = []
    if pass_share < 0.60:
        reasons.append("fewer than 60% of OOS windows passed all dual-benchmark gates")
    if bh_drawdowns and statistics.median(bh_drawdowns) <= 0.0:
        reasons.append("median OOS drawdown reduction versus buy&hold was not positive")
    if ew_drawdowns and statistics.median(ew_drawdowns) <= 0.0:
        reasons.append("median OOS drawdown reduction versus equal-weight was not positive")
    if bh_sharpes and statistics.median(bh_sharpes) < 0.0:
        reasons.append("median OOS Sharpe was worse than buy&hold")
    if ew_sharpes and statistics.median(ew_sharpes) < 0.0:
        reasons.append("median OOS Sharpe was worse than equal-weight")
    if bh_calmars and statistics.median(bh_calmars) < 0.0:
        reasons.append("median OOS Calmar was worse than buy&hold")
    if ew_calmars and statistics.median(ew_calmars) < 0.0:
        reasons.append("median OOS Calmar was worse than equal-weight")
    return {
        "windows": len(windows),
        "parameter_trials": parameter_trials,
        "pass_count": pass_count,
        "pass_share": round(pass_share, 6),
        "buy_hold": median_summary(bh_drawdowns, bh_sharpes, bh_calmars),
        "equal_weight": median_summary(ew_drawdowns, ew_sharpes, ew_calmars),
        "oos_stable": stable,
        "reason": "; ".join(reasons) if reasons else "OOS risk-adjusted behavior is stable",
    }


def benchmark_passed(comparison: dict[str, Any]) -> bool:
    return bool(
        comparison["drawdown_reduction_pass"]
        and comparison["sharpe_not_worse_pass"]
        and comparison["calmar_not_worse_pass"]
    )


def median_summary(
    drawdowns: list[float],
    sharpes: list[float],
    calmars: list[float],
) -> dict[str, float]:
    return {
        "median_drawdown_reduction_ratio": round(statistics.median(drawdowns), 6),
        "median_sharpe_delta": round(statistics.median(sharpes), 6),
        "median_calmar_delta": round(statistics.median(calmars), 6),
    }


def verdict_from_reports(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
    config: EvaluationConfig,
) -> tuple[str, list[str]]:
    comparison = cast(dict[str, Any], full_sample["comparison"])
    summary = cast(dict[str, Any], walk_forward.get("summary", {}))
    reasons: list[str] = []
    for benchmark_name in ("buy_hold", "equal_weight"):
        benchmark = cast(dict[str, Any], comparison[benchmark_name])
        if not benchmark["drawdown_reduction_pass"]:
            reasons.append(
                f"full-sample max drawdown reduction versus {benchmark_name} below "
                f"{config.min_drawdown_reduction_ratio:.0%}"
            )
        if not benchmark["sharpe_not_worse_pass"]:
            reasons.append(f"full-sample Sharpe worse than {benchmark_name}")
        if not benchmark["calmar_not_worse_pass"]:
            reasons.append(f"full-sample Calmar worse than {benchmark_name}")
    if walk_forward.get("status") != "OK":
        reasons.append("walk-forward data insufficient")
    elif summary.get("oos_stable") is not True:
        reasons.append(f"walk-forward OOS not stable: {summary.get('reason')}")
    if reasons:
        return VERDICT_FAIL, reasons
    return VERDICT_PASS, [
        (
            "drawdown significantly lower, Sharpe/Calmar not worse versus both benchmarks, "
            "OOS stable, costs included"
        )
    ]


def olympus28_contrast(walk_forward: dict[str, Any]) -> dict[str, Any]:
    summary = cast(dict[str, Any], walk_forward.get("summary", {}))
    return {
        "olympus28_verdict": "NO_ROBUST_PURE_RISK_ALLOCATION_EDGE",
        "olympus28_oos_note": (
            "BTC/ETH pure-risk candidate had 14 OOS windows, 6 passes, "
            "pass_share=0.428571, median Sharpe delta=-0.049006/-0.016257 versus "
            "buy&hold/equal-weight, and median Calmar delta=-0.040191/+0.019953"
        ),
        "diversified_pure_risk_oos_stable": summary.get("oos_stable"),
        "diversified_pure_risk_pass_share": summary.get("pass_share"),
        "interpretation": (
            "This report tests whether #28 failed because BTC/ETH was too correlated and the "
            "15% target-vol setting was too conservative; the final verdict is still based on "
            "dual-benchmark OOS gates, not that hypothesis."
        ),
    }


def insufficient_report(
    generated_at: datetime,
    strategy_config: StrategyConfig,
    evaluation_config: EvaluationConfig,
    universe_items: list[UniverseItem],
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
    reason: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "name": "pure_risk_allocation",
        "status": "INSUFFICIENT_DATA",
        "verdict": VERDICT_DATA,
        "verdict_reasons": [reason],
        "edge_thesis": EDGE_THESIS,
        "config": {
            "strategy": asdict(strategy_config),
            "evaluation": asdict(evaluation_config),
            "predeclared_parameter_grid": [
                asdict(params) for params in parameter_grid(strategy_config)
            ],
        },
        "universe": [asdict(item) for item in universe_items],
        "universe_assessment": universe_assessment(
            universe_items, frames, load_failures, pd.DataFrame()
        ),
        "data": data_summary(pd.DataFrame(), frames, load_failures),
        "full_sample": None,
        "walk_forward": {
            "status": "INSUFFICIENT_DATA",
            "summary": {"windows": 0, "oos_stable": False, "reason": reason},
            "windows": [],
        },
        "olympus28_contrast": {"status": "not_comparable_without_data"},
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
        "correlation_matrix": correlation_matrix(aligned),
        "average_abs_pairwise_correlation": average_abs_pairwise_corr(aligned),
        "annualized_volatility_pct": annualized_volatility_summary(aligned),
    }


def correlation_matrix(aligned: pd.DataFrame) -> dict[str, dict[str, float]] | None:
    if len(aligned.columns) < 2 or len(aligned) < 2:
        return None
    corr = aligned.pct_change().dropna().corr()
    return {
        str(index): {
            str(column): round(float(value), 6)
            for column, value in cast(pd.Series, row).items()
        }
        for index, row in corr.iterrows()
    }


def average_abs_pairwise_corr(aligned: pd.DataFrame) -> float | None:
    if len(aligned.columns) < 2 or len(aligned) < 2:
        return None
    corr = aligned.pct_change().dropna().corr().abs()
    values: list[float] = []
    columns = list(corr.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            values.append(float(corr.loc[left, right]))
    return round(statistics.mean(values), 6) if values else None


def annualized_volatility_summary(aligned: pd.DataFrame) -> dict[str, float] | None:
    if aligned.empty:
        return None
    vols = aligned.pct_change().dropna().std() * math.sqrt(TRADING_DAYS) * 100.0
    return {str(symbol): round(float(vol), 6) for symbol, vol in vols.items()}


def safety_statement() -> dict[str, Any]:
    return {
        "candidate_only": True,
        "paper_research_only": True,
        "directional_timing_added": False,
        "order_path_added": False,
        "strategy_plugin_registered": False,
        "risk_gate_changes": False,
        "live_trading_changes": False,
        "public_bind_changes": False,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"olympus29-diversified-pure-risk-allocation-{stamp}"
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
    universe_status = cast(dict[str, Any], report.get("universe_assessment", {})).get("status")
    lines = [
        "# Olympus #29 Diversified Pure Risk Allocation",
        "",
        f"- Generated: {report['generated_at']}",
        f"- JSON evidence: `{json_path}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Edge thesis: {report['edge_thesis']}",
        f"- Universe status: `{universe_status}`",
        "- Discipline: candidate-only, paper research, no order path, no directional timing.",
        "",
    ]
    full_sample = report.get("full_sample")
    if isinstance(full_sample, dict):
        strategy = cast(dict[str, Any], cast(dict[str, Any], full_sample["strategy"])["metrics"])
        benchmarks = cast(dict[str, Any], full_sample["benchmarks"])
        comparisons = cast(dict[str, Any], full_sample["comparison"])
        for name in ("buy_hold", "equal_weight"):
            benchmark = cast(dict[str, Any], cast(dict[str, Any], benchmarks[name])["metrics"])
            comparison = cast(dict[str, Any], comparisons[name])
            lines.extend(
                [
                    f"## Full Sample vs {name}",
                    "",
                    "| metric | strategy | benchmark | delta/pass |",
                    "|---|---:|---:|---|",
                    "| max_drawdown_pct | "
                    f"{strategy['max_drawdown_pct']} | {benchmark['max_drawdown_pct']} | "
                    f"reduction={comparison['drawdown_reduction_ratio']} "
                    f"pass={comparison['drawdown_reduction_pass']} |",
                    "| sharpe | "
                    f"{strategy['sharpe']} | {benchmark['sharpe']} | "
                    f"delta={comparison['sharpe_delta']} "
                    f"pass={comparison['sharpe_not_worse_pass']} |",
                    "| calmar | "
                    f"{strategy['calmar']} | {benchmark['calmar']} | "
                    f"delta={comparison['calmar_delta']} "
                    f"pass={comparison['calmar_not_worse_pass']} |",
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
            f"- Pass share: {summary.get('pass_share')}",
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
    if strategy.method not in ("vol_target", "risk_parity", "risk_parity_vol_target"):
        raise ValueError("method must be vol_target, risk_parity, or risk_parity_vol_target")
    positive_ints = {
        "vol_window": strategy.vol_window,
        "rebalance_bars": strategy.rebalance_bars,
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
    if not 0.0 <= strategy.min_asset_weight <= strategy.max_asset_weight <= 1.0:
        raise ValueError("asset weight bounds must be between 0 and 1")
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


def _date_value(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _env_int(name: str, default: int) -> int:
    raw = _env_text(name, "")
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env_text(name, "")
    return default if raw is None or raw == "" else float(raw)


def _env_text(name: str, default: str) -> str:
    for prefix in ("OLYMPUS29", "OLYMPUS28"):
        raw = os.getenv(f"{prefix}_{name}")
        if raw is not None and raw != "":
            return raw
    return default


if __name__ == "__main__":
    raise SystemExit(main())
