from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from pure_risk_allocation import (
    DEFAULT_END,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START,
    DEFAULT_TIMEFRAME,
    EvaluationConfig,
    StrategyConfig,
    UniverseItem,
    align_closes,
    data_summary,
    load_frames,
    parse_universe,
    public_strategy_report,
    result_from_daily,
    safety_statement,
    simulate_strategy,
    strategy_result_for_period,
    to_jsonable,
    universe_assessment,
)  # noqa: E402

VERDICT_PASS = "ROBUST_EQUAL_WEIGHT_BETA_CONFIG"
VERDICT_FAIL = "NO_ROBUST_BETA_CONFIG"
VERDICT_DATA = "EQUAL_WEIGHT_ALLOCATION_DATA_INSUFFICIENT"
EDGE_THESIS = (
    "低相关跨资产篮子按1/N等权持有,仅按预声明频率再平衡;不做择时、方向预测或风险加权,"
    "验证分散beta纪律是否比单资产持有更稳健"
)


@dataclass(frozen=True)
class EqualWeightConfig:
    rebalance_bars: int
    fee_bps: float
    slippage_bps: float


DEFAULT_REBALANCE_GRID: tuple[EqualWeightConfig, ...] = (
    EqualWeightConfig(rebalance_bars=21, fee_bps=10.0, slippage_bps=5.0),
    EqualWeightConfig(rebalance_bars=63, fee_bps=10.0, slippage_bps=5.0),
    EqualWeightConfig(rebalance_bars=126, fee_bps=10.0, slippage_bps=5.0),
)

RISK_PARITY_BASELINE = StrategyConfig(
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Olympus #30 equal-weight 1/N beta allocation candidate."
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
        evaluation_config=evaluation_config,
        universe=parse_universe(_env_text("UNIVERSE", "")),
        rebalance_grid=env_rebalance_grid(),
        verbose=True,
    )
    if not args.no_write:
        report["written_files"] = write_report(report, Path(str(args.output_dir)))
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["verdict"] != VERDICT_DATA else 1


def run_evaluation(
    *,
    evaluation_config: EvaluationConfig,
    universe: tuple[UniverseItem, ...],
    rebalance_grid: tuple[EqualWeightConfig, ...],
    verbose: bool = False,
) -> dict[str, Any]:
    validate_config(evaluation_config, rebalance_grid)
    generated_at = datetime.now(UTC)
    universe_items = list(universe)
    frames, load_failures = load_frames(universe_items, evaluation_config, verbose=verbose)
    aligned = align_closes(frames)
    universe_status = universe_assessment(universe_items, frames, load_failures, aligned)
    if universe_status["status"] == "DATA_INSUFFICIENT":
        return insufficient_report(
            generated_at,
            evaluation_config,
            rebalance_grid,
            universe_items,
            frames,
            load_failures,
            str(universe_status["reason"]),
        )
    min_bars = evaluation_config.train_bars + evaluation_config.test_bars
    if aligned.empty or len(aligned) < min_bars:
        return insufficient_report(
            generated_at,
            evaluation_config,
            rebalance_grid,
            universe_items,
            frames,
            load_failures,
            "not enough aligned bars for one walk-forward split",
        )

    full_sample = full_sample_report(aligned, rebalance_grid, evaluation_config)
    walk_forward = run_walk_forward(aligned, rebalance_grid, evaluation_config)
    verdict, reasons = verdict_from_reports(full_sample, walk_forward)
    return {
        "generated_at": generated_at.isoformat(),
        "name": "equal_weight_allocation",
        "status": "OK",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "edge_thesis": EDGE_THESIS,
        "summary": summary_fields(full_sample, walk_forward, verdict),
        "config": {
            "evaluation": asdict(evaluation_config),
            "predeclared_rebalance_grid": [asdict(item) for item in rebalance_grid],
            "only_tunable_parameter": "rebalance_bars",
            "risk_parity_baseline": asdict(RISK_PARITY_BASELINE),
        },
        "universe": [asdict(item) for item in universe_items],
        "universe_assessment": universe_status,
        "data": data_summary(aligned, frames, load_failures),
        "full_sample": full_sample,
        "walk_forward": walk_forward,
        "olympus29_contrast": olympus29_contrast(full_sample, walk_forward),
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only paper research; no trading signal or order path",
    }


def full_sample_report(
    closes: pd.DataFrame,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    single_assets = {
        column: public_strategy_report(simulate_single_asset_buy_hold(closes, column))
        for column in closes.columns
    }
    risk_parity = simulate_strategy(closes, RISK_PARITY_BASELINE)
    results: list[dict[str, Any]] = []
    for config in rebalance_grid:
        candidate = simulate_equal_weight_periodic(closes, config)
        comparisons = benchmark_comparisons(candidate, single_assets, risk_parity)
        results.append(
            {
                "rebalance_bars": config.rebalance_bars,
                "label": rebalance_label(config.rebalance_bars),
                "candidate": public_strategy_report(candidate),
                "benchmarks": {
                    "single_assets": single_assets,
                    "risk_parity": public_strategy_report(risk_parity),
                },
                "comparison": comparisons,
                "passes_all_gates": full_sample_passed(comparisons),
            }
        )
    return {
        "net_costs_included": True,
        "cost_model": "fee_bps + slippage_bps charged on explicit rebalance turnover",
        "single_asset_required_benchmarks": ["binance:BTCUSDT", "yfinance:SPY"],
        "rebalance_sweep": results,
        "summary": rebalance_sweep_summary(results, evaluation_config),
    }


def simulate_equal_weight_periodic(
    closes: pd.DataFrame,
    config: EqualWeightConfig,
) -> dict[str, Any]:
    returns = closes.pct_change().fillna(0.0)
    target = pd.Series(1.0 / len(closes.columns), index=closes.columns)
    weights = pd.Series(0.0, index=closes.columns)
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    equity = 1.0
    rows: list[dict[str, Any]] = []
    for offset, timestamp in enumerate(closes.index):
        if offset == 0 or offset % config.rebalance_bars == 0:
            desired = target
        else:
            desired = weights
        turnover = float((desired - weights).abs().sum())
        cost = turnover * cost_rate
        day_returns = cast(pd.Series, returns.loc[timestamp])
        gross_return = float((desired * day_returns).sum())
        net_return = gross_return - cost
        equity *= 1.0 + net_return
        after_return_weights = desired * (1.0 + day_returns)
        gross_weight_sum = float(after_return_weights.sum())
        weights = (
            cast(pd.Series, after_return_weights / gross_weight_sum)
            if gross_weight_sum > 0
            else target
        )
        rows.append(
            {
                "timestamp": timestamp,
                "return": net_return,
                "gross_return": gross_return,
                "cost": cost,
                "equity": equity,
                "gross_exposure": float(desired.abs().sum()),
                "turnover": turnover,
            }
        )
    daily = pd.DataFrame(rows).set_index("timestamp")
    return result_from_daily("equal_weight_1n_periodic_rebalanced", daily)


def simulate_single_asset_buy_hold(closes: pd.DataFrame, column: str) -> dict[str, Any]:
    returns = closes[column].pct_change().fillna(0.0)
    daily = pd.DataFrame(
        {"return": returns, "equity": (1.0 + returns).cumprod()},
        index=closes.index,
    )
    result = result_from_daily(f"single_asset_buy_hold:{column}", daily)
    return result


def benchmark_comparisons(
    candidate: dict[str, Any],
    single_assets: dict[str, dict[str, Any]],
    risk_parity: dict[str, Any],
) -> dict[str, Any]:
    single = {
        name: comparison_against(candidate, report)
        for name, report in sorted(single_assets.items())
    }
    required = {
        name: single[name]
        for name in ("binance:BTCUSDT", "yfinance:SPY")
        if name in single
    }
    return {
        "candidate": public_strategy_report(candidate),
        "single_assets": single,
        "required_single_assets": required,
        "risk_parity": comparison_against(candidate, public_strategy_report(risk_parity)),
        "net_costs_included": True,
    }


def comparison_against(candidate: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    candidate_metrics = cast(dict[str, float], candidate["metrics"])
    benchmark_metrics = cast(dict[str, float], benchmark["metrics"])
    benchmark_dd = benchmark_metrics["max_drawdown_pct"]
    candidate_dd = candidate_metrics["max_drawdown_pct"]
    drawdown_reduction = (
        (benchmark_dd - candidate_dd) / benchmark_dd if benchmark_dd > 0 else 0.0
    )
    sharpe_delta = candidate_metrics["sharpe"] - benchmark_metrics["sharpe"]
    calmar_delta = candidate_metrics["calmar"] - benchmark_metrics["calmar"]
    return {
        "drawdown_reduction_ratio": round(drawdown_reduction, 6),
        "drawdown_reduction_pass": drawdown_reduction > 0.0,
        "sharpe_delta": round(sharpe_delta, 6),
        "sharpe_not_worse_pass": sharpe_delta >= 0.0,
        "calmar_delta": round(calmar_delta, 6),
        "calmar_not_worse_pass": calmar_delta >= 0.0,
    }


def full_sample_passed(comparisons: dict[str, Any]) -> bool:
    required = cast(dict[str, Any], comparisons["required_single_assets"])
    risk_parity = cast(dict[str, Any], comparisons["risk_parity"])
    candidate = cast(dict[str, Any], comparisons["candidate"])
    return bool(
        required
        and all(benchmark_passed(item) for item in required.values())
        and risk_adjusted_not_worse(risk_parity)
        and float(candidate["metrics"]["total_return_pct"]) > 0.0
    )


def run_walk_forward(
    closes: pd.DataFrame,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
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
                "parameter_trials": len(rebalance_grid),
                "pass_count": 0,
                "pass_share": 0.0,
                "oos_stable": False,
                "reason": "not enough bars for one train->test walk-forward split",
            },
        }
    windows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        test = closes.iloc[
            start + evaluation_config.train_bars : start
            + evaluation_config.train_bars
            + evaluation_config.test_bars
        ]
        warm = closes.iloc[
            start : start + evaluation_config.train_bars + evaluation_config.test_bars
        ]
        risk_parity = strategy_result_for_period(warm, test.index, RISK_PARITY_BASELINE)
        single_assets = {
            column: public_strategy_report(simulate_single_asset_buy_hold(test, column))
            for column in test.columns
        }
        frequency_results: list[dict[str, Any]] = []
        for config in rebalance_grid:
            candidate = simulate_equal_weight_periodic(test, config)
            comparisons = benchmark_comparisons(candidate, single_assets, risk_parity)
            frequency_results.append(
                {
                    "rebalance_bars": config.rebalance_bars,
                    "label": rebalance_label(config.rebalance_bars),
                    "candidate": public_strategy_report(candidate),
                    "comparison": comparisons,
                    "passes_all_gates": full_sample_passed(comparisons),
                }
            )
        windows.append(
            {
                "index": index,
                "test_period": period(test),
                "frequency_results": frequency_results,
            }
        )
    return {
        "status": "OK",
        "windows": windows,
        "summary": walk_forward_summary(windows, len(rebalance_grid)),
        "selection_objective": "none; rebalance frequencies are predeclared and all reported",
    }


def walk_forward_summary(
    windows: list[dict[str, Any]],
    frequency_count: int,
) -> dict[str, Any]:
    frequency_rows = [
        item
        for window in windows
        for item in cast(list[dict[str, Any]], window["frequency_results"])
    ]
    pass_count = sum(1 for item in frequency_rows if item["passes_all_gates"])
    pass_share = pass_count / len(frequency_rows) if frequency_rows else 0.0
    by_frequency: list[dict[str, Any]] = []
    for rebalance_bars in sorted({int(item["rebalance_bars"]) for item in frequency_rows}):
        rows = [item for item in frequency_rows if item["rebalance_bars"] == rebalance_bars]
        by_frequency.append(
            {
                "rebalance_bars": rebalance_bars,
                "label": rebalance_label(rebalance_bars),
                "windows": len(rows),
                "pass_count": sum(1 for item in rows if item["passes_all_gates"]),
                "pass_share": round(
                    sum(1 for item in rows if item["passes_all_gates"]) / len(rows), 6
                )
                if rows
                else 0.0,
                "required_single_assets": median_required_single_asset_deltas(rows),
                "risk_parity": median_benchmark_deltas(rows, "risk_parity"),
            }
        )
    stable = bool(windows and pass_share >= 0.60)
    reasons: list[str] = []
    if pass_share < 0.60:
        reasons.append("fewer than 60% of OOS frequency-window trials passed all gates")
    for frequency in by_frequency:
        if float(frequency["pass_share"]) < 0.60:
            reasons.append(
                f"{frequency['label']} OOS pass_share below 0.60: {frequency['pass_share']}"
            )
    return {
        "windows": len(windows),
        "parameter_trials": frequency_count,
        "frequency_window_trials": len(frequency_rows),
        "pass_count": pass_count,
        "pass_share": round(pass_share, 6),
        "by_frequency": by_frequency,
        "oos_stable": stable,
        "reason": "; ".join(reasons) if reasons else "OOS pass_share met the 0.60 gate",
    }


def rebalance_sweep_summary(
    results: list[dict[str, Any]],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    pass_count = sum(1 for item in results if item["passes_all_gates"])
    pass_share = pass_count / len(results) if results else 0.0
    return {
        "frequency_count": len(results),
        "pass_count": pass_count,
        "pass_share": round(pass_share, 6),
        "required_oos_pass_share": 0.60,
        "min_drawdown_gate": "positive reduction versus BTC and SPY single-asset buy&hold",
        "risk_adjusted_gate": "Sharpe and Calmar not worse than BTC, SPY, and #29 risk parity",
        "net_costs_included": True,
        "evaluation_start": evaluation_config.start,
        "evaluation_end": evaluation_config.end,
    }


def median_required_single_asset_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            name
            for row in rows
            for name in cast(
                dict[str, Any],
                cast(dict[str, Any], row["comparison"])["required_single_assets"],
            )
        }
    )
    return {name: median_benchmark_deltas(rows, ("required_single_assets", name)) for name in names}


def median_benchmark_deltas(
    rows: list[dict[str, Any]],
    benchmark_path: str | tuple[str, str],
) -> dict[str, float]:
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        comparison = cast(dict[str, Any], row["comparison"])
        if isinstance(benchmark_path, tuple):
            group, name = benchmark_path
            comparisons.append(cast(dict[str, Any], cast(dict[str, Any], comparison[group])[name]))
        else:
            comparisons.append(cast(dict[str, Any], comparison[benchmark_path]))
    return {
        "median_drawdown_reduction_ratio": median_of(
            comparisons, "drawdown_reduction_ratio"
        ),
        "median_sharpe_delta": median_of(comparisons, "sharpe_delta"),
        "median_calmar_delta": median_of(comparisons, "calmar_delta"),
    }


def median_of(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(statistics.median(values), 6) if values else 0.0


def benchmark_passed(comparison: dict[str, Any]) -> bool:
    return bool(
        comparison["drawdown_reduction_pass"]
        and comparison["sharpe_not_worse_pass"]
        and comparison["calmar_not_worse_pass"]
    )


def risk_adjusted_not_worse(comparison: dict[str, Any]) -> bool:
    return bool(
        comparison["sharpe_not_worse_pass"] and comparison["calmar_not_worse_pass"]
    )


def verdict_from_reports(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    full_summary = cast(dict[str, Any], full_sample["summary"])
    wf_summary = cast(dict[str, Any], walk_forward.get("summary", {}))
    if float(full_summary["pass_share"]) < 1.0:
        reasons.append(
            f"full-sample rebalance sweep pass_share below 1.00: {full_summary['pass_share']}"
        )
    if walk_forward.get("status") != "OK":
        reasons.append("walk-forward data insufficient")
    elif wf_summary.get("oos_stable") is not True:
        reasons.append(f"walk-forward OOS not stable: {wf_summary.get('reason')}")
    if reasons:
        return VERDICT_FAIL, reasons
    return VERDICT_PASS, [
        "all predeclared rebalance frequencies passed full-sample gates and OOS pass_share >= 0.60"
    ]


def summary_fields(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "full_sample": cast(dict[str, Any], full_sample["summary"]),
        "wf": cast(dict[str, Any], walk_forward.get("summary", {})),
        "cost": {
            "net_costs_included": True,
            "cost_model": full_sample["cost_model"],
        },
        "rebalance_sweep": [
            {
                "rebalance_bars": item["rebalance_bars"],
                "label": item["label"],
                "passes_all_gates": item["passes_all_gates"],
                "candidate_metrics": cast(dict[str, Any], item["candidate"])["metrics"],
                "total_cost_pct": cast(dict[str, Any], item["candidate"])["total_cost_pct"],
            }
            for item in cast(list[dict[str, Any]], full_sample["rebalance_sweep"])
        ],
    }


def olympus29_contrast(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    return {
        "olympus29_verdict": "NO_ROBUST_DIVERSIFIED_RISK_ALLOCATION_EDGE",
        "olympus29_decisive_intel": (
            "risk parity beat naive buy&hold on drawdown but lost to equal-weight 1/N; "
            "diversification, not weighting complexity, was the useful signal"
        ),
        "equal_weight_full_sample_pass_share": cast(dict[str, Any], full_sample["summary"])[
            "pass_share"
        ],
        "equal_weight_oos_pass_share": cast(dict[str, Any], walk_forward.get("summary", {})).get(
            "pass_share"
        ),
        "interpretation": (
            "Olympus #30 treats 1/N as the candidate and checks whether the simple beta "
            "discipline itself is robust after costs and across predeclared rebalance frequencies."
        ),
    }


def insufficient_report(
    generated_at: datetime,
    evaluation_config: EvaluationConfig,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    universe_items: list[UniverseItem],
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
    reason: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "name": "equal_weight_allocation",
        "status": "INSUFFICIENT_DATA",
        "verdict": VERDICT_DATA,
        "verdict_reasons": [reason],
        "edge_thesis": EDGE_THESIS,
        "summary": {
            "verdict": VERDICT_DATA,
            "full_sample": None,
            "wf": {"oos_stable": False, "reason": reason},
            "cost": {"net_costs_included": True},
            "rebalance_sweep": [],
        },
        "config": {
            "evaluation": asdict(evaluation_config),
            "predeclared_rebalance_grid": [asdict(item) for item in rebalance_grid],
            "only_tunable_parameter": "rebalance_bars",
            "risk_parity_baseline": asdict(RISK_PARITY_BASELINE),
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
        "olympus29_contrast": {"status": "not_comparable_without_data"},
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only paper research; no trading signal or order path",
    }


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"olympus30-equal-weight-allocation-{stamp}"
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
    summary = cast(dict[str, Any], report["summary"])
    wf = cast(dict[str, Any], summary["wf"])
    lines = [
        "# Olympus #30 Equal Weight 1/N Allocation",
        "",
        f"- Generated: {report['generated_at']}",
        f"- JSON evidence: `{json_path}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Thesis: {report['edge_thesis']}",
        "- Discipline: candidate-only, paper research, no timing, no risk weighting, "
        "no order path.",
        "",
        "## Summary",
        "",
        "- Full-sample sweep pass_share: "
        f"{cast(dict[str, Any], summary['full_sample'])['pass_share']}",
        f"- Walk-forward OOS pass_share: {wf.get('pass_share')}",
        f"- OOS stable: {wf.get('oos_stable')}",
        f"- Reason: {wf.get('reason')}",
        "",
        "## Rebalance Sweep",
        "",
        "| frequency | rebalance_bars | max_dd_pct | sharpe | calmar | total_cost_pct | pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in cast(list[dict[str, Any]], summary["rebalance_sweep"]):
        metrics = cast(dict[str, Any], item["candidate_metrics"])
        lines.append(
            "| "
            f"{item['label']} | {item['rebalance_bars']} | "
            f"{metrics['max_drawdown_pct']} | {metrics['sharpe']} | {metrics['calmar']} | "
            f"{item['total_cost_pct']} | {item['passes_all_gates']} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- candidate-only research; no paper/live setting changed.",
            "- no strategy plugin, live order code, API key, or public bind added.",
        ]
    )
    return "\n".join(lines) + "\n"


def window_starts(total_bars: int, train_bars: int, test_bars: int, step_bars: int) -> list[int]:
    starts: list[int] = []
    start = 0
    while start + train_bars + test_bars <= total_bars:
        starts.append(start)
        start += step_bars
    return starts


def period(frame: pd.DataFrame) -> dict[str, str | int]:
    return {
        "start": str(frame.index[0].date()) if len(frame) else "",
        "end": str(frame.index[-1].date()) if len(frame) else "",
        "bars": len(frame),
    }


def rebalance_label(rebalance_bars: int) -> str:
    labels = {21: "monthly", 63: "quarterly", 126: "semiannual"}
    return labels.get(rebalance_bars, f"{rebalance_bars}_bars")


def env_evaluation_config(*, start: str, end: str, timeframe: str) -> EvaluationConfig:
    return EvaluationConfig(
        start=start,
        end=end,
        timeframe=timeframe,
        train_bars=_env_int("TRAIN_BARS", DEFAULT_EVALUATION_CONFIG.train_bars),
        test_bars=_env_int("TEST_BARS", DEFAULT_EVALUATION_CONFIG.test_bars),
        step_bars=_env_int("STEP_BARS", DEFAULT_EVALUATION_CONFIG.step_bars),
        min_drawdown_reduction_ratio=0.0,
        max_sharpe_shortfall=0.0,
        max_calmar_shortfall=0.0,
        max_parameter_trials=len(DEFAULT_REBALANCE_GRID),
    )


def env_rebalance_grid() -> tuple[EqualWeightConfig, ...]:
    raw = _env_text("REBALANCE_GRID", "")
    if not raw.strip():
        return DEFAULT_REBALANCE_GRID
    bars = [_parse_positive_int(chunk, "REBALANCE_GRID") for chunk in raw.split(",")]
    fee_bps = _env_float("FEE_BPS", DEFAULT_REBALANCE_GRID[0].fee_bps)
    slippage_bps = _env_float("SLIPPAGE_BPS", DEFAULT_REBALANCE_GRID[0].slippage_bps)
    return tuple(
        EqualWeightConfig(
            rebalance_bars=rebalance_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        for rebalance_bars in bars
    )


def validate_config(
    evaluation_config: EvaluationConfig,
    rebalance_grid: tuple[EqualWeightConfig, ...],
) -> None:
    if not rebalance_grid:
        raise ValueError("at least one predeclared rebalance frequency is required")
    if len(rebalance_grid) > evaluation_config.max_parameter_trials:
        raise ValueError("rebalance grid exceeds max_parameter_trials")
    if evaluation_config.train_bars <= 0 or evaluation_config.test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    for config in rebalance_grid:
        if config.rebalance_bars <= 0:
            raise ValueError("rebalance_bars must be positive")
        if not math.isfinite(config.fee_bps) or not math.isfinite(config.slippage_bps):
            raise ValueError("fee_bps and slippage_bps must be finite")


def _env_text(name: str, default: str) -> str:
    for candidate in (f"OLYMPUS30_{name}", name):
        raw = os.environ.get(candidate)
        if raw not in (None, ""):
            return raw
    return default


def _env_int(name: str, default: int) -> int:
    return _parse_positive_int(_env_text(name, str(default)), name)


def _env_float(name: str, default: float) -> float:
    raw = _env_text(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
