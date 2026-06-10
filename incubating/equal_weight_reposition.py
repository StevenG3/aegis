from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from equal_weight_allocation import (
    DEFAULT_END,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REBALANCE_GRID,
    DEFAULT_START,
    DEFAULT_TIMEFRAME,
    RISK_PARITY_BASELINE,
    EqualWeightConfig,
    benchmark_comparisons,
    env_evaluation_config,
    full_sample_passed,
    load_frames,
    parse_universe,
    period,
    public_strategy_report,
    rebalance_label,
    result_from_daily,
    simulate_equal_weight_periodic,
    simulate_single_asset_buy_hold,
    simulate_strategy,
    to_jsonable,
    universe_assessment,
    validate_config,
    window_starts,
)
from equal_weight_allocation import (
    EDGE_THESIS as OLYMPUS30_THESIS,
)
from equal_weight_allocation import (
    run_walk_forward as olympus30_walk_forward,
)
from pure_risk_allocation import (
    EvaluationConfig,
    UniverseItem,
    align_closes,
    data_summary,
    safety_statement,
)

VERDICT_ROBUST = "ROBUST_REPOSITION_VS_STATUSQUO"
VERDICT_TRIVIAL = "REPOSITION_TRIVIAL_DERISK_ONLY"
VERDICT_FAIL = "NO_ROBUST_REPOSITION"
VERDICT_DATA = "REPOSITION_DATA_INSUFFICIENT"
DEFAULT_STATUS_QUO_ASSET = "binance:BTCUSDT"
REQUIRED_OOS_PASS_SHARE = 0.60
EDGE_THESIS = (
    "用决策相关基准重评等权1/N: 与用户原本可能集中持有的status-quo相比是否少亏且"
    "风险调整不差,并用BTC+cash同波动基准检查多资产分散是否真有增量"
)
FUNDING_BORROW_NA = {
    "applicability": "N/A",
    "funding_borrow_cost_pct": 0.0,
    "reason": (
        "spot long-only paper basket/status-quo; no perp, leverage, short, borrow, or funding "
        "exposure"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Olympus #31 equal-weight repositioning versus status quo."
    )
    parser.add_argument("--start", default=_env_text("START", DEFAULT_START))
    parser.add_argument("--end", default=_env_text("END", DEFAULT_END))
    parser.add_argument("--timeframe", default=_env_text("TIMEFRAME", DEFAULT_TIMEFRAME))
    parser.add_argument("--output-dir", default=_env_text("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument(
        "--status-quo-asset",
        default=_env_text("STATUS_QUO_ASSET", DEFAULT_STATUS_QUO_ASSET),
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
        rebalance_grid=DEFAULT_REBALANCE_GRID,
        status_quo_asset=str(args.status_quo_asset),
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
    status_quo_asset: str = DEFAULT_STATUS_QUO_ASSET,
    verbose: bool = False,
) -> dict[str, Any]:
    validate_config(evaluation_config, rebalance_grid)
    generated_at = datetime.now(UTC)
    universe_items = list(universe)
    frames, load_failures = load_frames(universe_items, evaluation_config, verbose=verbose)
    aligned = align_closes(frames)
    universe_status = universe_assessment(universe_items, frames, load_failures, aligned)
    if universe_status["status"] == "DATA_INSUFFICIENT" or status_quo_asset not in aligned:
        reason = (
            "status-quo asset unavailable"
            if status_quo_asset not in aligned
            else str(universe_status["reason"])
        )
        return insufficient_report(
            generated_at,
            evaluation_config,
            rebalance_grid,
            universe_items,
            frames,
            load_failures,
            status_quo_asset,
            reason,
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
            status_quo_asset,
            "not enough aligned bars for one walk-forward split",
        )

    full_sample = full_sample_report(aligned, rebalance_grid, evaluation_config, status_quo_asset)
    walk_forward = walk_forward_report(aligned, rebalance_grid, evaluation_config, status_quo_asset)
    verdict, reasons = verdict_from_reports(full_sample, walk_forward)
    compact_failed_gates = compact_failed_gate_matrix(full_sample, walk_forward)
    return {
        "generated_at": generated_at.isoformat(),
        "name": "equal_weight_reposition_vs_statusquo",
        "status": "OK",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "compact_failed_gates": compact_failed_gates,
        "edge_thesis": EDGE_THESIS,
        "benchmark_revision": benchmark_revision(status_quo_asset),
        "config": {
            "evaluation": asdict(evaluation_config),
            "predeclared_rebalance_grid": [asdict(item) for item in rebalance_grid],
            "only_tunable_parameter": "rebalance_bars",
            "status_quo_asset": status_quo_asset,
            "integrity_benchmark": (
                "status_quo_asset + cash scaled to candidate annualized volatility"
            ),
            "full_cost_policy": {
                "fee_bps": [item.fee_bps for item in rebalance_grid],
                "slippage_bps": [item.slippage_bps for item in rebalance_grid],
                "funding_borrow": FUNDING_BORROW_NA,
                "net_cost_required": True,
            },
        },
        "universe": [asdict(item) for item in universe_items],
        "universe_assessment": universe_status,
        "data": data_summary(aligned, frames, load_failures),
        "full_sample": full_sample,
        "walk_forward": walk_forward,
        "olympus30_old_gate": {
            "verdict": "NO_ROBUST_BETA_CONFIG",
            "thesis": OLYMPUS30_THESIS,
            "kept_for_integrity": True,
            "summary": full_sample["old_olympus30_gate_summary"],
        },
        "what_would_change_next_research_question": what_would_change_next_question(verdict),
        "safety": safety_statement(),
        "disclaimer": "incubating candidate-only paper research; no trading signal or order path",
    }


def full_sample_report(
    closes: pd.DataFrame,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    evaluation_config: EvaluationConfig,
    status_quo_asset: str,
) -> dict[str, Any]:
    single_assets = {
        column: public_strategy_report(simulate_single_asset_buy_hold(closes, column))
        for column in closes.columns
    }
    status_quo = simulate_single_asset_buy_hold(closes, status_quo_asset)
    risk_parity = simulate_strategy(closes, RISK_PARITY_BASELINE)
    results: list[dict[str, Any]] = []
    for config in rebalance_grid:
        candidate = simulate_equal_weight_periodic(closes, config)
        btc_cash = simulate_same_vol_status_quo_cash(closes, status_quo_asset, candidate)
        status_comparison = comparison_against(candidate, public_strategy_report(status_quo))
        integrity_comparison = comparison_against(candidate, public_strategy_report(btc_cash))
        old_comparisons = benchmark_comparisons(candidate, single_assets, risk_parity)
        results.append(
            {
                "rebalance_bars": config.rebalance_bars,
                "label": rebalance_label(config.rebalance_bars),
                "candidate": report_with_standard_metrics(candidate),
                "benchmarks": {
                    "status_quo": report_with_standard_metrics(status_quo),
                    "same_vol_status_quo_cash": report_with_standard_metrics(btc_cash),
                },
                "status_quo_comparison": status_comparison,
                "same_vol_cash_integrity_check": integrity_comparison,
                "passes_reposition_gate": reposition_passed(status_comparison, candidate),
                "passes_same_vol_cash_integrity_check": integrity_passed(integrity_comparison),
                "old_olympus30_gate": {
                    "comparison": old_comparisons,
                    "passes_all_gates": full_sample_passed(old_comparisons),
                },
            }
        )
    return {
        "status_quo_asset": status_quo_asset,
        "status_quo_rationale": benchmark_revision(status_quo_asset)["status_quo_rationale"],
        "net_costs_included": True,
        "cost_model": "fee_bps + slippage_bps charged on explicit equal-weight rebalance turnover",
        "rebalance_sweep": results,
        "summary": sweep_summary(results, evaluation_config),
        "old_olympus30_gate_summary": old_gate_summary(results),
    }


def walk_forward_report(
    closes: pd.DataFrame,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    evaluation_config: EvaluationConfig,
    status_quo_asset: str,
) -> dict[str, Any]:
    starts = window_starts(
        len(closes),
        evaluation_config.train_bars,
        evaluation_config.test_bars,
        evaluation_config.step_bars,
    )
    windows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        test = closes.iloc[
            start + evaluation_config.train_bars : start
            + evaluation_config.train_bars
            + evaluation_config.test_bars
        ]
        status_quo = simulate_single_asset_buy_hold(test, status_quo_asset)
        frequency_results: list[dict[str, Any]] = []
        for config in rebalance_grid:
            candidate = simulate_equal_weight_periodic(test, config)
            btc_cash = simulate_same_vol_status_quo_cash(test, status_quo_asset, candidate)
            status_comparison = comparison_against(candidate, public_strategy_report(status_quo))
            integrity_comparison = comparison_against(candidate, public_strategy_report(btc_cash))
            frequency_results.append(
                {
                    "rebalance_bars": config.rebalance_bars,
                    "label": rebalance_label(config.rebalance_bars),
                    "candidate": report_with_standard_metrics(candidate),
                    "status_quo_comparison": status_comparison,
                    "same_vol_cash_integrity_check": integrity_comparison,
                    "passes_reposition_gate": reposition_passed(status_comparison, candidate),
                    "passes_same_vol_cash_integrity_check": integrity_passed(integrity_comparison),
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
        "status": "OK" if windows else "INSUFFICIENT_DATA",
        "windows": windows,
        "summary": walk_forward_summary(windows, len(rebalance_grid)),
        "old_olympus30_walk_forward_summary": olympus30_walk_forward(
            closes, rebalance_grid, evaluation_config
        )["summary"],
    }


def simulate_same_vol_status_quo_cash(
    closes: pd.DataFrame,
    status_quo_asset: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    btc_returns = closes[status_quo_asset].pct_change().fillna(0.0)
    btc_vol = float(btc_returns.std() * (252**0.5) * 100.0)
    candidate_metrics = cast(dict[str, float], candidate["metrics"])
    candidate_vol = float(candidate_metrics["ann_vol_pct"])
    weight = min(candidate_vol / btc_vol, 1.0) if btc_vol > 0 else 0.0
    daily = pd.DataFrame(
        {
            "return": btc_returns * weight,
            "cost": 0.0,
            "gross_exposure": weight,
            "turnover": 0.0,
            "equity": (1.0 + btc_returns * weight).cumprod(),
        },
        index=closes.index,
    )
    result = result_from_daily("status_quo_plus_cash_same_vol", daily)
    result["status_quo_asset"] = status_quo_asset
    result["status_quo_weight"] = round(weight, 6)
    result["target_ann_vol_pct"] = round(candidate_vol, 6)
    result["source_ann_vol_pct"] = round(btc_vol, 6)
    return result


def report_with_standard_metrics(
    result: dict[str, Any],
    *,
    oos_window_win_rate_vs_status_quo_pct: float | None = None,
) -> dict[str, Any]:
    report = public_strategy_report(result)
    report["standard_metrics"] = standard_metrics_block(
        result,
        oos_window_win_rate_vs_status_quo_pct=oos_window_win_rate_vs_status_quo_pct,
    )
    return report


def standard_metrics_block(
    result: dict[str, Any],
    *,
    oos_window_win_rate_vs_status_quo_pct: float | None = None,
) -> dict[str, Any]:
    metrics = cast(dict[str, float], result["metrics"])
    daily = cast(pd.DataFrame, result.get("daily", pd.DataFrame()))
    returns = (
        pd.to_numeric(cast(pd.Series, daily["return"]), errors="coerce").dropna()
        if "return" in daily
        else pd.Series(dtype=float)
    )
    ann_return_pct = float(metrics.get("ann_return_pct", 0.0))
    downside = returns[returns < 0]
    downside_vol = float(downside.std()) * (252**0.5) if not downside.empty else 0.0
    sortino = (ann_return_pct / 100.0) / downside_vol if downside_vol > 0 else 0.0
    period_positive_rate = float((returns > 0).mean()) * 100.0 if not returns.empty else 0.0
    turnover = cast(dict[str, float] | None, result.get("turnover"))
    daily_turnover = float(turnover.get("daily_average", 0.0)) if turnover else 0.0
    annualized_turnover = daily_turnover * 252.0
    return {
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "sortino": round(sortino, 6),
        "calmar": metrics.get("calmar", 0.0),
        "positive_period_rate_pct": round(period_positive_rate, 6),
        "oos_window_win_rate_vs_status_quo_pct": oos_window_win_rate_vs_status_quo_pct,
        "annualized_turnover": round(annualized_turnover, 6),
        "net_cost_pct": result.get("total_cost_pct", 0.0),
        "funding_borrow": FUNDING_BORROW_NA,
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


def reposition_passed(comparison: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        comparison["drawdown_reduction_pass"]
        and comparison["sharpe_not_worse_pass"]
        and comparison["calmar_not_worse_pass"]
        and float(candidate["metrics"]["total_return_pct"]) > 0.0
    )


def integrity_passed(comparison: dict[str, Any]) -> bool:
    return bool(comparison["sharpe_not_worse_pass"] and comparison["calmar_not_worse_pass"])


def sweep_summary(
    results: list[dict[str, Any]],
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    reposition_pass_count = sum(1 for item in results if item["passes_reposition_gate"])
    integrity_pass_count = sum(
        1 for item in results if item["passes_same_vol_cash_integrity_check"]
    )
    return {
        "frequency_count": len(results),
        "reposition_pass_count": reposition_pass_count,
        "reposition_pass_share": ratio(reposition_pass_count, len(results)),
        "same_vol_cash_integrity_pass_count": integrity_pass_count,
        "same_vol_cash_integrity_pass_share": ratio(integrity_pass_count, len(results)),
        "reposition_all_frequencies_pass": reposition_pass_count == len(results),
        "same_vol_cash_integrity_all_frequencies_pass": integrity_pass_count == len(results),
        "evaluation_start": evaluation_config.start,
        "evaluation_end": evaluation_config.end,
    }


def old_gate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for item in results if item["old_olympus30_gate"]["passes_all_gates"])
    return {
        "old_gate_pass_count": pass_count,
        "old_gate_pass_share": ratio(pass_count, len(results)),
        "old_gate_verdict": (
            "NO_ROBUST_BETA_CONFIG"
            if pass_count < len(results)
            else "ROBUST_EQUAL_WEIGHT_BETA_CONFIG"
        ),
        "kept_for_integrity": True,
    }


def walk_forward_summary(windows: list[dict[str, Any]], frequency_count: int) -> dict[str, Any]:
    rows = [
        item
        for window in windows
        for item in cast(list[dict[str, Any]], window["frequency_results"])
    ]
    reposition_pass_count = sum(1 for item in rows if item["passes_reposition_gate"])
    integrity_pass_count = sum(1 for item in rows if item["passes_same_vol_cash_integrity_check"])
    by_frequency = []
    for rebalance_bars in sorted({int(item["rebalance_bars"]) for item in rows}):
        frequency_rows = [item for item in rows if int(item["rebalance_bars"]) == rebalance_bars]
        by_frequency.append(
            {
                "rebalance_bars": rebalance_bars,
                "label": rebalance_label(rebalance_bars),
                "windows": len(frequency_rows),
                "reposition_pass_count": sum(
                    1 for item in frequency_rows if item["passes_reposition_gate"]
                ),
                "reposition_pass_share": ratio(
                    sum(1 for item in frequency_rows if item["passes_reposition_gate"]),
                    len(frequency_rows),
                ),
                "oos_window_win_rate_vs_status_quo_pct": ratio(
                    sum(1 for item in frequency_rows if item["passes_reposition_gate"]),
                    len(frequency_rows),
                )
                * 100.0,
                "same_vol_cash_integrity_pass_count": sum(
                    1
                    for item in frequency_rows
                    if item["passes_same_vol_cash_integrity_check"]
                ),
                "same_vol_cash_integrity_pass_share": ratio(
                    sum(
                        1
                        for item in frequency_rows
                        if item["passes_same_vol_cash_integrity_check"]
                    ),
                    len(frequency_rows),
                ),
                "status_quo_median_deltas": median_deltas(frequency_rows, "status_quo_comparison"),
                "same_vol_cash_median_deltas": median_deltas(
                    frequency_rows, "same_vol_cash_integrity_check"
                ),
            }
        )
    reposition_stable = bool(
        rows
        and ratio(reposition_pass_count, len(rows)) >= REQUIRED_OOS_PASS_SHARE
        and all(item["reposition_pass_share"] >= REQUIRED_OOS_PASS_SHARE for item in by_frequency)
    )
    integrity_stable = bool(
        rows
        and ratio(integrity_pass_count, len(rows)) >= REQUIRED_OOS_PASS_SHARE
        and all(
            item["same_vol_cash_integrity_pass_share"] >= REQUIRED_OOS_PASS_SHARE
            for item in by_frequency
        )
    )
    return {
        "windows": len(windows),
        "parameter_trials": frequency_count,
        "frequency_window_trials": len(rows),
        "reposition_pass_count": reposition_pass_count,
        "reposition_pass_share": ratio(reposition_pass_count, len(rows)),
        "oos_window_win_rate_vs_status_quo_pct": ratio(
            reposition_pass_count, len(rows)
        )
        * 100.0,
        "same_vol_cash_integrity_pass_count": integrity_pass_count,
        "same_vol_cash_integrity_pass_share": ratio(integrity_pass_count, len(rows)),
        "by_frequency": by_frequency,
        "reposition_oos_stable": reposition_stable,
        "same_vol_cash_integrity_oos_stable": integrity_stable,
        "required_oos_pass_share": REQUIRED_OOS_PASS_SHARE,
    }


def median_deltas(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    comparisons = [cast(dict[str, Any], item[key]) for item in rows]
    return {
        "median_drawdown_reduction_ratio": median_of(comparisons, "drawdown_reduction_ratio"),
        "median_sharpe_delta": median_of(comparisons, "sharpe_delta"),
        "median_calmar_delta": median_of(comparisons, "calmar_delta"),
    }


def median_of(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(statistics.median(values), 6) if values else 0.0


def verdict_from_reports(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
) -> tuple[str, list[str]]:
    full_summary = cast(dict[str, Any], full_sample["summary"])
    wf_summary = cast(dict[str, Any], walk_forward["summary"])
    reposition_ok = bool(
        full_summary["reposition_all_frequencies_pass"]
        and wf_summary["reposition_oos_stable"]
    )
    integrity_ok = bool(
        full_summary["same_vol_cash_integrity_all_frequencies_pass"]
        and wf_summary["same_vol_cash_integrity_oos_stable"]
    )
    reasons: list[str] = []
    if not reposition_ok:
        reasons.append(
            "reposition gate failed versus status-quo: "
            f"full={full_summary['reposition_pass_share']}, "
            f"oos={wf_summary['reposition_pass_share']}"
        )
        return VERDICT_FAIL, reasons
    if not integrity_ok:
        reasons.append(
            "status-quo gate passed, but same-vol BTC+cash integrity check failed: "
            f"full={full_summary['same_vol_cash_integrity_pass_share']}, "
            f"oos={wf_summary['same_vol_cash_integrity_pass_share']}"
        )
        return VERDICT_TRIVIAL, reasons
    return VERDICT_ROBUST, [
        "all frequencies passed status-quo reposition gates and same-vol BTC+cash integrity checks"
    ]


def compact_failed_gate_matrix(
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for item in cast(list[dict[str, Any]], full_sample["rebalance_sweep"]):
        rows.append(
            {
                "scope": "full_sample",
                "frequency": item["label"],
                "rebalance_bars": item["rebalance_bars"],
                "status_quo": failed_flags(item["status_quo_comparison"]),
                "same_vol_cash": failed_flags(item["same_vol_cash_integrity_check"]),
                "old_olympus30_gate_pass": item["old_olympus30_gate"]["passes_all_gates"],
            }
        )
    wf_summary = cast(dict[str, Any], walk_forward["summary"])
    for item in cast(list[dict[str, Any]], wf_summary["by_frequency"]):
        rows.append(
            {
                "scope": "walk_forward",
                "frequency": item["label"],
                "rebalance_bars": item["rebalance_bars"],
                "status_quo_reposition_pass_share": item["reposition_pass_share"],
                "same_vol_cash_integrity_pass_share": item[
                    "same_vol_cash_integrity_pass_share"
                ],
                "required_pass_share": REQUIRED_OOS_PASS_SHARE,
            }
        )
    return {
        "rows": rows,
        "summary": {
            "full_reposition_pass_share": full_sample["summary"]["reposition_pass_share"],
            "full_same_vol_cash_integrity_pass_share": full_sample["summary"][
                "same_vol_cash_integrity_pass_share"
            ],
            "oos_reposition_pass_share": wf_summary["reposition_pass_share"],
            "oos_same_vol_cash_integrity_pass_share": wf_summary[
                "same_vol_cash_integrity_pass_share"
            ],
            "old_olympus30_full_pass_share": full_sample["old_olympus30_gate_summary"][
                "old_gate_pass_share"
            ],
            "old_olympus30_oos_pass_share": walk_forward[
                "old_olympus30_walk_forward_summary"
            ]["pass_share"],
        },
    }


def failed_flags(comparison: dict[str, Any]) -> list[str]:
    flags = []
    if not comparison["drawdown_reduction_pass"]:
        flags.append("drawdown_reduction")
    if not comparison["sharpe_not_worse_pass"]:
        flags.append("sharpe_not_worse")
    if not comparison["calmar_not_worse_pass"]:
        flags.append("calmar_not_worse")
    return flags


def benchmark_revision(status_quo_asset: str) -> dict[str, str]:
    return {
        "status_quo_asset": status_quo_asset,
        "status_quo_rationale": (
            "The repositioning decision is whether replacing a concentrated high-volatility "
            "holding with a 1/N diversified basket improves drawdown and risk-adjusted returns. "
            "Therefore the primary benchmark is the concentrated status-quo asset, not the safest "
            "single leg of the basket. The old #30 gate remains reported for integrity."
        ),
        "not_a_gate_relaxation": (
            "This run predeclares the benchmark correction, keeps the old gate side-by-side, "
            "does not change the universe, rebalance grid, sample, or windows, "
            "and accepts any result."
        ),
    }


def what_would_change_next_question(verdict: str) -> str:
    if verdict == VERDICT_ROBUST:
        return (
            "Next question becomes portfolio-sizing and user approval: whether a paper-only "
            "candidate should move toward a non-live model portfolio specification."
        )
    if verdict == VERDICT_TRIVIAL:
        return (
            "Next question becomes whether simple volatility scaling or cash allocation is enough, "
            "because multi-asset diversification did not beat same-vol status-quo+cash."
        )
    return (
        "Next question becomes whether the repositioning route should be closed or whether the "
        "status-quo benchmark was misdeclared by the user, not whether gates should be relaxed."
    )


def insufficient_report(
    generated_at: datetime,
    evaluation_config: EvaluationConfig,
    rebalance_grid: tuple[EqualWeightConfig, ...],
    universe_items: list[UniverseItem],
    frames: dict[str, pd.DataFrame],
    load_failures: list[dict[str, str]],
    status_quo_asset: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "name": "equal_weight_reposition_vs_statusquo",
        "status": "INSUFFICIENT_DATA",
        "verdict": VERDICT_DATA,
        "verdict_reasons": [reason],
        "compact_failed_gates": {"summary": {"reason": reason}, "rows": []},
        "edge_thesis": EDGE_THESIS,
        "benchmark_revision": benchmark_revision(status_quo_asset),
        "config": {
            "evaluation": asdict(evaluation_config),
            "predeclared_rebalance_grid": [asdict(item) for item in rebalance_grid],
            "status_quo_asset": status_quo_asset,
        },
        "universe": [asdict(item) for item in universe_items],
        "data": data_summary(pd.DataFrame(), frames, load_failures),
        "safety": safety_statement(),
    }


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"olympus31-reposition-vs-statusquo-{stamp}"
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
    full = cast(dict[str, Any], report["full_sample"])
    wf = cast(dict[str, Any], report["walk_forward"])
    lines = [
        "# Olympus #31 Reposition Vs Status Quo",
        "",
        f"- Generated: {report['generated_at']}",
        f"- JSON evidence: `{json_path}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Status-quo asset: `{report['config']['status_quo_asset']}`",
        f"- Full reposition pass_share: {full['summary']['reposition_pass_share']}",
        f"- OOS reposition pass_share: {wf['summary']['reposition_pass_share']}",
        "- Full same-vol BTC+cash pass_share: "
        f"{full['summary']['same_vol_cash_integrity_pass_share']}",
        "- OOS same-vol BTC+cash pass_share: "
        f"{wf['summary']['same_vol_cash_integrity_pass_share']}",
        f"- Old #30 full pass_share: {full['old_olympus30_gate_summary']['old_gate_pass_share']}",
        "",
        "## Rebalance Sweep",
        "",
        "| frequency | status_quo_pass | same_vol_cash_pass | old_30_pass | "
        "sharpe | sortino | calmar | max_dd_pct | win_pct | ann_turnover | net_cost_pct |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cast(list[dict[str, Any]], full["rebalance_sweep"]):
        metrics = item["candidate"]["metrics"]
        standard = item["candidate"]["standard_metrics"]
        lines.append(
            f"| {item['label']} | {item['passes_reposition_gate']} | "
            f"{item['passes_same_vol_cash_integrity_check']} | "
            f"{item['old_olympus30_gate']['passes_all_gates']} | "
            f"{metrics['sharpe']} | {standard['sortino']} | {metrics['calmar']} | "
            f"{metrics['max_drawdown_pct']} | {standard['positive_period_rate_pct']} | "
            f"{standard['annualized_turnover']} | {standard['net_cost_pct']} |"
        )
    lines.extend(
        [
            "",
            "## Standard Metrics",
            "",
            "- JSON evidence includes `standard_metrics` for each full-sample candidate and "
            "benchmark: max drawdown, Sharpe, Sortino, Calmar, positive-period win rate, "
            "OOS window win rate where applicable, annualized turnover, and net cost.",
            "- Funding/borrow cost is explicitly `N/A` because this is a spot long-only "
            "paper basket with no perp, leverage, short, borrow, or funding exposure.",
            "",
            "## Integrity",
            "",
            "- The #30 gate remains reported side-by-side.",
            "- No universe, frequency grid, sample, or window selection changed.",
            "- candidate-only paper research; no live trading path or order path changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _env_text(name: str, default: str) -> str:
    for candidate in (f"OLYMPUS31_{name}", name):
        raw = os.environ.get(candidate)
        if raw not in (None, ""):
            return raw
    return default


if __name__ == "__main__":
    raise SystemExit(main())
