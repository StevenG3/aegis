from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.microstructure_perp_runner import run_microstructure_perp_from_spec
from aegis.private_paths import private_dir_from_cli

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import microstructure_evidence as base_evidence  # noqa: E402

BASE_GRID = {
    "funding_abs_bps": [1.0, 3.0],
    "imbalance_abs": [0.10, 0.20],
    "oi_drop_abs": [0.02, 0.05],
    "score_threshold": [1, 2],
}
BASE_CONTROL_GRID = (
    {
        "name": "base_no_impulse_no_guard_full_window",
        "btc_impulse": {
            "enabled": False,
            "lookback_bars": 3,
            "return_threshold": 0.0,
            "zscore_threshold": 0.0,
        },
        "liquidity_guard": {
            "enabled": False,
            "max_spread_bps": 25.0,
            "min_top_depth_usd": 50_000.0,
            "min_quote_volume_usd": 0.0,
        },
        "entry_window": {},
    },
)
IMPULSE_LOOKBACK_BARS = (3, 6)
IMPULSE_RETURN_THRESHOLDS = (0.005, 0.01, 0.02)
IMPULSE_ZSCORE_THRESHOLDS = (0.0, 1.0)
QUOTE_VOLUME_THRESHOLDS_USD = (250_000.0, 1_000_000.0)
MAX_SPREAD_BPS = 25.0
MIN_TOP_DEPTH_USD = 50_000.0


def main() -> int:
    generated_at = datetime.now(UTC)
    output_dir = private_dir_from_cli(
        os.getenv("MICROSTRUCTURE_IMPULSE_EVIDENCE_OUTPUT_DIR"),
        default_task="olympus65",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run = replace(base_evidence._run_from_env(), output_dir=output_dir)
    observations, coverage = load_observations(run)
    if not observations:
        raise base_evidence.EvidenceDataError(
            "0 aligned observation rows; check ccxt source/symbols/time window"
        )

    base_controls = tuple(dict(item) for item in BASE_CONTROL_GRID)
    impulse_controls = impulse_control_grid()
    base_run = run_backtest(
        _hypothesis_spec(
            run,
            observations,
            key="olympus65_microstructure_base_real_ccxt",
            control_grid=base_controls,
            signal_suffix=(),
        )
    )
    impulse_run = run_backtest(
        _hypothesis_spec(
            run,
            observations,
            key="olympus65_microstructure_impulse_real_ccxt",
            control_grid=impulse_controls,
            signal_suffix=("btc_impulse", "liquidity_guard", "entry_window"),
        )
    )
    base_payload = cast(Mapping[str, Any], base_run.payload)
    impulse_payload = cast(Mapping[str, Any], impulse_run.payload)
    artifact = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_65_MICROSTRUCTURE_IMPULSE_REAL_CCXT",
        "public_boundary": (
            "raw ccxt-derived observations and full verdict JSON are private-only"
        ),
        "input": base_evidence._run_to_dict(run),
        "coverage": coverage,
        "predeclared": {
            "base_grid": BASE_GRID,
            "base_control_grid": base_controls,
            "impulse_control_grid": impulse_controls,
            "trial_count_base": base_run.spec.trial_count_n,
            "trial_count_impulse": impulse_run.spec.trial_count_n,
            "survivor_light_ceiling": True,
        },
        "base": {
            "spec": _spec_summary(base_run.spec),
            "verdict": _verdict_to_dict(base_run.verdict),
            "report": base_payload,
        },
        "impulse": {
            "spec": _spec_summary(impulse_run.spec),
            "verdict": _verdict_to_dict(impulse_run.verdict),
            "report": impulse_payload,
        },
        "comparison": {
            "base": comparison_row(base_run.verdict, base_payload),
            "impulse": comparison_row(impulse_run.verdict, impulse_payload),
        },
        "decision_boundary": (
            "survivor-light data caps any positive outcome at SUGGESTIVE; "
            "NO_EDGE and INSUFFICIENT are completion states"
        ),
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"microstructure-impulse-evidence-{stamp}.json"
    md_path = output_dir / f"microstructure-impulse-evidence-{stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_summary(artifact, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_verdict": base_run.verdict.verdict,
                "impulse_verdict": impulse_run.verdict.verdict,
                "impulse_state": impulse_run.verdict.state,
                "reason": impulse_run.verdict.reason,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_observations(
    run: base_evidence.EvidenceRun,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    coverage_by_symbol: list[dict[str, Any]] = []
    fetch_failures: list[dict[str, str]] = []
    btc_reference = base_evidence._load_btc_reference(run, fetch_failures)
    for symbol in run.symbols:
        try:
            symbol_rows = base_evidence.load_symbol_observations(
                run, symbol, btc_reference=btc_reference
            )
            if not symbol_rows:
                raise base_evidence.EvidenceDataError(f"no aligned rows for {symbol}")
            observations.extend(symbol_rows)
            coverage_by_symbol.append(
                {
                    "symbol": symbol,
                    "rows": len(symbol_rows),
                    "funding_rows": len(symbol_rows),
                    "survivor_status": base_evidence._survivor_status(run, symbol),
                    "btc_close_rows": sum(1 for row in symbol_rows if row.get("btc_close")),
                    "quote_volume_rows": sum(
                        1 for row in symbol_rows if row.get("quote_volume_usd") is not None
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            fetch_failures.append({"symbol": symbol, "error": str(exc)})
    return observations, {
        "requested_symbols": len(run.symbols),
        "covered_symbols": len({str(row["symbol"]) for row in observations}),
        "requested_delisted_or_crash_symbols": len(run.delisted_crash_symbols),
        "covered_delisted_or_crash_symbols": sum(
            1 for item in coverage_by_symbol if item["symbol"] in run.delisted_crash_symbols
        ),
        "funding_rows": len(observations),
        "observation_rows": len(observations),
        "fetch_failures": fetch_failures,
        "by_symbol": coverage_by_symbol,
    }


def impulse_control_grid() -> tuple[dict[str, object], ...]:
    controls: list[dict[str, object]] = []
    for lookback, return_threshold, zscore, quote_volume in product(
        IMPULSE_LOOKBACK_BARS,
        IMPULSE_RETURN_THRESHOLDS,
        IMPULSE_ZSCORE_THRESHOLDS,
        QUOTE_VOLUME_THRESHOLDS_USD,
    ):
        controls.append(
            {
                "name": (
                    f"imp_lb{lookback}_ret{return_threshold:g}_"
                    f"z{zscore:g}_qv{quote_volume:g}_full"
                ),
                "btc_impulse": {
                    "enabled": True,
                    "lookback_bars": lookback,
                    "return_threshold": return_threshold,
                    "zscore_threshold": zscore,
                },
                "liquidity_guard": {
                    "enabled": True,
                    "max_spread_bps": MAX_SPREAD_BPS,
                    "min_top_depth_usd": MIN_TOP_DEPTH_USD,
                    "min_quote_volume_usd": quote_volume,
                },
                "entry_window": {},
            }
        )
    return tuple(controls)


def _hypothesis_spec(
    run: base_evidence.EvidenceRun,
    observations: Sequence[Mapping[str, Any]],
    *,
    key: str,
    control_grid: Sequence[Mapping[str, object]],
    signal_suffix: tuple[str, ...],
) -> HypothesisSpec:
    trial_n = (
        len(run.symbols)
        * len(BASE_GRID["funding_abs_bps"])
        * len(BASE_GRID["imbalance_abs"])
        * len(BASE_GRID["oi_drop_abs"])
        * len(BASE_GRID["score_threshold"])
        * len(control_grid)
    )
    spec_without_runner = HypothesisSpec(
        key=key,
        hypothesis_type="event",
        universe=run.symbols,
        predeclared_signals=(
            "funding_sign",
            "oi_price_divergence",
            "orderflow_imbalance",
            *signal_suffix,
        ),
        params={
            "observations": [dict(row) for row in observations],
            "grid": BASE_GRID,
            "control_grid": [dict(item) for item in control_grid],
            "locked_oos_fraction": 0.40,
            "fold_count": 6,
            "pbo_splits": 4,
            "pbo_threshold": 0.20,
            "annualization_periods": base_evidence._annualization_periods(run.timeframe),
            "fdr_alpha": 0.10,
            "max_trial_count": max(1, trial_n),
        },
        cost_model={
            "fee_bps": run.taker_fee_bps,
            "slippage_bps": run.slippage_bps,
            "funding_bps_per_period": 0.0,
            "funding_label": "perp funding debited from ccxt funding history observations",
        },
        benchmark="buy_and_hold",
        data_source=f"ccxt.{run.source}.funding+open_interest+perp_klines",
        trial_count_n=max(1, trial_n),
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=True,
    )
    return HypothesisSpec(
        key=spec_without_runner.key,
        hypothesis_type=spec_without_runner.hypothesis_type,
        universe=spec_without_runner.universe,
        predeclared_signals=spec_without_runner.predeclared_signals,
        params=spec_without_runner.params,
        cost_model=spec_without_runner.cost_model,
        benchmark=spec_without_runner.benchmark,
        data_source=spec_without_runner.data_source,
        trial_count_n=spec_without_runner.trial_count_n,
        discipline=spec_without_runner.discipline,
        survivor_light=spec_without_runner.survivor_light,
        runner=lambda: run_microstructure_perp_from_spec(spec_without_runner),
    )


def comparison_row(verdict: Any, payload: Mapping[str, Any]) -> dict[str, object]:
    multiple_testing = cast(Mapping[str, Any], payload.get("multiple_testing", {}))
    pbo_report = cast(Mapping[str, Any], multiple_testing.get("pbo", {}))
    trade_scorecard = cast(Mapping[str, Any], payload.get("trade_scorecard", {}))
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}))
    return {
        "verdict": verdict.verdict,
        "state": verdict.state,
        "data_adequacy": verdict.data_adequacy,
        "unlock_condition": verdict.unlock_condition,
        "candidate_count_n": verdict.candidate_count_n,
        "tested_candidates": multiple_testing.get("tested_candidates"),
        "fdr_after": multiple_testing.get("fdr_after"),
        "pbo_after_survivors": multiple_testing.get("pbo_after_survivors"),
        "min_p_value": multiple_testing.get("min_p_value"),
        "pbo": pbo_report.get("pbo"),
        "pbo_valid": pbo_report.get("valid"),
        "expectancy_per_trade": trade_scorecard.get("expectancy_per_trade"),
        "total_return": metrics.get("total_return"),
        "net_cost": metrics.get("net_cost"),
    }


def _spec_summary(spec: HypothesisSpec) -> dict[str, object]:
    return {
        "key": spec.key,
        "type": spec.hypothesis_type,
        "trial_n": spec.trial_count_n,
        "survivor_light": spec.survivor_light,
        "predeclared_signals": list(spec.predeclared_signals),
    }


def _verdict_to_dict(verdict: Any) -> dict[str, object]:
    return {
        "state": verdict.state,
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "data_adequacy": verdict.data_adequacy,
        "unlock_condition": verdict.unlock_condition,
        "candidate_count_n": verdict.candidate_count_n,
        "raw_survivors": verdict.raw_survivors,
        "fdr_survivors": verdict.fdr_survivors,
        "survivor_ceiling_applied": verdict.survivor_ceiling_applied,
        "multiple_testing": dict(verdict.multiple_testing),
        "safety": dict(verdict.safety),
    }


def markdown_summary(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload["coverage"])
    comparison = cast(Mapping[str, Mapping[str, object]], payload["comparison"])
    base = comparison["base"]
    impulse = comparison["impulse"]
    return "\n".join(
        [
            "# CODEX OLYMPUS 65 Microstructure Impulse Evidence",
            "",
            f"Generated: `{payload['generated_at']}`",
            f"JSON artifact: `{json_path}`",
            "",
            "## Verdict",
            "",
            f"- Base: `{base.get('verdict')}`",
            f"- Impulse: `{impulse.get('verdict')}`",
            "",
            "## Impulse vs Base",
            "",
            "| Variant | N | Tested | FDR after | PBO | PBO survivors | Min p | Expectancy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            _comparison_markdown_row("Base", base),
            _comparison_markdown_row("Impulse", impulse),
            "",
            "## Coverage",
            "",
            f"- Requested symbols: `{coverage.get('requested_symbols')}`",
            f"- Covered symbols: `{coverage.get('covered_symbols')}`",
            f"- Funding rows: `{coverage.get('funding_rows')}`",
            "- Covered delisted/crash symbols: "
            f"`{coverage.get('covered_delisted_or_crash_symbols')}`",
            "",
            "## Discipline",
            "",
            "- Same ccxt observation pool for base and impulse variants.",
            "- Signals observed at t; entries happen at t+1 inside the runner.",
            "- Fees, slippage, and perp funding are counted.",
            "- All base and impulse control-grid trials are counted in BH-FDR + PBO.",
            "- Survivor-light evidence caps any positive outcome below ROBUST.",
        ]
    ) + "\n"


def _comparison_markdown_row(label: str, row: Mapping[str, object]) -> str:
    return (
        f"| {label} | `{row.get('candidate_count_n')}` | `{row.get('tested_candidates')}` | "
        f"`{row.get('fdr_after')}` | `{row.get('pbo')}` | "
        f"`{row.get('pbo_after_survivors')}` | `{row.get('min_p_value')}` | "
        f"`{row.get('expectancy_per_trade')}` |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
