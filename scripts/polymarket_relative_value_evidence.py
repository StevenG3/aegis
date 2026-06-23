#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.polymarket_relative_value import (
    TRIAL_COUNT_N,
    RelativeValueConfig,
    run_relative_value_calibration,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus68"


def main() -> int:
    args = _parse_args()
    base_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else base_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not args.forward_dir:
        raise SystemExit("--forward-dir or POLYMARKET_RV_FORWARD_DIR is required")
    forward_dir = Path(args.forward_dir)
    rows = load_forward_rows(forward_dir)
    if not rows:
        raise SystemExit(f"forward data source has 0 rows: {forward_dir}")
    config = RelativeValueConfig(
        min_markets=args.min_markets,
        min_observations=args.min_observations,
        min_bucket_observations=args.min_bucket_observations,
        target_tolerance_seconds=args.target_tolerance_seconds,
    )
    spec = HypothesisSpec(
        key="olympus68_polymarket_btc_5m_relative_value_calibration",
        hypothesis_type="event",
        universe=("polymarket_btc_5m_updown_forward",),
        predeclared_signals=("favorite_mid_implied_probability", "settlement_label"),
        params={
            "time_to_close_seconds": list(config.time_to_close_seconds),
            "probability_bins": [list(value) for value in config.probability_bins],
            "target_tolerance_seconds": config.target_tolerance_seconds,
            "min_markets": config.min_markets,
            "min_observations": config.min_observations,
            "min_bucket_observations": config.min_bucket_observations,
            "forward_rows": len(rows),
        },
        cost_model={
            "execution": "none; calibration only",
            "spread_upper_bound": "favorite ask minus realized settlement frequency",
        },
        benchmark="well_calibrated_market",
        data_source="private_forward_polymarket_orderbook+settlement_labels",
        trial_count_n=TRIAL_COUNT_N,
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=True,
        runner=lambda: run_relative_value_calibration(rows, config=config),
    )
    run = run_backtest(spec)
    payload = cast(Mapping[str, Any], run.payload)
    generated_at = datetime.now(UTC)
    artifact = sanitized_artifact(
        generated_at=generated_at,
        forward_dir=forward_dir,
        rows=len(rows),
        spec=spec,
        verdict={
            "state": run.verdict.state,
            "verdict": run.verdict.verdict,
            "reason": run.verdict.reason,
            "data_adequacy": run.verdict.data_adequacy,
            "unlock_condition": run.verdict.unlock_condition,
            "candidate_count_n": run.verdict.candidate_count_n,
            "fdr_survivors": run.verdict.fdr_survivors,
            "survivor_ceiling_applied": run.verdict.survivor_ceiling_applied,
        },
        payload=payload,
    )
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = artifact_dir / f"polymarket-relative-value-{stamp}.json"
    md_path = artifact_dir / f"polymarket-relative-value-{stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_summary(artifact, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": run.verdict.state,
                "verdict": run.verdict.verdict,
                "reason": run.verdict.reason,
                "coverage": payload.get("coverage", {}),
                "fdr_after": cast(Mapping[str, Any], payload.get("multiple_testing", {})).get(
                    "fdr_after"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_forward_rows(forward_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(forward_dir.glob("date=*/hour=*/polymarket_btc_5m_forward.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, dict):
                rows.append(raw)
    return rows


def sanitized_artifact(
    *,
    generated_at: datetime,
    forward_dir: Path,
    rows: int,
    spec: HypothesisSpec,
    verdict: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_68_POLYMARKET_5M_RELATIVE_VALUE",
        "input": {
            "forward_dir": str(forward_dir),
            "rows": rows,
        },
        "spec": {
            "key": spec.key,
            "trial_n": spec.trial_count_n,
            "survivor_light": spec.survivor_light,
            "params": dict(spec.params),
        },
        "verdict": dict(verdict),
        "report": {
            "status": payload.get("status"),
            "verdict": payload.get("verdict"),
            "reason": payload.get("reason"),
            "coverage": dict(cast(Mapping[str, Any], payload.get("coverage", {}))),
            "standard_metrics": dict(
                cast(Mapping[str, Any], payload.get("standard_metrics", {}))
            ),
            "benchmark_metrics": dict(
                cast(Mapping[str, Any], payload.get("benchmark_metrics", {}))
            ),
            "multiple_testing": dict(
                cast(Mapping[str, Any], payload.get("multiple_testing", {}))
            ),
            "calibration_buckets": list(
                cast(list[Mapping[str, Any]], payload.get("calibration_buckets", []))
            ),
            "deviation_test": dict(
                cast(Mapping[str, Any], payload.get("deviation_test", {}))
            ),
            "safety": dict(cast(Mapping[str, Any], payload.get("safety", {}))),
        },
    }


def markdown_summary(artifact: Mapping[str, Any], json_path: Path) -> str:
    verdict = cast(Mapping[str, Any], artifact["verdict"])
    report = cast(Mapping[str, Any], artifact["report"])
    coverage = cast(Mapping[str, Any], report.get("coverage", {}))
    metrics = cast(Mapping[str, Any], report.get("standard_metrics", {}))
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    buckets = cast(list[Mapping[str, Any]], report.get("calibration_buckets", []))
    lines = [
        "# CODEX OLYMPUS 68 Polymarket Relative Value Evidence",
        "",
        f"- Verdict: `{verdict['verdict']}`",
        f"- State: `{verdict['state']}`",
        f"- Reason: {verdict['reason']}",
        f"- JSON: `{json_path}`",
        "",
        "## Coverage",
        f"- Markets: `{coverage.get('markets', 0)}`",
        f"- Settled markets: `{coverage.get('settled_markets', 0)}`",
        f"- Calibration observations: `{coverage.get('calibration_observations', 0)}`",
        f"- Snapshots: `{coverage.get('snapshots', 0)}`",
        "",
        "## Metrics",
        f"- Mean implied probability: `{metrics.get('mean_implied_probability')}`",
        f"- Actual favorite win rate: `{metrics.get('actual_favorite_win_rate')}`",
        f"- Mean favorite spread: `{metrics.get('mean_favorite_spread')}`",
        f"- Max positive edge after ask: `{metrics.get('max_positive_edge_after_ask')}`",
        "",
        "## Multiple Testing",
        f"- Candidate N: `{multiple.get('candidate_count_n', TRIAL_COUNT_N)}`",
        f"- Tested buckets: `{multiple.get('tested_buckets')}`",
        f"- FDR after: `{multiple.get('fdr_after')}`",
        f"- Exploitable after ask: `{multiple.get('exploitable_after_ask')}`",
        "",
        "## Calibration Buckets",
        "| ttc | bucket | N | implied | actual | CI | FDR miscalibrated | max edge after ask |",
        "|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for row in buckets:
        lines.append(
            "| {target} | {bucket} | {n} | {implied:.4f} | {actual:.4f} | "
            "[{low}, {high}] | {flag} | {edge} |".format(
                target=row.get("target_seconds"),
                bucket=row.get("bucket"),
                n=row.get("n"),
                implied=float(row.get("mean_implied_probability", 0.0)),
                actual=float(row.get("actual_win_rate", 0.0)),
                low=_fmt(row.get("binomial_ci_low")),
                high=_fmt(row.get("binomial_ci_high")),
                flag=row.get("fdr_miscalibrated"),
                edge=_fmt(row.get("max_edge_after_ask")),
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "- Offline calibration only; no wallet, order API, account API, live, or "
            "execution modeling.",
            "- Settlement is used only as the outcome label.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return "NA"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Polymarket BTC 5m relative-value calibration evidence."
    )
    parser.add_argument("--output-dir", default=os.getenv("POLYMARKET_RV_OUTPUT_DIR"))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--forward-dir",
        default=os.getenv("POLYMARKET_RV_FORWARD_DIR"),
    )
    parser.add_argument("--min-markets", type=int, default=100)
    parser.add_argument("--min-observations", type=int, default=50)
    parser.add_argument("--min-bucket-observations", type=int, default=5)
    parser.add_argument("--target-tolerance-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
