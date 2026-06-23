#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.polymarket_forward_execution import (
    TRIAL_COUNT_N,
    ForwardExecutionConfig,
    run_forward_execution_backtest,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus63"


def main() -> int:
    args = _parse_args()
    base_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir is not None
        else base_dir / "forward_execution_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    forward_dir = Path(args.forward_dir) if args.forward_dir else base_dir / "forward"
    rows = load_forward_rows(forward_dir)
    if not rows:
        raise SystemExit(f"forward data source has 0 rows: {forward_dir}")
    config = ForwardExecutionConfig(
        notional_usd=float(args.notional_usd),
        min_markets=int(args.min_markets),
        pbo_splits=int(args.pbo_splits),
        venue_geoblocked=bool(args.venue_geoblocked),
    )
    spec = HypothesisSpec(
        key="olympus63_polymarket_btc_5m_forward_execution",
        hypothesis_type="event",
        universe=("polymarket_btc_5m_updown_forward",),
        predeclared_signals=("chainlink_btc_impulse", "near_settlement_orderbook_ask"),
        params={
            "notional_usd": config.notional_usd,
            "min_markets": config.min_markets,
            "pbo_splits": config.pbo_splits,
            "venue_geoblocked": config.venue_geoblocked,
            "forward_rows": len(rows),
        },
        cost_model={
            "execution": "walk_actual_ask_book_and_preclose_bid_book",
            "fee_model": "public CLOB fee not separately added; spread/depth execution included",
        },
        benchmark="no_trade_cash",
        data_source="private_forward_polymarket_orderbook+chainlink_reference",
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
        runner=lambda: run_forward_execution_backtest(rows, config=config),
    )
    run = run_backtest(spec)
    payload = cast(Mapping[str, Any], run.payload)
    generated_at = datetime.now(UTC)
    artifact = sanitized_artifact(
        generated_at=generated_at,
        forward_dir=forward_dir,
        rows=len(rows),
        config=config,
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
    json_path = artifact_dir / f"polymarket-forward-execution-{stamp}.json"
    md_path = artifact_dir / f"polymarket-forward-execution-{stamp}.md"
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
                "raw_forward_dir": str(forward_dir),
                "artifact_dir": str(artifact_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def sanitized_artifact(
    *,
    generated_at: datetime,
    forward_dir: Path,
    rows: int,
    config: ForwardExecutionConfig,
    spec: HypothesisSpec,
    verdict: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}))
    multiple_testing = cast(Mapping[str, Any], payload.get("multiple_testing", {}))
    safety = cast(Mapping[str, Any], payload.get("safety", {}))
    fill_ratio = cast(Mapping[str, Any], payload.get("fill_ratio_distribution", {}))
    settlement_source = cast(Mapping[str, Any], payload.get("settlement_source", {}))
    return {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_63_POLYMARKET_5M_FORWARD_EXECUTION",
        "input": {
            "forward_dir": str(forward_dir),
            "rows": rows,
            "notional_usd": config.notional_usd,
            "min_markets": config.min_markets,
            "pbo_splits": config.pbo_splits,
            "venue_geoblocked": config.venue_geoblocked,
        },
        "spec": {
            "key": spec.key,
            "trial_n": spec.trial_count_n,
            "survivor_light": spec.survivor_light,
        },
        "verdict": dict(verdict),
        "report": {
            "status": payload.get("status"),
            "verdict": payload.get("verdict"),
            "reason": payload.get("reason"),
            "coverage": dict(coverage),
            "settlement_source": dict(settlement_source),
            "multiple_testing": {
                "method": multiple_testing.get("method"),
                "candidate_count_n": multiple_testing.get("candidate_count_n"),
                "tested_candidates": multiple_testing.get("tested_candidates"),
                "fdr_after": multiple_testing.get("fdr_after"),
                "preclose_survivors": multiple_testing.get("preclose_survivors"),
                "pbo": multiple_testing.get("pbo"),
                "settlement_is_control_only": multiple_testing.get(
                    "settlement_is_control_only"
                ),
            },
            "fill_ratio_distribution": dict(fill_ratio),
            "safety": dict(safety),
        },
    }


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


def markdown_summary(artifact: Mapping[str, Any], json_path: Path) -> str:
    verdict = cast(Mapping[str, Any], artifact["verdict"])
    report = cast(Mapping[str, Any], artifact["report"])
    coverage = cast(Mapping[str, Any], report.get("coverage", {}))
    multiple_testing = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    return "\n".join(
        [
            "# CODEX OLYMPUS 63 Forward Execution Evidence",
            "",
            f"- Verdict: `{verdict['verdict']}`",
            f"- State: `{verdict['state']}`",
            f"- Reason: {verdict['reason']}",
            f"- JSON: `{json_path}`",
            "",
            "## Coverage",
            f"- Markets: `{coverage.get('markets', 0)}`",
            f"- Settled markets: `{coverage.get('settled_markets', 0)}`",
            f"- Chainlink-ready markets: `{coverage.get('chainlink_ready_markets', 0)}`",
            f"- Verified settlement-source markets: "
            f"`{coverage.get('verified_settlement_source_markets', 0)}`",
            f"- Snapshots: `{coverage.get('snapshots', 0)}`",
            "",
            "## Multiple Testing",
            f"- Candidate N: `{multiple_testing.get('candidate_count_n', TRIAL_COUNT_N)}`",
            f"- FDR survivors: `{multiple_testing.get('fdr_after', 0)}`",
            f"- PBO: `{multiple_testing.get('pbo', {})}`",
            "",
            "## Safety",
            "- Read-only public data only; no wallet, no order API, no account API.",
            "- Settlement is a control exit only; preclose orderbook execution is the "
            "decision gate.",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run read-only Polymarket BTC 5m forward execution evidence."
    )
    parser.add_argument("--output-dir", default=os.getenv("POLYMARKET_FORWARD_OUTPUT_DIR"))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--forward-dir", default=None)
    parser.add_argument("--notional-usd", type=float, default=25.0)
    parser.add_argument("--min-markets", type=int, default=100)
    parser.add_argument("--pbo-splits", type=int, default=4)
    parser.add_argument("--venue-geoblocked", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
