#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

from aegis.edgar_factor_costs import run_edgar_factor_cost_diagnostic
from aegis.edgar_full_universe_ic import EdgarIcObservation
from aegis.private_paths import private_dir_from_cli, private_root_from_env


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task="olympus87")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_json = Path(args.source_json) if args.source_json else _latest_olympus83_json()
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    observations = _observations_from_payload(source_payload)
    report = run_edgar_factor_cost_diagnostic(
        observations,
        coverage={
            "source_json": str(source_json),
            "source_briefing": source_payload.get("briefing"),
            "source_universe_mode": source_payload.get("universe_mode"),
            "source_observation_count": source_payload.get("observation_count"),
            "source_coverage": source_payload.get("coverage", {}),
        },
    )
    generated_at = _utc_now()
    payload = {
        "briefing": "CODEX_OLYMPUS_87_EDGAR_NET_COST_LONGSHORT",
        "generated_at": generated_at.isoformat(),
        "source_json": str(source_json),
        "report": report,
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "public_edge_numbers": False,
        },
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"edgar-factor-costs-{stamp}.json"
    md_path = output_dir / f"edgar-factor-costs-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": report["state"],
                "verdict": report["verdict"],
                "data_adequacy": report["data_adequacy"],
                "reason": report["reason"],
                "sharadar_decision": report.get("sharadar_decision", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cost-aware portfolio diagnostics on #83 EDGAR as-of observations."
    )
    parser.add_argument("--source-json", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _latest_olympus83_json() -> Path:
    root = private_root_from_env() / "incubating" / "olympus83"
    paths = sorted(root.glob("edgar-panel-build-ic-*.json"))
    if not paths:
        raise FileNotFoundError(f"no #83 EDGAR panel JSON found under {root}")
    return paths[-1]


def _observations_from_payload(payload: Mapping[str, Any]) -> list[EdgarIcObservation]:
    raw = payload.get("observations", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("source payload observations must be a list")
    observations = [_observation_from_json(cast(Mapping[str, Any], item)) for item in raw]
    if not observations:
        raise ValueError("source payload has 0 observations; refusing silent INSUFFICIENT")
    return observations


def _observation_from_json(row: Mapping[str, Any]) -> EdgarIcObservation:
    return EdgarIcObservation(
        symbol=str(row["symbol"]),
        as_of=date.fromisoformat(str(row["as_of"])),
        available_on=date.fromisoformat(str(row["available_on"])),
        factors={
            str(key): float(value)
            for key, value in cast(Mapping[str, Any], row.get("factors", {})).items()
        },
        forward_returns={
            str(key): float(value)
            for key, value in cast(Mapping[str, Any], row.get("forward_returns", {})).items()
        },
        in_universe=bool(row.get("in_universe", True)),
    )


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    report = cast(Mapping[str, Any], payload["report"])
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    decision = cast(Mapping[str, Any], report.get("sharadar_decision", {}))
    base_rows = cast(Sequence[Mapping[str, Any]], report.get("base_long_short", ()))
    long_only = cast(Sequence[Mapping[str, Any]], report.get("personal_long_only", ()))
    lines = [
        "# Olympus #87 EDGAR Net-Cost Factor Portfolios",
        "",
        f"- State: `{report['state']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Data adequacy: `{report['data_adequacy']}`",
        f"- Reason: {report['reason']}",
        f"- Trial N: `{multiple.get('candidate_count_n')}`",
        f"- FDR survivors: `{multiple.get('fdr_survivors')}`",
        f"- Net survivors: `{multiple.get('net_survivors')}`",
        f"- Sharadar decision: `{decision.get('decision')}`",
        f"- JSON: `{json_path}`",
        "",
        "## Base Long/Short Net Cost Rows",
        "",
        "| Trial | Gross ann. | Net ann. | Turnover | Net cost | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(_portfolio_row(row) for row in base_rows)
    lines.extend(
        [
            "",
            "## Personal Long-Only Rows",
            "",
            "| Trial | Net ann. | MaxDD | Monthly win | Turnover | Net cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_long_only_row(row) for row in long_only)
    lines.append("")
    lines.append("Public repository contains only generic code; full numeric artifact is private.")
    return "\n".join(lines)


def _portfolio_row(row: Mapping[str, Any]) -> str:
    trial = f"{row.get('factor')} {row.get('horizon')}"
    return (
        f"| `{trial}` | {_pct(row.get('gross_annualized_return'))} | "
        f"{_pct(row.get('net_annualized_return'))} | {_num(row.get('turnover'))} | "
        f"{_pct(row.get('net_cost'))} | `{row.get('net_verdict')}` |"
    )


def _long_only_row(row: Mapping[str, Any]) -> str:
    trial = f"{row.get('factor')} {row.get('horizon')}"
    return (
        f"| `{trial}` | {_pct(row.get('net_annualized_return'))} | "
        f"{_pct(row.get('net_max_drawdown'))} | {_pct(row.get('monthly_win_rate'))} | "
        f"{_num(row.get('turnover'))} | {_pct(row.get('net_cost'))} |"
    )


def _pct(value: object) -> str:
    return f"{_object_float(value) * 100.0:.2f}%" if value is not None else "N/A"


def _num(value: object) -> str:
    return f"{_object_float(value):.2f}" if value is not None else "N/A"


def _object_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - host evidence runner is Python 3.10.


if __name__ == "__main__":
    raise SystemExit(main())
