#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

from aegis.edgar_full_universe_ic import (
    EdgarIcConfig,
    EdgarIcObservation,
    run_edgar_full_universe_ic,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_INPUT_RELATIVE = Path(
    "incubating/olympus41/olympus41b-real-universe-survivor-light-20260610T134539Z.json"
)
DEFAULT_TASK = "olympus81"


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = _input_path(args.input_json)
    observations, source_coverage = _load_legacy_observations(input_path)
    coverage = {
        **source_coverage,
        "sec_user_agent_configured": bool(os.environ.get("AEGIS_SEC_USER_AGENT", "").strip()),
        "data_gate_note": (
            "Existing private artifacts are used only when they already contain PIT EDGAR "
            "fundamentals and forward labels; this script does not fabricate full-universe data."
        ),
        "full_free_rebuild_status": (
            "blocked_without_AEGIS_SEC_USER_AGENT"
            if not os.environ.get("AEGIS_SEC_USER_AGENT", "").strip()
            else "not_attempted_by_default_to_avoid_unbounded_SEC_yfinance_batch"
        ),
    }
    report = run_edgar_full_universe_ic(
        observations,
        config=EdgarIcConfig(),
        coverage=coverage,
    )
    payload = {
        "briefing": "CODEX_OLYMPUS_81_EDGAR_FULL_UNIVERSE_IC",
        "generated_at": _utc_now().isoformat(),
        "source_input": str(input_path),
        "report": report,
        "sharadar_decision": _sharadar_decision(report),
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "public_edge_values_sanitized": True,
        },
    }
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"edgar-full-universe-ic-{stamp}.json"
    md_path = output_dir / f"edgar-full-universe-ic-{stamp}.md"
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
                "coverage": report["coverage"],
                "sharadar_decision": payload["sharadar_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Olympus #81 EDGAR full-universe IC evidence from private PIT rows."
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Private PIT observation JSON. Defaults to "
            "${AEGIS_STRATEGIES_ROOT}/incubating/olympus41/... when unset."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - host evidence runner is Python 3.10.


def _input_path(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    root = os.environ.get("AEGIS_STRATEGIES_ROOT", "").strip()
    if not root:
        return DEFAULT_INPUT_RELATIVE
    return Path(root) / DEFAULT_INPUT_RELATIVE


def _load_legacy_observations(path: Path) -> tuple[list[EdgarIcObservation], dict[str, Any]]:
    if not path.exists():
        return [], {"input_status": "missing", "input_path": str(path)}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = loaded.get("observations") if isinstance(loaded, dict) else None
    if not isinstance(raw_rows, list):
        return [], {"input_status": "invalid_no_observations", "input_path": str(path)}
    observations: list[EdgarIcObservation] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        parsed = _legacy_row_to_observation(item)
        if parsed is not None:
            observations.append(parsed)
    data_sources = loaded.get("data_sources", {}) if isinstance(loaded, dict) else {}
    return observations, {
        "input_status": "loaded",
        "input_path": str(path),
        "legacy_rows": len(raw_rows),
        "converted_rows": len(observations),
        "legacy_note": (
            "The #41B artifact is a bounded pipeline-validation sample, not the #81 "
            "full ~500-name monthly universe."
        ),
        "legacy_data_sources": data_sources if isinstance(data_sources, dict) else {},
    }


def _legacy_row_to_observation(row: dict[str, Any]) -> EdgarIcObservation | None:
    try:
        as_of = date.fromisoformat(str(row["rebalance_date"]))
        symbol = str(row["ticker"]).upper()
        raw_factors = cast(dict[str, Any], row.get("factors", {}))
        forward_return = float(row["forward_return"])
    except (KeyError, TypeError, ValueError):
        return None
    factors = _canonical_factors(raw_factors)
    if not factors:
        return None
    return EdgarIcObservation(
        symbol=symbol,
        as_of=as_of,
        available_on=as_of,
        factors=factors,
        forward_returns={"3m": forward_return},
        in_universe=True,
    )


def _canonical_factors(raw: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    mapping = {
        "earnings_yield_ep": "earnings_yield_ep",
        "book_to_price_bp": "book_to_price_bp",
        "free_cash_flow_to_price": "fcf_yield",
        "sales_to_price_sp": "sales_to_price_sp",
        "roe": "roe",
    }
    for source, target in mapping.items():
        if source in raw:
            result[target] = float(raw[source])
    if "accruals" in raw:
        result["low_accruals"] = -float(raw["accruals"])
    return result


def _sharadar_decision(report: dict[str, Any]) -> dict[str, str]:
    verdict = str(report.get("verdict"))
    if verdict == "SUGGESTIVE_NEEDS_PAID_CONFIRM":
        return {
            "decision": "PAY_TO_CONFIRM_ONLY_IF_BUDGET_ACCEPTS_SURVIVOR_UNLOCK",
            "reason": "free survivor-light IC found a signal that requires PIT paid validation",
        }
    if verdict == "NO_EDGE":
        return {
            "decision": "DO_NOT_PAY_FOR_THIS_FACTOR_SET_NOW",
            "reason": "full-scope free IC did not survive FDR/PBO gates",
        }
    return {
        "decision": "NO_PURCHASE_DECISION_FROM_CURRENT_DATA",
        "reason": "data gate did not reach full-universe monthly IC coverage",
    }


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    report = cast(dict[str, Any], payload["report"])
    coverage = cast(dict[str, Any], report.get("coverage", {}))
    decision = cast(dict[str, str], payload["sharadar_decision"])
    return "\n".join(
        [
            "# Olympus #81 EDGAR Full-Universe IC",
            "",
            f"- Verdict: `{report['verdict']}`",
            f"- State: `{report['state']}`",
            f"- Data adequacy: `{report['data_adequacy']}`",
            f"- Reason: {report['reason']}",
            f"- Trial count N: `{report['predeclared']['trial_count_n']}`",
            f"- Eligible rows: `{coverage.get('eligible_rows')}`",
            f"- Symbols: `{coverage.get('symbols')}`",
            f"- Periods: `{coverage.get('periods')}`",
            f"- Sharadar decision: `{decision['decision']}`",
            f"- JSON: `{json_path}`",
            "",
            "This is read-only research evidence. It does not access broker accounts, "
            "place orders, or claim a robust edge from survivor-light data.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
