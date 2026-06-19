from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "backtest-service"
sys.path.insert(0, str(SERVICE_DIR))

from data import DataLoadError  # noqa: E402
from funding_arb import (  # noqa: E402
    FundingArbConfig,
    FundingResearchConfig,
    FundingSource,
    _symbols,
    load_aligned_funding_events,
    run_funding_arb_research,
)

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_SOURCE: FundingSource = "bybit"
DEFAULT_START = "2023-07-01"
DEFAULT_END = "2026-01-01"
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "OLYMPUS_EVIDENCE_DIR",
        str(Path.home() / "aegis-strategies" / "incubating" / "olympus50"),
    )
)


@dataclass(frozen=True)
class EvidenceRun:
    source: FundingSource
    symbols: tuple[str, ...]
    start: str
    end: str
    cash: float
    max_events: int
    cash_rate_annual: float
    taker_fee_bps: float
    slippage_bps: float
    basis_cost_bps: float
    borrow_cost_bps_annual: float
    output_dir: Path


def main() -> int:
    generated_at = datetime.now(UTC)
    run = _run_from_env()
    run.output_dir.mkdir(parents=True, exist_ok=True)
    events_by_symbol: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    for symbol in run.symbols:
        try:
            config = _base_config(run, symbol)
            events_by_symbol[symbol] = load_aligned_funding_events(config, _symbols(symbol))
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})

    report = run_funding_arb_research(
        events_by_symbol,
        source=run.source,
        start=run.start,
        end=run.end,
        cash=run.cash,
        base_config=_base_config(run, run.symbols[0]),
        research_config=FundingResearchConfig(),
    )
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_50_FUNDING_ARB",
        "public_boundary": (
            "raw funding and price rows are private-only and omitted from public repo"
        ),
        "input": {
            "source": run.source,
            "symbols": list(run.symbols),
            "start": run.start,
            "end": run.end,
            "cash": run.cash,
            "cash_rate_annual": run.cash_rate_annual,
            "taker_fee_bps": run.taker_fee_bps,
            "slippage_bps": run.slippage_bps,
            "basis_cost_bps": run.basis_cost_bps,
            "borrow_cost_bps_annual": run.borrow_cost_bps_annual,
            "max_events": run.max_events,
        },
        "fetch_failures": failures,
        "report": report,
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = run.output_dir / f"funding-arb-research-{stamp}.json"
    md_path = run.output_dir / f"funding-arb-research-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "reason": report["reason"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _base_config(run: EvidenceRun, symbol: str) -> FundingArbConfig:
    return FundingArbConfig(
        symbol=symbol,
        source=run.source,
        start=run.start,
        end=run.end,
        cash=run.cash,
        taker_fee_bps=run.taker_fee_bps,
        slippage_bps=run.slippage_bps,
        basis_cost_bps=run.basis_cost_bps,
        borrow_cost_bps_annual=run.borrow_cost_bps_annual,
        cash_rate_annual=run.cash_rate_annual,
        max_funding_events=run.max_events,
    )


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    report = cast(dict[str, Any], payload["report"])
    lines = [
        "# CODEX OLYMPUS 50 Funding Arb Evidence",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        f"Reason: {report['reason']}",
        f"JSON artifact: `{json_path}`",
        "",
        "## Discipline",
        "",
        "- Spot long + equal-notional perpetual short; no speculative leverage.",
        "- Funding is settled event by event in the held short-perp direction.",
        "- Signal observes event t; entry/exit fills at the next aligned event price.",
        "- Benchmark is risk-free cash, not buy-and-hold.",
        "- Predeclared grid is included in BH-FDR; raw data stays private.",
        "",
        "## Summary",
        "",
        f"- Search space: `{report.get('search_space_n', 0)}`",
        f"- Tested candidates: `{report.get('tested_candidates', 0)}`",
        f"- FDR: `{report.get('fdr', {})}`",
        f"- Best candidate: `{report.get('best_candidate')}`",
        "",
        "## Failures",
        "",
    ]
    failures = cast(list[dict[str, str]], payload["fetch_failures"])
    if not failures:
        lines.append("No fetch failures.")
    else:
        for failure in failures:
            lines.append(f"- `{failure['symbol']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def _run_from_env() -> EvidenceRun:
    return EvidenceRun(
        source=cast(FundingSource, os.getenv("FUNDING_ARB_EVIDENCE_SOURCE", DEFAULT_SOURCE)),
        symbols=tuple(_env_csv("FUNDING_ARB_EVIDENCE_SYMBOLS", DEFAULT_SYMBOLS)),
        start=os.getenv("FUNDING_ARB_EVIDENCE_START", DEFAULT_START),
        end=os.getenv("FUNDING_ARB_EVIDENCE_END", DEFAULT_END),
        cash=_env_float("FUNDING_ARB_EVIDENCE_CASH", 10_000.0),
        max_events=_env_int("FUNDING_ARB_EVIDENCE_MAX_EVENTS", 1500),
        cash_rate_annual=_env_float("FUNDING_ARB_EVIDENCE_CASH_RATE_ANNUAL", 0.04),
        taker_fee_bps=_env_float("FUNDING_ARB_EVIDENCE_TAKER_FEE_BPS", 10.0),
        slippage_bps=_env_float("FUNDING_ARB_EVIDENCE_SLIPPAGE_BPS", 2.0),
        basis_cost_bps=_env_float("FUNDING_ARB_EVIDENCE_BASIS_COST_BPS", 0.0),
        borrow_cost_bps_annual=_env_float("FUNDING_ARB_EVIDENCE_BORROW_COST_BPS_ANNUAL", 0.0),
        output_dir=Path(os.getenv("FUNDING_ARB_EVIDENCE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )


def _env_csv(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataLoadError as exc:
        print(json.dumps({"verdict": "INSUFFICIENT", "reason": str(exc)}, indent=2))
        raise SystemExit(0) from exc
