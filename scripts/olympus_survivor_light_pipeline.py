#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from aegis.edgar_pit import (
    PitFundamentalStore,
    SecEdgarClient,
    extract_submission_metadata,
)
from aegis.olympus_survivor_light import (
    FreePriceSource,
    align_pit_fundamentals_with_prices,
    download_wikipedia_sp500_html,
    evaluate_survivor_light_ic,
    parse_wikipedia_sp500_snapshot,
    timestamp_slug,
)
from aegis.private_paths import private_dir_from_cli


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Olympus #41 survivor-light free-data pipeline."
    )
    parser.add_argument(
        "--private-dir",
        default=None,
        help="Private output directory; do not point inside public aegis.",
    )
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get(
            "AEGIS_SEC_USER_AGENT",
            "AegisOlympusResearch/0.1 contact=https://example.org/aegis",
        ),
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=8,
        help="Bound the private free-data E2E run after selecting from the real as-of universe.",
    )
    args = parser.parse_args()

    private_dir = private_dir_from_cli(args.private_dir, default_task="olympus41")
    cache_dir = private_dir / "cache"
    price_cache_dir = cache_dir / "prices"
    edgar_cache_dir = cache_dir / "edgar"
    constituent_cache_path = cache_dir / "constituents" / "wikipedia-sp500.html"
    private_dir.mkdir(parents=True, exist_ok=True)

    rebalance_dates = [
        date(2024, 3, 29),
        date(2024, 6, 28),
        date(2024, 9, 30),
        date(2024, 12, 31),
        date(2025, 3, 31),
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
    ]
    constituent_html = download_wikipedia_sp500_html(
        cache_path=constituent_cache_path,
        user_agent=str(args.sec_user_agent),
    )
    constituent_snapshot = parse_wikipedia_sp500_snapshot(
        constituent_html,
        as_of_date=date.today(),
    )
    universe_counts = {
        rebalance_date.isoformat(): len(constituent_snapshot.store.as_of(rebalance_date))
        for rebalance_date in rebalance_dates
    }
    selected_tickers = _select_tickers_from_asof_universe(
        constituent_snapshot.current,
        [
            ticker
            for rebalance_date in rebalance_dates
            for ticker in sorted(constituent_snapshot.store.as_of(rebalance_date))
        ],
        max_tickers=max(1, int(args.max_tickers)),
    )
    selected_ciks = {
        ticker: constituent_snapshot.current[ticker].cik
        for ticker in selected_tickers
        if constituent_snapshot.current[ticker].cik is not None
    }
    e2e_constituent_store = constituent_snapshot.store.filtered(selected_ciks.keys())

    client = SecEdgarClient(
        cache_dir=edgar_cache_dir,
        user_agent=str(args.sec_user_agent),
        requests_per_second=5,
    )
    parts: list[PitFundamentalStore] = []
    for ticker, cik in selected_ciks.items():
        assert cik is not None
        companyfacts = client.fetch_companyfacts(cik)
        submissions = client.fetch_submissions(cik)
        metadata = extract_submission_metadata(
            ticker=ticker,
            cik=cik,
            payload=submissions,
            pilot_status="wikipedia_asof_survivor_light_candidate",
        )
        parts.append(
            PitFundamentalStore.from_companyfacts(
                ticker=ticker,
                cik=cik,
                payload=companyfacts,
                company_metadata=metadata,
            )
        )

    fundamentals = PitFundamentalStore(
        facts=[fact for part in parts for fact in part.facts],
        restatements=[fact for part in parts for fact in part.restatements],
        company_metadata={
            ticker: metadata
            for part in parts
            for ticker, metadata in part.company_metadata.items()
        },
    )
    price_source = FreePriceSource(cache_dir=price_cache_dir, delisted_tickers={"ENRNQ"})
    observations = align_pit_fundamentals_with_prices(
        fundamentals=fundamentals,
        price_source=price_source,
        constituent_store=e2e_constituent_store,
        rebalance_dates=rebalance_dates,
    )
    benchmark = next(iter(selected_ciks), None)
    report = evaluate_survivor_light_ic(observations, benchmark_symbol=benchmark)
    slug = timestamp_slug()
    output = {
        "briefing": "CODEX_OLYMPUS_41B_REAL_UNIVERSE_AND_PAID_PRICE",
        "run_id": slug,
        "data_sources": {
            "fundamentals": "SEC EDGAR companyfacts/submissions",
            "prices": "free yfinance fallback or private CSV cache",
            "constituents": {
                "source": constituent_snapshot.source_url,
                "cache_path": str(constituent_cache_path),
                "method": (
                    "Wikipedia current S&P 500 table plus selected changes table, "
                    "normalized into HistoricalConstituentStore.as_of"
                ),
                "quality_caveat": constituent_snapshot.caveat,
                "current_count": len(constituent_snapshot.current),
                "change_events_used": len(constituent_snapshot.changes),
                "as_of_universe_counts": universe_counts,
                "e2e_selected_tickers": list(selected_ciks),
            },
        },
        "survivorship": "light",
        "warning": report["warning"],
        "paid_price_source_status": "STOP_AND_ASK_REQUIRED_BEFORE_B_SEGMENT",
        "observations": [_observation_to_dict(row) for row in observations],
        "report": report,
    }
    json_path = private_dir / f"olympus41b-real-universe-survivor-light-{slug}.json"
    md_path = private_dir / f"olympus41b-real-universe-survivor-light-{slug}.md"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(output, json_path), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "md": str(md_path)}, sort_keys=True))
    return 0


def _observation_to_dict(row: Any) -> dict[str, Any]:
    return {
        "ticker": row.ticker,
        "rebalance_date": row.rebalance_date.isoformat(),
        "price_date": row.price_date.isoformat(),
        "trade_date": row.trade_date.isoformat(),
        "forward_return": row.forward_return,
        "survivorship_status": row.survivorship_status,
        "factors": row.factors,
    }


def _render_markdown(output: dict[str, Any], json_path: Path) -> str:
    report = output["report"]
    return "\n".join(
        [
            "# Olympus #41 Survivor-Light Pipeline",
            "",
            f"- Verdict: {report['verdict']}",
            f"- Survivorship: {report['survivorship']}",
            f"- Warning: {report['warning']}",
            "- Constituents: Wikipedia S&P 500 current table plus selected changes table",
            "- Paid price source: STOP-AND-ASK before implementation",
            f"- JSON: `{json_path}`",
            "",
            "This is pipeline validation only. It is not edge evidence.",
        ]
    )


def _select_tickers_from_asof_universe(
    current: dict[str, Any],
    candidates: list[str],
    *,
    max_tickers: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for ticker in candidates:
        if ticker in seen or ticker not in current or current[ticker].cik is None:
            continue
        seen.add(ticker)
        selected.append(ticker)
        if len(selected) >= max_tickers:
            break
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
