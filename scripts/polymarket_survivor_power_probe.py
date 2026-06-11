#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.polymarket_onchain import (
    PolymarketDataApiClient,
    PolymarketTrade,
    SurvivorPowerThreshold,
    analyze_survivor_power_coverage,
    parse_closed_market,
    parse_trade,
    survivor_power_coverage_to_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a read-only Polymarket survivor-power coverage probe."
    )
    parser.add_argument(
        "--private-dir",
        default="/home/gggqqy/apps/aegis-strategies/incubating/olympus42",
        help="Private output directory outside the public aegis repository.",
    )
    parser.add_argument("--max-markets", type=int, default=1_000)
    parser.add_argument("--market-page-size", type=int, default=100)
    parser.add_argument("--trade-page-size", type=int, default=500)
    parser.add_argument("--max-trades-per-market", type=int, default=5_000)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-closed-markets", type=int, default=1_000)
    parser.add_argument("--min-markets-with-trades", type=int, default=650)
    args = parser.parse_args()

    started = time.monotonic()
    private_dir = Path(args.private_dir)
    output_dir = private_dir / "survivor_power"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = PolymarketDataApiClient(timeout_seconds=float(args.timeout_seconds))
    raw_markets: list[dict[str, Any]] = []
    parsed_condition_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    for raw_market in client.iter_closed_markets(
        limit=int(args.market_page_size),
        max_markets=int(args.max_markets),
        sleep_seconds=float(args.sleep_seconds),
    ):
        raw_markets.append(raw_market)
        parsed = parse_closed_market(raw_market)
        if parsed is not None:
            parsed_condition_ids.append(parsed.condition_id)
    print(
        json.dumps(
            {
                "closed_markets_fetched": len(raw_markets),
                "parsed_conditions": len(parsed_condition_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    trades_by_condition: dict[str, list[PolymarketTrade]] = {}
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_condition_trades,
                condition_id,
                float(args.timeout_seconds),
                int(args.trade_page_size),
                int(args.max_trades_per_market),
                float(args.sleep_seconds),
            ): condition_id
            for condition_id in parsed_condition_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            condition_id = futures[future]
            try:
                trades_by_condition[condition_id] = future.result()
            except Exception as exc:  # pragma: no cover - private runner resilience
                errors.append({"condition_id": condition_id, "error": repr(exc)})
            if index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "conditions": len(parsed_condition_ids),
                            "errors": len(errors),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    threshold = SurvivorPowerThreshold(
        min_closed_markets=int(args.min_closed_markets),
        min_markets_with_trades=int(args.min_markets_with_trades),
        target_closed_window_days=None,
    )
    coverage = analyze_survivor_power_coverage(
        raw_markets,
        trades_by_condition,
        threshold=threshold,
    )
    _write_payload(
        args=args,
        private_dir=private_dir,
        started=started,
        coverage=coverage,
        errors=errors,
    )
    return 0


def _fetch_condition_trades(
    condition_id: str,
    timeout_seconds: float,
    trade_page_size: int,
    max_trades_per_market: int,
    sleep_seconds: float,
) -> list[PolymarketTrade]:
    client = PolymarketDataApiClient(timeout_seconds=timeout_seconds)
    raw_trades = list(
        client.iter_trades(
            condition_id,
            limit=trade_page_size,
            max_trades=max_trades_per_market,
            sleep_seconds=sleep_seconds,
        )
    )
    return [trade for row in raw_trades if (trade := parse_trade(row)) is not None]


def _write_payload(
    *,
    args: argparse.Namespace,
    private_dir: Path,
    started: float,
    coverage: Any,
    errors: list[dict[str, Any]],
) -> None:
    output_dir = private_dir / "survivor_power"
    payload = {
        "briefing": "CODEX_OLYMPUS_42C_POLYMARKET_SURVIVOR_POWER",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "runtime_seconds": round(time.monotonic() - started, 3),
        "predeclared_power_threshold": {
            "min_closed_markets": coverage.threshold.min_closed_markets,
            "min_markets_with_trades": coverage.threshold.min_markets_with_trades,
            "rationale": (
                "Far above #42B's 80 recent-market probe; sufficient for a data-reachability "
                "verdict, not for a zero-tail or edge claim."
            ),
        },
        "execution_bounds": {
            "max_markets": int(args.max_markets),
            "market_page_size": int(args.market_page_size),
            "trade_page_size": int(args.trade_page_size),
            "max_trades_per_market": int(args.max_trades_per_market),
            "timeout_seconds": float(args.timeout_seconds),
            "sleep_seconds": float(args.sleep_seconds),
            "workers": int(args.workers),
        },
        "data_sources": {
            "gamma": "closed markets paginated by closedTime descending",
            "data_api": "per-condition paginated trades",
            "goldsky": (
                "public endpoint probed separately; usable as archive signal but old subgraph "
                "is incomplete after Polymarket v2 migration"
            ),
        },
        "coverage": survivor_power_coverage_to_dict(coverage),
        "trade_fetch_errors": errors,
        "no_backtest_run": True,
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "funds_access": False,
            "zero_risk_claim": "explicitly_rejected",
        },
    }
    slug = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    path = output_dir / f"polymarket-survivor-power-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "path": str(path), "verdict": coverage.verdict}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
