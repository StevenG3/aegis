#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
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
from aegis.polymarket_tail_backtest import (
    PolymarketTailCostConfig,
    build_tail_positions,
    positions_to_dict,
    summarize_tail_backtest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a read-only Polymarket tail-aware carry backtest."
    )
    parser.add_argument(
        "--private-dir",
        default="/home/gggqqy/apps/aegis-strategies/incubating/olympus42",
        help="Private output directory outside the public aegis repository.",
    )
    parser.add_argument("--survivor-power-json", default=None)
    parser.add_argument("--max-markets", type=int, default=1_000)
    parser.add_argument("--market-page-size", type=int, default=100)
    parser.add_argument("--trade-page-size", type=int, default=500)
    parser.add_argument("--max-trades-per-market", type=int, default=500)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lower-price", default="0.95")
    parser.add_argument("--upper-price", default="0.985")
    parser.add_argument("--notional-usd", default="100")
    parser.add_argument("--fee-coefficient", default="0.05")
    parser.add_argument("--slippage-bps", default="10")
    parser.add_argument("--gas-usd-per-entry", default="0.02")
    parser.add_argument("--risk-free-return-per-trade", default="0.0002")
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()

    started = time.monotonic()
    private_dir = Path(args.private_dir)
    output_dir = private_dir / "tailaware_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    survivor_gate = _load_survivor_gate(args.survivor_power_json, private_dir)
    if survivor_gate.get("verdict") != "SURVIVOR_GATE_SATISFIED":
        raise SystemExit("STOP: #42C survivor gate is not satisfied")

    client = PolymarketDataApiClient(timeout_seconds=float(args.timeout_seconds))
    raw_markets: list[dict[str, Any]] = []
    parsed_markets = []
    errors: list[dict[str, Any]] = []
    for raw_market in client.iter_closed_markets(
        limit=int(args.market_page_size),
        max_markets=int(args.max_markets),
        sleep_seconds=float(args.sleep_seconds),
    ):
        raw_markets.append(raw_market)
        parsed = parse_closed_market(raw_market)
        if parsed is not None:
            parsed_markets.append(parsed)
    print(
        json.dumps(
            {
                "closed_markets_fetched": len(raw_markets),
                "parsed_markets": len(parsed_markets),
                "survivor_gate": survivor_gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    trades_by_condition: dict[str, list[PolymarketTrade]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _fetch_condition_trades,
                market.condition_id,
                float(args.timeout_seconds),
                int(args.trade_page_size),
                int(args.max_trades_per_market),
                float(args.sleep_seconds),
            ): market.condition_id
            for market in parsed_markets
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
                            "conditions": len(parsed_markets),
                            "errors": len(errors),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    threshold = SurvivorPowerThreshold(min_closed_markets=1_000, min_markets_with_trades=650)
    coverage = analyze_survivor_power_coverage(
        raw_markets,
        trades_by_condition,
        threshold=threshold,
        lower=Decimal(str(args.lower_price)),
        upper=Decimal(str(args.upper_price)),
    )
    costs = PolymarketTailCostConfig(
        notional_usd=Decimal(str(args.notional_usd)),
        fee_coefficient=Decimal(str(args.fee_coefficient)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        gas_usd_per_entry=Decimal(str(args.gas_usd_per_entry)),
    )
    positions = build_tail_positions(
        parsed_markets,
        trades_by_condition,
        lower=Decimal(str(args.lower_price)),
        upper=Decimal(str(args.upper_price)),
        cost_config=costs,
    )
    summary = summarize_tail_backtest(
        positions,
        risk_free_return_per_trade=Decimal(str(args.risk_free_return_per_trade)),
        bootstrap_iterations=int(args.bootstrap_iterations),
    )
    path = _write_payload(
        args=args,
        output_dir=output_dir,
        started=started,
        survivor_gate=survivor_gate,
        coverage=survivor_power_coverage_to_dict(coverage),
        positions=positions_to_dict(positions),
        summary=summary,
        errors=errors,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(path),
                "verdict": summary["verdict"],
                "positions": summary["sample"]["positions"],
                "losses": summary["sample"]["losses"],
            },
            sort_keys=True,
        )
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


def _load_survivor_gate(path_text: str | None, private_dir: Path) -> dict[str, Any]:
    path = Path(path_text) if path_text else _latest_survivor_power(private_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    coverage = data.get("coverage", {})
    return {
        "path": str(path),
        "verdict": coverage.get("verdict"),
        "closed_markets_scanned": coverage.get("closed_markets_scanned"),
        "markets_with_trades": coverage.get("markets_with_trades"),
        "high_price_outcomes": coverage.get("high_price_outcomes"),
        "high_price_winning_outcomes": coverage.get("high_price_winning_outcomes"),
        "high_price_losing_outcomes": coverage.get("high_price_losing_outcomes"),
        "losing_samples": len(coverage.get("losing_samples") or []),
    }


def _latest_survivor_power(private_dir: Path) -> Path:
    paths = sorted((private_dir / "survivor_power").glob("polymarket-survivor-power-*.json"))
    if not paths:
        raise SystemExit("STOP: #42C survivor-power private JSON is missing")
    return paths[-1]


def _write_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    started: float,
    survivor_gate: dict[str, Any],
    coverage: dict[str, Any],
    positions: list[dict[str, Any]],
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> Path:
    payload = {
        "briefing": "CODEX_OLYMPUS_42D_POLYMARKET_TAILAWARE_BACKTEST",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "runtime_seconds": round(time.monotonic() - started, 3),
        "thesis": (
            "Buy 0.95-0.985 near-certain outcomes in a survivor-safe closed-market sample "
            "and hold to binary settlement; test whether net carry clears costs, tail losses, "
            "and a risk-free carry benchmark."
        ),
        "null_hypothesis": (
            "Net carry is not significantly positive, is consumed by rare losses/costs, "
            "or tail frequency is not measurable from the available loss events."
        ),
        "survivor_gate": survivor_gate,
        "execution_bounds": {
            "max_markets": int(args.max_markets),
            "market_page_size": int(args.market_page_size),
            "trade_page_size": int(args.trade_page_size),
            "max_trades_per_market": int(args.max_trades_per_market),
            "timeout_seconds": float(args.timeout_seconds),
            "sleep_seconds": float(args.sleep_seconds),
            "workers": int(args.workers),
            "price_band": [str(args.lower_price), str(args.upper_price)],
        },
        "cost_assumptions": {
            "notional_usd": str(args.notional_usd),
            "fee_coefficient": str(args.fee_coefficient),
            "fee_source": "Polymarket docs formula: theta * contracts * p * (1-p)",
            "slippage_bps": str(args.slippage_bps),
            "gas_usd_per_entry": str(args.gas_usd_per_entry),
            "funding": "not_applicable",
            "binary_hold_to_settlement": True,
        },
        "coverage_recomputed": coverage,
        "sample_alignment": _sample_alignment(survivor_gate, coverage),
        "summary": summary,
        "positions": positions,
        "trade_fetch_errors": errors,
        "safety": {
            "read_only": True,
            "wallet_or_order_access": False,
            "funds_access": False,
            "live_trading": False,
            "zero_risk_claim": "explicitly_rejected",
        },
    }
    slug = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    path = output_dir / f"polymarket-tailaware-backtest-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sample_alignment(
    survivor_gate: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "high_price_outcomes",
        "high_price_winning_outcomes",
        "high_price_losing_outcomes",
    )
    differences = {
        key: {
            "survivor_gate": survivor_gate.get(key),
            "recomputed": coverage.get(key),
        }
        for key in keys
        if survivor_gate.get(key) != coverage.get(key)
    }
    return {
        "matches_survivor_gate_counts": not differences,
        "differences": differences,
        "note": (
            "#42C private JSON stores aggregate counts and losing samples, not every winner "
            "row. This runner re-fetches closed markets/trades to build a full ledger; a "
            "moving closed-market window can differ from #42C counts."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
