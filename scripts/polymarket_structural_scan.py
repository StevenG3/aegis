from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from aegis.polymarket_onchain import PolymarketDataApiClient
from aegis.polymarket_structural_scan import (
    StructuralCostConfig,
    StructuralScanResult,
    clob_token_ids,
    scan_logic_subset_pairs,
    scan_neg_risk_groups,
    structural_scan_to_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Polymarket structural scan")
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--max-markets", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=12)
    parser.add_argument("--target-size", default="5")
    parser.add_argument("--fee-rate", default="0")
    parser.add_argument("--gas-usdc", default="0.02")
    parser.add_argument("--min-net-edge", default="0.001")
    args = parser.parse_args()

    client = PolymarketDataApiClient(timeout_seconds=args.timeout_seconds)
    raw_markets = list(
        client.iter_closed_markets(
            limit=100,
            max_markets=args.max_markets,
            order="volume",
            ascending=False,
            closed=False,
        )
    )
    active_markets = [market for market in raw_markets if market.get("closed") is False]
    token_ids = sorted({token for market in active_markets for token in clob_token_ids(market)})
    books_by_token: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(client.get_order_book, token_id): token_id for token_id in token_ids
        }
        for future in as_completed(futures):
            token_id = futures[future]
            try:
                books_by_token[token_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{token_id}:{exc.__class__.__name__}")

    costs = StructuralCostConfig(
        fee_rate=Decimal(args.fee_rate),
        gas_usdc=Decimal(args.gas_usdc),
        min_trade_size=Decimal(args.target_size),
        min_net_edge=Decimal(args.min_net_edge),
    )
    target_size = Decimal(args.target_size)
    neg_groups, neg_groups_with_books, neg_candidates = scan_neg_risk_groups(
        active_markets,
        books_by_token,
        costs=costs,
        target_size=target_size,
    )
    logic_pairs, logic_candidates = scan_logic_subset_pairs(
        active_markets,
        books_by_token,
        costs=costs,
        target_size=target_size,
    )
    result = StructuralScanResult(
        neg_risk_groups_scanned=neg_groups,
        neg_risk_groups_with_books=neg_groups_with_books,
        neg_risk_candidates=neg_candidates,
        logic_pairs_evaluated=logic_pairs,
        logic_candidates=logic_candidates,
        orderbook_errors=tuple(errors),
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_bounds": {
            "max_markets": args.max_markets,
            "active_markets": len(active_markets),
            "token_books_requested": len(token_ids),
            "token_books_loaded": len(books_by_token),
            "workers": args.workers,
            "target_size": args.target_size,
            "fee_rate": args.fee_rate,
            "gas_usdc": args.gas_usdc,
            "min_net_edge": args.min_net_edge,
        },
        "result": structural_scan_to_dict(result),
        "read_only": True,
        "wallet_order_funds_connected": False,
    }
    out_dir = Path(args.private_dir) / "structural_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polymarket-structural-scan-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "verdict": result.verdict}, sort_keys=True))


if __name__ == "__main__":
    main()
