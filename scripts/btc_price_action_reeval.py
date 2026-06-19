#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import time
from pathlib import Path
from typing import Any

from aegis.btc_price_action_reeval import (
    DEFAULT_PRICE_ACTION_COST_MODEL,
    ExternalContext,
    PriceActionConfig,
    report_to_dict,
    run_btc_price_action_reeval,
)
from aegis.combo_indicator_search import ComboBar, ComboCostModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Olympus #49 BTC 4H price-action reevaluation."
    )
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--exchange", default="okx")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--funding-symbol", default="BTC/USDT:USDT")
    parser.add_argument("--ethbtc-symbol", default="ETH/BTC")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--since", default="2023-05-01")
    parser.add_argument("--max-bars", type=int, default=7000)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    return parser.parse_args()


def iso_to_ms(value: str) -> int:
    parsed = dt.datetime.fromisoformat(value).replace(tzinfo=dt.UTC)
    return int(parsed.timestamp() * 1000)


def load_ccxt_exchange(exchange_id: str) -> Any:
    ccxt = importlib.import_module("ccxt")
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    *,
    timeframe: str,
    since_ms: int,
    max_bars: int,
) -> list[ComboBar]:
    rows: list[list[float]] = []
    cursor = since_ms
    while len(rows) < max_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=300)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < 300:
            break
    return [
        ComboBar(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows[:max_bars]
    ]


def fetch_funding_by_timestamp(
    exchange: Any,
    symbol: str,
    *,
    since_ms: int,
    max_rows: int,
) -> dict[int, float]:
    if not getattr(exchange, "has", {}).get("fetchFundingRateHistory"):
        return {}
    rows: list[dict[str, Any]] = []
    cursor = since_ms
    while len(rows) < max_rows:
        try:
            batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=100)
        except Exception:
            return {}
        if not batch:
            break
        rows.extend(batch)
        timestamp = int(batch[-1].get("timestamp") or 0)
        next_cursor = timestamp + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < 100:
            break
    return {
        int(row["timestamp"]): float(row["fundingRate"])
        for row in rows[:max_rows]
        if row.get("timestamp") is not None and row.get("fundingRate") is not None
    }


def main() -> int:
    args = parse_args()
    exchange = load_ccxt_exchange(args.exchange)
    since_ms = iso_to_ms(args.since)
    bars = fetch_ohlcv(
        exchange,
        args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        max_bars=args.max_bars,
    )
    ethbtc = fetch_ohlcv(
        exchange,
        args.ethbtc_symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        max_bars=args.max_bars,
    )
    funding = fetch_funding_by_timestamp(
        exchange,
        args.funding_symbol,
        since_ms=since_ms,
        max_rows=args.max_bars,
    )
    cost_model = ComboCostModel(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        funding_bps_per_period=DEFAULT_PRICE_ACTION_COST_MODEL.funding_bps_per_period,
        funding_label="short funding debited from ccxt funding history when available",
    )
    external = ExternalContext(ethbtc=tuple(ethbtc), funding_by_timestamp=funding)
    report = run_btc_price_action_reeval(
        bars,
        external=external,
        config=PriceActionConfig(),
        cost_model=cost_model,
    )
    payload = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "exchange": args.exchange,
        "symbol": args.symbol,
        "funding_symbol": args.funding_symbol,
        "timeframe": args.timeframe,
        "since": args.since,
        "bars": len(bars),
        "ethbtc_bars": len(ethbtc),
        "funding_rows": len(funding),
        "read_only": True,
        "wallet_or_order_api_used": False,
        "report": report_to_dict(report),
    }
    output_dir = Path(args.private_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"btc-price-action-reeval-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "status": report.status,
                "verdict": report.verdict,
                "reason": report.reason,
                "candidate_count_n": report.candidate_count_n,
                "raw_is_survivors": report.raw_is_survivors,
                "alpha_fdr_survivors": report.alpha_fdr_survivors,
                "risk_diff_fdr_survivors": report.risk_diff_fdr_survivors,
                "alpha_edge_count": report.alpha_edge_count,
                "risk_improved_count": report.risk_improved_count,
                "insufficient_count": report.insufficient_count,
                "bars": len(bars),
                "ethbtc_bars": len(ethbtc),
                "funding_rows": len(funding),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
