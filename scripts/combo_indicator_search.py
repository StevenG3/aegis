#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import time
from collections.abc import Sequence
from typing import Any

from aegis.combo_indicator_search import (
    ComboBar,
    ComboCostModel,
    ComboSearchConfig,
    report_to_dict,
    run_combo_indicator_search,
)
from aegis.private_paths import private_dir_from_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the read-only Olympus #45 combo indicator honesty search."
    )
    parser.add_argument("--private-dir", default=None)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"],
    )
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--max-bars", type=int, default=2500)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--funding-bps-per-period", type=float, default=0.0)
    parser.add_argument("--train-bars", type=int, default=730)
    parser.add_argument("--test-bars", type=int, default=180)
    parser.add_argument("--step-bars", type=int, default=180)
    parser.add_argument("--locked-oos-fraction", type=float, default=0.30)
    parser.add_argument("--top-k-oos", type=int, default=3)
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
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < 1000:
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


def main() -> int:
    args = parse_args()
    exchange = load_ccxt_exchange(args.exchange)
    since_ms = iso_to_ms(args.since)
    symbols = [str(symbol) for symbol in args.symbols]
    bars_by_symbol: dict[str, Sequence[ComboBar]] = {
        symbol: fetch_ohlcv(
            exchange,
            symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            max_bars=args.max_bars,
        )
        for symbol in symbols
    }
    config = ComboSearchConfig(
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        locked_oos_fraction=args.locked_oos_fraction,
        annualization_periods=365,
        top_k_oos=args.top_k_oos,
    )
    cost_model = ComboCostModel(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        funding_bps_per_period=args.funding_bps_per_period,
        funding_label="N/A for spot long-only; perp funding not used",
    )
    report = run_combo_indicator_search(
        bars_by_symbol,
        config=config,
        cost_model=cost_model,
    )
    payload = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "since": args.since,
        "bars": {symbol: len(bars) for symbol, bars in bars_by_symbol.items()},
        "read_only": True,
        "wallet_or_order_api_used": False,
        "report": report_to_dict(report),
    }
    output_dir = private_dir_from_cli(args.private_dir, default_task="olympus45")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"combo-indicator-search-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "status": report.status,
                "verdict": report.verdict,
                "reason": report.reason,
                "bars": payload["bars"],
                "search_space_n": report.search_space_n,
                "raw_is_survivors": report.raw_is_survivors,
                "fdr_is_survivors": report.fdr_is_survivors,
                "locked_oos_survivors": report.locked_oos_survivors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
