#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import time
from typing import Any

from aegis.btc_price_action_reeval import (
    DEFAULT_PRICE_ACTION_COST_MODEL,
    ExternalContext,
    PriceActionConfig,
    definitive_report_to_dict,
    run_price_action_definitive,
)
from aegis.combo_indicator_search import ComboBar, ComboCostModel
from aegis.private_paths import private_dir_from_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Olympus #49B definitive pooled 4H price-action evaluation."
    )
    parser.add_argument("--private-dir", default=None)
    parser.add_argument("--exchange", default="okx")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--funding-symbols", default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT")
    parser.add_argument("--ethbtc-symbol", default="ETH/BTC")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--since", default="2019-01-01")
    parser.add_argument("--max-bars", type=int, default=12000)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    parser.add_argument("--risk-bootstrap-samples", type=int, default=400)
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
        limit = min(1000, max_bars - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < limit:
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


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    exchange = load_ccxt_exchange(args.exchange)
    since_ms = iso_to_ms(args.since)
    symbols = _csv(args.symbols)
    funding_symbols = _csv(args.funding_symbols)
    funding_by_symbol = {
        symbol: funding_symbols[index] if index < len(funding_symbols) else ""
        for index, symbol in enumerate(symbols)
    }
    bars_by_symbol = {
        symbol: fetch_ohlcv(
            exchange,
            symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            max_bars=args.max_bars,
        )
        for symbol in symbols
    }
    ethbtc = fetch_ohlcv(
        exchange,
        args.ethbtc_symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        max_bars=args.max_bars,
    )
    external_by_symbol: dict[str, ExternalContext] = {}
    for symbol in symbols:
        funding_symbol = funding_by_symbol.get(symbol, "")
        funding = (
            fetch_funding_by_timestamp(
                exchange,
                funding_symbol,
                since_ms=since_ms,
                max_rows=args.max_bars,
            )
            if funding_symbol
            else {}
        )
        external_by_symbol[symbol] = ExternalContext(
            ethbtc=tuple(ethbtc),
            funding_by_timestamp=funding,
        )
    cost_model = ComboCostModel(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        funding_bps_per_period=DEFAULT_PRICE_ACTION_COST_MODEL.funding_bps_per_period,
        funding_label="short funding debited from ccxt funding history when available",
    )
    report = run_price_action_definitive(
        bars_by_symbol,
        external_by_symbol=external_by_symbol,
        config=PriceActionConfig(
            min_trades=30,
            risk_diff_bootstrap_samples=args.risk_bootstrap_samples,
        ),
        cost_model=cost_model,
    )
    payload = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "exchange": args.exchange,
        "symbols": symbols,
        "funding_symbols": funding_by_symbol,
        "timeframe": args.timeframe,
        "since": args.since,
        "bars": {symbol: len(bars) for symbol, bars in bars_by_symbol.items()},
        "ethbtc_bars": len(ethbtc),
        "funding_rows": {
            symbol: len(context.funding_by_timestamp or {})
            for symbol, context in external_by_symbol.items()
        },
        "read_only": True,
        "wallet_or_order_api_used": False,
        "report": definitive_report_to_dict(report),
    }
    output_dir = private_dir_from_cli(args.private_dir, default_task="olympus49b")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"price-action-definitive-"
        f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "status": report.status,
                "alpha_verdict": report.alpha_verdict,
                "risk_verdict": report.risk_verdict,
                "reason": report.reason,
                "candidate_count_n": report.candidate_count_n,
                "pooled_trade_count": report.pooled_trade_count,
                "max_candidate_trade_count": report.max_candidate_trade_count,
                "alpha_fdr_survivors": report.alpha_fdr_survivors,
                "risk_diff_fdr_survivors": report.risk_diff_fdr_survivors,
                "sparse_undeployable": report.sparse_undeployable,
                "bars": {symbol: len(bars) for symbol, bars in bars_by_symbol.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
