#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import time
from pathlib import Path
from typing import Any

from aegis.backtest_core import CostModel
from aegis.private_paths import private_dir_from_cli
from aegis.volume_divergence_rr import (
    VolumeDivergenceBar,
    VolumeDivergenceConfig,
    result_to_dict,
    run_volume_divergence_rr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the 4h lower-volume lower-low / higher-high divergence 1:3 RR rule."
    )
    parser.add_argument("--private-dir", default=None)
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
    )
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--max-bars", type=int, default=5000)
    parser.add_argument("--lookback-bars", type=int, default=20)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--require-rsi-divergence", action="store_true")
    parser.add_argument("--macd-fast-period", type=int, default=12)
    parser.add_argument("--macd-slow-period", type=int, default=26)
    parser.add_argument("--macd-signal-period", type=int, default=9)
    parser.add_argument("--require-macd-histogram-divergence", action="store_true")
    parser.add_argument("--require-liquidity-sweep", action="store_true")
    parser.add_argument("--require-choch-confirmation", action="store_true")
    parser.add_argument("--choch-lookback-bars", type=int, default=5)
    parser.add_argument("--reward-risk", type=float, default=3.0)
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=None,
        help="Optional fixed account risk per trade, expressed in percent, e.g. 0.5 or 1.0.",
    )
    parser.add_argument(
        "--max-position-notional",
        type=float,
        default=None,
        help="Optional max notional as a fraction of account equity, e.g. 1.0 for no leverage.",
    )
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--include-trades", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.long_only and args.short_only:
        raise SystemExit("--long-only and --short-only cannot both be set")
    exchange = _load_exchange(args.exchange)
    since_ms = _iso_to_ms(args.since)
    config = VolumeDivergenceConfig(
        lookback_bars=args.lookback_bars,
        rsi_period=args.rsi_period,
        require_rsi_divergence=args.require_rsi_divergence,
        macd_fast_period=args.macd_fast_period,
        macd_slow_period=args.macd_slow_period,
        macd_signal_period=args.macd_signal_period,
        require_macd_histogram_divergence=args.require_macd_histogram_divergence,
        require_liquidity_sweep=args.require_liquidity_sweep,
        require_choch_confirmation=args.require_choch_confirmation,
        choch_lookback_bars=args.choch_lookback_bars,
        reward_risk=args.reward_risk,
        risk_per_trade_fraction=(
            args.risk_per_trade_pct / 100.0 if args.risk_per_trade_pct is not None else None
        ),
        max_position_notional_fraction=args.max_position_notional,
        allow_long=not args.short_only,
        allow_short=not args.long_only,
        annualization_periods=_annualization_periods(args.timeframe),
    )
    cost_model = CostModel(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        funding_bps_per_period=0.0,
        funding_label=(
            "not modeled in this first-pass OHLCV-only divergence replay; "
            "use exchange funding history before treating perp shorts as deployable"
        ),
    )
    results: dict[str, object] = {}
    coverage: dict[str, object] = {}
    for symbol in args.symbols:
        bars = _fetch_ohlcv(
            exchange,
            symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            max_bars=args.max_bars,
        )
        if not bars:
            coverage[symbol] = {"bars": 0, "status": "no_data"}
            continue
        result = run_volume_divergence_rr(
            bars,
            symbol=symbol,
            config=config,
            cost_model=cost_model,
        )
        results[symbol] = result_to_dict(result, include_trades=args.include_trades)
        coverage[symbol] = {
            "bars": len(bars),
            "first_ts": _ms_to_iso(bars[0].timestamp),
            "last_ts": _ms_to_iso(bars[-1].timestamp),
            "trades": len(result.trades),
        }

    run_at = dt.datetime.now(dt.timezone.utc)
    payload = {
        "run_at": run_at.isoformat(),
        "read_only": True,
        "wallet_or_order_api_used": False,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "since": args.since,
        "symbols": args.symbols,
        "coverage": coverage,
        "strategy_definition": {
            "long": (
                "current 4h bar makes a new lookback low, low is below previous recorded low, "
                "and current volume is below previous low's volume; enter next bar open"
            ),
            "short": (
                "mirror rule: current 4h bar makes a new lookback high, high is above previous "
                "recorded high, and current volume is below previous high's volume; enter next bar open"
            ),
            "stop": "signal bar low for longs, signal bar high for shorts",
            "take_profit": f"{args.reward_risk:g}:1 fixed reward/risk",
            "position_sizing": (
                f"fixed account risk {args.risk_per_trade_pct:g}% per trade"
                if args.risk_per_trade_pct is not None
                else "full price exposure; no fixed account-risk sizing"
            ),
            "max_position_notional": (
                f"{args.max_position_notional:g}x account equity"
                if args.max_position_notional is not None
                else "not capped"
            ),
            "rsi_filter": (
                f"RSI({args.rsi_period}) price divergence required"
                if args.require_rsi_divergence
                else "not required"
            ),
            "macd_histogram_filter": (
                f"MACD({args.macd_fast_period},{args.macd_slow_period},"
                f"{args.macd_signal_period}) histogram divergence required"
                if args.require_macd_histogram_divergence
                else "not required"
            ),
            "liquidity_sweep_filter": (
                "signal bar must sweep the previous extreme and close back through it"
                if args.require_liquidity_sweep
                else "not required"
            ),
            "choch_confirmation": (
                f"after signal, require close through prior {args.choch_lookback_bars}-bar structure, "
                "then enter next bar open"
                if args.require_choch_confirmation
                else "not required"
            ),
            "same_bar_collision": "stop first",
            "funding": cost_model.funding_label,
        },
        "results": results,
    }
    output_dir = private_dir_from_cli(args.private_dir, default_task="manual_volume_divergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    side_mode = "long-only" if args.long_only else "short-only" if args.short_only else "long-short"
    rsi_mode = "rsi" if args.require_rsi_divergence else "no-rsi"
    macd_mode = "macd" if args.require_macd_histogram_divergence else "no-macd"
    sweep_mode = "sweep" if args.require_liquidity_sweep else "no-sweep"
    choch_mode = "choch" if args.require_choch_confirmation else "no-choch"
    risk_mode = (
        f"risk-{str(args.risk_per_trade_pct).replace('.', 'p')}pct"
        if args.risk_per_trade_pct is not None
        else "full-risk"
    )
    cap_mode = (
        f"cap-{str(args.max_position_notional).replace('.', 'p')}x"
        if args.max_position_notional is not None
        else "uncapped"
    )
    rr_mode = f"rr-{str(args.reward_risk).replace('.', 'p')}"
    mode = f"{side_mode}-{rsi_mode}-{macd_mode}-{sweep_mode}-{choch_mode}-{rr_mode}-{risk_mode}-{cap_mode}"
    output_path = output_dir / f"volume-divergence-rr-{mode}-{run_at:%Y%m%dT%H%M%SZ}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "coverage": coverage,
                "summary": _summary(results),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_exchange(exchange_id: str) -> Any:
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, exchange_id)
    exchange = factory({"enableRateLimit": True, "timeout": 20_000})
    exchange.load_markets()
    return exchange


def _fetch_ohlcv(
    exchange: Any,
    symbol: str,
    *,
    timeframe: str,
    since_ms: int,
    max_bars: int,
) -> list[VolumeDivergenceBar]:
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
        VolumeDivergenceBar(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows[:max_bars]
    ]


def _summary(results: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for symbol, result in results.items():
        if not isinstance(result, dict):
            continue
        metrics = result.get("metrics", {})
        scorecard = result.get("trade_scorecard", {})
        benchmark = result.get("benchmark_metrics", {})
        if isinstance(metrics, dict) and isinstance(scorecard, dict) and isinstance(benchmark, dict):
            summary[symbol] = {
                "annualized_return": metrics.get("annualized_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe": metrics.get("sharpe"),
                "total_return": metrics.get("total_return"),
                "trades": scorecard.get("total_trades"),
                "win_rate": scorecard.get("win_rate"),
                "profit_factor": scorecard.get("profit_factor"),
                "expectancy_per_trade": scorecard.get("expectancy_per_trade"),
                "buy_hold_annualized_return": benchmark.get("annualized_return"),
                "buy_hold_max_drawdown": benchmark.get("max_drawdown"),
            }
    return summary


def _iso_to_ms(value: str) -> int:
    parsed = dt.datetime.fromisoformat(value).replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def _ms_to_iso(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).isoformat()


def _annualization_periods(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized.endswith("h"):
        hours = int(normalized[:-1])
        return int(24 / hours * 365)
    if normalized.endswith("d"):
        days = int(normalized[:-1] or "1")
        return int(365 / days)
    return 365 * 6


if __name__ == "__main__":
    raise SystemExit(main())
