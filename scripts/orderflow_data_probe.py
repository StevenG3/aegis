from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_EXCHANGES = ("binance", "okx", "bybit")
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_SWAP_SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "OLYMPUS_EVIDENCE_DIR",
        str(Path(__file__).resolve().parents[2] / "aegis-strategies" / "incubating"),
    )
)


class ExchangeLike(Protocol):
    has: dict[str, object]
    rateLimit: int

    def fetch_trades(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        ...

    def fetch_order_book(self, symbol: str, limit: int | None = None) -> dict[str, Any]:
        ...

    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        ...


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe public order-flow/microstructure data availability."
    )
    parser.add_argument("--exchanges", default=",".join(DEFAULT_EXCHANGES))
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--swap-symbol", default=DEFAULT_SWAP_SYMBOL)
    parser.add_argument("--trade-limit", type=int, default=20)
    parser.add_argument("--book-limit", type=int, default=50)
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    ccxt_module = importlib.import_module("ccxt")
    report = build_report(
        ccxt_module,
        exchanges=_csv(args.exchanges),
        symbol=args.symbol,
        swap_symbol=args.swap_symbol,
        trade_limit=args.trade_limit,
        book_limit=args.book_limit,
        since_hours=args.since_hours,
    )
    if not args.no_write:
        report["written_files"] = write_report(report, Path(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def build_report(
    ccxt_module: object,
    *,
    exchanges: list[str],
    symbol: str,
    swap_symbol: str,
    trade_limit: int = 20,
    book_limit: int = 50,
    since_hours: int = 24,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for exchange_id in exchanges:
        rows.append(
            _probe_exchange(
                ccxt_module,
                exchange_id=exchange_id,
                symbol=symbol,
                swap_symbol=swap_symbol,
                trade_limit=trade_limit,
                book_limit=book_limit,
                since_hours=since_hours,
            )
        )
    report = {
        "generated_at": generated_at.isoformat(),
        "config": {
            "exchanges": exchanges,
            "symbol": symbol,
            "swap_symbol": swap_symbol,
            "trade_limit": trade_limit,
            "book_limit": book_limit,
            "since_hours": since_hours,
        },
        "exchanges": rows,
        "conclusion": _conclusion(rows),
        "disclaimer": "read-only public API probe; no API keys, no orders, no feature build",
    }
    report["human_readable"] = _markdown(report)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"orderflow-data-probe-{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(str(report["human_readable"]), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _probe_exchange(
    ccxt_module: object,
    *,
    exchange_id: str,
    symbol: str,
    swap_symbol: str,
    trade_limit: int,
    book_limit: int,
    since_hours: int,
) -> dict[str, Any]:
    try:
        factory = getattr(ccxt_module, exchange_id)
        exchange = cast(ExchangeLike, factory({"enableRateLimit": True, "timeout": 10_000}))
    except Exception as exc:  # noqa: BLE001
        return {"exchange": exchange_id, "status": "ERROR", "error": str(exc)}
    return {
        "exchange": exchange_id,
        "status": "OK",
        "rate_limit_ms": getattr(exchange, "rateLimit", None),
        "ccxt_has": {
            "fetchTrades": bool(exchange.has.get("fetchTrades")),
            "fetchOrderBook": bool(exchange.has.get("fetchOrderBook")),
            "fetchFundingRateHistory": bool(exchange.has.get("fetchFundingRateHistory")),
        },
        "trades": _probe_trades(exchange, symbol, trade_limit, since_hours),
        "order_book": _probe_order_book(exchange, symbol, book_limit),
        "funding": _probe_funding(exchange, swap_symbol),
    }


def _probe_trades(
    exchange: ExchangeLike, symbol: str, limit: int, since_hours: int
) -> dict[str, Any]:
    if not exchange.has.get("fetchTrades"):
        return {"availability": "unavailable", "reason": "ccxt.has.fetchTrades is false"}
    result: dict[str, Any] = {"availability": "unknown"}
    try:
        recent = exchange.fetch_trades(symbol, limit=limit)
        result |= _trade_sample("recent", recent, limit)
    except Exception as exc:  # noqa: BLE001
        result |= {"availability": "unavailable", "error": str(exc)}
        return result

    since = int((datetime.now(UTC) - timedelta(hours=since_hours)).timestamp() * 1000)
    try:
        historical = exchange.fetch_trades(symbol, since=since, limit=min(limit, 10))
        result["since_probe"] = _trade_sample("since", historical, min(limit, 10))
        result["history_depth_conclusion"] = _trade_history_conclusion(historical, since)
    except Exception as exc:  # noqa: BLE001
        result["since_probe"] = {"availability": "error", "error": str(exc)}
        result["history_depth_conclusion"] = "recent_only_or_since_not_supported"
    result["availability"] = "available" if result.get("sample_count", 0) else "unavailable"
    return result


def _trade_history_conclusion(trades: list[dict[str, Any]], requested_since: int) -> str:
    timestamps = [
        int(trade["timestamp"])
        for trade in trades
        if isinstance(trade.get("timestamp"), int | float)
    ]
    if not timestamps:
        return "recent_only_or_since_not_supported"
    five_minutes_ms = 5 * 60 * 1000
    if min(timestamps) <= requested_since + five_minutes_ms:
        return "partial_recent_history_since_supported"
    return "recent_only_or_since_ignored"


def _trade_sample(kind: str, trades: list[dict[str, Any]], requested_limit: int) -> dict[str, Any]:
    side_values = sorted(
        {
            str(trade.get("side"))
            for trade in trades
            if trade.get("side") is not None
        }
    )
    timestamps = [
        int(trade["timestamp"])
        for trade in trades
        if isinstance(trade.get("timestamp"), int | float)
    ]
    first = trades[0] if trades else {}
    return {
        "probe": kind,
        "requested_limit": requested_limit,
        "sample_count": len(trades),
        "first_timestamp": _timestamp(min(timestamps)) if timestamps else None,
        "last_timestamp": _timestamp(max(timestamps)) if timestamps else None,
        "fields": sorted(str(key) for key in first),
        "has_taker_side": bool(side_values),
        "side_values": side_values,
        "sample": _compact_trade(first) if first else None,
    }


def _probe_order_book(exchange: ExchangeLike, symbol: str, limit: int) -> dict[str, Any]:
    if not exchange.has.get("fetchOrderBook"):
        return {"availability": "unavailable", "reason": "ccxt.has.fetchOrderBook is false"}
    try:
        book = exchange.fetch_order_book(symbol, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"availability": "unavailable", "error": str(exc)}
    bids = book.get("bids") if isinstance(book, dict) else None
    asks = book.get("asks") if isinstance(book, dict) else None
    bid_rows = bids if isinstance(bids, list) else []
    ask_rows = asks if isinstance(asks, list) else []
    return {
        "availability": "available" if bid_rows or ask_rows else "unavailable",
        "requested_limit": limit,
        "bid_levels": len(bid_rows),
        "ask_levels": len(ask_rows),
        "top_bid": _price_size(bid_rows[0]) if bid_rows else None,
        "top_ask": _price_size(ask_rows[0]) if ask_rows else None,
        "snapshot_only": True,
        "historical_order_book": False,
        "history_depth_conclusion": "ccxt unified public API exposes snapshot, not history",
    }


def _probe_funding(exchange: ExchangeLike, swap_symbol: str) -> dict[str, Any]:
    if not exchange.has.get("fetchFundingRateHistory"):
        return {
            "availability": "unavailable",
            "reason": "ccxt.has.fetchFundingRateHistory is false",
        }
    since = int((datetime.now(UTC) - timedelta(days=7)).timestamp() * 1000)
    try:
        rows = exchange.fetch_funding_rate_history(swap_symbol, since=since, limit=5)
    except Exception as exc:  # noqa: BLE001
        return {"availability": "unavailable", "error": str(exc)}
    return {
        "availability": "available" if rows else "unavailable",
        "sample_count": len(rows),
        "first_timestamp": _timestamp_from_rows(rows, min),
        "last_timestamp": _timestamp_from_rows(rows, max),
        "fields": sorted(str(key) for key in rows[0]) if rows else [],
        "history_depth_conclusion": (
            "funding history available through ccxt when exchange supports it"
        ),
    }


def _conclusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trade_available = [row["exchange"] for row in rows if _available(row, "trades")]
    side_available = [
        row["exchange"]
        for row in rows
        if isinstance(row.get("trades"), dict) and row["trades"].get("has_taker_side")
    ]
    book_available = [row["exchange"] for row in rows if _available(row, "order_book")]
    funding_available = [row["exchange"] for row in rows if _available(row, "funding")]
    return {
        "footprint_feasibility": "partial",
        "volume_profile_feasibility": "partial",
        "historical_backtest_feasibility": "limited",
        "trade_available_exchanges": trade_available,
        "taker_side_available_exchanges": side_available,
        "order_book_available_exchanges": book_available,
        "funding_available_exchanges": funding_available,
        "summary": (
            "Public APIs can provide recent trades and live order-book snapshots, "
            "but not full historical order-book depth. Footprint/volume-profile "
            "features are feasible for forward collection or short recent windows, "
            "but weak for historical backtest funnels unless data is archived."
        ),
    }


def _available(row: dict[str, Any], section: str) -> bool:
    raw = row.get(section)
    return isinstance(raw, dict) and raw.get("availability") == "available"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Order-Flow Data Probe",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Conclusion",
        "",
        str(cast(dict[str, Any], report["conclusion"])["summary"]),
        "",
        "## Exchange Matrix",
        "",
        (
            "| Exchange | Trades | Side | Since probe | Book | Book history | "
            "Funding history |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in cast(list[dict[str, Any]], report["exchanges"]):
        trades = cast(dict[str, Any], row.get("trades", {}))
        book = cast(dict[str, Any], row.get("order_book", {}))
        funding = cast(dict[str, Any], row.get("funding", {}))
        lines.append(
            f"| {row.get('exchange')} | {trades.get('availability')} "
            f"({trades.get('sample_count', 0)}) | {trades.get('has_taker_side')} | "
            f"{trades.get('history_depth_conclusion')} | "
            f"{book.get('availability')} ({book.get('bid_levels', 0)}x"
            f"{book.get('ask_levels', 0)}) | "
            f"{book.get('history_depth_conclusion')} | "
            f"{funding.get('availability')} ({funding.get('sample_count', 0)}) |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Read-only public API calls.",
            "- No API keys.",
            "- No order placement or account access.",
            "- No microstructure feature implementation in this task.",
        ]
    )
    return "\n".join(lines)


def _compact_trade(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(cast(int | float, trade["timestamp"]))
        if isinstance(trade.get("timestamp"), int | float)
        else None,
        "price": trade.get("price"),
        "amount": trade.get("amount"),
        "side": trade.get("side"),
        "takerOrMaker": trade.get("takerOrMaker"),
    }


def _timestamp_from_rows(
    rows: list[dict[str, Any]], selector: Any
) -> str | None:
    timestamps = [
        int(row["timestamp"])
        for row in rows
        if isinstance(row.get("timestamp"), int | float)
    ]
    return _timestamp(selector(timestamps)) if timestamps else None


def _timestamp(value: int | float) -> str:
    return datetime.fromtimestamp(float(value) / 1000, tz=UTC).isoformat()


def _price_size(row: object) -> dict[str, float] | None:
    if not isinstance(row, list | tuple) or len(row) < 2:
        return None
    return {"price": float(row[0]), "size": float(row[1])}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
