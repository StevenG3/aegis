#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from aegis.polymarket_onchain import PolymarketDataApiClient, parse_closed_market
from aegis.polymarket_structural_scan import (
    OrderBookLevel,
    clob_token_ids,
    parse_order_book_levels,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus61"
DEFAULT_USER_AGENT = "aegis-polymarket-btc-5m-orderbook-capture/0.1 read-only"


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    interval_seconds: float
    duration_seconds: float | None
    max_iterations: int | None
    market_page_size: int
    max_markets: int
    timeout_seconds: float
    sleep_seconds: float
    depth_levels: int


@dataclass(frozen=True)
class ActiveBtcMarket:
    condition_id: str
    slug: str
    title: str
    end_ts: int | None
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]


@dataclass(frozen=True)
class BookSideSummary:
    top_price: Decimal | None
    top_size: Decimal | None
    depth_usd: Decimal
    levels: tuple[dict[str, str], ...]


def main() -> int:
    args = _parse_args()
    config = _config_from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    client = PolymarketDataApiClient(
        user_agent=DEFAULT_USER_AGENT,
        timeout_seconds=config.timeout_seconds,
    )
    summary = capture_order_books(client, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def capture_order_books(
    client: PolymarketDataApiClient,
    config: CaptureConfig,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    deadline = (
        None
        if config.duration_seconds is None
        else time.monotonic() + max(0.0, config.duration_seconds)
    )
    iterations = 0
    records_written = 0
    errors: list[dict[str, str]] = []
    last_file: Path | None = None

    while True:
        captured_at = datetime.now(UTC)
        markets = list(fetch_active_btc_5m_markets(client, config))
        for market in markets:
            for outcome, token_id in zip(market.outcomes, market.token_ids, strict=False):
                try:
                    book = client.get_order_book(token_id)
                    record = build_snapshot_record(
                        market,
                        outcome=outcome,
                        token_id=token_id,
                        raw_book=book,
                        captured_at=captured_at,
                        depth_levels=config.depth_levels,
                    )
                    last_file = append_jsonl(config.output_dir, captured_at, record)
                    records_written += 1
                    if config.sleep_seconds > 0:
                        time.sleep(config.sleep_seconds)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "condition_id": market.condition_id,
                            "token_id": token_id,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )

        iterations += 1
        if config.max_iterations is not None and iterations >= config.max_iterations:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if config.interval_seconds > 0:
            time.sleep(config.interval_seconds)

    finished = datetime.now(UTC)
    summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "iterations": iterations,
        "records_written": records_written,
        "output_dir": str(config.output_dir),
        "last_jsonl": str(last_file) if last_file is not None else None,
        "errors": errors[:50],
        "error_count": len(errors),
        "read_only_public_api": True,
        "wallet_order_account_connected": False,
    }
    summary_path = config.output_dir / f"capture-summary-{finished:%Y%m%dT%H%M%SZ}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def fetch_active_btc_5m_markets(
    client: PolymarketDataApiClient,
    config: CaptureConfig,
) -> Iterable[ActiveBtcMarket]:
    now_ts = int(datetime.now(UTC).timestamp())
    seen: set[str] = set()
    for event in client.iter_events(
        limit=config.market_page_size,
        max_events=config.max_markets,
        sleep_seconds=config.sleep_seconds,
        order="endDate",
        ascending=True,
        closed=False,
        active=True,
        tag_slug="crypto",
    ):
        raw_markets = event.get("markets")
        if not isinstance(raw_markets, list):
            continue
        for raw in raw_markets:
            if not isinstance(raw, Mapping):
                continue
            parsed = active_btc_market_from_raw(raw, now_ts=now_ts)
            if parsed is None or parsed.condition_id in seen:
                continue
            seen.add(parsed.condition_id)
            yield parsed

    for raw in client.iter_closed_markets(
        limit=config.market_page_size,
        max_markets=config.max_markets,
        sleep_seconds=config.sleep_seconds,
        order="endDate",
        ascending=True,
        closed=False,
    ):
        parsed = active_btc_market_from_raw(raw, now_ts=now_ts)
        if parsed is None or parsed.condition_id in seen:
            continue
        seen.add(parsed.condition_id)
        yield parsed


def active_btc_market_from_raw(
    raw: Mapping[str, Any],
    *,
    now_ts: int,
) -> ActiveBtcMarket | None:
    market = parse_closed_market(raw)
    token_ids = clob_token_ids(raw)
    if market is None or len(token_ids) != len(market.outcomes):
        return None
    if not is_btc_5m_market(market.slug, market.title):
        return None
    end_ts = end_timestamp(market.end_time)
    if end_ts is not None and end_ts < now_ts - 30:
        return None
    if "Up" not in market.outcomes or "Down" not in market.outcomes:
        return None
    return ActiveBtcMarket(
        condition_id=market.condition_id,
        slug=market.slug,
        title=market.title,
        end_ts=end_ts,
        outcomes=market.outcomes,
        token_ids=token_ids,
    )


def build_snapshot_record(
    market: ActiveBtcMarket,
    *,
    outcome: str,
    token_id: str,
    raw_book: Mapping[str, Any],
    captured_at: datetime,
    depth_levels: int,
) -> dict[str, Any]:
    bids = parse_order_book_levels(raw_book.get("bids"))
    asks = parse_order_book_levels(raw_book.get("asks"))
    bid_summary = summarize_bid_side(bids, depth_levels)
    ask_summary = summarize_ask_side(asks, depth_levels)
    mid = mid_price(bid_summary.top_price, ask_summary.top_price)
    spread = spread_bps(bid_summary.top_price, ask_summary.top_price, mid)
    seconds_to_close = (
        None if market.end_ts is None else market.end_ts - int(captured_at.timestamp())
    )
    return {
        "captured_at": captured_at.isoformat(),
        "captured_ts_ms": int(captured_at.timestamp() * 1000),
        "source": "polymarket_clob_book_public_read_only",
        "condition_id": market.condition_id,
        "slug": market.slug,
        "title": market.title,
        "end_ts": market.end_ts,
        "seconds_to_close": seconds_to_close,
        "outcome": outcome,
        "token_id": token_id,
        "best_bid": decimal_or_none(bid_summary.top_price),
        "best_bid_size": decimal_or_none(bid_summary.top_size),
        "best_ask": decimal_or_none(ask_summary.top_price),
        "best_ask_size": decimal_or_none(ask_summary.top_size),
        "mid": decimal_or_none(mid),
        "spread_bps": decimal_or_none(spread),
        "bid_depth_usd_top_n": decimal_or_none(bid_summary.depth_usd),
        "ask_depth_usd_top_n": decimal_or_none(ask_summary.depth_usd),
        "depth_levels_requested": depth_levels,
        "bid_levels": bid_summary.levels,
        "ask_levels": ask_summary.levels,
    }


def summarize_bid_side(levels: Sequence[OrderBookLevel], depth_levels: int) -> BookSideSummary:
    top_to_bottom = tuple(reversed(levels))
    return summarize_side(top_to_bottom, depth_levels)


def summarize_ask_side(levels: Sequence[OrderBookLevel], depth_levels: int) -> BookSideSummary:
    return summarize_side(levels, depth_levels)


def summarize_side(levels: Sequence[OrderBookLevel], depth_levels: int) -> BookSideSummary:
    selected = tuple(levels[: max(0, depth_levels)])
    top = selected[0] if selected else None
    depth = sum((level.price * level.size for level in selected), Decimal("0"))
    return BookSideSummary(
        top_price=None if top is None else top.price,
        top_size=None if top is None else top.size,
        depth_usd=depth,
        levels=tuple(
            {"price": decimal_to_str(level.price), "size": decimal_to_str(level.size)}
            for level in selected
        ),
    )


def mid_price(bid: Decimal | None, ask: Decimal | None) -> Decimal | None:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / Decimal("2")


def spread_bps(
    bid: Decimal | None,
    ask: Decimal | None,
    mid: Decimal | None,
) -> Decimal | None:
    if bid is None or ask is None or mid is None or mid <= 0:
        return None
    return (ask - bid) / mid * Decimal("10000")


def append_jsonl(output_dir: Path, captured_at: datetime, record: Mapping[str, Any]) -> Path:
    path = (
        output_dir
        / f"date={captured_at:%Y-%m-%d}"
        / f"hour={captured_at:%H}"
        / "polymarket_btc_5m_orderbooks.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def is_btc_5m_market(slug: str, title: str) -> bool:
    text = f"{slug} {title}".lower()
    compact = text.replace(" ", "").replace("-", "")
    return "btcupdown5m" in compact or (
        "bitcoinupordown" in compact and "5m" in compact
    )


def end_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def decimal_or_none(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return decimal_to_str(value)


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only forward capture of public Polymarket BTC 5m CLOB order books. "
            "Writes JSONL under a private AEGIS_STRATEGIES_ROOT incubating directory."
        )
    )
    parser.add_argument("--output-dir", default=os.getenv("POLYMARKET_BOOK_OUTPUT_DIR"))
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--market-page-size", type=int, default=100)
    parser.add_argument("--max-markets", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--depth-levels", type=int, default=20)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> CaptureConfig:
    base_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    return CaptureConfig(
        output_dir=base_dir / "orderbook_capture",
        interval_seconds=max(0.0, float(args.interval_seconds)),
        duration_seconds=None
        if args.duration_seconds is None
        else max(0.0, float(args.duration_seconds)),
        max_iterations=None if args.max_iterations is None else max(1, int(args.max_iterations)),
        market_page_size=max(1, int(args.market_page_size)),
        max_markets=max(1, int(args.max_markets)),
        timeout_seconds=max(0.1, float(args.timeout_seconds)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        depth_levels=max(1, int(args.depth_levels)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
