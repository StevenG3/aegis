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
from urllib.parse import urlencode

from aegis.polymarket_onchain import PolymarketDataApiClient, parse_closed_market
from aegis.polymarket_structural_scan import (
    OrderBookLevel,
    clob_token_ids,
    parse_order_book_levels,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus63"
DEFAULT_USER_AGENT = "aegis-polymarket-5m-forward-collector/0.1 read-only"
POLYMARKET_RTDS_URL = "wss://ws-live-data.polymarket.com"
CHAINLINK_BTC_SYMBOL = "btc/usd"
CHAINLINK_BTC_STREAM_URL = "https://data.chain.link/streams/btc-usd"


@dataclass(frozen=True)
class ForwardCollectorConfig:
    output_dir: Path
    interval_seconds: float
    duration_seconds: float | None
    max_iterations: int | None
    market_page_size: int
    max_markets: int
    timeout_seconds: float
    sleep_seconds: float
    depth_levels: int
    decision_window_start_seconds: int
    decision_window_end_seconds: int
    settlement_lag_seconds: int
    chainlink_price_url_template: str | None
    chainlink_rtds_enabled: bool
    chainlink_tick_match_tolerance_seconds: int


@dataclass(frozen=True)
class ActiveBtc5mMarket:
    condition_id: str
    slug: str
    title: str
    start_ts: int
    end_ts: int
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]
    resolution_source: str


@dataclass(frozen=True)
class BookSideSummary:
    top_price: Decimal | None
    top_size: Decimal | None
    depth_usd: Decimal
    levels: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ChainlinkTick:
    symbol: str
    price: float
    price_ts_ms: int
    captured_ts_ms: int
    source: str


def main() -> int:
    args = _parse_args()
    config = _config_from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    client = PolymarketDataApiClient(
        user_agent=DEFAULT_USER_AGENT,
        timeout_seconds=config.timeout_seconds,
    )
    summary = capture_forward_books(client, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def capture_forward_books(
    client: PolymarketDataApiClient,
    config: ForwardCollectorConfig,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    deadline = (
        None
        if config.duration_seconds is None
        else time.monotonic() + max(0.0, config.duration_seconds)
    )
    state = load_state(config.output_dir)
    iterations = 0
    records_written = 0
    settlement_records_written = 0
    chainlink_records_written = 0
    errors: list[dict[str, str]] = []
    last_file: Path | None = None
    last_hourly_log = time.monotonic()

    while True:
        captured_at = datetime.now(UTC)
        now_ts = int(captured_at.timestamp())
        chainlink_tick = fetch_chainlink_rtds_tick(config, captured_at)
        if chainlink_tick is not None:
            state = remember_chainlink_tick(state, chainlink_tick)
            last_file = append_jsonl(
                config.output_dir,
                captured_at,
                chainlink_tick_record(chainlink_tick),
            )
            records_written += 1
            chainlink_records_written += 1
        markets = tuple(fetch_current_btc_5m_markets(client, config, now_ts=now_ts))
        state = remember_markets(state, markets)
        for market in markets:
            seconds_to_close = market.end_ts - now_ts
            if not (
                config.decision_window_end_seconds
                <= seconds_to_close
                <= config.decision_window_start_seconds
            ):
                continue
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
                        chainlink_tick=chainlink_tick,
                        chainlink_start_price=chainlink_price_from_state(
                            state,
                            market.start_ts,
                            tolerance_seconds=config.chainlink_tick_match_tolerance_seconds,
                        ),
                    )
                    last_file = append_jsonl(config.output_dir, captured_at, record)
                    records_written += 1
                    if config.sleep_seconds > 0:
                        time.sleep(config.sleep_seconds)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "stage": "order_book",
                            "slug": market.slug,
                            "condition_id": market.condition_id,
                            "token_id": token_id,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
        for market in markets_due_for_settlement(state, now_ts, config.settlement_lag_seconds):
            try:
                settlement = settlement_record(client, config, state, market, captured_at)
                if settlement is None:
                    continue
                last_file = append_jsonl(config.output_dir, captured_at, settlement)
                records_written += 1
                settlement_records_written += 1
                state.setdefault("settled_slugs", {})[market.slug] = captured_at.isoformat()
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "settlement",
                        "slug": market.slug,
                        "condition_id": market.condition_id,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )

        save_state(config.output_dir, state)
        iterations += 1
        if time.monotonic() - last_hourly_log >= 3600:
            print(json.dumps(hourly_coverage(config.output_dir), sort_keys=True), flush=True)
            last_hourly_log = time.monotonic()
        if config.max_iterations is not None and iterations >= config.max_iterations:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if config.interval_seconds > 0:
            time.sleep(config.interval_seconds)

    finished = datetime.now(UTC)
    coverage = hourly_coverage(config.output_dir)
    summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "iterations": iterations,
        "records_written": records_written,
        "snapshot_records_written": (
            records_written - settlement_records_written - chainlink_records_written
        ),
        "settlement_records_written": settlement_records_written,
        "chainlink_records_written": chainlink_records_written,
        "output_dir": str(config.output_dir),
        "last_jsonl": str(last_file) if last_file is not None else None,
        "coverage": coverage,
        "errors": errors[:50],
        "error_count": len(errors),
        "read_only_public_api": True,
        "wallet_order_account_connected": False,
    }
    summary_path = config.output_dir / f"forward-collector-summary-{finished:%Y%m%dT%H%M%SZ}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def fetch_current_btc_5m_markets(
    client: PolymarketDataApiClient,
    config: ForwardCollectorConfig,
    *,
    now_ts: int,
) -> Iterable[ActiveBtc5mMarket]:
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
        markets = event.get("markets")
        if not isinstance(markets, list):
            continue
        for raw in markets:
            if not isinstance(raw, Mapping):
                continue
            parsed = active_market_from_raw(raw, now_ts=now_ts)
            if parsed is None or parsed.condition_id in seen:
                continue
            seen.add(parsed.condition_id)
            yield parsed


def active_market_from_raw(raw: Mapping[str, Any], *, now_ts: int) -> ActiveBtc5mMarket | None:
    market = parse_closed_market(raw)
    token_ids = clob_token_ids(raw)
    if market is None or len(token_ids) != len(market.outcomes):
        return None
    if not market.slug.startswith("btc-updown-5m-"):
        return None
    resolution_source = str(raw.get("resolutionSource") or "")
    if resolution_source != CHAINLINK_BTC_STREAM_URL:
        return None
    end_ts = end_timestamp(market.end_time)
    if end_ts is None or end_ts < now_ts - 30:
        return None
    start_ts = start_timestamp_from_slug(market.slug, end_ts)
    if "Up" not in market.outcomes or "Down" not in market.outcomes:
        return None
    return ActiveBtc5mMarket(
        condition_id=market.condition_id,
        slug=market.slug,
        title=market.title,
        start_ts=start_ts,
        end_ts=end_ts,
        outcomes=market.outcomes,
        token_ids=token_ids,
        resolution_source=resolution_source,
    )


def build_snapshot_record(
    market: ActiveBtc5mMarket,
    *,
    outcome: str,
    token_id: str,
    raw_book: Mapping[str, Any],
    captured_at: datetime,
    depth_levels: int,
    chainlink_tick: ChainlinkTick | None,
    chainlink_start_price: float | None,
) -> dict[str, Any]:
    bids = parse_order_book_levels(raw_book.get("bids"))
    asks = parse_order_book_levels(raw_book.get("asks"))
    bid_summary = summarize_bid_side(bids, depth_levels)
    ask_summary = summarize_ask_side(asks, depth_levels)
    mid = mid_price(bid_summary.top_price, ask_summary.top_price)
    spread = spread_bps(bid_summary.top_price, ask_summary.top_price, mid)
    return {
        "record_type": "snapshot",
        "captured_at": captured_at.isoformat(),
        "captured_ts_ms": int(captured_at.timestamp() * 1000),
        "source": "polymarket_clob_book_public_read_only",
        "actual_settlement_source": CHAINLINK_BTC_STREAM_URL,
        "actual_settlement_source_type": "chainlink_data_streams_not_aggregator_v3",
        "condition_id": market.condition_id,
        "slug": market.slug,
        "title": market.title,
        "start_ts": market.start_ts,
        "end_ts": market.end_ts,
        "seconds_to_close": market.end_ts - int(captured_at.timestamp()),
        "outcome": outcome,
        "token_id": token_id,
        "best_bid": decimal_or_none(bid_summary.top_price),
        "best_bid_size": decimal_or_none(bid_summary.top_size),
        "best_ask": decimal_or_none(ask_summary.top_price),
        "best_ask_size": decimal_or_none(ask_summary.top_size),
        "mid": decimal_or_none(mid),
        "last": None,
        "spread_bps": decimal_or_none(spread),
        "bid_depth_usd_top_n": decimal_or_none(bid_summary.depth_usd),
        "ask_depth_usd_top_n": decimal_or_none(ask_summary.depth_usd),
        "depth_levels_requested": depth_levels,
        "bid_levels": bid_summary.levels,
        "ask_levels": ask_summary.levels,
        "chainlink_start_price": chainlink_start_price,
        "chainlink_start_status": "ok_from_forward_rtds_cache"
        if chainlink_start_price is not None
        else "chainlink_start_tick_not_yet_available",
        "chainlink_reference_price": None if chainlink_tick is None else chainlink_tick.price,
        "chainlink_reference_status": "ok_from_polymarket_rtds_chainlink"
        if chainlink_tick is not None
        else "chainlink_rtds_not_available",
        "chainlink_reference_ts_ms": None if chainlink_tick is None else chainlink_tick.price_ts_ms,
        "chainlink_source": "polymarket_rtds_crypto_prices_chainlink",
    }


def settlement_record(
    client: PolymarketDataApiClient,
    config: ForwardCollectorConfig,
    state: Mapping[str, Any],
    market: ActiveBtc5mMarket,
    captured_at: datetime,
) -> dict[str, Any] | None:
    event = event_by_slug(client, market.slug)
    if event is None:
        return None
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        return None
    raw_market = markets[0]
    if not isinstance(raw_market, Mapping):
        return None
    direction = settlement_direction(raw_market)
    if direction is None:
        return None
    start_price = forward_chainlink_price(config, state, market.start_ts)
    end_price = forward_chainlink_price(config, state, market.end_ts)
    return {
        "record_type": "settlement",
        "captured_at": captured_at.isoformat(),
        "captured_ts_ms": int(captured_at.timestamp() * 1000),
        "source": "gamma_settlement_public_read_only",
        "actual_settlement_source": market.resolution_source,
        "actual_settlement_source_type": "chainlink_data_streams_not_aggregator_v3",
        "condition_id": market.condition_id,
        "slug": market.slug,
        "title": market.title,
        "start_ts": market.start_ts,
        "end_ts": market.end_ts,
        "settlement_direction": direction,
        "chainlink_start_price": start_price.get("price"),
        "chainlink_start_status": start_price.get("status"),
        "chainlink_reference_price": end_price.get("price"),
        "chainlink_reference_status": end_price.get("status"),
        "chainlink_source": end_price.get("source"),
    }


def fetch_chainlink_rtds_tick(
    config: ForwardCollectorConfig,
    captured_at: datetime,
) -> ChainlinkTick | None:
    if not config.chainlink_rtds_enabled:
        return None
    try:
        from websockets.sync.client import connect
    except ImportError:
        return None
    subscribe = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": json.dumps({"symbol": CHAINLINK_BTC_SYMBOL}),
            }
        ],
    }
    try:
        with connect(
            POLYMARKET_RTDS_URL,
            open_timeout=config.timeout_seconds,
            close_timeout=1.0,
        ) as websocket:
            websocket.send(json.dumps(subscribe))
            deadline = time.monotonic() + config.timeout_seconds
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                message = websocket.recv(timeout=remaining)
                tick = chainlink_tick_from_rtds_message(message, captured_at)
                if tick is not None:
                    return tick
    except Exception:  # noqa: BLE001
        return None
    return None


def chainlink_tick_from_rtds_message(
    message: object, captured_at: datetime
) -> ChainlinkTick | None:
    if isinstance(message, bytes):
        message = message.decode()
    if not isinstance(message, str) or message in {"PING", "PONG"}:
        return None
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        return None
    batched = payload.get("data")
    if isinstance(batched, list) and batched:
        latest = batched[-1]
        if isinstance(latest, Mapping):
            price = _float_value(latest.get("value"))
            price_ts_ms = _int_value(latest.get("timestamp"))
            if price is not None and price_ts_ms is not None:
                return ChainlinkTick(
                    symbol=CHAINLINK_BTC_SYMBOL,
                    price=price,
                    price_ts_ms=price_ts_ms,
                    captured_ts_ms=int(captured_at.timestamp() * 1000),
                    source="polymarket_rtds_crypto_prices_chainlink",
                )
    symbol = payload.get("symbol")
    if symbol != CHAINLINK_BTC_SYMBOL:
        return None
    price = _float_value(payload.get("value"))
    price_ts_ms = _int_value(payload.get("timestamp"))
    if price is None or price_ts_ms is None:
        return None
    return ChainlinkTick(
        symbol=CHAINLINK_BTC_SYMBOL,
        price=price,
        price_ts_ms=price_ts_ms,
        captured_ts_ms=int(captured_at.timestamp() * 1000),
        source="polymarket_rtds_crypto_prices_chainlink",
    )


def chainlink_tick_record(tick: ChainlinkTick) -> dict[str, Any]:
    return {
        "record_type": "chainlink_price",
        "source": tick.source,
        "symbol": tick.symbol,
        "price": tick.price,
        "price_ts_ms": tick.price_ts_ms,
        "captured_ts_ms": tick.captured_ts_ms,
    }


def event_by_slug(client: PolymarketDataApiClient, slug: str) -> Mapping[str, Any] | None:
    data = client._get_json(f"{client.gamma_api_base_url}/events?{urlencode({'slug': slug})}")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    return first if isinstance(first, Mapping) else None


def settlement_direction(raw_market: Mapping[str, Any]) -> str | None:
    outcomes = _json_list(raw_market.get("outcomes"))
    prices = _json_list(raw_market.get("outcomePrices"))
    if len(outcomes) != len(prices) or not prices:
        return None
    numeric = []
    for price in prices:
        parsed = _float_value(price)
        if parsed is None:
            return None
        numeric.append(parsed)
    if max(numeric) < 0.99:
        return None
    winner = outcomes[numeric.index(max(numeric))]
    return winner if winner in {"Up", "Down"} else None


def forward_chainlink_price(
    config: ForwardCollectorConfig,
    state: Mapping[str, Any],
    timestamp: int,
) -> dict[str, Any]:
    cached = chainlink_price_from_state(
        state,
        timestamp,
        tolerance_seconds=config.chainlink_tick_match_tolerance_seconds,
    )
    if cached is not None:
        return {
            "price": cached,
            "status": "ok_from_forward_rtds_cache",
            "source": "polymarket_rtds_crypto_prices_chainlink",
        }
    return fetch_chainlink_price(config, timestamp)


def fetch_chainlink_price(config: ForwardCollectorConfig, timestamp: int) -> dict[str, Any]:
    if config.chainlink_price_url_template is None:
        return {
            "price": None,
            "status": "chainlink_historical_source_not_configured",
            "source": "chainlink",
        }
    url = config.chainlink_price_url_template.format(timestamp=timestamp)
    try:
        data = PolymarketDataApiClient(timeout_seconds=config.timeout_seconds)._get_json(url)
        price = chainlink_price_from_json(data)
    except Exception as exc:  # noqa: BLE001
        return {"price": None, "status": f"chainlink_fetch_error:{exc}", "source": url}
    return {
        "price": price,
        "status": "ok" if price is not None else "chainlink_price_missing",
        "source": url,
    }


def remember_chainlink_tick(state: dict[str, Any], tick: ChainlinkTick) -> dict[str, Any]:
    raw_ticks = state.setdefault("chainlink_ticks", [])
    if not isinstance(raw_ticks, list):
        raw_ticks = []
        state["chainlink_ticks"] = raw_ticks
    raw_ticks.append(
        {
            "price": tick.price,
            "price_ts_ms": tick.price_ts_ms,
            "captured_ts_ms": tick.captured_ts_ms,
            "source": tick.source,
        }
    )
    state["chainlink_ticks"] = raw_ticks[-10000:]
    return state


def chainlink_price_from_state(
    state: Mapping[str, Any],
    timestamp: int,
    *,
    tolerance_seconds: int,
) -> float | None:
    raw_ticks = state.get("chainlink_ticks")
    if not isinstance(raw_ticks, list):
        return None
    target_ms = timestamp * 1000
    best_distance: int | None = None
    best_price: float | None = None
    for raw in raw_ticks:
        if not isinstance(raw, Mapping):
            continue
        price = _float_value(raw.get("price"))
        price_ts_ms = _int_value(raw.get("price_ts_ms"))
        if price is None or price_ts_ms is None:
            continue
        distance = abs(price_ts_ms - target_ms)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_price = price
    if best_distance is None or best_distance > tolerance_seconds * 1000:
        return None
    return best_price


def chainlink_price_from_json(data: object) -> float | None:
    if not isinstance(data, Mapping):
        return None
    for key in ("price", "value", "answer"):
        value = data.get(key)
        if isinstance(value, (int, float, str)):
            try:
                parsed = float(value)
            except ValueError:
                continue
            return parsed / 100_000_000 if parsed > 1_000_000_000 else parsed
    return None


def remember_markets(
    state: dict[str, Any], markets: Sequence[ActiveBtc5mMarket]
) -> dict[str, Any]:
    known = state.setdefault("markets", {})
    if not isinstance(known, dict):
        known = {}
        state["markets"] = known
    for market in markets:
        known[market.slug] = {
            "condition_id": market.condition_id,
            "slug": market.slug,
            "title": market.title,
            "start_ts": market.start_ts,
            "end_ts": market.end_ts,
            "outcomes": list(market.outcomes),
            "token_ids": list(market.token_ids),
            "resolution_source": market.resolution_source,
        }
    state.setdefault("settled_slugs", {})
    return state


def markets_due_for_settlement(
    state: Mapping[str, Any], now_ts: int, lag_seconds: int
) -> tuple[ActiveBtc5mMarket, ...]:
    raw_markets = state.get("markets")
    settled = state.get("settled_slugs")
    if not isinstance(raw_markets, Mapping):
        return ()
    settled_slugs = set(settled.keys()) if isinstance(settled, Mapping) else set()
    due: list[ActiveBtc5mMarket] = []
    for slug, raw in raw_markets.items():
        if not isinstance(slug, str) or slug in settled_slugs or not isinstance(raw, Mapping):
            continue
        end_ts = _int_value(raw.get("end_ts"))
        start_ts = _int_value(raw.get("start_ts"))
        if end_ts is None or start_ts is None or now_ts < end_ts + lag_seconds:
            continue
        outcomes = tuple(str(item) for item in _json_list(raw.get("outcomes")))
        token_ids = tuple(str(item) for item in _json_list(raw.get("token_ids")))
        due.append(
            ActiveBtc5mMarket(
                condition_id=str(raw.get("condition_id", "")),
                slug=slug,
                title=str(raw.get("title", "")),
                start_ts=start_ts,
                end_ts=end_ts,
                outcomes=outcomes,
                token_ids=token_ids,
                resolution_source=str(raw.get("resolution_source", "")),
            )
        )
    return tuple(due)


def hourly_coverage(output_dir: Path) -> dict[str, Any]:
    market_slugs: set[str] = set()
    settled_slugs: set[str] = set()
    snapshots = 0
    settlements = 0
    chainlink_ticks = 0
    for path in output_dir.glob("date=*/hour=*/polymarket_btc_5m_forward.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = row.get("slug")
            if isinstance(slug, str):
                market_slugs.add(slug)
            if row.get("record_type") == "settlement":
                settlements += 1
                if isinstance(slug, str):
                    settled_slugs.add(slug)
            elif row.get("record_type") == "snapshot":
                snapshots += 1
            elif row.get("record_type") == "chainlink_price":
                chainlink_ticks += 1
    return {
        "record_type": "coverage",
        "generated_at": datetime.now(UTC).isoformat(),
        "markets": len(market_slugs),
        "snapshots": snapshots,
        "settlements": settlements,
        "chainlink_ticks": chainlink_ticks,
        "settlement_completion_rate": len(settled_slugs) / len(market_slugs)
        if market_slugs
        else 0.0,
    }


def summarize_bid_side(levels: Sequence[OrderBookLevel], depth_levels: int) -> BookSideSummary:
    return summarize_side(tuple(reversed(levels)), depth_levels)


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
    if bid is None or ask is None or bid <= 0 or ask <= 0:
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
        / "polymarket_btc_5m_forward.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "forward-collector-state.json"
    if not path.exists():
        return {"markets": {}, "settled_slugs": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"markets": {}, "settled_slugs": {}}


def save_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    path = output_dir / "forward-collector-state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def end_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def start_timestamp_from_slug(slug: str, end_ts: int) -> int:
    parts = slug.rsplit("-", 1)
    if len(parts) == 2:
        parsed = _int_value(parts[1])
        if parsed is not None:
            return parsed
    return end_ts - 300


def decimal_or_none(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return decimal_to_str(value)


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only forward collector for Polymarket BTC 5m executable order books."
    )
    parser.add_argument("--output-dir", default=os.getenv("POLYMARKET_FORWARD_OUTPUT_DIR"))
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--market-page-size", type=int, default=100)
    parser.add_argument("--max-markets", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--decision-window-start-seconds", type=int, default=90)
    parser.add_argument("--decision-window-end-seconds", type=int, default=30)
    parser.add_argument("--settlement-lag-seconds", type=int, default=60)
    parser.add_argument("--chainlink-price-url-template", default=None)
    parser.add_argument("--disable-chainlink-rtds", action="store_true")
    parser.add_argument("--chainlink-tick-match-tolerance-seconds", type=int, default=10)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ForwardCollectorConfig:
    base_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    return ForwardCollectorConfig(
        output_dir=base_dir / "forward",
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
        decision_window_start_seconds=max(1, int(args.decision_window_start_seconds)),
        decision_window_end_seconds=max(0, int(args.decision_window_end_seconds)),
        settlement_lag_seconds=max(0, int(args.settlement_lag_seconds)),
        chainlink_price_url_template=args.chainlink_price_url_template,
        chainlink_rtds_enabled=not bool(args.disable_chainlink_rtds),
        chainlink_tick_match_tolerance_seconds=max(
            0, int(args.chainlink_tick_match_tolerance_seconds)
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
