#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from aegis.gefs_weather_probability import (
    GEFS_MEMBERS,
    STATIONS,
    SampledDecoderValueSelfCheck,
    bucket_probability_from_samples,
    station_daily_samples_from_gefs,
    target_date_from_temperature_slug,
)
from aegis.polymarket_weather_relative_value import (
    TemperatureBucket,
    parse_temperature_bucket,
    settlement_source_alignment,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus71"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
USER_AGENT = "aegis-polymarket-weather-forward-collector/0.1 read-only"
DEFAULT_CITY_SLUGS = (
    "nyc",
    "miami",
    "los-angeles",
    "chicago",
    "austin",
    "denver",
    "seattle",
    "dallas",
    "houston",
    "atlanta",
    "san-francisco",
)


@dataclass(frozen=True)
class BookSide:
    best_price: float | None
    best_size: float | None
    depth_usd: float
    levels: int
    best_bid: float | None = None
    best_bid_size: float | None = None
    bid_depth_usd: float = 0.0
    bid_levels: int = 0


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK) / "forward"
    target_dates = tuple(args.target_date) if args.target_date else (datetime.now(UTC).date(),)
    rows, summary = collect_weather_forward_snapshot(
        output_dir=output_dir,
        target_dates=target_dates,
        city_slugs=tuple(args.city) if args.city else DEFAULT_CITY_SLUGS,
        members=GEFS_MEMBERS[: args.member_count],
        max_events=args.max_events,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "rows": rows,
                "summary_path": str(summary),
                "restart_count": 0,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def collect_weather_forward_snapshot(
    *,
    output_dir: Path,
    target_dates: Sequence[date],
    city_slugs: Sequence[str],
    members: Sequence[str],
    max_events: int = 0,
    max_workers: int = 4,
) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "gefs-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    excluded: list[Mapping[str, object]] = []
    decoder = SampledDecoderValueSelfCheck(max_checks=24)
    stamp = datetime.now(UTC)
    capture_ts = int(stamp.timestamp())
    rows_path = _rows_path(output_dir, stamp)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    events_seen = 0
    with rows_path.open("a", encoding="utf-8") as handle:
        for target in target_dates:
            for city_slug in city_slugs:
                if max_events and events_seen >= max_events:
                    break
                event = _fetch_event_by_slug(_temperature_event_slug(city_slug, target))
                if event is None:
                    excluded.append(
                        {
                            "city": city_slug,
                            "date": target.isoformat(),
                            "reason": "event_missing",
                        }
                    )
                    continue
                events_seen += 1
                event_rows, event_excluded = _rows_for_event(
                    event=event,
                    capture_ts=capture_ts,
                    members=members,
                    cache_dir=cache_dir,
                    decoder=decoder,
                    max_workers=max_workers,
                )
                excluded.extend(event_excluded)
                for row in event_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    written += 1
                time.sleep(0.05)
    summary = {
        "generated_at": stamp.isoformat(),
        "rows_path": str(rows_path),
        "events_seen": events_seen,
        "rows_written": written,
        "excluded_count": len(excluded),
        "excluded_reasons": _reason_counts(excluded),
        "excluded_sample": list(excluded[:10]),
        "city_slugs": list(city_slugs),
        "target_dates": [target.isoformat() for target in target_dates],
        "member_count": len(members),
        "decoder_value_self_check_messages": decoder.checks_run,
        "restart_count": 0,
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }
    summary_path = output_dir / f"weather-forward-summary-{stamp:%Y%m%dT%H%M%SZ}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written, summary_path


def _rows_for_event(
    *,
    event: Mapping[str, Any],
    capture_ts: int,
    members: Sequence[str],
    cache_dir: Path,
    decoder: Any,
    max_workers: int,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    event_slug = _required_str(event.get("slug"))
    target_date = target_date_from_temperature_slug(event_slug)
    if target_date is None:
        return [], [{"event": event_slug, "reason": "target_date_unparseable"}]
    alignment = settlement_source_alignment(event)
    if not alignment.get("aligned"):
        return [], [{"event": event_slug, "reason": "settlement_source_not_aligned"}]
    station_code = _optional_str(alignment.get("station"))
    station = STATIONS.get(station_code or "")
    if station is None:
        return [], [{"event": event_slug, "reason": "station_coordinates_missing"}]
    market_end_ts = _event_end_ts(event)
    if market_end_ts is None:
        return [], [{"event": event_slug, "reason": "market_end_timestamp_missing"}]
    forecast_error: Mapping[str, object] | None = None
    try:
        samples, cycle, leads = station_daily_samples_from_gefs(
            station=station,
            target_date=target_date,
            decision_ts=capture_ts,
            members=members,
            text_fetcher=_cached_text_fetcher(cache_dir / "idx"),
            byte_fetcher=_cached_byte_fetcher(cache_dir / "messages"),
            decoder=decoder,
            max_workers=max_workers,
        )
    except Exception as exc:
        samples = []
        cycle = None
        leads = ()
        forecast_error = {
            "reason": "gefs_decode_or_interpolation_failed",
            "error": exc.__class__.__name__,
            "message": str(exc),
        }
    rows: list[Mapping[str, object]] = []
    excluded: list[Mapping[str, object]] = []
    for market in _markets(event):
        bucket = _bucket_from_market(market)
        if bucket is None:
            excluded.append(
                {
                    "event": event_slug,
                    "market": market.get("slug"),
                    "reason": "bucket_unparseable",
                }
            )
            continue
        yes_token = _token_at(market, 0)
        no_token = _token_at(market, 1)
        yes_book = _book_for_token(yes_token)
        no_book = _book_for_token(no_token)
        probability = (
            bucket_probability_from_samples(
                samples=samples,
                bucket=bucket,
                cycle=cycle,
                leads=leads,
                station=station,
            )
            if samples and cycle is not None
            else None
        )
        rows.append(
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "event_slug": event_slug,
                "market_slug": str(market.get("slug")),
                "city": _city_from_slug(event_slug),
                "station": station.code,
                "resolution_source": alignment.get("sources"),
                "target_date": target_date.isoformat(),
                "decision_ts": capture_ts,
                "market_end_ts": market_end_ts,
                "forecast_available": probability is not None,
                "forecast_error": dict(forecast_error) if forecast_error is not None else None,
                "forecast_issue_ts": probability.issue_ts if probability is not None else None,
                "bucket_label": bucket.label,
                "model_probability": probability.probability if probability is not None else None,
                "member_count": (
                    probability.member_count if probability is not None else len(members)
                ),
                "members_in_bucket": (
                    probability.members_in_bucket if probability is not None else None
                ),
                "yes_token": yes_token,
                "no_token": no_token,
                "yes_book": asdict(yes_book),
                "no_book": asdict(no_book),
                "yes_ask": yes_book.best_price,
                "yes_bid": yes_book.best_bid,
                "no_ask": no_book.best_price,
                "no_bid": no_book.best_bid,
                "yes_ask_depth_usd": yes_book.depth_usd,
                "no_ask_depth_usd": no_book.depth_usd,
                "yes_bid_depth_usd": yes_book.bid_depth_usd,
                "no_bid_depth_usd": no_book.bid_depth_usd,
                "actual_won": _market_yes_won(market) if bool(event.get("closed")) else None,
                "price_source": "current CLOB /book true ask/depth snapshot",
            }
        )
    return rows, excluded


def _book_for_token(token: str | None) -> BookSide:
    if token is None:
        return BookSide(best_price=None, best_size=None, depth_usd=0.0, levels=0)
    try:
        data = _get_json(f"{CLOB_BASE_URL}/book?{urllib.parse.urlencode({'token_id': token})}")
    except Exception:
        return BookSide(best_price=None, best_size=None, depth_usd=0.0, levels=0)
    asks = data.get("asks") if isinstance(data, Mapping) else None
    bids = data.get("bids") if isinstance(data, Mapping) else None
    if not isinstance(asks, list) and not isinstance(bids, list):
        return BookSide(best_price=None, best_size=None, depth_usd=0.0, levels=0)
    ask_levels = _book_levels(asks if isinstance(asks, list) else [])
    bid_levels = _book_levels(bids if isinstance(bids, list) else [])
    best_price: float | None = None
    best_size: float | None = None
    if ask_levels:
        ask_levels.sort(key=lambda value: value[0])
        best_price, best_size = ask_levels[0]
    best_bid: float | None = None
    best_bid_size: float | None = None
    if bid_levels:
        bid_levels.sort(key=lambda value: value[0], reverse=True)
        best_bid, best_bid_size = bid_levels[0]
    return BookSide(
        best_price=best_price,
        best_size=best_size,
        depth_usd=sum(price * size for price, size in ask_levels),
        levels=len(ask_levels),
        best_bid=best_bid,
        best_bid_size=best_bid_size,
        bid_depth_usd=sum(price * size for price, size in bid_levels),
        bid_levels=len(bid_levels),
    )


def _book_levels(items: Sequence[object]) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        price = _optional_float(item.get("price"))
        size = _optional_float(item.get("size"))
        if price is not None and size is not None and 0.0 < price <= 1.0 and size > 0:
            levels.append((price, size))
    return levels


def _cached_text_fetcher(cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(url: str) -> str:
        path = cache_dir / f"{_sha(url)}.idx"
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = _fetch_text(url)
        path.write_text(text, encoding="utf-8")
        return text

    return fetch


def _cached_byte_fetcher(cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(url: str, start: int, end: int) -> bytes:
        path = cache_dir / f"{_sha(f'{url}:{start}:{end}')}.grib2"
        if path.exists():
            return path.read_bytes()
        data = _fetch_range(url, start, end)
        path.write_bytes(data)
        return data

    return fetch


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _fetch_event_by_slug(slug: str) -> Mapping[str, Any] | None:
    try:
        data = _get_json(f"{GAMMA_BASE_URL}/events?{urllib.parse.urlencode({'slug': slug})}")
    except Exception:
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        return None
    return dict(data[0])


def _markets(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    markets = event.get("markets")
    if not isinstance(markets, list):
        return []
    return [market for market in markets if isinstance(market, Mapping)]


def _token_at(market: Mapping[str, Any], index: int) -> str | None:
    raw = market.get("clobTokenIds")
    if not isinstance(raw, str):
        return None
    try:
        tokens = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(tokens, list) and len(tokens) > index:
        return str(tokens[index])
    return None


def _market_yes_won(market: Mapping[str, Any]) -> bool:
    raw = market.get("outcomePrices")
    if not isinstance(raw, str):
        return False
    try:
        prices = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(prices, list) and prices and str(prices[0]) == "1")


def _bucket_from_market(market: Mapping[str, Any]) -> TemperatureBucket | None:
    question = str(market.get("question") or "")
    between = re.search(r"between\s+(-?\d+)-(-?\d+)°?F", question, flags=re.IGNORECASE)
    if between:
        return parse_temperature_bucket(f"{between.group(1)}-{between.group(2)}°F")
    below = re.search(r"(-?\d+)°?F\s+or\s+below", question, flags=re.IGNORECASE)
    if below:
        return parse_temperature_bucket(f"{below.group(1)}°F or below")
    above = re.search(r"(-?\d+)°?F\s+or\s+(?:higher|above)", question, flags=re.IGNORECASE)
    if above:
        return parse_temperature_bucket(f"{above.group(1)}°F or higher")
    slug = str(market.get("slug") or "")
    span = re.search(r"-(\d+)-(\d+)f$", slug)
    if span:
        return parse_temperature_bucket(f"{span.group(1)}-{span.group(2)}°F")
    low = re.search(r"-(\d+)forbelow$", slug)
    if low:
        return parse_temperature_bucket(f"{low.group(1)}°F or below")
    high = re.search(r"-(\d+)forhigher$", slug)
    if high:
        return parse_temperature_bucket(f"{high.group(1)}°F or higher")
    return None


def _temperature_event_slug(city_slug: str, day: date) -> str:
    month = day.strftime("%B").lower()
    return f"highest-temperature-in-{city_slug}-on-{month}-{day.day}-{day.year}"


def _city_from_slug(slug: str) -> str:
    for city_slug in sorted(DEFAULT_CITY_SLUGS, key=len, reverse=True):
        if f"highest-temperature-in-{city_slug}-" in slug:
            return city_slug
    return "unknown"


def _event_end_ts(event: Mapping[str, Any]) -> int | None:
    value = event.get("endDate")
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _rows_path(output_dir: Path, stamp: datetime) -> Path:
    return (
        output_dir
        / f"date={stamp:%Y-%m-%d}"
        / f"hour={stamp:%H}"
        / "polymarket_weather_forward.jsonl"
    )


def _reason_counts(records: Sequence[Mapping[str, object]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = _optional_str(record.get("reason")) or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def _fetch_range(url: str, byte_start: int, byte_end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": f"bytes={byte_start}-{byte_end}"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _required_str(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("required string missing")
    return value


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture read-only Polymarket weather CLOB + GEFS forward snapshots."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-date", type=date.fromisoformat, action="append", default=None)
    parser.add_argument("--city", action="append", default=None)
    parser.add_argument("--member-count", type=int, default=len(GEFS_MEMBERS))
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
