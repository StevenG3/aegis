#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from aegis.gefs_weather_probability import (
    GEFS_MEMBERS,
    STATIONS,
    GefsCycle,
    SampledDecoderValueSelfCheck,
    StationForecastSample,
    bucket_probability_from_samples,
    gefs_archive_availability,
    station_daily_samples_from_gefs,
    target_date_from_temperature_slug,
)
from aegis.polymarket_weather_precision_rule import (
    WeatherPrecisionRuleConfig,
    precision_data_blocked_report,
    run_weather_precision_rule,
)
from aegis.polymarket_weather_relative_value import (
    TemperatureBucket,
    parse_temperature_bucket,
    settlement_source_alignment,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus79"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
USER_AGENT = "aegis-polymarket-weather-gefs-precision/0.1 read-only"
CITY_SLUGS = (
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
    "london",
    "paris",
    "berlin",
    "madrid",
    "tokyo",
    "seoul",
    "singapore",
    "sydney",
    "toronto",
)
DEFAULT_START_DATE = date(2026, 4, 1)
DEFAULT_END_DATE = date(2026, 6, 23)
ASK_PROXY_SPREAD_COST = 0.02


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or output_dir / "gefs-grib-fragments"
    sample_cache_dir = args.sample_cache_dir or output_dir / "gefs-station-daily-samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_cache_dir.mkdir(parents=True, exist_ok=True)
    result = run_weather_gefs_precision_evidence(
        start_date=args.start_date,
        end_date=args.end_date,
        city_slugs=tuple(args.city) if args.city else CITY_SLUGS,
        members=GEFS_MEMBERS[: args.member_count],
        cache_dir=cache_dir,
        sample_cache_dir=sample_cache_dir,
        max_events=args.max_events,
        max_workers=args.max_workers,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"polymarket-weather-gefs-precision-{stamp}.json"
    md_path = output_dir / f"polymarket-weather-gefs-precision-{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(result, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": result["state"],
                "verdict": result["verdict"],
                "data_adequacy": result["data_adequacy"],
                "reason": result["reason"],
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_weather_gefs_precision_evidence(
    *,
    start_date: date,
    end_date: date,
    city_slugs: Sequence[str],
    members: Sequence[str],
    cache_dir: Path,
    sample_cache_dir: Path,
    max_events: int = 0,
    max_workers: int = 8,
) -> Mapping[str, Any]:
    events = _fetch_temperature_events(
        start_date=start_date,
        end_date=end_date,
        city_slugs=city_slugs,
        max_events=max_events,
    )
    rows: list[Mapping[str, object]] = []
    excluded: list[Mapping[str, object]] = []
    availability_checks: list[Mapping[str, object]] = []
    probability_sample: list[Mapping[str, object]] = []
    text_fetcher = _cached_text_fetcher(cache_dir / "idx")
    byte_fetcher = _cached_byte_fetcher(cache_dir / "messages")
    decoder_self_check = SampledDecoderValueSelfCheck(max_checks=24)

    processed_events = 0
    aligned_events = 0
    for event in events:
        event_slug = _str(event.get("slug"))
        if event_slug is None or not bool(event.get("closed")):
            continue
        target_date = target_date_from_temperature_slug(event_slug)
        if target_date is None:
            excluded.append({"event": event_slug, "reason": "target_date_unparseable"})
            continue
        alignment = settlement_source_alignment(event)
        if not alignment.get("aligned"):
            excluded.append({"event": event_slug, "reason": "settlement_source_not_aligned"})
            continue
        aligned_events += 1
        station_code = _str(alignment.get("station"))
        station = STATIONS.get(station_code or "")
        if station is None:
            excluded.append({"event": event_slug, "reason": "station_coordinates_missing"})
            continue
        decision_ts = _event_end_ts(event)
        if decision_ts is None:
            excluded.append({"event": event_slug, "reason": "decision_timestamp_missing"})
            continue
        availability = gefs_archive_availability(
            station=station,
            target_date=target_date,
            decision_ts=decision_ts,
            members=members,
            text_fetcher=text_fetcher,
        )
        availability_checks.append(
            {
                "event": event_slug,
                "station": station.code,
                "required_messages": availability.required_messages,
                "available_messages": availability.available_messages,
                "missing_messages": availability.missing_messages,
                "missing_examples": list(availability.missing_examples),
            }
        )
        if availability.missing_messages:
            excluded.append({"event": event_slug, "reason": "gefs_archive_missing"})
            continue
        try:
            print(
                f"[weather-gefs-precision] processing {event_slug} station={station.code}",
                file=sys.stderr,
                flush=True,
            )
            samples, cycle, leads = _load_or_compute_samples(
                sample_cache_dir=sample_cache_dir,
                event_slug=event_slug,
                station=station,
                target_date=target_date,
                decision_ts=decision_ts,
                members=members,
                text_fetcher=text_fetcher,
                byte_fetcher=byte_fetcher,
                decoder=decoder_self_check,
                max_workers=max_workers,
            )
        except Exception as exc:
            excluded.append(
                {
                    "event": event_slug,
                    "reason": "gefs_decode_or_interpolation_failed",
                    "error": exc.__class__.__name__,
                }
            )
            continue
        processed_events += 1
        event_rows, event_samples = _precision_rows_for_event(
            event=event,
            station_code=station.code,
            decision_ts=decision_ts,
            samples=samples,
            cycle=cycle,
            leads=leads,
        )
        rows.extend(event_rows)
        probability_sample.extend(event_samples[:3])
        time.sleep(0.02)

    config = WeatherPrecisionRuleConfig(
        min_observations=30,
        pbo_splits=4,
        slippage_rate=0.005,
        survivor_light=True,
        entry_windows=("market_end_proxy",),
    )
    if not rows:
        report = precision_data_blocked_report(
            reason="no GEFS-aligned weather precision rows after settlement/source gates",
            coverage={"events_found": len(events), "aligned_events": aligned_events},
            config=config,
        )
    else:
        report = run_weather_precision_rule(rows, config=config)
    coverage = dict(cast(Mapping[str, Any], report.get("coverage", {})))
    coverage.update(
        {
            "events_found": len(events),
            "settlement_aligned_events": aligned_events,
            "events_processed_with_gefs": processed_events,
            "events_excluded": len(excluded),
            "event_excluded_reasons": _reason_counts(excluded),
            "availability_checks": len(availability_checks),
            "member_count": len(members),
            "cities_requested": list(city_slugs),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "forecast_confidence_source": (
                "GEFS ensemble bucket probability at Wunderground settlement station"
            ),
            "price_source": (
                "CLOB prices-history latest YES/NO token prices before market end "
                "+ fixed ask proxy spread"
            ),
            "entry_window_source": "market_end_proxy_from_historical_prices_history",
            "ask_source_quality": "proxy_history",
            "true_ask_markets": 0,
            "proxy_ask_markets": len({str(row.get("market_slug")) for row in rows}),
            "ask_proxy_spread_cost": ASK_PROXY_SPREAD_COST,
        }
    )
    return {
        "briefing": "CODEX_OLYMPUS_79_WEATHER_GEFSPRECISION_CONSOLIDATED",
        "generated_at": datetime.now(UTC).isoformat(),
        "state": report.get("state"),
        "verdict": report.get("verdict"),
        "reason": report.get("reason"),
        "data_adequacy": report.get("data_adequacy"),
        "unlock_condition": report.get("unlock_condition"),
        "candidate_count_n": report.get("candidate_count_n"),
        "coverage": coverage,
        "standard_metrics": report.get("standard_metrics"),
        "benchmark_metrics": report.get("benchmark_metrics"),
        "multiple_testing": report.get("multiple_testing"),
        "best_candidate": report.get("best_candidate"),
        "gate_evidence": {
            "forecast_issue_rule": "latest GEFS 00/06/12/18 cycle strictly before decision_ts",
            "decoder": "eccodes.codes_grib_find_nearest",
            "decoder_value_self_check": (
                "sampled GRIB messages cross-check find_nearest against raw grid "
                "lat/lon/value scan; fail-loud if absolute difference exceeds 0.05K"
            ),
            "decoder_value_self_check_messages": decoder_self_check.checks_run,
            "interpolation": "nearest",
            "availability_sample": availability_checks[:20],
            "excluded_sample": excluded[:20],
            "probability_sample": probability_sample[:20],
            "sample_cache_dir": str(sample_cache_dir),
            "no_owm_weatherapi_dependency": True,
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "geoblock_gate": "yellow_research_only",
            "funding": "N/A prediction market",
            "survivor_light_ceiling": "SUGGESTIVE",
        },
    }


def _precision_rows_for_event(
    *,
    event: Mapping[str, Any],
    station_code: str,
    decision_ts: int,
    samples: Sequence[StationForecastSample],
    cycle: GefsCycle,
    leads: Sequence[int],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    rows: list[Mapping[str, object]] = []
    sample_rows: list[Mapping[str, object]] = []
    event_slug = str(event["slug"])
    city = _city_from_slug(event_slug)
    for market in _markets(event):
        bucket = _bucket_from_market(market)
        if bucket is None:
            continue
        yes_price = _latest_token_price(_token_at(market, 0), decision_ts)
        no_price = _latest_token_price(_token_at(market, 1), decision_ts)
        if yes_price is None or no_price is None:
            continue
        probability = bucket_probability_from_samples(
            samples=samples,
            bucket=bucket,
            cycle=cycle,
            leads=leads,
            station=STATIONS[station_code],
        )
        actual_won = _market_yes_won(market)
        yes_ask_proxy = min(1.0, yes_price + ASK_PROXY_SPREAD_COST)
        no_ask_proxy = min(1.0, no_price + ASK_PROXY_SPREAD_COST)
        row = {
            "event_slug": event_slug,
            "market_slug": str(market.get("slug")),
            "city": city,
            "station": station_code,
            "decision_ts": decision_ts,
            "forecast_issue_ts": probability.issue_ts,
            "entry_window": "market_end_proxy",
            "forecast_yes_probability": probability.probability,
            "yes_ask": yes_ask_proxy,
            "no_ask": no_ask_proxy,
            "actual_yes_won": actual_won,
            "member_count": probability.member_count,
            "members_in_bucket": probability.members_in_bucket,
            "bucket_label": bucket.label,
            "price_source_limit": (
                "historical CLOB ask unavailable; fixed spread added to YES/NO token prices-history"
            ),
        }
        rows.append(row)
        sample_rows.append(
            {
                "event_slug": event_slug,
                "market_slug": str(market.get("slug")),
                "bucket": bucket.label,
                "forecast_yes_probability": probability.probability,
                "yes_ask_proxy": yes_ask_proxy,
                "no_ask_proxy": no_ask_proxy,
                "actual_yes_won": actual_won,
                "member_count": probability.member_count,
            }
        )
    return rows, sample_rows


def _load_or_compute_samples(
    *,
    sample_cache_dir: Path,
    event_slug: str,
    station: Any,
    target_date: date,
    decision_ts: int,
    members: Sequence[str],
    text_fetcher: Callable[[str], str],
    byte_fetcher: Callable[[str, int, int], bytes],
    decoder: SampledDecoderValueSelfCheck,
    max_workers: int,
) -> tuple[list[StationForecastSample], GefsCycle, tuple[int, ...]]:
    key = hashlib.sha256(
        f"{event_slug}:{station.code}:{decision_ts}:{','.join(members)}:v3".encode()
    ).hexdigest()
    path = sample_cache_dir / f"{key}.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = [
            StationForecastSample(
                member=str(item["member"]),
                daily_max_f=float(item["daily_max_f"]),
                rounded_daily_max_f=int(item["rounded_daily_max_f"]),
                lead_hours_used=tuple(int(lead) for lead in item["lead_hours_used"]),
            )
            for item in raw["samples"]
        ]
        cycle = GefsCycle(
            issue_time=datetime.fromisoformat(str(raw["cycle"]["issue_time"])),
            cycle_hour=int(raw["cycle"]["cycle_hour"]),
        )
        leads = tuple(int(lead) for lead in raw["leads"])
        return samples, cycle, leads
    samples, cycle, leads = station_daily_samples_from_gefs(
        station=station,
        target_date=target_date,
        decision_ts=decision_ts,
        members=members,
        text_fetcher=text_fetcher,
        byte_fetcher=byte_fetcher,
        decoder=decoder,
        max_workers=max_workers,
    )
    path.write_text(
        json.dumps(
            {
                "event_slug": event_slug,
                "station": station.code,
                "decision_ts": decision_ts,
                "cycle": {
                    "issue_time": cycle.issue_time.isoformat(),
                    "cycle_hour": cycle.cycle_hour,
                },
                "leads": list(leads),
                "samples": [
                    {
                        "member": sample.member,
                        "daily_max_f": sample.daily_max_f,
                        "rounded_daily_max_f": sample.rounded_daily_max_f,
                        "lead_hours_used": list(sample.lead_hours_used),
                    }
                    for sample in samples
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return samples, cycle, leads


def _fetch_temperature_events(
    *,
    start_date: date,
    end_date: date,
    city_slugs: Sequence[str],
    max_events: int,
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    current = start_date
    while current <= end_date:
        for city_slug in city_slugs:
            if max_events and len(events) >= max_events:
                return events
            event = _fetch_event_by_slug(_temperature_event_slug(city_slug, current))
            if event is not None:
                events.append(event)
            time.sleep(0.01)
        current += timedelta(days=1)
    return events


def _cached_text_fetcher(cache_dir: Path) -> Callable[[str], str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(url: str) -> str:
        path = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.idx"
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = _fetch_text(url)
        path.write_text(text, encoding="utf-8")
        return text

    return fetch


def _cached_byte_fetcher(cache_dir: Path) -> Callable[[str, int, int], bytes]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(url: str, start: int, end: int) -> bytes:
        key = hashlib.sha256(f"{url}:{start}:{end}".encode()).hexdigest()
        path = cache_dir / f"{key}.grib2"
        if path.exists():
            return path.read_bytes()
        data = _fetch_range(url, start, end)
        path.write_bytes(data)
        return data

    return fetch


def _latest_token_price(token: str | None, decision_ts: int) -> float | None:
    if token is None:
        return None
    params = {
        "market": token,
        "startTs": decision_ts - 7 * 24 * 3600,
        "endTs": decision_ts,
        "fidelity": 60,
    }
    try:
        data = _get_json(f"{CLOB_BASE_URL}/prices-history?{urllib.parse.urlencode(params)}")
    except Exception:
        return None
    history = data.get("history") if isinstance(data, Mapping) else None
    if not isinstance(history, list):
        return None
    clean: list[float] = []
    for point in history:
        if not isinstance(point, Mapping) or _optional_int(point.get("t")) is None:
            continue
        price = _optional_float(point.get("p"))
        if price is not None and 0.0 < price < 1.0:
            clean.append(price)
    return clean[-1] if clean else None


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


def _temperature_event_slug(city_slug: str, day: date) -> str:
    month = day.strftime("%B").lower()
    return f"highest-temperature-in-{city_slug}-on-{month}-{day.day}-{day.year}"


def _city_from_slug(slug: str) -> str:
    for city_slug in sorted(CITY_SLUGS, key=len, reverse=True):
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


def _reason_counts(records: Sequence[Mapping[str, object]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = _str(record.get("reason")) or "unknown"
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


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}) or {})
    multiple = cast(Mapping[str, Any], payload.get("multiple_testing", {}) or {})
    lines = [
        "# CODEX OLYMPUS 79 Weather GEFS Precision Evidence",
        "",
        f"- State: `{payload.get('state')}`",
        f"- Verdict: `{payload.get('verdict')}`",
        f"- Data adequacy: `{payload.get('data_adequacy')}`",
        f"- Reason: {payload.get('reason')}",
        f"- Unlock condition: {payload.get('unlock_condition')}",
        f"- JSON: `{json_path}`",
        "",
        "## Coverage",
        f"- Events found: `{coverage.get('events_found')}`",
        f"- Settlement aligned events: `{coverage.get('settlement_aligned_events')}`",
        f"- Events processed with GEFS: `{coverage.get('events_processed_with_gefs')}`",
        f"- Observations: `{coverage.get('observations')}`",
        f"- Ask source quality: `{coverage.get('ask_source_quality')}`",
        "",
        "## Metrics",
        f"- Trades: `{metrics.get('trades')}`",
        f"- Wins: `{metrics.get('wins')}`",
        f"- Losses: `{metrics.get('losses')}`",
        f"- Win rate: `{metrics.get('win_rate')}`",
        f"- Mean net return: `{metrics.get('mean_net_return')}`",
        f"- Total net return: `{metrics.get('total_net_return')}`",
        "",
        "## Multiple Testing",
        f"- Candidate N: `{payload.get('candidate_count_n')}`",
        f"- FDR after: `{multiple.get('fdr_after')}`",
        f"- min_p: `{multiple.get('min_p')}`",
        f"- PBO: `{multiple.get('pbo')}`",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #79 Polymarket weather GEFS precision.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--city", action="append", default=None)
    parser.add_argument("--member-count", type=int, default=len(GEFS_MEMBERS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
