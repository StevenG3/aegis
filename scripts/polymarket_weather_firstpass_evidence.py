#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from aegis.polymarket_weather_relative_value import (
    TRIAL_COUNT_N,
    settlement_source_alignment,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus71"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
GEFS_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
USER_AGENT = "aegis-polymarket-weather-firstpass/0.1 read-only"

US_CITY_SLUGS = ("nyc", "miami", "los-angeles")
DEFAULT_START_DATE = date(2026, 5, 1)
DEFAULT_END_DATE = date(2026, 6, 22)


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_gate_probe(
        start_date=args.start_date,
        end_date=args.end_date,
        city_slugs=tuple(args.city),
        gamma_page_limit=args.gamma_page_limit,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"polymarket-weather-firstpass-{stamp}.json"
    md_path = output_dir / f"polymarket-weather-firstpass-{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(result, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "verdict": result["verdict"],
                "reason": result["reason"],
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_gate_probe(
    *,
    start_date: date,
    end_date: date,
    city_slugs: Sequence[str],
    gamma_page_limit: int = 2_100,
) -> Mapping[str, Any]:
    slug_results = []
    events: list[Mapping[str, Any]] = []
    for city_slug in city_slugs:
        for day in _date_range(start_date, end_date):
            slug = _temperature_event_slug(city_slug, day)
            status, event = _fetch_event_by_slug(slug)
            slug_results.append({"slug": slug, "status": status})
            if event is not None:
                events.append(event)
            time.sleep(0.02)

    gamma_wall = _gamma_422_probe(limit=100, max_offset=gamma_page_limit)
    aligned_events = []
    misaligned_events = []
    resolved_events = []
    price_history_ok = 0
    price_history_missing = 0
    gefs_idx_ok = 0
    gefs_idx_missing = 0

    for event in events:
        alignment = settlement_source_alignment(event)
        record = {
            "slug": event.get("slug"),
            "title": event.get("title"),
            "closed": event.get("closed"),
            "endDate": event.get("endDate"),
            "station": alignment.get("station"),
            "sources": alignment.get("sources"),
            "aligned": alignment.get("aligned"),
            "market_count": len(event.get("markets") or []),
        }
        if alignment.get("aligned"):
            aligned_events.append(record)
        else:
            misaligned_events.append(record)
        if bool(event.get("closed")):
            resolved_events.append(record)
            token = _first_yes_token(event)
            if token is not None and _prices_history_available(token, event):
                price_history_ok += 1
            else:
                price_history_missing += 1
            if _gefs_index_available(event):
                gefs_idx_ok += 1
            else:
                gefs_idx_missing += 1

    coverage = {
        "city_slugs": list(city_slugs),
        "slug_dates": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "slugs_probed": len(slug_results),
        "events_found": len(events),
        "aligned_settlement_source_events": len(aligned_events),
        "misaligned_settlement_source_events": len(misaligned_events),
        "resolved_events": len(resolved_events),
        "resolved_events_with_prices_history": price_history_ok,
        "resolved_events_missing_prices_history": price_history_missing,
        "resolved_events_with_gefs_idx": gefs_idx_ok,
        "resolved_events_missing_gefs_idx": gefs_idx_missing,
        "gamma_422_probe": gamma_wall,
        "candidate_count_n": TRIAL_COUNT_N,
    }
    reason = (
        "Polymarket city high-temperature metadata, settlement source, historical prices, "
        "and GEFS archive indexes are partially available, but this firstpass cannot compute "
        "station-level ensemble bucket probabilities without a vetted GRIB decoder/interpolator; "
        "fail-closed rather than substituting non-equivalent weather data"
    )
    return {
        "briefing": "CODEX_OLYMPUS_71_POLYMARKET_WEATHER_FIRSTPASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "coverage": coverage,
        "gate_evidence": {
            "sample_aligned_events": aligned_events[:10],
            "sample_misaligned_events": misaligned_events[:10],
            "slug_probe_sample": slug_results[:20],
            "forecast_source": {
                "gefs_archive": GEFS_BASE_URL,
                "idx_available": gefs_idx_ok,
                "limitation": (
                    "GEFS pgrb2sp25 index files expose TMAX/TMP records, but raw GRIB2 "
                    "needs a decoder and station interpolation before probabilities can be "
                    "computed without lookahead."
                ),
            },
        },
        "multiple_testing": {
            "candidate_count_n": TRIAL_COUNT_N,
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": "not run under INSUFFICIENT data gate"},
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
            "geoblock_gate": "yellow_research_only",
            "funding": "N/A prediction market",
        },
    }


def _fetch_event_by_slug(slug: str) -> tuple[str, Mapping[str, Any] | None]:
    url = f"{GAMMA_BASE_URL}/events?{urllib.parse.urlencode({'slug': slug})}"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}", None
    except Exception as exc:
        return f"error_{exc.__class__.__name__}", None
    if not isinstance(data, list) or not data:
        return "not_found", None
    event = data[0]
    if not isinstance(event, Mapping):
        return "unexpected_response", None
    return "found", dict(event)


def _gamma_422_probe(*, limit: int, max_offset: int) -> Mapping[str, Any]:
    last_ok: int | None = None
    for offset in range(0, max_offset + limit, limit):
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
        }
        url = f"{GAMMA_BASE_URL}/events?{urllib.parse.urlencode(params)}"
        try:
            data = _get_json(url)
        except urllib.error.HTTPError as exc:
            return {"hit_422": exc.code == 422, "offset": offset, "last_ok_offset": last_ok}
        if not isinstance(data, list) or not data:
            return {"hit_422": False, "offset": offset, "last_ok_offset": last_ok}
        last_ok = offset
        time.sleep(0.02)
    return {"hit_422": False, "offset": max_offset, "last_ok_offset": last_ok}


def _first_yes_token(event: Mapping[str, Any]) -> str | None:
    markets = event.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if not isinstance(market, Mapping):
            continue
        raw = market.get("clobTokenIds")
        if not isinstance(raw, str):
            continue
        try:
            tokens = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(tokens, list) and tokens:
            return str(tokens[0])
    return None


def _prices_history_available(token: str, event: Mapping[str, Any]) -> bool:
    end_ts = _event_end_ts(event)
    if end_ts is None:
        return False
    params = {
        "market": token,
        "startTs": end_ts - 3 * 24 * 3600,
        "endTs": end_ts + 24 * 3600,
        "fidelity": 60,
    }
    try:
        data = _get_json(f"{CLOB_BASE_URL}/prices-history?{urllib.parse.urlencode(params)}")
    except Exception:
        return False
    return isinstance(data, Mapping) and bool(data.get("history"))


def _gefs_index_available(event: Mapping[str, Any]) -> bool:
    end_ts = _event_end_ts(event)
    if end_ts is None:
        return False
    issued = datetime.fromtimestamp(end_ts, UTC) - timedelta(days=1)
    ymd = issued.strftime("%Y%m%d")
    url = (
        f"{GEFS_BASE_URL}/gefs.{ymd}/00/atmos/pgrb2sp25/"
        "gep01.t00z.pgrb2s.0p25.f003.idx"
    )
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as response:
            return int(response.status) == 200
    except Exception:
        return False


def _event_end_ts(event: Mapping[str, Any]) -> int | None:
    value = event.get("endDate")
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp())


def _temperature_event_slug(city_slug: str, day: date) -> str:
    month = day.strftime("%B").lower()
    return f"highest-temperature-in-{city_slug}-on-{month}-{day.day}-{day.year}"


def _date_range(start: date, end: date) -> Sequence[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload["coverage"])
    multiple = cast(Mapping[str, Any], payload["multiple_testing"])
    lines = [
        "# CODEX OLYMPUS 71 Polymarket Weather Firstpass",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Reason: {payload['reason']}",
        f"- JSON: `{json_path}`",
        "",
        "## Coverage",
        f"- Slugs probed: `{coverage.get('slugs_probed')}`",
        f"- Events found: `{coverage.get('events_found')}`",
        f"- Aligned settlement-source events: `{coverage.get('aligned_settlement_source_events')}`",
        f"- Resolved events: `{coverage.get('resolved_events')}`",
        f"- Resolved with prices-history: `{coverage.get('resolved_events_with_prices_history')}`",
        f"- Resolved with GEFS idx: `{coverage.get('resolved_events_with_gefs_idx')}`",
        f"- Gamma 422 probe: `{coverage.get('gamma_422_probe')}`",
        "",
        "## Multiple Testing",
        f"- Candidate N: `{multiple.get('candidate_count_n')}`",
        f"- FDR after: `{multiple.get('fdr_after')}`",
        f"- PBO: `{multiple.get('pbo')}`",
        "",
        "## Safety",
        "- Read-only public APIs only; no wallet, order API, account API, or live trading.",
        "- Funding is N/A for prediction markets.",
        "- Geoblock remains yellow/research-only.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Polymarket city high-temperature markets for #71 firstpass."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--city", action="append", default=list(US_CITY_SLUGS))
    parser.add_argument("--gamma-page-limit", type=int, default=2_100)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
