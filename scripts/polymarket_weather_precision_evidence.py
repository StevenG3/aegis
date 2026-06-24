#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from aegis.polymarket_weather_precision_rule import (
    WeatherPrecisionRuleConfig,
    precision_data_blocked_report,
    run_weather_precision_rule,
)
from aegis.polymarket_weather_relative_value import settlement_source_alignment
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus78"
DEFAULT_SPEC_NAME = "H-2026-06-24-consolidated-weather-precision-multi-source.json"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
USER_AGENT = "aegis-polymarket-weather-precision/0.1 read-only"
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


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.spec or output_dir / DEFAULT_SPEC_NAME
    payload = run_weather_precision_evidence(
        spec_path=spec_path,
        start_date=args.start_date,
        end_date=args.end_date,
        city_slugs=tuple(args.city) if args.city else CITY_SLUGS,
        max_events=args.max_events,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"polymarket-weather-precision-{stamp}.json"
    md_path = output_dir / f"polymarket-weather-precision-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": payload["state"],
                "verdict": payload["verdict"],
                "data_adequacy": payload["data_adequacy"],
                "reason": payload["reason"],
                "coverage": payload["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_weather_precision_evidence(
    *,
    spec_path: Path,
    start_date: date,
    end_date: date,
    city_slugs: Sequence[str],
    max_events: int = 0,
) -> Mapping[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    config = WeatherPrecisionRuleConfig()
    events = _fetch_temperature_events(
        start_date=start_date,
        end_date=end_date,
        city_slugs=city_slugs,
        max_events=max_events,
    )
    alignment = _alignment_summary(events)
    forecast_gate = _point_in_time_forecast_gate()
    coverage: dict[str, Any] = {
        "spec_id": spec.get("id"),
        "events_found": len(events),
        "city_slugs_requested": list(city_slugs),
        "cities_requested": len(city_slugs),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "settlement_source": "Wunderground station URL from Gamma resolutionSource",
        "settlement_alignment": alignment,
        "pit_forecast_gate": forecast_gate,
        "owm_weatherapi_required_by_spec": True,
        "minimum_city_gate": 20,
        "historical_price_gate": (
            "Polymarket Gamma resolved markets are enumerable by city slug; historical true "
            "CLOB ask/depth is not guaranteed by prices-history and must be treated separately."
        ),
    }
    if not forecast_gate["available"]:
        report = precision_data_blocked_report(
            reason=str(forecast_gate["reason"]),
            coverage=coverage,
            config=config,
        )
    else:
        # This branch is intentionally narrow: local PIT forecast archives may be supplied
        # in a future run, but the public repo never fabricates them from observations.
        rows = _load_point_in_time_rows(Path(str(forecast_gate["local_archive_path"])))
        report = run_weather_precision_rule(rows, config=config)
    return {
        "briefing": "CODEX_OLYMPUS_78_WEATHER_PRECISION_MULTI_SOURCE",
        "generated_at": datetime.now(UTC).isoformat(),
        "spec": _sanitized_spec(spec),
        "state": report.get("state"),
        "verdict": report.get("verdict"),
        "reason": report.get("reason"),
        "data_adequacy": report.get("data_adequacy"),
        "unlock_condition": report.get("unlock_condition"),
        "candidate_count_n": report.get("candidate_count_n"),
        "coverage": report.get("coverage"),
        "standard_metrics": report.get("standard_metrics"),
        "benchmark_metrics": report.get("benchmark_metrics"),
        "multiple_testing": report.get("multiple_testing"),
        "best_candidate": report.get("best_candidate"),
        "gate_evidence": {
            "data_feasibility_first": True,
            "forecast_issue_rule": (
                "forecast issue timestamp must be strictly before decision timestamp"
            ),
            "owm_official_capability": (
                "OpenWeather historical forecast is a History Forecast Bulk/export product; "
                "no configured local PIT export was present for this run."
            ),
            "weatherapi_official_capability": (
                "WeatherAPI historical endpoint is archived after the weather date; no configured "
                "local point-in-time forecast archive with issue timestamps was present."
            ),
            "no_substitution": (
                "GEFS #71D probabilities, historical observations, and current forecasts were not "
                "substituted for the spec-required OWM/WeatherAPI PIT forecasts."
            ),
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


def _point_in_time_forecast_gate() -> Mapping[str, Any]:
    archive_path = os.environ.get("AEGIS_WEATHER_PIT_FORECAST_ARCHIVE")
    if archive_path:
        path = Path(archive_path)
        if path.exists():
            return {
                "available": True,
                "source": "local_point_in_time_forecast_archive",
                "local_archive_path": str(path),
                "reason": "configured local PIT forecast archive exists",
            }
        return {
            "available": False,
            "source": "local_point_in_time_forecast_archive",
            "local_archive_path": archive_path,
            "reason": "configured AEGIS_WEATHER_PIT_FORECAST_ARCHIVE path does not exist",
        }
    has_owm_key = bool(os.environ.get("OPENWEATHERMAP_API_KEY") or os.environ.get("OWM_API_KEY"))
    has_weatherapi_key = bool(os.environ.get("WEATHERAPI_API_KEY"))
    return {
        "available": False,
        "source": "OpenWeatherMap/WeatherAPI",
        "local_archive_path": None,
        "has_openweathermap_key": has_owm_key,
        "has_weatherapi_key": has_weatherapi_key,
        "reason": (
            "missing point-in-time historical forecast archive with issue timestamps for "
            "OpenWeatherMap/WeatherAPI; current forecasts or historical observations cannot "
            "satisfy issue_ts < decision_ts"
        ),
    }


def _load_point_in_time_rows(path: Path) -> Sequence[Mapping[str, object]]:
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows: list[Mapping[str, object]] = []
    for child in sorted(path.glob("*.jsonl")):
        rows.extend(
            json.loads(line) for line in child.read_text(encoding="utf-8").splitlines() if line
        )
    return rows


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


def _alignment_summary(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    aligned = 0
    stations: set[str] = set()
    source_examples: list[str] = []
    for event in events:
        report = settlement_source_alignment(event)
        if report.get("aligned"):
            aligned += 1
        station = report.get("station")
        if isinstance(station, str):
            stations.add(station)
        sources = report.get("sources")
        if isinstance(sources, list) and sources and len(source_examples) < 5:
            source_examples.append(str(sources[0]))
    return {
        "events_checked": len(events),
        "aligned_events": aligned,
        "unaligned_events": len(events) - aligned,
        "stations": sorted(stations),
        "station_count": len(stations),
        "source_examples": source_examples,
    }


def _fetch_event_by_slug(slug: str) -> Mapping[str, Any] | None:
    try:
        data = _get_json(f"{GAMMA_BASE_URL}/events?{urllib.parse.urlencode({'slug': slug})}")
    except Exception:
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        return None
    return dict(data[0])


def _temperature_event_slug(city_slug: str, day: date) -> str:
    month = day.strftime("%B").lower()
    return f"highest-temperature-in-{city_slug}-on-{month}-{day.day}-{day.year}"


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _sanitized_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "id": spec.get("id"),
        "type": spec.get("type"),
        "alpha_or_carry": spec.get("alpha_or_carry"),
        "evidence_quality": spec.get("evidence_quality"),
        "status": spec.get("status"),
    }


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}) or {})
    multiple = cast(Mapping[str, Any], payload.get("multiple_testing", {}) or {})
    gate = cast(Mapping[str, Any], payload.get("gate_evidence", {}) or {})
    lines = [
        "# CODEX OLYMPUS 78 Weather Precision Multi-Source Evidence",
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
        f"- City slugs requested: `{coverage.get('cities_requested')}`",
        f"- Settlement alignment: `{coverage.get('settlement_alignment')}`",
        f"- PIT forecast gate: `{coverage.get('pit_forecast_gate')}`",
        "",
        "## Metrics",
        f"- Trades: `{metrics.get('trades')}`",
        f"- Wins: `{metrics.get('wins')}`",
        f"- Losses: `{metrics.get('losses')}`",
        f"- Mean net return: `{metrics.get('mean_net_return')}`",
        "",
        "## Multiple Testing",
        f"- Candidate N: `{payload.get('candidate_count_n')}`",
        f"- FDR after: `{multiple.get('fdr_after')}`",
        f"- PBO: `{multiple.get('pbo')}`",
        "",
        "## Gate Evidence",
        f"- Data feasibility first: `{gate.get('data_feasibility_first')}`",
        f"- No substitution: {gate.get('no_substitution')}",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #78 Polymarket weather precision evidence.")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--city", action="append", default=None)
    parser.add_argument("--max-events", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
