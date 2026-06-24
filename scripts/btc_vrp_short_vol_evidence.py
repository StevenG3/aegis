#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.btc_vrp_short_vol import ShortVolVrpConfig, btc_vrp_data_blocked_report
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus80"
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2/public"
BINANCE_BASE_URL = "https://api.binance.com"
USER_AGENT = "aegis-btc-vrp-short-vol/0.1 read-only"
CRASH_WINDOWS = {
    "covid_2020_03": ("2020-03-01", "2020-04-01"),
    "luna_2022_05": ("2022-05-01", "2022-06-01"),
    "ftx_2022_11": ("2022-11-01", "2022-12-01"),
}


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_btc_vrp_short_vol_evidence()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"btc-vrp-short-vol-{stamp}.json"
    md_path = output_dir / f"btc-vrp-short-vol-{stamp}.md"
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


def run_btc_vrp_short_vol_evidence() -> Mapping[str, Any]:
    config = ShortVolVrpConfig(max_drawdown_limit=-0.30)
    crash_coverage = {
        name: _crash_window_coverage(start, end) for name, (start, end) in CRASH_WINDOWS.items()
    }
    dvol_start = _dvol_window(_to_ms("2019-01-01"), _to_ms("2026-06-01"))
    coverage: dict[str, Any] = {
        "iv_source": "Deribit public get_volatility_index_data BTC DVOL",
        "price_source": "Binance public BTCUSDT 1d klines coverage probe",
        "required_crash_windows": list(CRASH_WINDOWS),
        "crash_window_coverage": crash_coverage,
        "dvol_long_probe": {
            "rows": len(dvol_start),
            "first": dvol_start[0] if dvol_start else None,
            "last": dvol_start[-1] if dvol_start else None,
        },
        "predeclared_structures": ["atm_straddle", "otm_strangle", "iron_condor"],
        "predeclared_tenors_days": [7, 14],
        "tail_max_drawdown_limit": config.max_drawdown_limit,
    }
    missing = [
        name
        for name, report in crash_coverage.items()
        if int(cast(Mapping[str, object], report)["dvol_rows"]) == 0
    ]
    if missing:
        reason = (
            "Deribit public BTC DVOL history is missing required crash windows: "
            f"{', '.join(missing)}; cannot test short-vol tail survivability without them"
        )
        report = btc_vrp_data_blocked_report(reason=reason, coverage=coverage, config=config)
    else:
        report = btc_vrp_data_blocked_report(
            reason=(
                "PIT option-chain bid/ask IV and hedge funding rows are required before "
                "running strategy P&L; DVOL-only coverage probe passed"
            ),
            coverage=coverage,
            config=config,
        )
    return {
        "briefing": "CODEX_OLYMPUS_80_BTC_SHORT_VOL_VRP",
        "generated_at": datetime.now(UTC).isoformat(),
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
        "gate_evidence": {
            "data_feasibility_first": True,
            "no_crash_exclusion": True,
            "required_tail_windows": CRASH_WINDOWS,
            "blocked_before_backtest": bool(missing),
            "option_chain_requirement": (
                "DVOL is a volatility index; executable short-vol backtest still needs PIT "
                "option bid/ask IV by tenor/strike plus hedge funding history."
            ),
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }


def _crash_window_coverage(start_date: str, end_date: str) -> Mapping[str, Any]:
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    dvol = _dvol_window(start_ms, end_ms)
    klines = _binance_klines(start_ms, end_ms)
    return {
        "start": start_date,
        "end": end_date,
        "dvol_rows": len(dvol),
        "dvol_first": dvol[0] if dvol else None,
        "dvol_last": dvol[-1] if dvol else None,
        "btc_price_rows": len(klines),
        "btc_price_first_open_time": klines[0][0] if klines else None,
    }


def _dvol_window(start_ms: int, end_ms: int) -> list[list[float]]:
    params = {
        "currency": "BTC",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "resolution": "1D",
    }
    data = _get_json(
        f"{DERIBIT_BASE_URL}/get_volatility_index_data?{urllib.parse.urlencode(params)}"
    )
    result = data.get("result") if isinstance(data, Mapping) else None
    rows = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(rows, list):
        return []
    clean: list[list[float]] = []
    for row in rows:
        if isinstance(row, list) and len(row) >= 5:
            clean.append([float(value) for value in row[:5]])
    return clean


def _binance_klines(start_ms: int, end_ms: int) -> list[list[object]]:
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    data = _get_json(f"{BINANCE_BASE_URL}/api/v3/klines?{urllib.parse.urlencode(params)}")
    return [row for row in data if isinstance(row, list)] if isinstance(data, list) else []


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _to_ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    lines = [
        "# CODEX OLYMPUS 80 BTC Short Vol VRP Evidence",
        "",
        f"- State: `{payload.get('state')}`",
        f"- Verdict: `{payload.get('verdict')}`",
        f"- Data adequacy: `{payload.get('data_adequacy')}`",
        f"- Reason: {payload.get('reason')}",
        f"- Unlock condition: {payload.get('unlock_condition')}",
        f"- JSON: `{json_path}`",
        "",
        "## Coverage",
        f"- IV source: `{coverage.get('iv_source')}`",
        f"- Required crash windows: `{coverage.get('required_crash_windows')}`",
        f"- Crash coverage: `{coverage.get('crash_window_coverage')}`",
        "",
        "## Tail Gate",
        f"- Max drawdown limit: `{coverage.get('tail_max_drawdown_limit')}`",
        "- Strategy P&L not run because required crash IV coverage is missing.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #80 BTC short-vol VRP data gate.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
