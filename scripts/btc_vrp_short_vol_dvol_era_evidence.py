#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.btc_vrp_short_vol import (
    ShortVolVrpConfig,
    btc_vrp_data_blocked_report,
    run_btc_short_vol_vrp,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus80"
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2/public"
BINANCE_SPOT_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
USER_AGENT = "aegis-btc-vrp-short-vol-dvol-era/0.1 read-only"
DVOL_START = "2021-03-24"
END_DATE = "2026-06-01"
REQUIRED_CRASH_WINDOWS = {
    "luna_2022_05": ("2022-05-01", "2022-06-01"),
    "ftx_2022_11": ("2022-11-01", "2022-12-01"),
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    tenor_days: int
    option_spread_cost: float
    hedge_notional: float
    tail_threshold: float
    tail_multiplier: float
    tail_cap: float | None


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("atm_straddle_7d", 7, 0.012, 0.50, 0.12, 2.0, None),
    VariantSpec("atm_straddle_14d", 14, 0.014, 0.50, 0.12, 2.0, None),
    VariantSpec("otm_strangle_7d", 7, 0.010, 0.35, 0.15, 1.5, None),
    VariantSpec("otm_strangle_14d", 14, 0.012, 0.35, 0.15, 1.5, None),
    VariantSpec("iron_condor_7d", 7, 0.008, 0.25, 0.18, 1.0, 0.25),
    VariantSpec("iron_condor_14d", 14, 0.010, 0.25, 0.18, 1.0, 0.25),
)


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_btc_vrp_short_vol_dvol_era_evidence()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"btc-vrp-short-vol-dvol-era-{stamp}.json"
    md_path = output_dir / f"btc-vrp-short-vol-dvol-era-{stamp}.md"
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


def run_btc_vrp_short_vol_dvol_era_evidence() -> Mapping[str, Any]:
    config = ShortVolVrpConfig(max_drawdown_limit=-0.30)
    start_ms = _to_ms(DVOL_START)
    end_ms = _to_ms(END_DATE)
    dvol = _dvol_history(start_ms, end_ms)
    prices = _spot_daily_closes(start_ms, end_ms + 20 * 24 * 3600 * 1000)
    funding = _funding_rates(start_ms, end_ms + 20 * 24 * 3600 * 1000)
    crash_coverage = {
        name: _crash_window_coverage(start, end, dvol, prices)
        for name, (start, end) in REQUIRED_CRASH_WINDOWS.items()
    }
    coverage: dict[str, Any] = {
        "iv_source": "Deribit public BTC DVOL index close (proxy IV, not executable chain bid/ask)",
        "price_source": "Binance public BTCUSDT 1d klines",
        "funding_source": "Binance public BTCUSDT perpetual fundingRate",
        "dvol_rows": len(dvol),
        "price_rows": len(prices),
        "funding_rows": len(funding),
        "dvol_first_ts": dvol[0][0] if dvol else None,
        "dvol_last_ts": dvol[-1][0] if dvol else None,
        "required_crash_windows": list(REQUIRED_CRASH_WINDOWS),
        "crash_window_coverage": crash_coverage,
        "predeclared_variants": [variant.name for variant in VARIANTS],
        "max_drawdown_limit": config.max_drawdown_limit,
        "missing_2020_covid": "DVOL did not exist; recorded as limitation, not hard gate",
    }
    missing_required = [
        name
        for name, report in crash_coverage.items()
        if int(cast(Mapping[str, object], report)["dvol_rows"]) == 0
        or int(cast(Mapping[str, object], report)["price_rows"]) == 0
    ]
    if missing_required:
        reason = f"DVOL-era required crash windows missing: {', '.join(missing_required)}"
        report = btc_vrp_data_blocked_report(reason=reason, coverage=coverage, config=config)
    elif not funding:
        report = btc_vrp_data_blocked_report(
            reason="Binance funding history unavailable; cannot compute delta-hedge funding cost",
            coverage=coverage,
            config=config,
        )
    else:
        rows = _build_proxy_rows(dvol=dvol, prices=prices, funding=funding)
        report = run_btc_short_vol_vrp(rows, config=config)
        coverage = {**coverage, **cast(Mapping[str, Any], report.get("coverage", {}))}
    return {
        "briefing": "CODEX_OLYMPUS_80B_BTC_SHORT_VOL_VRP_DVOL_ERA",
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
            "dvol_era_only": True,
            "covid_2020_not_hard_gate": True,
            "required_tail_windows": REQUIRED_CRASH_WINDOWS,
            "proxy_limit": (
                "DVOL index is not executable option-chain bid/ask; spread and tail costs are "
                "predeclared conservative proxies."
            ),
            "no_live_or_order_access": True,
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }


def _build_proxy_rows(
    *,
    dvol: Sequence[tuple[int, float]],
    prices: Mapping[int, float],
    funding: Mapping[int, float],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    price_days = sorted(prices)
    for timestamp, dvol_close in dvol:
        current_day = _day_start(timestamp)
        if current_day not in prices:
            continue
        for variant in VARIANTS:
            expiry_day = current_day + variant.tenor_days * 24 * 3600 * 1000
            window_days = [day for day in price_days if current_day <= day <= expiry_day]
            if len(window_days) < variant.tenor_days + 1:
                continue
            returns = [
                math.log(prices[window_days[index]] / prices[window_days[index - 1]])
                for index in range(1, len(window_days))
            ]
            realized_vol = math.sqrt(sum(value * value for value in returns) * 365.0 / len(returns))
            max_abs_daily = max(abs(value) for value in returns) if returns else 0.0
            tail_excess = max(0.0, max_abs_daily - variant.tail_threshold)
            raw_tail = tail_excess * variant.tail_multiplier
            tail_loss = (
                min(raw_tail, variant.tail_cap) if variant.tail_cap is not None else raw_tail
            )
            funding_cost = _funding_cost(
                funding=funding,
                start_day=current_day,
                end_day=expiry_day,
                hedge_notional=variant.hedge_notional,
            )
            rebalances = len(returns)
            rows.append(
                {
                    "variant": variant.name,
                    "iv_ts": current_day,
                    "expiry_ts": expiry_day,
                    "implied_vol": dvol_close / 100.0,
                    "realized_vol": realized_vol,
                    "variance_year_fraction": variant.tenor_days / 365.0,
                    "option_spread_cost": variant.option_spread_cost,
                    "hedge_fee_cost": rebalances * 0.0004 * variant.hedge_notional,
                    "hedge_slippage_cost": rebalances * 0.0003 * variant.hedge_notional,
                    "funding_cost": funding_cost,
                    "tail_loss": tail_loss,
                }
            )
    return rows


def _funding_cost(
    *,
    funding: Mapping[int, float],
    start_day: int,
    end_day: int,
    hedge_notional: float,
) -> float:
    selected = [abs(rate) for ts, rate in funding.items() if start_day <= ts <= end_day]
    return sum(selected) * hedge_notional


def _crash_window_coverage(
    start_date: str,
    end_date: str,
    dvol: Sequence[tuple[int, float]],
    prices: Mapping[int, float],
) -> Mapping[str, Any]:
    start = _to_ms(start_date)
    end = _to_ms(end_date)
    dvol_rows = [row for row in dvol if start <= row[0] <= end]
    price_rows = [ts for ts in prices if start <= ts <= end]
    return {
        "start": start_date,
        "end": end_date,
        "dvol_rows": len(dvol_rows),
        "price_rows": len(price_rows),
        "dvol_first": dvol_rows[0] if dvol_rows else None,
        "dvol_last": dvol_rows[-1] if dvol_rows else None,
    }


def _dvol_history(start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    cursor = start_ms
    chunk = 240 * 24 * 3600 * 1000
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk)
        params = {
            "currency": "BTC",
            "start_timestamp": cursor,
            "end_timestamp": chunk_end,
            "resolution": "1D",
        }
        data = _get_json(
            f"{DERIBIT_BASE_URL}/get_volatility_index_data?{urllib.parse.urlencode(params)}"
        )
        result = data.get("result") if isinstance(data, Mapping) else None
        raw_rows = result.get("data") if isinstance(result, Mapping) else None
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, list) and len(row) >= 5:
                    rows.append((int(float(row[0])), float(row[4])))
        cursor = chunk_end + 24 * 3600 * 1000
    dedup = {timestamp: close for timestamp, close in rows}
    return sorted(dedup.items())


def _spot_daily_closes(start_ms: int, end_ms: int) -> dict[int, float]:
    rows: dict[int, float] = {}
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get_json(f"{BINANCE_SPOT_URL}/api/v3/klines?{urllib.parse.urlencode(params)}")
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if isinstance(row, list) and len(row) >= 5:
                rows[int(row[0])] = float(row[4])
        last = int(data[-1][0])
        next_cursor = last + 24 * 3600 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def _funding_rates(start_ms: int, end_ms: int) -> dict[int, float]:
    rows: dict[int, float] = {}
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get_json(
            f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate?{urllib.parse.urlencode(params)}"
        )
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if isinstance(row, Mapping):
                rows[int(row["fundingTime"])] = float(row["fundingRate"])
        last = int(cast(Mapping[str, object], data[-1])["fundingTime"])
        next_cursor = last + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _to_ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def _day_start(timestamp_ms: int) -> int:
    day = datetime.fromtimestamp(timestamp_ms / 1000, UTC).date()
    return int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}) or {})
    multiple = cast(Mapping[str, Any], payload.get("multiple_testing", {}) or {})
    return "\n".join(
        [
            "# CODEX OLYMPUS 80B BTC Short Vol VRP DVOL-Era Evidence",
            "",
            f"- State: `{payload.get('state')}`",
            f"- Verdict: `{payload.get('verdict')}`",
            f"- Data adequacy: `{payload.get('data_adequacy')}`",
            f"- Reason: {payload.get('reason')}",
            f"- Unlock condition: {payload.get('unlock_condition')}",
            f"- JSON: `{json_path}`",
            "",
            "## Coverage",
            f"- DVOL rows: `{coverage.get('dvol_rows')}`",
            f"- Price rows: `{coverage.get('price_rows')}`",
            f"- Funding rows: `{coverage.get('funding_rows')}`",
            f"- Crash coverage: `{coverage.get('crash_window_coverage')}`",
            "",
            "## Metrics",
            f"- Variant: `{metrics.get('variant')}`",
            f"- Mean net return: `{metrics.get('mean_net_return')}`",
            f"- MaxDD: `{metrics.get('max_drawdown')}`",
            f"- CVaR95: `{metrics.get('cvar_95')}`",
            f"- CVaR99: `{metrics.get('cvar_99')}`",
            f"- Worst trade: `{metrics.get('worst_trade')}`",
            f"- Worst month: `{metrics.get('worst_month')}`",
            "",
            "## Multiple Testing",
            f"- FDR after: `{multiple.get('fdr_after')}`",
            f"- PBO: `{multiple.get('pbo')}`",
        ]
    ) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #80B BTC short-vol VRP DVOL-era evidence.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
