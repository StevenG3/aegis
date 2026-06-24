from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2/public"
BINANCE_SPOT_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
USER_AGENT = "aegis-btc-vrp-public-data/0.1 read-only"
DVOL_START = "2021-03-24"
END_DATE = "2026-06-01"
DAY_MS = 24 * 3600 * 1000
REQUIRED_CRASH_WINDOWS = {
    "luna_2022_05": ("2022-05-01", "2022-06-01"),
    "ftx_2022_11": ("2022-11-01", "2022-12-01"),
}


def dvol_history(start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    cursor = start_ms
    chunk = 240 * DAY_MS
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk)
        params = {
            "currency": "BTC",
            "start_timestamp": cursor,
            "end_timestamp": chunk_end,
            "resolution": "1D",
        }
        data = get_json(
            f"{DERIBIT_BASE_URL}/get_volatility_index_data?{urllib.parse.urlencode(params)}"
        )
        result = data.get("result") if isinstance(data, Mapping) else None
        raw_rows = result.get("data") if isinstance(result, Mapping) else None
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, list) and len(row) >= 5:
                    rows.append((int(float(row[0])), float(row[4])))
        cursor = chunk_end + DAY_MS
    dedup = {timestamp: close for timestamp, close in rows}
    return sorted(dedup.items())


def spot_daily_closes(start_ms: int, end_ms: int) -> dict[int, float]:
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
        data = get_json(f"{BINANCE_SPOT_URL}/api/v3/klines?{urllib.parse.urlencode(params)}")
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if isinstance(row, list) and len(row) >= 5:
                rows[int(row[0])] = float(row[4])
        last = int(data[-1][0])
        next_cursor = last + DAY_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def funding_rates(start_ms: int, end_ms: int) -> dict[int, float]:
    rows: dict[int, float] = {}
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = get_json(
            f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate?{urllib.parse.urlencode(params)}"
        )
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if isinstance(row, Mapping):
                rows[int(row["fundingTime"])] = float(row["fundingRate"])
        last_row = data[-1]
        if not isinstance(last_row, Mapping):
            break
        last = int(cast(str | int | float, last_row["fundingTime"]))
        next_cursor = last + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def crash_window_coverage(
    start_date: str,
    end_date: str,
    dvol: list[tuple[int, float]],
    prices: Mapping[int, float],
) -> Mapping[str, Any]:
    start = to_ms(start_date)
    end = to_ms(end_date)
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


def get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def to_ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
