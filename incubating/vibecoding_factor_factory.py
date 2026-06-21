from __future__ import annotations

import importlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import CostModel
from aegis.private_paths import private_dir_from_cli
from aegis.vibecoding_factor_factory import (
    DEFAULT_COST_MODEL,
    VibeBar,
    VibeFactoryConfig,
    run_vibecoding_factor_factory,
)

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DEFAULT_TIMEFRAMES = ("1h", "4h")
DEFAULT_START = "2021-01-01T00:00:00Z"
DEFAULT_END = "2026-06-01T00:00:00Z"


def main() -> int:
    generated_at = datetime.now(UTC)
    output_dir = private_dir_from_cli(
        os.getenv("VIBECODING_FACTORY_OUTPUT_DIR"),
        default_task="olympus69",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source = os.getenv("VIBECODING_FACTORY_SOURCE", "binance").strip() or "binance"
    symbols = _csv_env("VIBECODING_FACTORY_SYMBOLS", DEFAULT_SYMBOLS)
    timeframes = _csv_env("VIBECODING_FACTORY_TIMEFRAMES", DEFAULT_TIMEFRAMES)
    start = os.getenv("VIBECODING_FACTORY_START", DEFAULT_START)
    end = os.getenv("VIBECODING_FACTORY_END", DEFAULT_END)
    exchange = _exchange(source)
    start_ms = _parse_date(exchange, start)
    end_ms = _parse_date(exchange, end)
    frames: dict[str, list[VibeBar]] = {}
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        for timeframe in timeframes:
            key = f"{symbol.replace('/', '')}:{timeframe}"
            try:
                rows = _fetch_ohlcv(exchange, symbol, timeframe, since_ms=start_ms, end_ms=end_ms)
                bars = _bars_from_rows(rows)
                if bars:
                    frames[key] = bars
            except Exception as exc:  # noqa: BLE001
                failures.append({"symbol": symbol, "timeframe": timeframe, "error": str(exc)})
    _close_exchange(exchange)

    result = dict(
        run_vibecoding_factor_factory(
            frames,
            config=VibeFactoryConfig(),
            cost_model=CostModel(
                fee_bps=_float_env("VIBECODING_FACTORY_FEE_BPS", DEFAULT_COST_MODEL.fee_bps),
                slippage_bps=_float_env(
                    "VIBECODING_FACTORY_SLIPPAGE_BPS", DEFAULT_COST_MODEL.slippage_bps
                ),
                funding_label=DEFAULT_COST_MODEL.funding_label,
            ),
            generated_at=generated_at,
        )
    )
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_69_VIBECODING_FACTOR_FACTORY",
        "input": {
            "source": source,
            "symbols": symbols,
            "timeframes": timeframes,
            "requested_start": start,
            "requested_end": end,
            "actual_ranges": {
                key: {
                    "bars": len(bars),
                    "start": _iso(bars[0].timestamp) if bars else None,
                    "end": _iso(bars[-1].timestamp) if bars else None,
                }
                for key, bars in frames.items()
            },
            "fetch_failures": failures,
        },
        "public_boundary": (
            "Detailed evidence artifact is private. Public repo contains only generic code and "
            "synthetic tests; no credentials, account data, or raw private research results."
        ),
        "result": result,
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"vibecoding-factor-factory-{stamp}.json"
    md_path = output_dir / f"vibecoding-factor-factory-{stamp}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "frames": len(frames),
                "fetch_failures": failures,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _exchange(source: str) -> Any:
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, source)
    return factory({"enableRateLimit": True, "timeout": 20_000})


def _fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    end_ms: int,
) -> list[list[float]]:
    rows: list[list[float]] = []
    cursor = since_ms
    seen = set[int]()
    while cursor < end_ms:
        batch = cast(
            list[list[float]],
            exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000),
        )
        if not batch:
            break
        advanced = False
        for raw in batch:
            timestamp = int(raw[0])
            if timestamp >= end_ms:
                continue
            if timestamp not in seen:
                rows.append(raw)
                seen.add(timestamp)
            if timestamp >= cursor:
                cursor = timestamp + 1
                advanced = True
        if not advanced:
            break
        time.sleep(float(getattr(exchange, "rateLimit", 200)) / 1000.0)
    return sorted(rows, key=lambda row: row[0])


def _bars_from_rows(rows: list[list[float]]) -> list[VibeBar]:
    return [
        VibeBar(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
        if len(row) >= 6
    ]


def _parse_date(exchange: Any, value: str) -> int:
    parsed = exchange.parse8601(value)
    if parsed is None:
        raise ValueError(f"could not parse date {value!r}")
    return int(parsed)


def _close_exchange(exchange: Any) -> None:
    close = getattr(exchange, "close", None)
    if callable(close):
        close()


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    result = cast(dict[str, Any], payload["result"])
    multiple = cast(dict[str, Any], result.get("multiple_testing", {}))
    health = cast(dict[str, Any], result.get("health", {}))
    data = cast(dict[str, Any], result.get("data", {}))
    frames = cast(dict[str, Any], data.get("frames", {}))
    return "\n".join(
        [
            "# Olympus #69 Vibecoding Factor Factory Evidence",
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- verdict: {result.get('verdict')}",
            f"- reason: {result.get('reason')}",
            f"- candidate_count_n: {multiple.get('candidate_count_n')}",
            f"- fdr_survivors: {multiple.get('fdr_survivors')}",
            f"- health: {health.get('score')}",
            f"- frames: {len(frames)}",
            f"- json: {json_path}",
            "",
            "Funding is N/A because this script evaluates Binance spot only.",
            "This artifact is private evidence, not a trading signal.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
