from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import CostModel
from aegis.domestic_futures_adx_prefilter import (
    DEFAULT_CONFIG,
    FuturesBar,
    run_underlying_adx_prefilter,
)
from aegis.private_paths import private_dir_from_cli

BRIEFING = "CODEX_OLYMPUS_73_UNDERLYING_ADX_PREFILTER"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2026-06-20"
DEFAULT_SYMBOLS: dict[str, str] = {
    "黄金": "AU0",
    "沪铜": "CU0",
    "PTA": "TA0",
    "铁矿石": "I0",
    "豆粕": "M0",
    "玉米": "C0",
    "豆油": "Y0",
}


def main() -> int:
    generated_at = datetime.now(timezone.utc)  # noqa: UP017 - host Python can be 3.10.
    output_dir = private_dir_from_cli(
        os.getenv("DOMESTIC_FUTURES_ADX_OUTPUT_DIR"),
        default_task="olympus73",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start = os.getenv("DOMESTIC_FUTURES_ADX_START", DEFAULT_START)
    end = os.getenv("DOMESTIC_FUTURES_ADX_END", DEFAULT_END)
    symbols = _symbols_from_env()
    fetch_report = _fetch_akshare_main_continuous(symbols, start=start, end=end)
    cost_model = CostModel(
        fee_bps=_float_env("DOMESTIC_FUTURES_ADX_FEE_BPS", 2.0),
        slippage_bps=_float_env("DOMESTIC_FUTURES_ADX_SLIPPAGE_BPS", 3.0),
        funding_label="N/A for listed domestic futures; no perp funding",
    )
    result = run_underlying_adx_prefilter(
        fetch_report["bars_by_symbol"],
        required_symbols=tuple(symbols),
        config=DEFAULT_CONFIG,
        cost_model=cost_model,
        data_source="akshare.futures_main_sina",
        roll_method=(
            "Sina current/main continuous contract code *0; not a self-rebuilt PIT contract "
            "chain. Roll construction is accepted only for cheap prefilter and caps verdict."
        ),
    )
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": BRIEFING,
        "ev_newness": (
            "确认模式: 技术趋势类(EMA/ADX)在商品期货标的层的廉价预筛; P(正)低。"
            "价值是先证伪期权买方方向内核,避免购买期权 PIT 数据。"
        ),
        "requested_range": {"start": start, "end": end},
        "symbols": symbols,
        "data_fetch": {
            "source": "akshare.futures_main_sina",
            "source_codes": fetch_report["source_codes"],
            "failures": fetch_report["failures"],
            "ranges": fetch_report["ranges"],
            "roll_method": "Sina main continuous *0; vendor-constructed continuous series.",
            "source_limit": (
                "This is not a self-rebuilt PIT roll from all individual contracts. It is "
                "sufficient for the requested cheap underlying prefilter, but any positive "
                "result remains capped and cannot validate the option strategy."
            ),
        },
        "result": result,
        "public_boundary": (
            "Detailed evidence is private. Public repo contains generic code and synthetic tests; "
            "no credentials, account data, broker GUI, or order path."
        ),
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"domestic-futures-adx-prefilter-{stamp}.json"
    md_path = output_dir / f"domestic-futures-adx-prefilter-{stamp}.md"
    json_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "candidate_count_n": result.get("candidate_count_n"),
                "fdr_survivors": result.get("fdr_survivors"),
                "pbo": (
                    cast(dict[str, Any], result.get("multiple_testing", {}))
                    .get("pbo", {})
                    .get("pbo")
                    if isinstance(
                        cast(dict[str, Any], result.get("multiple_testing", {})).get("pbo"),
                        dict,
                    )
                    else None
                ),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _fetch_akshare_main_continuous(
    symbols: dict[str, str],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    ak = importlib.import_module("akshare")
    bars_by_symbol: dict[str, list[FuturesBar]] = {}
    failures: list[dict[str, str]] = []
    ranges: dict[str, dict[str, object]] = {}
    source_codes: dict[str, str] = {}
    for name, code in symbols.items():
        source_codes[name] = code
        try:
            raw = ak.futures_main_sina(symbol=code)
            rows = _rows_from_dataframe(raw)
            bars = _bars_from_rows(rows, start=start, end=end)
            if bars:
                bars_by_symbol[name] = bars
                ranges[name] = {
                    "bars": len(bars),
                    "start": bars[0].timestamp,
                    "end": bars[-1].timestamp,
                }
            else:
                failures.append({"symbol": name, "code": code, "error": "no rows in range"})
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": name, "code": code, "error": repr(exc)})
        time.sleep(0.2)
    return {
        "bars_by_symbol": bars_by_symbol,
        "failures": failures,
        "ranges": ranges,
        "source_codes": source_codes,
    }


def _rows_from_dataframe(frame: Any) -> list[dict[str, Any]]:
    records = frame.to_dict("records")
    return cast(list[dict[str, Any]], records)


def _bars_from_rows(rows: list[dict[str, Any]], *, start: str, end: str) -> list[FuturesBar]:
    start_key = int(start.replace("-", ""))
    end_key = int(end.replace("-", ""))
    bars: list[FuturesBar] = []
    for row in rows:
        date_raw = str(row.get("日期", row.get("date", "")))[:10]
        date_key = int(date_raw.replace("-", ""))
        if date_key < start_key or date_key > end_key:
            continue
        bars.append(
            FuturesBar(
                timestamp=date_key,
                open=float(row.get("开盘价", row.get("open"))),
                high=float(row.get("最高价", row.get("high"))),
                low=float(row.get("最低价", row.get("low"))),
                close=float(row.get("收盘价", row.get("close"))),
                volume=float(row.get("成交量", row.get("volume", 0.0))),
            )
        )
    bars.sort(key=lambda item: item.timestamp)
    return bars


def _symbols_from_env() -> dict[str, str]:
    raw = os.getenv("DOMESTIC_FUTURES_ADX_SYMBOLS")
    if not raw:
        return dict(DEFAULT_SYMBOLS)
    result: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, code = item.split(":", 1)
        result[name.strip()] = code.strip()
    return result


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    return value


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    result = cast(dict[str, Any], payload["result"])
    multiple = cast(dict[str, Any], result.get("multiple_testing", {}))
    pbo_report = cast(dict[str, Any], multiple.get("pbo", {}))
    return "\n".join(
        [
            "# Olympus #73 Domestic Futures ADX Prefilter Evidence",
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- verdict: {result.get('verdict')}",
            f"- reason: {result.get('reason')}",
            f"- candidate_count_n: {result.get('candidate_count_n')}",
            f"- fdr_survivors: {result.get('fdr_survivors')}",
            f"- pbo: {pbo_report.get('pbo')}",
            f"- json: {json_path}",
            "",
            "Data source: akshare.futures_main_sina, Sina main continuous *0 series.",
            "Funding: N/A for listed domestic futures.",
            "This artifact is private evidence, not a trading signal.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
