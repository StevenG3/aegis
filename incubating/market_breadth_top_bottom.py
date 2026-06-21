import csv
import importlib
import json
import math
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

from aegis.backtest_core import CostModel
from aegis.market_breadth_top_bottom import (
    DEFAULT_BREADTH_CONFIG,
    DEFAULT_COST_MODEL,
    BreadthBar,
    run_market_breadth_study,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-06-01"
DEFAULT_CRYPTO_TOP_N = 50
DEFAULT_US_MAX_SYMBOLS = 0
SP500_CONSTITUENTS_CSV_URLS = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
)
FALLBACK_SP500_SAMPLE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "BRK-B",
    "JPM",
    "AVGO",
    "LLY",
    "TSLA",
    "UNH",
    "V",
    "MA",
    "XOM",
    "HD",
    "PG",
    "COST",
    "JNJ",
    "ABBV",
    "NFLX",
    "CRM",
    "WMT",
    "BAC",
    "ORCL",
    "KO",
    "CVX",
    "AMD",
    "PEP",
    "ADBE",
)
EXCLUDED_CRYPTO_BASES = {
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "USDP",
    "USD1",
    "DAI",
    "EUR",
    "UST",
    "USTC",
    "WBTC",
    "WETH",
}


def main() -> int:
    generated_at = datetime.now(UTC)
    output_dir = private_dir_from_cli(
        os.getenv("MARKET_BREADTH_OUTPUT_DIR"),
        default_task="olympus70",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start = os.getenv("MARKET_BREADTH_START", DEFAULT_START)
    end = os.getenv("MARKET_BREADTH_END", DEFAULT_END)
    fee_bps = _float_env("MARKET_BREADTH_FEE_BPS", DEFAULT_COST_MODEL.fee_bps)
    slippage_bps = _float_env("MARKET_BREADTH_SLIPPAGE_BPS", DEFAULT_COST_MODEL.slippage_bps)
    cost_model = CostModel(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        funding_label=DEFAULT_COST_MODEL.funding_label,
    )

    us_report, us_input = _run_us_study(start=start, end=end, cost_model=cost_model)
    crypto_reports, crypto_input = _run_crypto_studies(
        start=start,
        end=end,
        cost_model=cost_model,
    )
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_70_MARKET_BREADTH_TOP_BOTTOM",
        "ev_newness": (
            "市场内部宽度/背离(非已证伪的免费技术指标方向赌博)。美股 %>MA 是公开经典指标、"
            "指数层面预期被套利(EV 低); Crypto 宽度(altseason 内部)少有人系统测、"
            "是真正未测维度。"
        ),
        "requested_range": {"start": start, "end": end},
        "public_boundary": (
            "Detailed evidence artifact is private. Public repo contains only generic code and "
            "synthetic tests; no credentials, account data, raw private research output, or orders."
        ),
        "inputs": {"us": us_input, "crypto": crypto_input},
        "results": {"us": us_report, "crypto": crypto_reports},
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"market-breadth-top-bottom-{stamp}.json"
    md_path = output_dir / f"market-breadth-top-bottom-{stamp}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "us_verdict": _result_get(us_report, "verdict"),
                "crypto_btc_verdict": _result_get(crypto_reports.get("btc_benchmark"), "verdict"),
                "crypto_equal_weight_verdict": _result_get(
                    crypto_reports.get("equal_weight_benchmark"),
                    "verdict",
                ),
                "us_symbols": us_input.get("symbols_loaded"),
                "crypto_symbols": crypto_input.get("symbols_loaded"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_us_study(
    *,
    start: str,
    end: str,
    cost_model: CostModel,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[dict[str, str]] = []
    tickers = _sp500_tickers(failures)
    max_symbols = _int_env("MARKET_BREADTH_US_MAX_SYMBOLS", DEFAULT_US_MAX_SYMBOLS)
    if max_symbols > 0:
        tickers = tickers[:max_symbols]
    benchmark = _fetch_yfinance_daily("SPY", start=start, end=end)
    members = _fetch_yfinance_daily_many(tickers, start=start, end=end, failures=failures)
    report = dict(
        run_market_breadth_study(
            universe_name="US_current_SP500_survivor_light",
            member_bars=members,
            benchmark_bars=benchmark,
            config=DEFAULT_BREADTH_CONFIG,
            cost_model=cost_model,
            data_source="yfinance_current_sp500_constituents_survivor_light",
            benchmark_name="SPY",
            survivor_light=True,
        )
    )
    return report, {
        "source": "yfinance",
        "constituent_source": "wikipedia_current_sp500_or_fallback_sample",
        "point_in_time_constituents": False,
        "symbols_requested": len(tickers),
        "symbols_loaded": len(members),
        "benchmark_bars": len(benchmark),
        "max_symbols_env": max_symbols,
        "failures": failures[:50],
    }


def _run_crypto_studies(
    *,
    start: str,
    end: str,
    cost_model: CostModel,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, str]] = []
    exchange = _ccxt_exchange("binance")
    top_n = _int_env("MARKET_BREADTH_CRYPTO_TOP_N", DEFAULT_CRYPTO_TOP_N)
    symbols = _crypto_symbols(exchange, top_n=top_n, failures=failures)
    start_ms = _parse_exchange_date(exchange, f"{start}T00:00:00Z")
    end_ms = _parse_exchange_date(exchange, f"{end}T00:00:00Z")
    members: dict[str, list[BreadthBar]] = {}
    for symbol in symbols:
        try:
            bars = _fetch_ccxt_daily(exchange, symbol, since_ms=start_ms, end_ms=end_ms)
            if bars:
                members[symbol] = bars
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
    btc_bars = _fetch_ccxt_daily(exchange, "BTC/USDT", since_ms=start_ms, end_ms=end_ms)
    _close_exchange(exchange)
    equal_weight = _equal_weight_benchmark(members)
    reports = {
        "btc_benchmark": dict(
            run_market_breadth_study(
                universe_name="Crypto_Binance_spot_topN_survivor_light",
                member_bars=members,
                benchmark_bars=btc_bars,
                config=DEFAULT_BREADTH_CONFIG,
                cost_model=cost_model,
                data_source="ccxt_binance_spot_current_listings_survivor_light",
                benchmark_name="BTC/USDT",
                survivor_light=True,
            )
        ),
        "equal_weight_benchmark": dict(
            run_market_breadth_study(
                universe_name="Crypto_Binance_spot_topN_survivor_light",
                member_bars=members,
                benchmark_bars=equal_weight,
                config=DEFAULT_BREADTH_CONFIG,
                cost_model=cost_model,
                data_source="ccxt_binance_spot_current_listings_survivor_light",
                benchmark_name="equal_weight_topN_alt_index",
                survivor_light=True,
            )
        ),
    }
    return reports, {
        "source": "ccxt.binance.spot",
        "point_in_time_constituents": False,
        "top_n_requested": top_n,
        "symbols_requested": len(symbols),
        "symbols_loaded": len(members),
        "benchmark_btc_bars": len(btc_bars),
        "equal_weight_benchmark_bars": len(equal_weight),
        "symbols": tuple(members),
        "failures": failures[:50],
    }


def _sp500_tickers(failures: list[dict[str, str]]) -> list[str]:
    try:
        pandas = importlib.import_module("pandas")
        tables = pandas.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        frame = tables[0]
        values = list(frame["Symbol"])
        tickers = [str(value).strip().replace(".", "-") for value in values if str(value).strip()]
        if tickers:
            return tickers
    except Exception as exc:  # noqa: BLE001
        failures.append({"symbol": "SP500_CONSTITUENTS", "error": str(exc)})
    csv_tickers = _sp500_tickers_from_csv(failures)
    if csv_tickers:
        return csv_tickers
    return list(FALLBACK_SP500_SAMPLE)


def _sp500_tickers_from_csv(failures: list[dict[str, str]]) -> list[str]:
    for url in SP500_CONSTITUENTS_CSV_URLS:
        try:
            with urlopen(url, timeout=20) as response:  # noqa: S310
                content = response.read().decode("utf-8")
            rows = csv.DictReader(content.splitlines())
            tickers = [
                str(row.get("Symbol", "")).strip().replace(".", "-")
                for row in rows
                if str(row.get("Symbol", "")).strip()
            ]
            if tickers:
                return tickers
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": "SP500_CONSTITUENTS_CSV", "error": f"{url}: {exc}"})
    return []


def _fetch_yfinance_daily(symbol: str, *, start: str, end: str) -> list[BreadthBar]:
    yfinance = importlib.import_module("yfinance")
    frame = yfinance.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        return []
    return _bars_from_yfinance_frame(frame)


def _fetch_yfinance_daily_many(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    failures: list[dict[str, str]],
) -> dict[str, list[BreadthBar]]:
    if not symbols:
        return {}
    yfinance = importlib.import_module("yfinance")
    try:
        frame = yfinance.download(
            list(symbols),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:  # noqa: BLE001
        failures.append({"symbol": "BATCH_DOWNLOAD", "error": str(exc)})
        return _fetch_yfinance_daily_many_fallback(symbols, start=start, end=end, failures=failures)
    if frame is None or frame.empty:
        failures.append({"symbol": "BATCH_DOWNLOAD", "error": "empty yfinance batch response"})
        return _fetch_yfinance_daily_many_fallback(symbols, start=start, end=end, failures=failures)
    out: dict[str, list[BreadthBar]] = {}
    columns = getattr(frame, "columns", None)
    if len(symbols) == 1:
        bars = _bars_from_yfinance_frame(frame)
        if bars:
            out[symbols[0]] = bars
        return out
    top_level = set(columns.get_level_values(0)) if getattr(columns, "nlevels", 1) > 1 else set()
    for symbol in symbols:
        try:
            if symbol not in top_level:
                failures.append({"symbol": symbol, "error": "missing from yfinance batch response"})
                continue
            bars = _bars_from_yfinance_frame(frame[symbol])
            if bars:
                out[symbol] = bars
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
    return out


def _fetch_yfinance_daily_many_fallback(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    failures: list[dict[str, str]],
) -> dict[str, list[BreadthBar]]:
    out: dict[str, list[BreadthBar]] = {}
    for symbol in symbols:
        try:
            bars = _fetch_yfinance_daily(symbol, start=start, end=end)
            if bars:
                out[symbol] = bars
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
    return out


def _bars_from_yfinance_frame(frame: Any) -> list[BreadthBar]:
    out: list[BreadthBar] = []
    for timestamp, row in frame.iterrows():
        close = _row_value(row, "Close")
        open_price = _row_value(row, "Open", default=close)
        high = _row_value(row, "High", default=max(open_price, close))
        low = _row_value(row, "Low", default=min(open_price, close))
        volume = _row_value(row, "Volume", default=0.0)
        if (
            close <= 0.0
            or not math.isfinite(close)
            or not math.isfinite(open_price)
            or not math.isfinite(high)
            or not math.isfinite(low)
            or not math.isfinite(volume)
        ):
            continue
        out.append(
            BreadthBar(
                timestamp=int(timestamp.to_pydatetime().replace(tzinfo=UTC).timestamp()),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return out


def _row_value(row: Any, column: str, *, default: float | None = None) -> float:
    try:
        value = row[column]
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        return float(value)
    except Exception:  # noqa: BLE001
        if default is None:
            raise
        return default


def _ccxt_exchange(name: str) -> Any:
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, name)
    return factory({"enableRateLimit": True, "timeout": 20_000})


def _crypto_symbols(exchange: Any, *, top_n: int, failures: list[dict[str, str]]) -> list[str]:
    markets = cast(dict[str, dict[str, Any]], exchange.load_markets())
    candidates = [
        symbol
        for symbol, market in markets.items()
        if bool(market.get("spot"))
        and bool(market.get("active", True))
        and symbol.endswith("/USDT")
        and ":" not in symbol
        and _crypto_base(symbol) not in EXCLUDED_CRYPTO_BASES
        and "UP/" not in symbol
        and "DOWN/" not in symbol
        and "BULL/" not in symbol
        and "BEAR/" not in symbol
    ]
    try:
        tickers = cast(dict[str, dict[str, Any]], exchange.fetch_tickers())
        ranked = sorted(
            candidates,
            key=lambda symbol: float(tickers.get(symbol, {}).get("quoteVolume") or 0.0),
            reverse=True,
        )
        return ranked[:top_n]
    except Exception as exc:  # noqa: BLE001
        failures.append({"symbol": "fetch_tickers", "error": str(exc)})
        return candidates[:top_n]


def _crypto_base(symbol: str) -> str:
    return symbol.split("/", maxsplit=1)[0].upper()


def _fetch_ccxt_daily(
    exchange: Any,
    symbol: str,
    *,
    since_ms: int,
    end_ms: int,
) -> list[BreadthBar]:
    rows: list[list[float]] = []
    cursor = since_ms
    seen: set[int] = set()
    while cursor < end_ms:
        batch = cast(
            list[list[float]],
            exchange.fetch_ohlcv(symbol, timeframe="1d", since=cursor, limit=1000),
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
    return [
        BreadthBar(
            timestamp=int(row[0] // 1000),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in sorted(rows, key=lambda item: item[0])
        if len(row) >= 6 and float(row[4]) > 0.0
    ]


def _equal_weight_benchmark(members: dict[str, list[BreadthBar]]) -> list[BreadthBar]:
    by_symbol = {symbol: {bar.timestamp: bar for bar in bars} for symbol, bars in members.items()}
    timestamps = sorted({timestamp for bars in by_symbol.values() for timestamp in bars})
    previous: dict[str, float] = {}
    close = 100.0
    out: list[BreadthBar] = []
    for timestamp in timestamps:
        returns: list[float] = []
        for symbol, bars in by_symbol.items():
            bar = bars.get(timestamp)
            prior = previous.get(symbol)
            if bar is None:
                continue
            if prior is not None and prior > 0.0:
                returns.append(bar.close / prior - 1.0)
            previous[symbol] = bar.close
        if len(returns) < 2:
            continue
        close *= 1.0 + sum(returns) / len(returns)
        out.append(
            BreadthBar(
                timestamp=timestamp,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=0.0,
            )
        )
    return out


def _parse_exchange_date(exchange: Any, value: str) -> int:
    parsed = exchange.parse8601(value)
    if parsed is None:
        raise ValueError(f"could not parse exchange date {value!r}")
    return int(parsed)


def _close_exchange(exchange: Any) -> None:
    close = getattr(exchange, "close", None)
    if callable(close):
        close()


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _result_get(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    us = cast(dict[str, Any], payload["results"]["us"])
    crypto = cast(dict[str, Any], payload["results"]["crypto"])
    btc = cast(dict[str, Any], crypto.get("btc_benchmark", {}))
    equal = cast(dict[str, Any], crypto.get("equal_weight_benchmark", {}))
    us_multiple = cast(dict[str, Any], us.get("multiple_testing", {}))
    btc_multiple = cast(dict[str, Any], btc.get("multiple_testing", {}))
    equal_multiple = cast(dict[str, Any], equal.get("multiple_testing", {}))
    return "\n".join(
        [
            "# Olympus #70 Market Breadth Top/Bottom Evidence",
            "",
            str(payload["ev_newness"]),
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- requested_range: {payload['requested_range']}",
            f"- us_verdict: {us.get('verdict')} ({us.get('health_status')})",
            f"- us_reason: {us.get('reason')}",
            f"- us_trials: {us_multiple.get('candidate_count_n')}",
            f"- us_fdr_survivors: {us_multiple.get('fdr_survivors')}",
            f"- crypto_btc_verdict: {btc.get('verdict')} ({btc.get('health_status')})",
            f"- crypto_btc_reason: {btc.get('reason')}",
            f"- crypto_btc_trials: {btc_multiple.get('candidate_count_n')}",
            f"- crypto_btc_fdr_survivors: {btc_multiple.get('fdr_survivors')}",
            f"- crypto_equal_weight_verdict: {equal.get('verdict')} ({equal.get('health_status')})",
            f"- crypto_equal_weight_reason: {equal.get('reason')}",
            f"- crypto_equal_weight_trials: {equal_multiple.get('candidate_count_n')}",
            f"- crypto_equal_weight_fdr_survivors: {equal_multiple.get('fdr_survivors')}",
            f"- json: {json_path}",
            "",
            "Funding is N/A because this event study uses spot daily bars only.",
            "All results are survivor-light and capped below ROBUST/EDGE.",
            "This artifact is private evidence, not a trading signal.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
