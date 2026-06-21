import csv
import importlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import replace
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
DEFAULT_INACTIVE_STRESS_MAX = 30
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
KNOWN_REMOVED_SP500_SAMPLE = (
    "GE",
    "T",
    "F",
    "AAL",
    "DOW",
    "DD",
    "KSS",
    "M",
    "GPS",
    "BBBY",
    "SIVBQ",
    "FRCB",
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
        default_task="olympus70b",
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

    config = replace(
        DEFAULT_BREADTH_CONFIG,
        overlap_correction=True,
        block_bootstrap_samples=_int_env("MARKET_BREADTH_BLOCK_BOOTSTRAP_SAMPLES", 300),
    )
    us_report, us_input = _run_us_study(
        start=start,
        end=end,
        cost_model=cost_model,
        config=config,
    )
    crypto_reports, crypto_input = _run_crypto_studies(
        start=start,
        end=end,
        cost_model=cost_model,
        config=config,
    )
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_70B_BREADTH_OVERLAP_SURVIVORSHIP",
        "ev_newness": (
            "非新猎场,是给 #70 唯一穿过四重门的 survivor 做去伪存真:事件重叠/自相关"
            "校正 + 幸存者偏差压力测试。预期 survivor 数下降甚至归零。"
        ),
        "protocol": {
            "predeclared": True,
            "same_signal_grid_as_70": True,
            "overlap_correction": "disjoint_per_trial_plus_block_bootstrap",
            "survivorship_stress": "available_free_data_only_not_PIT_unbiased",
            "max_positive_verdict": "SUGGESTIVE",
        },
        "requested_range": {"start": start, "end": end},
        "public_boundary": (
            "Detailed evidence artifact is private. Public repo contains only generic code and "
            "synthetic tests; no credentials, account data, raw private research output, or orders."
        ),
        "inputs": {"us": us_input, "crypto": crypto_input},
        "results": {"us": us_report, "crypto": crypto_reports},
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"market-breadth-top-bottom-70b-{stamp}.json"
    md_path = output_dir / f"market-breadth-top-bottom-70b-{stamp}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "us_current_verdict": _result_get(us_report.get("current_constituents"), "verdict"),
                "us_removed_stress_verdict": _result_get(
                    us_report.get("known_removed_sample_stress"),
                    "verdict",
                ),
                "crypto_btc_verdict": _result_get(crypto_reports.get("btc_benchmark"), "verdict"),
                "crypto_equal_weight_verdict": _result_get(
                    crypto_reports.get("equal_weight_benchmark"),
                    "verdict",
                ),
                "crypto_inactive_stress_btc_verdict": _result_get(
                    crypto_reports.get("inactive_stress_btc_benchmark"),
                    "verdict",
                ),
                "crypto_inactive_stress_equal_weight_verdict": _result_get(
                    crypto_reports.get("inactive_stress_equal_weight_benchmark"),
                    "verdict",
                ),
                "us_symbols": us_input.get("symbols_loaded"),
                "crypto_symbols": crypto_input.get("symbols_loaded"),
                "crypto_fetchable_inactive": crypto_input.get("fetchable_inactive_loaded"),
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
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[dict[str, str]] = []
    tickers = _sp500_tickers(failures)
    max_symbols = _int_env("MARKET_BREADTH_US_MAX_SYMBOLS", DEFAULT_US_MAX_SYMBOLS)
    if max_symbols > 0:
        tickers = tickers[:max_symbols]
    benchmark = _fetch_yfinance_daily("SPY", start=start, end=end)
    members = _fetch_yfinance_daily_many(tickers, start=start, end=end, failures=failures)
    current_report = dict(
        run_market_breadth_study(
            universe_name="US_current_SP500_survivor_light",
            member_bars=members,
            benchmark_bars=benchmark,
            config=config,
            cost_model=cost_model,
            data_source="yfinance_current_sp500_constituents_survivor_light",
            benchmark_name="SPY",
            survivor_light=True,
        )
    )
    removed_symbols = _csv_env("MARKET_BREADTH_US_REMOVED_SYMBOLS", KNOWN_REMOVED_SP500_SAMPLE)
    removed_failures: list[dict[str, str]] = []
    removed_members = _fetch_yfinance_daily_many(
        removed_symbols,
        start=start,
        end=end,
        failures=removed_failures,
    )
    stress_members = dict(members)
    stress_members.update({f"REMOVED:{key}": value for key, value in removed_members.items()})
    removed_stress_report = dict(
        run_market_breadth_study(
            universe_name="US_current_SP500_plus_removed_sample_survivor_stress",
            member_bars=stress_members,
            benchmark_bars=benchmark,
            config=config,
            cost_model=cost_model,
            data_source="yfinance_current_sp500_plus_known_removed_sample_not_PIT",
            benchmark_name="SPY",
            survivor_light=True,
        )
    )
    report = {
        "current_constituents": current_report,
        "known_removed_sample_stress": removed_stress_report,
        "survivorship_stress": {
            "pit_constituents_available": False,
            "not_unbiased_reason": (
                "Free current S&P500/yfinance data cannot reconstruct point-in-time historical "
                "index membership; removed-symbol sample is a directional stress test only."
            ),
            "known_removed_symbols_requested": tuple(removed_symbols),
            "known_removed_symbols_loaded": len(removed_members),
            "failed_removed_symbols": removed_failures[:50],
            "current_break_even_haircut": _break_even_haircut_report(current_report),
            "removed_stress_break_even_haircut": _break_even_haircut_report(
                removed_stress_report
            ),
            "comparison": _result_comparison(current_report, removed_stress_report),
        },
    }
    return report, {
        "source": "yfinance",
        "constituent_source": "wikipedia_current_sp500_or_fallback_sample",
        "point_in_time_constituents": False,
        "symbols_requested": len(tickers),
        "symbols_loaded": len(members),
        "known_removed_symbols_requested": len(removed_symbols),
        "known_removed_symbols_loaded": len(removed_members),
        "benchmark_bars": len(benchmark),
        "max_symbols_env": max_symbols,
        "failures": failures[:50],
        "removed_failures": removed_failures[:50],
    }


def _run_crypto_studies(
    *,
    start: str,
    end: str,
    cost_model: CostModel,
    config: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, str]] = []
    exchange = _ccxt_exchange("binance")
    top_n = _int_env("MARKET_BREADTH_CRYPTO_TOP_N", DEFAULT_CRYPTO_TOP_N)
    inactive_limit = _int_env("MARKET_BREADTH_INACTIVE_STRESS_MAX", DEFAULT_INACTIVE_STRESS_MAX)
    markets = cast(dict[str, dict[str, Any]], exchange.load_markets())
    symbols = _crypto_symbols(exchange, markets=markets, top_n=top_n, failures=failures)
    inactive_candidates = _inactive_crypto_symbols(markets, limit=inactive_limit)
    start_ms = _parse_exchange_date(exchange, f"{start}T00:00:00Z")
    end_ms = _parse_exchange_date(exchange, f"{end}T00:00:00Z")
    members: dict[str, list[BreadthBar]] = {}
    symbol_manifest: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            bars = _fetch_ccxt_daily(exchange, symbol, since_ms=start_ms, end_ms=end_ms)
            if bars:
                members[symbol] = bars
            symbol_manifest.append(
                _symbol_manifest_row(
                    symbol,
                    markets.get(symbol, {}),
                    bars,
                    fetch_status="loaded" if bars else "empty_ohlcv",
                    included_current_active=bool(bars),
                    included_inactive_stress=False,
                    exclude_reason=None if bars else "empty_ohlcv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
            symbol_manifest.append(
                _symbol_manifest_row(
                    symbol,
                    markets.get(symbol, {}),
                    [],
                    fetch_status="failed_fetch",
                    included_current_active=False,
                    included_inactive_stress=False,
                    exclude_reason=str(exc),
                )
            )
    inactive_members: dict[str, list[BreadthBar]] = {}
    for symbol in inactive_candidates:
        try:
            bars = _fetch_ccxt_daily(exchange, symbol, since_ms=start_ms, end_ms=end_ms)
            included = bool(bars)
            if included:
                inactive_members[f"INACTIVE:{symbol}"] = bars
            symbol_manifest.append(
                _symbol_manifest_row(
                    symbol,
                    markets.get(symbol, {}),
                    bars,
                    fetch_status="loaded" if bars else "empty_ohlcv",
                    included_current_active=False,
                    included_inactive_stress=included,
                    exclude_reason=None if included else "empty_ohlcv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
            symbol_manifest.append(
                _symbol_manifest_row(
                    symbol,
                    markets.get(symbol, {}),
                    [],
                    fetch_status="failed_fetch",
                    included_current_active=False,
                    included_inactive_stress=False,
                    exclude_reason=str(exc),
                )
            )
    btc_bars = _fetch_ccxt_daily(exchange, "BTC/USDT", since_ms=start_ms, end_ms=end_ms)
    _close_exchange(exchange)
    equal_weight = _equal_weight_benchmark(members)
    stress_members = dict(members)
    stress_members.update(inactive_members)
    stress_equal_weight = _equal_weight_benchmark(stress_members)
    current_btc = dict(
        run_market_breadth_study(
            universe_name="Crypto_Binance_spot_topN_survivor_light",
            member_bars=members,
            benchmark_bars=btc_bars,
            config=config,
            cost_model=cost_model,
            data_source="ccxt_binance_spot_current_listings_survivor_light",
            benchmark_name="BTC/USDT",
            survivor_light=True,
        )
    )
    current_equal = dict(
        run_market_breadth_study(
            universe_name="Crypto_Binance_spot_topN_survivor_light",
            member_bars=members,
            benchmark_bars=equal_weight,
            config=config,
            cost_model=cost_model,
            data_source="ccxt_binance_spot_current_listings_survivor_light",
            benchmark_name="equal_weight_topN_alt_index",
            survivor_light=True,
        )
    )
    stress_btc = dict(
        run_market_breadth_study(
            universe_name="Crypto_Binance_spot_topN_plus_fetchable_inactive_stress",
            member_bars=stress_members,
            benchmark_bars=btc_bars,
            config=config,
            cost_model=cost_model,
            data_source="ccxt_binance_spot_active_plus_fetchable_inactive_not_PIT",
            benchmark_name="BTC/USDT",
            survivor_light=True,
        )
    )
    stress_equal = dict(
        run_market_breadth_study(
            universe_name="Crypto_Binance_spot_topN_plus_fetchable_inactive_stress",
            member_bars=stress_members,
            benchmark_bars=stress_equal_weight,
            config=config,
            cost_model=cost_model,
            data_source="ccxt_binance_spot_active_plus_fetchable_inactive_not_PIT",
            benchmark_name="equal_weight_topN_plus_inactive_alt_index",
            survivor_light=True,
        )
    )
    reports = {
        "btc_benchmark": current_btc,
        "equal_weight_benchmark": current_equal,
        "inactive_stress_btc_benchmark": stress_btc,
        "inactive_stress_equal_weight_benchmark": stress_equal,
        "survivorship_comparison": {
            "btc_benchmark": _result_comparison(current_btc, stress_btc),
            "equal_weight_benchmark": _result_comparison(current_equal, stress_equal),
        },
    }
    active_usdt_count = sum(
        1 for symbol, market in markets.items() if _is_crypto_usdt_spot(symbol, market, active=True)
    )
    inactive_usdt_count = sum(
        1
        for symbol, market in markets.items()
        if _is_crypto_usdt_spot(symbol, market, active=False)
    )
    return reports, {
        "source": "ccxt.binance.spot",
        "point_in_time_constituents": False,
        "top_n_requested": top_n,
        "inactive_stress_max": inactive_limit,
        "symbols_requested": len(symbols),
        "symbols_loaded": len(members),
        "inactive_candidates": len(inactive_candidates),
        "fetchable_inactive_loaded": len(inactive_members),
        "benchmark_btc_bars": len(btc_bars),
        "equal_weight_benchmark_bars": len(equal_weight),
        "stress_equal_weight_benchmark_bars": len(stress_equal_weight),
        "symbols": tuple(members),
        "inactive_symbols_loaded": tuple(inactive_members),
        "inactive_probe_summary": {
            "markets_total": len(markets),
            "active_usdt_count": active_usdt_count,
            "inactive_usdt_count": inactive_usdt_count,
            "fetchable_inactive_count": len(inactive_members),
            "empty_ohlcv_count": sum(
                1 for row in symbol_manifest if row["fetch_status"] == "empty_ohlcv"
            ),
            "failed_fetch_count": sum(
                1 for row in symbol_manifest if row["fetch_status"] == "failed_fetch"
            ),
        },
        "symbol_manifest": symbol_manifest,
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
            else:
                failures.append({"symbol": symbol, "error": "empty yfinance bars"})
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
            else:
                failures.append({"symbol": symbol, "error": "empty yfinance bars"})
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


def _crypto_symbols(
    exchange: Any,
    *,
    markets: dict[str, dict[str, Any]],
    top_n: int,
    failures: list[dict[str, str]],
) -> list[str]:
    candidates = [
        symbol
        for symbol, market in markets.items()
        if _is_crypto_usdt_spot(symbol, market, active=True)
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


def _inactive_crypto_symbols(
    markets: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    return [
        symbol
        for symbol, market in sorted(markets.items())
        if _is_crypto_usdt_spot(symbol, market, active=False)
    ][:limit]


def _is_crypto_usdt_spot(symbol: str, market: dict[str, Any], *, active: bool) -> bool:
    market_active = market.get("active")
    if active and market_active is False:
        return False
    if not active and market_active is not False:
        return False
    return (
        bool(market.get("spot"))
        and symbol.endswith("/USDT")
        and ":" not in symbol
        and _crypto_base(symbol) not in EXCLUDED_CRYPTO_BASES
        and "UP/" not in symbol
        and "DOWN/" not in symbol
        and "BULL/" not in symbol
        and "BEAR/" not in symbol
    )


def _symbol_manifest_row(
    symbol: str,
    market: dict[str, Any],
    bars: Sequence[BreadthBar],
    *,
    fetch_status: str,
    included_current_active: bool,
    included_inactive_stress: bool,
    exclude_reason: str | None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "market_id": market.get("id"),
        "base": market.get("base"),
        "quote": market.get("quote"),
        "spot": bool(market.get("spot")),
        "active": market.get("active"),
        "status_raw": market.get("status"),
        "first_bar_ts": bars[0].timestamp if bars else None,
        "last_bar_ts": bars[-1].timestamp if bars else None,
        "bar_count": len(bars),
        "fetch_status": fetch_status,
        "included_current_active": included_current_active,
        "included_inactive_stress": included_inactive_stress,
        "exclude_reason": exclude_reason,
    }


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


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _result_get(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _survivor_count(result: dict[str, Any]) -> int:
    multiple = result.get("multiple_testing", {})
    if isinstance(multiple, dict):
        return int(multiple.get("fdr_survivors", 0) or 0)
    return 0


def _mean_excess_survivors(result: dict[str, Any]) -> float:
    candidates = result.get("candidate_statistics", [])
    if not isinstance(candidates, list):
        return 0.0
    values = [
        float(row.get("mean_excess_return", 0.0))
        for row in candidates
        if isinstance(row, dict)
        and bool(row.get("bh_fdr_pass"))
        and bool(row.get("deflated_sharpe_pass"))
        and float(row.get("mean_excess_return", 0.0)) > 0.0
    ]
    return sum(values) / len(values) if values else 0.0


def _result_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_survivors = _survivor_count(before)
    after_survivors = _survivor_count(after)
    before_excess = _mean_excess_survivors(before)
    after_excess = _mean_excess_survivors(after)
    return {
        "before_verdict": before.get("verdict"),
        "after_verdict": after.get("verdict"),
        "before_survivors": before_survivors,
        "after_survivors": after_survivors,
        "survivor_delta": after_survivors - before_survivors,
        "before_mean_excess_survivors": before_excess,
        "after_mean_excess_survivors": after_excess,
        "mean_excess_delta": after_excess - before_excess,
        "direction_reversed": before_excess > 0.0 and after_excess <= 0.0,
        "verdict_change": before.get("verdict") != after.get("verdict"),
    }


def _break_even_haircut_report(result: dict[str, Any]) -> dict[str, Any]:
    candidates = result.get("candidate_statistics", [])
    if not isinstance(candidates, list):
        return {"survivors": 0, "min_haircut_to_zero_mean_excess": None, "by_horizon": {}}
    survivors = [
        row
        for row in candidates
        if isinstance(row, dict)
        and bool(row.get("bh_fdr_pass"))
        and bool(row.get("deflated_sharpe_pass"))
        and float(row.get("mean_excess_return", 0.0)) > 0.0
    ]
    by_horizon: dict[str, float] = {}
    for row in survivors:
        horizon = str(row.get("horizon", "unknown"))
        value = float(row.get("mean_excess_return", 0.0))
        by_horizon[horizon] = (
            value if horizon not in by_horizon else min(by_horizon[horizon], value)
        )
    min_haircut = min(by_horizon.values()) if by_horizon else None
    return {
        "survivors": len(survivors),
        "min_haircut_to_zero_mean_excess": min_haircut,
        "by_horizon": by_horizon,
        "interpretation": (
            "A survivor-light edge this small can be erased by an omitted-constituent adverse "
            "haircut of this size to mean excess return."
        ),
    }


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
    us_current = cast(dict[str, Any], us.get("current_constituents", {}))
    us_stress = cast(dict[str, Any], us.get("known_removed_sample_stress", {}))
    crypto = cast(dict[str, Any], payload["results"]["crypto"])
    btc = cast(dict[str, Any], crypto.get("btc_benchmark", {}))
    equal = cast(dict[str, Any], crypto.get("equal_weight_benchmark", {}))
    stress_btc = cast(dict[str, Any], crypto.get("inactive_stress_btc_benchmark", {}))
    stress_equal = cast(
        dict[str, Any],
        crypto.get("inactive_stress_equal_weight_benchmark", {}),
    )
    us_multiple = cast(dict[str, Any], us_current.get("multiple_testing", {}))
    us_stress_multiple = cast(dict[str, Any], us_stress.get("multiple_testing", {}))
    btc_multiple = cast(dict[str, Any], btc.get("multiple_testing", {}))
    equal_multiple = cast(dict[str, Any], equal.get("multiple_testing", {}))
    stress_btc_multiple = cast(dict[str, Any], stress_btc.get("multiple_testing", {}))
    stress_equal_multiple = cast(dict[str, Any], stress_equal.get("multiple_testing", {}))
    return "\n".join(
        [
            "# Olympus #70B Market Breadth Overlap/Survivorship Evidence",
            "",
            str(payload["ev_newness"]),
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- requested_range: {payload['requested_range']}",
            "- us_current_verdict: "
            f"{us_current.get('verdict')} ({us_current.get('health_status')})",
            f"- us_current_reason: {us_current.get('reason')}",
            f"- us_trials: {us_multiple.get('candidate_count_n')}",
            f"- us_fdr_survivors: {us_multiple.get('fdr_survivors')}",
            "- us_removed_stress_verdict: "
            f"{us_stress.get('verdict')} ({us_stress.get('health_status')})",
            f"- us_removed_stress_survivors: {us_stress_multiple.get('fdr_survivors')}",
            f"- crypto_btc_verdict: {btc.get('verdict')} ({btc.get('health_status')})",
            f"- crypto_btc_reason: {btc.get('reason')}",
            f"- crypto_btc_trials: {btc_multiple.get('candidate_count_n')}",
            f"- crypto_btc_fdr_survivors: {btc_multiple.get('fdr_survivors')}",
            f"- crypto_equal_weight_verdict: {equal.get('verdict')} ({equal.get('health_status')})",
            f"- crypto_equal_weight_reason: {equal.get('reason')}",
            f"- crypto_equal_weight_trials: {equal_multiple.get('candidate_count_n')}",
            f"- crypto_equal_weight_fdr_survivors: {equal_multiple.get('fdr_survivors')}",
            "- crypto_inactive_stress_btc_verdict: "
            f"{stress_btc.get('verdict')} ({stress_btc.get('health_status')})",
            "- crypto_inactive_stress_btc_survivors: "
            f"{stress_btc_multiple.get('fdr_survivors')}",
            "- crypto_inactive_stress_equal_weight_verdict: "
            f"{stress_equal.get('verdict')} ({stress_equal.get('health_status')})",
            "- crypto_inactive_stress_equal_weight_survivors: "
            f"{stress_equal_multiple.get('fdr_survivors')}",
            f"- json: {json_path}",
            "",
            "Funding is N/A because this event study uses spot daily bars only.",
            "P-values for #70B gates use disjoint events plus block bootstrap; "
            "sign-test is report-only.",
            "All results are survivor-light and capped below ROBUST/EDGE.",
            "This artifact is private evidence, not a trading signal.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
