from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.backtest_core import CostModel
from aegis.crypto_cross_sectional import (
    DEFAULT_COST_MODEL,
    MAIN_CONFIG,
    CrossSectionalCryptoBar,
    CrossSectionalCryptoConfig,
    run_crypto_cross_sectional_momentum,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_SOURCE = "binanceusdm"
STABLE_QUOTES = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE"}
WRAPPED_BASES = {"WBTC", "WETH", "WSOL", "WBNB"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")
RETRY_COUNT = 2


@dataclass(frozen=True)
class EvidenceRun:
    source: str
    start: str
    end: str
    max_symbols: int
    prefetch_symbols: int
    output_dir: Path
    taker_fee_bps: float
    slippage_bps: float
    min_proxy_volume_usd: float


def main() -> int:
    generated_at = datetime.now(UTC)
    run = _run_from_env()
    run.output_dir.mkdir(parents=True, exist_ok=True)
    exchange = _exchange(run.source)
    start_ms = _parse_date(run.start)
    end_ms = _parse_date(run.end)
    markets = _request_with_retry(exchange.load_markets)
    cross_listed = _cross_listed_bases()
    candidates = _ranked_candidates(
        exchange,
        markets=markets,
        cross_listed_bases=cross_listed,
        prefetch_symbols=run.prefetch_symbols,
    )
    bars_by_symbol: dict[str, list[CrossSectionalCryptoBar]] = {}
    failures: list[dict[str, str]] = []
    funding_failures: list[dict[str, str]] = []
    funding_rows = 0
    for symbol, exchange_count in candidates[: run.max_symbols]:
        try:
            ohlcv = _fetch_ohlcv(exchange, symbol, since_ms=start_ms, end_ms=end_ms)
            try:
                funding = _fetch_daily_funding(
                    exchange, symbol, since_ms=start_ms, end_ms=end_ms
                )
            except Exception as exc:  # noqa: BLE001
                funding = {}
                funding_failures.append({"symbol": symbol, "error": str(exc)})
            funding_rows += sum(1 for value in funding.values() if value != 0.0)
            bars = _bars_from_ohlcv(
                ohlcv,
                funding_by_day=funding,
                exchange_count=exchange_count,
            )
            if len(bars) >= _minimum_evidence_bars(config=MAIN_CONFIG):
                bars_by_symbol[symbol] = bars
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
    _close_exchange(exchange)

    config = _evidence_config(run)
    result = dict(
        run_crypto_cross_sectional_momentum(
            bars_by_symbol,
            config=config,
            cost_model=CostModel(
                fee_bps=run.taker_fee_bps,
                slippage_bps=run.slippage_bps,
                funding_label=DEFAULT_COST_MODEL.funding_label,
            ),
            survivor_light=True,
            data_source="ccxt_free_binance_usdt_perpetuals_survivor_light",
        )
    )
    if bars_by_symbol and funding_failures:
        result = {
            **result,
            "status": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "reason": (
                "funding history unavailable for one or more included perpetual symbols; "
                "full-cost confirmation must fail closed"
            ),
        }
    payload = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_67_CRYPTO_CROSS_SECTIONAL_FACTOR",
        "public_boundary": (
            "artifact is private; raw OHLCV/funding rows, positions, and credentials are not "
            "written to the public repository"
        ),
        "input": _run_to_dict(run),
        "discipline": {
            "mode": "confirmation",
            "main_spec_only_for_verdict": True,
            "robustness_grid_side_evidence_only": True,
            "read_only_ccxt_public_data": True,
            "wallet_or_order_access": False,
            "survivor_light_ceiling": True,
        },
        "coverage": {
            "candidate_symbols": len(candidates),
            "symbols_with_bars": len(bars_by_symbol),
            "bars_min": min((len(values) for values in bars_by_symbol.values()), default=0),
            "bars_max": max((len(values) for values in bars_by_symbol.values()), default=0),
            "funding_nonzero_daily_rows": funding_rows,
            "fetch_failures": failures,
            "funding_failures": funding_failures,
            "point_in_time_limit": (
                "free ccxt market metadata is current-listing oriented; delisted historical "
                "universe and historical market cap are not complete, so survivor-light "
                "ceiling remains active"
            ),
        },
        "result": result,
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = run.output_dir / f"crypto-cross-sectional-evidence-{stamp}.json"
    md_path = run.output_dir / f"crypto-cross-sectional-evidence-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "symbols_with_bars": len(bars_by_symbol),
                "funding_nonzero_daily_rows": funding_rows,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_from_env() -> EvidenceRun:
    return EvidenceRun(
        source=os.getenv("CRYPTO_XSEC_EVIDENCE_SOURCE", DEFAULT_SOURCE).strip() or DEFAULT_SOURCE,
        start=os.getenv("CRYPTO_XSEC_EVIDENCE_START", DEFAULT_START),
        end=os.getenv("CRYPTO_XSEC_EVIDENCE_END", DEFAULT_END),
        max_symbols=_env_int("CRYPTO_XSEC_EVIDENCE_MAX_SYMBOLS", 30),
        prefetch_symbols=_env_int("CRYPTO_XSEC_EVIDENCE_PREFETCH_SYMBOLS", 60),
        taker_fee_bps=_env_float("CRYPTO_XSEC_EVIDENCE_TAKER_FEE_BPS", 5.0),
        slippage_bps=_env_float("CRYPTO_XSEC_EVIDENCE_SLIPPAGE_BPS", 5.0),
        min_proxy_volume_usd=_env_float("CRYPTO_XSEC_EVIDENCE_MIN_PROXY_VOLUME_USD", 50_000_000.0),
        output_dir=private_dir_from_cli(
            os.getenv("CRYPTO_XSEC_EVIDENCE_OUTPUT_DIR"),
            default_task="olympus67",
        ),
    )


def _evidence_config(run: EvidenceRun) -> CrossSectionalCryptoConfig:
    return replace(MAIN_CONFIG, min_proxy_volume_usd=run.min_proxy_volume_usd)


def _exchange(source: str) -> Any:
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, source)
    return factory({"enableRateLimit": True, "timeout": 20_000})


def _cross_listed_bases() -> set[str]:
    bases: set[str] = set()
    for source in ("okx", "bybit"):
        exchange = _exchange(source)
        try:
            markets = cast(dict[str, Any], _request_with_retry(exchange.load_markets))
            for market in markets.values():
                if _is_usdt_perp_market(market):
                    bases.add(str(market.get("base", "")).upper())
        except Exception:
            continue
        finally:
            _close_exchange(exchange)
    return bases


def _ranked_candidates(
    exchange: Any,
    *,
    markets: Any,
    cross_listed_bases: set[str],
    prefetch_symbols: int,
) -> list[tuple[str, int]]:
    ranked: list[tuple[str, int, float]] = []
    tickers = cast(dict[str, Any], _request_with_retry(exchange.fetch_tickers))
    for symbol, market in cast(dict[str, Any], markets).items():
        if not _is_usdt_perp_market(market):
            continue
        base = str(market.get("base", "")).upper()
        if _excluded_base(base):
            continue
        ticker = tickers.get(symbol, {})
        quote_volume = _float_scalar(ticker.get("quoteVolume"))
        if quote_volume is None:
            base_volume = _float_scalar(ticker.get("baseVolume"))
            last = _float_scalar(ticker.get("last"))
            quote_volume = (base_volume or 0.0) * (last or 0.0)
        exchange_count = 1 + int(base in cross_listed_bases)
        if exchange_count < MAIN_CONFIG.min_exchange_count:
            continue
        ranked.append((symbol, exchange_count, quote_volume or 0.0))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return [
        (symbol, exchange_count)
        for symbol, exchange_count, _volume in ranked[:prefetch_symbols]
    ]


def _is_usdt_perp_market(market: Any) -> bool:
    if not isinstance(market, dict):
        return False
    return (
        bool(market.get("swap"))
        and str(market.get("quote", "")).upper() == "USDT"
        and bool(market.get("linear", True))
    )


def _excluded_base(base: str) -> bool:
    if base in STABLE_QUOTES:
        return True
    if base in WRAPPED_BASES:
        return True
    return any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES)


def _fetch_ohlcv(exchange: Any, symbol: str, *, since_ms: int, end_ms: int) -> list[list[float]]:
    rows: list[list[float]] = []
    cursor = since_ms
    while cursor < end_ms:
        batch = _request_with_retry(exchange.fetch_ohlcv, symbol, "1d", since=cursor, limit=1000)
        parsed = [cast(list[float], row) for row in batch if int(row[0]) < end_ms]
        if not parsed:
            break
        rows.extend(parsed)
        next_cursor = int(parsed[-1][0]) + 86_400_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    return rows


def _fetch_daily_funding(
    exchange: Any, symbol: str, *, since_ms: int, end_ms: int
) -> dict[int, float]:
    if not hasattr(exchange, "fetch_funding_rate_history"):
        return {}
    rows: dict[int, float] = {}
    cursor = since_ms
    while cursor < end_ms:
        batch = _request_with_retry(
            exchange.fetch_funding_rate_history,
            symbol,
            since=cursor,
            limit=1000,
        )
        parsed = [cast(dict[str, Any], item) for item in batch]
        if not parsed:
            break
        advanced = False
        for item in parsed:
            timestamp = _int_value(item.get("timestamp"))
            rate = _float_scalar(item.get("fundingRate"))
            if timestamp is None or timestamp >= end_ms or rate is None:
                continue
            day = timestamp - timestamp % 86_400_000
            rows[day] = rows.get(day, 0.0) + rate
            if timestamp >= cursor:
                cursor = timestamp + 1
                advanced = True
        if not advanced:
            break
        time.sleep(0.05)
    return rows


def _bars_from_ohlcv(
    ohlcv: list[list[float]],
    *,
    funding_by_day: dict[int, float],
    exchange_count: int,
) -> list[CrossSectionalCryptoBar]:
    if not ohlcv:
        return []
    listed_at = int(ohlcv[0][0])
    bars: list[CrossSectionalCryptoBar] = []
    for row in ohlcv:
        timestamp = int(row[0])
        open_price = float(row[1])
        close_price = float(row[4])
        base_volume = float(row[5])
        quote_volume = base_volume * close_price
        bars.append(
            CrossSectionalCryptoBar(
                timestamp=timestamp,
                open=open_price,
                close=close_price,
                quote_volume_usd=quote_volume,
                funding_rate=funding_by_day.get(timestamp - timestamp % 86_400_000, 0.0),
                market_cap_usd=None,
                exchange_count=exchange_count,
                listed_at=listed_at,
                is_stable=False,
                is_wrapped=False,
                is_leveraged=False,
            )
        )
    return bars


def _minimum_evidence_bars(*, config: CrossSectionalCryptoConfig) -> int:
    return max(
        config.min_history_days,
        config.momentum_lookback_days + config.skip_recent_days + 2,
        config.vol_lookback_days + 2,
    ) + 30


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    result = cast(dict[str, Any], payload["result"])
    coverage = cast(dict[str, Any], payload["coverage"])
    multiple = cast(dict[str, Any], result.get("multiple_testing", {}))
    return "\n".join(
        [
            "# CODEX OLYMPUS 67 Crypto Cross-Sectional Evidence",
            "",
            f"Generated: `{payload['generated_at']}`",
            f"Verdict: `{result.get('verdict')}`",
            f"Reason: {result.get('reason')}",
            f"JSON artifact: `{json_path}`",
            "",
            "## Coverage",
            "",
            f"- Candidate symbols: `{coverage.get('candidate_symbols')}`",
            f"- Symbols with bars: `{coverage.get('symbols_with_bars')}`",
            f"- Bars min/max: `{coverage.get('bars_min')}` / `{coverage.get('bars_max')}`",
            f"- Funding nonzero daily rows: `{coverage.get('funding_nonzero_daily_rows')}`",
            f"- Fetch failures: `{len(coverage.get('fetch_failures', []))}`",
            f"- Funding failures: `{len(coverage.get('funding_failures', []))}`",
            f"- PIT limit: {coverage.get('point_in_time_limit')}",
            "",
            "## Multiple Testing",
            "",
            f"- Main N: `{multiple.get('candidate_count_n')}`",
            f"- FDR survivors: `{multiple.get('fdr_after')}`",
            f"- PBO: `{multiple.get('pbo')}`",
            "",
            "## Discipline",
            "",
            "- Main specification is predeclared; robustness grid is side evidence.",
            "- Signals observe prior daily data; fills use the next daily open.",
            "- Taker fee, slippage, and perpetual funding observations are counted.",
            "- Benchmarks are cash-neutral, equal-weight long beta, and BTC buy-and-hold.",
            "- Free ccxt coverage is survivor-light and cannot produce ROBUST.",
        ]
    ) + "\n"


def _run_to_dict(run: EvidenceRun) -> dict[str, Any]:
    return {
        "source": run.source,
        "start": run.start,
        "end": run.end,
        "max_symbols": run.max_symbols,
        "prefetch_symbols": run.prefetch_symbols,
        "taker_fee_bps": run.taker_fee_bps,
        "slippage_bps": run.slippage_bps,
        "min_proxy_volume_usd": run.min_proxy_volume_usd,
    }


def _parse_date(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def _request_with_retry(func: Any, *args: object, **kwargs: object) -> Any:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(str(last_error) if last_error else "ccxt request failed")


def _close_exchange(exchange: Any) -> None:
    close = getattr(exchange, "close", None)
    if callable(close):
        close()


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


if __name__ == "__main__":
    raise SystemExit(main())
