from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.microstructure_perp_runner import run_microstructure_perp_from_spec
from aegis.private_paths import private_dir_from_cli

Source = Literal["binance", "bybit", "okx"]

REQUEST_LIMIT = 1000
RETRY_COUNT = 2
DEFAULT_SOURCE: Source = "binance"
DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "1000PEPE/USDT:USDT",
    "WIF/USDT:USDT",
    "FTT/USDT:USDT",
    "LUNA/USDT:USDT",
)
DEFAULT_START = "2024-01-01T00:00:00+00:00"
DEFAULT_END = "2026-01-01T00:00:00+00:00"
HIGH_LIQUIDITY_DEFAULTS = frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"})
DELISTED_CRASH_DEFAULTS = frozenset({"FTT/USDT:USDT", "LUNA/USDT:USDT"})


@dataclass(frozen=True)
class EvidenceRun:
    source: Source
    symbols: tuple[str, ...]
    start: str
    end: str
    timeframe: str
    max_bars_per_symbol: int
    taker_fee_bps: float
    slippage_bps: float
    output_dir: Path
    high_liquidity_symbols: tuple[str, ...]
    delisted_crash_symbols: tuple[str, ...]
    btc_reference_symbol: str


def main() -> int:
    generated_at = datetime.now(UTC)
    run = _run_from_env()
    run.output_dir.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    fetch_failures: list[dict[str, str]] = []
    btc_reference = _load_btc_reference(run, fetch_failures)
    for symbol in run.symbols:
        try:
            symbol_rows = load_symbol_observations(run, symbol, btc_reference=btc_reference)
            if not symbol_rows:
                raise EvidenceDataError(f"no aligned rows for {symbol}")
            observations.extend(symbol_rows)
            coverage.append(
                {
                    "symbol": symbol,
                    "rows": len(symbol_rows),
                    "survivor_status": _survivor_status(run, symbol),
                    "funding_rows": len(symbol_rows),
                }
            )
        except Exception as exc:  # noqa: BLE001
            fetch_failures.append({"symbol": symbol, "error": str(exc)})

    if not observations:
        raise EvidenceDataError("0 aligned observation rows; check ccxt source/symbols/time window")

    spec = _hypothesis_spec(run, observations)
    result = run_backtest(spec)
    payload = cast(Mapping[str, Any], result.payload)
    artifact = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_60B_MICROSTRUCTURE_EVIDENCE",
        "public_boundary": (
            "raw ccxt-derived observations and full verdict JSON are private-only"
        ),
        "input": _run_to_dict(run),
        "coverage": {
            "requested_symbols": len(run.symbols),
            "covered_symbols": len({str(row['symbol']) for row in observations}),
            "requested_delisted_or_crash_symbols": len(run.delisted_crash_symbols),
            "covered_delisted_or_crash_symbols": sum(
                1 for item in coverage if item["symbol"] in run.delisted_crash_symbols
            ),
            "funding_rows": len(observations),
            "observation_rows": len(observations),
            "fetch_failures": fetch_failures,
            "by_symbol": coverage,
        },
        "spec": {
            "key": spec.key,
            "type": spec.hypothesis_type,
            "trial_n": spec.trial_count_n,
            "survivor_light": spec.survivor_light,
        },
        "verdict": _verdict_to_dict(result.verdict),
        "report": payload,
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = run.output_dir / f"microstructure-evidence-{stamp}.json"
    md_path = run.output_dir / f"microstructure-evidence-{stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(artifact, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result.verdict.verdict,
                "state": result.verdict.state,
                "reason": result.verdict.reason,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_symbol_observations(
    run: EvidenceRun,
    symbol: str,
    *,
    btc_reference: Sequence[Mapping[str, float | int]],
) -> list[dict[str, Any]]:
    start_dt = _parse_date(run.start)
    end_dt = _parse_date(run.end)
    funding = _load_funding_history(
        run.source, symbol, start_dt, end_dt, max_rows=run.max_bars_per_symbol
    )
    price_flow = _load_price_flow(
        run.source,
        symbol,
        run.timeframe,
        start_dt,
        end_dt,
        max_rows=run.max_bars_per_symbol,
    )
    oi = _load_open_interest_history(
        run.source,
        symbol,
        run.timeframe,
        start_dt,
        end_dt,
        max_rows=run.max_bars_per_symbol,
    )
    return _align_observations(
        symbol=symbol,
        funding=funding,
        price_flow=price_flow,
        btc_reference=btc_reference,
        open_interest=oi,
        survivor_status=_survivor_status(run, symbol),
    )


def _load_btc_reference(
    run: EvidenceRun, fetch_failures: list[dict[str, str]]
) -> list[dict[str, float | int]]:
    try:
        return _load_price_flow(
            run.source,
            run.btc_reference_symbol,
            run.timeframe,
            _parse_date(run.start),
            _parse_date(run.end),
            max_rows=run.max_bars_per_symbol,
        )
    except Exception as exc:  # noqa: BLE001
        fetch_failures.append(
            {"symbol": run.btc_reference_symbol, "error": f"btc_reference: {exc}"}
        )
        return []


def _hypothesis_spec(run: EvidenceRun, observations: Sequence[Mapping[str, Any]]) -> HypothesisSpec:
    grid = {
        "funding_abs_bps": [1.0, 3.0],
        "imbalance_abs": [0.10, 0.20],
        "oi_drop_abs": [0.02, 0.05],
        "score_threshold": [1, 2],
    }
    trial_n = (
        len(run.symbols)
        * len(grid["funding_abs_bps"])
        * len(grid["imbalance_abs"])
        * len(grid["oi_drop_abs"])
        * len(grid["score_threshold"])
    )
    spec_without_runner = HypothesisSpec(
        key="olympus60_microstructure_real_ccxt",
        hypothesis_type="event",
        universe=run.symbols,
        predeclared_signals=("funding_sign", "oi_price_divergence", "orderflow_imbalance"),
        params={
            "observations": [dict(row) for row in observations],
            "grid": grid,
            "locked_oos_fraction": 0.40,
            "fold_count": 6,
            "pbo_splits": 4,
            "pbo_threshold": 0.20,
            "annualization_periods": _annualization_periods(run.timeframe),
            "fdr_alpha": 0.10,
        },
        cost_model={
            "fee_bps": run.taker_fee_bps,
            "slippage_bps": run.slippage_bps,
            "funding_bps_per_period": 0.0,
            "funding_label": "perp funding debited from ccxt funding history observations",
        },
        benchmark="buy_and_hold",
        data_source=f"ccxt.{run.source}.funding+open_interest+perp_klines",
        trial_count_n=max(1, trial_n),
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=True,
    )
    return HypothesisSpec(
        key=spec_without_runner.key,
        hypothesis_type=spec_without_runner.hypothesis_type,
        universe=spec_without_runner.universe,
        predeclared_signals=spec_without_runner.predeclared_signals,
        params=spec_without_runner.params,
        cost_model=spec_without_runner.cost_model,
        benchmark=spec_without_runner.benchmark,
        data_source=spec_without_runner.data_source,
        trial_count_n=spec_without_runner.trial_count_n,
        discipline=spec_without_runner.discipline,
        survivor_light=spec_without_runner.survivor_light,
        runner=lambda: run_microstructure_perp_from_spec(spec_without_runner),
    )


def _load_funding_history(
    source: Source,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    exchange = _exchange(source)
    if not hasattr(exchange, "fetch_funding_rate_history"):
        raise EvidenceDataError(f"{source} does not expose fetch_funding_rate_history")
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows: list[dict[str, float | int]] = []
    seen: set[int] = set()
    while since < end_ms and len(rows) < max_rows:
        batch = _request_with_retry(
            exchange.fetch_funding_rate_history,
            symbol,
            since=since,
            limit=min(REQUEST_LIMIT, max_rows - len(rows)),
        )
        if not isinstance(batch, list) or not batch:
            break
        last_ts = None
        for item in batch:
            if not isinstance(item, Mapping):
                continue
            ts = _timestamp_ms(item)
            rate = _funding_rate(item)
            last_ts = ts if ts is not None else last_ts
            if ts is None or rate is None or ts < since or ts >= end_ms or ts in seen:
                continue
            rows.append({"timestamp": ts, "funding_rate": rate})
            seen.add(ts)
        if last_ts is None or last_ts + 1 <= since:
            break
        since = last_ts + 1
    rows.sort(key=lambda row: int(row["timestamp"]))
    if not rows:
        raise EvidenceDataError(f"no funding history returned for {symbol}")
    return rows


def _load_price_flow(
    source: Source,
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    if source == "binance":
        return _load_binance_futures_klines(symbol, timeframe, start_dt, end_dt, max_rows=max_rows)
    raise EvidenceDataError(
        f"{source} price-flow loader is not implemented; "
        "Binance futures klines provide taker-buy volume"
    )


def _load_binance_futures_klines(
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    exchange = _exchange("binance")
    method = getattr(exchange, "fapiPublicGetKlines", None)
    if method is None:
        raise EvidenceDataError("binance ccxt client lacks fapiPublicGetKlines")
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    market_id = _binance_market_id(symbol)
    rows: list[dict[str, float | int]] = []
    while since < end_ms and len(rows) < max_rows:
        batch = _request_with_retry(
            method,
            {
                "symbol": market_id,
                "interval": timeframe,
                "startTime": since,
                "endTime": end_ms,
                "limit": min(REQUEST_LIMIT, max_rows - len(rows)),
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        last_open = None
        for raw in batch:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 10:
                continue
            ts = _int_value(raw[0])
            close = _float_scalar(raw[4])
            volume = _float_scalar(raw[5])
            quote_volume = _float_scalar(raw[7]) if len(raw) > 7 else None
            taker_buy = _float_scalar(raw[9])
            if ts is None or close is None or volume is None or taker_buy is None:
                continue
            last_open = ts
            if ts < since or ts >= end_ms:
                continue
            buy_volume = max(taker_buy, 0.0)
            sell_volume = max(volume - buy_volume, 0.0)
            rows.append(
                {
                    "timestamp": ts,
                    "close": close,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "quote_volume_usd": max(quote_volume or 0.0, 0.0),
                }
            )
        if last_open is None:
            break
        next_since = last_open + _timeframe_ms(timeframe)
        if next_since <= since:
            break
        since = next_since
    if not rows:
        raise EvidenceDataError(f"no Binance futures kline rows returned for {symbol}")
    return rows


def _load_open_interest_history(
    source: Source,
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    exchange = _exchange(source)
    if hasattr(exchange, "fetch_open_interest_history"):
        rows = _load_unified_open_interest_history(
            exchange,
            symbol,
            timeframe,
            start_dt,
            end_dt,
            max_rows=max_rows,
        )
        if rows:
            return rows
    if source == "binance":
        return _load_binance_open_interest_history(
            symbol, timeframe, start_dt, end_dt, max_rows=max_rows
        )
    raise EvidenceDataError(f"no open interest history returned for {symbol}")


def _load_unified_open_interest_history(
    exchange: Any,
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows: list[dict[str, float | int]] = []
    while since < end_ms and len(rows) < max_rows:
        batch = _request_with_retry(
            exchange.fetch_open_interest_history,
            symbol,
            timeframe,
            since=since,
            limit=min(REQUEST_LIMIT, max_rows - len(rows)),
        )
        if not isinstance(batch, list) or not batch:
            break
        last_ts = None
        for item in batch:
            if not isinstance(item, Mapping):
                continue
            ts = _timestamp_ms(item)
            oi = _open_interest_value(item)
            last_ts = ts if ts is not None else last_ts
            if ts is None or oi is None or ts < since or ts >= end_ms:
                continue
            rows.append({"timestamp": ts, "open_interest": oi})
        if last_ts is None or last_ts + _timeframe_ms(timeframe) <= since:
            break
        since = last_ts + _timeframe_ms(timeframe)
    return rows


def _load_binance_open_interest_history(
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_rows: int,
) -> list[dict[str, float | int]]:
    exchange = _exchange("binance")
    method = getattr(exchange, "fapiDataGetOpenInterestHist", None)
    if method is None:
        raise EvidenceDataError("binance ccxt client lacks fapiDataGetOpenInterestHist")
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows: list[dict[str, float | int]] = []
    while since < end_ms and len(rows) < max_rows:
        batch = _request_with_retry(
            method,
            {
                "symbol": _binance_market_id(symbol),
                "period": timeframe,
                "startTime": since,
                "endTime": end_ms,
                "limit": min(REQUEST_LIMIT, max_rows - len(rows)),
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        last_ts = None
        for item in batch:
            if not isinstance(item, Mapping):
                continue
            ts = _timestamp_ms(item)
            oi = _open_interest_value(item)
            last_ts = ts if ts is not None else last_ts
            if ts is None or oi is None or ts < since or ts >= end_ms:
                continue
            rows.append({"timestamp": ts, "open_interest": oi})
        if last_ts is None or last_ts + _timeframe_ms(timeframe) <= since:
            break
        since = last_ts + _timeframe_ms(timeframe)
    if not rows:
        raise EvidenceDataError(f"no Binance open interest history returned for {symbol}")
    return rows


def _align_observations(
    *,
    symbol: str,
    funding: Sequence[Mapping[str, float | int]],
    price_flow: Sequence[Mapping[str, float | int]],
    open_interest: Sequence[Mapping[str, float | int]],
    survivor_status: str,
    btc_reference: Sequence[Mapping[str, float | int]] = (),
) -> list[dict[str, Any]]:
    price_sorted = sorted(price_flow, key=lambda item: int(item["timestamp"]))
    btc_sorted = sorted(btc_reference, key=lambda item: int(item["timestamp"]))
    oi_sorted = sorted(open_interest, key=lambda item: int(item["timestamp"]))
    rows: list[dict[str, Any]] = []
    for event in sorted(funding, key=lambda item: int(item["timestamp"])):
        ts = int(event["timestamp"])
        price = _last_at_or_before(price_sorted, ts)
        btc_price = _last_at_or_before(btc_sorted, ts)
        oi = _last_at_or_before(oi_sorted, ts)
        if price is None or oi is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "close": float(price["close"]),
                "btc_close": float(btc_price["close"]) if btc_price is not None else None,
                "open_interest": float(oi["open_interest"]),
                "funding_rate": float(event["funding_rate"]),
                "buy_volume": float(price["buy_volume"]),
                "sell_volume": float(price["sell_volume"]),
                "quote_volume_usd": float(price.get("quote_volume_usd", 0.0)),
                "survivor_status": survivor_status,
            }
        )
    return rows


def _last_at_or_before(
    rows: Sequence[Mapping[str, float | int]], timestamp: int
) -> Mapping[str, float | int] | None:
    candidate: Mapping[str, float | int] | None = None
    for row in rows:
        if int(row["timestamp"]) <= timestamp:
            candidate = row
        else:
            break
    return candidate


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    report = cast(Mapping[str, Any], payload["report"])
    verdict = cast(Mapping[str, Any], payload["verdict"])
    multiple_testing = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    coverage = cast(Mapping[str, Any], payload["coverage"])
    return "\n".join(
        [
            "# CODEX OLYMPUS 60 Microstructure Evidence",
            "",
            f"Generated: `{payload['generated_at']}`",
            f"Verdict: `{verdict.get('verdict')}`",
            f"Reason: {verdict.get('reason')}",
            f"JSON artifact: `{json_path}`",
            "",
            "## Coverage",
            "",
            f"- Requested symbols: `{coverage.get('requested_symbols')}`",
            f"- Covered symbols: `{coverage.get('covered_symbols')}`",
            "- Requested delisted/crash symbols: "
            f"`{coverage.get('requested_delisted_or_crash_symbols')}`",
            "- Covered delisted/crash symbols: "
            f"`{coverage.get('covered_delisted_or_crash_symbols')}`",
            f"- Funding rows: `{coverage.get('funding_rows')}`",
            "",
            "## Multiple Testing",
            "",
            f"- N: `{multiple_testing.get('candidate_count_n')}`",
            f"- FDR before: `{multiple_testing.get('fdr_before')}`",
            f"- FDR after: `{multiple_testing.get('fdr_after')}`",
            f"- PBO after survivors: `{multiple_testing.get('pbo_after_survivors')}`",
            "",
            "## Discipline",
            "",
            "- Funding sign + OI/price divergence + order-flow imbalance.",
            "- Signals observed at t; returns filled at t+1 inside the runner.",
            "- Fees, slippage, and perpetual funding are counted.",
            "- Benchmark is buy-and-hold.",
            "- Survivor-light data is capped below ROBUST.",
            "- Order-book event rate above 15000/hr is data-blocked.",
        ]
    ) + "\n"


def _run_from_env() -> EvidenceRun:
    symbols = tuple(_env_csv("MICROSTRUCTURE_EVIDENCE_SYMBOLS", DEFAULT_SYMBOLS))
    return EvidenceRun(
        source=cast(Source, os.getenv("MICROSTRUCTURE_EVIDENCE_SOURCE", DEFAULT_SOURCE)),
        symbols=symbols,
        start=os.getenv("MICROSTRUCTURE_EVIDENCE_START", DEFAULT_START),
        end=os.getenv("MICROSTRUCTURE_EVIDENCE_END", DEFAULT_END),
        timeframe=os.getenv("MICROSTRUCTURE_EVIDENCE_TIMEFRAME", "4h"),
        max_bars_per_symbol=_env_int("MICROSTRUCTURE_EVIDENCE_MAX_BARS", 1500),
        taker_fee_bps=_env_float("MICROSTRUCTURE_EVIDENCE_TAKER_FEE_BPS", 5.0),
        slippage_bps=_env_float("MICROSTRUCTURE_EVIDENCE_SLIPPAGE_BPS", 2.0),
        output_dir=private_dir_from_cli(
            os.getenv("MICROSTRUCTURE_EVIDENCE_OUTPUT_DIR"),
            default_task="olympus60",
        ),
        high_liquidity_symbols=tuple(
            _env_csv("MICROSTRUCTURE_EVIDENCE_HIGH_LIQUIDITY", tuple(HIGH_LIQUIDITY_DEFAULTS))
        ),
        delisted_crash_symbols=tuple(
            _env_csv("MICROSTRUCTURE_EVIDENCE_DELISTED_CRASH", tuple(DELISTED_CRASH_DEFAULTS))
        ),
        btc_reference_symbol=(
            os.getenv("MICROSTRUCTURE_EVIDENCE_BTC_REFERENCE_SYMBOL", "BTC/USDT:USDT").strip()
            or "BTC/USDT:USDT"
        ),
    )


def _run_to_dict(run: EvidenceRun) -> dict[str, Any]:
    return {
        "source": run.source,
        "symbols": list(run.symbols),
        "start": run.start,
        "end": run.end,
        "timeframe": run.timeframe,
        "max_bars_per_symbol": run.max_bars_per_symbol,
        "taker_fee_bps": run.taker_fee_bps,
        "slippage_bps": run.slippage_bps,
        "high_liquidity_symbols": list(run.high_liquidity_symbols),
        "delisted_crash_symbols": list(run.delisted_crash_symbols),
        "btc_reference_symbol": run.btc_reference_symbol,
    }


def _survivor_status(run: EvidenceRun, symbol: str) -> str:
    if symbol in run.delisted_crash_symbols:
        return "delisted_or_crash_basket_requested"
    if symbol in run.high_liquidity_symbols:
        return "active_high_liquidity_requested"
    return "survivor_light_requested"


def _exchange(source: Source) -> Any:
    ccxt = importlib.import_module("ccxt")
    if source == "binance":
        factory = ccxt.binanceusdm
        return factory({"enableRateLimit": True, "timeout": 10_000})
    factory = getattr(ccxt, source)
    return factory({"enableRateLimit": True, "timeout": 10_000, "options": {"defaultType": "swap"}})


def _request_with_retry(func: Any, *args: object, **kwargs: object) -> Any:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.25 * (attempt + 1))
    raise EvidenceDataError(str(last_error) if last_error else "ccxt request failed")


def _timestamp_ms(item: Mapping[str, Any]) -> int | None:
    for key in ("timestamp", "time", "sumOpenInterestValue"):
        if key in item and key != "sumOpenInterestValue":
            value = _int_value(item[key])
            if value is not None:
                return value
    info = item.get("info")
    if isinstance(info, Mapping):
        for key in ("timestamp", "time", "fundingTime"):
            value = _int_value(info.get(key))
            if value is not None:
                return value
    return None


def _funding_rate(item: Mapping[str, Any]) -> float | None:
    for key in ("fundingRate", "rate"):
        value = _float_scalar(item.get(key))
        if value is not None:
            return value
    info = item.get("info")
    if isinstance(info, Mapping):
        for key in ("fundingRate", "rate"):
            value = _float_scalar(info.get(key))
            if value is not None:
                return value
    return None


def _open_interest_value(item: Mapping[str, Any]) -> float | None:
    for key in ("openInterestAmount", "openInterestValue", "openInterest", "sumOpenInterest"):
        value = _float_scalar(item.get(key))
        if value is not None:
            return value
    info = item.get("info")
    if isinstance(info, Mapping):
        for key in ("sumOpenInterest", "openInterest", "openInterestAmount"):
            value = _float_scalar(info.get(key))
            if value is not None:
                return value
    return None


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


def _binance_market_id(symbol: str) -> str:
    base = symbol.split(":", 1)[0]
    return base.replace("/", "").upper()


def _parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timeframe_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 60 * 60_000
    if unit == "d":
        return amount * 24 * 60 * 60_000
    raise EvidenceDataError(f"unsupported timeframe: {timeframe}")


def _annualization_periods(timeframe: str) -> int:
    return max(1, int((365 * 24 * 60 * 60 * 1000) / _timeframe_ms(timeframe)))


def _env_csv(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return max(int(raw), 1) if raw else default


def _verdict_to_dict(verdict: Any) -> dict[str, Any]:
    return {
        "state": verdict.state,
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "data_adequacy": verdict.data_adequacy,
        "unlock_condition": verdict.unlock_condition,
        "candidate_count_n": verdict.candidate_count_n,
        "raw_survivors": verdict.raw_survivors,
        "fdr_survivors": verdict.fdr_survivors,
        "multiple_testing": dict(verdict.multiple_testing),
        "safety": dict(verdict.safety),
        "survivor_ceiling_applied": verdict.survivor_ceiling_applied,
    }


class EvidenceDataError(RuntimeError):
    pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceDataError as exc:
        print(json.dumps({"verdict": "INSUFFICIENT", "reason": str(exc)}, indent=2))
        raise SystemExit(0) from exc
