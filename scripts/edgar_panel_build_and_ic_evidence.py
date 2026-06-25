#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from aegis.edgar_full_universe_ic import (
    EdgarIcConfig,
    EdgarIcObservation,
    run_edgar_full_universe_ic,
)
from aegis.edgar_panel_builder import PanelBuildConfig, build_edgar_ic_panel
from aegis.edgar_pit import PitFundamentalStore, SecEdgarClient, extract_submission_metadata
from aegis.olympus_survivor_light import (
    PriceBar,
    download_wikipedia_sp500_html,
    parse_wikipedia_sp500_snapshot,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus82"
DEFAULT_START = date(2021, 3, 31)
DEFAULT_END = date(2025, 11, 30)


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_edgar_panel_build_and_ic_evidence(
        output_dir=output_dir,
        cache_dir=_cache_dir(args.cache_dir, output_dir),
        max_tickers=args.max_tickers,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        requests_per_second=args.sec_rps,
        price_chunk_size=args.price_chunk_size,
    )
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"edgar-panel-build-ic-{stamp}.json"
    md_path = output_dir / f"edgar-panel-build-ic-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    report = cast(dict[str, Any], payload["report"])
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": report["state"],
                "verdict": report["verdict"],
                "data_adequacy": report["data_adequacy"],
                "reason": report["reason"],
                "coverage": report["coverage"],
                "sharadar_decision": payload["sharadar_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_edgar_panel_build_and_ic_evidence(
    *,
    output_dir: Path,
    cache_dir: Path,
    max_tickers: int | None,
    start: date,
    end: date,
    requests_per_second: float,
    price_chunk_size: int,
) -> dict[str, Any]:
    ua = os.environ.get("AEGIS_SEC_USER_AGENT", "").strip()
    if not ua:
        report = run_edgar_full_universe_ic(
            [],
            coverage={
                "sec_user_agent_configured": False,
                "data_gate": "AEGIS_SEC_USER_AGENT is required for SEC EDGAR requests",
            },
        )
        return _payload(report=report, observations=[], coverage={"blocked_before_fetch": True})

    constituent_html = download_wikipedia_sp500_html(
        cache_path=cache_dir / "constituents" / "wikipedia-sp500.html",
        user_agent="AegisOlympusResearch/0.1",
    )
    snapshot = parse_wikipedia_sp500_snapshot(constituent_html, as_of_date=date.today())
    selected = _selected_current_symbols(snapshot.current, max_tickers=max_tickers)
    constituent_store = snapshot.store.filtered(selected)
    edgar, edgar_coverage = _fetch_edgar_stores(
        selected,
        snapshot.current,
        cache_dir=cache_dir / "edgar",
        user_agent=ua,
        requests_per_second=requests_per_second,
    )
    price_start = start - timedelta(days=10)
    price_end = _add_months(end, 7) + timedelta(days=10)
    prices, price_coverage = _load_or_fetch_prices(
        selected,
        cache_dir=cache_dir / "prices",
        start=price_start,
        end=price_end,
        chunk_size=price_chunk_size,
    )
    observations, panel_coverage = build_edgar_ic_panel(
        fundamentals=edgar,
        prices=prices,
        constituent_store=constituent_store,
        config=PanelBuildConfig(start=start, end=end),
    )
    coverage = {
        "sec_user_agent_configured": True,
        "sec_user_agent_recorded_in_public": False,
        "universe_source": snapshot.source_url,
        "current_constituents": len(snapshot.current),
        "change_events_used": len(snapshot.changes),
        "selected_symbols": len(selected),
        "as_of_universe_counts": {
            value.isoformat(): len(constituent_store.as_of(value))
            for value in (start, date(2023, 12, 31), end)
        },
        "edgar": edgar_coverage,
        "prices": price_coverage,
        "panel": panel_coverage,
        "survivorship": "survivor_light_free_current_plus_wikipedia_changes",
        "cache_dir": str(cache_dir),
    }
    report = run_edgar_full_universe_ic(observations, config=EdgarIcConfig(), coverage=coverage)
    return _payload(report=report, observations=observations, coverage=coverage)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a free EDGAR/yfinance monthly panel and run Olympus #81 IC."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=DEFAULT_END.isoformat())
    parser.add_argument("--sec-rps", type=float, default=5.0)
    parser.add_argument("--price-chunk-size", type=int, default=60)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Raw data cache directory. Defaults to AEGIS_EVIDENCE_CACHE_ROOT/<task>/cache "
            "or /mnt/blockstorage/aegis-strategies/incubating/<task>/cache when present."
        ),
    )
    return parser.parse_args()


def _cache_dir(raw: str | None, output_dir: Path) -> Path:
    if raw:
        return Path(raw)
    env_root = os.environ.get("AEGIS_EVIDENCE_CACHE_ROOT", "").strip()
    if env_root:
        return Path(env_root) / DEFAULT_TASK / "cache"
    blockstorage = Path("/mnt/blockstorage")
    if blockstorage.exists():
        return blockstorage / "aegis-strategies" / "incubating" / DEFAULT_TASK / "cache"
    return output_dir / "cache"


def _selected_current_symbols(current: Mapping[str, Any], *, max_tickers: int | None) -> list[str]:
    selected = [
        symbol
        for symbol, row in sorted(current.items())
        if getattr(row, "cik", None) is not None
    ]
    return selected[:max_tickers] if max_tickers is not None else selected


def _fetch_edgar_stores(
    symbols: Sequence[str],
    current: Mapping[str, Any],
    *,
    cache_dir: Path,
    user_agent: str,
    requests_per_second: float,
) -> tuple[dict[str, PitFundamentalStore], dict[str, Any]]:
    client = SecEdgarClient(
        cache_dir=cache_dir,
        user_agent=user_agent,
        requests_per_second=requests_per_second,
        timeout_seconds=30.0,
    )
    stores: dict[str, PitFundamentalStore] = {}
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        row = current[symbol]
        cik = getattr(row, "cik", None)
        if cik is None:
            continue
        try:
            companyfacts = client.fetch_companyfacts(str(cik))
            submissions = client.fetch_submissions(str(cik))
            metadata = extract_submission_metadata(
                ticker=symbol,
                cik=str(cik),
                payload=submissions,
                pilot_status="olympus82_full_free_panel_survivor_light",
            )
            stores[symbol] = PitFundamentalStore.from_companyfacts(
                ticker=symbol,
                cik=str(cik),
                payload=companyfacts,
                company_metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - evidence must continue bounded batches.
            failures.append({"symbol": symbol, "error": type(exc).__name__, "message": str(exc)})
    return stores, {
        "requested_symbols": len(symbols),
        "stores_built": len(stores),
        "failures": failures[:25],
        "failure_count": len(failures),
        "fetch_records": len(client.fetch_records),
        "cache_hits": sum(1 for item in client.fetch_records if item.cache_hit),
        "network_fetches": sum(1 for item in client.fetch_records if not item.cache_hit),
        "requests_per_second": requests_per_second,
    }


def _load_or_fetch_prices(
    symbols: Sequence[str],
    *,
    cache_dir: Path,
    start: date,
    end: date,
    chunk_size: int,
) -> tuple[dict[str, list[PriceBar]], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices: dict[str, list[PriceBar]] = {}
    missing: list[str] = []
    for symbol in symbols:
        bars = _read_price_csv(cache_dir / f"{_safe_file_symbol(symbol)}.csv")
        if bars:
            prices[symbol] = [bar for bar in bars if start <= bar.date <= end]
        else:
            missing.append(symbol)
    fetch_failures: list[dict[str, str]] = []
    for chunk in _chunks(missing, max(1, chunk_size)):
        fetched = _fetch_yfinance_chunk(chunk, start=start, end=end)
        for symbol in chunk:
            bars = fetched.get(symbol, [])
            if bars:
                _write_price_csv(cache_dir / f"{_safe_file_symbol(symbol)}.csv", bars)
                prices[symbol] = bars
            else:
                fetch_failures.append({"symbol": symbol, "reason": "no_bars_returned"})
    return prices, {
        "requested_symbols": len(symbols),
        "symbols_with_prices": len(prices),
        "cache_hits": len(symbols) - len(missing),
        "network_requested": len(missing),
        "failure_count": len(fetch_failures),
        "failures": fetch_failures[:25],
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def _fetch_yfinance_chunk(
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
) -> dict[str, list[PriceBar]]:
    import yfinance as yf  # type: ignore[import-untyped]

    if not symbols:
        return {}
    yahoo_by_symbol = {symbol: _yahoo_symbol(symbol) for symbol in symbols}
    frame = yf.download(
        list(yahoo_by_symbol.values()),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    result: dict[str, list[PriceBar]] = {}
    for symbol, yahoo_symbol in yahoo_by_symbol.items():
        try:
            subframe = frame[yahoo_symbol] if len(symbols) > 1 else frame
        except Exception:  # noqa: BLE001 - absent ticker in multi-index frame.
            result[symbol] = []
            continue
        result[symbol] = _bars_from_frame(subframe)
    return result


def _bars_from_frame(frame: Any) -> list[PriceBar]:
    if frame is None or bool(getattr(frame, "empty", False)):
        return []
    bars: list[PriceBar] = []
    for raw_index, row in frame.iterrows():
        close = _float_or_none(row.get("Close"))
        if close is None or close <= 0.0:
            continue
        bars.append(
            PriceBar(
                date=_coerce_date(raw_index),
                open=_float_or_default(row.get("Open"), close),
                high=_float_or_default(row.get("High"), close),
                low=_float_or_default(row.get("Low"), close),
                close=close,
                adj_close=_float_or_none(row.get("Adj Close")),
                volume=_float_or_default(row.get("Volume"), 0.0),
            )
        )
    return bars


def _read_price_csv(path: Path) -> list[PriceBar]:
    if not path.exists():
        return []
    bars: list[PriceBar] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            close = _float_or_none(row.get("Close"))
            raw_date = row.get("Date")
            if raw_date is None or close is None:
                continue
            bars.append(
                PriceBar(
                    date=date.fromisoformat(raw_date),
                    open=_float_or_default(row.get("Open"), close),
                    high=_float_or_default(row.get("High"), close),
                    low=_float_or_default(row.get("Low"), close),
                    close=close,
                    adj_close=_float_or_none(row.get("Adj Close")),
                    volume=_float_or_default(row.get("Volume"), 0.0),
                )
            )
    return bars


def _write_price_csv(path: Path, bars: Sequence[PriceBar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "Date": bar.date.isoformat(),
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Adj Close": bar.adj_close if bar.adj_close is not None else "",
                    "Volume": bar.volume,
                }
            )


def _payload(
    *,
    report: Mapping[str, Any],
    observations: Sequence[EdgarIcObservation],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "briefing": "CODEX_OLYMPUS_82_EDGAR_PANEL_BUILD_AND_IC",
        "generated_at": _utc_now().isoformat(),
        "report": report,
        "observation_count": len(observations),
        "observations": [_observation_to_json(row) for row in observations],
        "coverage": coverage,
        "sharadar_decision": _sharadar_decision(report),
        "safety": {
            "read_only": True,
            "live_trading": False,
            "orders": False,
            "wallet_or_account_access": False,
            "sec_user_agent_value_redacted": True,
        },
    }


def _observation_to_json(row: EdgarIcObservation) -> dict[str, Any]:
    data = asdict(row)
    data["as_of"] = row.as_of.isoformat()
    data["available_on"] = row.available_on.isoformat()
    return data


def _sharadar_decision(report: Mapping[str, Any]) -> dict[str, str]:
    verdict = str(report.get("verdict"))
    if verdict == "SUGGESTIVE_NEEDS_PAID_CONFIRM":
        return {
            "decision": "SHARADAR_WORTH_PAYING_TO_CONFIRM",
            "reason": (
                "free survivor-light panel found FDR/PBO-surviving IC; paid PIT data "
                "can unlock robust validation"
            ),
        }
    if verdict == "NO_EDGE":
        return {
            "decision": "DO_NOT_PAY_FOR_THIS_FACTOR_SET_NOW",
            "reason": "full free survivor-light panel found no factor-horizon IC after FDR/PBO",
        }
    return {
        "decision": "NO_PURCHASE_DECISION_FROM_CURRENT_DATA",
        "reason": "data gate did not reach full-universe monthly IC coverage",
    }


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    report = cast(Mapping[str, Any], payload["report"])
    coverage = cast(Mapping[str, Any], report.get("coverage", {}))
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    decision = cast(Mapping[str, str], payload["sharadar_decision"])
    return "\n".join(
        [
            "# Olympus #82 EDGAR Panel Build + IC",
            "",
            f"- Verdict: `{report['verdict']}`",
            f"- State: `{report['state']}`",
            f"- Data adequacy: `{report['data_adequacy']}`",
            f"- Reason: {report['reason']}",
            f"- Eligible rows: `{coverage.get('eligible_rows')}`",
            f"- Symbols: `{coverage.get('symbols')}`",
            f"- Periods: `{coverage.get('periods')}`",
            f"- Trial N: `{multiple.get('candidate_count_n')}`",
            f"- FDR survivors: `{multiple.get('fdr_survivors')}`",
            f"- Sharadar decision: `{decision['decision']}`",
            f"- JSON: `{json_path}`",
            "",
            "SEC user-agent value is intentionally not written to this artifact.",
        ]
    )


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _safe_file_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(".", "_")


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _float_or_default(value: object, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, 28)
    return date(year, month, day)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - host evidence runner is Python 3.10.


if __name__ == "__main__":
    raise SystemExit(main())
