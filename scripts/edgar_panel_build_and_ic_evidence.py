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
from typing import Any, Literal, cast
from urllib.request import Request, urlopen

from aegis.edgar_full_universe_ic import (
    EdgarIcConfig,
    EdgarIcObservation,
    run_edgar_full_universe_ic,
)
from aegis.edgar_panel_builder import (
    PanelBuildConfig,
    build_edgar_ic_panel,
    historical_universe_symbols,
    month_end_dates,
)
from aegis.edgar_pit import PitFundamentalStore, SecEdgarClient, extract_submission_metadata
from aegis.olympus_survivor_light import (
    HistoricalConstituentStore,
    PriceBar,
    download_wikipedia_sp500_html,
    parse_wikipedia_sp500_snapshot,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus82"
DEFAULT_START = date(2021, 3, 31)
DEFAULT_END = date(2025, 11, 30)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class _CikRow:
    def __init__(self, cik: str | None) -> None:
        self.cik = cik


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_edgar_panel_build_and_ic_evidence(
        output_dir=output_dir,
        cache_dir=_cache_dir(args.cache_dir, output_dir, task=args.task),
        task=args.task,
        universe_mode=args.universe_mode,
        comparison_json=Path(args.comparison_json) if args.comparison_json else None,
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
    task: str,
    universe_mode: Literal["current", "asof"],
    comparison_json: Path | None,
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
        return _payload(
            task=task,
            universe_mode=universe_mode,
            report=report,
            observations=[],
            coverage={"blocked_before_fetch": True},
            comparison=None,
        )

    constituent_html = download_wikipedia_sp500_html(
        cache_path=cache_dir / "constituents" / "wikipedia-sp500.html",
        user_agent="AegisOlympusResearch/0.1",
    )
    snapshot = parse_wikipedia_sp500_snapshot(constituent_html, as_of_date=date.today())
    rebalance_dates = month_end_dates(start, end)
    selected, cik_rows, universe_coverage = _selected_symbols_and_cik_rows(
        snapshot_current=snapshot.current,
        constituent_store=snapshot.store,
        rebalance_dates=rebalance_dates,
        cache_dir=cache_dir / "constituents",
        user_agent=ua,
        universe_mode=universe_mode,
        max_tickers=max_tickers,
    )
    constituent_store = snapshot.store.filtered(selected)
    edgar, edgar_coverage = _fetch_edgar_stores(
        selected,
        cik_rows,
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
    universe_coverage.update(
        _non_current_price_coverage(
            selected=selected,
            current_symbols=set(snapshot.current),
            prices=prices,
            raw_non_current_count=int(
                universe_coverage.get("non_current_historical_symbols_raw", 0)
            ),
            missing_cik_count=int(universe_coverage.get("symbols_missing_cik", 0)),
        )
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
        **universe_coverage,
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
    comparison = _comparison_to_current_only(report, comparison_json) if comparison_json else None
    return _payload(
        task=task,
        universe_mode=universe_mode,
        report=report,
        observations=observations,
        coverage=coverage,
        comparison=comparison,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a free EDGAR/yfinance monthly panel and run Olympus #81 IC."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--universe-mode", choices=("current", "asof"), default="current")
    parser.add_argument("--comparison-json", default=None)
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


def _cache_dir(raw: str | None, output_dir: Path, *, task: str) -> Path:
    if raw:
        return Path(raw)
    env_root = os.environ.get("AEGIS_EVIDENCE_CACHE_ROOT", "").strip()
    if env_root:
        return Path(env_root) / task / "cache"
    blockstorage = Path("/mnt/blockstorage")
    if blockstorage.exists():
        return blockstorage / "aegis-strategies" / "incubating" / task / "cache"
    return output_dir / "cache"


def _selected_current_symbols(current: Mapping[str, Any], *, max_tickers: int | None) -> list[str]:
    selected = [
        symbol
        for symbol, row in sorted(current.items())
        if getattr(row, "cik", None) is not None
    ]
    return selected[:max_tickers] if max_tickers is not None else selected


def _selected_symbols_and_cik_rows(
    *,
    snapshot_current: Mapping[str, Any],
    constituent_store: HistoricalConstituentStore,
    rebalance_dates: Sequence[date],
    cache_dir: Path,
    user_agent: str,
    universe_mode: Literal["current", "asof"],
    max_tickers: int | None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    current_symbols = _selected_current_symbols(snapshot_current, max_tickers=None)
    if universe_mode == "current":
        selected = current_symbols[:max_tickers] if max_tickers is not None else current_symbols
        return selected, dict(snapshot_current), {"universe_mode": "current"}

    raw_asof_symbols = sorted(historical_universe_symbols(constituent_store, rebalance_dates))
    if max_tickers is not None:
        raw_asof_symbols = raw_asof_symbols[:max_tickers]
    current_set = set(snapshot_current)
    sec_rows = _load_sec_company_ticker_rows(
        cache_dir / "sec-company-tickers.json",
        user_agent=user_agent,
    )
    cik_rows: dict[str, Any] = {symbol: row for symbol, row in snapshot_current.items()}
    cik_rows.update({symbol: _CikRow(cik) for symbol, cik in sec_rows.items()})
    selected = [
        symbol
        for symbol in raw_asof_symbols
        if symbol in cik_rows and getattr(cik_rows[symbol], "cik", None) is not None
    ]
    removed_raw = sorted(set(raw_asof_symbols) - current_set)
    removed_resolved = sorted(set(selected) - current_set)
    missing_cik = sorted(set(raw_asof_symbols) - set(selected))
    return selected, cik_rows, {
        "universe_mode": "asof",
        "raw_asof_universe_symbols": len(raw_asof_symbols),
        "current_symbols_in_snapshot": len(current_set),
        "non_current_historical_symbols_raw": len(removed_raw),
        "non_current_historical_symbols_cik_resolved": len(removed_resolved),
        "symbols_missing_cik": len(missing_cik),
        "missing_cik_examples": missing_cik[:25],
        "free_sec_ticker_map_used": True,
        "free_sec_ticker_map_cache": str(cache_dir / "sec-company-tickers.json"),
    }


def _load_sec_company_ticker_rows(path: Path, *, user_agent: str) -> dict[str, str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(SEC_COMPANY_TICKERS_URL, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=30.0) as response:  # noqa: S310 - fixed SEC HTTPS URL.
            payload = json.loads(response.read().decode("utf-8"))
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    rows: dict[str, str] = {}
    for raw_row in payload.values():
        if not isinstance(raw_row, dict):
            continue
        ticker = str(raw_row.get("ticker", "")).upper().strip()
        cik = str(raw_row.get("cik_str", "")).strip()
        if ticker and cik:
            rows[ticker] = cik
    return rows


def _non_current_price_coverage(
    *,
    selected: Sequence[str],
    current_symbols: set[str],
    prices: Mapping[str, Sequence[PriceBar]],
    raw_non_current_count: int,
    missing_cik_count: int,
) -> dict[str, Any]:
    non_current_selected = sorted(set(selected) - current_symbols)
    non_current_with_prices = sorted(
        symbol for symbol in non_current_selected if prices.get(symbol)
    )
    missing_price = sorted(set(non_current_selected) - set(non_current_with_prices))
    missing_total = max(0, missing_cik_count) + len(missing_price)
    denominator = raw_non_current_count if raw_non_current_count > 0 else 1
    return {
        "non_current_historical_symbols_price_requested": len(non_current_selected),
        "non_current_historical_symbols_with_prices": len(non_current_with_prices),
        "non_current_historical_symbols_missing_prices": len(missing_price),
        "non_current_historical_price_missing_examples": missing_price[:25],
        "non_current_historical_total_missing_rate": missing_total / denominator,
    }


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
    task: str,
    universe_mode: Literal["current", "asof"],
    report: Mapping[str, Any],
    observations: Sequence[EdgarIcObservation],
    coverage: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "briefing": _briefing_name(task),
        "generated_at": _utc_now().isoformat(),
        "report": report,
        "observation_count": len(observations),
        "observations": [_observation_to_json(row) for row in observations],
        "coverage": coverage,
        "universe_mode": universe_mode,
        "comparison_to_current_only": comparison,
        "sharadar_decision": _sharadar_decision(report, comparison=comparison),
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


def _briefing_name(task: str) -> str:
    if task == "olympus83":
        return "CODEX_OLYMPUS_83_EDGAR_DEBIASED_UNIVERSE_IC"
    return "CODEX_OLYMPUS_82_EDGAR_PANEL_BUILD_AND_IC"


def _sharadar_decision(
    report: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None,
) -> dict[str, str]:
    if comparison is not None:
        survivors = cast(Sequence[str], comparison.get("debiased_fdr_survivors", []))
        retained = cast(Sequence[str], comparison.get("retained_from_current_survivors", []))
        if survivors and retained:
            return {
                "decision": "SHARADAR_WORTH_PAYING_TO_CONFIRM",
                "reason": (
                    "free as-of universe retained FDR/PBO-surviving IC after adding "
                    "historical removed names; paid PIT data is the next confirmation gate"
                ),
            }
        return {
            "decision": "DO_NOT_PAY_SHARADAR_FOR_THIS_FACTOR_SET_NOW",
            "reason": (
                "current-only IC did not survive the free as-of universe debiasing gate"
            ),
        }
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


def _comparison_to_current_only(
    report: Mapping[str, Any],
    comparison_json: Path,
) -> dict[str, Any]:
    current_payload = json.loads(comparison_json.read_text(encoding="utf-8"))
    current_report = cast(Mapping[str, Any], current_payload["report"])
    current_multiple = cast(Mapping[str, Any], current_report.get("multiple_testing", {}))
    debiased_multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    current_survivors = [
        str(item) for item in current_multiple.get("fdr_survivor_keys", [])
    ]
    debiased_survivors = [
        str(item) for item in debiased_multiple.get("fdr_survivor_keys", [])
    ]
    retained = sorted(set(current_survivors) & set(debiased_survivors))
    trial_comparison: dict[str, dict[str, Any]] = {}
    current_trials = cast(Mapping[str, Any], current_report.get("trials", {}))
    debiased_trials = cast(Mapping[str, Any], report.get("trials", {}))
    for key in sorted(set(current_trials) | set(debiased_trials)):
        current_trial = cast(Mapping[str, Any], current_trials.get(key, {}))
        debiased_trial = cast(Mapping[str, Any], debiased_trials.get(key, {}))
        trial_comparison[key] = {
            "current_is_mean_ic": _nested_float(current_trial, "is_rank_ic", "mean"),
            "current_oos_mean_ic": _nested_float(current_trial, "oos_rank_ic", "mean"),
            "current_fdr": bool(current_trial.get("fdr_discovery", False)),
            "debiased_is_mean_ic": _nested_float(debiased_trial, "is_rank_ic", "mean"),
            "debiased_oos_mean_ic": _nested_float(debiased_trial, "oos_rank_ic", "mean"),
            "debiased_fdr": bool(debiased_trial.get("fdr_discovery", False)),
        }
    conclusion = (
        "SURVIVED_FREE_DEBIASING_PAY_SHARADAR_TO_CONFIRM"
        if debiased_survivors and retained
        else "COLLAPSED_AFTER_FREE_DEBIASING_DO_NOT_PAY_NOW"
    )
    return {
        "source_json": str(comparison_json),
        "current_fdr_survivors": current_survivors,
        "debiased_fdr_survivors": debiased_survivors,
        "retained_from_current_survivors": retained,
        "debiased_conclusion": conclusion,
        "trial_comparison": trial_comparison,
    }


def _nested_float(row: Mapping[str, Any], section: str, key: str) -> float | None:
    nested = row.get(section)
    if not isinstance(nested, Mapping):
        return None
    value = nested.get(key)
    if value is None:
        return None
    return float(value)


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    report = cast(Mapping[str, Any], payload["report"])
    coverage = cast(Mapping[str, Any], report.get("coverage", {}))
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    decision = cast(Mapping[str, str], payload["sharadar_decision"])
    comparison = cast(Mapping[str, Any] | None, payload.get("comparison_to_current_only"))
    title = (
        "# Olympus #83 EDGAR Debiased Universe IC"
        if payload.get("briefing") == "CODEX_OLYMPUS_83_EDGAR_DEBIASED_UNIVERSE_IC"
        else "# Olympus #82 EDGAR Panel Build + IC"
    )
    lines = [
        title,
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
    ]
    if comparison is not None:
        lines.extend(
            [
                f"- Current-only FDR survivors: `{comparison.get('current_fdr_survivors')}`",
                f"- Debiased FDR survivors: `{comparison.get('debiased_fdr_survivors')}`",
                "- Retained current survivors: "
                f"`{comparison.get('retained_from_current_survivors')}`",
                f"- Debias conclusion: `{comparison.get('debiased_conclusion')}`",
            ]
        )
    lines.extend(["", "SEC user-agent value is intentionally not written to this artifact."])
    return "\n".join(lines)


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
