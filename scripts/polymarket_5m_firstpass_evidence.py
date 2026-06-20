#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.polymarket_5m_firstpass import (
    UNMODELED_EXECUTION_COSTS,
    run_polymarket_5m_firstpass,
)
from aegis.polymarket_onchain import PolymarketDataApiClient, parse_closed_market, parse_trade
from aegis.polymarket_structural_scan import clob_token_ids
from aegis.private_paths import private_dir_from_cli

DEFAULT_USER_AGENT = "aegis-polymarket-5m-firstpass/0.1 read-only"
CLOB_BASE_URL = "https://clob.polymarket.com"


@dataclass(frozen=True)
class EvidenceRun:
    output_dir: Path
    max_markets: int
    market_page_size: int
    sleep_seconds: float
    timeout_seconds: float
    ccxt_source: str
    btc_symbol: str
    lookback_days: int
    min_aligned_markets: int
    min_entries: int
    max_trades_per_market: int


@dataclass(frozen=True)
class BtcMarket:
    condition_id: str
    slug: str
    title: str
    start_ts: int
    end_ts: int
    settlement_direction: str
    up_token_id: str
    down_token_id: str
    up_outcome_index: int
    down_outcome_index: int


def main() -> int:
    generated_at = datetime.now(UTC)
    run = _run_from_env()
    run.output_dir.mkdir(parents=True, exist_ok=True)
    raw_markets, btc_markets, enumeration = _load_btc_5m_markets(run)
    observations, price_errors = _build_observations(run, btc_markets)

    spec_without_runner = HypothesisSpec(
        key="olympus62_polymarket_btc_5m_onchain_fills",
        hypothesis_type="event",
        universe=("polymarket_btc_5m_updown",),
        predeclared_signals=("btc_5m_impulse", "near_settlement_direction_price"),
        params={
            "observations": observations,
            "optimistic_only": False,
            "unmodeled_execution_costs": UNMODELED_EXECUTION_COSTS,
        },
        cost_model={
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "funding_bps_per_period": 0.0,
            "funding_label": "not_applicable_binary_event_market",
            "execution_costs": "intentionally_unmodeled_in_optimistic_firstpass",
        },
        benchmark="no_trade_random_direction_no_impulse_filter",
        data_source="gamma_closed_markets+clob_prices_history+data_api_chain_indexed_fills+ccxt_btc_1m",
        trial_count_n=96,
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
    )
    spec = HypothesisSpec(
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
        runner=lambda: run_polymarket_5m_firstpass(observations),
    )
    backtest = run_backtest(spec)
    payload = cast(Mapping[str, Any], backtest.payload)
    coverage_gate = _coverage_gate(payload, run)
    final_verdict = (
        "INSUFFICIENT" if coverage_gate["status"] != "PASS" else str(backtest.verdict.verdict)
    )
    final_state = "INSUFFICIENT" if final_verdict == "INSUFFICIENT" else backtest.verdict.state
    final_reason = (
        str(coverage_gate["reason"])
        if coverage_gate["status"] != "PASS"
        else str(backtest.verdict.reason)
    )
    artifact = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_62_POLYMARKET_5M_ONCHAIN_FILLS",
        "input": _run_to_dict(run),
        "coverage": {
            "gamma_closed_markets_fetched": len(raw_markets),
            "gamma_enumeration": enumeration,
            "btc_5m_markets": len(btc_markets),
            "aligned_observations": len(observations),
            "aligned_observations_with_onchain_fills": sum(
                1
                for row in observations
                if row.get("up_onchain_fills") or row.get("down_onchain_fills")
            ),
            "date_range": payload.get("coverage", {}).get("date_range", {}),
            "entry_count": payload.get("coverage", {}).get("entry_count", 0),
            "entry_count_by_price_source": payload.get("coverage", {}).get(
                "entry_count_by_price_source", {}
            ),
            "entry_count_by_move_threshold": payload.get("coverage", {}).get(
                "entry_count_by_move_threshold", {}
            ),
            "price_history_errors": price_errors,
            "coverage_gate": coverage_gate,
        },
        "spec": {
            "key": spec.key,
            "trial_n": spec.trial_count_n,
            "price_source_dimension": ("observed_price", "onchain_fill"),
        },
        "verdict": {
            "state": final_state,
            "verdict": final_verdict,
            "reason": final_reason,
            "raw_backtest_verdict": backtest.verdict.verdict,
            "raw_backtest_state": backtest.verdict.state,
            "raw_backtest_reason": backtest.verdict.reason,
            "candidate_count_n": backtest.verdict.candidate_count_n,
            "fdr_survivors": backtest.verdict.fdr_survivors,
            "survivor_ceiling_applied": backtest.verdict.survivor_ceiling_applied,
        },
        "report": payload,
        "observations": observations,
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = run.output_dir / f"polymarket-5m-firstpass-{stamp}.json"
    md_path = run.output_dir / f"polymarket-5m-firstpass-{stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(artifact, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": backtest.verdict.verdict,
                "final_verdict": final_verdict,
                "state": final_state,
                "reason": final_reason,
                "json": str(json_path),
                "markdown": str(md_path),
                "markets": len(btc_markets),
                "observations": len(observations),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_btc_5m_markets(
    run: EvidenceRun,
) -> tuple[list[dict[str, Any]], list[BtcMarket], dict[str, Any]]:
    client = PolymarketDataApiClient(timeout_seconds=run.timeout_seconds)
    raw_markets: list[dict[str, Any]] = []
    parsed: list[BtcMarket] = []
    cutoff_ts = int(datetime.now(UTC).timestamp()) - run.lookback_days * 86_400
    offset = 0
    reached_end = False
    pagination_error: str | None = None
    while len(raw_markets) < run.max_markets:
        page_limit = min(run.market_page_size, run.max_markets - len(raw_markets))
        try:
            page = client.get_closed_markets(
                limit=page_limit,
                offset=offset,
                order="closedTime",
                ascending=False,
            )
        except HTTPError as exc:
            pagination_error = f"Gamma markets pagination stopped at offset {offset}: {exc}"
            break
        if not page:
            reached_end = True
            break
        raw_markets.extend(page)
        page_oldest_ts: int | None = None
        for raw in page:
            maybe_market = _btc_market_from_raw(raw)
            if maybe_market is not None:
                page_oldest_ts = (
                    maybe_market.end_ts
                    if page_oldest_ts is None
                    else min(page_oldest_ts, maybe_market.end_ts)
                )
                if maybe_market.end_ts >= cutoff_ts:
                    parsed.append(maybe_market)
        if len(page) < page_limit:
            reached_end = True
            break
        if page_oldest_ts is not None and page_oldest_ts < cutoff_ts:
            reached_end = True
            break
        offset += page_limit
        if run.sleep_seconds > 0:
            time.sleep(run.sleep_seconds)
    enumeration = {
        "closed_time_desc_pages": (len(raw_markets) + run.market_page_size - 1)
        // run.market_page_size,
        "lookback_days": run.lookback_days,
        "cutoff_ts": cutoff_ts,
        "reached_end_or_cutoff": reached_end,
        "hit_max_markets_limit": len(raw_markets) >= run.max_markets and not reached_end,
        "pagination_error": pagination_error,
    }
    return raw_markets, parsed, enumeration


def _btc_market_from_raw(raw: Mapping[str, Any]) -> BtcMarket | None:
    market = parse_closed_market(raw)
    token_ids = clob_token_ids(raw)
    if market is None or len(token_ids) != len(market.outcomes):
        return None
    if not _is_btc_5m_market(market.slug, market.title):
        return None
    outcome_to_token = dict(zip(market.outcomes, token_ids, strict=False))
    if "Up" not in outcome_to_token or "Down" not in outcome_to_token:
        return None
    outcome_to_index = {outcome: index for index, outcome in enumerate(market.outcomes)}
    end_ts = _end_ts(market.end_time)
    if end_ts is None:
        return None
    settlement = _settlement_direction(
        market.outcomes, tuple(float(price) for price in market.outcome_prices)
    )
    if settlement is None:
        return None
    return BtcMarket(
        condition_id=market.condition_id,
        slug=market.slug,
        title=market.title,
        start_ts=end_ts - 300,
        end_ts=end_ts,
        settlement_direction=settlement,
        up_token_id=outcome_to_token["Up"],
        down_token_id=outcome_to_token["Down"],
        up_outcome_index=outcome_to_index["Up"],
        down_outcome_index=outcome_to_index["Down"],
    )


def _build_observations(
    run: EvidenceRun,
    markets: Sequence[BtcMarket],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not markets:
        return [], []
    btc_prices = _load_btc_1m_prices(
        run,
        min(m.start_ts for m in markets),
        max(m.end_ts for m in markets),
    )
    observations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for market in markets:
        try:
            up_prices = _load_prices_history(
                market.up_token_id, market.start_ts, market.end_ts, run
            )
            down_prices = _load_prices_history(
                market.down_token_id, market.start_ts, market.end_ts, run
            )
            up_fills: list[dict[str, float | int]]
            down_fills: list[dict[str, float | int]]
            try:
                up_fills, down_fills = _load_onchain_fill_prices(market, run)
            except Exception as exc:  # noqa: BLE001
                up_fills, down_fills = [], []
                errors.append(
                    {
                        "condition_id": market.condition_id,
                        "slug": market.slug,
                        "stage": "onchain_fill_history",
                        "error": str(exc),
                    }
                )
            move = _btc_move_usd(btc_prices, market.start_ts, market.end_ts)
            if move is None:
                errors.append({"condition_id": market.condition_id, "error": "missing_btc_1m_move"})
                continue
            observations.append(
                {
                    "condition_id": market.condition_id,
                    "slug": market.slug,
                    "title": market.title,
                    "start_ts": market.start_ts,
                    "end_ts": market.end_ts,
                    "settlement_direction": market.settlement_direction,
                    "btc_move_usd": move,
                    "btc_direction": "Up" if move >= 0 else "Down",
                    "up_prices": up_prices,
                    "down_prices": down_prices,
                    "up_onchain_fills": up_fills,
                    "down_onchain_fills": down_fills,
                    "onchain_fill_source": "polymarket_data_api_chain_indexed_trades",
                    "enable_onchain_price_source": True,
                }
            )
            time.sleep(run.sleep_seconds)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "condition_id": market.condition_id,
                    "slug": market.slug,
                    "stage": "observed_price_history_or_btc_alignment",
                    "start_ts": str(market.start_ts),
                    "end_ts": str(market.end_ts),
                    "up_token_id": market.up_token_id,
                    "down_token_id": market.down_token_id,
                    "error": str(exc),
                }
            )
    return observations, errors


def _load_onchain_fill_prices(
    market: BtcMarket,
    run: EvidenceRun,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    client = PolymarketDataApiClient(timeout_seconds=run.timeout_seconds)
    up: list[dict[str, float | int]] = []
    down: list[dict[str, float | int]] = []
    offset = 0
    limit = 500
    loaded = 0
    while loaded < run.max_trades_per_market:
        page_limit = min(limit, run.max_trades_per_market - loaded)
        page: list[dict[str, Any]] | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                page = client.get_trades(
                    market.condition_id,
                    limit=page_limit,
                    offset=offset,
                    taker_only=False,
                )
                break
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {400, 429, 500, 502, 503, 504}:
                    raise
                time.sleep(run.sleep_seconds * (attempt + 1) + 0.25)
        if page is None:
            if loaded > 0:
                break
            if last_error is not None:
                raise last_error
            break
        if not page:
            break
        for raw in page:
            timestamp = _int_value(raw.get("timestamp"))
            price = _float_value(raw.get("price"))
            if timestamp is None or price is None or not 0.0 <= price <= 1.0:
                continue
            if timestamp < market.start_ts or timestamp > market.end_ts:
                continue
            row = {"timestamp": timestamp, "price": price}
            asset = str(raw.get("asset") or "")
            if asset == market.up_token_id:
                up.append(row)
                continue
            if asset == market.down_token_id:
                down.append(row)
                continue
            trade = parse_trade(raw)
            if trade is None:
                continue
            if trade.outcome_index == market.up_outcome_index:
                up.append(row)
            elif trade.outcome_index == market.down_outcome_index:
                down.append(row)
        loaded += len(page)
        if len(page) < page_limit:
            break
        offset += page_limit
        if run.sleep_seconds > 0:
            time.sleep(run.sleep_seconds)
    return sorted(up, key=lambda row: int(row["timestamp"])), sorted(
        down, key=lambda row: int(row["timestamp"])
    )


def _load_btc_1m_prices(
    run: EvidenceRun, start_ts: int, end_ts: int
) -> list[dict[str, float | int]]:
    ccxt = importlib.import_module("ccxt")
    factory = getattr(ccxt, run.ccxt_source)
    exchange = factory({"enableRateLimit": True, "timeout": 20_000})
    since = (start_ts - 60) * 1000
    end_ms = (end_ts + 60) * 1000
    rows: list[dict[str, float | int]] = []
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(run.btc_symbol, timeframe="1m", since=since, limit=1000)
        if not isinstance(batch, list) or not batch:
            break
        last_ts = None
        for raw in batch:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 5:
                continue
            ts = _int_value(raw[0])
            close = _float_value(raw[4])
            if ts is None or close is None:
                continue
            last_ts = ts
            if ts <= end_ms:
                rows.append({"timestamp": ts // 1000, "close": close})
        if last_ts is None or last_ts + 60_000 <= since or last_ts > end_ms:
            break
        since = last_ts + 60_000
    if not rows:
        raise EvidenceDataError("ccxt returned 0 BTC 1m kline rows")
    dedup = {int(row["timestamp"]): row for row in rows}
    return [dedup[key] for key in sorted(dedup)]


def _load_prices_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    run: EvidenceRun,
) -> list[dict[str, float | int]]:
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1}
    request = Request(
        f"{CLOB_BASE_URL}/prices-history?{urlencode(params)}",
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    data: Any = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=run.timeout_seconds) as response:
                data = json.loads(response.read().decode())
            break
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {400, 429, 500, 502, 503, 504}:
                raise
            time.sleep(run.sleep_seconds * (attempt + 1) + 0.25)
    if data is None:
        if last_error is not None:
            raise last_error
        raise EvidenceDataError("empty CLOB prices-history response")
    history = data.get("history") if isinstance(data, Mapping) else None
    if not isinstance(history, list):
        raise EvidenceDataError("unexpected CLOB prices-history response")
    rows: list[dict[str, float | int]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        ts = _int_value(item.get("t"))
        price = _float_value(item.get("p"))
        if ts is None or price is None or not 0.0 <= price <= 1.0:
            continue
        rows.append({"timestamp": ts, "price": price})
    return rows


def _btc_move_usd(
    prices: Sequence[Mapping[str, float | int]],
    start_ts: int,
    end_ts: int,
) -> float | None:
    start = _last_price_at_or_before(prices, start_ts)
    end = _last_price_at_or_before(prices, end_ts)
    if start is None or end is None:
        return None
    return end - start


def _last_price_at_or_before(
    prices: Sequence[Mapping[str, float | int]],
    timestamp: int,
) -> float | None:
    value: float | None = None
    for row in prices:
        if int(row["timestamp"]) <= timestamp:
            value = float(row["close"])
        else:
            break
    return value


def _is_btc_5m_market(slug: str, title: str) -> bool:
    slug_text = slug.lower()
    title_text = title.lower()
    return slug_text.startswith("btc-updown-5m-") or (
        "bitcoin up or down" in title_text
        and (" 5m" in title_text or " 5 min" in title_text or " 5 minute" in title_text)
    )


def _settlement_direction(outcomes: Sequence[str], prices: Sequence[float]) -> str | None:
    if len(outcomes) != len(prices):
        return None
    winners = [outcome for outcome, price in zip(outcomes, prices, strict=False) if price >= 0.999]
    if len(winners) != 1:
        return None
    if winners[0] in {"Up", "Down"}:
        return winners[0]
    return None


def _end_ts(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    verdict = cast(Mapping[str, Any], payload["verdict"])
    coverage = cast(Mapping[str, Any], payload["coverage"])
    report = cast(Mapping[str, Any], payload["report"])
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}))
    return "\n".join(
        [
            "# CODEX OLYMPUS 62 Polymarket 5m Onchain Fills",
            "",
            f"Verdict: `{verdict.get('verdict')}`",
            f"Reason: {verdict.get('reason')}",
            f"JSON artifact: `{json_path}`",
            "",
            "## Coverage",
            "",
            f"- Gamma markets fetched: `{coverage.get('gamma_closed_markets_fetched')}`",
            f"- BTC 5m markets: `{coverage.get('btc_5m_markets')}`",
            f"- Aligned observations: `{coverage.get('aligned_observations')}`",
            "- Aligned observations with onchain fills: "
            f"`{coverage.get('aligned_observations_with_onchain_fills')}`",
            f"- Entry count: `{coverage.get('entry_count')}`",
            f"- Entry count by price source: `{coverage.get('entry_count_by_price_source')}`",
            "",
            "## Multiple Testing",
            "",
            f"- N: `{multiple.get('candidate_count_n')}`",
            f"- FDR after: `{multiple.get('fdr_after')}`",
            f"- PBO after survivors: `{multiple.get('pbo_after_survivors')}`",
            "",
            "## Optimistic Boundary",
            "",
            "- Compares observed CLOB prices-history with chain-indexed historical fills "
            "when available.",
            "- Queue priority, market impact, final-second cancellations, FAK non-fill, "
            "stale quote, and depth are still unmodeled.",
            "- Positive results remain capped at SUGGESTIVE_NEEDS_EXECUTION_VALIDATION.",
        ]
    ) + "\n"


def _coverage_gate(payload: Mapping[str, Any], run: EvidenceRun) -> dict[str, object]:
    coverage = payload.get("coverage", {})
    if not isinstance(coverage, Mapping):
        return {"status": "FAIL", "reason": "missing coverage payload"}
    markets = _int_from_mapping(coverage, "market_count")
    entries = _int_from_mapping(coverage, "entry_count")
    onchain_markets = _int_from_mapping(coverage, "onchain_fill_observation_count")
    if markets < run.min_aligned_markets:
        return {
            "status": "FAIL",
            "reason": (
                f"aligned markets {markets} < required {run.min_aligned_markets}; "
                "free historical coverage remains insufficient"
            ),
            "aligned_markets": markets,
            "entry_count": entries,
            "onchain_fill_observation_count": onchain_markets,
        }
    if entries < run.min_entries:
        return {
            "status": "FAIL",
            "reason": (
                f"grid entries {entries} < required {run.min_entries}; "
                "sample is too small for a #62 decision"
            ),
            "aligned_markets": markets,
            "entry_count": entries,
            "onchain_fill_observation_count": onchain_markets,
        }
    return {
        "status": "PASS",
        "reason": "coverage gate passed",
        "aligned_markets": markets,
        "entry_count": entries,
        "onchain_fill_observation_count": onchain_markets,
    }


def _run_from_env() -> EvidenceRun:
    return EvidenceRun(
        output_dir=private_dir_from_cli(
            os.getenv("POLYMARKET_5M_OUTPUT_DIR"),
            default_task=os.getenv("POLYMARKET_5M_OUTPUT_TASK", "olympus62"),
        ),
        max_markets=_env_int("POLYMARKET_5M_MAX_MARKETS", 10_000),
        market_page_size=_env_int("POLYMARKET_5M_MARKET_PAGE_SIZE", 100),
        sleep_seconds=_env_float("POLYMARKET_5M_SLEEP_SECONDS", 0.05),
        timeout_seconds=_env_float("POLYMARKET_5M_TIMEOUT_SECONDS", 20.0),
        ccxt_source=os.getenv("POLYMARKET_5M_CCXT_SOURCE", "binanceusdm"),
        btc_symbol=os.getenv("POLYMARKET_5M_BTC_SYMBOL", "BTC/USDT:USDT"),
        lookback_days=_env_int("POLYMARKET_5M_LOOKBACK_DAYS", 180),
        min_aligned_markets=_env_int("POLYMARKET_5M_MIN_ALIGNED_MARKETS", 100),
        min_entries=_env_int("POLYMARKET_5M_MIN_ENTRIES", 100),
        max_trades_per_market=_env_int("POLYMARKET_5M_MAX_TRADES_PER_MARKET", 5_000),
    )


def _run_to_dict(run: EvidenceRun) -> dict[str, Any]:
    return {
        "max_markets": run.max_markets,
        "market_page_size": run.market_page_size,
        "sleep_seconds": run.sleep_seconds,
        "timeout_seconds": run.timeout_seconds,
        "ccxt_source": run.ccxt_source,
        "btc_symbol": run.btc_symbol,
        "lookback_days": run.lookback_days,
        "min_aligned_markets": run.min_aligned_markets,
        "min_entries": run.min_entries,
        "max_trades_per_market": run.max_trades_per_market,
    }


def _int_from_mapping(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return max(1, int(value)) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


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


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class EvidenceDataError(RuntimeError):
    pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceDataError as exc:
        print(json.dumps({"verdict": "INSUFFICIENT", "reason": str(exc)}, indent=2))
        raise SystemExit(0) from exc
