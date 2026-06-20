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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, run_backtest
from aegis.polymarket_5m_firstpass import (
    UNMODELED_EXECUTION_COSTS,
    run_polymarket_5m_firstpass,
)
from aegis.polymarket_onchain import PolymarketDataApiClient, parse_closed_market
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


def main() -> int:
    generated_at = datetime.now(UTC)
    run = _run_from_env()
    run.output_dir.mkdir(parents=True, exist_ok=True)
    raw_markets, btc_markets = _load_btc_5m_markets(run)
    observations, price_errors = _build_observations(run, btc_markets)
    if not observations:
        raise EvidenceDataError("0 aligned BTC 5m Polymarket observations")

    spec_without_runner = HypothesisSpec(
        key="olympus61_polymarket_btc_5m_firstpass",
        hypothesis_type="event",
        universe=("polymarket_btc_5m_updown",),
        predeclared_signals=("btc_5m_impulse", "near_settlement_direction_price"),
        params={
            "observations": observations,
            "optimistic_only": True,
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
        data_source="gamma_closed_markets+clob_prices_history+ccxt_btc_1m",
        trial_count_n=48,
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
    artifact = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_61_POLYMARKET_5M_FIRSTPASS",
        "input": _run_to_dict(run),
        "coverage": {
            "gamma_closed_markets_fetched": len(raw_markets),
            "btc_5m_markets": len(btc_markets),
            "aligned_observations": len(observations),
            "date_range": payload.get("coverage", {}).get("date_range", {}),
            "entry_count": payload.get("coverage", {}).get("entry_count", 0),
            "entry_count_by_move_threshold": payload.get("coverage", {}).get(
                "entry_count_by_move_threshold", {}
            ),
            "price_history_errors": price_errors,
        },
        "spec": {
            "key": spec.key,
            "trial_n": spec.trial_count_n,
            "optimistic_only": True,
        },
        "verdict": {
            "state": backtest.verdict.state,
            "verdict": backtest.verdict.verdict,
            "reason": backtest.verdict.reason,
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
                "state": backtest.verdict.state,
                "reason": backtest.verdict.reason,
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


def _load_btc_5m_markets(run: EvidenceRun) -> tuple[list[dict[str, Any]], list[BtcMarket]]:
    client = PolymarketDataApiClient(timeout_seconds=run.timeout_seconds)
    raw_markets = list(
        client.iter_closed_markets(
            limit=run.market_page_size,
            max_markets=run.max_markets,
            sleep_seconds=run.sleep_seconds,
            order="closedTime",
            ascending=False,
        )
    )
    parsed: list[BtcMarket] = []
    for raw in raw_markets:
        market = parse_closed_market(raw)
        token_ids = clob_token_ids(raw)
        if market is None or len(token_ids) != len(market.outcomes):
            continue
        if not _is_btc_5m_market(market.slug, market.title):
            continue
        outcome_to_token = dict(zip(market.outcomes, token_ids, strict=False))
        if "Up" not in outcome_to_token or "Down" not in outcome_to_token:
            continue
        end_ts = _end_ts(market.end_time)
        if end_ts is None:
            continue
        settlement = _settlement_direction(
            market.outcomes, tuple(float(p) for p in market.outcome_prices)
        )
        if settlement is None:
            continue
        parsed.append(
            BtcMarket(
                condition_id=market.condition_id,
                slug=market.slug,
                title=market.title,
                start_ts=end_ts - 300,
                end_ts=end_ts,
                settlement_direction=settlement,
                up_token_id=outcome_to_token["Up"],
                down_token_id=outcome_to_token["Down"],
            )
        )
    return raw_markets, parsed


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
                }
            )
            time.sleep(run.sleep_seconds)
        except Exception as exc:  # noqa: BLE001
            errors.append({"condition_id": market.condition_id, "error": str(exc)})
    return observations, errors


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
    with urlopen(request, timeout=run.timeout_seconds) as response:
        data = json.loads(response.read().decode())
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
    text = f"{slug} {title}".lower()
    return ("btc-updown-5m" in text) or (
        "bitcoin up or down" in text and "5m" in text.replace(" ", "")
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
            "# CODEX OLYMPUS 61 Polymarket 5m Firstpass",
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
            f"- Entry count: `{coverage.get('entry_count')}`",
            "",
            "## Multiple Testing",
            "",
            f"- N: `{multiple.get('candidate_count_n')}`",
            f"- FDR after: `{multiple.get('fdr_after')}`",
            f"- PBO after survivors: `{multiple.get('pbo_after_survivors')}`",
            "",
            "## Optimistic Boundary",
            "",
            "- Uses observed price history, not historical executable asks.",
            "- Spread, depth, FAK non-fill, stale quote, and last-second reversal are unmodeled.",
            "- Positive results are capped at SUGGESTIVE_NEEDS_EXECUTION_VALIDATION.",
        ]
    ) + "\n"


def _run_from_env() -> EvidenceRun:
    return EvidenceRun(
        output_dir=private_dir_from_cli(
            os.getenv("POLYMARKET_5M_OUTPUT_DIR"),
            default_task="olympus61",
        ),
        max_markets=_env_int("POLYMARKET_5M_MAX_MARKETS", 500),
        market_page_size=_env_int("POLYMARKET_5M_MARKET_PAGE_SIZE", 100),
        sleep_seconds=_env_float("POLYMARKET_5M_SLEEP_SECONDS", 0.05),
        timeout_seconds=_env_float("POLYMARKET_5M_TIMEOUT_SECONDS", 20.0),
        ccxt_source=os.getenv("POLYMARKET_5M_CCXT_SOURCE", "binanceusdm"),
        btc_symbol=os.getenv("POLYMARKET_5M_BTC_SYMBOL", "BTC/USDT:USDT"),
    )


def _run_to_dict(run: EvidenceRun) -> dict[str, Any]:
    return {
        "max_markets": run.max_markets,
        "market_page_size": run.market_page_size,
        "sleep_seconds": run.sleep_seconds,
        "timeout_seconds": run.timeout_seconds,
        "ccxt_source": run.ccxt_source,
        "btc_symbol": run.btc_symbol,
    }


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
