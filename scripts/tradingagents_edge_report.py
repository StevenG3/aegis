from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sqlite3
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from aegis.private_paths import private_dir_from_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = REPO_ROOT / "services" / "orchestrator"
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from db import connect  # noqa: E402

DEFAULT_SOURCE = "tradingagents"
DEFAULT_BENCHMARK_SOURCE = "binance"
DEFAULT_HORIZON_HOURS = 24
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_FUNDING_BPS_PER_8H = 0.0
DEFAULT_CASH_RATE_ANNUAL = 0.04
DEFAULT_MIN_N = 30
DEFAULT_BUCKET_MIN_N = 10
DEFAULT_ANALYST_MIN_N = 10
DEFAULT_FDR_ALPHA = 0.10

AgentVerdict = Literal["AGENT_EDGE", "NO_EDGE", "INSUFFICIENT"]
Action = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class CostModel:
    fee_bps: float
    slippage_bps: float
    funding_bps_per_8h: float
    cash_rate_annual: float

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0

    def funding_cost(self, action: str, horizon_hours: int) -> float:
        if action.lower() != "sell":
            return 0.0
        periods = max(horizon_hours / 8.0, 0.0)
        return periods * self.funding_bps_per_8h / 10_000.0

    def cash_return(self, horizon_hours: int) -> float:
        years = max(horizon_hours, 0) / (365.0 * 24.0)
        return float((1.0 + self.cash_rate_annual) ** years - 1.0)


@dataclass(frozen=True)
class ForwardPrices:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float


@dataclass(frozen=True)
class Recommendation:
    scorecard_id: str
    actor: str
    symbol: str
    action: Action
    source: str
    created_at: datetime
    conviction: float | None
    factors: tuple[dict[str, object], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ForwardSample:
    scorecard_id: str
    actor: str
    symbol: str
    action: Action
    created_at: datetime
    entry_time: datetime
    exit_time: datetime
    conviction: float | None
    confidence_bucket: str
    analyst_names: tuple[str, ...]
    strategy_return: float
    buy_hold_return: float
    cash_return: float
    excess_vs_buy_hold: float
    excess_vs_cash: float
    gross_directional_return: float
    total_cost: float


PriceFetcher = Callable[[str, datetime, int], ForwardPrices]
RecommendationLoader = Callable[..., list[Recommendation]]


class EmptyDataSourceError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TradingAgents forward recommendation edge."
    )
    parser.add_argument("--actor", default=None)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--benchmark-source", default=DEFAULT_BENCHMARK_SOURCE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--funding-bps-per-8h", type=float, default=DEFAULT_FUNDING_BPS_PER_8H)
    parser.add_argument("--cash-rate-annual", type=float, default=DEFAULT_CASH_RATE_ANNUAL)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--bucket-min-n", type=int, default=DEFAULT_BUCKET_MIN_N)
    parser.add_argument("--analyst-min-n", type=int, default=DEFAULT_ANALYST_MIN_N)
    parser.add_argument("--fdr-alpha", type=float, default=DEFAULT_FDR_ALPHA)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    try:
        report = build_report(
            actor=args.actor,
            source=args.source,
            benchmark_source=args.benchmark_source,
            horizon_hours=args.horizon_hours,
            costs=CostModel(
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                funding_bps_per_8h=args.funding_bps_per_8h,
                cash_rate_annual=args.cash_rate_annual,
            ),
            min_n=args.min_n,
            bucket_min_n=args.bucket_min_n,
            analyst_min_n=args.analyst_min_n,
            fdr_alpha=args.fdr_alpha,
        )
    except EmptyDataSourceError as exc:
        print(
            json.dumps(
                {
                    "error": "EMPTY_DATA_SOURCE",
                    "reason": str(exc),
                    "hint": "Connect the real orchestrator database before rerunning.",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if not args.no_write:
        output_dir = private_dir_from_cli(args.output_dir, default_task="olympus51")
        report["written_files"] = write_report(report, output_dir)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def build_report(
    *,
    actor: str | None,
    source: str,
    benchmark_source: str,
    horizon_hours: int,
    costs: CostModel,
    min_n: int = DEFAULT_MIN_N,
    bucket_min_n: int = DEFAULT_BUCKET_MIN_N,
    analyst_min_n: int = DEFAULT_ANALYST_MIN_N,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    price_fetcher: PriceFetcher | None = None,
    recommendation_loader: RecommendationLoader | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    loader = recommendation_loader or _load_recommendations
    recommendations = loader(actor=actor, source=source)
    if not recommendations:
        raise EmptyDataSourceError(
            "TradingAgents recommendation source returned 0 rows for "
            f"source={source!r}. This is a data-source/configuration error, not INSUFFICIENT."
        )
    fetcher = price_fetcher or _ccxt_forward_price_fetcher(benchmark_source)
    samples: list[ForwardSample] = []
    skipped: list[dict[str, str]] = []
    for recommendation in recommendations:
        sample = _evaluate_recommendation(recommendation, horizon_hours, costs, fetcher, skipped)
        if sample is not None:
            samples.append(sample)

    overall = _group_stats(samples, min_n=min_n)
    buckets = _bucket_stats(samples, min_n=bucket_min_n, fdr_alpha=fdr_alpha)
    analysts = _analyst_stats(samples, min_n=analyst_min_n, fdr_alpha=fdr_alpha)
    verdict, reason = _verdict(overall, min_n)
    report = {
        "generated_at": generated_at.isoformat(),
        "briefing": "CODEX_OLYMPUS_51_TRADINGAGENTS_EDGE",
        "verdict": verdict,
        "reason": reason,
        "scope": {
            "source": source,
            "actor_filter_applied": actor is not None,
            "horizon_hours": horizon_hours,
            "benchmark_source": benchmark_source,
        },
        "predeclared": {
            "entry": "first 1h bar strictly after scorecard.created_at (t+1)",
            "exit": "first 1h bar at or after entry_time + horizon_hours",
            "benchmarks": ["same-symbol buy&hold", "cash"],
            "min_n": min_n,
            "bucket_min_n": bucket_min_n,
            "analyst_min_n": analyst_min_n,
            "fdr_alpha": fdr_alpha,
        },
        "cost_model": {
            "fee_bps": costs.fee_bps,
            "slippage_bps": costs.slippage_bps,
            "funding_bps_per_8h": costs.funding_bps_per_8h,
            "cash_rate_annual": costs.cash_rate_annual,
            "round_trip_cost": costs.round_trip_cost,
        },
        "sample_counts": {
            "recommendations_loaded": len(recommendations),
            "evaluated_n": len(samples),
            "skipped_n": len(skipped),
        },
        "overall": overall,
        "buckets": buckets,
        "analysts": analysts,
        "private_rows": [_sample_to_private_row(sample) for sample in samples],
        "skipped": skipped,
        "sanitized_public_summary": _sanitized_summary(verdict, overall, buckets, analysts),
        "safety": {
            "mode": "read-only evaluation",
            "orders": "disabled",
            "wallet_or_account_access": "none",
            "live_trading": "disabled",
        },
    }
    report["human_readable"] = _markdown(report)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(report["generated_at"])).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"tradingagents-edge-{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(str(report["human_readable"]), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _load_recommendations(*, actor: str | None, source: str) -> list[Recommendation]:
    clauses = ["source = ?"]
    params: list[object] = [source]
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    where = " where " + " and ".join(clauses)
    with connect() as conn:
        rows = conn.execute(
            f"""
            select scorecard_id, actor, symbol, action, source, payload_json, created_at
            from scorecards
            {where}
            order by created_at asc
            """,
            params,
        ).fetchall()
    recommendations: list[Recommendation] = []
    for row in rows:
        recommendation = _recommendation_from_row(row)
        if recommendation is not None:
            recommendations.append(recommendation)
    return recommendations


def _recommendation_from_row(row: sqlite3.Row) -> Recommendation | None:
    created_at = _parse_dt(row["created_at"])
    if created_at is None:
        return None
    action_raw = str(row["action"]).lower()
    if action_raw not in {"buy", "sell", "hold"}:
        return None
    payload = _json_dict(row["payload_json"])
    metadata = cast_dict(payload.get("metadata"))
    conviction = (
        _float_or_none(metadata.get("calibrated_conviction"))
        or _float_or_none(metadata.get("heuristic_conviction"))
        or _float_or_none(payload.get("conviction"))
    )
    return Recommendation(
        scorecard_id=str(row["scorecard_id"]),
        actor=str(row["actor"]),
        symbol=str(row["symbol"]),
        action=cast(Action, action_raw),
        source=str(row["source"]),
        created_at=created_at,
        conviction=conviction,
        factors=tuple(_factors(payload)),
        metadata=metadata,
    )


def _evaluate_recommendation(
    recommendation: Recommendation,
    horizon_hours: int,
    costs: CostModel,
    price_fetcher: PriceFetcher,
    skipped: list[dict[str, str]],
) -> ForwardSample | None:
    try:
        prices = price_fetcher(recommendation.symbol, recommendation.created_at, horizon_hours)
    except Exception as exc:  # noqa: BLE001
        skipped.append({"scorecard_id": recommendation.scorecard_id, "reason": str(exc)})
        return None
    if prices.entry_price <= 0 or prices.exit_price <= 0:
        skipped.append(
            {"scorecard_id": recommendation.scorecard_id, "reason": "non-positive price"}
        )
        return None
    gross_directional = _directional_return(
        recommendation.action, prices.entry_price, prices.exit_price
    )
    total_cost = 0.0 if recommendation.action == "hold" else costs.round_trip_cost
    total_cost += costs.funding_cost(recommendation.action, horizon_hours)
    strategy_return = gross_directional - total_cost
    buy_hold_return = prices.exit_price / prices.entry_price - 1.0 - costs.round_trip_cost
    cash_return = costs.cash_return(horizon_hours)
    return ForwardSample(
        scorecard_id=recommendation.scorecard_id,
        actor=recommendation.actor,
        symbol=recommendation.symbol,
        action=recommendation.action,
        created_at=recommendation.created_at,
        entry_time=prices.entry_time,
        exit_time=prices.exit_time,
        conviction=recommendation.conviction,
        confidence_bucket=_confidence_bucket(recommendation.conviction),
        analyst_names=tuple(
            sorted(
                {
                    str(factor["name"])
                    for factor in recommendation.factors
                    if isinstance(factor.get("name"), str)
                }
            )
        ),
        strategy_return=strategy_return,
        buy_hold_return=buy_hold_return,
        cash_return=cash_return,
        excess_vs_buy_hold=strategy_return - buy_hold_return,
        excess_vs_cash=strategy_return - cash_return,
        gross_directional_return=gross_directional,
        total_cost=total_cost,
    )


def _directional_return(action: str, entry: float, exit_: float) -> float:
    if action == "buy":
        return exit_ / entry - 1.0
    if action == "sell":
        return entry / exit_ - 1.0
    return 0.0


def _ccxt_forward_price_fetcher(source: str) -> PriceFetcher:
    cache: dict[tuple[str, str, int], ForwardPrices] = {}

    def fetch(symbol: str, created_at: datetime, horizon_hours: int) -> ForwardPrices:
        key = (symbol, created_at.isoformat(), horizon_hours)
        if key in cache:
            return cache[key]
        ccxt = importlib.import_module("ccxt")
        exchange = getattr(ccxt, source)({"enableRateLimit": True, "timeout": 10_000})
        market_symbol = _ccxt_symbol(symbol)
        start_ms = int((created_at - timedelta(hours=1)).timestamp() * 1000)
        limit = max(horizon_hours + 8, 30)
        rows = exchange.fetch_ohlcv(market_symbol, "1h", since=start_ms, limit=limit)
        prices = _forward_prices_from_ohlcv(rows, created_at, horizon_hours, market_symbol)
        cache[key] = prices
        return prices

    return fetch


def _forward_prices_from_ohlcv(
    rows: object,
    created_at: datetime,
    horizon_hours: int,
    symbol: str,
) -> ForwardPrices:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"no OHLCV returned for {symbol}")
    parsed: list[tuple[datetime, float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        timestamp = _int_or_none(row[0])
        close = _float_or_none(row[4])
        if timestamp is None or close is None:
            continue
        parsed.append((datetime.fromtimestamp(timestamp / 1000, tz=UTC), close))
    parsed.sort(key=lambda item: item[0])
    entry = next((item for item in parsed if item[0] > created_at), None)
    if entry is None:
        raise RuntimeError(f"no t+1 entry bar after {created_at.isoformat()} for {symbol}")
    target_exit = entry[0] + timedelta(hours=horizon_hours)
    exit_row = next((item for item in parsed if item[0] >= target_exit), None)
    if exit_row is None:
        raise RuntimeError(f"no exit bar after {target_exit.isoformat()} for {symbol}")
    return ForwardPrices(
        entry_time=entry[0],
        exit_time=exit_row[0],
        entry_price=entry[1],
        exit_price=exit_row[1],
    )


def _group_stats(samples: list[ForwardSample], *, min_n: int) -> dict[str, Any]:
    strategy = [sample.strategy_return for sample in samples]
    buy_hold = [sample.buy_hold_return for sample in samples]
    cash = [sample.cash_return for sample in samples]
    excess_bh = [sample.excess_vs_buy_hold for sample in samples]
    excess_cash = [sample.excess_vs_cash for sample in samples]
    enough = len(samples) >= min_n
    p_bh = _positive_sign_test_p_value(excess_bh)
    p_cash = _positive_sign_test_p_value(excess_cash)
    ci_bh = _bootstrap_mean_ci(excess_bh)
    ci_cash = _bootstrap_mean_ci(excess_cash)
    return {
        "n": len(samples),
        "min_n": min_n,
        "status": "TESTED" if enough else "INSUFFICIENT",
        "strategy": _return_stats(strategy),
        "buy_hold": _return_stats(buy_hold),
        "cash": _return_stats(cash),
        "excess_vs_buy_hold": _return_stats(excess_bh),
        "excess_vs_cash": _return_stats(excess_cash),
        "sign_test": {
            "method": "one-sided positive sign test",
            "p_value_vs_buy_hold": p_bh,
            "p_value_vs_cash": p_cash,
        },
        "bootstrap": {
            "method": "iid bootstrap mean CI",
            "mean_excess_vs_buy_hold_ci": ci_bh,
            "mean_excess_vs_cash_ci": ci_cash,
        },
        "risk_adjusted": {
            "strategy_sharpe": _sharpe(strategy),
            "buy_hold_sharpe": _sharpe(buy_hold),
            "cash_sharpe": _sharpe(cash),
        },
    }


def _bucket_stats(
    samples: list[ForwardSample], *, min_n: int, fdr_alpha: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[ForwardSample]] = {}
    for sample in samples:
        groups.setdefault(("symbol", sample.symbol), []).append(sample)
        groups.setdefault(("action", sample.action), []).append(sample)
        groups.setdefault(("confidence", sample.confidence_bucket), []).append(sample)
    return _stats_with_fdr(groups, min_n=min_n, fdr_alpha=fdr_alpha)


def _analyst_stats(
    samples: list[ForwardSample], *, min_n: int, fdr_alpha: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[ForwardSample]] = {}
    for sample in samples:
        for analyst in sample.analyst_names:
            groups.setdefault(("analyst", analyst), []).append(sample)
    return _stats_with_fdr(groups, min_n=min_n, fdr_alpha=fdr_alpha)


def _stats_with_fdr(
    groups: dict[tuple[str, str], list[ForwardSample]], *, min_n: int, fdr_alpha: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    tested_indices: list[int] = []
    for (kind, name), group_samples in sorted(groups.items()):
        stats = _group_stats(group_samples, min_n=min_n)
        row = {
            "kind": kind,
            "name": name,
            "n": len(group_samples),
            "min_n": min_n,
            "status": stats["status"],
            "median_excess_vs_buy_hold": stats["excess_vs_buy_hold"]["median"],
            "median_excess_vs_cash": stats["excess_vs_cash"]["median"],
            "win_rate_vs_buy_hold": stats["excess_vs_buy_hold"]["win_rate"],
            "win_rate_vs_cash": stats["excess_vs_cash"]["win_rate"],
            "p_value_vs_buy_hold": stats["sign_test"]["p_value_vs_buy_hold"],
            "p_value_vs_cash": stats["sign_test"]["p_value_vs_cash"],
            "bh_fdr_discovery": False,
        }
        if stats["status"] == "TESTED":
            tested_indices.append(len(rows))
            p_values.append(max(float(row["p_value_vs_buy_hold"]), float(row["p_value_vs_cash"])))
        rows.append(row)
    discoveries = _benjamini_hochberg(p_values, alpha=fdr_alpha)
    for passed, row_index in zip(discoveries, tested_indices, strict=True):
        rows[row_index]["bh_fdr_discovery"] = passed
    return rows


def _verdict(overall: dict[str, Any], min_n: int) -> tuple[AgentVerdict, str]:
    if int(overall["n"]) < min_n:
        return "INSUFFICIENT", "recommendation sample is below the predeclared minimum N"
    median_bh = float(overall["excess_vs_buy_hold"]["median"] or 0.0)
    median_cash = float(overall["excess_vs_cash"]["median"] or 0.0)
    p_bh = float(overall["sign_test"]["p_value_vs_buy_hold"])
    p_cash = float(overall["sign_test"]["p_value_vs_cash"])
    ci_bh = cast(dict[str, float | None], overall["bootstrap"]["mean_excess_vs_buy_hold_ci"])
    ci_cash = cast(dict[str, float | None], overall["bootstrap"]["mean_excess_vs_cash_ci"])
    strategy_sharpe = float(overall["risk_adjusted"]["strategy_sharpe"])
    buy_hold_sharpe = float(overall["risk_adjusted"]["buy_hold_sharpe"])
    if (
        median_bh > 0.0
        and median_cash > 0.0
        and p_bh <= 0.05
        and p_cash <= 0.05
        and (ci_bh["p05"] or 0.0) > 0.0
        and (ci_cash["p05"] or 0.0) > 0.0
        and strategy_sharpe > buy_hold_sharpe
    ):
        return "AGENT_EDGE", "forward recommendations beat buy&hold and cash after costs"
    return "NO_EDGE", "recommendations did not pass the full-cost benchmark and significance gates"


def _return_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": _round(statistics.fmean(values)) if values else None,
        "median": _round(statistics.median(values)) if values else None,
        "total": _round(sum(values)) if values else None,
        "win_rate": _share([value > 0.0 for value in values]),
        "stdev": _round(statistics.stdev(values)) if len(values) >= 2 else None,
    }


def _positive_sign_test_p_value(values: list[float]) -> float:
    non_zero = [value for value in values if value != 0.0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0.0)
    return min(1.0, float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)))


def _bootstrap_mean_ci(
    values: list[float], *, iterations: int = 1000, seed: int = 51
) -> dict[str, float | None]:
    if not values:
        return {"p05": None, "p50": None, "p95": None}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(iterations))
    return {
        "p05": _round(means[int(iterations * 0.05)]),
        "p50": _round(means[int(iterations * 0.50)]),
        "p95": _round(means[int(iterations * 0.95)]),
    }


def _benjamini_hochberg(p_values: list[float], *, alpha: float) -> list[bool]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    passed = [False for _ in p_values]
    max_rank = -1
    tests = len(indexed)
    for rank, (_index, p_value) in enumerate(indexed, start=1):
        if p_value <= alpha * rank / tests:
            max_rank = rank
    if max_rank >= 1:
        for rank, (index, _p_value) in enumerate(indexed, start=1):
            if rank <= max_rank:
                passed[index] = True
    return passed


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return 0.0
    return _round(statistics.fmean(values) / stdev * math.sqrt(len(values))) or 0.0


def _sanitized_summary(
    verdict: AgentVerdict,
    overall: dict[str, Any],
    buckets: list[dict[str, Any]],
    analysts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "overall_n": overall["n"],
        "overall_status": overall["status"],
        "bucket_groups": len(buckets),
        "analyst_groups": len(analysts),
        "contains_actor_or_recommendation_rows": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    overall = cast(dict[str, Any], report["overall"])
    lines = [
        "# TradingAgents Edge Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        f"Reason: {report['reason']}",
        "",
        "## Discipline",
        "",
        "- Inputs are recorded scorecards only; no LLM calls and no order path.",
        "- Entry is the first 1h bar strictly after recommendation creation.",
        "- Benchmarks are same-symbol buy&hold and cash.",
        "- Costs include fee, slippage, and short funding when configured.",
        "- Bucket and analyst tests are BH-FDR adjusted.",
        "",
        "## Overall",
        "",
        f"- N: `{overall['n']}` / min `{overall['min_n']}`",
        f"- Status: `{overall['status']}`",
        f"- Median excess vs buy&hold: `{overall['excess_vs_buy_hold']['median']}`",
        f"- Median excess vs cash: `{overall['excess_vs_cash']['median']}`",
        f"- p vs buy&hold: `{overall['sign_test']['p_value_vs_buy_hold']}`",
        f"- p vs cash: `{overall['sign_test']['p_value_vs_cash']}`",
        "",
        "## Limits",
        "",
        "- Full recommendation rows and actors are private-only.",
        "- Public summary is sanitized and excludes actor/recommendation rows.",
        "- Evaluation quality depends on historical OHLCV coverage after each recommendation.",
    ]
    return "\n".join(lines) + "\n"


def _sample_to_private_row(sample: ForwardSample) -> dict[str, Any]:
    return {
        "scorecard_id": sample.scorecard_id,
        "actor": sample.actor,
        "symbol": sample.symbol,
        "action": sample.action,
        "created_at": sample.created_at.isoformat(),
        "entry_time": sample.entry_time.isoformat(),
        "exit_time": sample.exit_time.isoformat(),
        "conviction": sample.conviction,
        "confidence_bucket": sample.confidence_bucket,
        "analyst_names": list(sample.analyst_names),
        "strategy_return": _round(sample.strategy_return),
        "buy_hold_return": _round(sample.buy_hold_return),
        "cash_return": _round(sample.cash_return),
        "excess_vs_buy_hold": _round(sample.excess_vs_buy_hold),
        "excess_vs_cash": _round(sample.excess_vs_cash),
        "gross_directional_return": _round(sample.gross_directional_return),
        "total_cost": _round(sample.total_cost),
    }


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_dict(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def cast_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _factors(payload: dict[str, object]) -> list[dict[str, object]]:
    raw = payload.get("factors")
    if not isinstance(raw, list):
        return []
    factors: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        direction = item.get("direction")
        if isinstance(name, str):
            factors.append({"name": name, "direction": direction, "score": item.get("score")})
    return factors


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.5:
        return "0.00-0.50"
    if value < 0.65:
        return "0.50-0.65"
    if value < 0.8:
        return "0.65-0.80"
    return "0.80-1.01"


def _ccxt_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" in normalized:
        return normalized
    for quote in ("USDT", "USDC", "USD"):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return f"{normalized[:-len(quote)]}/{quote}"
    return normalized


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _share(values: list[bool]) -> float | None:
    if not values:
        return None
    return _round(sum(1 for value in values if value) / len(values))


def _round(value: float | None) -> float | None:
    return round(float(value), 8) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
