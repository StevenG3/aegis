from __future__ import annotations

import argparse
import importlib
import json
import os
import sqlite3
import statistics
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = REPO_ROOT / "services" / "orchestrator"
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from db import connect  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "OLYMPUS_EVIDENCE_DIR",
        str(Path(__file__).resolve().parents[2] / "aegis-strategies" / "incubating"),
    )
)
DEFAULT_SOURCE = "tradingagents"
DEFAULT_BENCHMARK = "BTC/USDT"
DEFAULT_BENCHMARK_SOURCE = "binance"
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_FUNDING_BPS = 0.0
DEFAULT_MIN_N = 30
DEFAULT_BUCKET_MIN_N = 10
DEFAULT_ANALYST_MIN_N = 10


@dataclass(frozen=True)
class CostModel:
    fee_bps: float
    slippage_bps: float
    funding_bps: float

    @property
    def round_trip_cost_pct(self) -> float:
        return ((self.fee_bps + self.slippage_bps) * 2 + self.funding_bps) / 10_000


@dataclass(frozen=True)
class OutcomeSample:
    outcome_id: str
    scorecard_id: str
    actor: str
    symbol: str
    action: str
    opened_at: datetime
    closed_at: datetime
    gross_return_pct: float
    net_return_pct: float
    btc_hold_return_pct: float | None
    alpha_vs_btc_pct: float | None
    conviction: float | None
    heuristic_conviction: float | None
    calibrated_conviction: float | None
    factors: list[dict[str, object]]
    data_origin: str
    gate_conviction: str | None


PriceFetcher = Callable[[datetime, datetime], tuple[float, float]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TradingAgents closed-outcome edge.")
    parser.add_argument("--actor", default=None)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--benchmark-source", default=DEFAULT_BENCHMARK_SOURCE)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--funding-bps", type=float, default=DEFAULT_FUNDING_BPS)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--bucket-min-n", type=int, default=DEFAULT_BUCKET_MIN_N)
    parser.add_argument("--analyst-min-n", type=int, default=DEFAULT_ANALYST_MIN_N)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = build_report(
        actor=args.actor,
        source=args.source,
        benchmark=args.benchmark,
        benchmark_source=args.benchmark_source,
        costs=CostModel(args.fee_bps, args.slippage_bps, args.funding_bps),
        min_n=args.min_n,
        bucket_min_n=args.bucket_min_n,
        analyst_min_n=args.analyst_min_n,
    )
    if not args.no_write:
        report["written_files"] = write_report(report, Path(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def build_report(
    *,
    actor: str | None,
    source: str,
    benchmark: str,
    benchmark_source: str,
    costs: CostModel,
    min_n: int = DEFAULT_MIN_N,
    bucket_min_n: int = DEFAULT_BUCKET_MIN_N,
    analyst_min_n: int = DEFAULT_ANALYST_MIN_N,
    price_fetcher: PriceFetcher | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    raw_rows = _load_closed_rows(actor=actor, source=source)
    benchmark_errors: list[dict[str, str]] = []
    samples: list[OutcomeSample] = []
    fetcher = price_fetcher or _benchmark_price_fetcher(benchmark, benchmark_source)
    for row in raw_rows:
        sample = _sample_from_row(row, costs, fetcher, benchmark_errors)
        if sample is not None:
            samples.append(sample)

    rows = [_sample_to_row(sample) for sample in samples]
    summary = _summary(samples, min_n=min_n)
    calibration = _conviction_calibration(samples, bucket_min_n=bucket_min_n)
    analysts = _analyst_attribution(samples, analyst_min_n=analyst_min_n)
    accumulation = _accumulation_plan(samples, min_n=min_n, bucket_min_n=bucket_min_n)
    recommendation = _recommendation(summary, calibration, analysts)
    payload = {
        "generated_at": generated_at.isoformat(),
        "scope": {
            "source": source,
            "actor": actor,
            "benchmark": benchmark,
            "benchmark_source": benchmark_source,
        },
        "cost_model": {
            "fee_bps": costs.fee_bps,
            "slippage_bps": costs.slippage_bps,
            "funding_bps": costs.funding_bps,
            "round_trip_cost_pct": _round(costs.round_trip_cost_pct),
            "net_return_formula": "closed_return_pct - round_trip_cost_pct",
            "alpha_formula": "net_return_pct - btc_hold_return_pct",
        },
        "sample_sufficiency": {
            "closed_outcomes_n": len(samples),
            "min_n_for_edge_claim": min_n,
            "bucket_min_n_for_calibration": bucket_min_n,
            "analyst_min_n": analyst_min_n,
            "verdict": "INSUFFICIENT_DATA" if len(samples) < min_n else "ENOUGH_FOR_FIRST_PASS",
            "honesty_rule": "do not claim TradingAgents edge below min_n",
        },
        "summary": summary,
        "conviction_calibration": calibration,
        "analyst_attribution": analysts,
        "accumulation_plan": accumulation,
        "recommendation": recommendation,
        "rows": rows,
        "benchmark_errors": benchmark_errors,
        "disclaimer": (
            "read-only paper evaluation; no TradingAgents, risk, execution, "
            "or analyst gating changes"
        ),
    }
    payload["human_readable"] = _markdown(payload)
    return payload


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


def _load_closed_rows(*, actor: str | None, source: str) -> list[sqlite3.Row]:
    clauses = ["o.status = 'closed'", "o.source = ?"]
    params: list[object] = [source]
    if actor:
        clauses.append("o.actor = ?")
        params.append(actor)
    where = " where " + " and ".join(clauses)
    with connect() as conn:
        return conn.execute(
            f"""
            select o.outcome_id, o.scorecard_id, o.actor, o.symbol, o.source, o.action,
                   o.opened_at, o.opened_avg_cost, o.opened_cost_basis,
                   o.closed_at, o.closed_realized_pnl, o.closed_return_pct,
                   s.payload_json
            from scorecard_outcomes o
            join scorecards s on s.scorecard_id = o.scorecard_id
            {where}
            order by o.closed_at asc
            """,
            params,
        ).fetchall()


def _sample_from_row(
    row: sqlite3.Row,
    costs: CostModel,
    price_fetcher: PriceFetcher,
    benchmark_errors: list[dict[str, str]],
) -> OutcomeSample | None:
    opened_at = _parse_dt(row["opened_at"])
    closed_at = _parse_dt(row["closed_at"])
    gross_return = _float_or_none(row["closed_return_pct"])
    if opened_at is None or closed_at is None or gross_return is None:
        return None
    payload = _json_dict(row["payload_json"])
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    net_return = gross_return - costs.round_trip_cost_pct
    btc_hold: float | None = None
    alpha: float | None = None
    try:
        btc_open, btc_close = price_fetcher(opened_at, closed_at)
        if btc_open > 0:
            btc_hold = btc_close / btc_open - 1
            alpha = net_return - btc_hold
    except Exception as exc:  # noqa: BLE001
        benchmark_errors.append({"outcome_id": str(row["outcome_id"]), "error": str(exc)})
    return OutcomeSample(
        outcome_id=str(row["outcome_id"]),
        scorecard_id=str(row["scorecard_id"]),
        actor=str(row["actor"]),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        opened_at=opened_at,
        closed_at=closed_at,
        gross_return_pct=gross_return,
        net_return_pct=net_return,
        btc_hold_return_pct=btc_hold,
        alpha_vs_btc_pct=alpha,
        conviction=_float_or_none(payload.get("conviction")),
        heuristic_conviction=_float_or_none(cast_dict(metadata).get("heuristic_conviction")),
        calibrated_conviction=_float_or_none(cast_dict(metadata).get("calibrated_conviction")),
        factors=_factors(payload),
        data_origin=str(
            cast_dict(metadata).get("data_origin")
            or cast_dict(metadata).get("origin")
            or "unknown"
        ),
        gate_conviction=_str_or_none(cast_dict(metadata).get("gate_conviction")),
    )


def _benchmark_price_fetcher(benchmark: str, source: str) -> PriceFetcher:
    cache: dict[tuple[str, str], tuple[float, float]] = {}

    def fetch(opened_at: datetime, closed_at: datetime) -> tuple[float, float]:
        start = opened_at.replace(minute=0, second=0, microsecond=0)
        end = closed_at.replace(minute=0, second=0, microsecond=0)
        key = (start.isoformat(), end.isoformat())
        if key in cache:
            return cache[key]
        ccxt = importlib.import_module("ccxt")
        exchange = getattr(ccxt, source)({"enableRateLimit": True, "timeout": 10_000})
        symbol = _ccxt_symbol(benchmark)
        open_price = _ohlcv_close_at_or_before(exchange, symbol, start)
        close_price = _ohlcv_close_at_or_before(exchange, symbol, end)
        cache[key] = (open_price, close_price)
        return cache[key]

    return fetch


def _ohlcv_close_at_or_before(exchange: object, symbol: str, target: datetime) -> float:
    since = int((target.timestamp() - 3 * 86_400) * 1000)
    rows = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=100)  # type: ignore[attr-defined]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"no benchmark OHLCV returned for {symbol}")
    target_ms = int(target.timestamp() * 1000)
    eligible = [
        row
        for row in rows
        if isinstance(row, list) and len(row) >= 5 and int(row[0]) <= target_ms
    ]
    if not eligible:
        raise RuntimeError(f"no benchmark bar at or before {target.isoformat()}")
    return float(eligible[-1][4])


def _ccxt_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" in normalized:
        return normalized
    if normalized.endswith("USDT"):
        return f"{normalized[:-4]}/USDT"
    return normalized


def _summary(samples: list[OutcomeSample], *, min_n: int) -> dict[str, Any]:
    net_returns = [sample.net_return_pct for sample in samples]
    alpha_returns = [
        sample.alpha_vs_btc_pct for sample in samples if sample.alpha_vs_btc_pct is not None
    ]
    return {
        "n": len(samples),
        "benchmark_available_n": len(alpha_returns),
        "total_gross_return_pct": _round(sum(sample.gross_return_pct for sample in samples)),
        "total_net_return_pct": _round(sum(net_returns)),
        "total_alpha_vs_btc_pct": _round(sum(alpha_returns)) if alpha_returns else None,
        "avg_net_return_pct": _mean(net_returns),
        "median_net_return_pct": _median(net_returns),
        "net_win_rate": _share(value > 0 for value in net_returns),
        "avg_alpha_vs_btc_pct": _mean(alpha_returns),
        "median_alpha_vs_btc_pct": _median(alpha_returns),
        "alpha_win_rate": _share(value > 0 for value in alpha_returns),
        "edge_claim": (
            "NO_CLAIM_INSUFFICIENT_DATA" if len(samples) < min_n else "FIRST_PASS_ALLOWED"
        ),
        "preliminary_signal": _preliminary_signal(len(samples), alpha_returns),
    }


def _conviction_calibration(
    samples: list[OutcomeSample], *, bucket_min_n: int
) -> dict[str, Any]:
    buckets: dict[str, list[OutcomeSample]] = {}
    for sample in samples:
        conviction = (
            sample.calibrated_conviction or sample.heuristic_conviction or sample.conviction
        )
        bucket = _conviction_bucket(conviction)
        buckets.setdefault(bucket, []).append(sample)
    items = []
    for bucket in ("unknown", "0.00-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.01"):
        rows = buckets.get(bucket, [])
        alpha = [sample.alpha_vs_btc_pct for sample in rows if sample.alpha_vs_btc_pct is not None]
        net = [sample.net_return_pct for sample in rows]
        items.append(
            {
                "bucket": bucket,
                "n": len(rows),
                "alpha_available_n": len(alpha),
                "net_win_rate": _share(value > 0 for value in net),
                "alpha_win_rate": _share(value > 0 for value in alpha),
                "avg_alpha_vs_btc_pct": _mean(alpha),
                "median_alpha_vs_btc_pct": _median(alpha),
                "reliable": len(rows) >= bucket_min_n,
            }
        )
    non_empty = [item for item in items if item["bucket"] != "unknown" and int(item["n"]) > 0]
    alpha_rates = [
        item["alpha_win_rate"] for item in non_empty if item["alpha_win_rate"] is not None
    ]
    monotonic = _non_decreasing(alpha_rates) if len(alpha_rates) >= 2 else None
    return {
        "bucket_min_n": bucket_min_n,
        "items": items,
        "monotonic_alpha_win_rate": monotonic,
        "verdict": _calibration_verdict(non_empty, monotonic, bucket_min_n),
    }


def _analyst_attribution(
    samples: list[OutcomeSample], *, analyst_min_n: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[OutcomeSample]] = {}
    for sample in samples:
        for factor in sample.factors:
            name = str(factor.get("name") or "")
            direction = str(factor.get("direction") or "")
            if not name or direction not in {"support", "oppose", "neutral"}:
                continue
            grouped.setdefault((name, direction), []).append(sample)
    items = []
    for (name, direction), rows in sorted(grouped.items()):
        alpha = [sample.alpha_vs_btc_pct for sample in rows if sample.alpha_vs_btc_pct is not None]
        correct = [_factor_correct(direction, value) for value in alpha]
        hit_rate = _share(value for value in correct if value is not None)
        items.append(
            {
                "analyst": name,
                "direction": direction,
                "n": len(rows),
                "alpha_available_n": len(alpha),
                "directional_hit_rate_vs_btc": hit_rate,
                "avg_alpha_vs_btc_pct": _mean(alpha),
                "median_alpha_vs_btc_pct": _median(alpha),
                "reliable": len(rows) >= analyst_min_n,
                "preliminary_label": _analyst_label(len(rows), hit_rate, analyst_min_n),
            }
        )
    return {
        "analyst_min_n": analyst_min_n,
        "items": items,
        "noise_candidates": [
            item for item in items if item["preliminary_label"] in {"possible_noise", "wrong_way"}
        ],
        "guardrail": "human gate only; do not auto-remove analysts from TradingAgents",
    }


def _accumulation_plan(
    samples: list[OutcomeSample], *, min_n: int, bucket_min_n: int
) -> dict[str, Any]:
    n = len(samples)
    closed_dates = sorted(sample.closed_at for sample in samples)
    rate_per_day: float | None = None
    days_to_min: float | None = None
    if len(closed_dates) >= 2:
        span_days = max((closed_dates[-1] - closed_dates[0]).total_seconds() / 86_400, 1 / 24)
        rate_per_day = len(closed_dates) / span_days
        if rate_per_day > 0 and n < min_n:
            days_to_min = (min_n - n) / rate_per_day
    return {
        "current_closed_outcomes": n,
        "min_n_for_first_edge_claim": min_n,
        "remaining_to_min_n": max(min_n - n, 0),
        "bucket_min_n": bucket_min_n,
        "observed_closed_outcomes_per_day": (
            _round(rate_per_day) if rate_per_day is not None else None
        ),
        "estimated_days_to_min_n": _round(days_to_min) if days_to_min is not None else None,
        "primary_path": (
            "continue paper feedback bootstrap; do not lower safety gates just to create samples"
        ),
        "optional_shadow_replay": (
            "small bounded historical replay only; each replay may require one "
            "TradingAgents/LLM call, "
            "so do not run large batches without explicit cost approval"
        ),
    }


def _recommendation(
    summary: dict[str, Any],
    calibration: dict[str, Any],
    analysts: dict[str, Any],
) -> dict[str, str]:
    if summary["edge_claim"] == "NO_CLAIM_INSUFFICIENT_DATA":
        status = "insufficient_data_continue_bootstrap"
        reason = "closed-outcome sample is below the minimum required for an edge claim"
    elif (
        summary.get("median_alpha_vs_btc_pct") is not None
        and summary["median_alpha_vs_btc_pct"] > 0
    ):
        status = "first_pass_positive_but_human_review_required"
        reason = "median alpha is positive after costs, but promotion still requires human review"
    else:
        status = "no_edge_detected"
        reason = "net alpha versus BTC hold is not positive after costs"
    if calibration["verdict"] == "NOT_MONOTONIC":
        reason += "; conviction is not monotonically calibrated"
    if analysts["noise_candidates"]:
        reason += "; some analyst signals look like possible noise but require human gate"
    return {
        "status": status,
        "reason": reason,
        "decision_policy": "report only; no TradingAgents, risk, execution, or analyst changes",
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    suff = report["sample_sufficiency"]
    rec = report["recommendation"]
    lines = [
        "# TradingAgents Edge Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Source: `{report['scope']['source']}`",
        f"Benchmark: `{report['scope']['benchmark']}` via `{report['scope']['benchmark_source']}`",
        "",
        "## Recommendation",
        "",
        f"- Status: `{rec['status']}`",
        f"- Reason: {rec['reason']}",
        f"- Policy: {rec['decision_policy']}",
        "",
        "## Sample Sufficiency",
        "",
        f"- Closed outcomes: `{suff['closed_outcomes_n']}`",
        f"- Minimum for edge claim: `{suff['min_n_for_edge_claim']}`",
        f"- Verdict: `{suff['verdict']}`",
        "",
        "## Net Cost vs BTC Hold",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "n",
        "benchmark_available_n",
        "total_gross_return_pct",
        "total_net_return_pct",
        "total_alpha_vs_btc_pct",
        "median_net_return_pct",
        "median_alpha_vs_btc_pct",
        "net_win_rate",
        "alpha_win_rate",
    ):
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(["", "## Conviction Calibration", ""])
    lines.extend(
        [
            "| Bucket | N | Alpha N | Net win | Alpha win | Median alpha | Reliable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["conviction_calibration"]["items"]:
        lines.append(
            "| {bucket} | {n} | {alpha_available_n} | {net_win_rate} | {alpha_win_rate} | "
            "{median_alpha_vs_btc_pct} | {reliable} |".format(**item)
        )
    lines.extend(["", "## Analyst Attribution", ""])
    lines.extend(
        [
            "| Analyst | Direction | N | Alpha N | Hit vs BTC | Median alpha | Label | Reliable |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in report["analyst_attribution"]["items"]:
        lines.append(
            "| {analyst} | {direction} | {n} | {alpha_available_n} | "
            "{directional_hit_rate_vs_btc} | {median_alpha_vs_btc_pct} | "
            "{preliminary_label} | {reliable} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Accumulation Plan",
            "",
        ]
    )
    for key, value in report["accumulation_plan"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Cost Assumptions",
            "",
            f"- Fee bps: `{report['cost_model']['fee_bps']}`",
            f"- Slippage bps: `{report['cost_model']['slippage_bps']}`",
            f"- Funding bps: `{report['cost_model']['funding_bps']}`",
            f"- Round-trip cost pct: `{report['cost_model']['round_trip_cost_pct']}`",
            "",
            "## Limits",
            "",
            "- Read-only paper evaluation; no decision path changed.",
            "- BTC benchmark depends on public OHLCV availability.",
            "- Small n must be treated as directional only, not statistical proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sample_to_row(sample: OutcomeSample) -> dict[str, Any]:
    return {
        "outcome_id": sample.outcome_id,
        "scorecard_id": sample.scorecard_id,
        "actor": sample.actor,
        "symbol": sample.symbol,
        "action": sample.action,
        "opened_at": sample.opened_at.isoformat(),
        "closed_at": sample.closed_at.isoformat(),
        "gross_return_pct": _round(sample.gross_return_pct),
        "net_return_pct": _round(sample.net_return_pct),
        "btc_hold_return_pct": _round(sample.btc_hold_return_pct),
        "alpha_vs_btc_pct": _round(sample.alpha_vs_btc_pct),
        "conviction": sample.conviction,
        "heuristic_conviction": sample.heuristic_conviction,
        "calibrated_conviction": sample.calibrated_conviction,
        "factors": sample.factors,
        "data_origin": sample.data_origin,
        "gate_conviction": sample.gate_conviction,
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
        if isinstance(name, str) and direction in {"support", "oppose", "neutral"}:
            factors.append({"name": name, "direction": direction, "score": item.get("score")})
    return factors


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _conviction_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.5:
        return "0.00-0.50"
    if value < 0.65:
        return "0.50-0.65"
    if value < 0.8:
        return "0.65-0.80"
    return "0.80-1.01"


def _factor_correct(direction: str, alpha: float) -> bool | None:
    if direction == "support":
        return alpha > 0
    if direction == "oppose":
        return alpha <= 0
    return None


def _analyst_label(n: int, hit_rate: float | None, analyst_min_n: int) -> str:
    if n < analyst_min_n:
        return "insufficient_data"
    if hit_rate is None:
        return "not_directional"
    if hit_rate >= 0.6:
        return "possible_signal"
    if hit_rate <= 0.4:
        return "wrong_way"
    return "possible_noise"


def _calibration_verdict(
    items: list[dict[str, Any]], monotonic: bool | None, bucket_min_n: int
) -> str:
    if sum(int(item["n"]) for item in items) == 0:
        return "NO_DATA"
    if any(int(item["n"]) < bucket_min_n for item in items):
        return "INSUFFICIENT_BUCKET_DATA"
    if monotonic is False:
        return "NOT_MONOTONIC"
    if monotonic is True:
        return "MONOTONIC"
    return "INSUFFICIENT_BUCKET_SPREAD"


def _preliminary_signal(n: int, alpha_returns: list[float]) -> str:
    if n == 0:
        return "no_closed_outcomes"
    if not alpha_returns:
        return "benchmark_unavailable"
    median = statistics.median(alpha_returns)
    if median > 0:
        return "positive_but_unreliable" if n < DEFAULT_MIN_N else "positive_first_pass"
    if median < 0:
        return "negative_but_unreliable" if n < DEFAULT_MIN_N else "negative_first_pass"
    return "flat_or_inconclusive"


def _non_decreasing(values: list[float | None]) -> bool | None:
    concrete = [value for value in values if value is not None]
    if len(concrete) < 2:
        return None
    return all(right >= left for left, right in zip(concrete, concrete[1:], strict=False))


def _mean(values: list[float]) -> float | None:
    return _round(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return _round(statistics.median(values)) if values else None


def _share(values: Iterable[bool]) -> float | None:
    concrete = list(values)
    if not concrete:
        return None
    return _round(sum(1 for value in concrete if value) / len(concrete))


def _round(value: float | None) -> float | None:
    return round(float(value), 8) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
