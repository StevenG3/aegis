from __future__ import annotations

import json
import os
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "backtest-service"
sys.path.insert(0, str(SERVICE_DIR))

from data import DataLoadError  # noqa: E402
from funding_arb import (  # noqa: E402
    FundingArbConfig,
    FundingSource,
    _align_events,
    _cost_model,
    _load_funding_history,
    _load_ohlcv,
    _simulate,
    _symbols,
)

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
DEFAULT_SOURCE: FundingSource = "bybit"
DEFAULT_PERIODS = (
    ("2023_h2_range", "2023-07-01", "2023-10-01", "range / post-bear recovery"),
    ("2024_q1_bull", "2024-01-01", "2024-04-01", "bull / high beta"),
    ("2024_q2_cooldown", "2024-04-01", "2024-07-01", "post-rally cooldown"),
    ("2024_q3_range", "2024-07-01", "2024-10-01", "range / lower momentum"),
    ("2025_q1_mixed", "2025-01-01", "2025-04-01", "mixed regime sample"),
    ("2025_q2_mixed", "2025-04-01", "2025-07-01", "mixed regime sample"),
)
DEFAULT_TAKER_FEE_BPS = (5.0, 10.0, 15.0)
DEFAULT_SLIPPAGE_BPS = (1.0, 2.0, 5.0)
DEFAULT_MIN_FUNDING_BPS = (1.0, 3.0, 5.0, 8.0)
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "OLYMPUS_EVIDENCE_DIR",
        str(Path(__file__).resolve().parents[2] / "aegis-strategies" / "incubating"),
    )
)


@dataclass(frozen=True)
class EvidencePeriod:
    name: str
    start: str
    end: str
    note: str


@dataclass(frozen=True)
class LoadedCase:
    symbol: str
    period: EvidencePeriod
    aligned_events: list[dict[str, Any]]
    symbols: Any


def main() -> int:
    generated_at = datetime.now(UTC)
    source = cast(FundingSource, os.getenv("FUNDING_ARB_EVIDENCE_SOURCE", DEFAULT_SOURCE))
    symbols = _env_csv("FUNDING_ARB_EVIDENCE_SYMBOLS", DEFAULT_SYMBOLS)
    periods = _periods_from_env()
    output_dir = Path(os.getenv("FUNDING_ARB_EVIDENCE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    taker_fees = _env_float_csv("FUNDING_ARB_EVIDENCE_TAKER_FEE_BPS", DEFAULT_TAKER_FEE_BPS)
    slippages = _env_float_csv("FUNDING_ARB_EVIDENCE_SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS)
    min_fundings = _env_float_csv("FUNDING_ARB_EVIDENCE_MIN_FUNDING_BPS", DEFAULT_MIN_FUNDING_BPS)
    cash = _env_float("FUNDING_ARB_EVIDENCE_CASH", 10_000.0)
    max_events = _env_int("FUNDING_ARB_EVIDENCE_MAX_EVENTS", 400)

    baseline: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for symbol in symbols:
        for period in periods:
            try:
                loaded = _load_case(
                    source=source,
                    symbol=symbol,
                    period=period,
                    cash=cash,
                    max_events=max_events,
                )
                base_row = _run_loaded_case(
                    loaded,
                    source=source,
                    cash=cash,
                    taker_fee_bps=10.0,
                    slippage_bps=2.0,
                    min_funding_bps=3.0,
                    max_events=max_events,
                )
                baseline.append(base_row)
                for taker_fee_bps in taker_fees:
                    for slippage_bps in slippages:
                        for min_funding_bps in min_fundings:
                            row = _run_loaded_case(
                                loaded,
                                source=source,
                                cash=cash,
                                taker_fee_bps=taker_fee_bps,
                                slippage_bps=slippage_bps,
                                min_funding_bps=min_funding_bps,
                                max_events=max_events,
                            )
                            sensitivity.append(row)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "symbol": symbol,
                        "period": period.name,
                        "start": period.start,
                        "end": period.end,
                        "error": str(exc),
                    }
                )

    robustness = _robustness_summary(baseline, sensitivity)
    recommendation = _recommendation(robustness)
    payload = {
        "generated_at": generated_at.isoformat(),
        "source": source,
        "config": {
            "symbols": symbols,
            "periods": [period.__dict__ for period in periods],
            "baseline": {
                "taker_fee_bps": 10.0,
                "slippage_bps": 2.0,
                "min_funding_bps": 3.0,
                "exit_funding_bps": 0.0,
                "basis_cost_bps": 0.0,
                "borrow_cost_bps_annual": 0.0,
                "cash": cash,
                "max_events": max_events,
            },
            "sensitivity": {
                "taker_fee_bps": taker_fees,
                "slippage_bps": slippages,
                "min_funding_bps": min_fundings,
            },
        },
        "baseline": baseline,
        "sensitivity": sensitivity,
        "sensitivity_summary": _sensitivity_summary(sensitivity),
        "robustness": robustness,
        "recommendation": recommendation,
        "failures": failures,
        "disclaimer": (
            "backtest-only evidence; no graduation, paper trading, or live trading action"
        ),
    }

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"funding-arb-evidence-{stamp}.json"
    md_path = output_dir / f"funding-arb-evidence-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), **robustness}, indent=2))
    return 0 if baseline else 1


def _load_case(
    *,
    source: FundingSource,
    symbol: str,
    period: EvidencePeriod,
    cash: float,
    max_events: int,
) -> LoadedCase:
    if cash <= 0:
        raise DataLoadError("cash must be positive")
    start_dt = datetime.fromisoformat(period.start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(period.end).replace(tzinfo=UTC)
    symbols = _symbols(symbol)
    funding_events = _load_funding_history(source, symbols.swap, start_dt, end_dt, max_events)
    spot_frame = _load_ohlcv(source, symbols.spot, "spot", "1h", start_dt, end_dt)
    perp_frame = _load_ohlcv(source, symbols.swap, "swap", "1h", start_dt, end_dt)
    aligned = _align_events(funding_events, spot_frame, perp_frame)
    if not aligned:
        raise DataLoadError("funding history and price bars could not be aligned")
    return LoadedCase(symbol=symbol, period=period, aligned_events=aligned, symbols=symbols)


def _run_loaded_case(
    loaded: LoadedCase,
    *,
    source: FundingSource,
    cash: float,
    taker_fee_bps: float,
    slippage_bps: float,
    min_funding_bps: float,
    max_events: int,
) -> dict[str, Any]:
    config = FundingArbConfig(
        symbol=loaded.symbol,
        source=source,
        start=loaded.period.start,
        end=loaded.period.end,
        cash=cash,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        min_funding_bps=min_funding_bps,
        exit_funding_bps=0.0,
        basis_cost_bps=0.0,
        borrow_cost_bps_annual=0.0,
        max_funding_events=max_events,
    )
    result = _simulate(loaded.aligned_events, config, _cost_model(config), loaded.symbols)
    stats = cast(dict[str, Any], result["stats"])
    data = cast(dict[str, Any], result["data"])
    data_start = str(data["start"])
    data_end = str(data["end"])
    rates = [float(event["funding_rate"]) * 10_000 for event in loaded.aligned_events]
    coverage = _coverage(
        requested_start=loaded.period.start,
        requested_end=loaded.period.end,
        data_start=data_start,
        data_end=data_end,
    )
    return {
        "symbol": loaded.symbol,
        "period": loaded.period.name,
        "start": loaded.period.start,
        "end": loaded.period.end,
        "period_note": loaded.period.note,
        "observed_regime": _observed_regime(rates),
        "taker_fee_bps": taker_fee_bps,
        "slippage_bps": slippage_bps,
        "min_funding_bps": min_funding_bps,
        "net_return_pct": _round(stats["net_return_pct"]),
        "gross_return_pct": _round(stats["gross_return_pct"]),
        "gross_funding_return_pct": _round(stats["gross_funding_return_pct"]),
        "basis_return_pct": _round(stats["basis_return_pct"]),
        "fee_cost_pct": _round(stats["fee_cost_pct"]),
        "slippage_cost_pct": _round(stats["slippage_cost_pct"]),
        "basis_cost_pct": _round(stats["basis_cost_pct"]),
        "borrow_cost_pct": _round(stats["borrow_cost_pct"]),
        "annualized_return_pct": _round(stats["annualized_return_pct"]),
        "sharpe": _round(stats["sharpe"]),
        "max_drawdown_pct": _round(stats["max_drawdown_pct"]),
        "negative_funding_period_share": _round(stats["negative_funding_period_share"]),
        "entries": int(stats["num_trades"]),
        "funding_events": int(stats["funding_events"]),
        "held_funding_events": int(stats["held_funding_events"]),
        "exposure_pct": _round(stats["exposure_pct"]),
        "exit_breakdown": stats["exit_breakdown"],
        "data_start": data_start,
        "data_end": data_end,
        **coverage,
    }


def _observed_regime(funding_bps: list[float]) -> str:
    if not funding_bps:
        return "unknown"
    median = statistics.median(funding_bps)
    negative_share = sum(1 for value in funding_bps if value < 0) / len(funding_bps)
    if negative_share >= 0.35:
        return "negative_funding_heavy"
    if median >= 4.0:
        return "high_positive_funding"
    if median <= 1.0:
        return "low_funding"
    return "mixed_positive_funding"


def _coverage(
    *,
    requested_start: str,
    requested_end: str,
    data_start: str,
    data_end: str,
) -> dict[str, float]:
    requested_start_dt = datetime.fromisoformat(requested_start).replace(tzinfo=UTC)
    requested_end_dt = datetime.fromisoformat(requested_end).replace(tzinfo=UTC)
    data_start_dt = datetime.fromisoformat(data_start)
    data_end_dt = datetime.fromisoformat(data_end)
    requested_days = max((requested_end_dt - requested_start_dt).total_seconds() / 86_400, 0.0)
    data_days = max((data_end_dt - data_start_dt).total_seconds() / 86_400, 0.0)
    return {
        "requested_days": _round(requested_days),
        "data_coverage_days": _round(data_days),
        "data_coverage_pct": _round(data_days / requested_days * 100 if requested_days else 0.0),
    }


def _robustness_summary(
    baseline: list[dict[str, Any]], sensitivity: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_returns = [float(row["net_return_pct"]) for row in baseline]
    sensitivity_returns = [float(row["net_return_pct"]) for row in sensitivity]
    return {
        "baseline_runs": len(baseline),
        "sensitivity_runs": len(sensitivity),
        "baseline_positive_share": _positive_share(baseline_returns),
        "sensitivity_positive_share": _positive_share(sensitivity_returns),
        "baseline_median_net_return_pct": _median(baseline_returns),
        "sensitivity_median_net_return_pct": _median(sensitivity_returns),
        "baseline_worst_net_return_pct": min(baseline_returns) if baseline_returns else 0.0,
        "sensitivity_worst_net_return_pct": (
            min(sensitivity_returns) if sensitivity_returns else 0.0
        ),
        "baseline_empty_entry_share": _empty_entry_share(baseline),
        "regime_counts": _counts(str(row["observed_regime"]) for row in baseline),
    }


def _recommendation(robustness: dict[str, Any]) -> dict[str, str]:
    baseline_positive = float(robustness["baseline_positive_share"])
    sensitivity_positive = float(robustness["sensitivity_positive_share"])
    baseline_median = float(robustness["baseline_median_net_return_pct"])
    sensitivity_median = float(robustness["sensitivity_median_net_return_pct"])
    empty_entry_share = float(robustness["baseline_empty_entry_share"])
    if (
        baseline_positive >= 0.7
        and sensitivity_positive >= 0.6
        and baseline_median >= 0.25
        and sensitivity_median >= 0.1
        and empty_entry_share <= 0.2
    ):
        status = "small_paper_candidate_human_gate_only"
        reason = "mostly positive across regimes and still survives conservative sensitivity."
    elif baseline_positive >= 0.45 and baseline_median > 0:
        status = "keep_incubating"
        reason = "baseline has some edge, but robustness is not strong enough for promotion."
    else:
        status = "eliminate_or_hold_research_only"
        reason = "edge is not reliably positive after conservative costs."
    return {
        "status": status,
        "reason": reason,
        "guardrail": "no auto-graduation; any paper allocation requires explicit human approval",
    }


def _sensitivity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[float]] = {}
    for row in rows:
        key = (
            float(row["taker_fee_bps"]),
            float(row["slippage_bps"]),
            float(row["min_funding_bps"]),
        )
        grouped.setdefault(key, []).append(float(row["net_return_pct"]))
    summary = []
    for key, values in grouped.items():
        summary.append(
            {
                "taker_fee_bps": key[0],
                "slippage_bps": key[1],
                "min_funding_bps": key[2],
                "runs": len(values),
                "positive_share": _positive_share(values),
                "median_net_return_pct": _median(values),
                "worst_net_return_pct": min(values),
                "edge_fragile": _positive_share(values) < 0.5 or _median(values) <= 0,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            float(row["taker_fee_bps"]),
            float(row["slippage_bps"]),
            float(row["min_funding_bps"]),
        ),
    )


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    baseline = cast(list[dict[str, Any]], payload["baseline"])
    sensitivity_summary = cast(list[dict[str, Any]], payload["sensitivity_summary"])
    robustness = cast(dict[str, Any], payload["robustness"])
    recommendation = cast(dict[str, str], payload["recommendation"])
    failures = cast(list[dict[str, str]], payload["failures"])
    lines = [
        "# Funding Arb Evidence Matrix",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Source: `ccxt.{payload['source']}` true historical funding + spot/swap OHLCV",
        f"JSON artifact: `{json_path}`",
        "",
        "## Recommendation",
        "",
        f"- Status: `{recommendation['status']}`",
        f"- Reason: {recommendation['reason']}",
        f"- Guardrail: {recommendation['guardrail']}",
        "",
        "## Robustness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in robustness.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Baseline Evidence",
            "",
            "| Symbol | Period | Data UTC | Coverage % | Observed regime | Net % | "
            "Annualized % | Sharpe | Max DD % | Neg funding share | Entries | Gross % | "
            "Fees % | Slippage % |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )
    for row in baseline:
        lines.append(
            "| {symbol} | {period} | {data_start}..{data_end} | {data_coverage_pct:.2f} | "
            "{observed_regime} | {net_return_pct:.4f} | {annualized_return_pct:.4f} | "
            "{sharpe:.4f} | {max_drawdown_pct:.4f} | {negative_funding_period_share:.4f} | "
            "{entries} | {gross_return_pct:.4f} | {fee_cost_pct:.4f} | "
            "{slippage_cost_pct:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Sensitivity Summary",
            "",
            "| Taker bps | Slippage bps | Min funding bps | Runs | Positive share | "
            "Median net % | Worst net % | Fragile |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sensitivity_summary:
        lines.append(
            "| {taker_fee_bps:.1f} | {slippage_bps:.1f} | {min_funding_bps:.1f} | "
            "{runs} | {positive_share:.4f} | {median_net_return_pct:.4f} | "
            "{worst_net_return_pct:.4f} | {edge_fragile} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    if not failures:
        lines.append("No data-fetch failures.")
    else:
        for failure in failures:
            lines.append(
                "- {symbol} {period} {start}..{end}: {error}".format(**failure)
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Backtest-only; no candidate graduation, paper trading, or live trading action.",
            "- Uses aligned public candles around funding timestamps; no order book depth, "
            "margin liquidation path, borrow availability, or exchange outage modeling.",
            "- Sensitivity changes costs and entry threshold over the same fetched historical "
            "events; it does not optimize parameters for the future.",
        ]
    )
    return "\n".join(lines) + "\n"


def _periods_from_env() -> list[EvidencePeriod]:
    raw = os.getenv("FUNDING_ARB_EVIDENCE_PERIODS", "").strip()
    if not raw:
        return [EvidencePeriod(*period) for period in DEFAULT_PERIODS]
    periods: list[EvidencePeriod] = []
    for item in raw.split(";"):
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 4:
            raise ValueError(
                "FUNDING_ARB_EVIDENCE_PERIODS entries must be name|start|end|note"
            )
        periods.append(EvidencePeriod(parts[0], parts[1], parts[2], parts[3]))
    return periods


def _env_csv(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _env_float_csv(name: str, default: tuple[float, ...]) -> list[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _round(value: object) -> float:
    return round(float(value), 6)


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 6) if values else 0.0


def _positive_share(values: list[float]) -> float:
    return round(sum(1 for value in values if value > 0) / len(values), 6) if values else 0.0


def _empty_entry_share(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if int(row["entries"]) == 0) / len(rows), 6)


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
