#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from aegis.btc_vrp_short_vol import (
    ShortVolVrpConfig,
    btc_vrp_data_blocked_report,
    run_btc_short_vol_vrp,
)
from aegis.private_paths import private_dir_from_cli
from aegis.spx_vrp_harvest import (
    REQUIRED_SPX_CRASH_WINDOWS,
    SpxDailyBar,
    build_always_short_rows,
    build_spx_vrp_deployment_rows,
    build_spx_vrp_rows,
    crash_window_coverage,
    garman_klass_vol,
    gross_vrp_self_check,
    locked_oos_variant_report,
    spx_vrp_deployment_variant_names,
    spx_vrp_risk_curve,
)

DEFAULT_TASK = "olympus85"
DEFAULT_START = date(1990, 1, 2)
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
USER_AGENT = "aegis-spx-vrp-harvest/0.1 read-only"


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_spx_vrp_harvest_evidence(cache_dir=_cache_dir(args.cache_dir, output_dir))
    stamp = datetime.now(timezone.utc).strftime(  # noqa: UP017 - host evidence uses py3.10.
        "%Y%m%dT%H%M%SZ"
    )
    json_path = output_dir / f"spx-vrp-harvest-{stamp}.json"
    md_path = output_dir / f"spx-vrp-harvest-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "state": payload["state"],
                "verdict": payload["verdict"],
                "data_adequacy": payload["data_adequacy"],
                "tail_conclusion": payload["tail_conclusion"],
                "reason": payload["reason"],
                "coverage": payload["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_spx_vrp_harvest_evidence(*, cache_dir: Path) -> Mapping[str, Any]:
    config = ShortVolVrpConfig(max_drawdown_limit=-0.30)
    vix = load_cboe_vix(cache_dir / "vix-history.csv")
    spx = load_spx_yfinance(cache_dir / "spx-yfinance.csv")
    coverage: dict[str, Any] = {
        "iv_source": (
            "CBOE VIX daily close via public VIX_History.csv "
            "(index proxy, not executable bid/ask)"
        ),
        "price_source": "yfinance ^GSPC daily OHLC",
        "start": min(vix).isoformat() if vix else None,
        "end": max(set(vix) & set(spx)).isoformat() if vix and spx else None,
        "vix_rows": len(vix),
        "spx_rows": len(spx),
        "required_crash_windows": list(REQUIRED_SPX_CRASH_WINDOWS),
        "max_drawdown_limit": config.max_drawdown_limit,
        "predeclared_configs": list(spx_vrp_deployment_variant_names()),
        "predeclared_config_n": len(spx_vrp_deployment_variant_names()),
        "costs": {
            "option_spread_proxy": "3% of premium credit, same scale as VRP premium",
            "hard_cap_insurance_proxy": "2% of premium credit for capped-wing proxy",
            "hedge_fee_proxy": "delta-hedge turnover x 2 bps",
            "hedge_slippage_proxy": "delta-hedge turnover x 5 bps",
            "funding": "N/A for equity index option proxy",
        },
        "exposure_policy": "non_overlapping_per_variant",
    }
    crash_coverage = crash_window_coverage(vix=vix, spx=spx)
    coverage["crash_window_coverage"] = crash_coverage
    missing = [
        name
        for name, item in crash_coverage.items()
        if not bool(item["covered"])
    ]
    report: Mapping[str, Any]
    always_report: Mapping[str, Any]
    locked_oos: Mapping[str, Any]
    if missing:
        reason = f"required SPX/VIX crash windows missing: {', '.join(missing)}"
        report = btc_vrp_data_blocked_report(reason=reason, coverage=coverage, config=config)
        always_report = report
        diagnostics: Mapping[str, Any] = {}
        always_diagnostics: Mapping[str, Any] = {}
    else:
        rows, diagnostics = build_spx_vrp_deployment_rows(vix=vix, spx=spx)
        base_rows, base_diagnostics = build_spx_vrp_rows(vix=vix, spx=spx)
        always_rows, always_diagnostics = build_always_short_rows(vix=vix, spx=spx)
        gross_self_check = gross_vrp_self_check(rows)
        base_gross_self_check = gross_vrp_self_check(base_rows)
        always_gross_self_check = gross_vrp_self_check(always_rows)
        if not bool(gross_self_check.get("valid")):
            report = {
                "state": "INSUFFICIENT",
                "verdict": "MODEL_INVALID_GROSS_VRP",
                "reason": (
                    "model failed gross VRP positivity self-check; P&L representation "
                    "does not capture the known positive average IV-minus-RV premium"
                ),
                "candidate_count_n": len(rows),
            }
            locked_oos = {"valid": False, "reason": "gross VRP self-check failed"}
            always_report = report
        else:
            report = run_btc_short_vol_vrp(
                rows,
                config=config,
                all_variants=spx_vrp_deployment_variant_names(),
            )
            locked_oos = locked_oos_variant_report(rows)
            always_report = run_btc_short_vol_vrp(
                always_rows,
                config=config,
                all_variants=("always_short_vrp_21d_cap15_tv10",),
            )
        coverage = dict({
            **coverage,
            **_mapping_or_empty(report.get("coverage")),
            "diagnostics": diagnostics,
            "base_diagnostics": base_diagnostics,
            "always_short_diagnostics": always_diagnostics,
            "gross_vrp_self_check": gross_self_check,
            "base_gross_vrp_self_check": base_gross_self_check,
            "always_short_gross_vrp_self_check": always_gross_self_check,
            "rv_estimators": _rv_estimator_summary(spx),
            "locked_oos": locked_oos,
            "risk_curve": spx_vrp_risk_curve(rows),
        })
    if missing:
        locked_oos = {"valid": False, "reason": "data gate blocked"}
    tail_conclusion = _tail_conclusion(report)
    return {
        "briefing": "CODEX_OLYMPUS_85C_SPX_VRP_RISK_DEPLOYMENT",
        "generated_at": datetime.now(  # noqa: UP017 - host evidence uses py3.10.
            timezone.utc  # noqa: UP017 - host evidence uses py3.10.
        ).isoformat(),
        "state": report.get("state"),
        "verdict": report.get("verdict"),
        "tail_conclusion": tail_conclusion,
        "reason": report.get("reason"),
        "data_adequacy": "limited" if report.get("state") != "INSUFFICIENT" else "blocked",
        "unlock_condition": (
            "paid PIT SPX/SPY option chain bid/ask by strike and tenor, executable depth, "
            "margin model, and ETF/put-write benchmark total returns"
        ),
        "candidate_count_n": report.get("candidate_count_n"),
        "coverage": coverage,
        "standard_metrics": report.get("standard_metrics"),
        "benchmark_metrics": {
            "cash": {"mean_return": 0.0},
            "always_short_vol": always_report.get("standard_metrics"),
            "buy_hold_spx": _buy_hold_spx_metrics(spx),
            "put_index": _load_benchmark_metrics(cache_dir, "^PUT"),
            "putw_etf": _load_benchmark_metrics(cache_dir, "PUTW"),
            "svol_etf": _load_benchmark_metrics(cache_dir, "SVOL"),
            "put_write_etf_comparability": (
                "Risk-matched comparison is approximate: VIX/SPX proxy is not "
                "strike-level executable option evidence, while PUT/PUTW/SVOL are "
                "index/ETF total-return proxies when available."
            ),
        },
        "multiple_testing": report.get("multiple_testing"),
        "locked_oos": locked_oos,
        "best_candidate": report.get("best_candidate"),
        "always_short_comparison": {
            "state": always_report.get("state"),
            "verdict": always_report.get("verdict"),
            "best_candidate": always_report.get("best_candidate"),
            "standard_metrics": always_report.get("standard_metrics"),
            "multiple_testing": always_report.get("multiple_testing"),
        },
        "gate_evidence": {
            "no_lookahead": (
                "VIX/forecast/regime use t and earlier; "
                "realized variance starts after t"
            ),
            "hard_cap": "net_return_override is clipped at scaled max single-trade loss",
            "portfolio_exposure": (
                "base non-overlap per variant plus predeclared risk-tier max_exposure cap"
            ),
            "vol_target": "position scale uses forecast RV at t only",
            "gross_vrp_self_check": (
                "gross mean must be positive before net-cost strategy verdict is allowed"
            ),
            "proxy_limit": "VIX index is not executable option-chain bid/ask",
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }


def load_cboe_vix(path: Path) -> dict[date, float]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(CBOE_VIX_URL, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=30.0) as response:  # noqa: S310 - fixed CBOE HTTPS URL.
            path.write_bytes(response.read())
    result: dict[date, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_date = row.get("DATE") or row.get("Date") or row.get("date")
            raw_close = row.get("CLOSE") or row.get("Close") or row.get("close")
            if raw_date is None or raw_close is None:
                continue
            parsed_date = _parse_date(raw_date)
            if parsed_date is None:
                continue
            try:
                result[parsed_date] = float(raw_close)
            except ValueError:
                continue
    return {day: value for day, value in result.items() if day >= DEFAULT_START and value > 0.0}


def load_spx_yfinance(path: Path) -> dict[date, SpxDailyBar]:
    if not path.exists():
        yf = importlib.import_module("yfinance")

        path.parent.mkdir(parents=True, exist_ok=True)
        frame = yf.download(
            "^GSPC",
            start=DEFAULT_START.isoformat(),
            end=(date.today()).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Date", "Open", "High", "Low", "Close"])
            writer.writeheader()
            for raw_index, row in frame.iterrows():
                writer.writerow(
                    {
                        "Date": str(raw_index)[:10],
                        "Open": _float_or_default(
                            _row_value(row, "Open"), _row_value(row, "Close")
                        ),
                        "High": _float_or_default(
                            _row_value(row, "High"), _row_value(row, "Close")
                        ),
                        "Low": _float_or_default(
                            _row_value(row, "Low"), _row_value(row, "Close")
                        ),
                        "Close": _float_or_default(_row_value(row, "Close"), None),
                    }
                )
    result: dict[date, SpxDailyBar] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = _bar_from_row(row)
            if parsed is not None:
                result[parsed.date] = parsed
    return result


def _bar_from_row(row: Mapping[str, str]) -> SpxDailyBar | None:
    raw_date = row.get("Date")
    open_ = _float_or_none(row.get("Open"))
    high = _float_or_none(row.get("High"))
    low = _float_or_none(row.get("Low"))
    close = _float_or_none(row.get("Close"))
    if (
        raw_date is None
        or open_ is None
        or high is None
        or low is None
        or close is None
        or min(open_, high, low, close) <= 0.0
    ):
        return None
    return SpxDailyBar(
        date.fromisoformat(raw_date),
        open_,
        high,
        low,
        close,
    )


def _rv_estimator_summary(spx: Mapping[date, SpxDailyBar]) -> Mapping[str, Any]:
    bars = tuple(spx[day] for day in sorted(spx)[-252:])
    return {
        "close_to_close_primary": True,
        "garman_klass_last_252d": garman_klass_vol(bars),
        "parkinson_gk_available": True,
    }


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _buy_hold_spx_metrics(spx: Mapping[date, SpxDailyBar]) -> Mapping[str, Any]:
    days = sorted(spx)
    if len(days) < 2:
        return {"valid": False, "reason": "insufficient SPX rows"}
    closes = [spx[day].close for day in days if spx[day].close > 0.0]
    if len(closes) < 2:
        return {"valid": False, "reason": "insufficient positive SPX closes"}
    peak = closes[0]
    max_drawdown = 0.0
    returns: list[float] = []
    for index in range(1, len(closes)):
        previous = closes[index - 1]
        current = closes[index]
        returns.append(current / previous - 1.0)
        peak = max(peak, current)
        max_drawdown = min(max_drawdown, current / peak - 1.0)
    return {
        "valid": True,
        "start": days[0].isoformat(),
        "end": days[-1].isoformat(),
        "total_return": closes[-1] / closes[0] - 1.0,
        "mean_daily_return": sum(returns) / len(returns),
        "max_drawdown": max_drawdown,
        **_daily_return_metrics(returns),
    }


def _load_benchmark_metrics(cache_dir: Path, symbol: str) -> Mapping[str, Any]:
    try:
        closes = _load_yfinance_closes(cache_dir / f"{_safe_symbol(symbol)}.csv", symbol)
    except Exception as exc:  # noqa: BLE001 - benchmark availability is reported, not fatal.
        return {"valid": False, "symbol": symbol, "reason": str(exc)}
    if len(closes) < 30:
        return {"valid": False, "symbol": symbol, "reason": "insufficient rows"}
    days = sorted(closes)
    prices = [closes[day] for day in days if closes[day] > 0.0]
    if len(prices) < 30:
        return {"valid": False, "symbol": symbol, "reason": "insufficient positive closes"}
    returns = [prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))]
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        max_drawdown = min(max_drawdown, price / peak - 1.0)
    return {
        "valid": True,
        "symbol": symbol,
        "start": days[0].isoformat(),
        "end": days[-1].isoformat(),
        "rows": len(prices),
        "total_return": prices[-1] / prices[0] - 1.0,
        "max_drawdown": max_drawdown,
        **_daily_return_metrics(returns),
    }


def _load_yfinance_closes(path: Path, symbol: str) -> dict[date, float]:
    if not path.exists():
        yf = importlib.import_module("yfinance")

        path.parent.mkdir(parents=True, exist_ok=True)
        frame = yf.download(
            symbol,
            start=DEFAULT_START.isoformat(),
            end=(date.today()).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Date", "Close"])
            writer.writeheader()
            for raw_index, row in frame.iterrows():
                writer.writerow(
                    {
                        "Date": str(raw_index)[:10],
                        "Close": _float_or_default(_row_value(row, "Close"), None),
                    }
                )
    result: dict[date, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_date = row.get("Date")
            close = _float_or_none(row.get("Close"))
            if raw_date is None or close is None or close <= 0.0:
                continue
            result[date.fromisoformat(raw_date)] = close
    return result


def _daily_return_metrics(returns: list[float]) -> Mapping[str, float]:
    if not returns:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }
    total = math.prod(1.0 + value for value in returns) - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    annualized_return = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
    stdev = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    downside = [value for value in returns if value < 0.0]
    downside_stdev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    mean = statistics.fmean(returns)
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": stdev * math.sqrt(252.0),
        "sharpe": mean / stdev * math.sqrt(252.0) if stdev > 0.0 else 0.0,
        "sortino": mean / downside_stdev * math.sqrt(252.0) if downside_stdev > 0.0 else 0.0,
    }


def _safe_symbol(symbol: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in symbol).strip("_").lower()


def _tail_conclusion(report: Mapping[str, Any]) -> str:
    if report.get("state") == "INSUFFICIENT":
        return "INSUFFICIENT_DATA_GATE"
    if report.get("verdict") == "PREMIUM_EXISTS_BUT_TAIL_UNSAFE":
        return "PREMIUM_EXISTS_BUT_TAIL_UNSAFE"
    metrics = cast(Mapping[str, Any], report.get("standard_metrics", {}) or {})
    mean_return = _float_or_none(metrics.get("mean_net_return")) or 0.0
    maxdd = _float_or_none(metrics.get("max_drawdown")) or -1.0
    if mean_return > 0.0 and maxdd >= -0.30:
        return "TAIL_SURVIVABLE_POSITIVE_EV_PROXY"
    if maxdd >= -0.30:
        return "TAIL_SURVIVABLE_EV_NOT_ESTABLISHED"
    return "TAIL_UNSAFE"


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}) or {})
    multiple = cast(Mapping[str, Any], payload.get("multiple_testing", {}) or {})
    best = cast(Mapping[str, Any], payload.get("best_candidate", {}) or {})
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    return "\n".join(
        [
            "# CODEX OLYMPUS 85C SPX VRP Risk Deployment Evidence",
            "",
            f"- State: `{payload.get('state')}`",
            f"- Verdict: `{payload.get('verdict')}`",
            f"- Data adequacy: `{payload.get('data_adequacy')}`",
            f"- Tail conclusion: `{payload.get('tail_conclusion')}`",
            f"- Reason: {payload.get('reason')}",
            f"- VIX rows: `{coverage.get('vix_rows')}`",
            f"- SPX rows: `{coverage.get('spx_rows')}`",
            f"- Candidate N: `{payload.get('candidate_count_n')}`",
            f"- FDR after: `{multiple.get('fdr_after')}`",
            f"- Best variant: `{best.get('variant')}`",
            f"- Trades: `{metrics.get('trades')}`",
            f"- Mean net return: `{metrics.get('mean_net_return')}`",
            f"- MaxDD: `{metrics.get('max_drawdown')}`",
            f"- CVaR99: `{metrics.get('cvar_99')}`",
            f"- JSON: `{json_path}`",
            "",
            "This is a VIX/SPX proxy, not executable option-chain bid/ask evidence.",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #85 SPX/VIX short-vol VRP evidence.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def _cache_dir(raw: str | None, output_dir: Path) -> Path:
    if raw:
        return Path(raw)
    blockstorage = Path("/mnt/blockstorage")
    if blockstorage.exists():
        return blockstorage / "aegis-strategies" / "incubating" / DEFAULT_TASK / "cache"
    return output_dir / "cache"


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_value(row: Any, name: str) -> object:
    for key in getattr(row, "index", ()):
        if isinstance(key, tuple) and name in key:
            return row.get(key)
    value = row.get(name)
    if value is not None:
        return value
    return None


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _float_or_default(value: object, default: object) -> float:
    parsed = _float_or_none(value)
    if parsed is not None:
        return parsed
    fallback = _float_or_none(default)
    return fallback if fallback is not None else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
