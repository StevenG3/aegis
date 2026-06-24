#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from aegis.btc_vrp_condor_timed import (
    build_timed_condor_rows,
    timed_condor_variant_names,
)
from aegis.btc_vrp_public_data import (
    DVOL_START,
    END_DATE,
    REQUIRED_CRASH_WINDOWS,
    crash_window_coverage,
    dvol_history,
    funding_rates,
    spot_daily_closes,
    to_ms,
)
from aegis.btc_vrp_short_vol import (
    ShortVolVrpConfig,
    btc_vrp_data_blocked_report,
    run_btc_short_vol_vrp,
)
from aegis.private_paths import private_dir_from_cli

DEFAULT_TASK = "olympus80"
NAIVE_VARIANTS = (
    "atm_straddle_7d",
    "atm_straddle_14d",
    "otm_strangle_7d",
    "otm_strangle_14d",
    "iron_condor_7d",
    "iron_condor_14d",
)


def main() -> int:
    args = _parse_args()
    output_dir = private_dir_from_cli(args.output_dir, default_task=DEFAULT_TASK)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_btc_vrp_condor_timed_evidence()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"btc-vrp-condor-timed-{stamp}.json"
    md_path = output_dir / f"btc-vrp-condor-timed-{stamp}.md"
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
                "abc_conclusion": payload["abc_conclusion"],
                "reason": payload["reason"],
                "coverage": payload["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_btc_vrp_condor_timed_evidence() -> Mapping[str, Any]:
    config = ShortVolVrpConfig(max_drawdown_limit=-0.30)
    start_ms = to_ms(DVOL_START)
    end_ms = to_ms(END_DATE)
    dvol = dvol_history(start_ms, end_ms)
    prices = spot_daily_closes(start_ms, end_ms + 20 * 24 * 3600 * 1000)
    funding = funding_rates(start_ms, end_ms + 20 * 24 * 3600 * 1000)
    crash_coverage = {
        name: crash_window_coverage(start, end, dvol, prices)
        for name, (start, end) in REQUIRED_CRASH_WINDOWS.items()
    }
    coverage: dict[str, Any] = {
        "iv_source": "Deribit public BTC DVOL index close (proxy IV, not executable chain bid/ask)",
        "price_source": "Binance public BTCUSDT 1d klines",
        "funding_source": "Binance public BTCUSDT perpetual fundingRate",
        "dvol_rows": len(dvol),
        "price_rows": len(prices),
        "funding_rows": len(funding),
        "required_crash_windows": list(REQUIRED_CRASH_WINDOWS),
        "crash_window_coverage": crash_coverage,
        "predeclared_timed_configs": list(timed_condor_variant_names()),
        "predeclared_timed_config_n": len(timed_condor_variant_names()),
        "max_drawdown_limit": config.max_drawdown_limit,
        "missing_2020_covid": "DVOL did not exist; recorded as limitation, not hard gate",
    }
    missing_required = [
        name
        for name, report in crash_coverage.items()
        if int(cast(Mapping[str, object], report)["dvol_rows"]) == 0
        or int(cast(Mapping[str, object], report)["price_rows"]) == 0
    ]
    if missing_required:
        reason = f"DVOL-era required crash windows missing: {', '.join(missing_required)}"
        report = btc_vrp_data_blocked_report(reason=reason, coverage=coverage, config=config)
        naive_report = report
        diagnostics: Mapping[str, Any] = {}
    elif not funding:
        report = btc_vrp_data_blocked_report(
            reason="Binance funding history unavailable; cannot compute delta-hedge funding cost",
            coverage=coverage,
            config=config,
        )
        naive_report = report
        diagnostics = {}
    else:
        timed_rows, diagnostics = build_timed_condor_rows(dvol=dvol, prices=prices, funding=funding)
        naive_rows = _build_naive_proxy_rows(dvol=dvol, prices=prices, funding=funding)
        report = run_btc_short_vol_vrp(
            timed_rows,
            config=config,
            all_variants=timed_condor_variant_names(),
        )
        naive_report = run_btc_short_vol_vrp(
            naive_rows,
            config=config,
            all_variants=NAIVE_VARIANTS,
        )
        coverage = {
            **coverage,
            **cast(Mapping[str, Any], report.get("coverage", {})),
            "timed_diagnostics": diagnostics,
            "naive_best_candidate": naive_report.get("best_candidate"),
            "naive_standard_metrics": naive_report.get("standard_metrics"),
            "naive_multiple_testing": naive_report.get("multiple_testing"),
        }
    abc_conclusion = _abc_conclusion(report=report, naive_report=naive_report)
    return {
        "briefing": "CODEX_OLYMPUS_80C_VRP_CONDOR_TIMED",
        "generated_at": datetime.now(UTC).isoformat(),
        "state": report.get("state"),
        "verdict": report.get("verdict"),
        "abc_conclusion": abc_conclusion,
        "reason": report.get("reason"),
        "data_adequacy": report.get("data_adequacy"),
        "unlock_condition": (
            "paid PIT Deribit/Tardis BTC option chain bid/ask by strike and tenor, executable "
            "depth, and longer crash history including a 2020-03-like regime"
        ),
        "candidate_count_n": report.get("candidate_count_n"),
        "coverage": coverage,
        "standard_metrics": report.get("standard_metrics"),
        "benchmark_metrics": report.get("benchmark_metrics"),
        "multiple_testing": report.get("multiple_testing"),
        "best_candidate": report.get("best_candidate"),
        "naive_comparison": {
            "state": naive_report.get("state"),
            "verdict": naive_report.get("verdict"),
            "best_candidate": naive_report.get("best_candidate"),
            "standard_metrics": naive_report.get("standard_metrics"),
            "multiple_testing": naive_report.get("multiple_testing"),
        },
        "gate_evidence": {
            "condor_cap": "return_floor caps each trade loss after paying insurance proxy",
            "iv_timing": "forecast_RV EWMA and DVOL percentile use current/prior data only",
            "regime_filter": "skips recent RV surge, DVOL jump, and 7d drawdown stress",
            "proxy_limit": "DVOL index is not executable option-chain bid/ask",
            "no_live_or_order_access": True,
        },
        "safety": {
            "read_only_public_apis": True,
            "wallet_or_order_access": False,
            "live_trading": False,
            "account_access": False,
        },
    }


def _abc_conclusion(*, report: Mapping[str, Any], naive_report: Mapping[str, Any]) -> str:
    if report.get("state") == "INSUFFICIENT":
        return "INSUFFICIENT_DATA_GATE"
    metrics = cast(Mapping[str, Any], report.get("standard_metrics", {}) or {})
    naive_metrics = cast(Mapping[str, Any], naive_report.get("standard_metrics", {}) or {})
    timed_mean = _float(metrics.get("mean_net_return"))
    naive_mean = _float(naive_metrics.get("mean_net_return"))
    timed_maxdd = _float(metrics.get("max_drawdown"))
    naive_maxdd = _float(naive_metrics.get("max_drawdown"))
    multiple = cast(Mapping[str, Any], report.get("multiple_testing", {}) or {})
    fdr_after = int(cast(int | None, multiple.get("fdr_after")) or 0)
    pbo_report = cast(Mapping[str, Any], multiple.get("pbo", {}) or {})
    pbo_valid = bool(pbo_report.get("valid"))
    pbo_value = _float(pbo_report.get("pbo"), default=1.0)
    tail_safe = timed_maxdd >= -0.30
    materially_better = timed_mean > naive_mean and timed_maxdd > naive_maxdd
    if (
        materially_better
        and timed_mean > 0.0
        and fdr_after > 0
        and pbo_valid
        and pbo_value <= 0.20
        and tail_safe
    ):
        return "A_SUGGESTIVE_WORTH_TARDIS_CONFIRMATION"
    if materially_better:
        return "B_DIRECTIONAL_IMPROVEMENT_BUT_GATE_FAILED_OR_PROXY_LIMITED"
    return "C_NO_IMPROVEMENT_NOT_WORTH_PAID_CHAIN_YET"


def _build_naive_proxy_rows(
    *,
    dvol: list[tuple[int, float]],
    prices: Mapping[int, float],
    funding: Mapping[int, float],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    price_days = sorted(prices)
    for timestamp, dvol_close in dvol:
        current_day = timestamp - (timestamp % (24 * 3600 * 1000))
        if current_day not in prices:
            continue
        for variant in NAIVE_VARIANTS:
            tenor_days = 14 if variant.endswith("14d") else 7
            hedge_notional = 0.25 if variant.startswith("iron_condor") else 0.50
            expiry_day = current_day + tenor_days * 24 * 3600 * 1000
            window_days = [day for day in price_days if current_day <= day <= expiry_day]
            if len(window_days) < tenor_days + 1:
                continue
            returns = [
                math.log(prices[window_days[index]] / prices[window_days[index - 1]])
                for index in range(1, len(window_days))
            ]
            realized_vol = math.sqrt(sum(value * value for value in returns) * 365.0 / len(returns))
            max_abs_daily = max(abs(value) for value in returns) if returns else 0.0
            tail_threshold = 0.18 if variant.startswith("iron_condor") else 0.12
            tail_cap = 0.25 if variant.startswith("iron_condor") else None
            raw_tail = max(0.0, max_abs_daily - tail_threshold)
            tail_loss = min(raw_tail, tail_cap) if tail_cap is not None else raw_tail
            rows.append(
                {
                    "variant": variant,
                    "iv_ts": current_day,
                    "expiry_ts": expiry_day,
                    "implied_vol": dvol_close / 100.0,
                    "realized_vol": realized_vol,
                    "variance_year_fraction": tenor_days / 365.0,
                    "option_spread_cost": 0.010,
                    "hedge_fee_cost": len(returns) * 0.0004 * hedge_notional,
                    "hedge_slippage_cost": len(returns) * 0.0003 * hedge_notional,
                    "funding_cost": _funding_cost(
                        funding=funding,
                        start_day=current_day,
                        end_day=expiry_day,
                        hedge_notional=hedge_notional,
                    ),
                    "tail_loss": tail_loss,
                }
            )
    return rows


def _funding_cost(
    *,
    funding: Mapping[int, float],
    start_day: int,
    end_day: int,
    hedge_notional: float,
) -> float:
    selected = [abs(rate) for ts, rate in funding.items() if start_day <= ts <= end_day]
    return sum(selected) * hedge_notional


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (float, int)):
        return float(value)
    return default


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    metrics = cast(Mapping[str, Any], payload.get("standard_metrics", {}) or {})
    naive = cast(Mapping[str, Any], payload.get("naive_comparison", {}) or {})
    naive_metrics = cast(Mapping[str, Any], naive.get("standard_metrics", {}) or {})
    multiple = cast(Mapping[str, Any], payload.get("multiple_testing", {}) or {})
    coverage = cast(Mapping[str, Any], payload.get("coverage", {}) or {})
    return "\n".join(
        [
            "# CODEX OLYMPUS 80C BTC Timed Condor VRP Evidence",
            "",
            f"- State: `{payload.get('state')}`",
            f"- Verdict: `{payload.get('verdict')}`",
            f"- ABC conclusion: `{payload.get('abc_conclusion')}`",
            f"- Data adequacy: `{payload.get('data_adequacy')}`",
            f"- Reason: {payload.get('reason')}",
            f"- JSON: `{json_path}`",
            "",
            "## Timed Condor",
            f"- Variant: `{metrics.get('variant')}`",
            f"- Mean net return: `{metrics.get('mean_net_return')}`",
            f"- Win rate: `{metrics.get('positive_period_win_rate')}`",
            f"- MaxDD: `{metrics.get('max_drawdown')}`",
            f"- CVaR95: `{metrics.get('cvar_95')}`",
            f"- CVaR99: `{metrics.get('cvar_99')}`",
            f"- Worst trade: `{metrics.get('worst_trade')}`",
            f"- Worst month: `{metrics.get('worst_month')}`",
            "",
            "## Naive #80B Comparison",
            f"- Mean net return: `{naive_metrics.get('mean_net_return')}`",
            f"- Win rate: `{naive_metrics.get('positive_period_win_rate')}`",
            f"- MaxDD: `{naive_metrics.get('max_drawdown')}`",
            f"- CVaR95: `{naive_metrics.get('cvar_95')}`",
            f"- CVaR99: `{naive_metrics.get('cvar_99')}`",
            "",
            "## Gates",
            f"- Candidate N: `{payload.get('candidate_count_n')}`",
            f"- Multiple testing: `{multiple}`",
            f"- Coverage: `{coverage}`",
        ]
    ) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #80C BTC timed-condor VRP evidence.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
