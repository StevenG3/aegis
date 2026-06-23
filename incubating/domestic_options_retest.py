from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.domestic_options_retest import (
    DomesticOptionsConfig,
    evaluate_data_feasibility,
    nan_to_none,
    requirement_keys,
    trial_count,
)
from aegis.private_paths import private_dir_from_cli

BRIEFING = "CODEX_OLYMPUS_72_DOMESTIC_OPTIONS_STRICT_RETEST"
EV_NEWNESS = (
    "确认模式: 机制①ADX门控期权买方=技术趋势类但工具是期权凸性; 机制②MO+IM "
    "Delta中性=vol-arb新维度。厂商报告为同段8年in-sample网格寻优,本轮只接受"
    "PIT可执行数据+全成本后的独立证据。"
)
PUBLIC_SOURCES = {
    "CZCE historical option daily": "https://www.czce.com.cn/cn/jysj/lshqxz/H077003019index_1.htm",
    "SHFE data download": "https://www.shfe.com.cn/reports/tradedata/datadownload/",
    "CFFEX historical data download": "https://www.cffex.com.cn/cn/lssjxz.html",
    "AKShare option docs": "https://akshare.akfamily.xyz/data/option/option.html",
}


def main() -> int:
    generated_at = datetime.now(timezone.utc)  # noqa: UP017 - host Python can be 3.10.
    output_dir = private_dir_from_cli(
        os.getenv("DOMESTIC_OPTIONS_OUTPUT_DIR"),
        default_task="olympus72",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DomesticOptionsConfig()
    available = _availability_from_env()
    feasibility = evaluate_data_feasibility(available, config=config, product_count=8)
    payload: dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "briefing": BRIEFING,
        "ev_newness": EV_NEWNESS,
        "mode": "CONFIRMATION_DATA_GATE",
        "verdict": feasibility["verdict"],
        "reason": feasibility["reason"],
        "public_sources_checked": PUBLIC_SOURCES,
        "report_claims": {
            "commodity_adx_option_buyer": {
                "reported_parameters_are_in_sample": True,
                "reported_search_period": "2018-06-01 to 2026-06-20",
                "vendor_parameters_treated_as_grid_points_only": {
                    "ADX": 22,
                    "EMA": "13/34",
                    "delta_range": [0.35, 0.65],
                    "hard_or_step_stop": [0.40, 0.30],
                    "atr_stop": "1.5x",
                    "min_dte": 30,
                    "iv_drop_stop": 0.08,
                },
            },
            "mo_im_delta_neutral": {
                "reported_parameters_are_in_sample": True,
                "requires_second_level_rebalance_data": True,
                "vendor_parameters_treated_as_grid_points_only": {
                    "delta_neutral_band": 0.20,
                    "rebalance_frequency": "1 second",
                    "iv_discount": 0.08,
                },
            },
        },
        "predeclared_grid": {
            "adx_thresholds": config.adx_thresholds,
            "ema_fast_windows": config.ema_fast_windows,
            "ema_slow_windows": config.ema_slow_windows,
            "delta_min_values": config.delta_min_values,
            "delta_max_values": config.delta_max_values,
            "hard_stop_values": config.hard_stop_values,
            "trail_drawdown_values": config.trail_drawdown_values,
            "atr_stop_multipliers": config.atr_stop_multipliers,
            "min_dte_values": config.min_dte_values,
            "iv_drop_stop_values": config.iv_drop_stop_values,
            "mo_delta_bands": config.mo_delta_bands,
            "hedge_check_seconds": config.hedge_check_seconds,
            "mo_iv_discount_values": config.mo_iv_discount_values,
            "commodity_trial_count_n": trial_count(
                config,
                "commodity_adx_option_buyer",
                product_count=8,
            ),
            "mo_im_trial_count_n": trial_count(
                config,
                "mo_im_delta_neutral",
                product_count=8,
            ),
        },
        "data_gate": feasibility,
        "discipline": {
            "t_plus_1_execution_required": True,
            "locked_oos_required": True,
            "walk_forward_required": True,
            "full_costs_required": (
                "enter ask, exit bid, spread, fee, slippage; mechanism② also counts every IM "
                "rebalance fee/slippage and margin capital"
            ),
            "funding": "N/A for listed domestic options/futures; margin is required for IM hedge",
            "max_positive_verdict": "SUGGESTIVE",
            "live_trading": False,
            "broker_gui": False,
        },
        "known_limits": [
            "Free public daily exchange files can expose some settlement/OHLC fields, but do not "
            "provide complete PIT executable option-chain snapshots with bid/ask, greeks, and IV "
            "for all report symbols.",
            "MO+IM second-level dynamic hedging cannot be validated from daily bars; it requires "
            "aligned MO option quotes/greeks and IM futures executable quotes for every rebalance.",
            "No vendor report number is treated as evidence because the report states parameters "
            "were optimized on the same 2018-2026 sample.",
        ],
        "public_boundary": (
            "Detailed evidence is private. Public repo contains generic feasibility/cost "
            "discipline "
            "code and synthetic tests only; no account, credential, GUI automation, or proprietary "
            "data is used."
        ),
    }
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"domestic-options-retest-{stamp}.json"
    md_path = output_dir / f"domestic-options-retest-{stamp}.md"
    json_path.write_text(
        json.dumps(nan_to_none(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "reason": payload["reason"],
                "json": str(json_path),
                "markdown": str(md_path),
                "commodity_missing": _missing(payload, "commodity_adx_option_buyer"),
                "mo_im_missing": _missing(payload, "mo_im_delta_neutral"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _availability_from_env() -> dict[str, bool]:
    available: dict[str, bool] = {}
    all_keys = sorted(
        set(requirement_keys("commodity_adx_option_buyer"))
        | set(requirement_keys("mo_im_delta_neutral"))
    )
    for key in all_keys:
        available[key] = _truthy(os.getenv(f"DOMESTIC_OPTIONS_HAVE_{key.upper()}"))
    return available


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _missing(payload: dict[str, Any], mechanism: str) -> list[str]:
    mechanisms = payload["data_gate"]["mechanisms"]
    return list(mechanisms[mechanism]["missing_requirements"])


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    data_gate = payload["data_gate"]
    mechanisms = data_gate["mechanisms"]
    lines = [
        "# Olympus #72 Domestic Options Strict Retest Evidence",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- verdict: {payload['verdict']}",
        f"- reason: {payload['reason']}",
        f"- json: {json_path}",
        "",
        "## Mechanism Verdicts",
    ]
    for mechanism in ("commodity_adx_option_buyer", "mo_im_delta_neutral"):
        report = mechanisms[mechanism]
        lines.extend(
            [
                f"- {mechanism}: {report['verdict']}",
                f"  - trial_count_n: {report['trial_count_n']}",
                f"  - missing: {', '.join(report['missing_requirements'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Data Gate",
            "",
            "Public/free sources can provide some daily exchange files, but this task requires PIT "
            "contract chains, executable bid/ask, greeks/IV or repricing inputs, and for MO+IM "
            "second-level aligned option/futures quotes plus rebalance costs and margin rules.",
            "",
            "Funding: N/A for listed options/futures. IM margin and hedge execution costs are "
            "mandatory for mechanism②.",
            "",
            "This artifact is private evidence, not a trading signal.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
