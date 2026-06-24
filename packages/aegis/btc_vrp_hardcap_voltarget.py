from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aegis.btc_vrp_condor_timed import (
    DAY_MS,
    annualized_vol,
    forecast_rv_ewma,
    funding_cost,
    future_log_returns,
    realized_vol_window,
    trailing_change,
    trailing_drawdown,
    trailing_percentile,
)


@dataclass(frozen=True)
class HardCapVolTargetSpec:
    name: str
    tenor_days: int
    cap_multiple: float
    target_vol: float
    timing_rule: str
    iv_edge_threshold: float | None = None
    dvol_percentile_threshold: float | None = None


HARDCAP_VOLTARGET_SPECS: tuple[HardCapVolTargetSpec, ...] = (
    HardCapVolTargetSpec("hardcap_7d_c15_tv10_edge10", 7, 1.5, 0.10, "iv_edge", 0.10),
    HardCapVolTargetSpec("hardcap_7d_c15_tv10_pct75", 7, 1.5, 0.10, "dvol_percentile", None, 0.75),
    HardCapVolTargetSpec("hardcap_7d_c15_tv15_edge10", 7, 1.5, 0.15, "iv_edge", 0.10),
    HardCapVolTargetSpec("hardcap_7d_c15_tv15_pct75", 7, 1.5, 0.15, "dvol_percentile", None, 0.75),
    HardCapVolTargetSpec("hardcap_7d_c20_tv10_edge10", 7, 2.0, 0.10, "iv_edge", 0.10),
    HardCapVolTargetSpec("hardcap_7d_c20_tv10_pct75", 7, 2.0, 0.10, "dvol_percentile", None, 0.75),
    HardCapVolTargetSpec("hardcap_7d_c20_tv15_edge10", 7, 2.0, 0.15, "iv_edge", 0.10),
    HardCapVolTargetSpec("hardcap_7d_c20_tv15_pct75", 7, 2.0, 0.15, "dvol_percentile", None, 0.75),
    HardCapVolTargetSpec("hardcap_14d_c15_tv10_edge10", 14, 1.5, 0.10, "iv_edge", 0.10),
    HardCapVolTargetSpec(
        "hardcap_14d_c15_tv10_pct75", 14, 1.5, 0.10, "dvol_percentile", None, 0.75
    ),
    HardCapVolTargetSpec("hardcap_14d_c15_tv15_edge10", 14, 1.5, 0.15, "iv_edge", 0.10),
    HardCapVolTargetSpec(
        "hardcap_14d_c15_tv15_pct75", 14, 1.5, 0.15, "dvol_percentile", None, 0.75
    ),
    HardCapVolTargetSpec("hardcap_14d_c20_tv10_edge10", 14, 2.0, 0.10, "iv_edge", 0.10),
    HardCapVolTargetSpec(
        "hardcap_14d_c20_tv10_pct75", 14, 2.0, 0.10, "dvol_percentile", None, 0.75
    ),
    HardCapVolTargetSpec("hardcap_14d_c20_tv15_edge10", 14, 2.0, 0.15, "iv_edge", 0.10),
    HardCapVolTargetSpec(
        "hardcap_14d_c20_tv15_pct75", 14, 2.0, 0.15, "dvol_percentile", None, 0.75
    ),
)


def hardcap_voltarget_variant_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in HARDCAP_VOLTARGET_SPECS)


def build_hardcap_voltarget_rows(
    *,
    dvol: Sequence[tuple[int, float]],
    prices: Mapping[int, float],
    funding: Mapping[int, float],
) -> tuple[list[Mapping[str, object]], Mapping[str, Any]]:
    rows: list[Mapping[str, object]] = []
    diagnostics: dict[str, Any] = {
        "predeclared_hardcap_voltarget_configs": hardcap_voltarget_variant_names(),
        "skipped_reasons": {},
        "trade_counts_by_variant": {name: 0 for name in hardcap_voltarget_variant_names()},
        "hard_cap_bindings": 0,
        "hard_cap_violations": 0,
        "max_position_scale": 0.0,
    }
    dvol_by_day = {_day_start(timestamp): close / 100.0 for timestamp, close in dvol}
    dvol_days = sorted(dvol_by_day)
    price_days = sorted(prices)
    for current_day in dvol_days:
        if current_day not in prices:
            _count_skip(diagnostics, "missing_current_price")
            continue
        forecast_rv = forecast_rv_ewma(price_days=price_days, prices=prices, day=current_day)
        recent_rv = realized_vol_window(
            price_days=price_days, prices=prices, day=current_day, days=7
        )
        dvol_percentile = trailing_percentile(
            values_by_day=dvol_by_day, ordered_days=dvol_days, day=current_day, lookback=180
        )
        dvol_change_3d = trailing_change(values_by_day=dvol_by_day, day=current_day, lag_days=3)
        drawdown_7d = trailing_drawdown(
            price_days=price_days, prices=prices, day=current_day, days=7
        )
        if (
            forecast_rv is None
            or recent_rv is None
            or dvol_percentile is None
            or dvol_change_3d is None
            or drawdown_7d is None
        ):
            _count_skip(diagnostics, "insufficient_history_for_sizing")
            continue
        implied_vol = dvol_by_day[current_day]
        for spec in HARDCAP_VOLTARGET_SPECS:
            expiry_day = current_day + spec.tenor_days * DAY_MS
            future_returns = future_log_returns(
                price_days=price_days,
                prices=prices,
                start_day=current_day,
                end_day=expiry_day,
            )
            if len(future_returns) < spec.tenor_days:
                _count_skip(diagnostics, "insufficient_future_price_window")
                continue
            if not _timing_pass(
                spec=spec,
                implied_vol=implied_vol,
                forecast_rv=forecast_rv,
                dvol_percentile=dvol_percentile,
                recent_rv=recent_rv,
                dvol_change_3d=dvol_change_3d,
                drawdown_7d=drawdown_7d,
            ):
                _count_skip(diagnostics, f"timing_filter_{spec.name}")
                continue
            realized_vol = annualized_vol(future_returns)
            gross_credit = (implied_vol**2) * (spec.tenor_days / 365.0)
            hard_cap = max(gross_credit * (spec.cap_multiple - 1.0), gross_credit * 0.10)
            insurance_cost = gross_credit / (spec.cap_multiple * 2.5)
            rebalances = len(future_returns)
            hedge_fee = rebalances * 0.0004 * 0.20
            hedge_slippage = rebalances * 0.0003 * 0.20
            hedge_funding = funding_cost(
                funding=funding,
                start_day=current_day,
                end_day=expiry_day,
                hedge_notional=0.20,
            )
            raw_unscaled = (
                (implied_vol**2 - realized_vol**2) * (spec.tenor_days / 365.0)
                - (0.004 + insurance_cost)
                - hedge_fee
                - hedge_slippage
                - hedge_funding
            )
            capped_unscaled = max(raw_unscaled, -hard_cap)
            if capped_unscaled != raw_unscaled:
                diagnostics["hard_cap_bindings"] = int(diagnostics["hard_cap_bindings"]) + 1
            position_scale = min(1.0, spec.target_vol / max(forecast_rv, 0.01))
            diagnostics["max_position_scale"] = max(
                float(diagnostics["max_position_scale"]), position_scale
            )
            net_return = capped_unscaled * position_scale
            scaled_cap = hard_cap * position_scale
            if net_return < -scaled_cap - 1e-12:
                diagnostics["hard_cap_violations"] = int(diagnostics["hard_cap_violations"]) + 1
            rows.append(
                {
                    "variant": spec.name,
                    "iv_ts": current_day,
                    "expiry_ts": expiry_day,
                    "implied_vol": implied_vol,
                    "realized_vol": realized_vol,
                    "variance_year_fraction": spec.tenor_days / 365.0,
                    "option_spread_cost": (0.004 + insurance_cost) * position_scale,
                    "hedge_fee_cost": hedge_fee * position_scale,
                    "hedge_slippage_cost": hedge_slippage * position_scale,
                    "funding_cost": hedge_funding * position_scale,
                    "tail_loss": 0.0,
                    "return_floor": scaled_cap,
                    "net_return_override": net_return,
                    "forecast_rv": forecast_rv,
                    "position_scale": position_scale,
                    "hard_cap_unscaled": hard_cap,
                    "hard_cap_scaled": scaled_cap,
                    "raw_unscaled_return": raw_unscaled,
                    "capped_unscaled_return": capped_unscaled,
                }
            )
            counts = diagnostics["trade_counts_by_variant"]
            if isinstance(counts, dict):
                counts[spec.name] = int(counts[spec.name]) + 1
    return rows, diagnostics


def _timing_pass(
    *,
    spec: HardCapVolTargetSpec,
    implied_vol: float,
    forecast_rv: float,
    dvol_percentile: float,
    recent_rv: float,
    dvol_change_3d: float,
    drawdown_7d: float,
) -> bool:
    if recent_rv > forecast_rv * 1.50:
        return False
    if dvol_change_3d > 0.35:
        return False
    if drawdown_7d < -0.20:
        return False
    if spec.timing_rule == "iv_edge":
        return (
            spec.iv_edge_threshold is not None
            and implied_vol - forecast_rv > spec.iv_edge_threshold
        )
    if spec.timing_rule == "dvol_percentile":
        return (
            spec.dvol_percentile_threshold is not None
            and dvol_percentile >= spec.dvol_percentile_threshold
        )
    return False


def _day_start(timestamp_ms: int) -> int:
    return timestamp_ms - (timestamp_ms % DAY_MS)


def _count_skip(diagnostics: dict[str, Any], reason: str) -> None:
    skipped = diagnostics["skipped_reasons"]
    if isinstance(skipped, dict):
        skipped[reason] = int(skipped.get(reason, 0)) + 1
