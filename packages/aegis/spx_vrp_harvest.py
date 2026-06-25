from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Literal

DAY_MS = 24 * 60 * 60 * 1000
TRADING_DAYS_PER_YEAR = 252.0

TimingRule = Literal["iv_edge", "vix_percentile"]


@dataclass(frozen=True)
class SpxDailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class SpxVrpSpec:
    name: str
    tenor_days: int
    max_loss_cap: float
    target_vol: float
    timing_rule: TimingRule
    iv_edge_threshold: float | None = None
    vix_percentile_threshold: float | None = None


SPX_VRP_SPECS: tuple[SpxVrpSpec, ...] = (
    SpxVrpSpec("spx_vrp_21d_cap10_tv10_edge03", 21, 0.10, 0.10, "iv_edge", 0.03),
    SpxVrpSpec("spx_vrp_21d_cap10_tv10_pct75", 21, 0.10, 0.10, "vix_percentile", None, 0.75),
    SpxVrpSpec("spx_vrp_21d_cap10_tv15_edge03", 21, 0.10, 0.15, "iv_edge", 0.03),
    SpxVrpSpec("spx_vrp_21d_cap10_tv15_pct75", 21, 0.10, 0.15, "vix_percentile", None, 0.75),
    SpxVrpSpec("spx_vrp_21d_cap15_tv10_edge06", 21, 0.15, 0.10, "iv_edge", 0.06),
    SpxVrpSpec("spx_vrp_21d_cap15_tv10_pct80", 21, 0.15, 0.10, "vix_percentile", None, 0.80),
    SpxVrpSpec("spx_vrp_21d_cap15_tv15_edge06", 21, 0.15, 0.15, "iv_edge", 0.06),
    SpxVrpSpec("spx_vrp_21d_cap15_tv15_pct80", 21, 0.15, 0.15, "vix_percentile", None, 0.80),
    SpxVrpSpec("spx_vrp_42d_cap10_tv10_edge03", 42, 0.10, 0.10, "iv_edge", 0.03),
    SpxVrpSpec("spx_vrp_42d_cap10_tv10_pct75", 42, 0.10, 0.10, "vix_percentile", None, 0.75),
    SpxVrpSpec("spx_vrp_42d_cap10_tv15_edge03", 42, 0.10, 0.15, "iv_edge", 0.03),
    SpxVrpSpec("spx_vrp_42d_cap10_tv15_pct75", 42, 0.10, 0.15, "vix_percentile", None, 0.75),
    SpxVrpSpec("spx_vrp_42d_cap15_tv10_edge06", 42, 0.15, 0.10, "iv_edge", 0.06),
    SpxVrpSpec("spx_vrp_42d_cap15_tv10_pct80", 42, 0.15, 0.10, "vix_percentile", None, 0.80),
    SpxVrpSpec("spx_vrp_42d_cap15_tv15_edge06", 42, 0.15, 0.15, "iv_edge", 0.06),
    SpxVrpSpec("spx_vrp_42d_cap15_tv15_pct80", 42, 0.15, 0.15, "vix_percentile", None, 0.80),
)

REQUIRED_SPX_CRASH_WINDOWS: Mapping[str, tuple[date, date]] = {
    "dotcom_2000_2002": (date(2000, 3, 1), date(2002, 10, 31)),
    "gfc_2008": (date(2008, 9, 1), date(2009, 3, 31)),
    "volmageddon_2018": (date(2018, 2, 1), date(2018, 2, 28)),
    "covid_2020": (date(2020, 2, 15), date(2020, 4, 15)),
}


def spx_vrp_variant_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in SPX_VRP_SPECS)


def build_spx_vrp_rows(
    *,
    vix: Mapping[date, float],
    spx: Mapping[date, SpxDailyBar],
) -> tuple[list[Mapping[str, object]], Mapping[str, Any]]:
    rows: list[Mapping[str, object]] = []
    ordered_days = sorted(set(vix) & set(spx))
    diagnostics: dict[str, Any] = {
        "predeclared_configs": list(spx_vrp_variant_names()),
        "trade_counts_by_variant": {name: 0 for name in spx_vrp_variant_names()},
        "skipped_reasons": {},
        "hard_cap_bindings": 0,
        "hard_cap_violations": 0,
        "max_position_scale": 0.0,
    }
    for index, current_day in enumerate(ordered_days):
        forecast_rv = forecast_realized_vol(spx=spx, ordered_days=ordered_days, index=index)
        recent_rv = trailing_realized_vol(spx=spx, ordered_days=ordered_days, index=index, days=10)
        vix_percentile = trailing_percentile(
            values_by_day=vix,
            ordered_days=ordered_days,
            index=index,
            lookback=252,
        )
        vix_change_3d = trailing_change(values_by_day=vix, ordered_days=ordered_days, index=index)
        drawdown_10d = trailing_drawdown(spx=spx, ordered_days=ordered_days, index=index, days=10)
        if (
            forecast_rv is None
            or recent_rv is None
            or vix_percentile is None
            or vix_change_3d is None
            or drawdown_10d is None
        ):
            _count_skip(diagnostics, "insufficient_history")
            continue
        implied_vol = vix[current_day] / 100.0
        for spec in SPX_VRP_SPECS:
            expiry_index = index + spec.tenor_days
            if expiry_index >= len(ordered_days):
                _count_skip(diagnostics, "insufficient_future_window")
                continue
            if not timing_pass(
                spec=spec,
                implied_vol=implied_vol,
                forecast_rv=forecast_rv,
                recent_rv=recent_rv,
                vix_percentile=vix_percentile,
                vix_change_3d=vix_change_3d,
                drawdown_10d=drawdown_10d,
            ):
                _count_skip(diagnostics, f"timing_filter_{spec.name}")
                continue
            future_returns = close_to_close_returns(
                spx=spx,
                ordered_days=ordered_days,
                start_index=index,
                tenor_days=spec.tenor_days,
            )
            if len(future_returns) < spec.tenor_days:
                _count_skip(diagnostics, "future_return_gap")
                continue
            realized_vol = annualized_vol(future_returns)
            variance_year_fraction = spec.tenor_days / TRADING_DAYS_PER_YEAR
            insurance_cost = min(0.0030, spec.max_loss_cap * 0.08)
            option_spread_cost = 0.0025 + insurance_cost
            hedge_fee_cost = spec.tenor_days * 0.00002 * 0.15
            hedge_slippage_cost = spec.tenor_days * 0.00005 * 0.15
            raw_unscaled = (
                (implied_vol * implied_vol - realized_vol * realized_vol)
                * variance_year_fraction
                - option_spread_cost
                - hedge_fee_cost
                - hedge_slippage_cost
            )
            capped_unscaled = max(raw_unscaled, -spec.max_loss_cap)
            if capped_unscaled != raw_unscaled:
                diagnostics["hard_cap_bindings"] = int(diagnostics["hard_cap_bindings"]) + 1
            position_scale = min(1.0, spec.target_vol / max(forecast_rv, 0.01))
            diagnostics["max_position_scale"] = max(
                float(diagnostics["max_position_scale"]), position_scale
            )
            net_return = capped_unscaled * position_scale
            scaled_cap = spec.max_loss_cap * position_scale
            if net_return < -scaled_cap - 1e-12:
                diagnostics["hard_cap_violations"] = int(diagnostics["hard_cap_violations"]) + 1
            rows.append(
                {
                    "variant": spec.name,
                    "iv_ts": date_to_ms(current_day),
                    "expiry_ts": date_to_ms(ordered_days[expiry_index]),
                    "implied_vol": implied_vol,
                    "realized_vol": realized_vol,
                    "variance_year_fraction": variance_year_fraction,
                    "option_spread_cost": option_spread_cost * position_scale,
                    "hedge_fee_cost": hedge_fee_cost * position_scale,
                    "hedge_slippage_cost": hedge_slippage_cost * position_scale,
                    "funding_cost": 0.0,
                    "tail_loss": 0.0,
                    "return_floor": scaled_cap,
                    "net_return_override": net_return,
                    "forecast_rv": forecast_rv,
                    "recent_rv": recent_rv,
                    "vix_percentile": vix_percentile,
                    "vix_change_3d": vix_change_3d,
                    "drawdown_10d": drawdown_10d,
                    "position_scale": position_scale,
                    "hard_cap_scaled": scaled_cap,
                    "raw_unscaled_return": raw_unscaled,
                    "capped_unscaled_return": capped_unscaled,
                }
            )
            counts = diagnostics["trade_counts_by_variant"]
            if isinstance(counts, dict):
                counts[spec.name] = int(counts[spec.name]) + 1
    return rows, diagnostics


def build_always_short_rows(
    *,
    vix: Mapping[date, float],
    spx: Mapping[date, SpxDailyBar],
    tenor_days: int = 21,
    max_loss_cap: float = 0.15,
    target_vol: float = 0.10,
) -> tuple[list[Mapping[str, object]], Mapping[str, Any]]:
    rows: list[Mapping[str, object]] = []
    ordered_days = sorted(set(vix) & set(spx))
    diagnostics = {"hard_cap_violations": 0, "hard_cap_bindings": 0}
    for index, current_day in enumerate(ordered_days):
        forecast_rv = forecast_realized_vol(spx=spx, ordered_days=ordered_days, index=index)
        if forecast_rv is None:
            continue
        expiry_index = index + tenor_days
        if expiry_index >= len(ordered_days):
            continue
        future_returns = close_to_close_returns(
            spx=spx,
            ordered_days=ordered_days,
            start_index=index,
            tenor_days=tenor_days,
        )
        realized_vol = annualized_vol(future_returns)
        implied_vol = vix[current_day] / 100.0
        year_fraction = tenor_days / TRADING_DAYS_PER_YEAR
        raw_unscaled = (
            (implied_vol * implied_vol - realized_vol * realized_vol) * year_fraction
            - 0.0035
            - tenor_days * 0.00007 * 0.15
        )
        capped_unscaled = max(raw_unscaled, -max_loss_cap)
        if capped_unscaled != raw_unscaled:
            diagnostics["hard_cap_bindings"] = int(diagnostics["hard_cap_bindings"]) + 1
        position_scale = min(1.0, target_vol / max(forecast_rv, 0.01))
        net_return = capped_unscaled * position_scale
        scaled_cap = max_loss_cap * position_scale
        if net_return < -scaled_cap - 1e-12:
            diagnostics["hard_cap_violations"] = int(diagnostics["hard_cap_violations"]) + 1
        rows.append(
            {
                "variant": "always_short_vrp_21d_cap15_tv10",
                "iv_ts": date_to_ms(current_day),
                "expiry_ts": date_to_ms(ordered_days[expiry_index]),
                "implied_vol": implied_vol,
                "realized_vol": realized_vol,
                "variance_year_fraction": year_fraction,
                "option_spread_cost": 0.0035 * position_scale,
                "hedge_fee_cost": tenor_days * 0.00002 * 0.15 * position_scale,
                "hedge_slippage_cost": tenor_days * 0.00005 * 0.15 * position_scale,
                "funding_cost": 0.0,
                "tail_loss": 0.0,
                "return_floor": scaled_cap,
                "net_return_override": net_return,
            }
        )
    return rows, diagnostics


def locked_oos_variant_report(
    rows: Sequence[Mapping[str, object]],
    *,
    split_fraction: float = 0.70,
) -> Mapping[str, Any]:
    times = sorted({value for row in rows if (value := _int(row.get("iv_ts"))) is not None})
    if len(times) < 4:
        return {"valid": False, "reason": "insufficient timestamps for locked OOS"}
    split_index = min(max(1, int(len(times) * split_fraction)), len(times) - 1)
    split_ts = times[split_index]
    by_variant = sorted({str(row.get("variant")) for row in rows if row.get("variant")})
    is_reports: list[tuple[str, float, int]] = []
    for variant in by_variant:
        is_returns = [
            _float(row.get("net_return_override"))
            for row in rows
            if row.get("variant") == variant and (_int(row.get("iv_ts")) or 0) < split_ts
        ]
        clean_is = [value for value in is_returns if value is not None]
        if clean_is:
            is_reports.append((variant, statistics.fmean(clean_is), len(clean_is)))
    if not is_reports:
        return {"valid": False, "reason": "no IS returns"}
    selected, is_mean, is_count = max(is_reports, key=lambda item: item[1])
    oos_returns = [
        _float(row.get("net_return_override"))
        for row in rows
        if row.get("variant") == selected and (_int(row.get("iv_ts")) or 0) >= split_ts
    ]
    clean_oos = [value for value in oos_returns if value is not None]
    if not clean_oos:
        return {"valid": False, "reason": "no OOS returns", "selected_variant": selected}
    return {
        "valid": True,
        "split_ts": split_ts,
        "selected_variant": selected,
        "is_trade_count": is_count,
        "is_mean_net_return": is_mean,
        "oos_trade_count": len(clean_oos),
        "oos_mean_net_return": statistics.fmean(clean_oos),
        "oos_total_net_return": sum(clean_oos),
        "oos_win_rate": sum(1 for value in clean_oos if value > 0.0) / len(clean_oos),
    }


def timing_pass(
    *,
    spec: SpxVrpSpec,
    implied_vol: float,
    forecast_rv: float,
    recent_rv: float,
    vix_percentile: float,
    vix_change_3d: float,
    drawdown_10d: float,
) -> bool:
    if recent_rv > forecast_rv * 1.75:
        return False
    if vix_change_3d > 0.30:
        return False
    if drawdown_10d < -0.08:
        return False
    if spec.timing_rule == "iv_edge":
        return (
            spec.iv_edge_threshold is not None
            and implied_vol - forecast_rv > spec.iv_edge_threshold
        )
    return (
        spec.vix_percentile_threshold is not None
        and vix_percentile >= spec.vix_percentile_threshold
    )


def crash_window_coverage(
    *,
    vix: Mapping[date, float],
    spx: Mapping[date, SpxDailyBar],
) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for name, (start, end) in REQUIRED_SPX_CRASH_WINDOWS.items():
        vix_days = [day for day in vix if start <= day <= end]
        spx_days = [day for day in spx if start <= day <= end]
        result[name] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "vix_rows": len(vix_days),
            "spx_rows": len(spx_days),
            "covered": bool(vix_days and spx_days),
        }
    return result


def forecast_realized_vol(
    *,
    spx: Mapping[date, SpxDailyBar],
    ordered_days: Sequence[date],
    index: int,
    lookback: int = 60,
) -> float | None:
    returns = trailing_returns(spx=spx, ordered_days=ordered_days, index=index, days=lookback)
    if len(returns) < max(20, lookback // 2):
        return None
    weights = [0.94 ** offset for offset in range(len(returns) - 1, -1, -1)]
    weighted_var = sum(
        weight * value * value for weight, value in zip(weights, returns, strict=True)
    ) / sum(weights)
    return math.sqrt(weighted_var * TRADING_DAYS_PER_YEAR)


def trailing_realized_vol(
    *,
    spx: Mapping[date, SpxDailyBar],
    ordered_days: Sequence[date],
    index: int,
    days: int,
) -> float | None:
    returns = trailing_returns(spx=spx, ordered_days=ordered_days, index=index, days=days)
    if len(returns) < days:
        return None
    return annualized_vol(returns)


def trailing_returns(
    *,
    spx: Mapping[date, SpxDailyBar],
    ordered_days: Sequence[date],
    index: int,
    days: int,
) -> list[float]:
    start = max(1, index - days + 1)
    returns: list[float] = []
    for pos in range(start, index + 1):
        prev_close = spx[ordered_days[pos - 1]].close
        current_close = spx[ordered_days[pos]].close
        if prev_close > 0.0 and current_close > 0.0:
            returns.append(math.log(current_close / prev_close))
    return returns


def close_to_close_returns(
    *,
    spx: Mapping[date, SpxDailyBar],
    ordered_days: Sequence[date],
    start_index: int,
    tenor_days: int,
) -> list[float]:
    returns: list[float] = []
    for pos in range(start_index + 1, min(len(ordered_days), start_index + tenor_days + 1)):
        prev_close = spx[ordered_days[pos - 1]].close
        current_close = spx[ordered_days[pos]].close
        if prev_close > 0.0 and current_close > 0.0:
            returns.append(math.log(current_close / prev_close))
    return returns


def annualized_vol(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    return math.sqrt(sum(value * value for value in returns) * TRADING_DAYS_PER_YEAR / len(returns))


def garman_klass_vol(bars: Sequence[SpxDailyBar]) -> float | None:
    if not bars:
        return None
    estimates: list[float] = []
    for bar in bars:
        if min(bar.open, bar.high, bar.low, bar.close) <= 0.0:
            continue
        high_low = math.log(bar.high / bar.low)
        close_open = math.log(bar.close / bar.open)
        estimates.append(0.5 * high_low * high_low - (2.0 * math.log(2.0) - 1.0) * close_open**2)
    if not estimates:
        return None
    return math.sqrt(max(statistics.fmean(estimates), 0.0) * TRADING_DAYS_PER_YEAR)


def trailing_percentile(
    *,
    values_by_day: Mapping[date, float],
    ordered_days: Sequence[date],
    index: int,
    lookback: int,
) -> float | None:
    current = values_by_day[ordered_days[index]]
    history = [
        values_by_day[ordered_days[pos]]
        for pos in range(max(0, index - lookback), index)
        if ordered_days[pos] in values_by_day
    ]
    if len(history) < max(30, lookback // 3):
        return None
    return sum(1 for value in history if value <= current) / len(history)


def trailing_change(
    *,
    values_by_day: Mapping[date, float],
    ordered_days: Sequence[date],
    index: int,
    lag: int = 3,
) -> float | None:
    if index < lag:
        return None
    current = values_by_day[ordered_days[index]]
    previous = values_by_day[ordered_days[index - lag]]
    if previous <= 0.0:
        return None
    return current / previous - 1.0


def trailing_drawdown(
    *,
    spx: Mapping[date, SpxDailyBar],
    ordered_days: Sequence[date],
    index: int,
    days: int,
) -> float | None:
    if index < days:
        return None
    closes = [spx[ordered_days[pos]].close for pos in range(index - days, index + 1)]
    peak = max(closes)
    if peak <= 0.0:
        return None
    return closes[-1] / peak - 1.0


def date_to_ms(value: date) -> int:
    return int(
        datetime.combine(
            value,
            time.min,
            tzinfo=timezone.utc,  # noqa: UP017 - host evidence runner uses Python 3.10.
        ).timestamp()
        * 1000
    )


def _count_skip(diagnostics: dict[str, Any], reason: str) -> None:
    skipped = diagnostics["skipped_reasons"]
    if isinstance(skipped, dict):
        skipped[reason] = int(skipped.get(reason, 0)) + 1


def _float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    return None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None
