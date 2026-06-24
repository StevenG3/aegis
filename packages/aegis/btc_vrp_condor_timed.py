from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DAY_MS = 24 * 3600 * 1000


@dataclass(frozen=True)
class TimedCondorSpec:
    name: str
    tenor_days: int
    wing_multiple: float
    timing_rule: str
    iv_edge_threshold: float | None = None
    dvol_percentile_threshold: float | None = None


TIMED_CONDOR_SPECS: tuple[TimedCondorSpec, ...] = (
    TimedCondorSpec("condor_7d_w2_edge10", 7, 2.0, "iv_edge", iv_edge_threshold=0.10),
    TimedCondorSpec("condor_7d_w2_edge20", 7, 2.0, "iv_edge", iv_edge_threshold=0.20),
    TimedCondorSpec(
        "condor_7d_w2_pct75", 7, 2.0, "dvol_percentile", dvol_percentile_threshold=0.75
    ),
    TimedCondorSpec("condor_7d_w3_edge10", 7, 3.0, "iv_edge", iv_edge_threshold=0.10),
    TimedCondorSpec("condor_7d_w3_edge20", 7, 3.0, "iv_edge", iv_edge_threshold=0.20),
    TimedCondorSpec(
        "condor_7d_w3_pct75", 7, 3.0, "dvol_percentile", dvol_percentile_threshold=0.75
    ),
    TimedCondorSpec("condor_14d_w2_edge10", 14, 2.0, "iv_edge", iv_edge_threshold=0.10),
    TimedCondorSpec("condor_14d_w2_edge20", 14, 2.0, "iv_edge", iv_edge_threshold=0.20),
    TimedCondorSpec(
        "condor_14d_w2_pct75", 14, 2.0, "dvol_percentile", dvol_percentile_threshold=0.75
    ),
    TimedCondorSpec("condor_14d_w3_edge10", 14, 3.0, "iv_edge", iv_edge_threshold=0.10),
    TimedCondorSpec("condor_14d_w3_edge20", 14, 3.0, "iv_edge", iv_edge_threshold=0.20),
    TimedCondorSpec(
        "condor_14d_w3_pct75", 14, 3.0, "dvol_percentile", dvol_percentile_threshold=0.75
    ),
)


def timed_condor_variant_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in TIMED_CONDOR_SPECS)


def build_timed_condor_rows(
    *,
    dvol: Sequence[tuple[int, float]],
    prices: Mapping[int, float],
    funding: Mapping[int, float],
) -> tuple[list[Mapping[str, object]], Mapping[str, Any]]:
    rows: list[Mapping[str, object]] = []
    diagnostics: dict[str, Any] = {
        "predeclared_timed_configs": timed_condor_variant_names(),
        "skipped_reasons": {},
        "trade_counts_by_variant": {name: 0 for name in timed_condor_variant_names()},
        "max_loss_cap_breaches": 0,
    }
    dvol_by_day = {_day_start(timestamp): close / 100.0 for timestamp, close in dvol}
    dvol_days = sorted(dvol_by_day)
    price_days = sorted(prices)
    for current_day in dvol_days:
        price_index = _index_of(price_days, current_day)
        if price_index is None:
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
            _count_skip(diagnostics, "insufficient_history_for_timing")
            continue
        implied_vol = dvol_by_day[current_day]
        for spec in TIMED_CONDOR_SPECS:
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
            max_abs_daily = max(abs(value) for value in future_returns)
            gross_credit = (implied_vol**2) * (spec.tenor_days / 365.0)
            insurance_cost = gross_credit / (spec.wing_multiple * 4.0)
            return_floor = gross_credit * spec.wing_multiple
            uncapped_tail_loss = max(0.0, max_abs_daily - 0.18) * 1.0
            capped_tail_loss = min(uncapped_tail_loss, return_floor)
            if uncapped_tail_loss > capped_tail_loss:
                diagnostics["max_loss_cap_breaches"] = int(diagnostics["max_loss_cap_breaches"]) + 1
            rebalances = len(future_returns)
            hedge_fee = rebalances * 0.0004 * 0.20
            hedge_slippage = rebalances * 0.0003 * 0.20
            hedge_funding = funding_cost(
                funding=funding,
                start_day=current_day,
                end_day=expiry_day,
                hedge_notional=0.20,
            )
            raw_net = (
                (implied_vol**2 - realized_vol**2) * (spec.tenor_days / 365.0)
                - (0.006 + insurance_cost)
                - hedge_fee
                - hedge_slippage
                - hedge_funding
                - capped_tail_loss
            )
            if raw_net < -return_floor:
                diagnostics["max_loss_cap_breaches"] = int(diagnostics["max_loss_cap_breaches"]) + 1
            row = {
                "variant": spec.name,
                "iv_ts": current_day,
                "expiry_ts": expiry_day,
                "implied_vol": implied_vol,
                "realized_vol": realized_vol,
                "variance_year_fraction": spec.tenor_days / 365.0,
                "option_spread_cost": 0.006 + insurance_cost,
                "hedge_fee_cost": hedge_fee,
                "hedge_slippage_cost": hedge_slippage,
                "funding_cost": hedge_funding,
                "tail_loss": capped_tail_loss,
                "return_floor": return_floor,
                "forecast_rv": forecast_rv,
                "recent_rv": recent_rv,
                "dvol_percentile": dvol_percentile,
                "dvol_change_3d": dvol_change_3d,
                "drawdown_7d": drawdown_7d,
            }
            rows.append(row)
            counts = diagnostics["trade_counts_by_variant"]
            if isinstance(counts, dict):
                counts[spec.name] = int(counts[spec.name]) + 1
    return rows, diagnostics


def forecast_rv_ewma(
    *,
    price_days: Sequence[int],
    prices: Mapping[int, float],
    day: int,
    lookback: int = 30,
    decay: float = 0.94,
) -> float | None:
    index = _index_of(price_days, day)
    if index is None or index < lookback:
        return None
    returns = [
        math.log(prices[price_days[position]] / prices[price_days[position - 1]])
        for position in range(index - lookback + 1, index + 1)
    ]
    variance = 0.0
    weight_sum = 0.0
    weight = 1.0
    for value in reversed(returns):
        variance += weight * value * value
        weight_sum += weight
        weight *= decay
    return math.sqrt((variance / weight_sum) * 365.0) if weight_sum > 0.0 else None


def realized_vol_window(
    *,
    price_days: Sequence[int],
    prices: Mapping[int, float],
    day: int,
    days: int,
) -> float | None:
    index = _index_of(price_days, day)
    if index is None or index < days:
        return None
    returns = [
        math.log(prices[price_days[position]] / prices[price_days[position - 1]])
        for position in range(index - days + 1, index + 1)
    ]
    return annualized_vol(returns)


def future_log_returns(
    *,
    price_days: Sequence[int],
    prices: Mapping[int, float],
    start_day: int,
    end_day: int,
) -> tuple[float, ...]:
    window = [day for day in price_days if start_day <= day <= end_day]
    return tuple(
        math.log(prices[window[index]] / prices[window[index - 1]])
        for index in range(1, len(window))
    )


def annualized_vol(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    return math.sqrt(sum(value * value for value in returns) * 365.0 / len(returns))


def trailing_percentile(
    *,
    values_by_day: Mapping[int, float],
    ordered_days: Sequence[int],
    day: int,
    lookback: int,
) -> float | None:
    index = _index_of(ordered_days, day)
    if index is None or index < lookback:
        return None
    current = values_by_day[day]
    history = [values_by_day[ordered_days[position]] for position in range(index - lookback, index)]
    below_or_equal = sum(1 for value in history if value <= current)
    return below_or_equal / len(history) if history else None


def trailing_change(
    *,
    values_by_day: Mapping[int, float],
    day: int,
    lag_days: int,
) -> float | None:
    previous_day = day - lag_days * DAY_MS
    if previous_day not in values_by_day:
        return None
    previous = values_by_day[previous_day]
    if previous <= 0.0:
        return None
    return values_by_day[day] / previous - 1.0


def trailing_drawdown(
    *,
    price_days: Sequence[int],
    prices: Mapping[int, float],
    day: int,
    days: int,
) -> float | None:
    index = _index_of(price_days, day)
    if index is None or index < days:
        return None
    window = [prices[price_days[position]] for position in range(index - days, index + 1)]
    peak = max(window)
    return prices[day] / peak - 1.0 if peak > 0.0 else None


def funding_cost(
    *,
    funding: Mapping[int, float],
    start_day: int,
    end_day: int,
    hedge_notional: float,
) -> float:
    selected = [abs(rate) for ts, rate in funding.items() if start_day <= ts <= end_day]
    return sum(selected) * hedge_notional


def _timing_pass(
    *,
    spec: TimedCondorSpec,
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


def _index_of(values: Sequence[int], target: int) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None


def _day_start(timestamp_ms: int) -> int:
    return timestamp_ms - (timestamp_ms % DAY_MS)


def _count_skip(diagnostics: dict[str, Any], reason: str) -> None:
    skipped = diagnostics["skipped_reasons"]
    if isinstance(skipped, dict):
        skipped[reason] = int(skipped.get(reason, 0)) + 1
