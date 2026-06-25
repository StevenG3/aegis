from __future__ import annotations

from datetime import date, timedelta
from typing import cast

from aegis.btc_vrp_short_vol import ShortVolVrpConfig, run_btc_short_vol_vrp
from aegis.spx_vrp_harvest import (
    SpxDailyBar,
    build_spx_vrp_deployment_rows,
    build_spx_vrp_rows,
    date_to_ms,
    gross_vrp_self_check,
    locked_oos_variant_report,
    spx_vrp_risk_curve,
    spx_vrp_variant_names,
)


def _bars(*, shock_after: int | None = None, high_vol: bool = False) -> dict[date, SpxDailyBar]:
    start = date(2020, 1, 1)
    result: dict[date, SpxDailyBar] = {}
    close = 100.0
    for index in range(380):
        day = start + timedelta(days=index)
        if shock_after is not None and index == shock_after:
            close *= 0.65
        elif high_vol:
            close *= 1.0 + (0.02 if index % 2 == 0 else -0.019)
        else:
            close *= 1.0005
        result[day] = SpxDailyBar(day, close * 0.999, close * 1.002, close * 0.998, close)
    return result


def _vix(value: float = 35.0) -> dict[date, float]:
    start = date(2020, 1, 1)
    return {start + timedelta(days=index): value for index in range(380)}


def test_spx_vrp_no_lookahead_forecast_ignores_future_crash() -> None:
    spx = _bars(shock_after=300)
    rows, _ = build_spx_vrp_rows(vix=_vix(35.0), spx=spx)
    crash_day = date(2020, 1, 1) + timedelta(days=300)
    pre_crash_rows = [
        row for row in rows if cast(int, row["iv_ts"]) < date_to_ms(crash_day)
    ]

    assert pre_crash_rows
    assert max(cast(float, row["forecast_rv"]) for row in pre_crash_rows) < 0.05


def test_spx_vrp_hard_cap_has_no_single_trade_cap_violation() -> None:
    rows, diagnostics = build_spx_vrp_rows(vix=_vix(80.0), spx=_bars(shock_after=300))

    assert rows
    assert diagnostics["hard_cap_violations"] == 0
    assert set(spx_vrp_variant_names()) == set(diagnostics["trade_counts_by_variant"])
    assert all(
        isinstance(row["net_return_override"], float)
        and isinstance(row["hard_cap_scaled"], float)
        and row["net_return_override"] >= -row["hard_cap_scaled"] - 1e-12
        for row in rows
    )


def test_spx_vrp_sizing_reduces_position_when_forecast_vol_is_high() -> None:
    low_rows, _ = build_spx_vrp_rows(vix=_vix(45.0), spx=_bars(high_vol=False))
    high_rows, _ = build_spx_vrp_rows(vix=_vix(80.0), spx=_bars(high_vol=True))

    low_scales = [cast(float, row["position_scale"]) for row in low_rows]
    high_scales = [cast(float, row["position_scale"]) for row in high_rows]
    assert low_scales and high_scales
    assert min(high_scales) < min(low_scales)


def test_spx_vrp_costs_reduce_net_return() -> None:
    base = {
        "variant": "costed",
        "iv_ts": 1_000,
        "expiry_ts": 2_000,
        "implied_vol": 0.30,
        "realized_vol": 0.10,
        "variance_year_fraction": 21.0 / 252.0,
        "tail_loss": 0.0,
    }
    cheap = run_btc_short_vol_vrp(
        [
            base
            | {
                "option_spread_cost": 0.0,
                "hedge_fee_cost": 0.0,
                "hedge_slippage_cost": 0.0,
                "funding_cost": 0.0,
            }
        ],
        config=ShortVolVrpConfig(),
    )
    expensive = run_btc_short_vol_vrp(
        [
            base
            | {
                "option_spread_cost": 0.01,
                "hedge_fee_cost": 0.002,
                "hedge_slippage_cost": 0.003,
                "funding_cost": 0.0,
            }
        ],
        config=ShortVolVrpConfig(),
    )

    assert expensive["best_candidate"]["mean_net_return"] < cheap["best_candidate"][
        "mean_net_return"
    ]


def test_spx_vrp_costs_scale_with_premium_not_absolute_constant() -> None:
    low_rows, _ = build_spx_vrp_rows(vix=_vix(25.0), spx=_bars(high_vol=False))
    high_rows, _ = build_spx_vrp_rows(vix=_vix(50.0), spx=_bars(high_vol=False))

    low = low_rows[0]
    high = high_rows[0]

    assert cast(float, low["premium_credit"]) > 0.0
    assert cast(float, high["premium_credit"]) > cast(float, low["premium_credit"])
    assert cast(float, high["option_spread_cost"]) > cast(
        float, low["option_spread_cost"]
    )
    assert cast(float, low["option_spread_cost"]) / cast(
        float, low["premium_credit"]
    ) == cast(float, high["option_spread_cost"]) / cast(float, high["premium_credit"])


def test_spx_vrp_gross_vrp_self_check_requires_positive_gross_mean() -> None:
    positive = gross_vrp_self_check(
        [
            {"gross_vrp_return": 0.01},
            {"gross_vrp_return": 0.02},
            {"gross_vrp_return": -0.005},
        ]
    )
    negative = gross_vrp_self_check(
        [{"gross_vrp_return": -0.01}, {"gross_vrp_return": -0.02}]
    )

    assert positive["valid"] is True
    assert negative["valid"] is False


def test_spx_vrp_uses_non_overlapping_exposure_per_variant() -> None:
    rows, diagnostics = build_spx_vrp_rows(vix=_vix(50.0), spx=_bars(high_vol=False))

    assert diagnostics["exposure_policy"] == "non_overlapping_per_variant"
    by_variant: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(
            (cast(int, row["iv_ts"]), cast(int, row["expiry_ts"]))
        )
    assert by_variant
    for trades in by_variant.values():
        previous_expiry = -1
        for entry, expiry in sorted(trades):
            assert entry > previous_expiry
            previous_expiry = expiry


def test_spx_vrp_deployment_respects_exposure_cap() -> None:
    rows, diagnostics = build_spx_vrp_deployment_rows(
        vix=_vix(50.0), spx=_bars(high_vol=False)
    )

    assert rows
    assert diagnostics["portfolio_exposure_policy"]
    assert all(
        cast(float, row["deployment_exposure"]) <= cast(float, row["max_exposure"])
        for row in rows
    )


def test_spx_vrp_deployment_scales_return_and_cost_components() -> None:
    base_rows, _ = build_spx_vrp_rows(vix=_vix(50.0), spx=_bars(high_vol=False))
    deployed_rows, _ = build_spx_vrp_deployment_rows(
        vix=_vix(50.0), spx=_bars(high_vol=False)
    )
    base = base_rows[0]
    deployed = next(
        row
        for row in deployed_rows
        if row["base_variant"] == base["variant"] and row["iv_ts"] == base["iv_ts"]
    )
    exposure = cast(float, deployed["deployment_exposure"])

    assert cast(float, deployed["net_return_override"]) == cast(
        float, base["net_return_override"]
    ) * exposure
    assert cast(float, deployed["total_cost"]) == cast(float, base["total_cost"]) * exposure


def test_spx_vrp_risk_curve_reports_annualized_metrics_by_risk_tier() -> None:
    rows, _ = build_spx_vrp_deployment_rows(vix=_vix(50.0), spx=_bars(high_vol=False))
    curve = spx_vrp_risk_curve(rows)

    assert {"risk08_cap4x", "risk12_cap6x", "risk16_cap8x"} <= set(curve)
    assert all(cast(dict[str, object], metrics)["valid"] is True for metrics in curve.values())
    assert all(
        "annualized_return" in cast(dict[str, object], metrics)
        for metrics in curve.values()
    )


def test_locked_oos_variant_report_selects_on_is_then_reports_oos() -> None:
    rows = []
    base_ts = 1_000
    for index in range(10):
        rows.append({"variant": "a", "iv_ts": base_ts + index, "net_return_override": 0.02})
        rows.append({"variant": "b", "iv_ts": base_ts + index, "net_return_override": -0.01})
    for index in range(10, 14):
        rows.append({"variant": "a", "iv_ts": base_ts + index, "net_return_override": -0.03})
        rows.append({"variant": "b", "iv_ts": base_ts + index, "net_return_override": 0.04})

    report = locked_oos_variant_report(rows, split_fraction=0.70)

    assert report["valid"] is True
    assert report["selected_variant"] == "a"
    assert report["oos_mean_net_return"] < 0.0
