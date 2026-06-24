from __future__ import annotations

from aegis.btc_vrp_condor_timed import DAY_MS
from aegis.btc_vrp_hardcap_voltarget import (
    build_hardcap_voltarget_rows,
    hardcap_voltarget_variant_names,
)
from aegis.btc_vrp_short_vol import ShortVolVrpConfig, run_btc_short_vol_vrp


def test_hardcap_voltarget_has_no_single_trade_cap_violation() -> None:
    base = 1_704_067_200_000
    price_days = tuple(base + index * DAY_MS for index in range(260))
    prices = {
        day: 100.0 + index * 0.2 if index != 220 else 70.0
        for index, day in enumerate(price_days)
    }
    dvol = [(day, 90.0 if index >= 190 else 45.0) for index, day in enumerate(price_days[:-20])]
    funding = {day: 0.0001 for day in price_days}
    rows, diagnostics = build_hardcap_voltarget_rows(dvol=dvol, prices=prices, funding=funding)
    assert rows
    assert diagnostics["hard_cap_violations"] == 0
    assert set(hardcap_voltarget_variant_names()) == set(diagnostics["trade_counts_by_variant"])
    assert all(
        isinstance(row["net_return_override"], float)
        and isinstance(row["hard_cap_scaled"], float)
        and row["net_return_override"] >= -row["hard_cap_scaled"] - 1e-12
        for row in rows
    )


def test_hardcap_voltarget_sizing_reduces_position_when_forecast_vol_is_high() -> None:
    base = 1_704_067_200_000
    price_days = tuple(base + index * DAY_MS for index in range(260))
    prices = {
        day: 100.0 + ((-1) ** index) * index * 0.05
        for index, day in enumerate(price_days)
    }
    dvol = [(day, 95.0 if index >= 190 else 45.0) for index, day in enumerate(price_days[:-20])]
    funding = {day: 0.0001 for day in price_days}
    rows, _ = build_hardcap_voltarget_rows(dvol=dvol, prices=prices, funding=funding)
    assert rows
    scales = [row["position_scale"] for row in rows if isinstance(row["position_scale"], float)]
    assert scales
    assert max(scales) <= 1.0
    assert min(scales) < 1.0


def test_net_return_override_is_used_by_short_vol_runner() -> None:
    report = run_btc_short_vol_vrp(
        [
            {
                "variant": "override",
                "iv_ts": 1_000,
                "expiry_ts": 2_000,
                "implied_vol": 0.0,
                "realized_vol": 0.0,
                "variance_year_fraction": 1.0,
                "option_spread_cost": 0.0,
                "hedge_fee_cost": 0.0,
                "hedge_slippage_cost": 0.0,
                "funding_cost": 0.0,
                "tail_loss": 0.0,
                "net_return_override": -0.123,
            }
        ],
        config=ShortVolVrpConfig(),
    )
    assert report["best_candidate"]["mean_net_return"] == -0.123
