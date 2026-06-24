from __future__ import annotations

from aegis.btc_vrp_condor_timed import (
    DAY_MS,
    build_timed_condor_rows,
    forecast_rv_ewma,
    timed_condor_variant_names,
)


def test_forecast_rv_uses_only_current_and_prior_prices() -> None:
    base = 1_704_067_200_000
    price_days = tuple(base + index * DAY_MS for index in range(45))
    prices = {day: 100.0 + index for index, day in enumerate(price_days)}
    day = price_days[35]
    original = forecast_rv_ewma(price_days=price_days, prices=prices, day=day)
    mutated = dict(prices)
    for future_day in price_days[36:]:
        mutated[future_day] = mutated[future_day] * 5.0
    changed_future = forecast_rv_ewma(price_days=price_days, prices=mutated, day=day)
    assert original == changed_future


def test_timed_condor_builder_applies_timing_and_cap_fields() -> None:
    base = 1_704_067_200_000
    price_days = tuple(base + index * DAY_MS for index in range(260))
    prices = {day: 100.0 + index * 0.1 for index, day in enumerate(price_days)}
    dvol = [(day, 85.0 if index >= 190 else 45.0) for index, day in enumerate(price_days[:-20])]
    funding = {day: 0.0001 for day in price_days}
    rows, diagnostics = build_timed_condor_rows(dvol=dvol, prices=prices, funding=funding)
    assert rows
    assert set(timed_condor_variant_names()) == set(diagnostics["trade_counts_by_variant"])
    assert all("return_floor" in row for row in rows)
    assert all(
        isinstance(row["iv_ts"], int)
        and isinstance(row["expiry_ts"], int)
        and row["iv_ts"] < row["expiry_ts"]
        for row in rows
    )
