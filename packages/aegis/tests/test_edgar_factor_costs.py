from __future__ import annotations

from datetime import date
from typing import cast

import pytest

from aegis.edgar_factor_costs import (
    EdgarFactorCostConfig,
    FactorCostProfile,
    run_edgar_factor_cost_diagnostic,
    simulate_factor_portfolio,
)
from aegis.edgar_full_universe_ic import EdgarIcObservation


def _observation(
    symbol: str,
    as_of: date,
    score: float,
    *,
    available_on: date | None = None,
    in_universe: bool = True,
) -> EdgarIcObservation:
    forward = {
        "1m": score * 0.002,
        "3m": score * 0.004,
        "6m": score * 0.006,
    }
    factors = {
        "earnings_yield_ep": score,
        "fcf_yield": score,
        "sales_to_price_sp": score,
    }
    return EdgarIcObservation(
        symbol=symbol,
        as_of=as_of,
        available_on=available_on or as_of,
        factors=factors,
        forward_returns=forward,
        in_universe=in_universe,
    )


def _rows(symbols: int = 40, periods: int = 8) -> list[EdgarIcObservation]:
    rows: list[EdgarIcObservation] = []
    for month in range(1, periods + 1):
        as_of = date(2024, month, 28)
        for index in range(symbols):
            rows.append(_observation(f"S{index:03d}", as_of, float(index - symbols / 2)))
    return rows


def test_factor_cost_report_filters_future_filings_and_non_universe_rows() -> None:
    rows = _rows(symbols=35, periods=6)
    rows.append(
        _observation(
            "FUTURE",
            date(2024, 6, 28),
            999.0,
            available_on=date(2024, 7, 15),
        )
    )
    rows.append(_observation("OUT", date(2024, 6, 28), 999.0, in_universe=False))

    report = run_edgar_factor_cost_diagnostic(
        rows,
        config=EdgarFactorCostConfig(min_cross_section=20),
    )

    coverage = cast(dict[str, object], report["coverage"])
    assert report["status"] == "OK"
    assert coverage["eligible_rows"] == 35 * 6
    assert report["data_adequacy"] == "limited"
    assert report["multiple_testing"]["candidate_count_n"] == 6


def test_long_short_costs_include_turnover_and_borrow_drag() -> None:
    rows = _rows(symbols=30, periods=5)
    config = EdgarFactorCostConfig(min_cross_section=20)
    cheap = simulate_factor_portfolio(
        rows,
        factor="earnings_yield_ep",
        horizon="3m",
        kind="long_short",
        cost_profile=FactorCostProfile("cheap", 6.0, short_borrow_bps_per_year=50.0),
        config=config,
    )
    expensive = simulate_factor_portfolio(
        rows,
        factor="earnings_yield_ep",
        horizon="3m",
        kind="long_short",
        cost_profile=FactorCostProfile("expensive", 30.0, short_borrow_bps_per_year=50.0),
        config=config,
    )

    assert cheap
    assert cheap[0].turnover == pytest.approx(2.0)
    assert cheap[0].borrow_cost > 0.0
    assert cheap[0].net_return < cheap[0].gross_return
    assert expensive[0].net_return < cheap[0].net_return


def test_personal_long_only_has_no_short_borrow_and_costs_reduce_return() -> None:
    rows = _rows(symbols=30, periods=5)
    config = EdgarFactorCostConfig(min_cross_section=20)

    portfolio = simulate_factor_portfolio(
        rows,
        factor="fcf_yield",
        horizon="6m",
        kind="long_only",
        cost_profile=FactorCostProfile("base", 6.0, short_borrow_bps_per_year=50.0),
        config=config,
    )

    assert portfolio
    assert portfolio[0].short_count == 0
    assert portfolio[0].borrow_cost == 0.0
    assert portfolio[0].trading_cost > 0.0
    assert portfolio[0].net_return < portfolio[0].gross_return


def test_factor_cost_report_includes_long_only_and_sharadar_decision() -> None:
    report = run_edgar_factor_cost_diagnostic(
        _rows(symbols=40, periods=8),
        config=EdgarFactorCostConfig(min_cross_section=20),
    )

    assert report["status"] == "OK"
    assert report["state"] in {"EDGE", "NO_EDGE"}
    assert len(cast(list[object], report["base_long_short"])) == 6
    assert len(cast(list[object], report["personal_long_only"])) == 6
    assert cast(dict[str, object], report["multiple_testing"])["pbo"]
    assert cast(dict[str, object], report["sharadar_decision"])["decision"] in {
        "SHARADAR_WORTH_PAYING_TO_CONFIRM_NET_COSTS",
        "DO_NOT_PAY_SHARADAR_FOR_THIS_FACTOR_SET_NOW",
    }
