from __future__ import annotations

from datetime import date

from aegis.edgar_full_universe_ic import (
    FACTOR_NAMES,
    HORIZONS,
    TRIAL_COUNT_N,
    EdgarIcConfig,
    EdgarIcObservation,
    run_edgar_full_universe_ic,
)


def _observation(
    symbol: str,
    as_of: date,
    score: float,
    *,
    available_on: date | None = None,
) -> EdgarIcObservation:
    factors = {name: score for name in FACTOR_NAMES}
    returns = {
        "1m": score * 0.001,
        "3m": score * 0.002,
        "6m": score * 0.003,
    }
    return EdgarIcObservation(
        symbol=symbol,
        as_of=as_of,
        available_on=available_on or as_of,
        factors=factors,
        forward_returns=returns,
    )


def _rows(symbols: int = 40, periods: int = 8) -> list[EdgarIcObservation]:
    result: list[EdgarIcObservation] = []
    for month in range(1, periods + 1):
        as_of = date(2024, month, 28)
        for index in range(symbols):
            result.append(_observation(f"S{index:03d}", as_of, float(index)))
    return result


def test_full_scope_factor_horizon_trials_and_survivor_ceiling() -> None:
    report = run_edgar_full_universe_ic(
        _rows(),
        config=EdgarIcConfig(min_symbols=30, min_periods=6, min_cross_section=20),
    )

    assert report["status"] == "OK"
    assert report["data_adequacy"] == "limited"
    assert report["predeclared"]["trial_count_n"] == TRIAL_COUNT_N
    assert report["multiple_testing"]["candidate_count_n"] == len(FACTOR_NAMES) * len(HORIZONS)
    assert report["multiple_testing"]["pbo"]["valid"] is True
    assert report["verdict"] in {"SUGGESTIVE_NEEDS_PAID_CONFIRM", "NO_EDGE"}
    assert "ROBUST" not in str(report)


def test_future_filing_rows_are_excluded_before_ic() -> None:
    rows = _rows(symbols=35, periods=6)
    rows.append(
        _observation(
            "FUTURE",
            date(2024, 6, 28),
            999.0,
            available_on=date(2024, 7, 15),
        )
    )

    report = run_edgar_full_universe_ic(
        rows,
        config=EdgarIcConfig(min_symbols=30, min_periods=6, min_cross_section=20),
    )

    assert report["coverage"]["excluded_rows"] == 1
    assert report["coverage"]["eligible_rows"] == 35 * 6


def test_insufficient_data_is_blocked_and_counts_all_trials() -> None:
    report = run_edgar_full_universe_ic(
        _rows(symbols=5, periods=2),
        config=EdgarIcConfig(min_symbols=30, min_periods=6, min_cross_section=20),
    )

    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["state"] == "INSUFFICIENT"
    assert report["data_adequacy"] == "blocked"
    assert report["multiple_testing"]["candidate_count_n"] == TRIAL_COUNT_N
    assert report["multiple_testing"]["fdr_survivors"] == 0


def test_missing_horizon_does_not_silently_create_trial_signal() -> None:
    rows = [
        EdgarIcObservation(
            symbol=f"S{index:03d}",
            as_of=date(2024, 1, 31),
            available_on=date(2024, 1, 15),
            factors={"earnings_yield_ep": float(index)},
            forward_returns={"3m": index * 0.001},
        )
        for index in range(40)
    ]

    report = run_edgar_full_universe_ic(
        rows,
        config=EdgarIcConfig(min_symbols=30, min_periods=2, min_cross_section=20),
    )

    assert report["data_adequacy"] == "blocked"
    assert "period count" in report["reason"]
