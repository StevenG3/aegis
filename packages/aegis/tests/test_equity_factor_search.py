from __future__ import annotations

from datetime import date

from aegis.edgar_pit import EdgarFact
from aegis.equity_factor_search import (
    PREDECLARED_COMPOSITES,
    PREDECLARED_FACTORS,
    SURVIVOR_LIGHT_VERDICT_CEILING,
    PriceObservation,
    SearchConfig,
    benjamini_hochberg,
    build_factor_values,
    run_equity_factor_search,
    sanitized_report,
)


def fact(ticker: str, concept: str, value: float, available_on: date) -> EdgarFact:
    return EdgarFact(
        ticker=ticker,
        cik=f"{abs(hash(ticker)) % 999999:06d}",
        concept=concept,
        unit="USD",
        value=value,
        filed=available_on,
        available_on=available_on,
        period_end=available_on,
        accession=None,
        form="10-Q",
        fiscal_year=available_on.year,
        fiscal_period="Q1",
        frame=None,
    )


def fundamental_snapshot(
    ticker: str,
    *,
    available_on: date,
    shares: float,
    assets: float,
    liabilities: float,
    equity: float,
    net_income: float,
    gross_profit: float,
    operating_income: float,
    operating_cash_flow: float,
    capex: float,
) -> dict[str, EdgarFact]:
    values = {
        "CommonStockSharesOutstanding": shares,
        "Assets": assets,
        "Liabilities": liabilities,
        "StockholdersEquity": equity,
        "NetIncomeLoss": net_income,
        "GrossProfit": gross_profit,
        "OperatingIncomeLoss": operating_income,
        "NetCashProvidedByUsedInOperatingActivities": operating_cash_flow,
        "PaymentsToAcquirePropertyPlantAndEquipment": capex,
        "CashAndCashEquivalentsAtCarryingValue": 10.0,
        "LongTermDebt": 20.0,
    }
    return {
        concept: fact(ticker, concept, value, available_on)
        for concept, value in values.items()
    }


def test_build_factor_values_uses_only_visible_facts() -> None:
    early = fundamental_snapshot(
        "AAA",
        available_on=date(2024, 1, 15),
        shares=10,
        assets=100,
        liabilities=40,
        equity=60,
        net_income=6,
        gross_profit=30,
        operating_income=7,
        operating_cash_flow=8,
        capex=2,
    )
    future = fundamental_snapshot(
        "AAA",
        available_on=date(2024, 4, 15),
        shares=10,
        assets=100,
        liabilities=40,
        equity=60,
        net_income=60,
        gross_profit=30,
        operating_income=7,
        operating_cash_flow=8,
        capex=2,
    )
    price = PriceObservation("AAA", date(2024, 3, 31), close=10, forward_return=0.01)

    visible_values = build_factor_values(early, price)
    leaked_values = build_factor_values(future, price)

    assert visible_values["earnings_yield_ep"] == 0.06
    assert leaked_values["earnings_yield_ep"] == 0.6
    assert visible_values["earnings_yield_ep"] != leaked_values["earnings_yield_ep"]


def test_equity_factor_search_reports_predeclared_n_and_survivor_ceiling() -> None:
    fundamentals = {
        symbol: {
            date(2024, 1, 15): fundamental_snapshot(
                symbol,
                available_on=date(2024, 1, 15),
                shares=10 + index,
                assets=100 + index * 10,
                liabilities=30 + index,
                equity=70 + index,
                net_income=4 + index,
                gross_profit=20 + index,
                operating_income=5 + index,
                operating_cash_flow=6 + index,
                capex=1,
            ),
            date(2024, 4, 15): fundamental_snapshot(
                symbol,
                available_on=date(2024, 4, 15),
                shares=10 + index,
                assets=105 + index * 10,
                liabilities=32 + index,
                equity=73 + index,
                net_income=5 + index,
                gross_profit=21 + index,
                operating_income=6 + index,
                operating_cash_flow=7 + index,
                capex=1,
            ),
        }
        for index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
    }
    observations = []
    for month, as_of in enumerate(
        (
            date(2024, 2, 29),
            date(2024, 3, 31),
            date(2024, 4, 30),
            date(2024, 5, 31),
            date(2024, 6, 30),
            date(2024, 7, 31),
        )
    ):
        for index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
            forward = (index - 2.5) * 0.01 + month * 0.0001
            observations.append(
                PriceObservation(
                    symbol,
                    as_of,
                    close=10 + index,
                    forward_return=forward,
                    market_cap=(10 + index) * (10 + index),
                    volatility_252=0.30 - index * 0.01,
                    momentum_12_1=(index - 2.5) * 0.02,
                )
            )

    report = run_equity_factor_search(
        observations,
        fundamentals,
        config=SearchConfig(locked_oos_fraction=0.34, groups=3, fdr_alpha=0.10),
    )

    expected_n = len(PREDECLARED_FACTORS) + len(PREDECLARED_COMPOSITES)
    assert report["status"] == "OK"
    assert report["verdict"] in {
        "SUGGESTIVE_NEEDS_PAID_CONFIRM",
        "NO_EDGE",
        "INSUFFICIENT",
    }
    assert "ROBUST" not in str(report["verdict"])
    assert report["verdict_ceiling"] == SURVIVOR_LIGHT_VERDICT_CEILING
    assert report["predeclared"]["trial_count_n"] == expected_n
    assert report["multiple_testing"]["trial_count_n"] == expected_n
    assert report["counts"]["n"] == expected_n
    assert report["cost_model"]["round_trip_bps"] == 12.0
    assert report["benchmarks"]["equal_weight"]["periods"] > 0
    assert report["benchmarks"]["market_cap_weight"]["periods"] > 0
    assert report["split"]["first_oos_signal_date"] == report["split"]["locked_oos_start_date"]


def test_bh_fdr_counts_every_predeclared_trial() -> None:
    discoveries = benjamini_hochberg([0.001, 0.02, 0.20, 0.90], alpha=0.10)
    assert discoveries == [True, True, False, False]
    conservative = benjamini_hochberg([0.04, 0.05, 0.06, 0.07], alpha=0.05)
    assert conservative == [False, False, False, False]


def test_sanitized_report_omits_edge_metrics() -> None:
    report = {
        "status": "OK",
        "verdict": "NO_EDGE",
        "verdict_ceiling": SURVIVOR_LIGHT_VERDICT_CEILING,
        "survivorship": "survivor_light",
        "predeclared": {"trial_count_n": 16},
        "multiple_testing": {"fdr_survivors": 0},
        "split": {"locked_oos_start_date": "2024-06-30"},
        "cost_model": {"round_trip_bps": 12.0},
        "counts": {"n": 16},
        "safety": {"read_only": True},
        "factors": {"secret_edge_numbers": {}},
    }

    sanitized = sanitized_report(report)

    assert sanitized["verdict"] == "NO_EDGE"
    assert "factors" not in sanitized
    assert sanitized["note"].startswith("edge metrics intentionally omitted")
