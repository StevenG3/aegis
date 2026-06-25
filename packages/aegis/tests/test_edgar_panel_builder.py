from __future__ import annotations

from datetime import date

from aegis.edgar_panel_builder import (
    PanelBuildConfig,
    build_edgar_ic_panel,
    factor_values_from_facts,
    historical_universe_symbols,
)
from aegis.edgar_pit import EdgarFact, PitFundamentalStore
from aegis.olympus_survivor_light import (
    ConstituentChange,
    HistoricalConstituentStore,
    PriceBar,
)


def _fact(concept: str, value: float, available_on: date) -> EdgarFact:
    return EdgarFact(
        ticker="AAA",
        cik="0000000001",
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


def _snapshot(available_on: date, *, net_income: float = 10.0) -> list[EdgarFact]:
    return [
        _fact("CommonStockSharesOutstanding", 10.0, available_on),
        _fact("NetIncomeLoss", net_income, available_on),
        _fact("StockholdersEquity", 50.0, available_on),
        _fact("NetCashProvidedByUsedInOperatingActivities", 12.0, available_on),
        _fact("PaymentsToAcquirePropertyPlantAndEquipment", 2.0, available_on),
        _fact("Revenues", 100.0, available_on),
        _fact("GrossProfit", 40.0, available_on),
        _fact("Assets", 200.0, available_on),
    ]


def _bar(value: date, close: float) -> PriceBar:
    return PriceBar(
        date=value,
        open=close,
        high=close,
        low=close,
        close=close,
        adj_close=None,
        volume=1000.0,
    )


def test_factor_values_match_predeclared_names() -> None:
    values = factor_values_from_facts(
        {fact.concept: fact for fact in _snapshot(date(2024, 1, 15))},
        close=10.0,
    )

    assert values["earnings_yield_ep"] == 0.1
    assert values["book_to_price_bp"] == 0.5
    assert values["fcf_yield"] == 0.1
    assert values["sales_to_price_sp"] == 1.0
    assert values["roe"] == 0.2
    assert values["gross_margin"] == 0.4
    assert values["low_accruals"] == 0.01
    assert values["asset_turnover"] == 0.5


def test_panel_builder_uses_only_facts_available_on_or_before_asof_and_t_plus_one() -> None:
    store = PitFundamentalStore(
        facts=[
            *_snapshot(date(2024, 1, 15), net_income=10.0),
            *_snapshot(date(2024, 3, 15), net_income=100.0),
        ]
    )
    prices = {
        "AAA": [
            _bar(date(2024, 1, 31), 10.0),
            _bar(date(2024, 2, 1), 11.0),
            _bar(date(2024, 2, 29), 12.0),
            _bar(date(2024, 5, 1), 13.0),
            _bar(date(2024, 8, 1), 14.0),
        ]
    }

    observations, coverage = build_edgar_ic_panel(
        fundamentals={"AAA": store},
        prices=prices,
        constituent_store=HistoricalConstituentStore(["AAA"], []),
        config=PanelBuildConfig(start=date(2024, 1, 31), end=date(2024, 1, 31)),
    )

    assert coverage["observations"] == 1
    row = observations[0]
    assert row.available_on == date(2024, 1, 15)
    assert row.factors["earnings_yield_ep"] == 10.0 / (10.0 * 11.0)
    assert row.forward_returns["1m"] == 12.0 / 11.0 - 1.0
    assert row.forward_returns["3m"] == 13.0 / 11.0 - 1.0
    assert row.forward_returns["6m"] == 14.0 / 11.0 - 1.0


def test_historical_universe_symbols_includes_removed_non_current_names() -> None:
    store = HistoricalConstituentStore.from_current_members_and_changes(
        ["BBB", "CCC"],
        [
            ConstituentChange(date(2024, 2, 1), "CCC", "add", "synthetic"),
            ConstituentChange(date(2024, 3, 1), "AAA", "remove", "synthetic"),
        ],
    )

    symbols = historical_universe_symbols(
        store,
        [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)],
    )

    assert symbols == {"AAA", "BBB", "CCC"}
    assert "AAA" not in {"BBB", "CCC"}
