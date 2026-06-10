from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from aegis.edgar_pit import PitFundamentalStore, derive_net_debt
from aegis.olympus_survivor_light import (
    ALLOWED_SURVIVOR_LIGHT_VERDICTS,
    LOCKED_VALUE_QUALITY_FACTORS,
    ConstituentChange,
    FreePriceSource,
    HistoricalConstituentStore,
    PriceBar,
    align_pit_fundamentals_with_prices,
    evaluate_survivor_light_ic,
    first_price_after,
    latest_price_on_or_before,
    parse_wikipedia_sp500_snapshot,
    write_sanitized_pipeline_status,
)


def _payload(
    revenue: float,
    net_income: float,
    shares: float,
    priceish: float,
) -> dict[str, object]:
    concepts = {
        "Revenues": revenue,
        "NetIncomeLoss": net_income,
        "CommonStockSharesOutstanding": shares,
        "StockholdersEquity": revenue * 0.4,
        "GrossProfit": revenue * 0.35,
        "Assets": revenue * 0.8,
        "OperatingIncomeLoss": net_income * 1.1,
        "DepreciationDepletionAndAmortization": priceish,
        "NetCashProvidedByUsedInOperatingActivities": net_income * 1.05,
        "PaymentsToAcquirePropertyPlantAndEquipment": priceish * 0.4,
        "LongTermDebt": priceish * 2,
        "LongTermDebtCurrent": priceish * 0.2,
        "ShortTermBorrowings": priceish * 0.3,
        "CommercialPaper": priceish * 0.1,
        "CashAndCashEquivalentsAtCarryingValue": priceish,
    }
    return {
        "facts": {
            "us-gaap": {
                concept: {
                    "units": {
                        "USD" if concept != "CommonStockSharesOutstanding" else "shares": [
                            {
                                "val": value,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-01-05",
                                "end": "2023-12-31",
                                "accn": f"000-{concept}",
                            }
                        ]
                    }
                }
                for concept, value in concepts.items()
            }
        }
    }


def _store() -> PitFundamentalStore:
    parts = [
        PitFundamentalStore.from_companyfacts(
            ticker="AAA", cik="1", payload=_payload(1000, 90, 10, 20)
        ),
        PitFundamentalStore.from_companyfacts(
            ticker="BBB", cik="2", payload=_payload(1000, 70, 10, 30)
        ),
        PitFundamentalStore.from_companyfacts(
            ticker="CCC", cik="3", payload=_payload(1000, 50, 10, 40)
        ),
    ]
    return PitFundamentalStore(
        facts=[fact for part in parts for fact in part.facts],
        restatements=[],
        company_metadata={},
    )


def _write_prices(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[float]] = {
        "AAA": [10, 11, 12, 13, 14],
        "BBB": [10, 10.5, 10.8, 11.0, 11.2],
        "CCC": [10, 9.8, 9.7, 9.6, 9.5],
    }
    dates = ["2024-01-10", "2024-02-10", "2024-03-10", "2024-04-10", "2024-05-10"]
    for ticker, closes in rows.items():
        lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
        for raw_date, close in zip(dates, closes, strict=True):
            lines.append(f"{raw_date},{close},{close},{close},{close},{close},1000")
        (cache_dir / f"{ticker}.csv").write_text("\n".join(lines), encoding="utf-8")


def test_free_price_source_marks_delisted_missing_without_fake_bars(tmp_path: Path) -> None:
    source = FreePriceSource(cache_dir=tmp_path, delisted_tickers={"ENRNQ"})

    payload = source.get_prices("ENRNQ", date(2001, 1, 1), date(2002, 1, 1))

    assert payload["bars"] == []
    assert payload["survivorship"] == "light"
    assert payload["survivorship_status"]["status"] == "delisted_price_missing_free_source"


def test_historical_constituents_as_of_applies_changes_on_effective_date() -> None:
    store = HistoricalConstituentStore.from_rows(
        ["AAA", "BBB"],
        [
            {"effective_date": "2024-02-01", "ticker": "CCC", "action": "add"},
            {"effective_date": "2024-03-01", "ticker": "AAA", "action": "remove"},
        ],
        source="synthetic",
    )

    assert store.as_of("2024-01-31") == {"AAA", "BBB"}
    assert store.as_of("2024-02-01") == {"AAA", "BBB", "CCC"}
    assert store.as_of("2024-03-01") == {"BBB", "CCC"}
    assert "free historical constituents" in store.caveat()


def test_historical_constituents_can_reconstruct_from_current_members() -> None:
    store = HistoricalConstituentStore.from_current_members_and_changes(
        ["BBB", "CCC"],
        [
            ConstituentChange(date(2024, 2, 1), "CCC", "add", "synthetic"),
            ConstituentChange(date(2024, 3, 1), "AAA", "remove", "synthetic"),
        ],
    )

    assert store.as_of("2024-01-31") == {"AAA", "BBB"}
    assert store.as_of("2024-02-01") == {"AAA", "BBB", "CCC"}
    assert store.as_of("2024-03-01") == {"BBB", "CCC"}
    assert store.filtered({"CCC"}).as_of("2024-01-31") == set()
    assert store.filtered({"CCC"}).as_of("2024-02-01") == {"CCC"}


def test_wikipedia_sp500_snapshot_parser_builds_asof_store() -> None:
    html = """
    <table>
      <tr><th>Symbol</th><th>Security</th><th>Date added</th><th>CIK</th></tr>
      <tr><td>BBB</td><td>Beta</td><td>2020-01-01</td><td>0000000002</td></tr>
      <tr><td>CCC</td><td>Gamma</td><td>2024-02-01</td><td>0000000003</td></tr>
    </table>
    <table>
      <tr><th>Effective Date</th><th>Added</th><th>Removed</th><th>Reason</th></tr>
      <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
      <tr><td>February 1, 2024</td><td>CCC</td><td>Gamma</td><td></td><td></td><td>add</td></tr>
      <tr><td>March 1, 2024</td><td></td><td></td><td>AAA</td><td>Alpha</td><td>remove</td></tr>
      <tr><td>January 1, 2025</td><td>DDD</td><td>Delta</td><td></td><td></td><td>future</td></tr>
    </table>
    """

    snapshot = parse_wikipedia_sp500_snapshot(html, as_of_date=date(2024, 12, 31))

    assert snapshot.current["BBB"].cik == "2"
    assert snapshot.current["CCC"].date_added == date(2024, 2, 1)
    assert [change.ticker for change in snapshot.changes] == ["CCC", "AAA"]
    assert snapshot.store.as_of("2024-01-31") == {"AAA", "BBB"}
    assert snapshot.store.as_of("2024-02-01") == {"AAA", "BBB", "CCC"}
    assert snapshot.store.as_of("2024-03-01") == {"BBB", "CCC"}
    assert "paid-source validation" in snapshot.caveat


def test_price_alignment_uses_price_on_or_before_and_next_day_trade() -> None:
    bars = [
        PriceBar(date(2024, 1, 5), 10, 10, 10, 10, 10, 1),
        PriceBar(date(2024, 1, 8), 11, 11, 11, 11, 11, 1),
        PriceBar(date(2024, 1, 10), 12, 12, 12, 12, 12, 1),
    ]

    latest = latest_price_on_or_before(bars, date(2024, 1, 9))
    first = first_price_after(bars, date(2024, 1, 9))

    assert latest is not None
    assert latest.date == date(2024, 1, 8)
    assert first is not None
    assert first.date == date(2024, 1, 10)


def test_net_debt_includes_short_term_borrowings_and_commercial_paper() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="AAA",
        cik="1",
        payload=_payload(1000, 90, 10, 20),
    )

    net_debt = derive_net_debt(store.as_of("AAA", "2024-01-08"))

    assert net_debt is not None
    assert net_debt.value == 32.0


def test_survivor_light_pipeline_outputs_capped_verdict_and_warning(tmp_path: Path) -> None:
    _write_prices(tmp_path)
    source = FreePriceSource(cache_dir=tmp_path)
    universe = HistoricalConstituentStore(["AAA", "BBB", "CCC"], [])

    observations = align_pit_fundamentals_with_prices(
        fundamentals=_store(),
        price_source=source,
        constituent_store=universe,
        rebalance_dates=[
            date(2024, 1, 10),
            date(2024, 2, 10),
            date(2024, 3, 10),
            date(2024, 4, 10),
            date(2024, 5, 10),
        ],
    )
    report = evaluate_survivor_light_ic(observations)

    assert set(report["factors"]) == set(LOCKED_VALUE_QUALITY_FACTORS)
    assert report["verdict"] in ALLOWED_SURVIVOR_LIGHT_VERDICTS
    assert report["verdict"] != "ROBUST_VALUE_FACTOR_EDGE"
    assert report["survivorship"] == "light"
    for factor in report["factors"].values():
        assert factor["survivorship"] == "light"
        assert "must not be used as edge evidence" in factor["warning"]
        assert factor["full_cost"]["funding"] == "N/A"
    assert report["benchmarks"]["equal_weight"]["status"] == "OK"

    sanitized_path = tmp_path / "sanitized" / "status.json"
    write_sanitized_pipeline_status(sanitized_path, report)
    sanitized = json.loads(sanitized_path.read_text(encoding="utf-8"))
    assert sanitized["private_results"].startswith("redacted")
    assert sanitized["verdict"] in ALLOWED_SURVIVOR_LIGHT_VERDICTS


def test_sharadar_not_configured_degrades_gracefully() -> None:
    from aegis.olympus_survivor_light import SharadarPriceSource

    src = SharadarPriceSource(api_key="")  # explicit empty = not configured
    assert src.configured is False
    out = src.get_prices("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert out["status"] == "paid_source_not_configured"
    assert out["bars"] == []
    assert out["survivorship_status"]["status"] == "paid_source_not_configured"


def test_sharadar_parses_sep_and_tickers_with_injected_http() -> None:
    from aegis.olympus_survivor_light import SharadarPriceSource

    def fake_http(url: str) -> dict[str, object]:
        if "TICKERS" in url:
            return {
                "datatable": {
                    "columns": [
                        {"name": "ticker"},
                        {"name": "isdelisted"},
                        {"name": "firstpricedate"},
                        {"name": "lastpricedate"},
                    ],
                    "data": [["TWTR", "Y", "2013-11-07", "2022-10-27"]],
                }
            }
        return {
            "datatable": {
                "columns": [
                    {"name": "ticker"},
                    {"name": "date"},
                    {"name": "open"},
                    {"name": "high"},
                    {"name": "low"},
                    {"name": "close"},
                    {"name": "closeadj"},
                    {"name": "volume"},
                ],
                "data": [
                    ["TWTR", "2022-10-26", 53.0, 53.9, 52.8, 53.7, 53.7, 1000.0],
                    ["TWTR", "2022-10-25", 52.0, 52.5, 51.5, 52.2, 52.2, 900.0],
                ],
            }
        }

    src = SharadarPriceSource(api_key="test-key", http_get=fake_http)
    assert src.configured is True
    status = src.survivorship_status("TWTR")
    assert status["status"] == "delisted" and status["isdelisted"] == "Y"

    out = src.get_prices("TWTR", date(2022, 10, 1), date(2022, 10, 31))
    assert out["source"] == "sharadar_sep"
    assert out["survivorship"] == "full"  # delisting-aware, not survivor-light
    assert [b["date"] for b in out["bars"]] == ["2022-10-25", "2022-10-26"]  # sorted asc
    assert out["bars"][-1]["close"] == 53.7


def test_select_price_source_prefers_sharadar_only_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aegis.olympus_survivor_light import (
        FreePriceSource,
        SharadarPriceSource,
        select_price_source,
    )

    for env in ("NASDAQ_DATA_LINK_API_KEY", "SHARADAR_API_KEY", "QUANDL_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert isinstance(select_price_source(delisted_tickers=["TWTR"]), FreePriceSource)

    monkeypatch.setenv("NASDAG_PLACEHOLDER", "x")  # noise
    monkeypatch.setenv("SHARADAR_API_KEY", "abc123")
    assert isinstance(select_price_source(), SharadarPriceSource)
