from __future__ import annotations

from datetime import date
from pathlib import Path

from aegis.edgar_pit import (
    PitFundamentalStore,
    SecEdgarClient,
    add_business_days,
    build_coverage_matrix,
    derive_ebitda,
    derive_fcf,
    derive_net_debt,
    extract_submission_metadata,
    validate_sec_user_agent,
)


def _companyfacts_payload() -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 100.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000001",
                            },
                            {
                                "val": 105.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K/A",
                                "filed": "2024-03-20",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000002",
                            },
                            {
                                "val": 130.0,
                                "fy": 2024,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2024-05-06",
                                "end": "2024-03-31",
                                "accn": "0000000000-24-000003",
                            },
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 10.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000001",
                            }
                        ]
                    }
                },
            }
        }
    }


def _derived_payload() -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 100.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000010",
                            }
                        ]
                    }
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            {
                                "val": 25.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000011",
                            }
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "val": 80.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000012",
                            }
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "val": 30.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000013",
                            }
                        ]
                    }
                },
            }
        }
    }


def _net_debt_payload() -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {
                                "val": 200.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15",
                                "end": "2023-12-31",
                            }
                        ]
                    }
                },
                "LongTermDebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "val": 20.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15",
                                "end": "2023-12-31",
                            }
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "val": 80.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15",
                                "end": "2023-12-31",
                            }
                        ]
                    }
                },
            }
        }
    }


def test_as_of_hides_fact_before_filing_and_before_lag() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_companyfacts_payload(),
    )

    assert store.as_of("ACME", "2023-12-31") == {}
    assert store.as_of("ACME", "2024-02-16") == {}


def test_as_of_uses_filed_date_plus_next_business_day_not_period_end() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_companyfacts_payload(),
    )

    visible = store.as_of("ACME", "2024-02-20")

    assert visible["Revenues"].value == 100.0
    assert visible["Revenues"].period_end == date(2023, 12, 31)
    assert visible["Revenues"].available_on == date(2024, 2, 19)
    assert visible["NetIncomeLoss"].value == 10.0


def test_as_filed_primary_is_not_replaced_by_restatement_for_same_period() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_companyfacts_payload(),
    )

    visible = store.as_of("ACME", "2024-04-01")

    assert visible["Revenues"].value == 100.0
    assert len(store.restatements) == 1
    assert store.restatements[0].value == 105.0
    assert store.restatements[0].is_restatement is True


def test_as_of_moves_to_newer_period_only_after_that_filing_is_available() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_companyfacts_payload(),
    )

    assert store.as_of("ACME", "2024-05-06")["Revenues"].value == 100.0
    assert store.as_of("ACME", "2024-05-07")["Revenues"].value == 130.0


def test_business_day_lag_skips_weekend() -> None:
    assert add_business_days(date(2024, 2, 16), 1) == date(2024, 2, 19)


def test_derived_ebitda_and_fcf_compute_when_atoms_are_present() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_derived_payload(),
    )

    visible = store.as_of("ACME", "2024-02-20")
    ebitda = derive_ebitda(visible)
    fcf = derive_fcf(visible)

    assert ebitda is not None
    assert ebitda.concept == "Ebitda"
    assert ebitda.value == 125.0
    assert ebitda.filed == date(2024, 2, 16)
    assert ebitda.available_on == date(2024, 2, 19)
    assert fcf is not None
    assert fcf.concept == "FreeCashFlow"
    assert fcf.value == 50.0


def test_derived_concepts_return_none_when_an_atom_is_missing() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_derived_payload(),
        concepts={"OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities"},
    )

    visible = store.as_of("ACME", "2024-02-20")

    assert derive_ebitda(visible) is None
    assert derive_fcf(visible) is None


def test_derived_available_on_uses_later_atom_filing_lag() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_derived_payload(),
    )

    early = store.as_of("ACME", "2024-02-16")
    available = store.as_of("ACME", "2024-02-19")

    assert "OperatingIncomeLoss" in early
    assert "DepreciationDepletionAndAmortization" not in early
    assert derive_ebitda(early) is None
    ebitda = derive_ebitda(available)
    assert ebitda is not None
    assert ebitda.available_on == date(2024, 2, 19)


def test_coverage_flags_post_event_cik_history_gap() -> None:
    metadata = extract_submission_metadata(
        ticker="GM",
        cik="1467858",
        payload={
            "name": "GENERAL MOTORS CO",
            "tickers": ["GM"],
            "formerNames": [{"name": "GENERAL MOTORS HOLDINGS LLC"}],
            "filings": {"recent": {"filingDate": ["2010-08-19", "2011-02-28"]}},
        },
        pilot_status="active_restructured",
    )
    store = PitFundamentalStore.from_companyfacts(
        ticker="GM",
        cik="1467858",
        payload=_companyfacts_payload(),
        company_metadata=metadata,
    )

    company = store.coverage()["companies"][0]

    assert company["earliest_filing_date"] == "2010-08-19"
    assert company["former_names_count"] == 1
    assert company["pilot_status"] == "active_restructured"
    assert company["coverage_window_gap"]["pre_event_history_missing"] is True


def test_submission_metadata_uses_true_earliest_from_submission_files() -> None:
    metadata = extract_submission_metadata(
        ticker="AAPL",
        cik="320193",
        payload={
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
            "formerNames": [],
            "filings": {
                "recent": {"filingDate": ["2015-05-13", "2016-01-01"]},
                "files": [
                    {
                        "name": "CIK0000320193-submissions-001.json",
                        "filingFrom": "1994-01-26",
                        "filingTo": "2015-05-11",
                    }
                ],
            },
        },
        pilot_status="active",
    )
    store = PitFundamentalStore.from_companyfacts(
        ticker="AAPL",
        cik="320193",
        payload=_companyfacts_payload(),
        company_metadata=metadata,
    )

    company = store.coverage()["companies"][0]

    assert company["earliest_filing_date"] == "1994-01-26"
    assert company["earliest_recent_block_filing_date"] == "2015-05-13"
    assert company["earliest_filing_date_source"] == "recent_plus_files"
    assert company["coverage_window_gap"]["pre_event_history_missing"] is False


def test_restatement_distribution_is_reported_by_concept_and_form() -> None:
    store = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_companyfacts_payload(),
    )

    distribution = store.coverage()["restatement_distribution"]

    assert distribution["by_concept"] == {"Revenues": 1}
    assert distribution["by_form"] == {"10-K/A": 1}


def test_net_debt_derivation_requires_atoms_and_never_fills_zero() -> None:
    full = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_net_debt_payload(),
    )
    partial = PitFundamentalStore.from_companyfacts(
        ticker="ACME",
        cik="1234",
        payload=_net_debt_payload(),
        concepts={"LongTermDebt", "LongTermDebtCurrent"},
    )

    net_debt = derive_net_debt(full.as_of("ACME", "2024-02-20"))

    assert net_debt is not None
    assert net_debt.concept == "NetDebt"
    assert net_debt.value == 140.0
    assert derive_net_debt(partial.as_of("ACME", "2024-02-20")) is None


def test_sec_user_agent_placeholder_is_rejected_before_network_request(tmp_path: Path) -> None:
    client = SecEdgarClient(cache_dir=tmp_path)

    try:
        client.fetch_submissions("320193", force=True)
    except ValueError as exc:
        assert "AEGIS_SEC_USER_AGENT" in str(exc)
    else:
        raise AssertionError("expected placeholder SEC user-agent to be rejected")


def test_sec_user_agent_requires_real_contact() -> None:
    try:
        validate_sec_user_agent("AegisOlympusResearch/0.1 research@example.com")
    except ValueError as exc:
        assert "AEGIS_SEC_USER_AGENT" in str(exc)
    else:
        raise AssertionError("expected example.com SEC user-agent to be rejected")

    validate_sec_user_agent("AegisOlympusResearch/0.1 contact=https://example.org/aegis")


def test_coverage_matrix_marks_missing_cells_false_including_derived() -> None:
    full = PitFundamentalStore.from_companyfacts(
        ticker="AAA",
        cik="1",
        payload=_derived_payload(),
    )
    partial = PitFundamentalStore.from_companyfacts(
        ticker="BBB",
        cik="2",
        payload=_derived_payload(),
        concepts={"OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities"},
    )
    store = PitFundamentalStore(
        facts=full.facts + partial.facts,
        restatements=full.restatements + partial.restatements,
        company_metadata={
            "CCC": extract_submission_metadata(
                ticker="CCC",
                cik="3",
                payload={
                    "name": "metadata only issuer",
                    "tickers": ["CCC"],
                    "formerNames": [],
                    "filings": {"recent": {"filingDate": ["2024-02-16"]}},
                },
                pilot_status="active",
            )
        },
    )

    matrix = build_coverage_matrix(
        store,
        concepts={"OperatingIncomeLoss", "Ebitda", "FreeCashFlow"},
        fiscal_years={2023, 2024},
    )

    assert matrix["matrix"]["AAA"]["OperatingIncomeLoss"]["2023"] is True
    assert matrix["matrix"]["AAA"]["Ebitda"]["2023"] is True
    assert matrix["matrix"]["AAA"]["FreeCashFlow"]["2023"] is True
    assert matrix["matrix"]["AAA"]["Ebitda"]["2024"] is False
    assert matrix["matrix"]["BBB"]["OperatingIncomeLoss"]["2023"] is True
    assert matrix["matrix"]["BBB"]["Ebitda"]["2023"] is False
    assert matrix["matrix"]["BBB"]["FreeCashFlow"]["2023"] is False
    assert matrix["matrix"]["CCC"]["OperatingIncomeLoss"]["2023"] is False
    assert matrix["matrix"]["CCC"]["Ebitda"]["2024"] is False


def test_sec_client_rejects_rate_above_sec_limit() -> None:
    try:
        SecEdgarClient(requests_per_second=10.1)
    except ValueError as exc:
        assert "SEC requests_per_second" in str(exc)
    else:
        raise AssertionError("expected SEC rate limit guard")
