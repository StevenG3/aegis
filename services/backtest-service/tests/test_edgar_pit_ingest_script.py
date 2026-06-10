from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_ingest_script():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "edgar_pit_ingest.py"
    spec = importlib.util.spec_from_file_location("edgar_pit_ingest_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["edgar_pit_ingest_script"] = module
    spec.loader.exec_module(module)
    return module


def test_predeclared_pilot_universe_loads_from_metadata() -> None:
    module = load_ingest_script()
    repo_root = Path(__file__).resolve().parents[3]

    companies = module._load_pilot_universe(
        repo_root / "incubating" / "edgar_pit_fundamentals.meta.json"
    )

    assert len(companies) == 32
    assert companies[0]["ticker"] == "AAPL"
    assert any(company["status"].startswith("acquired_delisted") for company in companies)
    assert any(company["status"] == "bankrupt_delisted" for company in companies)


def test_default_matrix_out_uses_private_incubating_path(monkeypatch) -> None:
    module = load_ingest_script()
    monkeypatch.delenv("AEGIS_EDGAR_MATRIX_OUT", raising=False)

    assert str(module._default_matrix_out()).endswith(
        "apps/aegis-strategies/incubating/olympus37/matrix.json"
    )


def test_matrix_out_can_be_disabled_with_empty_env(monkeypatch) -> None:
    module = load_ingest_script()
    monkeypatch.setenv("AEGIS_EDGAR_MATRIX_OUT", "")

    assert module._default_matrix_out() is None


def test_main_writes_matrix_out_with_missing_derived_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_ingest_script()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "pilot_universe": [
                    {"ticker": "AAA", "cik": "1", "status": "active"},
                    {"ticker": "BBB", "cik": "2", "status": "active"},
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_out = tmp_path / "matrix.json"

    class FakeClient:
        def fetch_submissions(self, cik: str, *, force: bool = False) -> dict[str, object]:
            return {
                "name": f"issuer {cik}",
                "tickers": ["AAA" if cik == "1" else "BBB"],
                "formerNames": [],
                "filings": {"recent": {"filingDate": ["2024-02-16"]}},
            }

        def fetch_companyfacts(self, cik: str, *, force: bool = False) -> dict[str, object]:
            us_gaap: dict[str, object] = {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 100.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                            }
                        ]
                    }
                }
            }
            if cik == "1":
                us_gaap["DepreciationDepletionAndAmortization"] = {
                    "units": {
                        "USD": [
                            {
                                "val": 20.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-16",
                                "end": "2023-12-31",
                            }
                        ]
                    }
                }
            return {"facts": {"us-gaap": us_gaap}}

    monkeypatch.setattr(module, "SecEdgarClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgar_pit_ingest.py",
            "--meta",
            str(meta_path),
            "--matrix-out",
            str(matrix_out),
            "--coverage-out",
            str(tmp_path / "coverage.json"),
        ],
    )

    assert module.main() == 0

    loaded = json.loads(matrix_out.read_text(encoding="utf-8"))
    assert loaded["matrix"]["AAA"]["OperatingIncomeLoss"]["2023"] is True
    assert loaded["matrix"]["AAA"]["Ebitda"]["2023"] is True
    assert loaded["matrix"]["BBB"]["OperatingIncomeLoss"]["2023"] is True
    assert loaded["matrix"]["BBB"]["Ebitda"]["2023"] is False
