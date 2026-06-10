from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from starlette.testclient import TestClient


def load_service_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in ("app", "data"):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("screener_service_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["screener_service_app"] = module
    spec.loader.exec_module(module)
    return module


def sample_rows(symbols: list[str]) -> list[dict[str, object]]:
    fixtures = {
        "AAA": {
            "symbol": "AAA",
            "name": "Alpha",
            "sector": "Tech",
            "industry": "Software",
            "trailing_pe": "12",
            "forward_pe": "10",
            "peg": "1.1",
            "price_to_book": "3",
            "dividend_yield": "0.01",
            "market_cap": "1000000000",
            "price": "50",
            "error": None,
        },
        "BBB": {
            "symbol": "BBB",
            "name": "Beta",
            "sector": "Tech",
            "industry": "Hardware",
            "trailing_pe": "30",
            "forward_pe": "24",
            "peg": "2.2",
            "price_to_book": "8",
            "dividend_yield": "0",
            "market_cap": "2000000000",
            "price": "80",
            "error": None,
        },
        "CCC": {
            "symbol": "CCC",
            "name": "Core",
            "sector": "Financials",
            "industry": "Banks",
            "trailing_pe": "8",
            "forward_pe": "9",
            "peg": None,
            "price_to_book": "1.2",
            "dividend_yield": "0.04",
            "market_cap": "3000000000",
            "price": "30",
            "error": None,
        },
        "BAD": {
            "symbol": "BAD",
            "name": None,
            "sector": None,
            "industry": None,
            "trailing_pe": None,
            "forward_pe": None,
            "peg": None,
            "price_to_book": None,
            "dividend_yield": None,
            "market_cap": None,
            "price": None,
            "error": "not found",
        },
    }
    return [fixtures[symbol] for symbol in symbols]


def test_screen_filters_sorts_and_aggregates(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "fetch_valuation", sample_rows)

    response = TestClient(module.app).post(
        "/screen",
        json={
            "universe": ["AAA", "BBB", "CCC"],
            "filters": {"max_pe": "20"},
            "sort_by": "trailing_pe",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["disclaimer"] == "valuation screen, candidates only, not a buy signal"
    assert [row["symbol"] for row in body["valuations"]] == ["CCC", "AAA"]
    assert body["sectors"] == [
        {
            "sector": "Financials",
            "count": 1,
            "valid_trailing_pe_count": 1,
            "median_trailing_pe": "8",
            "median_forward_pe": "9",
            "median_price_to_book": "1.2",
        },
        {
            "sector": "Tech",
            "count": 1,
            "valid_trailing_pe_count": 1,
            "median_trailing_pe": "12",
            "median_forward_pe": "10",
            "median_price_to_book": "3",
        },
    ]


def test_symbol_errors_are_returned_without_breaking_batch(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "fetch_valuation", sample_rows)

    response = TestClient(module.app).post("/screen", json={"universe": ["AAA", "BAD"]})

    assert response.status_code == 200
    body = response.json()
    assert [row["symbol"] for row in body["valuations"]] == ["AAA", "BAD"]
    assert body["errors"] == [sample_rows(["BAD"])[0]]


def test_empty_universe_is_friendly_error() -> None:
    module = load_service_app()
    response = TestClient(module.app).post("/screen", json={"universe": []})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]


def test_read_only_boundary_terms_absent() -> None:
    service_dir = Path(__file__).resolve().parents[1]
    terms = [
        ("create_", "order"),
        ("pl", "ace"),
        ("can", "cel"),
        ("with", "draw"),
        ("api_?", "key"),
        ("se", "cret"),
    ]
    pattern = "|".join(left + right for left, right in terms)
    completed = subprocess.run(
        ["grep", "-riE", pattern, str(service_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.stdout == ""
