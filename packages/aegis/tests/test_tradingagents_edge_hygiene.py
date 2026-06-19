from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "tradingagents_edge_report.py"
    spec = importlib.util.spec_from_file_location(
        "tradingagents_edge_report_for_tests",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load tradingagents_edge_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tradingagents_zero_rows_fail_loud_not_insufficient(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    module = cast(Any, _load_script_module())

    def empty_loader(**_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(module, "_load_recommendations", empty_loader)
    monkeypatch.setattr(sys, "argv", ["tradingagents_edge_report.py", "--no-write"])

    assert module.main() == 2
    captured = capsys.readouterr()
    assert "EMPTY_DATA_SOURCE" in captured.err
    assert "not INSUFFICIENT" in captured.err


def test_tradingagents_positive_but_below_min_n_remains_insufficient() -> None:
    module = cast(Any, _load_script_module())
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    recommendation = module.Recommendation(
        scorecard_id="scorecard-1",
        actor="private_actor",
        symbol="BTC/USDT",
        action="buy",
        source="tradingagents",
        created_at=created_at,
        conviction=None,
        factors=(),
        metadata={},
    )

    def one_row_loader(**_kwargs: object) -> list[Any]:
        return [recommendation]

    def price_fetcher(_symbol: str, created: datetime, horizon_hours: int) -> Any:
        return module.ForwardPrices(
            entry_time=created + timedelta(hours=1),
            exit_time=created + timedelta(hours=horizon_hours + 1),
            entry_price=100.0,
            exit_price=101.0,
        )

    report = module.build_report(
        actor=None,
        source="tradingagents",
        benchmark_source="synthetic",
        horizon_hours=24,
        costs=module.CostModel(
            fee_bps=0.0,
            slippage_bps=0.0,
            funding_bps_per_8h=0.0,
            cash_rate_annual=0.0,
        ),
        min_n=2,
        bucket_min_n=2,
        analyst_min_n=2,
        price_fetcher=price_fetcher,
        recommendation_loader=one_row_loader,
    )

    assert report["verdict"] == "INSUFFICIENT"
    assert report["sample_counts"] == {
        "recommendations_loaded": 1,
        "evaluated_n": 1,
        "skipped_n": 0,
    }
