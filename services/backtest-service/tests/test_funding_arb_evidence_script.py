from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_evidence_script():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "funding_arb_evidence.py"
    spec = importlib.util.spec_from_file_location("funding_arb_evidence_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["funding_arb_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


def test_periods_from_env_accepts_explicit_matrix(monkeypatch) -> None:
    module = load_evidence_script()
    monkeypatch.setenv(
        "FUNDING_ARB_EVIDENCE_PERIODS",
        "high|2024-01-01|2024-02-01|positive funding;low|2024-02-01|2024-03-01|chop",
    )

    periods = module._periods_from_env()

    assert [period.name for period in periods] == ["high", "low"]
    assert periods[0].start == "2024-01-01"
    assert periods[1].note == "chop"


def test_sensitivity_summary_marks_fragile_edges() -> None:
    module = load_evidence_script()

    summary = module._sensitivity_summary(
        [
            {
                "taker_fee_bps": 10,
                "slippage_bps": 2,
                "min_funding_bps": 3,
                "net_return_pct": 0.4,
            },
            {
                "taker_fee_bps": 10,
                "slippage_bps": 2,
                "min_funding_bps": 3,
                "net_return_pct": -0.1,
            },
            {
                "taker_fee_bps": 15,
                "slippage_bps": 5,
                "min_funding_bps": 8,
                "net_return_pct": -0.2,
            },
        ]
    )

    assert summary[0]["runs"] == 2
    assert summary[0]["positive_share"] == 0.5
    assert summary[0]["edge_fragile"] is False
    assert summary[1]["edge_fragile"] is True


def test_recommendation_never_auto_graduates() -> None:
    module = load_evidence_script()

    recommendation = module._recommendation(
        {
            "baseline_positive_share": 0.9,
            "sensitivity_positive_share": 0.8,
            "baseline_median_net_return_pct": 0.4,
            "sensitivity_median_net_return_pct": 0.2,
            "baseline_empty_entry_share": 0.0,
        }
    )

    assert recommendation["status"] == "small_paper_candidate_human_gate_only"
    assert "no auto-graduation" in recommendation["guardrail"]
