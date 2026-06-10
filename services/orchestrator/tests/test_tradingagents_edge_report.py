from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def load_edge_script(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "tradingagents_edge_report.py"
    sys.modules.pop("db", None)
    sys.modules.pop("tradingagents_edge_report_script", None)
    spec = importlib.util.spec_from_file_location("tradingagents_edge_report_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tradingagents_edge_report_script"] = module
    spec.loader.exec_module(module)
    return module


def insert_closed_outcome(
    module,
    *,
    scorecard_id: str,
    outcome_id: str,
    actor: str = "tg_1",
    source: str = "tradingagents",
    conviction: str = "0.70",
    closed_return_pct: str = "0.03000000",
    closed_realized_pnl: str = "3.00000000",
    opened_at: str = "2026-05-25T00:00:00+00:00",
    closed_at: str = "2026-05-26T00:00:00+00:00",
    factors: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "actor": actor,
        "symbol": "BTCUSDT",
        "action": "buy",
        "source": source,
        "conviction": conviction,
        "metadata": {
            "asset_type": "crypto",
            "heuristic_conviction": conviction,
            "calibrated_conviction": conviction,
            "origin": "paper_feedback_bootstrap",
        },
        "factors": factors or [
            {"name": "market", "direction": "support"},
            {"name": "news", "direction": "oppose"},
        ],
    }
    with module.connect() as conn:
        conn.execute(
            """
            insert into scorecards
              (scorecard_id, actor, symbol, action, source, payload_json, created_at, expires_at)
            values (?,?,?,?,?,?,?,?)
            """,
            (
                scorecard_id,
                actor,
                "BTCUSDT",
                "buy",
                source,
                json.dumps(payload),
                opened_at,
                closed_at,
            ),
        )
        conn.execute(
            """
            insert into scorecard_outcomes
              (outcome_id, scorecard_id, actor, symbol, source, action,
               opened_intent_id, opened_at, opened_qty, opened_avg_cost,
               opened_cost_basis, status, closed_at, closed_realized_pnl,
               closed_return_pct, notes)
            values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
            """,
            (
                outcome_id,
                scorecard_id,
                actor,
                "BTCUSDT",
                source,
                "buy",
                f"intent-{outcome_id}",
                opened_at,
                "0.00100000",
                "100000.00000000",
                "100.00000000",
                "closed",
                closed_at,
                closed_realized_pnl,
                closed_return_pct,
            ),
        )
        conn.commit()


def test_build_report_compares_net_return_to_btc_hold(monkeypatch, tmp_path: Path) -> None:
    module = load_edge_script(monkeypatch, tmp_path)
    insert_closed_outcome(
        module,
        scorecard_id="sc-good",
        outcome_id="out-good",
        conviction="0.85",
        closed_return_pct="0.03000000",
        factors=[{"name": "market", "direction": "support"}],
    )
    insert_closed_outcome(
        module,
        scorecard_id="sc-bad",
        outcome_id="out-bad",
        conviction="0.55",
        closed_return_pct="-0.01000000",
        factors=[{"name": "market", "direction": "support"}],
    )

    def fake_btc_hold(opened_at: datetime, closed_at: datetime) -> tuple[float, float]:
        assert opened_at.tzinfo == UTC
        assert closed_at.tzinfo == UTC
        return 100.0, 101.0

    report = module.build_report(
        actor="tg_1",
        source="tradingagents",
        benchmark="BTC/USDT",
        benchmark_source="fixture",
        costs=module.CostModel(fee_bps=10, slippage_bps=2, funding_bps=0),
        min_n=30,
        price_fetcher=fake_btc_hold,
    )

    assert report["sample_sufficiency"]["closed_outcomes_n"] == 2
    assert report["sample_sufficiency"]["verdict"] == "INSUFFICIENT_DATA"
    assert report["summary"]["edge_claim"] == "NO_CLAIM_INSUFFICIENT_DATA"
    assert report["summary"]["benchmark_available_n"] == 2
    assert report["summary"]["total_net_return_pct"] == 0.0152
    assert report["summary"]["total_alpha_vs_btc_pct"] == -0.0048
    assert report["recommendation"]["status"] == "insufficient_data_continue_bootstrap"


def test_conviction_and_analyst_attribution_use_alpha_direction(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_edge_script(monkeypatch, tmp_path)
    insert_closed_outcome(
        module,
        scorecard_id="sc-high",
        outcome_id="out-high",
        conviction="0.90",
        closed_return_pct="0.05000000",
        factors=[
            {"name": "market", "direction": "support"},
            {"name": "news", "direction": "oppose"},
        ],
    )
    insert_closed_outcome(
        module,
        scorecard_id="sc-low",
        outcome_id="out-low",
        conviction="0.60",
        closed_return_pct="-0.02000000",
        factors=[
            {"name": "market", "direction": "support"},
            {"name": "news", "direction": "oppose"},
        ],
    )

    report = module.build_report(
        actor=None,
        source="tradingagents",
        benchmark="BTC/USDT",
        benchmark_source="fixture",
        costs=module.CostModel(fee_bps=0, slippage_bps=0, funding_bps=0),
        bucket_min_n=1,
        analyst_min_n=1,
        price_fetcher=lambda opened_at, closed_at: (100.0, 100.0),
    )

    buckets = {
        item["bucket"]: item for item in report["conviction_calibration"]["items"]
    }
    assert buckets["0.50-0.65"]["alpha_win_rate"] == 0.0
    assert buckets["0.80-1.01"]["alpha_win_rate"] == 1.0
    assert report["conviction_calibration"]["monotonic_alpha_win_rate"] is True

    analysts = {
        (item["analyst"], item["direction"]): item
        for item in report["analyst_attribution"]["items"]
    }
    assert analysts[("market", "support")]["directional_hit_rate_vs_btc"] == 0.5
    assert analysts[("market", "support")]["preliminary_label"] == "possible_noise"
    assert analysts[("news", "oppose")]["directional_hit_rate_vs_btc"] == 0.5
