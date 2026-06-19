from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
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


def insert_scorecard(
    module,
    *,
    scorecard_id: str,
    actor: str = "fixture_actor",
    symbol: str = "BTCUSDT",
    action: str = "buy",
    source: str = "tradingagents",
    conviction: str = "0.70",
    created_at: str = "2026-05-25T00:30:00+00:00",
    factors: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "actor": actor,
        "symbol": symbol,
        "action": action,
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
                symbol,
                action,
                source,
                json.dumps(payload),
                created_at,
                (datetime.fromisoformat(created_at) + timedelta(hours=24)).isoformat(),
            ),
        )
        conn.commit()


def test_forward_prices_use_first_bar_after_recommendation() -> None:
    module = load_edge_script_no_db()
    created_at = datetime(2026, 5, 25, 0, 30, tzinfo=UTC)
    rows = [
        [int(datetime(2026, 5, 25, 0, 0, tzinfo=UTC).timestamp() * 1000), 0, 0, 0, 99, 0],
        [int(datetime(2026, 5, 25, 1, 0, tzinfo=UTC).timestamp() * 1000), 0, 0, 0, 100, 0],
        [int(datetime(2026, 5, 26, 1, 0, tzinfo=UTC).timestamp() * 1000), 0, 0, 0, 110, 0],
    ]

    prices = module._forward_prices_from_ohlcv(rows, created_at, 24, "BTC/USDT")

    assert prices.entry_time == datetime(2026, 5, 25, 1, 0, tzinfo=UTC)
    assert prices.exit_time == datetime(2026, 5, 26, 1, 0, tzinfo=UTC)
    assert prices.entry_price == 100
    assert prices.exit_price == 110


def test_build_report_scores_recommendations_against_buy_hold_and_cash(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_edge_script(monkeypatch, tmp_path)
    insert_scorecard(module, scorecard_id="sc-buy", action="buy", conviction="0.85")
    insert_scorecard(
        module,
        scorecard_id="sc-sell",
        action="sell",
        conviction="0.55",
        created_at="2026-05-25T02:30:00+00:00",
    )

    def fake_prices(symbol: str, created_at: datetime, horizon_hours: int):
        assert symbol == "BTCUSDT"
        entry = created_at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return module.ForwardPrices(
            entry_time=entry,
            exit_time=entry + timedelta(hours=horizon_hours),
            entry_price=100.0,
            exit_price=110.0,
        )

    report = module.build_report(
        actor=None,
        source="tradingagents",
        benchmark_source="fixture",
        horizon_hours=24,
        costs=module.CostModel(
            fee_bps=10,
            slippage_bps=2,
            funding_bps_per_8h=1,
            cash_rate_annual=0.04,
        ),
        min_n=30,
        price_fetcher=fake_prices,
    )

    assert report["verdict"] == "INSUFFICIENT"
    assert report["sample_counts"]["recommendations_loaded"] == 2
    assert report["sample_counts"]["evaluated_n"] == 2
    rows = report["private_rows"]
    assert rows[0]["strategy_return"] == 0.0976
    assert rows[0]["buy_hold_return"] == 0.0976
    assert rows[1]["strategy_return"] < 0
    assert rows[1]["excess_vs_buy_hold"] < 0
    assert report["overall"]["cash"]["median"] > 0
    assert report["sanitized_public_summary"]["contains_actor_or_recommendation_rows"] is False


def test_min_sample_gate_prevents_edge_claim(monkeypatch, tmp_path: Path) -> None:
    module = load_edge_script(monkeypatch, tmp_path)
    for index in range(5):
        insert_scorecard(module, scorecard_id=f"sc-{index}")

    report = module.build_report(
        actor=None,
        source="tradingagents",
        benchmark_source="fixture",
        horizon_hours=24,
        costs=module.CostModel(0, 0, 0, 0),
        min_n=30,
        price_fetcher=lambda symbol, created_at, horizon_hours: module.ForwardPrices(
            entry_time=created_at + timedelta(hours=1),
            exit_time=created_at + timedelta(hours=25),
            entry_price=100,
            exit_price=120,
        ),
    )

    assert report["overall"]["status"] == "INSUFFICIENT"
    assert report["verdict"] == "INSUFFICIENT"


def test_bucket_and_analyst_fdr_are_reported(monkeypatch, tmp_path: Path) -> None:
    module = load_edge_script(monkeypatch, tmp_path)
    base = datetime(2026, 5, 25, 0, 30, tzinfo=UTC)
    for index in range(12):
        insert_scorecard(
            module,
            scorecard_id=f"good-{index}",
            conviction="0.90",
            created_at=(base + timedelta(hours=index * 2)).isoformat(),
            factors=[{"name": "trend", "direction": "support"}],
        )
    for index in range(12):
        insert_scorecard(
            module,
            scorecard_id=f"bad-{index}",
            action="sell",
            conviction="0.60",
            created_at=(base + timedelta(hours=100 + index * 2)).isoformat(),
            factors=[{"name": "macro", "direction": "support"}],
        )

    def fake_prices(symbol: str, created_at: datetime, horizon_hours: int):
        entry = created_at + timedelta(hours=1)
        if created_at.hour < 12 and created_at.day == 25:
            exit_price = 105
        else:
            exit_price = 103
        return module.ForwardPrices(entry, entry + timedelta(hours=horizon_hours), 100, exit_price)

    report = module.build_report(
        actor=None,
        source="tradingagents",
        benchmark_source="fixture",
        horizon_hours=24,
        costs=module.CostModel(0, 0, 0, 0),
        min_n=20,
        bucket_min_n=10,
        analyst_min_n=10,
        price_fetcher=fake_prices,
    )

    assert report["overall"]["status"] == "TESTED"
    assert any(row["kind"] == "confidence" for row in report["buckets"])
    assert any(row["kind"] == "analyst" for row in report["analysts"])
    assert all("bh_fdr_discovery" in row for row in report["buckets"])
    assert all("bh_fdr_discovery" in row for row in report["analysts"])


def test_sign_test_is_one_sided_for_positive_excess() -> None:
    module = load_edge_script_no_db()

    assert module._positive_sign_test_p_value([-1.0, -0.5, -0.1]) == 1.0
    assert module._positive_sign_test_p_value([1.0, 0.5, 0.1]) < 0.20


def load_edge_script_no_db():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "tradingagents_edge_report.py"
    sys.modules.pop("tradingagents_edge_report_script_no_db", None)
    spec = importlib.util.spec_from_file_location(
        "tradingagents_edge_report_script_no_db", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tradingagents_edge_report_script_no_db"] = module
    spec.loader.exec_module(module)
    return module
