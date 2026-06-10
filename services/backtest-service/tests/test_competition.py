from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient


def load_competition():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    sys.modules.pop("competition", None)
    path = service_dir / "competition.py"
    spec = importlib.util.spec_from_file_location("competition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["competition"] = module
    spec.loader.exec_module(module)
    return module


def load_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in (
        "app",
        "competition",
        "data",
        "factor_ic",
        "funding_arb",
        "healthcheck",
        "strategies",
        "walk_forward",
    ):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("backtest_service_app_competition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app_competition"] = module
    spec.loader.exec_module(module)
    return module


def entry(
    strategy: str,
    *,
    verdict: str = "PASS",
    median_return: float,
    beat_bh_share: float,
    sharpe: float,
    max_dd: float,
    bottom_streak: int = 0,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "params": {"fast": 20, "slow": 50},
        "healthcheck": {"verdict": verdict},
        "key_metrics": {
            "median_return": median_return,
            "beat_bh_share": beat_bh_share,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "exit_breakdown": {"signal": 3},
        },
        "history": {"bottom_streak": bottom_streak},
    }


def test_blocked_healthcheck_is_forced_to_bottom_even_with_best_score() -> None:
    module = load_competition()

    leaderboard = module.rank_strategies(
        [
            entry(
                "blocked",
                verdict="BLOCK",
                median_return=80,
                beat_bh_share=1,
                sharpe=5,
                max_dd=5,
            ),
            entry("passing", median_return=2, beat_bh_share=0.4, sharpe=0.5, max_dd=20),
        ],
        promote_top_n=1,
    )

    assert [row["strategy"] for row in leaderboard] == ["passing", "blocked"]
    assert leaderboard[0]["status"] == "promote_candidate"
    assert leaderboard[1]["healthcheck_verdict"] == "BLOCK"
    assert leaderboard[1]["status"] == "retire_candidate"


def test_ranking_key_prioritizes_median_and_beat_buy_hold_share() -> None:
    module = load_competition()

    leaderboard = module.rank_strategies(
        [
            entry("higher_sharpe", median_return=4, beat_bh_share=0.4, sharpe=3, max_dd=5),
            entry("better_median_beat", median_return=10, beat_bh_share=0.8, sharpe=0.2, max_dd=8),
        ],
        promote_top_n=1,
    )

    assert leaderboard[0]["strategy"] == "better_median_beat"
    assert leaderboard[0]["score"] > leaderboard[1]["score"]


def test_ranking_accepts_beat_benchmark_share_alias() -> None:
    module = load_competition()

    leaderboard = module.rank_strategies(
        [
            {
                "strategy": "funding_arb",
                "params": {"min_funding_bps": 3},
                "healthcheck_verdict": "PASS_WITH_WARN",
                "key_metrics": {
                    "median_return": 1,
                    "beat_benchmark_share": 1,
                    "sharpe": 1,
                    "max_dd": 2,
                },
            }
        ],
        promote_top_n=0,
    )

    assert leaderboard[0]["key_metrics"]["beat_bh_share"] == 1
    assert leaderboard[0]["status"] == "hold"


def test_promote_hold_and_long_term_retire_statuses() -> None:
    module = load_competition()

    leaderboard = module.rank_strategies(
        [
            entry("winner", median_return=12, beat_bh_share=0.9, sharpe=1.2, max_dd=8),
            entry(
                "warned",
                verdict="PASS_WITH_WARN",
                median_return=10,
                beat_bh_share=0.9,
                sharpe=1.2,
                max_dd=8,
            ),
            entry(
                "stale_loser",
                median_return=-1,
                beat_bh_share=0.1,
                sharpe=-0.5,
                max_dd=30,
                bottom_streak=3,
            ),
        ],
        promote_top_n=2,
    )

    by_strategy = {row["strategy"]: row for row in leaderboard}
    assert by_strategy["winner"]["status"] == "promote_candidate"
    assert by_strategy["warned"]["status"] == "hold"
    assert by_strategy["stale_loser"]["status"] == "retire_candidate"


def test_strategy_competition_endpoint() -> None:
    module = load_app()

    response = TestClient(module.app).post(
        "/strategy-competition",
        json={
            "entries": [
                entry("alpha", median_return=6, beat_bh_share=0.7, sharpe=1, max_dd=10),
                entry(
                    "beta",
                    verdict="BLOCK",
                    median_return=30,
                    beat_bh_share=1,
                    sharpe=4,
                    max_dd=5,
                ),
            ],
            "promote_top_n": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body[0]["strategy"] == "alpha"
    assert body[0]["rank"] == 1
    assert body[0]["status"] == "promote_candidate"
    assert body[1]["status"] == "retire_candidate"


def test_latest_competition_returns_unavailable_when_summary_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_app()
    monkeypatch.setenv("COMPETITION_LATEST_PATH", str(tmp_path / "missing.json"))

    response = TestClient(module.app).get("/competition/latest")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "reason": "no competition run yet",
        "entries": [],
    }


def test_latest_competition_returns_sanitized_leaderboard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_app()
    path = tmp_path / "competition_latest.json"
    monkeypatch.setenv("COMPETITION_LATEST_PATH", str(path))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "universe": {"symbols": ["IBM", "MSFT"], "periods": ["recent"]},
        "disclaimer": "competition candidates only; no auto graduation, no trading",
        "entries": [
            {
                "strategy": "golden_cross_2r",
                "params": {"param_set": "base"},
                "rank": 1,
                "score": 0.42,
                "healthcheck_verdict": "PASS",
                "status": "promote_candidate",
                "key_metrics": {
                    "median_return": 6.5,
                    "beat_bh_share": 0.7,
                    "sharpe": 1.1,
                    "max_dd": 12.0,
                },
                "private_notes": "must not be included",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    response = TestClient(module.app).get("/competition/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["generated_at"] == payload["generated_at"]
    assert body["universe"] == payload["universe"]
    assert body["disclaimer"] == payload["disclaimer"]
    assert body["entries"] == [
        {
            "strategy": "golden_cross_2r",
            "params": {"param_set": "base"},
            "rank": 1,
            "score": 0.42,
            "healthcheck_verdict": "PASS",
            "status": "promote_candidate",
            "key_metrics": {
                "median_return": 6.5,
                "beat_bh_share": 0.7,
                "sharpe": 1.1,
                "max_dd": 12.0,
            },
        }
    ]
    assert "private_notes" not in body["entries"][0]
