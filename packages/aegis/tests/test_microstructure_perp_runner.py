from __future__ import annotations

from typing import Any

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec
from aegis.microstructure_perp_runner import run_microstructure_perp_from_spec


def _bars(*, event_rate: float = 0.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closes = [100.0, 101.0, 103.0, 105.0, 100.0, 96.0, 98.0, 101.0, 97.0, 94.0]
    for index, close in enumerate(closes):
        rows.append(
            {
                "symbol": "BTC/USDT:USDT",
                "timestamp": 1_700_000_000 + index * 14_400,
                "close": close,
                "open_interest": max(1.0, 1_000.0 - index * 35.0),
                "funding_rate": 0.0002,
                "buy_volume": 10.0,
                "sell_volume": 30.0,
                "order_book_event_rate_per_hour": event_rate,
                "survivor_status": "active",
            }
        )
    rows.extend(
        {
            "symbol": "DELISTED/USDT:USDT",
            "timestamp": 1_700_000_000 + index * 14_400,
            "close": max(1.0, 20.0 - index),
            "open_interest": max(1.0, 500.0 - index * 20.0),
            "funding_rate": -0.0001,
            "buy_volume": 25.0,
            "sell_volume": 10.0,
            "order_book_event_rate_per_hour": event_rate,
            "survivor_status": "delisted",
        }
        for index in range(10)
    )
    return rows


def _spec(observations: list[dict[str, Any]]) -> HypothesisSpec:
    return HypothesisSpec(
        key="microstructure_unit",
        hypothesis_type="event",
        universe=("BTC/USDT:USDT", "DELISTED/USDT:USDT"),
        predeclared_signals=("funding_sign", "oi_price_divergence", "orderflow_imbalance"),
        params={
            "observations": observations,
            "grid": {
                "funding_abs_bps": [1.0, 2.0],
                "imbalance_abs": [0.2],
                "oi_drop_abs": [0.02],
                "score_threshold": [1, 2],
            },
            "locked_oos_fraction": 0.8,
            "fold_count": 4,
            "pbo_splits": 4,
            "annualization_periods": 365 * 6,
        },
        cost_model={
            "fee_bps": 1.0,
            "slippage_bps": 2.0,
            "funding_bps_per_period": 0.0,
            "funding_label": "perp funding debited from observations",
        },
        benchmark="buy_and_hold",
        data_source="synthetic_offline_microstructure_fixture",
        trial_count_n=8,
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=True,
        ),
        survivor_light=True,
    )


def test_microstructure_runner_reports_fdr_pbo_and_full_cost_safety() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars()))

    assert payload["status"] == "OK"
    assert payload["candidate_count_n"] == 8
    assert payload["multiple_testing"]["method"] == "BH-FDR + CSCV_PBO"
    assert "pbo_after_survivors" in payload["multiple_testing"]
    assert payload["safety"]["t_plus_1_execution"] is True
    assert payload["safety"]["perp_funding_counted"] is True
    assert payload["standard_metrics"]["net_cost"] > 0
    assert "buy_and_hold" in payload["benchmark_metrics"]
    assert "DELISTED/USDT:USDT" in payload["universe"]["usable_symbols"]


def test_microstructure_runner_data_blocks_high_orderbook_event_rate() -> None:
    payload = run_microstructure_perp_from_spec(_spec(_bars(event_rate=20_000.0)))

    assert payload["status"] == "INSUFFICIENT"
    assert "data-block" in payload["reason"]
    assert "BTC/USDT:USDT" in payload["safety"]["excluded_data_blocked_symbols"]
