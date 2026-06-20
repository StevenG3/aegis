from __future__ import annotations

from aegis.polymarket_5m_firstpass import run_polymarket_5m_firstpass


def _observations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = 1_800_000_000
    for index in range(10):
        start = base + index * 300
        end = start + 300
        move = 120.0 if index % 2 == 0 else -120.0
        direction = "Up" if move > 0 else "Down"
        settlement = direction if index != 7 else ("Down" if direction == "Up" else "Up")
        up_price = 0.82 if direction == "Up" else 0.18
        down_price = 0.82 if direction == "Down" else 0.18
        rows.append(
            {
                "condition_id": f"condition-{index}",
                "slug": f"btc-updown-5m-{start}",
                "title": "Bitcoin Up or Down - synthetic",
                "start_ts": start,
                "end_ts": end,
                "settlement_direction": settlement,
                "btc_move_usd": move,
                "btc_direction": direction,
                "up_prices": [
                    {"timestamp": end - 120, "price": up_price},
                    {"timestamp": end - 30, "price": min(0.95, up_price + 0.05)},
                ],
                "down_prices": [
                    {"timestamp": end - 120, "price": down_price},
                    {"timestamp": end - 30, "price": min(0.95, down_price + 0.05)},
                ],
                "up_onchain_fills": [
                    {"timestamp": end - 115, "price": min(0.99, up_price + 0.01)}
                ],
                "down_onchain_fills": [
                    {"timestamp": end - 115, "price": min(0.99, down_price + 0.01)}
                ],
            }
        )
    return rows


def test_polymarket_5m_firstpass_reports_optimistic_boundary_and_fdr_pbo() -> None:
    payload = run_polymarket_5m_firstpass(_observations())

    assert payload["status"] == "OK"
    assert payload["candidate_count_n"] == 96
    assert payload["multiple_testing"]["method"] == "BH-FDR + CSCV_PBO"
    assert payload["optimistic_boundary"]["optimistic_only"] is False
    assert payload["optimistic_boundary"]["robust_or_edge_claim_allowed"] is False
    assert payload["optimistic_boundary"]["positive_verdict_ceiling"] == (
        "SUGGESTIVE_NEEDS_EXECUTION_VALIDATION"
    )
    assert "historical_depth_missing" in payload["optimistic_boundary"][
        "unmodeled_execution_costs"
    ]
    assert payload["coverage"]["market_count"] == 10
    assert payload["coverage"]["entry_count"] > 0
    assert payload["coverage"]["entry_count_by_price_source"]["onchain_fill"] > 0
    assert set(payload["coverage"]["entry_count_by_move_threshold"]) == {
        "50",
        "70",
        "100",
        "150",
    }
    assert payload["verdict"] in {"NO_EDGE", "SUGGESTIVE_NEEDS_EXECUTION_VALIDATION"}


def test_polymarket_5m_firstpass_insufficient_without_observations() -> None:
    payload = run_polymarket_5m_firstpass([])

    assert payload["status"] == "INSUFFICIENT"
    assert payload["verdict"] == "INSUFFICIENT"
    assert payload["coverage"]["market_count"] == 0


def test_polymarket_5m_firstpass_keeps_legacy_observed_only_grid() -> None:
    observations = _observations()
    for row in observations:
        row.pop("up_onchain_fills")
        row.pop("down_onchain_fills")

    payload = run_polymarket_5m_firstpass(observations)

    assert payload["candidate_count_n"] == 48
    assert payload["optimistic_boundary"]["optimistic_only"] is True
    assert payload["coverage"]["entry_count_by_price_source"]["onchain_fill"] == 0
