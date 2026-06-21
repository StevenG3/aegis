from __future__ import annotations

from aegis.calibrated_oos_gate import (
    CalibratedOosGateConfig,
    OosWindow,
    annualization_periods_for_instrument,
    evaluate_calibrated_oos_gate,
    signed_rank_p_value_greater,
)


def test_synthetic_true_edge_passes_calibrated_gate() -> None:
    windows = [
        OosWindow(
            candidate_score=0.45 + index * 0.01,
            static_score=0.05,
            buy_hold_score=0.0,
            candidate_return=0.03,
            static_return=0.0,
            buy_hold_return=-0.01,
        )
        for index in range(10)
    ]
    returns = [0.01, 0.012, 0.009, 0.011, 0.013, 0.01, 0.012, 0.009] * 6

    result = evaluate_calibrated_oos_gate(
        windows,
        returns,
        config=CalibratedOosGateConfig(
            min_windows=3,
            alpha=0.05,
            bootstrap_iterations=200,
            trial_count=4,
            annualization_periods=365,
        ),
    )

    assert result["verdict"] == "ROBUST_OOS_EDGE"
    assert all(check["passed"] for check in result["checks"])


def test_toy_no_edge_fails_with_specific_reasons() -> None:
    windows = [
        OosWindow(
            candidate_score=0.0,
            static_score=0.05 if index % 2 == 0 else -0.01,
            buy_hold_score=0.04,
            candidate_return=0.0,
            static_return=0.01,
            buy_hold_return=0.02,
        )
        for index in range(8)
    ]

    result = evaluate_calibrated_oos_gate(
        windows,
        [0.0, -0.002, 0.001, 0.0, -0.001, 0.0],
        config=CalibratedOosGateConfig(bootstrap_iterations=100, trial_count=8),
    )

    assert result["verdict"] == "NO_ROBUST_EDGE"
    assert "buy-and-hold" in str(result["reason"])
    assert "deflated Sharpe" in str(result["reason"])


def test_gate_does_not_require_every_window_to_win() -> None:
    windows = []
    returns = []
    for index in range(17):
        edge = -0.02 if index == 3 else 0.25
        windows.append(
            OosWindow(
                candidate_score=edge,
                static_score=0.0,
                buy_hold_score=0.0,
                candidate_return=0.02 if edge > 0 else -0.002,
                static_return=0.0,
                buy_hold_return=0.0,
            )
        )
        returns.append(0.02 if edge > 0 else -0.002)

    result = evaluate_calibrated_oos_gate(
        windows,
        returns,
        config=CalibratedOosGateConfig(
            min_windows=3,
            alpha=0.05,
            bootstrap_iterations=200,
            trial_count=4,
            annualization_periods=365,
        ),
    )

    assert result["verdict"] == "ROBUST_OOS_EDGE"
    assert "does not require every OOS window" in str(result["legacy_all_windows_note"])


def test_insufficient_window_count_is_not_forced_no_edge() -> None:
    result = evaluate_calibrated_oos_gate(
        [
            OosWindow(
                candidate_score=1.0,
                static_score=0.0,
                buy_hold_score=0.0,
                candidate_return=0.1,
                static_return=0.0,
                buy_hold_return=0.0,
            )
        ],
        [0.1],
    )

    assert result["verdict"] == "OOS_DATA_INSUFFICIENT"


def test_annualization_mapping_uses_crypto_365_and_intraday_bars() -> None:
    assert annualization_periods_for_instrument("spot", "1d") == 365
    assert annualization_periods_for_instrument("equity", "1d") == 252
    assert annualization_periods_for_instrument("spot", "4h") == 2190
    assert annualization_periods_for_instrument("spot", "1h") == 8760


def test_signed_rank_detects_positive_shift() -> None:
    assert signed_rank_p_value_greater([0.2, 0.1, -0.01, 0.3, 0.4]) < 0.10
