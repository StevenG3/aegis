from __future__ import annotations

from aegis.btc_vrp_short_vol import (
    ShortVolVrpConfig,
    btc_vrp_data_blocked_report,
    run_btc_short_vol_vrp,
)


def _row(
    *,
    variant: str = "atm_straddle_7d",
    iv_ts: int = 1_000,
    expiry_ts: int = 2_000,
    implied_vol: float = 0.80,
    realized_vol: float = 0.60,
    variance_year_fraction: float = 7.0 / 365.0,
    option_spread_cost: float = 0.01,
    hedge_fee_cost: float = 0.005,
    hedge_slippage_cost: float = 0.005,
    funding_cost: float = 0.01,
    tail_loss: float = 0.0,
) -> dict[str, object]:
    return {
        "variant": variant,
        "iv_ts": iv_ts,
        "expiry_ts": expiry_ts,
        "implied_vol": implied_vol,
        "realized_vol": realized_vol,
        "variance_year_fraction": variance_year_fraction,
        "option_spread_cost": option_spread_cost,
        "hedge_fee_cost": hedge_fee_cost,
        "hedge_slippage_cost": hedge_slippage_cost,
        "funding_cost": funding_cost,
        "tail_loss": tail_loss,
    }


def test_short_vol_vrp_rejects_lookahead_iv_timestamp() -> None:
    report = run_btc_short_vol_vrp(
        [_row(iv_ts=2_000, expiry_ts=2_000)],
        config=ShortVolVrpConfig(),
    )
    assert report["state"] == "INSUFFICIENT"
    assert report["data_adequacy"] == "blocked"
    assert report["coverage"]["excluded_reasons"] == {"iv_timestamp_not_before_expiry": 1}


def test_short_vol_vrp_counts_hedge_funding_cost() -> None:
    low_funding = run_btc_short_vol_vrp(
        [_row(funding_cost=0.0), _row(iv_ts=1_100, expiry_ts=2_100, funding_cost=0.0)],
        config=ShortVolVrpConfig(),
    )
    high_funding = run_btc_short_vol_vrp(
        [_row(funding_cost=0.10), _row(iv_ts=1_100, expiry_ts=2_100, funding_cost=0.10)],
        config=ShortVolVrpConfig(),
    )
    assert high_funding["best_candidate"]["mean_net_return"] < low_funding["best_candidate"][
        "mean_net_return"
    ]


def test_short_vol_vrp_positive_premium_but_tail_unsafe() -> None:
    rows = [
        _row(
            variant="atm_straddle_7d",
            iv_ts=1_000 + index,
            expiry_ts=2_000 + index,
            implied_vol=5.00,
            realized_vol=0.20,
        )
        for index in range(5)
    ]
    rows.append(
        _row(
            variant="atm_straddle_7d",
            iv_ts=9_000,
            expiry_ts=10_000,
            implied_vol=10.00,
            realized_vol=0.30,
            tail_loss=3.00,
        )
    )
    report = run_btc_short_vol_vrp(rows, config=ShortVolVrpConfig(max_drawdown_limit=-0.30))
    assert report["state"] == "NO_EDGE"
    assert report["verdict"] == "PREMIUM_EXISTS_BUT_TAIL_UNSAFE"
    assert report["standard_metrics"]["mean_net_return"] > 0.0
    assert report["standard_metrics"]["max_drawdown"] < -0.30


def test_btc_vrp_blocked_report_sets_unlock_condition() -> None:
    report = btc_vrp_data_blocked_report(
        reason="missing 2020 crash IV",
        coverage={"dvol_rows": 0},
    )
    assert report["state"] == "INSUFFICIENT"
    assert report["data_adequacy"] == "blocked"
    assert report["unlock_condition"] == "missing 2020 crash IV"
