from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import cast

from aegis.mo_im_vol_gap_diagnostic import (
    IndexBar,
    IvObservation,
    OptionQuote,
    VolGapConfig,
    atm_iv_snapshot_from_quotes,
    implied_volatility_from_option,
    run_vol_gap_diagnostic,
)


def test_no_historical_iv_fails_closed_to_forward_collection() -> None:
    bars = _bars(count=80, daily_moves=(0.01, -0.01))

    result = run_vol_gap_diagnostic(bars, [], config=VolGapConfig(horizons=(20,)))

    assert result["verdict"] == "INSUFFICIENT"
    assert result["fdr_survivors"] == 0
    assert result["forward_collection"] == {
        "required": True,
        "target": "daily ATM MO IV plus 000852 OHLC",
        "private_path": "${AEGIS_STRATEGIES_ROOT}/incubating/olympus74/forward/",
    }


def test_iv_t_is_used_against_forward_rv_without_future_iv_leakage() -> None:
    bars = _bars(count=25, daily_moves=(0.03, -0.02))
    config = VolGapConfig(horizons=(20,), rv_methods=("close_to_close",), min_samples_per_cell=1)
    first_iv = [IvObservation(timestamp=bars[0].timestamp, annualized_iv=0.10, source="synthetic")]
    with_future_iv = [
        *first_iv,
        IvObservation(
            timestamp=bars[10].timestamp,
            annualized_iv=9.99,
            source="future_should_not_enter",
        ),
    ]

    baseline = run_vol_gap_diagnostic(bars, first_iv, config=config)
    mutated = run_vol_gap_diagnostic(bars, with_future_iv, config=config)

    assert baseline["summary"] == mutated["summary"]
    assert baseline["cells"] == mutated["cells"]


def test_low_iv_high_forward_rv_favors_buyer_in_gross_diagnostic() -> None:
    bars = _bars(count=90, daily_moves=(0.04, -0.035, 0.03, -0.025))
    observations = [
        IvObservation(timestamp=bar.timestamp, annualized_iv=0.05, source="synthetic")
        for bar in bars
    ]

    result = run_vol_gap_diagnostic(
        bars,
        observations,
        config=VolGapConfig(
            horizons=(20,),
            rv_methods=("close_to_close",),
            min_samples_per_cell=20,
            buyer_win_rate_threshold=0.90,
        ),
    )

    assert result["verdict"] == "GAP_FAVORS_BUYER"
    assert result["fdr_survivors"] == 1


def test_high_iv_low_forward_rv_favors_seller() -> None:
    bars = _bars(count=90, daily_moves=(0.002, -0.001))
    observations = [
        IvObservation(timestamp=bar.timestamp, annualized_iv=0.40, source="synthetic")
        for bar in bars
    ]

    result = run_vol_gap_diagnostic(
        bars,
        observations,
        config=VolGapConfig(
            horizons=(20,),
            rv_methods=("close_to_close",),
            min_samples_per_cell=20,
        ),
    )

    assert result["verdict"] == "GAP_FAVORS_SELLER"
    assert result["fdr_survivors"] == 0
    summary = cast(Mapping[str, object], result["summary"])
    assert summary["seller_positive_rate"] == 1.0


def test_atm_forward_snapshot_inverts_option_mid_quotes() -> None:
    quote = OptionQuote(
        option_type="call",
        strike=100.0,
        bid=7.8,
        ask=8.2,
        expiry=date(2026, 7, 17),
        underlying=100.0,
        as_of=date(2026, 6, 17),
    )

    iv = implied_volatility_from_option(quote, risk_free_rate=0.0)
    snapshot = atm_iv_snapshot_from_quotes([quote], risk_free_rate=0.0)

    assert iv is not None
    assert 0.50 < iv < 0.90
    assert snapshot["ok"] is True
    assert snapshot["iv_count"] == 1
    assert snapshot["source_quality"] == "forward_proxy_from_atm_mid_quotes"


def _bars(*, count: int, daily_moves: tuple[float, ...]) -> list[IndexBar]:
    close = 1000.0
    bars: list[IndexBar] = []
    for index in range(count):
        open_price = close
        move = daily_moves[index % len(daily_moves)]
        close = open_price * (1.0 + move)
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        bars.append(
            IndexBar(
                timestamp=20200101 + index,
                open=open_price,
                high=high,
                low=low,
                close=close,
            )
        )
    return bars
