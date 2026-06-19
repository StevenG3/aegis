from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def load_funding_arb():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in ("funding_arb", "data"):
        sys.modules.pop(name, None)
    path = service_dir / "funding_arb.py"
    spec = importlib.util.spec_from_file_location("funding_arb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["funding_arb"] = module
    spec.loader.exec_module(module)
    return module


def events(
    rates: list[float],
    *,
    spot_start: float = 100.0,
    perp_start: float = 100.2,
    basis_drift: float = 0.0,
) -> list[dict[str, object]]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index, rate in enumerate(rates):
        spot = spot_start + index
        perp = perp_start + index + basis_drift * index
        rows.append(
            {
                "time": start + timedelta(hours=8 * index),
                "funding_rate": rate,
                "spot_close": spot,
                "perp_close": perp,
            }
        )
    return rows


def test_t_plus_one_entry_skips_signal_event_funding() -> None:
    module = load_funding_arb()
    config = module.FundingArbConfig(
        symbol="BTCUSDT",
        source="binance",
        start="2024-01-01",
        end="2024-01-04",
        cash=10_000,
        min_funding_bps=5,
        exit_funding_bps=-100,
        taker_fee_bps=0,
        slippage_bps=0,
        cash_rate_annual=0,
        max_holding_events=2,
    )

    result = module._simulate(
        events([0.0100, 0.0001, 0.0001, 0.0001]),
        config,
        module._cost_model(config),
        module._symbols("BTCUSDT"),
    )

    trade = result["trades"][0]
    assert trade["signal_time"] == "2024-01-01T00:00:00+00:00"
    assert trade["entry_time"] == "2024-01-01T08:00:00+00:00"
    assert trade["funding_return_pct"] < 0.05


def test_short_perp_receives_positive_funding_and_pays_negative_funding() -> None:
    module = load_funding_arb()
    config = module.FundingArbConfig(
        symbol="BTCUSDT",
        source="binance",
        start="2024-01-01",
        end="2024-01-04",
        cash=10_000,
        min_funding_bps=1,
        exit_funding_bps=-100,
        taker_fee_bps=0,
        slippage_bps=0,
        cash_rate_annual=0,
        max_holding_events=3,
    )

    positive = module._simulate(
        events([0.0002, 0.0003, 0.0003, 0.0003]),
        config,
        module._cost_model(config),
        module._symbols("BTCUSDT"),
    )
    negative = module._simulate(
        events([0.0002, -0.0003, -0.0003, -0.0003]),
        config,
        module._cost_model(config),
        module._symbols("BTCUSDT"),
    )

    assert positive["stats"]["gross_funding_return_pct"] > 0
    assert negative["stats"]["gross_funding_return_pct"] < 0


def test_double_leg_costs_and_basis_risk_reduce_result() -> None:
    module = load_funding_arb()
    base = module.FundingArbConfig(
        symbol="BTCUSDT",
        source="binance",
        start="2024-01-01",
        end="2024-01-04",
        cash=10_000,
        min_funding_bps=1,
        exit_funding_bps=-100,
        taker_fee_bps=0,
        slippage_bps=0,
        cash_rate_annual=0,
        max_holding_events=2,
    )
    costly = module.FundingArbConfig(
        **{**base.__dict__, "taker_fee_bps": 10, "slippage_bps": 2, "basis_cost_bps": 5}
    )

    cheap_result = module._simulate(
        events([0.0005, 0.0005, 0.0005, 0.0005], basis_drift=0.0),
        base,
        module._cost_model(base),
        module._symbols("BTCUSDT"),
    )
    costly_result = module._simulate(
        events([0.0005, 0.0005, 0.0005, 0.0005], basis_drift=0.5),
        costly,
        module._cost_model(costly),
        module._symbols("BTCUSDT"),
    )

    assert costly_result["stats"]["fee_cost_pct"] > 0
    assert costly_result["stats"]["slippage_cost_pct"] > 0
    assert costly_result["stats"]["basis_cost_pct"] > 0
    assert costly_result["stats"]["net_return_pct"] < cheap_result["stats"]["net_return_pct"]
    assert costly_result["stats"]["worst_trade_basis_move_pct"] < 0


def test_research_report_uses_cash_benchmark_and_bh_fdr() -> None:
    module = load_funding_arb()
    rows = events([0.0006 for _ in range(260)])
    report = module.run_funding_arb_research(
        {"BTCUSDT": rows},
        source="binance",
        start="2024-01-01",
        end="2024-04-01",
        base_config=module.FundingArbConfig(
            symbol="BTCUSDT",
            source="binance",
            start="2024-01-01",
            end="2024-04-01",
            taker_fee_bps=0,
            slippage_bps=0,
            cash_rate_annual=0,
        ),
        research_config=module.FundingResearchConfig(
            train_events=80,
            test_events=30,
            step_events=30,
            min_total_events=180,
            min_locked_oos_trades=1,
        ),
    )

    assert len(report["predeclared_grid"]) == 8
    assert report["benchmark"] == "risk-free cash, not buy-and-hold"
    assert report["fdr"]["method"].startswith("Benjamini-Hochberg")
    assert report["search_space_n"] == 8
    assert report["verdict"] in {"ROBUST_CARRY", "NO_ROBUST_EDGE"}


def test_research_report_marks_short_history_insufficient() -> None:
    module = load_funding_arb()

    report = module.run_funding_arb_research(
        {"BTCUSDT": events([0.0005 for _ in range(20)])},
        source="binance",
        start="2024-01-01",
        end="2024-01-10",
    )

    assert report["verdict"] == "INSUFFICIENT"
    assert report["tested_candidates"] == 0


def test_sign_test_is_one_sided_for_positive_cash_excess() -> None:
    module = load_funding_arb()

    assert module._sign_test_p_value([-1.0, -0.5, -0.25, -0.1]) == 1.0
    assert module._sign_test_p_value([1.0, 0.5, 0.25, 0.1]) < 0.10
