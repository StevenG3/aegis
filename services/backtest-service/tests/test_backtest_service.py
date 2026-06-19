from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
from starlette.testclient import TestClient


def load_service_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in ("app", "data", "factor_ic", "funding_arb", "strategies", "walk_forward"):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("backtest_service_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app"] = module
    spec.loader.exec_module(module)
    return module


def sample_frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = []
    value = 100.0
    for item in range(rows):
        if item < 30:
            value -= 0.5
        elif item < 70:
            value += 1.2
        elif item < 90:
            value -= 1.5
        else:
            value += 0.8
        close.append(value)
    frame = pd.DataFrame(
        {
            "Open": [value - 0.5 for value in close],
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [1000.0 + item for item in range(rows)],
        },
        index=index,
    )
    return frame


def base_payload() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "source": "binance",
        "timeframe": "1d",
        "start": "2023-01-01",
        "end": "2023-05-01",
        "strategy": "ma_cross",
        "params": {"fast": 5, "slow": 10, "trend": 20},
        "cash": 10000,
        "commission": 0.001,
    }


class FakeFundingExchange:
    def __init__(self, market_type: str) -> None:
        self.market_type = market_type

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        base = int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000)
        rates = [0.0005, 0.0004, 0.0003, -0.0002, 0.0006, 0.0004, 0.0001, -0.0001]
        rows = [
            {"timestamp": base + index * 8 * 60 * 60 * 1000, "fundingRate": rate}
            for index, rate in enumerate(rates)
        ]
        lower = since or 0
        filtered = [row for row in rows if int(row["timestamp"]) >= lower]
        return filtered[: limit or len(filtered)]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[object]]:
        base = int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000)
        lower = since or 0
        rows: list[list[object]] = []
        for index in range(80):
            timestamp = base + index * 60 * 60 * 1000
            if timestamp < lower:
                continue
            spot_close = 100 + index * 0.2
            basis = 0.1 if self.market_type == "swap" else 0.0
            close = spot_close + basis
            rows.append([timestamp, close - 0.1, close + 0.2, close - 0.2, close, 1000])
        return rows[: limit or len(rows)]


def fake_ccxt_module() -> SimpleNamespace:
    return SimpleNamespace(
        binance=lambda config: FakeFundingExchange("spot"),
        binanceusdm=lambda config: FakeFundingExchange("swap"),
        okx=lambda config: FakeFundingExchange(str(config.get("options", {}).get("defaultType"))),
        bybit=lambda config: FakeFundingExchange(str(config.get("options", {}).get("defaultType"))),
    )


def test_strategies_lists_defaults() -> None:
    module = load_service_app()
    response = TestClient(module.app).get("/strategies")
    assert response.status_code == 200
    assert response.json() == [
        {"name": "ma_cross", "default_params": {"fast": 20, "slow": 50, "trend": 200}}
    ]


def test_strategy_plugin_loader_adds_external_strategy(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "external.py").write_text(
        """
from backtesting import Strategy

class ExternalStrategy(Strategy):
    lookback = 5

    def init(self):
        pass

    def next(self):
        pass

STRATEGY = {
    "name": "external_fixture",
    "cls": ExternalStrategy,
    "default_params": {"lookback": 5, "threshold": 1.5, "enabled": True, "entry_mode": "state"},
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STRATEGY_PLUGINS_DIR", str(tmp_path))

    module = load_service_app()
    response = TestClient(module.app).get("/strategies")

    assert response.status_code == 200
    body = response.json()
    assert {"name": "ma_cross", "default_params": {"fast": 20, "slow": 50, "trend": 200}} in body
    assert {
        "name": "external_fixture",
        "default_params": {
            "lookback": 5,
            "threshold": 1.5,
            "enabled": True,
            "entry_mode": "state",
        },
    } in body
    assert "external_fixture" in module.STRATEGIES


def test_strategy_plugin_loader_skips_bad_plugin(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("raise RuntimeError('broken plugin')\n", encoding="utf-8")
    monkeypatch.setenv("STRATEGY_PLUGINS_DIR", str(tmp_path))

    module = load_service_app()
    response = TestClient(module.app).get("/strategies")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "ma_cross", "default_params": {"fast": 20, "slow": 50, "trend": 200}}
    ]


def test_strategy_plugin_loader_does_not_override_builtin(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "conflict.py").write_text(
        """
from backtesting import Strategy

class ConflictingStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        pass

STRATEGY = {
    "name": "ma_cross",
    "cls": ConflictingStrategy,
    "default_params": {"fast": 1, "slow": 2, "trend": 3},
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STRATEGY_PLUGINS_DIR", str(tmp_path))

    module = load_service_app()
    response = TestClient(module.app).get("/strategies")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "ma_cross", "default_params": {"fast": 20, "slow": 50, "trend": 200}}
    ]
    assert module.DEFAULT_PARAMS["ma_cross"] == {"fast": 20, "slow": 50, "trend": 200}


def test_backtest_returns_stats_equity_and_trades(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())

    response = TestClient(module.app).post("/backtest", json=base_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body["stats"]) == {
        "return_pct",
        "buy_hold_return_pct",
        "max_drawdown_pct",
        "win_rate",
        "sharpe",
        "num_trades",
        "exposure_pct",
        "exit_breakdown",
    }
    assert body["equity_curve"]
    assert set(body["equity_curve"][0]) == {"date", "equity"}
    assert isinstance(body["trades"], list)
    assert "open_position" in body
    if body["trades"]:
        assert set(body["trades"][0]) == {
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "sl",
            "tp",
            "exit_reason",
            "entry_regime_up",
            "pnl_pct",
            "size",
        }


def test_funding_arb_endpoint_uses_funding_history_and_costs(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt_module())

    response = TestClient(module.app).post(
        "/funding-arb/backtest",
        json={
            "symbol": "BTCUSDT",
            "source": "binance",
            "timeframe": "1h",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "cash": 10000,
            "min_funding_bps": 2,
            "exit_funding_bps": 0,
            "taker_fee_bps": 1,
            "slippage_bps": 0.5,
            "basis_cost_bps": 0.25,
            "settlement_hours": 8,
            "cash_rate_annual": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    stats = body["stats"]
    assert body["data"]["funding_source"] == "ccxt.binance.fetch_funding_rate_history"
    assert body["market_symbols"] == {"spot": "BTC/USDT", "swap": "BTC/USDT:USDT"}
    assert stats["funding_events"] == 8
    assert stats["negative_funding_period_share"] == 0.25
    assert stats["gross_funding_return_pct"] > 0
    assert stats["fee_cost_pct"] > 0
    assert stats["slippage_cost_pct"] > 0
    assert stats["benchmark_return_pct"] == 0
    assert body["trades"]
    assert body["cost_model"]["taker_fee_bps"] == 1
    assert body["cost_model"]["basis_cost_bps"] == 0.25


def test_yfinance_multi_ticker_frame_selects_requested_symbol(monkeypatch) -> None:
    module = load_service_app()
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["MSFT", "SPY"]],
        names=["Price", "Ticker"],
    )
    rows = []
    for item in range(len(index)):
        rows.append(
            [
                200 + item,
                400 + item,
                201 + item,
                401 + item,
                199 + item,
                399 + item,
                200.5 + item,
                400.5 + item,
                1_000 + item,
                2_000 + item,
            ]
        )
    raw = pd.DataFrame(rows, index=index, columns=columns)
    fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: raw)
    monkeypatch.setattr(module.data_module.importlib, "import_module", lambda name: fake_yfinance)

    frame = module.data_module._load_yfinance_ohlcv(
        "MSFT",
        "1d",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 28, tzinfo=UTC),
    )

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame["Close"].iloc[0] == 200.5
    assert frame["Volume"].iloc[0] == 1000


def test_trade_exit_reason_classification() -> None:
    module = load_service_app()
    frame = pd.DataFrame(
        [
            {
                "EntryTime": pd.Timestamp("2023-01-01"),
                "ExitTime": pd.Timestamp("2023-01-02"),
                "EntryPrice": 100,
                "ExitPrice": 120.05,
                "SL": 90,
                "TP": 120,
                "ReturnPct": 0.2,
                "Size": 1,
                "Entry_RegimeUp": True,
            },
            {
                "EntryTime": pd.Timestamp("2023-01-03"),
                "ExitTime": pd.Timestamp("2023-01-04"),
                "EntryPrice": 100,
                "ExitPrice": 95.01,
                "SL": 95,
                "TP": 130,
                "ReturnPct": -0.05,
                "Size": 1,
                "Entry_RegimeUp": False,
            },
            {
                "EntryTime": pd.Timestamp("2023-01-05"),
                "ExitTime": pd.Timestamp("2023-01-06"),
                "EntryPrice": 100,
                "ExitPrice": 102,
                "SL": None,
                "TP": None,
                "ReturnPct": 0.02,
                "Size": 1,
                "Entry_RegimeUp": True,
            },
            {
                "EntryTime": pd.Timestamp("2023-01-07"),
                "ExitTime": pd.Timestamp("2023-01-10"),
                "EntryPrice": 100,
                "ExitPrice": 103,
                "SL": 90,
                "TP": 130,
                "ReturnPct": 0.03,
                "Size": 1,
                "Entry_RegimeUp": True,
            },
        ]
    )
    stats = {"_trades": frame}

    price_frame = pd.DataFrame(
        {"RegimeUp": [True, False, True, True]},
        index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05", "2023-01-10"]),
    )
    trades = module._trades(stats, price_frame)

    assert [trade.exit_reason for trade in trades] == [
        "take_profit",
        "stop_loss",
        "signal",
        "end_of_data",
    ]
    assert trades[0].entry_price == 100
    assert trades[0].exit_price == 120.05
    assert trades[0].sl == 90
    assert trades[0].tp == 120
    assert trades[1].entry_regime_up is False
    assert module._exit_breakdown(trades) == {
        "take_profit": 1,
        "stop_loss": 1,
        "signal": 1,
        "end_of_data": 1,
        "unknown": 0,
    }


def test_unclosed_position_is_not_reported_as_closed_trade_performance() -> None:
    module = load_service_app()
    stats = {
        "Return [%]": 7.2,
        "Buy & Hold Return [%]": 22.0,
        "Max. Drawdown [%]": -1.0,
        "Win Rate [%]": float("nan"),
        "Sharpe Ratio": 0.67,
        "Exposure Time [%]": 0.0,
        "_equity_curve": pd.DataFrame(
            {"Equity": [10_000.0, 10_000.0, 10_720.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
    }
    trades: list[object] = []

    open_position = module._open_position_info(stats, trades)
    response_stats = module._response_stats(stats, trades, open_position)

    assert open_position is not None
    assert round(open_position.mark_to_market_return_pct, 2) == 7.2
    assert response_stats.return_pct == 0.0
    assert response_stats.num_trades == 0
    assert response_stats.exposure_pct == 0.0
    assert response_stats.sharpe == 0.0
    assert response_stats.exit_breakdown == {
        "take_profit": 0,
        "stop_loss": 0,
        "signal": 0,
        "end_of_data": 0,
        "unknown": 0,
    }


def test_regime_alignment_is_backward_only() -> None:
    module = load_service_app()
    main = pd.DataFrame(
        {
            "Open": [10, 10, 10],
            "High": [11, 11, 11],
            "Low": [9, 9, 9],
            "Close": [10, 10, 10],
            "Volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
    )
    regime = pd.DataFrame(
        {
            "Open": [1, 1, 10],
            "High": [1, 1, 10],
            "Low": [1, 1, 10],
            "Close": [1, 1, 10],
            "Volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-06"]),
    )

    aligned = module._add_regime_column(main, regime, 2)

    assert bool(aligned.loc[pd.Timestamp("2023-01-03"), "RegimeUp"]) is False
    assert bool(aligned.loc[pd.Timestamp("2023-01-04"), "RegimeUp"]) is False
    # The 2023-01-06 regime turn must not leak into 2023-01-05.
    assert bool(aligned.loc[pd.Timestamp("2023-01-05"), "RegimeUp"]) is False


def test_unknown_strategy_returns_400(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())
    payload = dict(base_payload(), strategy="missing")

    response = TestClient(module.app).post("/backtest", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNKNOWN_STRATEGY"


def test_insufficient_data_returns_friendly_error(monkeypatch) -> None:
    module = load_service_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame(15))

    response = TestClient(module.app).post("/backtest", json=base_payload())

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INSUFFICIENT_BARS"


def test_excessive_date_range_returns_friendly_error(monkeypatch) -> None:
    module = load_service_app()

    def fail_if_called(*args):
        raise AssertionError("load_ohlcv should not be called for oversized requests")

    monkeypatch.setattr(module.data_module, "load_ohlcv", fail_if_called)
    payload = dict(
        base_payload(),
        timeframe="1m",
        start="2023-01-01",
        end="2023-01-10",
    )

    response = TestClient(module.app).post("/backtest", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "TOO_MANY_BARS"
    assert detail["max_bars"] == 5000


def test_source_contains_no_mutating_exchange_calls() -> None:
    service_dir = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text() for path in service_dir.rglob("*.py"))
    blocked = [
        "_".join(("create", "order")),
        "_".join(("place", "order")),
        "can" + "cel",
        "with" + "draw",
    ]
    assert all(term not in source for term in blocked)
