from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from starlette.testclient import TestClient


def load_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in (
        "app",
        "data",
        "factor_ic",
        "funding_arb",
        "healthcheck",
        "strategies",
        "walk_forward",
    ):
        sys.modules.pop(name, None)
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("backtest_service_app_walk_forward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app_walk_forward"] = module
    spec.loader.exec_module(module)
    return module


def load_walk_forward():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    for name in ("strategies", "walk_forward"):
        sys.modules.pop(name, None)
    path = service_dir / "walk_forward.py"
    spec = importlib.util.spec_from_file_location("walk_forward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["walk_forward"] = module
    spec.loader.exec_module(module)
    return module


def sample_frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close: list[float] = []
    price = 100.0
    for item in range(rows):
        price *= 1 + 0.003 + math.sin(item / 13) * 0.001
        close.append(price)
    return pd.DataFrame(
        {
            "Open": [value * 0.999 for value in close],
            "High": [value * 1.002 for value in close],
            "Low": [value * 0.998 for value in close],
            "Close": close,
            "Volume": [1000.0 + item for item in range(rows)],
        },
        index=index,
    )


def test_walk_forward_endpoint_returns_oos_windows(monkeypatch: Any) -> None:
    module = load_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())

    response = TestClient(module.app).post(
        "/walk-forward",
        json={
            "symbol": "SPY",
            "source": "yfinance",
            "timeframe": "1d",
            "start": "2023-01-01",
            "end": "2023-09-30",
            "strategy": "ma_cross",
            "param_grid": [
                {"fast": 3, "slow": 8, "trend": 20},
                {"fast": 5, "slow": 12, "trend": 30},
            ],
            "train_bars": 100,
            "test_bars": 50,
            "step_bars": 50,
            "cash": 10000,
            "commission": 0.001,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert len(body["windows"]) >= 2
    first = body["windows"][0]
    assert first["train_period"]["end"] < first["test_period"]["start"]
    assert set(first["selected_params"]) == {"fast", "slow", "trend"}
    assert "no trading signal" in body["disclaimer"]
    assert "Walk-forward evaluation" in body["readable_report"]


def test_walk_forward_reports_insufficient_data_honestly() -> None:
    module = load_walk_forward()
    from backtesting import Strategy  # type: ignore[import-untyped]

    class NoopStrategy(Strategy):  # type: ignore[misc]
        def init(self) -> None:
            pass

        def next(self) -> None:
            pass

    report = module.run_walk_forward(
        sample_frame(rows=40),
        NoopStrategy,
        [{}],
        train_bars=30,
        test_bars=30,
    )

    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["summary"]["windows"] == 0


def test_walk_forward_endpoint_rejects_invalid_param_order(monkeypatch: Any) -> None:
    module = load_app()
    monkeypatch.setattr(module.data_module, "load_ohlcv", lambda *args: sample_frame())

    response = TestClient(module.app).post(
        "/walk-forward",
        json={
            "symbol": "SPY",
            "source": "yfinance",
            "timeframe": "1d",
            "start": "2023-01-01",
            "end": "2023-09-30",
            "strategy": "ma_cross",
            "param_grid": [{"fast": 20, "slow": 10, "trend": 30}],
            "train_bars": 100,
            "test_bars": 50,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "FAST_MUST_BE_LT_SLOW"
