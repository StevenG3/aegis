from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from starlette.testclient import TestClient


def load_factor_ic():
    service_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(service_dir))
    sys.modules.pop("factor_ic", None)
    path = service_dir / "factor_ic.py"
    spec = importlib.util.spec_from_file_location("factor_ic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["factor_ic"] = module
    spec.loader.exec_module(module)
    return module


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
    spec = importlib.util.spec_from_file_location("backtest_service_app_factor_ic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_service_app_factor_ic"] = module
    spec.loader.exec_module(module)
    return module


def predictive_frame(rows: int = 180, *, drift: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close: list[float] = []
    price = 100.0 + drift
    for item in range(rows):
        step_return = 0.001 + item * 0.00005 + math.sin(item / 9) * 0.0002 + drift * 0.00001
        price *= 1 + step_return
        close.append(price)
    return pd.DataFrame(
        {
            "Open": [value * 0.999 for value in close],
            "High": [value * 1.002 for value in close],
            "Low": [value * 0.998 for value in close],
            "Close": close,
            "Volume": [10_000 + item * 11 + drift for item in range(rows)],
        },
        index=index,
    )


def test_time_series_factor_ic_reports_predictive_metrics_and_redundancy() -> None:
    module = load_factor_ic()

    report = module.evaluate_ohlcv_factor_ic(
        {"AAA": predictive_frame()},
        [
            {"name": "ret_1", "kind": "return", "window": 1},
            {"name": "ret_1_clone", "kind": "return", "window": 1},
            {"name": "volume_z_10", "kind": "volume_zscore", "window": 10},
        ],
        label_periods=1,
        groups=5,
        mode="time_series",
        ic_window=30,
        redundancy_threshold=0.9,
    )

    ret_1 = report["factors"]["ret_1"]
    assert ret_1["rank_ic"]["status"] == "OK"
    assert ret_1["rank_ic"]["n"] > 0
    assert ret_1["pearson_ic"]["status"] == "OK"
    assert ret_1["monotonicity"]["status"] == "OK"
    assert ret_1["monotonicity"]["is_monotonic"] is True
    assert ret_1["monotonicity"]["top_bottom_return"] > 0
    assert ret_1["autocorrelation"]["status"] == "OK"
    assert "no trading signal" in report["disclaimer"]

    pair = report["redundancy"]["high_correlation_pairs"][0]
    assert {pair["left"], pair["right"]} == {"ret_1", "ret_1_clone"}
    assert "ret_1_clone" in report["redundancy"]["suggested_drop"]
    assert "Factor IC evaluation" in report["readable_report"]


def test_cross_sectional_factor_ic_uses_per_period_rank_correlations() -> None:
    module = load_factor_ic()
    frames = {
        f"SYM{item}": predictive_frame(rows=90, drift=float(item * 10))
        for item in range(5)
    }

    report = module.evaluate_ohlcv_factor_ic(
        frames,
        [{"name": "ret_1", "kind": "return", "window": 1}],
        label_periods=1,
        groups=3,
        mode="cross_sectional",
        redundancy_threshold=0.9,
    )

    factor = report["factors"]["ret_1"]
    assert factor["rank_ic"]["status"] == "OK"
    assert factor["rank_ic"]["n"] > 0
    assert factor["monotonicity"]["status"] == "OK"


def test_insufficient_factor_ic_samples_are_reported_honestly() -> None:
    module = load_factor_ic()

    report = module.evaluate_ohlcv_factor_ic(
        {"AAA": predictive_frame(rows=12)},
        [{"name": "ret_1", "kind": "return", "window": 1}],
        label_periods=1,
        groups=5,
        mode="time_series",
        ic_window=30,
    )

    factor = report["factors"]["ret_1"]
    assert factor["rank_ic"]["status"] == "INSUFFICIENT_DATA"
    assert factor["edge"]["has_predictive_power"] is False
    assert "sample insufficient" in factor["edge"]["reason"]


def test_factor_ic_endpoint_loads_ohlcv_and_returns_report(monkeypatch: Any) -> None:
    module = load_app()

    def fake_load_ohlcv(
        symbol: str,
        source: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        assert symbol == "AAA"
        assert source == "yfinance"
        assert timeframe == "1d"
        assert start is not None
        assert end is not None
        return predictive_frame()

    monkeypatch.setattr(module.data_module, "load_ohlcv", fake_load_ohlcv)

    response = TestClient(module.app).post(
        "/factor-ic",
        json={
            "symbol": "AAA",
            "source": "yfinance",
            "timeframe": "1d",
            "start": "2024-01-01",
            "end": "2024-06-01",
            "factors": [{"name": "ret_1", "kind": "return", "window": 1}],
            "label_periods": 1,
            "groups": 5,
            "mode": "time_series",
            "ic_window": 30,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["symbols"] == ["AAA"]
    assert body["factors"]["ret_1"]["rank_ic"]["status"] == "OK"
    assert body["disclaimer"] == (
        "candidates-only factor evaluation; no trading signal or order path"
    )


def test_factor_ic_endpoint_rejects_missing_symbols() -> None:
    module = load_app()

    response = TestClient(module.app).post(
        "/factor-ic",
        json={
            "source": "yfinance",
            "timeframe": "1d",
            "start": "2024-01-01",
            "end": "2024-06-01",
            "factors": [{"name": "ret_1", "kind": "return", "window": 1}],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "SYMBOLS_REQUIRED"
