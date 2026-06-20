from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "polymarket_5m_firstpass_evidence.py"
    spec = importlib.util.spec_from_file_location("polymarket_5m_firstpass_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["polymarket_5m_firstpass_script"] = module
    spec.loader.exec_module(module)
    return module


class FakePolymarketClient:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def iter_closed_markets(
        self,
        *,
        limit: int,
        max_markets: int,
        sleep_seconds: float,
        order: str,
        ascending: bool,
    ) -> list[dict[str, object]]:
        _ = limit, max_markets, sleep_seconds, order, ascending
        return [
            {
                "conditionId": "condition-1",
                "slug": "btc-updown-5m-1800000000",
                "question": "Bitcoin Up or Down - synthetic",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["1", "0"]',
                "clobTokenIds": '["up-token", "down-token"]',
                "endDate": "2027-01-15T08:05:00Z",
                "closedTime": "2027-01-15 08:05:20+00",
                "closed": True,
            },
            {
                "conditionId": "condition-2",
                "slug": "not-btc",
                "question": "Some other market",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["1", "0"]',
                "clobTokenIds": '["yes-token", "no-token"]',
                "endDate": "2027-01-15T08:05:00Z",
                "closed": True,
            },
        ]


class FakeExchange:
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float | int]]:
        _ = symbol, timeframe, limit
        start = 1_800_000_000_000
        rows = [
            [start + index * 60_000, 100_000.0, 100_000.0, 100_000.0, 100_000.0 + index * 30]
            for index in range(8)
        ]
        return [row for row in rows if int(row[0]) >= since][:limit]


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


def test_script_main_writes_private_firstpass_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setenv("POLYMARKET_5M_MAX_MARKETS", "2")
    monkeypatch.setattr(module, "PolymarketDataApiClient", FakePolymarketClient)
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=lambda _: None))
    fake_ccxt = SimpleNamespace(binanceusdm=lambda config: FakeExchange())
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        text = str(getattr(request, "full_url", ""))
        price = 0.82 if "up-token" in text else 0.18
        return FakeResponse(
            {
                "history": [
                    {"t": 1_800_000_300 - 120, "p": price},
                    {"t": 1_800_000_300 - 30, "p": min(0.95, price + 0.05)},
                ]
            }
        )

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    json_path = Path(summary["json"])
    assert json_path.is_relative_to(private_root / "incubating" / "olympus61")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["coverage"]["gamma_closed_markets_fetched"] == 2
    assert payload["coverage"]["btc_5m_markets"] == 1
    assert payload["coverage"]["aligned_observations"] == 1
    assert payload["spec"]["trial_n"] == 48
    assert payload["report"]["optimistic_boundary"]["optimistic_only"] is True
