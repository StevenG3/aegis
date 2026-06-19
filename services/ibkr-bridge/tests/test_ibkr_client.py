from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibkr_client import IBKRClient, IBKRConfig, PlaceOrderRequest, _is_live_gateway_port


class FakeEvent:
    def __init__(self) -> None:
        self._handlers: list[Callable[..., None]] = []

    def __iadd__(self, handler: Callable[..., None]) -> FakeEvent:
        self._handlers.append(handler)
        return self

    def __isub__(self, handler: Callable[..., None]) -> FakeEvent:
        if handler in self._handlers:
            self._handlers.remove(handler)
        return self

    def emit(self, *args: object) -> None:
        for handler in list(self._handlers):
            handler(*args)


class FakeIB:
    def __init__(self) -> None:
        self.connected = False
        self.qualified: list[object] = []
        self.placed: list[tuple[object, object]] = []
        self.disconnected = False
        self.positionEvent = FakeEvent()
        self.positionEndEvent = FakeEvent()
        self.req_positions_calls = 0
        self.cancel_positions_calls = 0
        self.connect_readonly: bool | None = None
        self.connect_account: str | None = None
        self.current_market_data_type = 1
        self.market_data_type_requests: list[int] = []
        self.cancel_mkt_data_contracts: list[object] = []
        self.market_prices_by_type: dict[int, dict[str, Decimal | None]] = {
            1: {"NVDA": Decimal("452.50"), "MSFT": Decimal("430.00")}
        }
        self.seed_positions = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol="NVDA"),
                position=10.0,
                avgCost=450.25,
            ),
            SimpleNamespace(
                contract=SimpleNamespace(symbol="MSFT"),
                position=5.0,
                avgCost=420.00,
            ),
        ]

    def connect(
        self,
        host: str,
        port: int,
        clientId: int,
        timeout: float,
        readonly: bool = False,
        account: str = "",
    ) -> bool:  # noqa: N803
        _ = host, port, clientId, timeout
        self.connect_readonly = readonly
        self.connect_account = account
        self.connected = True
        return True

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    def qualifyContracts(self, contract: object) -> list[object]:  # noqa: N802
        self.qualified.append(contract)
        return [contract]

    def placeOrder(self, contract: object, order: object) -> object:  # noqa: N802
        self.placed.append((contract, order))
        return SimpleNamespace(
            order=SimpleNamespace(orderId=123),
            orderStatus=SimpleNamespace(status="Filled", filled=Decimal("2"), remaining=0),
            fills=[
                SimpleNamespace(
                    execution=SimpleNamespace(
                        shares=Decimal("2"),
                        price=Decimal("451.25"),
                        time="2026-05-28T00:00:00Z",
                    ),
                    commissionReport=SimpleNamespace(commission=Decimal("1.00")),
                )
            ],
        )

    def reqMktData(self, contract: object, *args: object, **kwargs: object) -> object:  # noqa: N802
        _ = args, kwargs
        symbol = str(getattr(contract, "symbol", "")).upper()
        price = self.market_prices_by_type.get(self.current_market_data_type, {}).get(
            symbol, Decimal("452.50")
        )
        return SimpleNamespace(
            contract=contract,
            last=price,
            close=None,
            bid=None,
            ask=None,
            marketPrice=lambda: price if price is not None else Decimal("-1"),
        )

    def sleep(self, seconds: float) -> None:
        assert seconds in {1.0, 3.0}

    def reqMarketDataType(self, market_data_type: int) -> None:  # noqa: N802
        self.current_market_data_type = market_data_type
        self.market_data_type_requests.append(market_data_type)

    def cancelMktData(self, contract: object) -> None:  # noqa: N802
        self.cancel_mkt_data_contracts.append(contract)

    def accountSummary(self, account: str = "") -> list[object]:  # noqa: N802
        account_code = account or "DU123"
        return [
            SimpleNamespace(
                account=account_code,
                tag="NetLiquidation",
                value="10000",
                currency="USD",
            ),
            SimpleNamespace(
                account=account_code,
                tag="TotalCashValue",
                value="1000",
                currency="USD",
            ),
            SimpleNamespace(
                account=account_code,
                tag="GrossPositionValue",
                value="9000",
                currency="USD",
            ),
        ]

    def reqPositions(self) -> None:  # noqa: N802
        self.req_positions_calls += 1
        for position in self.seed_positions:
            self.positionEvent.emit(position)
        self.positionEndEvent.emit()

    def cancelPositions(self) -> None:  # noqa: N802
        self.cancel_positions_calls += 1


def test_connect_disconnect_and_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(
        IBKRConfig(host="host.docker.internal", port=4002, client_id=7, timeout_sec=3.0)
    )
    assert client.is_ready() is False
    client.connect()
    assert client.is_ready() is True
    client.disconnect()
    assert client.is_ready() is False


def test_connect_uses_read_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig(account_code="DU123"))
    client.connect()
    assert client._ib.connect_readonly is True  # type: ignore[union-attr]
    assert client._ib.connect_account == "DU123"  # type: ignore[union-attr]


def test_place_market_order_normalizes_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(
            symbol=symbol, exchange=exchange, currency=currency
        ),
    )
    monkeypatch.setattr(
        "ibkr_client.MarketOrder",
        lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity),
    )
    client = IBKRClient(
        IBKRConfig(host="host.docker.internal", port=4002, client_id=7, timeout_sec=3.0)
    )
    client.connect()
    result = client.place_order(
        PlaceOrderRequest(
            idempotency_key="abc",
            symbol="NVDA",
            side="buy",
            order_type="market",
            quantity=Decimal("2"),
            limit_price=None,
            time_in_force="GTC",
        )
    )
    assert result["id"] == "ibkr-123"
    assert result["status"] == "filled"
    assert result["avg_price"] == "451.25"
    assert result["filled_qty"] == "2"
    assert result["fills"][0]["fee_asset"] == "USD"


def test_duplicate_idempotency_returns_same_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol),
    )
    monkeypatch.setattr(
        "ibkr_client.MarketOrder",
        lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity),
    )
    client = IBKRClient(IBKRConfig())
    client.connect()
    request = PlaceOrderRequest(
        idempotency_key="dup",
        symbol="MSFT",
        side="sell",
        order_type="market",
        quantity=Decimal("1"),
        limit_price=None,
        time_in_force="GTC",
    )
    first = client.place_order(request)
    second = client.place_order(request)
    assert second == first


def test_limit_order_requires_limit_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    with pytest.raises(ValueError, match="limit_price"):
        client.place_order(
            PlaceOrderRequest(
                idempotency_key="limit-missing",
                symbol="NVDA",
                side="buy",
                order_type="limit",
                quantity=Decimal("1"),
                limit_price=None,
                time_in_force="GTC",
            )
        )


def test_live_gateway_port_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    assert _is_live_gateway_port(7496) is True
    assert "live" in caplog.text.lower()


def test_live_port_refused_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_instantiated() -> NoReturn:
        raise AssertionError("IBKR network client should not be created")

    monkeypatch.setattr("ibkr_client.IB", fail_if_instantiated)
    client = IBKRClient(IBKRConfig(port=7496, allow_live_port=False))
    with pytest.raises(RuntimeError, match="LIVE_PORT_NOT_AUTHORIZED"):
        client.connect()


def test_live_port_4001_refused_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_instantiated() -> NoReturn:
        raise AssertionError("IBKR network client should not be created")

    monkeypatch.setattr("ibkr_client.IB", fail_if_instantiated)
    client = IBKRClient(IBKRConfig(port=4001, allow_live_port=False))
    with pytest.raises(RuntimeError, match="LIVE_PORT_NOT_AUTHORIZED"):
        client.connect()


def test_live_port_connects_when_authorized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    caplog.set_level(logging.WARNING)
    client = IBKRClient(IBKRConfig(port=7496, allow_live_port=True))
    client.connect()
    assert client.is_ready() is True
    assert "IBKR_AUDIT live_port_authorized" in caplog.text


def test_paper_port_logs_audit_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    caplog.set_level(logging.INFO)
    client = IBKRClient(IBKRConfig(port=4002, allow_live_port=False))
    client.connect()
    assert client.is_ready() is True
    assert "IBKR_AUDIT paper_port" in caplog.text


def test_positions_returns_list_when_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    result = client.positions()
    assert isinstance(result, list)
    assert len(result) == 2
    symbols = {p["symbol"] for p in result}
    assert "NVDA" in symbols
    assert "MSFT" in symbols
    nvda = next(p for p in result if p["symbol"] == "NVDA")
    assert nvda["qty"] == "10.00000000"
    assert nvda["avg_cost"] == "450.25000000"


def test_positions_raises_when_not_connected() -> None:
    client = IBKRClient(IBKRConfig())
    with pytest.raises(RuntimeError, match="not connected"):
        client.positions()


def test_positions_skips_zero_qty(monkeypatch: pytest.MonkeyPatch) -> None:
    class ZeroQtyIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            self.seed_positions = [
                SimpleNamespace(
                    contract=SimpleNamespace(symbol="AAPL"),
                    position=0.0,
                    avgCost=150.0,
                ),
                SimpleNamespace(
                    contract=SimpleNamespace(symbol="NVDA"),
                    position=5.0,
                    avgCost=450.0,
                ),
            ]

    monkeypatch.setattr("ibkr_client.IB", ZeroQtyIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    result = client.positions()
    assert len(result) == 1
    assert result[0]["symbol"] == "NVDA"


def test_positions_skips_empty_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptySymbolIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            self.seed_positions = [
                SimpleNamespace(
                    contract=SimpleNamespace(symbol=""),
                    position=10.0,
                    avgCost=100.0,
                ),
                SimpleNamespace(
                    contract=SimpleNamespace(symbol="MSFT"),
                    position=3.0,
                    avgCost=420.0,
                ),
            ]

    monkeypatch.setattr("ibkr_client.IB", EmptySymbolIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    result = client.positions()
    assert len(result) == 1
    assert result[0]["symbol"] == "MSFT"


def test_position_event_populates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptySeedIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            self.seed_positions = []

    monkeypatch.setattr("ibkr_client.IB", EmptySeedIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    client._ib.positionEvent.emit(  # type: ignore[union-attr]
        SimpleNamespace(
            contract=SimpleNamespace(symbol="nvda"),
            position=10,
            avgCost=450.25,
        )
    )
    assert client.positions() == [
        {"symbol": "NVDA", "qty": "10.00000000", "avg_cost": "450.25000000"}
    ]
    assert client._ib.placed == []  # type: ignore[union-attr]


def test_position_event_zero_qty_removes_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    assert any(p["symbol"] == "NVDA" for p in client.positions())
    client._ib.positionEvent.emit(  # type: ignore[union-attr]
        SimpleNamespace(
            contract=SimpleNamespace(symbol="NVDA"),
            position=0,
            avgCost=450.25,
        )
    )
    assert all(p["symbol"] != "NVDA" for p in client.positions())


def test_positions_ready_false_before_first_event() -> None:
    client = IBKRClient(IBKRConfig())
    assert client.positions_ready() is False


def test_positions_ready_true_after_event(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoInitialEventIB(FakeIB):
        def reqPositions(self) -> None:  # noqa: N802
            self.req_positions_calls += 1

    monkeypatch.setattr("ibkr_client.IB", NoInitialEventIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    assert client.positions_ready() is False
    client._ib.positionEvent.emit(  # type: ignore[union-attr]
        SimpleNamespace(contract=SimpleNamespace(symbol="NVDA"), position=1, avgCost=2)
    )
    assert client.positions_ready() is True


def test_disconnect_clears_cache_and_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    assert client.positions_ready() is True
    assert client.positions()
    client.disconnect()
    assert client.positions_ready() is False
    with pytest.raises(RuntimeError, match="not connected"):
        client.positions()


def test_positions_does_not_call_reqPositions_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    fake_ib = client._ib
    assert fake_ib.req_positions_calls == 1
    client.positions()
    client.positions()
    assert fake_ib.req_positions_calls == 1


def test_positions_returns_copy_on_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    first = client.positions()
    first[0]["qty"] = "999"
    second = client.positions()
    assert second[0]["qty"] == "10.00000000"


def test_snapshot_returns_account_summary_positions_and_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ibkr_client.IB", FakeIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(
            symbol=symbol, exchange=exchange, currency=currency, conId=f"{symbol}-1"
        ),
    )
    client = IBKRClient(IBKRConfig(account_code="DU123"))
    client.connect()

    snapshot = client.snapshot()

    assert snapshot["ok"] is True
    assert snapshot["source"] == "ibkr-bridge"
    assert snapshot["positions_count"] == 2
    assert snapshot["market_data_type"] == 1
    assert snapshot["market_data_fallback_used"] is False
    account_summary = snapshot["account_summary"]
    assert isinstance(account_summary, dict)
    assert account_summary["DU123"]["NetLiquidation"]["numeric"] == "10000"
    positions = snapshot["positions"]
    assert isinstance(positions, list)
    nvda = next(p for p in positions if p["symbol"] == "NVDA")
    assert nvda["market_price"] == "452.50"
    assert nvda["market_value"] == "4525.0000000000"
    assert snapshot["gross_position_value_computed"] == "6675.0000000000"
    market_data = snapshot["market_data"]
    assert isinstance(market_data, list)
    assert {row["symbol"] for row in market_data} == {"NVDA", "MSFT"}
    fake_ib = client._ib
    assert fake_ib is not None
    assert fake_ib.market_data_type_requests == [1]
    assert len(fake_ib.cancel_mkt_data_contracts) == 2
    assert fake_ib.placed == []


def test_snapshot_falls_back_from_realtime_to_delayed_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedOnlyIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            self.market_prices_by_type = {
                1: {"NVDA": None, "MSFT": None},
                3: {"NVDA": Decimal("453.00"), "MSFT": Decimal("431.00")},
            }

    monkeypatch.setattr("ibkr_client.IB", DelayedOnlyIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(
            symbol=symbol, exchange=exchange, currency=currency, conId=f"{symbol}-1"
        ),
    )
    client = IBKRClient(IBKRConfig())
    client.connect()

    snapshot = client.snapshot()

    assert snapshot["market_data_type"] == 3
    assert snapshot["market_data_fallback_used"] is True
    positions = snapshot["positions"]
    assert isinstance(positions, list)
    nvda = next(p for p in positions if p["symbol"] == "NVDA")
    assert nvda["market_price"] == "453.00"
    assert nvda["market_value"] == "4530.0000000000"
    fake_ib = client._ib
    assert fake_ib is not None
    assert fake_ib.market_data_type_requests == [1, 3]
    assert len(fake_ib.cancel_mkt_data_contracts) == 4


def test_snapshot_returns_fallback_metadata_when_market_data_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoMarketDataIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            self.market_prices_by_type = {
                1: {"NVDA": None, "MSFT": None},
                3: {"NVDA": None, "MSFT": None},
            }

    monkeypatch.setattr("ibkr_client.IB", NoMarketDataIB)
    monkeypatch.setattr(
        "ibkr_client.Stock",
        lambda symbol, exchange, currency: SimpleNamespace(
            symbol=symbol, exchange=exchange, currency=currency, conId=f"{symbol}-1"
        ),
    )
    client = IBKRClient(IBKRConfig())
    client.connect()

    snapshot = client.snapshot()

    assert snapshot["market_data_type"] == 3
    assert snapshot["market_data_fallback_used"] is True
    assert snapshot["gross_position_value_computed"] is None
    positions = snapshot["positions"]
    assert isinstance(positions, list)
    assert all(position["market_price"] is None for position in positions)
    assert all(position["market_value"] is None for position in positions)


def test_reconnect_rebuilds_position_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[FakeIB] = []

    class ReconnectIB(FakeIB):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)
            if len(instances) == 1:
                self.seed_positions = [
                    SimpleNamespace(
                        contract=SimpleNamespace(symbol="NVDA"),
                        position=10,
                        avgCost=450.25,
                    )
                ]
            else:
                self.seed_positions = [
                    SimpleNamespace(
                        contract=SimpleNamespace(symbol="MSFT"),
                        position=2,
                        avgCost=420,
                    )
                ]

    monkeypatch.setattr("ibkr_client.IB", ReconnectIB)
    client = IBKRClient(IBKRConfig())
    client.connect()
    assert client.positions() == [
        {"symbol": "NVDA", "qty": "10.00000000", "avg_cost": "450.25000000"}
    ]
    client.disconnect()
    assert client.positions_ready() is False
    client.connect()
    assert client.positions() == [
        {"symbol": "MSFT", "qty": "2.00000000", "avg_cost": "420.00000000"}
    ]
