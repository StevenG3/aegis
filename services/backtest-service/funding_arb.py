from __future__ import annotations

import importlib
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]

from data import TIMEFRAME_MS, DataLoadError

FundingSource = Literal["binance", "okx", "bybit"]

REQUEST_LIMIT = 1000
RETRY_COUNT = 2
DEFAULT_MAX_FUNDING_EVENTS = 1000


@dataclass(frozen=True)
class FundingArbConfig:
    symbol: str
    source: FundingSource
    start: date | datetime | str
    end: date | datetime | str
    timeframe: str = "1h"
    cash: float = 10_000.0
    min_funding_bps: float | None = None
    exit_funding_bps: float | None = None
    taker_fee_bps: float | None = None
    maker_fee_bps: float | None = None
    slippage_bps: float | None = None
    basis_cost_bps: float | None = None
    borrow_cost_bps_annual: float | None = None
    settlement_hours: float | None = None
    use_maker_fees: bool = False
    max_funding_events: int | None = None


@dataclass(frozen=True)
class FundingCostModel:
    taker_fee_bps: float
    maker_fee_bps: float
    active_fee_bps: float
    slippage_bps: float
    basis_cost_bps: float
    borrow_cost_bps_annual: float
    settlement_hours: float
    min_funding_bps: float
    exit_funding_bps: float
    use_maker_fees: bool
    basis_model: str

    @property
    def round_trip_cost_pct(self) -> float:
        per_side_bps = 2 * (self.active_fee_bps + self.slippage_bps)
        round_trip_bps = 2 * per_side_bps + self.basis_cost_bps
        return round_trip_bps / 100


@dataclass
class OpenPosition:
    entry_time: datetime
    entry_spot: float
    entry_perp: float
    entry_signal_bps: float
    periods_held: int = 0
    funding_return: float = 0.0
    borrow_cost: float = 0.0


def run_funding_arb_backtest(config: FundingArbConfig) -> dict[str, Any]:
    start_dt = _parse_date(config.start)
    end_dt = _parse_date(config.end)
    if end_dt <= start_dt:
        raise DataLoadError("end must be after start")
    if config.cash <= 0:
        raise DataLoadError("cash must be positive")
    if config.timeframe not in TIMEFRAME_MS:
        raise DataLoadError(f"unsupported timeframe: {config.timeframe}")

    costs = _cost_model(config)
    symbols = _symbols(config.symbol)
    funding_events = _load_funding_history(
        config.source,
        symbols.swap,
        start_dt,
        end_dt,
        config.max_funding_events or _env_int("FUNDING_ARB_MAX_EVENTS", DEFAULT_MAX_FUNDING_EVENTS),
    )
    spot_frame = _load_ohlcv(
        config.source,
        symbols.spot,
        "spot",
        config.timeframe,
        start_dt,
        end_dt,
    )
    perp_frame = _load_ohlcv(
        config.source,
        symbols.swap,
        "swap",
        config.timeframe,
        start_dt,
        end_dt,
    )
    aligned = _align_events(funding_events, spot_frame, perp_frame)
    if not aligned:
        raise DataLoadError("funding history and price bars could not be aligned")

    return _simulate(aligned, config, costs, symbols)


@dataclass(frozen=True)
class Symbols:
    spot: str
    swap: str


def _symbols(symbol: str) -> Symbols:
    normalized = symbol.strip().upper().replace("-", "/")
    if not normalized:
        raise DataLoadError("symbol is required")
    if ":" in normalized:
        swap = normalized
        spot = normalized.split(":", 1)[0]
        return Symbols(spot=spot, swap=swap)
    for quote in ("USDT", "USDC", "USD"):
        if normalized.endswith(quote) and "/" not in normalized and len(normalized) > len(quote):
            spot = f"{normalized[:-len(quote)]}/{quote}"
            return Symbols(spot=spot, swap=f"{spot}:{quote}")
    if "/" in normalized:
        quote = normalized.split("/", 1)[1]
        return Symbols(spot=normalized, swap=f"{normalized}:{quote}")
    raise DataLoadError(f"unsupported symbol format: {symbol}")


def _parse_date(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _cost_model(config: FundingArbConfig) -> FundingCostModel:
    taker_fee_bps = _value_or_env(config.taker_fee_bps, "FUNDING_ARB_TAKER_FEE_BPS", 10.0)
    maker_fee_bps = _value_or_env(config.maker_fee_bps, "FUNDING_ARB_MAKER_FEE_BPS", 2.0)
    slippage_bps = _value_or_env(config.slippage_bps, "FUNDING_ARB_SLIPPAGE_BPS", 2.0)
    basis_cost_bps = _value_or_env(config.basis_cost_bps, "FUNDING_ARB_BASIS_COST_BPS", 0.0)
    borrow_cost_bps_annual = _value_or_env(
        config.borrow_cost_bps_annual,
        "FUNDING_ARB_BORROW_COST_BPS_ANNUAL",
        0.0,
    )
    settlement_hours = _value_or_env(
        config.settlement_hours,
        "FUNDING_ARB_SETTLEMENT_HOURS",
        8.0,
    )
    min_funding_bps = _value_or_env(config.min_funding_bps, "FUNDING_ARB_MIN_FUNDING_BPS", 3.0)
    exit_funding_bps = _value_or_env(config.exit_funding_bps, "FUNDING_ARB_EXIT_FUNDING_BPS", 0.0)
    active_fee_bps = maker_fee_bps if config.use_maker_fees else taker_fee_bps
    if settlement_hours <= 0:
        raise DataLoadError("settlement_hours must be positive")
    return FundingCostModel(
        taker_fee_bps=max(taker_fee_bps, 0.0),
        maker_fee_bps=max(maker_fee_bps, 0.0),
        active_fee_bps=max(active_fee_bps, 0.0),
        slippage_bps=max(slippage_bps, 0.0),
        basis_cost_bps=max(basis_cost_bps, 0.0),
        borrow_cost_bps_annual=max(borrow_cost_bps_annual, 0.0),
        settlement_hours=settlement_hours,
        min_funding_bps=min_funding_bps,
        exit_funding_bps=exit_funding_bps,
        use_maker_fees=config.use_maker_fees,
        basis_model="actual aligned spot/perp close; residual basis_cost_bps applied on exit",
    )


def _value_or_env(value: float | None, env_name: str, default: float) -> float:
    if value is not None:
        return value
    return _env_float(env_name, default)


def _request_with_retry(func: Any, *args: object, **kwargs: object) -> Any:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.25 * (attempt + 1))
    raise DataLoadError(str(last_error) if last_error else "historical data request failed")


def _exchange(source: FundingSource, market_type: Literal["spot", "swap"]) -> Any:
    ccxt = importlib.import_module("ccxt")
    if source == "binance":
        factory_name = "binance" if market_type == "spot" else "binanceusdm"
        factory = getattr(ccxt, factory_name)
        return factory({"enableRateLimit": True, "timeout": 10_000})
    options = {"defaultType": market_type}
    factory = getattr(ccxt, source)
    return factory({"enableRateLimit": True, "timeout": 10_000, "options": options})


def _load_funding_history(
    source: FundingSource,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    max_events: int,
) -> list[dict[str, Any]]:
    exchange = _exchange(source, "swap")
    if not hasattr(exchange, "fetch_funding_rate_history"):
        raise DataLoadError(f"{source} does not expose funding rate history")
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while since < end_ms and len(rows) < max_events:
        batch = _request_with_retry(
            exchange.fetch_funding_rate_history,
            symbol,
            since=since,
            limit=min(REQUEST_LIMIT, max_events - len(rows)),
        )
        if not isinstance(batch, list) or not batch:
            break
        advanced = False
        for item in batch:
            if not isinstance(item, dict):
                continue
            timestamp = _event_timestamp(item)
            rate = _event_rate(item)
            if timestamp is None or rate is None:
                continue
            if timestamp < int(start_dt.timestamp() * 1000) or timestamp >= end_ms:
                continue
            if timestamp in seen:
                continue
            seen.add(timestamp)
            rows.append({"timestamp": timestamp, "fundingRate": rate})
            advanced = True
        last_ts = _event_timestamp(batch[-1]) if isinstance(batch[-1], dict) else None
        if last_ts is None:
            break
        next_since = last_ts + 1
        if not advanced or next_since <= since:
            break
        since = next_since
    rows.sort(key=lambda item: cast(int, item["timestamp"]))
    if not rows:
        raise DataLoadError(f"no real funding rate history returned for {symbol} on {source}")
    return rows


def _event_timestamp(item: dict[str, Any]) -> int | None:
    raw = item.get("timestamp")
    if raw is None and isinstance(item.get("info"), dict):
        info = cast(dict[str, Any], item["info"])
        raw = info.get("fundingTime") or info.get("time") or info.get("timestamp")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float | str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _event_rate(item: dict[str, Any]) -> float | None:
    raw = item.get("fundingRate")
    if raw is None:
        raw = item.get("rate")
    if raw is None and isinstance(item.get("info"), dict):
        info = cast(dict[str, Any], item["info"])
        raw = info.get("fundingRate") or info.get("rate")
    if isinstance(raw, int | float | str):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _load_ohlcv(
    source: FundingSource,
    symbol: str,
    market_type: Literal["spot", "swap"],
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    exchange = _exchange(source, market_type)
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = TIMEFRAME_MS[timeframe]
    rows: list[list[object]] = []
    seen: set[int] = set()
    while since < end_ms:
        batch = _request_with_retry(
            exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since=since,
            limit=REQUEST_LIMIT,
        )
        if not isinstance(batch, list) or not batch:
            break
        advanced = False
        for raw in batch:
            if not isinstance(raw, list) or len(raw) < 6:
                continue
            ts = _int_timestamp(raw[0])
            if ts is None or ts >= end_ms or ts in seen:
                continue
            seen.add(ts)
            rows.append(raw[:6])
            advanced = True
        last_ts = (
            _int_timestamp(batch[-1][0])
            if isinstance(batch[-1], list) and batch[-1]
            else None
        )
        if last_ts is None:
            break
        next_since = last_ts + step_ms
        if not advanced or next_since <= since:
            break
        since = next_since
    if not rows:
        raise DataLoadError(f"no {market_type} OHLCV returned for {symbol} on {source}")
    frame = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    frame = frame.set_index("ts").sort_index()
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise DataLoadError(f"{market_type} OHLCV had no usable rows for {symbol}")
    return cast(pd.DataFrame, frame)


def _int_timestamp(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _align_events(
    funding_events: list[dict[str, Any]],
    spot_frame: pd.DataFrame,
    perp_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    events = pd.DataFrame(funding_events)
    events["Date"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    events = events.sort_values("Date")
    spot = spot_frame[["Close"]].rename(columns={"Close": "spot_close"}).reset_index()
    spot = spot.rename(columns={spot.columns[0]: "Date"}).sort_values("Date")
    perp = perp_frame[["Close"]].rename(columns={"Close": "perp_close"}).reset_index()
    perp = perp.rename(columns={perp.columns[0]: "Date"}).sort_values("Date")
    aligned = pd.merge_asof(events, spot, on="Date", direction="backward")
    aligned = pd.merge_asof(aligned, perp, on="Date", direction="backward")
    aligned = aligned.dropna(subset=["fundingRate", "spot_close", "perp_close"])
    result: list[dict[str, Any]] = []
    for row in aligned.itertuples(index=False):
        result.append(
            {
                "time": cast(pd.Timestamp, row.Date).to_pydatetime(),
                "funding_rate": float(row.fundingRate),
                "spot_close": float(row.spot_close),
                "perp_close": float(row.perp_close),
            }
        )
    return result


def _simulate(
    events: list[dict[str, Any]],
    config: FundingArbConfig,
    costs: FundingCostModel,
    symbols: Symbols,
) -> dict[str, Any]:
    cash = config.cash
    realized = 0.0
    gross_funding = 0.0
    realized_basis = 0.0
    fee_cost = 0.0
    slippage_cost = 0.0
    basis_cost = 0.0
    borrow_cost = 0.0
    position: OpenPosition | None = None
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    negative_periods = 0
    held_periods = 0

    for event in events:
        timestamp = cast(datetime, event["time"])
        spot = cast(float, event["spot_close"])
        perp = cast(float, event["perp_close"])
        rate = cast(float, event["funding_rate"])
        if rate < 0:
            negative_periods += 1

        if position is not None:
            period_funding = cash * rate
            period_borrow = cash * (costs.borrow_cost_bps_annual / 10_000) * (
                costs.settlement_hours / (365 * 24)
            )
            realized += period_funding - period_borrow
            gross_funding += period_funding
            borrow_cost += period_borrow
            position.funding_return += period_funding / cash
            position.borrow_cost += period_borrow / cash
            position.periods_held += 1
            held_periods += 1

        should_exit = position is not None and rate * 10_000 < costs.exit_funding_bps
        if should_exit and position is not None:
            pnl = _basis_pnl(cash, position.entry_spot, position.entry_perp, spot, perp)
            exit_fee = _two_leg_cost(cash, costs.active_fee_bps)
            exit_slippage = _two_leg_cost(cash, costs.slippage_bps)
            residual_basis = cash * costs.basis_cost_bps / 10_000
            realized += pnl - exit_fee - exit_slippage - residual_basis
            realized_basis += pnl
            fee_cost += exit_fee
            slippage_cost += exit_slippage
            basis_cost += residual_basis
            trades.append(
                _trade(
                    position,
                    cash,
                    timestamp,
                    spot,
                    perp,
                    rate,
                    pnl,
                    exit_fee,
                    exit_slippage,
                    residual_basis,
                )
            )
            position = None

        if position is None and rate * 10_000 >= costs.min_funding_bps:
            entry_fee = _two_leg_cost(cash, costs.active_fee_bps)
            entry_slippage = _two_leg_cost(cash, costs.slippage_bps)
            realized -= entry_fee + entry_slippage
            fee_cost += entry_fee
            slippage_cost += entry_slippage
            position = OpenPosition(
                entry_time=timestamp,
                entry_spot=spot,
                entry_perp=perp,
                entry_signal_bps=rate * 10_000,
            )

        unrealized_basis = (
            _basis_pnl(cash, position.entry_spot, position.entry_perp, spot, perp)
            if position is not None
            else 0.0
        )
        equity_curve.append(
            {
                "date": timestamp.isoformat(),
                "equity": cash + realized + unrealized_basis,
                "funding_rate": rate,
                "funding_bps": rate * 10_000,
                "position_open": position is not None,
                "spot_close": spot,
                "perp_close": perp,
            }
        )

    if position is not None:
        last = events[-1]
        spot = cast(float, last["spot_close"])
        perp = cast(float, last["perp_close"])
        rate = cast(float, last["funding_rate"])
        timestamp = cast(datetime, last["time"])
        pnl = _basis_pnl(cash, position.entry_spot, position.entry_perp, spot, perp)
        exit_fee = _two_leg_cost(cash, costs.active_fee_bps)
        exit_slippage = _two_leg_cost(cash, costs.slippage_bps)
        residual_basis = cash * costs.basis_cost_bps / 10_000
        realized += pnl - exit_fee - exit_slippage - residual_basis
        realized_basis += pnl
        fee_cost += exit_fee
        slippage_cost += exit_slippage
        basis_cost += residual_basis
        trades.append(
            _trade(
                position,
                cash,
                timestamp,
                spot,
                perp,
                rate,
                pnl,
                exit_fee,
                exit_slippage,
                residual_basis,
                exit_reason="end_of_data",
            )
        )
        equity_curve.append(
            {
                "date": timestamp.isoformat(),
                "equity": cash + realized,
                "funding_rate": rate,
                "funding_bps": rate * 10_000,
                "position_open": False,
                "spot_close": spot,
                "perp_close": perp,
            }
        )

    net_return = realized / cash
    gross_return = (gross_funding + realized_basis) / cash
    return {
        "strategy": "funding_arb",
        "source": config.source,
        "symbol": config.symbol,
        "market_symbols": {"spot": symbols.spot, "swap": symbols.swap},
        "stats": {
            "return_pct": net_return * 100,
            "net_return_pct": net_return * 100,
            "gross_return_pct": gross_return * 100,
            "gross_funding_return_pct": gross_funding / cash * 100,
            "basis_return_pct": realized_basis / cash * 100,
            "fee_cost_pct": fee_cost / cash * 100,
            "slippage_cost_pct": slippage_cost / cash * 100,
            "basis_cost_pct": basis_cost / cash * 100,
            "borrow_cost_pct": borrow_cost / cash * 100,
            "annualized_return_pct": _annualized_return_pct(net_return, events),
            "sharpe": _sharpe(equity_curve, costs.settlement_hours),
            "max_drawdown_pct": _max_drawdown_pct(equity_curve),
            "num_trades": len(trades),
            "funding_events": len(events),
            "held_funding_events": held_periods,
            "exposure_pct": held_periods / len(events) * 100 if events else 0.0,
            "negative_funding_period_share": negative_periods / len(events) if events else 0.0,
            "held_negative_funding_period_share": _held_negative_share(trades),
            "benchmark_return_pct": 0.0,
            "beat_benchmark": net_return > 0,
            "net_cost_positive": net_return > 0,
            "exit_breakdown": _exit_breakdown(trades),
        },
        "cost_model": _cost_model_dict(costs),
        "equity_curve": _sample_equity_curve(equity_curve),
        "trades": trades,
        "data": {
            "funding_source": f"ccxt.{config.source}.fetch_funding_rate_history",
            "price_source": f"ccxt.{config.source}.fetch_ohlcv spot + swap",
            "start": events[0]["time"].isoformat(),
            "end": events[-1]["time"].isoformat(),
        },
        "disclaimer": "backtest-only candidate; no trading integration or graduation action",
    }


def _basis_pnl(
    cash: float,
    entry_spot: float,
    entry_perp: float,
    exit_spot: float,
    exit_perp: float,
) -> float:
    spot_pnl = cash * (exit_spot / entry_spot - 1)
    short_perp_pnl = cash * (1 - exit_perp / entry_perp)
    return spot_pnl + short_perp_pnl


def _two_leg_cost(cash: float, bps: float) -> float:
    return cash * 2 * bps / 10_000


def _trade(
    position: OpenPosition,
    cash: float,
    exit_time: datetime,
    exit_spot: float,
    exit_perp: float,
    exit_rate: float,
    basis_pnl: float,
    exit_fee: float,
    exit_slippage: float,
    residual_basis: float,
    *,
    exit_reason: str = "funding_below_exit_threshold",
) -> dict[str, Any]:
    return {
        "entry_time": position.entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_spot": position.entry_spot,
        "entry_perp": position.entry_perp,
        "exit_spot": exit_spot,
        "exit_perp": exit_perp,
        "entry_signal_bps": position.entry_signal_bps,
        "exit_funding_bps": exit_rate * 10_000,
        "periods_held": position.periods_held,
        "funding_return_pct": position.funding_return * 100,
        "basis_return_pct": basis_pnl / cash * 100,
        "borrow_cost_pct": position.borrow_cost * 100,
        "exit_fee": exit_fee,
        "exit_slippage": exit_slippage,
        "basis_cost": residual_basis,
        "exit_reason": exit_reason,
        "factors": [
            "spot long plus perpetual short",
            "entry gated by previous real funding rate",
            "exit when real funding rate fell below threshold",
        ],
    }


def _exit_breakdown(trades: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "funding_below_exit_threshold": 0,
        "end_of_data": 0,
        "unknown": 0,
    }
    for trade in trades:
        reason = str(trade.get("exit_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _held_negative_share(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    held = 0
    negative_exits = 0
    for trade in trades:
        periods = int(trade.get("periods_held", 0))
        held += periods
        exit_bps = float(trade.get("exit_funding_bps", 0.0))
        if exit_bps < 0:
            negative_exits += 1
    if held == 0:
        return 0.0
    return negative_exits / held


def _sample_equity_curve(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= 250:
        return points
    step = max(len(points) // 250, 1)
    sampled = [point for index, point in enumerate(points) if index % step == 0]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _annualized_return_pct(net_return: float, events: list[dict[str, Any]]) -> float:
    if len(events) < 2:
        return net_return * 100
    start = cast(datetime, events[0]["time"])
    end = cast(datetime, events[-1]["time"])
    days = max((end - start).total_seconds() / 86_400, 1 / 24)
    if net_return <= -1:
        return -100.0
    return float(((1 + net_return) ** (365 / days) - 1) * 100)


def _sharpe(points: list[dict[str, Any]], settlement_hours: float) -> float:
    if len(points) < 3:
        return 0.0
    returns: list[float] = []
    previous = float(points[0]["equity"])
    for point in points[1:]:
        current = float(point["equity"])
        if previous > 0:
            returns.append(current / previous - 1)
        previous = current
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    periods_per_year = 365 * 24 / settlement_hours
    return statistics.fmean(returns) / std * math.sqrt(periods_per_year)


def _max_drawdown_pct(points: list[dict[str, Any]]) -> float:
    peak: float | None = None
    max_drawdown = 0.0
    for point in points:
        equity = float(point["equity"])
        peak = equity if peak is None else max(peak, equity)
        if peak <= 0:
            continue
        drawdown = equity / peak - 1
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown * 100


def _cost_model_dict(costs: FundingCostModel) -> dict[str, Any]:
    return {
        "taker_fee_bps": costs.taker_fee_bps,
        "maker_fee_bps": costs.maker_fee_bps,
        "active_fee_bps": costs.active_fee_bps,
        "slippage_bps": costs.slippage_bps,
        "basis_cost_bps": costs.basis_cost_bps,
        "borrow_cost_bps_annual": costs.borrow_cost_bps_annual,
        "settlement_hours": costs.settlement_hours,
        "min_funding_bps": costs.min_funding_bps,
        "exit_funding_bps": costs.exit_funding_bps,
        "use_maker_fees": costs.use_maker_fees,
        "basis_model": costs.basis_model,
        "round_trip_cost_pct": costs.round_trip_cost_pct,
    }
