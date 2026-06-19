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
FundingVerdict = Literal["ROBUST_CARRY", "NO_ROBUST_EDGE", "INSUFFICIENT"]

REQUEST_LIMIT = 1000
RETRY_COUNT = 2
DEFAULT_MAX_FUNDING_EVENTS = 1500


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
    cash_rate_annual: float | None = None
    max_holding_events: int | None = None
    use_maker_fees: bool = False
    max_funding_events: int | None = None


@dataclass(frozen=True)
class FundingResearchConfig:
    locked_oos_fraction: float = 0.30
    train_events: int = 120
    test_events: int = 40
    step_events: int = 40
    fdr_alpha: float = 0.10
    min_total_events: int = 220
    min_locked_oos_trades: int = 3
    min_walk_forward_folds: int = 2
    min_locked_oos_excess_return: float = 0.0
    min_locked_oos_sharpe: float = 0.0


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
    cash_rate_annual: float
    max_holding_events: int
    use_maker_fees: bool
    basis_model: str
    leverage_policy: str

    @property
    def round_trip_cost_pct(self) -> float:
        per_event_bps = 4 * (self.active_fee_bps + self.slippage_bps) + self.basis_cost_bps
        return per_event_bps / 100.0


@dataclass(frozen=True)
class FundingGridParams:
    min_funding_bps: float
    exit_funding_bps: float
    max_holding_events: int

    @property
    def key(self) -> str:
        return (
            f"min{self.min_funding_bps:g}_exit{self.exit_funding_bps:g}"
            f"_hold{self.max_holding_events}"
        )


@dataclass(frozen=True)
class Symbols:
    spot: str
    swap: str


@dataclass
class OpenPosition:
    signal_time: datetime
    entry_time: datetime
    entry_spot: float
    entry_perp: float
    entry_signal_bps: float
    periods_held: int = 0
    funding_return: float = 0.0
    borrow_cost: float = 0.0
    worst_basis_move_pct: float = 0.0


def run_funding_arb_backtest(config: FundingArbConfig) -> dict[str, Any]:
    start_dt = _parse_date(config.start)
    end_dt = _parse_date(config.end)
    if end_dt <= start_dt:
        raise DataLoadError("end must be after start")
    if config.cash <= 0:
        raise DataLoadError("cash must be positive")
    if config.timeframe not in TIMEFRAME_MS:
        raise DataLoadError(f"unsupported timeframe: {config.timeframe}")

    symbols = _symbols(config.symbol)
    aligned = load_aligned_funding_events(config, symbols)
    return _simulate(aligned, config, _cost_model(config), symbols)


def run_funding_arb_research(
    events_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    source: FundingSource,
    start: date | datetime | str,
    end: date | datetime | str,
    cash: float = 10_000.0,
    base_config: FundingArbConfig | None = None,
    research_config: FundingResearchConfig | None = None,
) -> dict[str, Any]:
    research_config = research_config or FundingResearchConfig()
    if not events_by_symbol:
        return _research_insufficient("no aligned funding events supplied", source, start, end)

    search_space = predeclared_funding_grid()
    candidates: list[dict[str, Any]] = []
    insufficient: list[str] = []
    p_values: list[float] = []
    tested_indices: list[int] = []

    for symbol, events in sorted(events_by_symbol.items()):
        if len(events) < research_config.min_total_events:
            insufficient.append(f"{symbol}: only {len(events)} events")
            continue
        locked_start = int(len(events) * (1.0 - research_config.locked_oos_fraction))
        if locked_start < research_config.train_events + research_config.test_events:
            insufficient.append(f"{symbol}: not enough in-sample events before locked OOS")
            continue
        symbols = _symbols(symbol)
        for params in search_space:
            config = _config_for_params(
                symbol,
                source,
                start,
                end,
                cash,
                params,
                base_config,
            )
            costs = _cost_model(config)
            folds = _walk_forward_fold_scores(
                events,
                config,
                costs,
                symbols,
                locked_start,
                research_config,
            )
            key = f"{symbol}:{params.key}"
            if len(folds) < research_config.min_walk_forward_folds:
                candidates.append({"key": key, "status": "INSUFFICIENT", "folds": len(folds)})
                insufficient.append(f"{key}: insufficient walk-forward folds")
                continue
            p_value = _sign_test_p_value([float(fold["excess_return_pct"]) for fold in folds])
            tested_indices.append(len(candidates))
            p_values.append(p_value)
            locked = _evaluate_slice(events, locked_start, len(events), config, costs, symbols)
            candidates.append(
                {
                    "key": key,
                    "symbol": symbol,
                    "params": params.__dict__,
                    "status": "TESTED",
                    "walk_forward": {
                        "folds": len(folds),
                        "positive_excess_share": _positive_share(
                            [float(fold["excess_return_pct"]) for fold in folds]
                        ),
                        "median_excess_return_pct": _median(
                            [float(fold["excess_return_pct"]) for fold in folds]
                        ),
                        "p_value": p_value,
                    },
                    "locked_oos": _compact_result(locked),
                }
            )

    if not tested_indices:
        return _research_insufficient(
            "; ".join(insufficient) or "no candidate had enough events",
            source,
            start,
            end,
            search_space_n=len(events_by_symbol) * len(search_space),
            failures=insufficient,
        )

    discoveries = _benjamini_hochberg(p_values, alpha=research_config.fdr_alpha)
    for passed, candidate_index in zip(discoveries, tested_indices, strict=True):
        candidates[candidate_index]["walk_forward"]["bh_fdr_discovery"] = passed
    robust = [
        candidates[candidate_index]
        for passed, candidate_index in zip(discoveries, tested_indices, strict=True)
        if passed
        and _passes_locked_oos_gate(
            cast(dict[str, Any], candidates[candidate_index]["locked_oos"]),
            research_config,
        )
    ]
    verdict: FundingVerdict = "ROBUST_CARRY" if robust else "NO_ROBUST_EDGE"
    reason = (
        "at least one predeclared carry grid member passed walk-forward BH-FDR and locked OOS"
        if robust
        else "no predeclared carry grid member passed both BH-FDR and locked OOS cash gates"
    )
    return {
        "strategy": "funding_arb_neutral_carry",
        "verdict": verdict,
        "reason": reason,
        "source": source,
        "data_window": {
            "start": _parse_date(start).isoformat(),
            "end": _parse_date(end).isoformat(),
        },
        "research_config": research_config.__dict__,
        "predeclared_grid": [params.__dict__ for params in search_space],
        "search_space_n": len(events_by_symbol) * len(search_space),
        "tested_candidates": len(tested_indices),
        "fdr": {
            "method": "Benjamini-Hochberg on in-sample walk-forward sign-test p-values",
            "alpha": research_config.fdr_alpha,
            "tests": len(p_values),
            "discoveries": sum(1 for value in discoveries if value),
            "min_p_value": min(p_values) if p_values else None,
        },
        "best_candidate": robust[0] if robust else _best_candidate(candidates),
        "candidates": candidates,
        "insufficient": insufficient,
        "benchmark": "risk-free cash, not buy-and-hold",
        "execution": (
            "signals observed at funding event t; entries/exits filled at t+1 aligned price"
        ),
        "safety": {
            "mode": "read-only research",
            "orders": "disabled",
            "wallet_or_account_access": "none",
            "leverage_policy": "spot long plus equal-notional perp short; no speculative leverage",
        },
    }


def predeclared_funding_grid() -> tuple[FundingGridParams, ...]:
    return (
        FundingGridParams(1.0, 0.0, 6),
        FundingGridParams(1.0, -1.0, 12),
        FundingGridParams(3.0, 0.0, 6),
        FundingGridParams(3.0, 1.0, 12),
        FundingGridParams(5.0, 0.0, 6),
        FundingGridParams(5.0, 1.0, 12),
        FundingGridParams(8.0, 0.0, 6),
        FundingGridParams(8.0, 2.0, 12),
    )


def load_aligned_funding_events(config: FundingArbConfig, symbols: Symbols) -> list[dict[str, Any]]:
    start_dt = _parse_date(config.start)
    end_dt = _parse_date(config.end)
    funding_events = _load_funding_history(
        config.source,
        symbols.swap,
        start_dt,
        end_dt,
        config.max_funding_events or _env_int("FUNDING_ARB_MAX_EVENTS", DEFAULT_MAX_FUNDING_EVENTS),
    )
    spot_frame = _load_ohlcv(
        config.source, symbols.spot, "spot", config.timeframe, start_dt, end_dt
    )
    perp_frame = _load_ohlcv(
        config.source, symbols.swap, "swap", config.timeframe, start_dt, end_dt
    )
    aligned = _align_events(funding_events, spot_frame, perp_frame)
    if not aligned:
        raise DataLoadError("funding history and price bars could not be aligned")
    return aligned


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
    settlement_hours = _value_or_env(config.settlement_hours, "FUNDING_ARB_SETTLEMENT_HOURS", 8.0)
    min_funding_bps = _value_or_env(config.min_funding_bps, "FUNDING_ARB_MIN_FUNDING_BPS", 3.0)
    exit_funding_bps = _value_or_env(config.exit_funding_bps, "FUNDING_ARB_EXIT_FUNDING_BPS", 0.0)
    cash_rate_annual = _value_or_env(config.cash_rate_annual, "FUNDING_ARB_CASH_RATE_ANNUAL", 0.04)
    max_holding_events = int(
        _value_or_env(config.max_holding_events, "FUNDING_ARB_MAX_HOLDING_EVENTS", 12.0)
    )
    active_fee_bps = maker_fee_bps if config.use_maker_fees else taker_fee_bps
    if settlement_hours <= 0:
        raise DataLoadError("settlement_hours must be positive")
    if max_holding_events <= 0:
        raise DataLoadError("max_holding_events must be positive")
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
        cash_rate_annual=max(cash_rate_annual, 0.0),
        max_holding_events=max_holding_events,
        use_maker_fees=config.use_maker_fees,
        basis_model="actual aligned spot/perp close; residual basis_cost_bps charged on exit",
        leverage_policy="spot long plus equal-notional perpetual short; no speculative leverage",
    )


def _value_or_env(value: float | int | None, env_name: str, default: float) -> float:
    if value is not None:
        return float(value)
    raw = os.getenv(env_name)
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
    return _evaluate_slice(events, 0, len(events), config, costs, symbols)


def _evaluate_slice(
    events: list[dict[str, Any]],
    start: int,
    end: int,
    config: FundingArbConfig,
    costs: FundingCostModel,
    symbols: Symbols,
) -> dict[str, Any]:
    sliced = events[start:end]
    if len(sliced) < 3:
        raise DataLoadError("at least three aligned funding events are required")
    cash = config.cash
    realized = 0.0
    gross_funding = 0.0
    realized_basis = 0.0
    fee_cost = 0.0
    slippage_cost = 0.0
    basis_cost = 0.0
    borrow_cost = 0.0
    position: OpenPosition | None = None
    pending_entry_signal: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    period_returns: list[float] = []
    negative_periods = 0
    held_periods = 0
    cash_returns: list[float] = []

    previous_equity = cash
    for index, event in enumerate(sliced):
        timestamp = cast(datetime, event["time"])
        spot = cast(float, event["spot_close"])
        perp = cast(float, event["perp_close"])
        rate = cast(float, event["funding_rate"])
        if rate < 0:
            negative_periods += 1

        if pending_entry_signal is not None and position is None:
            entry_fee = _two_leg_cost(cash, costs.active_fee_bps)
            entry_slippage = _two_leg_cost(cash, costs.slippage_bps)
            realized -= entry_fee + entry_slippage
            fee_cost += entry_fee
            slippage_cost += entry_slippage
            position = OpenPosition(
                signal_time=cast(datetime, pending_entry_signal["time"]),
                entry_time=timestamp,
                entry_spot=spot,
                entry_perp=perp,
                entry_signal_bps=float(pending_entry_signal["funding_rate"]) * 10_000,
            )
            pending_entry_signal = None

        if position is not None:
            period_funding = cash * rate
            period_borrow = cash * (costs.borrow_cost_bps_annual / 10_000.0) * (
                costs.settlement_hours / (365 * 24)
            )
            realized += period_funding - period_borrow
            gross_funding += period_funding
            borrow_cost += period_borrow
            position.funding_return += period_funding / cash
            position.borrow_cost += period_borrow / cash
            position.periods_held += 1
            held_periods += 1
            basis_move = (
                _basis_pnl(cash, position.entry_spot, position.entry_perp, spot, perp) / cash
            )
            position.worst_basis_move_pct = min(position.worst_basis_move_pct, basis_move * 100)

        should_exit = (
            position is not None
            and (rate * 10_000 < costs.exit_funding_bps
                 or position.periods_held >= costs.max_holding_events)
        )
        if should_exit and position is not None:
            exit_reason = (
                "max_holding_events"
                if position.periods_held >= costs.max_holding_events
                else "funding_below_exit_threshold"
            )
            trade_costs = _close_position(
                trades,
                position,
                cash,
                timestamp,
                spot,
                perp,
                rate,
                costs,
                exit_reason=exit_reason,
            )
            realized += trade_costs["basis_pnl"] - trade_costs["exit_fee"]
            realized -= trade_costs["exit_slippage"] + trade_costs["residual_basis"]
            realized_basis += trade_costs["basis_pnl"]
            fee_cost += trade_costs["exit_fee"]
            slippage_cost += trade_costs["exit_slippage"]
            basis_cost += trade_costs["residual_basis"]
            position = None

        if (
            position is None
            and pending_entry_signal is None
            and index + 1 < len(sliced)
            and rate * 10_000 >= costs.min_funding_bps
        ):
            pending_entry_signal = event

        unrealized_basis = (
            _basis_pnl(cash, position.entry_spot, position.entry_perp, spot, perp)
            if position is not None
            else 0.0
        )
        equity = cash + realized + unrealized_basis
        period_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        if equity_curve:
            period_returns.append(period_return)
            cash_returns.append(_cash_period_return(costs))
        previous_equity = equity
        equity_curve.append(
            {
                "date": timestamp.isoformat(),
                "equity": equity,
                "funding_rate": rate,
                "funding_bps": rate * 10_000,
                "position_open": position is not None,
                "spot_close": spot,
                "perp_close": perp,
            }
        )

    if position is not None:
        last = sliced[-1]
        timestamp = cast(datetime, last["time"])
        trade_costs = _close_position(
            trades,
            position,
            cash,
            timestamp,
            cast(float, last["spot_close"]),
            cast(float, last["perp_close"]),
            cast(float, last["funding_rate"]),
            costs,
            exit_reason="end_of_data",
        )
        realized += trade_costs["basis_pnl"] - trade_costs["exit_fee"]
        realized -= trade_costs["exit_slippage"] + trade_costs["residual_basis"]
        realized_basis += trade_costs["basis_pnl"]
        fee_cost += trade_costs["exit_fee"]
        slippage_cost += trade_costs["exit_slippage"]
        basis_cost += trade_costs["residual_basis"]
        equity_curve[-1]["equity"] = cash + realized
        equity_curve[-1]["position_open"] = False

    net_return = realized / cash
    gross_return = (gross_funding + realized_basis) / cash
    cash_return = _cash_total_return(costs, sliced)
    excess_return = net_return - cash_return
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
            "annualized_return_pct": _annualized_return_pct(net_return, sliced),
            "sharpe": _sharpe_from_returns(period_returns, costs.settlement_hours),
            "sortino": _sortino_from_returns(period_returns, costs.settlement_hours),
            "calmar": _calmar(net_return, sliced, _max_drawdown_pct(equity_curve)),
            "max_drawdown_pct": _max_drawdown_pct(equity_curve),
            "num_trades": len(trades),
            "funding_events": len(sliced),
            "held_funding_events": held_periods,
            "exposure_pct": held_periods / len(sliced) * 100 if sliced else 0.0,
            "negative_funding_period_share": negative_periods / len(sliced) if sliced else 0.0,
            "held_negative_funding_period_share": _held_negative_share(trades),
            "cash_benchmark_return_pct": cash_return * 100,
            "benchmark_return_pct": cash_return * 100,
            "excess_cash_return_pct": excess_return * 100,
            "beat_cash_benchmark": excess_return > 0,
            "beat_benchmark": excess_return > 0,
            "net_cost_positive": net_return > 0,
            "exit_breakdown": _exit_breakdown(trades),
            "worst_trade_basis_move_pct": min(
                (float(trade["worst_basis_move_pct"]) for trade in trades),
                default=0.0,
            ),
        },
        "cost_model": _cost_model_dict(costs),
        "equity_curve": _sample_equity_curve(equity_curve),
        "period_returns": period_returns,
        "cash_period_returns": cash_returns,
        "trades": trades,
        "data": {
            "funding_source": f"ccxt.{config.source}.fetch_funding_rate_history",
            "price_source": f"ccxt.{config.source}.fetch_ohlcv spot + swap",
            "start": sliced[0]["time"].isoformat(),
            "end": sliced[-1]["time"].isoformat(),
        },
        "disclaimer": (
            "read-only backtest; no order path, wallet access, or live trading integration"
        ),
    }


def _close_position(
    trades: list[dict[str, Any]],
    position: OpenPosition,
    cash: float,
    exit_time: datetime,
    exit_spot: float,
    exit_perp: float,
    exit_rate: float,
    costs: FundingCostModel,
    *,
    exit_reason: str,
) -> dict[str, float]:
    basis_pnl = _basis_pnl(cash, position.entry_spot, position.entry_perp, exit_spot, exit_perp)
    exit_fee = _two_leg_cost(cash, costs.active_fee_bps)
    exit_slippage = _two_leg_cost(cash, costs.slippage_bps)
    residual_basis = cash * costs.basis_cost_bps / 10_000.0
    trades.append(
        _trade(
            position,
            cash,
            exit_time,
            exit_spot,
            exit_perp,
            exit_rate,
            basis_pnl,
            exit_fee,
            exit_slippage,
            residual_basis,
            exit_reason=exit_reason,
        )
    )
    return {
        "basis_pnl": basis_pnl,
        "exit_fee": exit_fee,
        "exit_slippage": exit_slippage,
        "residual_basis": residual_basis,
    }


def _basis_pnl(
    cash: float,
    entry_spot: float,
    entry_perp: float,
    exit_spot: float,
    exit_perp: float,
) -> float:
    spot_pnl = cash * (exit_spot / entry_spot - 1.0)
    short_perp_pnl = cash * (1.0 - exit_perp / entry_perp)
    return spot_pnl + short_perp_pnl


def _two_leg_cost(cash: float, bps: float) -> float:
    return cash * 2.0 * bps / 10_000.0


def _cash_period_return(costs: FundingCostModel) -> float:
    return costs.cash_rate_annual * costs.settlement_hours / (365.0 * 24.0)


def _cash_total_return(costs: FundingCostModel, events: list[dict[str, Any]]) -> float:
    if len(events) < 2:
        return 0.0
    start = cast(datetime, events[0]["time"])
    end = cast(datetime, events[-1]["time"])
    years = max((end - start).total_seconds() / (365.0 * 24.0 * 3600.0), 0.0)
    return float((1.0 + costs.cash_rate_annual) ** years - 1.0)


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
    exit_reason: str,
) -> dict[str, Any]:
    return {
        "signal_time": position.signal_time.isoformat(),
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
        "worst_basis_move_pct": position.worst_basis_move_pct,
        "exit_fee": exit_fee,
        "exit_slippage": exit_slippage,
        "basis_cost": residual_basis,
        "exit_reason": exit_reason,
        "factors": [
            "spot long plus perpetual short",
            "entry filled one event after observed funding signal",
            "funding settled on real held settlement events by short-perp direction",
        ],
    }


def _exit_breakdown(trades: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "funding_below_exit_threshold": 0,
        "max_holding_events": 0,
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
        return net_return * 100.0
    start = cast(datetime, events[0]["time"])
    end = cast(datetime, events[-1]["time"])
    days = max((end - start).total_seconds() / 86_400.0, 1 / 24)
    if net_return <= -1:
        return -100.0
    return float(((1.0 + net_return) ** (365.0 / days) - 1.0) * 100.0)


def _sharpe_from_returns(returns: list[float], settlement_hours: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    periods_per_year = 365.0 * 24.0 / settlement_hours
    return statistics.fmean(returns) / std * math.sqrt(periods_per_year)


def _sortino_from_returns(returns: list[float], settlement_hours: float) -> float:
    downside = [value for value in returns if value < 0]
    if len(returns) < 2 or len(downside) < 2:
        return 0.0
    downside_std = statistics.stdev(downside)
    if downside_std == 0:
        return 0.0
    periods_per_year = 365.0 * 24.0 / settlement_hours
    return statistics.fmean(returns) / downside_std * math.sqrt(periods_per_year)


def _calmar(net_return: float, events: list[dict[str, Any]], max_drawdown_pct: float) -> float:
    drawdown = abs(max_drawdown_pct) / 100.0
    if drawdown == 0:
        return 0.0
    return (_annualized_return_pct(net_return, events) / 100.0) / drawdown


def _max_drawdown_pct(points: list[dict[str, Any]]) -> float:
    peak: float | None = None
    max_drawdown = 0.0
    for point in points:
        equity = float(point["equity"])
        peak = equity if peak is None else max(peak, equity)
        if peak <= 0:
            continue
        drawdown = equity / peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown * 100.0


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
        "cash_rate_annual": costs.cash_rate_annual,
        "max_holding_events": costs.max_holding_events,
        "use_maker_fees": costs.use_maker_fees,
        "basis_model": costs.basis_model,
        "leverage_policy": costs.leverage_policy,
        "round_trip_cost_pct": costs.round_trip_cost_pct,
    }


def _config_for_params(
    symbol: str,
    source: FundingSource,
    start: date | datetime | str,
    end: date | datetime | str,
    cash: float,
    params: FundingGridParams,
    base_config: FundingArbConfig | None,
) -> FundingArbConfig:
    template = base_config or FundingArbConfig(symbol=symbol, source=source, start=start, end=end)
    return FundingArbConfig(
        symbol=symbol,
        source=source,
        start=start,
        end=end,
        timeframe=template.timeframe,
        cash=cash,
        min_funding_bps=params.min_funding_bps,
        exit_funding_bps=params.exit_funding_bps,
        taker_fee_bps=template.taker_fee_bps,
        maker_fee_bps=template.maker_fee_bps,
        slippage_bps=template.slippage_bps,
        basis_cost_bps=template.basis_cost_bps,
        borrow_cost_bps_annual=template.borrow_cost_bps_annual,
        settlement_hours=template.settlement_hours,
        cash_rate_annual=template.cash_rate_annual,
        max_holding_events=params.max_holding_events,
        use_maker_fees=template.use_maker_fees,
        max_funding_events=template.max_funding_events,
    )


def _walk_forward_fold_scores(
    events: list[dict[str, Any]],
    config: FundingArbConfig,
    costs: FundingCostModel,
    symbols: Symbols,
    locked_start: int,
    research_config: FundingResearchConfig,
) -> list[dict[str, float | int]]:
    folds: list[dict[str, float | int]] = []
    start = 0
    while start + research_config.train_events + research_config.test_events <= locked_start:
        test_start = start + research_config.train_events
        test_end = test_start + research_config.test_events
        result = _evaluate_slice(events, test_start, test_end, config, costs, symbols)
        stats = cast(dict[str, Any], result["stats"])
        folds.append(
            {
                "test_start": test_start,
                "test_end": test_end,
                "net_return_pct": float(stats["net_return_pct"]),
                "cash_return_pct": float(stats["cash_benchmark_return_pct"]),
                "excess_return_pct": float(stats["excess_cash_return_pct"]),
                "trades": int(stats["num_trades"]),
            }
        )
        start += research_config.step_events
    return folds


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    stats = cast(dict[str, Any], result["stats"])
    return {
        "net_return_pct": stats["net_return_pct"],
        "cash_benchmark_return_pct": stats["cash_benchmark_return_pct"],
        "excess_cash_return_pct": stats["excess_cash_return_pct"],
        "annualized_return_pct": stats["annualized_return_pct"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "calmar": stats["calmar"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
        "num_trades": stats["num_trades"],
        "funding_events": stats["funding_events"],
        "held_funding_events": stats["held_funding_events"],
        "exposure_pct": stats["exposure_pct"],
        "worst_trade_basis_move_pct": stats["worst_trade_basis_move_pct"],
        "exit_breakdown": stats["exit_breakdown"],
    }


def _passes_locked_oos_gate(
    locked_oos: dict[str, Any],
    research_config: FundingResearchConfig,
) -> bool:
    return (
        float(locked_oos["excess_cash_return_pct"])
        > research_config.min_locked_oos_excess_return * 100.0
        and float(locked_oos["sharpe"]) > research_config.min_locked_oos_sharpe
        and int(locked_oos["num_trades"]) >= research_config.min_locked_oos_trades
    )


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    tested = [candidate for candidate in candidates if candidate.get("status") == "TESTED"]
    if not tested:
        return None
    return max(
        tested,
        key=lambda candidate: float(
            cast(dict[str, Any], candidate["locked_oos"])["excess_cash_return_pct"]
        ),
    )


def _research_insufficient(
    reason: str,
    source: FundingSource,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    search_space_n: int = 0,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "strategy": "funding_arb_neutral_carry",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "source": source,
        "data_window": {
            "start": _parse_date(start).isoformat(),
            "end": _parse_date(end).isoformat(),
        },
        "search_space_n": search_space_n,
        "tested_candidates": 0,
        "predeclared_grid": [params.__dict__ for params in predeclared_funding_grid()],
        "insufficient": failures or [reason],
        "benchmark": "risk-free cash, not buy-and-hold",
        "safety": {
            "mode": "read-only research",
            "orders": "disabled",
            "wallet_or_account_access": "none",
        },
    }


def _sign_test_p_value(excess_returns: list[float]) -> float:
    non_zero = [value for value in excess_returns if value != 0.0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0.0)
    return min(1.0, float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)))


def _benjamini_hochberg(p_values: list[float], *, alpha: float) -> list[bool]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    passed = [False for _ in p_values]
    max_rank = -1
    tests = len(indexed)
    for rank, (_index, p_value) in enumerate(indexed, start=1):
        if p_value <= alpha * rank / tests:
            max_rank = rank
    if max_rank >= 1:
        for rank, (index, _p_value) in enumerate(indexed, start=1):
            if rank <= max_rank:
                passed[index] = True
    return passed


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 6) if values else 0.0


def _positive_share(values: list[float]) -> float:
    return round(sum(1 for value in values if value > 0.0) / len(values), 6) if values else 0.0
