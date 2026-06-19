from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from backtesting import Backtest, Strategy  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import data as data_module
from competition import rank_strategies
from data import MAX_BARS, TIMEFRAME_MS, DataLoadError, Source
from factor_ic import FactorKind, FactorMode, evaluate_ohlcv_factor_ic
from funding_arb import FundingArbConfig, FundingSource, run_funding_arb_backtest
from healthcheck import evaluate_strategy_health
from strategies import DEFAULT_PARAMS, STRATEGIES, StrategyParams, load_plugins
from walk_forward import Objective, run_walk_forward

load_plugins()

app = FastAPI(title="backtest-service", version="0.1.0")
DEFAULT_COMPETITION_LATEST_PATH = "/plugins/competition_latest.json"
DEFAULT_COMPETITION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


class BacktestRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=40)
    source: Source
    timeframe: str = "1d"
    start: date
    end: date
    strategy: str = "ma_cross"
    params: StrategyParams = Field(default_factory=dict)
    cash: float = Field(default=10_000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.2)
    regime_symbol: str | None = Field(default=None, min_length=1, max_length=40)
    regime_ma: int = Field(default=200, gt=0)

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: StrategyParams) -> StrategyParams:
        for key, raw in value.items():
            if not key.strip():
                raise ValueError("param names must be non-empty")
            if isinstance(raw, bool | str):
                continue
            if raw <= 0:
                raise ValueError(f"{key} must be positive")
        return value


class BacktestStats(BaseModel):
    return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    sharpe: float
    num_trades: int
    exposure_pct: float
    exit_breakdown: dict[str, int]


class OpenPositionInfo(BaseModel):
    detected: bool
    mark_to_market_return_pct: float
    first_equity: float
    last_equity: float
    note: str


class EquityPoint(BaseModel):
    date: str
    equity: float


class TradeItem(BaseModel):
    entry_time: str
    exit_time: str
    entry_price: float | None
    exit_price: float | None
    sl: float | None
    tp: float | None
    exit_reason: str
    entry_regime_up: bool | None
    pnl_pct: float
    size: float


class BacktestResponse(BaseModel):
    stats: BacktestStats
    equity_curve: list[EquityPoint]
    trades: list[TradeItem]
    open_position: OpenPositionInfo | None = None


class StrategyInfo(BaseModel):
    name: str
    default_params: StrategyParams


class StrategyHealthcheckRequest(BaseModel):
    runs: list[dict[str, Any]]
    edge_thesis: str = ""
    thresholds: dict[str, float] | None = None
    cost_bps: float | None = None
    factor_report: dict[str, Any] | None = None
    walk_forward_report: dict[str, Any] | None = None


class StrategyCompetitionRequest(BaseModel):
    entries: list[dict[str, Any]]
    weights: dict[str, float] | None = None
    promote_top_n: int = Field(default=3, ge=0)
    retire_bottom_streak: int = Field(default=3, ge=0)


class FundingArbRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=40)
    source: FundingSource = "binance"
    timeframe: str = "1h"
    start: date
    end: date
    cash: float = Field(default=10_000, gt=0)
    min_funding_bps: float | None = None
    exit_funding_bps: float | None = None
    taker_fee_bps: float | None = Field(default=None, ge=0)
    maker_fee_bps: float | None = Field(default=None, ge=0)
    slippage_bps: float | None = Field(default=None, ge=0)
    basis_cost_bps: float | None = Field(default=None, ge=0)
    borrow_cost_bps_annual: float | None = Field(default=None, ge=0)
    settlement_hours: float | None = Field(default=None, gt=0)
    cash_rate_annual: float | None = Field(default=None, ge=0)
    max_holding_events: int | None = Field(default=None, gt=0)
    use_maker_fees: bool = False
    max_funding_events: int | None = Field(default=None, gt=0)


class FactorSpec(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    kind: FactorKind
    window: int = Field(default=1, gt=0, le=1000)


class FactorICRequest(BaseModel):
    symbol: str | None = Field(default=None, min_length=1, max_length=40)
    symbols: list[str] | None = None
    source: Source
    timeframe: str = "1d"
    start: date
    end: date
    factors: list[FactorSpec] = Field(min_length=1, max_length=20)
    label_periods: int = Field(default=1, gt=0, le=250)
    groups: int = Field(default=5, ge=2, le=20)
    mode: FactorMode = "time_series"
    ic_window: int = Field(default=60, ge=3, le=1000)
    redundancy_threshold: float = Field(default=0.8, gt=0, le=1)
    icir_threshold: float = Field(default=0.3, ge=0)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("symbols must not be empty")
        if len(cleaned) > 50:
            raise ValueError("symbols supports at most 50 entries")
        return cleaned


class WalkForwardRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=40)
    source: Source
    timeframe: str = "1d"
    start: date
    end: date
    strategy: str = "ma_cross"
    param_grid: list[StrategyParams] = Field(min_length=1, max_length=100)
    train_bars: int = Field(gt=0, le=5000)
    test_bars: int = Field(gt=0, le=5000)
    step_bars: int | None = Field(default=None, gt=0, le=5000)
    cash: float = Field(default=10_000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.2)
    objective: Objective = "return_pct"
    min_oos_return_pct: float = 0.0
    min_oos_is_return_ratio: float = Field(default=0.5, ge=0)
    min_oos_is_sharpe_ratio: float = Field(default=0.5, ge=0)
    max_parameter_trials: int = Field(default=20, ge=1)

    @field_validator("param_grid")
    @classmethod
    def validate_param_grid(cls, value: list[StrategyParams]) -> list[StrategyParams]:
        for params in value:
            for key, raw in params.items():
                if not key.strip():
                    raise ValueError("param names must be non-empty")
                if isinstance(raw, bool | str):
                    continue
                if raw <= 0:
                    raise ValueError(f"{key} must be positive")
        return value


def _competition_latest_path() -> Path:
    return Path(os.environ.get("COMPETITION_LATEST_PATH", DEFAULT_COMPETITION_LATEST_PATH))


def _competition_max_age_seconds() -> int:
    raw = os.environ.get("COMPETITION_LATEST_MAX_AGE_SECONDS", "")
    if not raw:
        return DEFAULT_COMPETITION_MAX_AGE_SECONDS
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_COMPETITION_MAX_AGE_SECONDS


def _parse_generated_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def _competition_latest_unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "entries": []}


def _competition_latest_entry(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "strategy": "",
            "params": {},
            "rank": 0,
            "score": 0.0,
            "healthcheck_verdict": "PASS_WITH_WARN",
            "status": "hold",
            "key_metrics": {},
        }
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    metrics = row.get("key_metrics") if isinstance(row.get("key_metrics"), dict) else {}
    return {
        "strategy": str(row.get("strategy", "")),
        "params": params,
        "rank": int(row.get("rank", 0)) if isinstance(row.get("rank"), int | float | str) else 0,
        "score": float(row.get("score", 0.0))
        if isinstance(row.get("score"), int | float | str)
        else 0.0,
        "healthcheck_verdict": str(row.get("healthcheck_verdict", "")),
        "status": str(row.get("status", "")),
        "key_metrics": metrics,
    }


def _competition_latest_payload(data: object) -> dict[str, Any]:
    if not isinstance(data, dict):
        return _competition_latest_unavailable("invalid competition summary")
    rows = data.get("entries")
    if not isinstance(rows, list):
        rows = data.get("leaderboard")
    if not isinstance(rows, list):
        rows = []
    generated_at = data.get("generated_at")
    return {
        "available": True,
        "generated_at": generated_at,
        "universe": data.get("universe", {}),
        "entries": [_competition_latest_entry(row) for row in rows],
        "disclaimer": data.get(
            "disclaimer",
            "competition candidates only; no auto graduation, no trading",
        ),
    }


def _estimated_bars(request: BacktestRequest) -> int | None:
    timeframe_ms = TIMEFRAME_MS.get(request.timeframe)
    if timeframe_ms is None:
        return None
    start_dt = datetime.combine(request.start, datetime.min.time())
    end_dt = datetime.combine(request.end, datetime.min.time())
    if end_dt <= start_dt:
        return 0
    delta_ms = int((end_dt - start_dt).total_seconds() * 1000)
    return max((delta_ms + timeframe_ms - 1) // timeframe_ms, 0)


def _estimated_bars_for_range(start: date, end: date, timeframe: str) -> int | None:
    timeframe_ms = TIMEFRAME_MS.get(timeframe)
    if timeframe_ms is None:
        return None
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.min.time())
    if end_dt <= start_dt:
        return 0
    delta_ms = int((end_dt - start_dt).total_seconds() * 1000)
    return max((delta_ms + timeframe_ms - 1) // timeframe_ms, 0)


def _float_stat(stats: Any, key: str) -> float:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_stat(stats: Any, key: str) -> int:
    value = stats.get(key, 0) if hasattr(stats, "get") else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _configured_strategy(base_cls: type[Strategy], params: StrategyParams) -> type[Strategy]:
    attrs: dict[str, Any] = dict(params)
    return cast(type[Strategy], type("ConfiguredStrategy", (base_cls,), attrs))


def _merged_params(strategy: str, params: StrategyParams) -> StrategyParams:
    return {**DEFAULT_PARAMS.get(strategy, {}), **params}


def _validate_ordered_params(params: StrategyParams) -> None:
    fast = _positive_number(params, "fast")
    slow = _positive_number(params, "slow")
    trend = _positive_number(params, "trend")
    if fast is not None and slow is not None and fast >= slow:
        raise ValueError("FAST_MUST_BE_LT_SLOW")
    if slow is not None and trend is not None and slow >= trend:
        raise ValueError("SLOW_MUST_BE_LT_TREND")


def _positive_number(params: StrategyParams, key: str) -> int | float | None:
    value = params.get(key)
    if isinstance(value, bool | str) or value is None:
        return None
    return value


def _date_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _safe_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _truthy(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)


def _same_timestamp(left: object, right: object) -> bool:
    try:
        return bool(pd.Timestamp(left) == pd.Timestamp(right))
    except Exception:  # noqa: BLE001
        return False


def _near_price(price: float | None, target: float | None) -> bool:
    if price is None or target is None:
        return False
    tolerance = max(abs(target) * 0.001, 1e-9)
    return abs(price - target) <= tolerance


def _factor_symbols(request: FactorICRequest) -> list[str]:
    raw_symbols = request.symbols if request.symbols is not None else []
    if request.symbol is not None:
        raw_symbols = [request.symbol, *raw_symbols]
    symbols = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        symbol = raw_symbol.strip()
        normalized = symbol.upper()
        if not symbol or normalized in seen:
            continue
        seen.add(normalized)
        symbols.append(symbol)
    return symbols


def _exit_reason(row: Any, last_bar_time: object | None) -> str:
    exit_price = _optional_float(_trade_value(row, "ExitPrice"))
    sl = _optional_float(_trade_value(row, "SL"))
    tp = _optional_float(_trade_value(row, "TP"))
    exit_time = _trade_value(row, "ExitTime")
    if _near_price(exit_price, tp):
        return "take_profit"
    if _near_price(exit_price, sl):
        return "stop_loss"
    if last_bar_time is not None and _same_timestamp(exit_time, last_bar_time):
        return "end_of_data"
    if exit_price is not None:
        return "signal"
    return "unknown"


def _empty_exit_breakdown() -> dict[str, int]:
    return {
        "take_profit": 0,
        "stop_loss": 0,
        "signal": 0,
        "end_of_data": 0,
        "unknown": 0,
    }


def _add_regime_column(
    frame: pd.DataFrame, regime_frame: pd.DataFrame, regime_ma: int
) -> pd.DataFrame:
    regime = regime_frame[["Close"]].copy()
    regime["RegimeSma"] = regime["Close"].rolling(regime_ma).mean()
    regime["RegimeUp"] = (regime["Close"] > regime["RegimeSma"]).fillna(False)
    left = frame.sort_index().reset_index(names="Date")
    right = regime[["RegimeUp"]].sort_index().reset_index(names="Date")
    aligned = pd.merge_asof(left, right, on="Date", direction="backward")
    aligned["RegimeUp"] = aligned["RegimeUp"].fillna(False).astype(bool)
    aligned = aligned.set_index("Date")
    aligned.index.name = frame.index.name
    return cast(pd.DataFrame, aligned.reindex(frame.index))


def _load_regime_frame(request: BacktestRequest) -> pd.DataFrame | None:
    if request.regime_symbol is None:
        return None
    # Calendar-day prewarm intentionally looks backward only; merge_asof below is backward-only.
    prewarm_start = request.start - timedelta(days=request.regime_ma * 3)
    return data_module.load_ohlcv(
        request.regime_symbol,
        request.source,
        request.timeframe,
        prewarm_start,
        request.end,
    )


def _sample_equity_curve(stats: Any) -> list[EquityPoint]:
    curve = stats.get("_equity_curve") if hasattr(stats, "get") else None
    if curve is None or not hasattr(curve, "iterrows"):
        return []
    length = len(curve)
    if length <= 250:
        step = 1
    else:
        step = max(length // 250, 1)
    points: list[EquityPoint] = []
    for index, (row_index, row) in enumerate(curve.iterrows()):
        if index % step != 0 and index != length - 1:
            continue
        equity = getattr(row, "Equity", None)
        if equity is None and hasattr(row, "get"):
            equity = row.get("Equity")
        points.append(EquityPoint(date=_date_str(row_index), equity=_safe_float(equity)))
    return points


def _trade_value(row: Any, *names: str) -> object:
    for name in names:
        if hasattr(row, "get"):
            value = row.get(name)
            if value is not None:
                return value
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _entry_regime_up(frame: pd.DataFrame | None, entry_time: object) -> bool | None:
    if frame is None or "RegimeUp" not in frame.columns:
        return None
    try:
        timestamp = pd.Timestamp(entry_time)
    except Exception:  # noqa: BLE001
        return None
    if timestamp not in frame.index:
        return None
    return bool(frame.loc[timestamp, "RegimeUp"])


def _trades(stats: Any, frame_data: pd.DataFrame | None) -> list[TradeItem]:
    frame = stats.get("_trades") if hasattr(stats, "get") else None
    if frame is None or not hasattr(frame, "iterrows"):
        return []
    last_bar_time = frame_data.index[-1] if frame_data is not None and len(frame_data) else None
    trades: list[TradeItem] = []
    for _, row in frame.iterrows():
        entry = _trade_value(row, "EntryTime")
        exit_time = _trade_value(row, "ExitTime")
        entry_price = _optional_float(_trade_value(row, "EntryPrice"))
        exit_price = _optional_float(_trade_value(row, "ExitPrice"))
        sl = _optional_float(_trade_value(row, "SL"))
        tp = _optional_float(_trade_value(row, "TP"))
        reason = _exit_reason(row, last_bar_time)
        entry_regime_up = _truthy(
            _trade_value(row, "Entry_RegimeUp", "EntryRegimeUp")
        )
        if entry_regime_up is None:
            entry_regime_up = _entry_regime_up(frame_data, entry)
        pnl_pct_raw = _trade_value(row, "ReturnPct")
        size_raw = _trade_value(row, "Size")
        pnl_pct = _safe_float(pnl_pct_raw) * 100
        size = _safe_float(size_raw)
        trades.append(
            TradeItem(
                entry_time=_date_str(entry),
                exit_time=_date_str(exit_time),
                entry_price=entry_price,
                exit_price=exit_price,
                sl=sl,
                tp=tp,
                exit_reason=reason,
                entry_regime_up=entry_regime_up,
                pnl_pct=pnl_pct,
                size=size,
            )
        )
    return trades


def _exit_breakdown(trades: list[TradeItem]) -> dict[str, int]:
    breakdown = _empty_exit_breakdown()
    for trade in trades:
        breakdown[trade.exit_reason] = breakdown.get(trade.exit_reason, 0) + 1
    return breakdown


def _open_position_info(stats: Any, trades: list[TradeItem]) -> OpenPositionInfo | None:
    if trades:
        return None
    curve = stats.get("_equity_curve") if hasattr(stats, "get") else None
    if curve is None or not hasattr(curve, "empty") or curve.empty:
        return None
    first_equity = _optional_float(curve.iloc[0].get("Equity"))
    last_equity = _optional_float(curve.iloc[-1].get("Equity"))
    if first_equity is None or last_equity is None or first_equity == 0:
        return None
    if abs(last_equity - first_equity) <= max(abs(first_equity) * 1e-9, 1e-9):
        return None
    return OpenPositionInfo(
        detected=True,
        mark_to_market_return_pct=(last_equity / first_equity - 1) * 100,
        first_equity=first_equity,
        last_equity=last_equity,
        note=(
            "No closed trades were reported by the backtest engine, but the equity curve "
            "changed. Closed-trade stats are zeroed to avoid reporting unrealized PnL as "
            "realized strategy performance."
        ),
    )


def _response_stats(
    stats: Any, trades: list[TradeItem], open_position: OpenPositionInfo | None
) -> BacktestStats:
    if open_position is not None:
        return BacktestStats(
            return_pct=0.0,
            buy_hold_return_pct=_float_stat(stats, "Buy & Hold Return [%]"),
            max_drawdown_pct=0.0,
            win_rate=0.0,
            sharpe=0.0,
            num_trades=0,
            exposure_pct=0.0,
            exit_breakdown=_empty_exit_breakdown(),
        )
    return BacktestStats(
        return_pct=_float_stat(stats, "Return [%]"),
        buy_hold_return_pct=_float_stat(stats, "Buy & Hold Return [%]"),
        max_drawdown_pct=_float_stat(stats, "Max. Drawdown [%]"),
        win_rate=_float_stat(stats, "Win Rate [%]"),
        sharpe=_float_stat(stats, "Sharpe Ratio"),
        num_trades=len(trades),
        exposure_pct=_float_stat(stats, "Exposure Time [%]"),
        exit_breakdown=_exit_breakdown(trades),
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies", response_model=list[StrategyInfo])
def strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(name=name, default_params=DEFAULT_PARAMS[name])
        for name in sorted(STRATEGIES)
    ]


@app.post("/strategy-healthcheck")
def strategy_healthcheck(request: StrategyHealthcheckRequest) -> dict[str, Any]:
    return evaluate_strategy_health(
        request.runs,
        edge_thesis=request.edge_thesis,
        thresholds=request.thresholds,
        cost_bps=request.cost_bps,
        factor_report=request.factor_report,
        walk_forward_report=request.walk_forward_report,
    )


@app.post("/strategy-competition")
def strategy_competition(request: StrategyCompetitionRequest) -> list[dict[str, Any]]:
    return rank_strategies(
        request.entries,
        weights=request.weights,
        promote_top_n=request.promote_top_n,
        retire_bottom_streak=request.retire_bottom_streak,
    )


@app.post("/funding-arb/backtest")
def funding_arb_backtest(request: FundingArbRequest) -> dict[str, Any]:
    if request.end <= request.start:
        raise HTTPException(status_code=422, detail={"code": "END_MUST_BE_AFTER_START"})
    try:
        return run_funding_arb_backtest(
            FundingArbConfig(
                symbol=request.symbol,
                source=request.source,
                start=request.start,
                end=request.end,
                timeframe=request.timeframe,
                cash=request.cash,
                min_funding_bps=request.min_funding_bps,
                exit_funding_bps=request.exit_funding_bps,
                taker_fee_bps=request.taker_fee_bps,
                maker_fee_bps=request.maker_fee_bps,
                slippage_bps=request.slippage_bps,
                basis_cost_bps=request.basis_cost_bps,
                borrow_cost_bps_annual=request.borrow_cost_bps_annual,
                settlement_hours=request.settlement_hours,
                cash_rate_annual=request.cash_rate_annual,
                max_holding_events=request.max_holding_events,
                use_maker_fees=request.use_maker_fees,
                max_funding_events=request.max_funding_events,
            )
        )
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc


@app.post("/factor-ic")
def factor_ic(request: FactorICRequest) -> dict[str, Any]:
    if request.end <= request.start:
        raise HTTPException(status_code=422, detail={"code": "END_MUST_BE_AFTER_START"})
    symbols = _factor_symbols(request)
    if not symbols:
        raise HTTPException(status_code=422, detail={"code": "SYMBOLS_REQUIRED"})
    estimated_bars = _estimated_bars_for_range(request.start, request.end, request.timeframe)
    if estimated_bars is not None and estimated_bars > MAX_BARS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TOO_MANY_BARS",
                "message": f"requested range estimates {estimated_bars} bars; max is {MAX_BARS}",
                "max_bars": MAX_BARS,
            },
        )
    frames: dict[str, pd.DataFrame] = {}
    try:
        for symbol in symbols:
            frames[symbol] = data_module.load_ohlcv(
                symbol,
                request.source,
                request.timeframe,
                request.start,
                request.end,
            )
        return evaluate_ohlcv_factor_ic(
            frames,
            [factor.model_dump(exclude_none=True) for factor in request.factors],
            label_periods=request.label_periods,
            groups=request.groups,
            mode=request.mode,
            ic_window=request.ic_window,
            redundancy_threshold=request.redundancy_threshold,
            icir_threshold=request.icir_threshold,
        )
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "FACTOR_IC_FAILED", "message": str(exc)},
        ) from exc


@app.post("/walk-forward")
def walk_forward(request: WalkForwardRequest) -> dict[str, Any]:
    if request.end <= request.start:
        raise HTTPException(status_code=422, detail={"code": "END_MUST_BE_AFTER_START"})
    strategy_cls = STRATEGIES.get(request.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail={"code": "UNKNOWN_STRATEGY"})
    estimated_bars = _estimated_bars_for_range(request.start, request.end, request.timeframe)
    if estimated_bars is not None and estimated_bars > MAX_BARS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TOO_MANY_BARS",
                "message": f"requested range estimates {estimated_bars} bars; max is {MAX_BARS}",
                "max_bars": MAX_BARS,
            },
        )
    param_grid = [_merged_params(request.strategy, params) for params in request.param_grid]
    try:
        for params in param_grid:
            _validate_ordered_params(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": str(exc)}) from exc
    try:
        frame = data_module.load_ohlcv(
            request.symbol,
            request.source,
            request.timeframe,
            request.start,
            request.end,
        )
        return run_walk_forward(
            frame,
            strategy_cls,
            param_grid,
            train_bars=request.train_bars,
            test_bars=request.test_bars,
            step_bars=request.step_bars,
            cash=request.cash,
            commission=request.commission,
            objective=request.objective,
            min_oos_return_pct=request.min_oos_return_pct,
            min_oos_is_return_ratio=request.min_oos_is_return_ratio,
            min_oos_is_sharpe_ratio=request.min_oos_is_sharpe_ratio,
            max_parameter_trials=request.max_parameter_trials,
        )
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "WALK_FORWARD_FAILED", "message": str(exc)},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail={"code": "BACKTEST_FAILED", "message": str(exc)},
        ) from exc


@app.get("/competition/latest")
def latest_competition() -> dict[str, Any]:
    path = _competition_latest_path()
    if not path.exists():
        return _competition_latest_unavailable("no competition run yet")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _competition_latest_unavailable("invalid competition summary")

    if isinstance(data, dict):
        generated_at = _parse_generated_at(data.get("generated_at"))
        max_age_seconds = _competition_max_age_seconds()
        if generated_at is not None and max_age_seconds > 0:
            age_seconds = (datetime.now(generated_at.tzinfo) - generated_at).total_seconds()
            if age_seconds > max_age_seconds:
                return _competition_latest_unavailable("competition run is stale")
    return _competition_latest_payload(data)


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest) -> BacktestResponse:
    strategy_cls = STRATEGIES.get(request.strategy)
    if strategy_cls is None:
        raise HTTPException(status_code=400, detail={"code": "UNKNOWN_STRATEGY"})

    params = {**DEFAULT_PARAMS.get(request.strategy, {}), **request.params}
    fast = _positive_number(params, "fast")
    slow = _positive_number(params, "slow")
    trend = _positive_number(params, "trend")
    if fast is not None and slow is not None and fast >= slow:
        raise HTTPException(status_code=400, detail={"code": "FAST_MUST_BE_LT_SLOW"})
    if slow is not None and trend is not None and slow >= trend:
        raise HTTPException(status_code=400, detail={"code": "SLOW_MUST_BE_LT_TREND"})
    estimated_bars = _estimated_bars(request)
    if estimated_bars is not None and estimated_bars > MAX_BARS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TOO_MANY_BARS",
                "message": f"requested range estimates {estimated_bars} bars; max is {MAX_BARS}",
                "max_bars": MAX_BARS,
            },
        )

    try:
        frame = data_module.load_ohlcv(
            request.symbol,
            request.source,
            request.timeframe,
            request.start,
            request.end,
        )
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc
    try:
        regime_frame = _load_regime_frame(request)
    except DataLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "REGIME_DATA_UNAVAILABLE", "message": str(exc)},
        ) from exc

    if regime_frame is not None:
        frame = _add_regime_column(frame, regime_frame, request.regime_ma)
    else:
        frame = frame.copy()
        frame["RegimeUp"] = True

    if trend is not None and len(frame) <= trend:
        raise HTTPException(status_code=422, detail={"code": "INSUFFICIENT_BARS"})

    configured = _configured_strategy(strategy_cls, params)
    try:
        backtest = Backtest(
            frame,
            configured,
            cash=request.cash,
            commission=request.commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = backtest.run()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail={"code": "BACKTEST_FAILED", "message": str(exc)},
        ) from exc

    response_trades = _trades(stats, frame)
    open_position = _open_position_info(stats, response_trades)
    response_stats = _response_stats(stats, response_trades, open_position)
    return BacktestResponse(
        stats=response_stats,
        equity_curve=_sample_equity_curve(stats),
        trades=response_trades,
        open_position=open_position,
    )
