from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from aegis.backtest_core import (
    CostModel,
    metrics_from_returns,
    trade_scorecard,
    trade_scorecard_to_dict,
)

Side = Literal["long", "short"]
ExitReason = Literal["take_profit", "stop_loss", "end_of_data"]


@dataclass(frozen=True)
class VolumeDivergenceBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class VolumeDivergenceConfig:
    lookback_bars: int = 20
    rsi_period: int = 14
    require_rsi_divergence: bool = False
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    require_macd_histogram_divergence: bool = False
    require_liquidity_sweep: bool = False
    require_choch_confirmation: bool = False
    choch_lookback_bars: int = 5
    reward_risk: float = 3.0
    risk_per_trade_fraction: float | None = None
    max_position_notional_fraction: float | None = None
    allow_long: bool = True
    allow_short: bool = True
    annualization_periods: int = 365 * 6


@dataclass(frozen=True)
class VolumeDivergenceTrade:
    symbol: str
    side: Side
    signal_timestamp: int
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    gross_return: float
    net_return_after_costs: float
    fee_cost: float
    slippage_cost: float
    risk_per_trade_fraction: float | None
    position_notional_fraction: float
    stop_distance_fraction: float
    exit_reason: ExitReason
    previous_extreme_price: float
    previous_extreme_volume: float
    previous_extreme_rsi: float | None
    previous_extreme_macd_histogram: float | None
    signal_extreme_price: float
    signal_extreme_volume: float
    signal_extreme_rsi: float | None
    signal_extreme_macd_histogram: float | None
    liquidity_sweep_pass: bool
    choch_level: float | None
    choch_confirmation_timestamp: int | None


@dataclass(frozen=True)
class VolumeDivergenceResult:
    symbol: str
    bars: int
    config: VolumeDivergenceConfig
    cost_model: CostModel
    returns: tuple[float, ...]
    buy_hold_returns: tuple[float, ...]
    trades: tuple[VolumeDivergenceTrade, ...]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    trade_scorecard: dict[str, float | int]
    discipline: dict[str, object]


@dataclass
class _Extreme:
    index: int
    price: float
    volume: float
    rsi: float | None
    macd_histogram: float | None


@dataclass
class _PendingEntry:
    side: Side
    signal_index: int
    entry_index: int
    stop_price: float
    take_profit_price: float
    previous_extreme: _Extreme
    signal_extreme: _Extreme
    liquidity_sweep_pass: bool
    choch_level: float | None
    choch_confirmation_index: int | None


@dataclass
class _AwaitingConfirmation:
    side: Side
    signal_index: int
    stop_price: float
    previous_extreme: _Extreme
    signal_extreme: _Extreme
    liquidity_sweep_pass: bool
    choch_level: float


@dataclass
class _OpenPosition:
    side: Side
    signal_index: int
    entry_index: int
    entry_price: float
    mark_price: float
    position_notional_fraction: float
    stop_distance_fraction: float
    stop_price: float
    take_profit_price: float
    previous_extreme: _Extreme
    signal_extreme: _Extreme
    liquidity_sweep_pass: bool
    choch_level: float | None
    choch_confirmation_index: int | None


def run_volume_divergence_rr(
    bars: list[VolumeDivergenceBar] | tuple[VolumeDivergenceBar, ...],
    *,
    symbol: str,
    config: VolumeDivergenceConfig = VolumeDivergenceConfig(),
    cost_model: CostModel = CostModel(fee_bps=5.0, slippage_bps=5.0),
) -> VolumeDivergenceResult:
    """Backtest the 4h volume-divergence fixed 1:3 reward/risk rule.

    The signal is deliberately left-looking only:
    - Long: current bar makes a new lookback low, is lower than the previous
      recorded low, and volume is lower than that previous low's volume.
    - Short: mirror image for highs.
    A signal at bar ``t`` enters at bar ``t+1`` open. Same-bar stop/take-profit
    collisions are resolved conservatively as stop first.
    """

    if config.lookback_bars < 1:
        raise ValueError("lookback_bars must be >= 1")
    if config.rsi_period < 1:
        raise ValueError("rsi_period must be >= 1")
    if config.macd_fast_period < 1 or config.macd_slow_period < 1 or config.macd_signal_period < 1:
        raise ValueError("MACD periods must be >= 1")
    if config.macd_fast_period >= config.macd_slow_period:
        raise ValueError("macd_fast_period must be less than macd_slow_period")
    if config.choch_lookback_bars < 1:
        raise ValueError("choch_lookback_bars must be >= 1")
    if config.reward_risk <= 0:
        raise ValueError("reward_risk must be positive")
    if config.risk_per_trade_fraction is not None and config.risk_per_trade_fraction <= 0:
        raise ValueError("risk_per_trade_fraction must be positive when provided")
    if config.max_position_notional_fraction is not None and config.max_position_notional_fraction <= 0:
        raise ValueError("max_position_notional_fraction must be positive when provided")
    if len(bars) < config.lookback_bars + 3:
        raise ValueError("not enough bars for volume divergence backtest")

    ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
    rsi_values = _rsi_values(ordered, period=config.rsi_period)
    macd_histogram_values = _macd_histogram_values(
        ordered,
        fast_period=config.macd_fast_period,
        slow_period=config.macd_slow_period,
        signal_period=config.macd_signal_period,
    )
    returns = [0.0 for _ in ordered]
    buy_hold = _buy_hold_returns(ordered)
    trades: list[VolumeDivergenceTrade] = []
    last_low: _Extreme | None = None
    last_high: _Extreme | None = None
    pending: _PendingEntry | None = None
    awaiting_confirmation: _AwaitingConfirmation | None = None
    position: _OpenPosition | None = None
    total_cost = 0.0

    for index, bar in enumerate(ordered):
        if pending is not None and pending.entry_index == index:
            entry_price = bar.open
            risk = _risk(pending.side, entry_price, pending.stop_price)
            if risk > 0:
                stop_distance_fraction = _stop_distance_fraction(pending.side, entry_price, pending.stop_price)
                position_notional_fraction = _position_notional_fraction(
                    stop_distance_fraction=stop_distance_fraction,
                    risk_per_trade_fraction=config.risk_per_trade_fraction,
                    max_position_notional_fraction=config.max_position_notional_fraction,
                )
                position = _OpenPosition(
                    side=pending.side,
                    signal_index=pending.signal_index,
                    entry_index=index,
                    entry_price=entry_price,
                    mark_price=entry_price,
                    position_notional_fraction=position_notional_fraction,
                    stop_distance_fraction=stop_distance_fraction,
                    stop_price=pending.stop_price,
                    take_profit_price=pending.take_profit_price,
                    previous_extreme=pending.previous_extreme,
                    signal_extreme=pending.signal_extreme,
                    liquidity_sweep_pass=pending.liquidity_sweep_pass,
                    choch_level=pending.choch_level,
                    choch_confirmation_index=pending.choch_confirmation_index,
                )
                entry_cost = cost_model.one_way_cost * position_notional_fraction
                returns[index] -= entry_cost
                total_cost += entry_cost
            pending = None

        if position is not None:
            exit_price, exit_reason = _exit_for_bar(position, bar)
            if exit_price is None:
                period_return = (
                    _signed_return(position.side, position.mark_price, bar.close)
                    * position.position_notional_fraction
                )
                returns[index] += period_return
                position.mark_price = bar.close
            else:
                period_return = (
                    _signed_return(position.side, position.mark_price, exit_price)
                    * position.position_notional_fraction
                )
                exit_cost = cost_model.one_way_cost * position.position_notional_fraction
                returns[index] += period_return - exit_cost
                total_cost += exit_cost
                gross = (
                    _signed_return(position.side, position.entry_price, exit_price)
                    * position.position_notional_fraction
                )
                net = gross - cost_model.round_trip_cost * position.position_notional_fraction
                trades.append(
                    VolumeDivergenceTrade(
                        symbol=symbol,
                        side=position.side,
                        signal_timestamp=ordered[position.signal_index].timestamp,
                        entry_timestamp=ordered[position.entry_index].timestamp,
                        exit_timestamp=bar.timestamp,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        stop_price=position.stop_price,
                        take_profit_price=position.take_profit_price,
                        gross_return=gross,
                        net_return_after_costs=net,
                        fee_cost=2.0 * cost_model.fee_bps / 10_000.0 * position.position_notional_fraction,
                        slippage_cost=(
                            2.0 * cost_model.slippage_bps / 10_000.0 * position.position_notional_fraction
                        ),
                        risk_per_trade_fraction=config.risk_per_trade_fraction,
                        position_notional_fraction=position.position_notional_fraction,
                        stop_distance_fraction=position.stop_distance_fraction,
                        exit_reason=exit_reason,
                        previous_extreme_price=position.previous_extreme.price,
                        previous_extreme_volume=position.previous_extreme.volume,
                        previous_extreme_rsi=position.previous_extreme.rsi,
                        previous_extreme_macd_histogram=position.previous_extreme.macd_histogram,
                        signal_extreme_price=position.signal_extreme.price,
                        signal_extreme_volume=position.signal_extreme.volume,
                        signal_extreme_rsi=position.signal_extreme.rsi,
                        signal_extreme_macd_histogram=position.signal_extreme.macd_histogram,
                        liquidity_sweep_pass=position.liquidity_sweep_pass,
                        choch_level=position.choch_level,
                        choch_confirmation_timestamp=(
                            ordered[position.choch_confirmation_index].timestamp
                            if position.choch_confirmation_index is not None
                            else None
                        ),
                    )
                )
                position = None

        if (
            awaiting_confirmation is not None
            and pending is None
            and position is None
            and index > awaiting_confirmation.signal_index
        ):
            if _pre_entry_stop_breached(awaiting_confirmation, bar):
                awaiting_confirmation = None
            elif _choch_confirmation_pass(awaiting_confirmation, bar) and index + 1 < len(ordered):
                entry_price = ordered[index + 1].open
                risk = _risk(awaiting_confirmation.side, entry_price, awaiting_confirmation.stop_price)
                if risk > 0:
                    pending = _PendingEntry(
                        side=awaiting_confirmation.side,
                        signal_index=awaiting_confirmation.signal_index,
                        entry_index=index + 1,
                        stop_price=awaiting_confirmation.stop_price,
                        take_profit_price=(
                            entry_price + config.reward_risk * risk
                            if awaiting_confirmation.side == "long"
                            else entry_price - config.reward_risk * risk
                        ),
                        previous_extreme=awaiting_confirmation.previous_extreme,
                        signal_extreme=awaiting_confirmation.signal_extreme,
                        liquidity_sweep_pass=awaiting_confirmation.liquidity_sweep_pass,
                        choch_level=awaiting_confirmation.choch_level,
                        choch_confirmation_index=index,
                    )
                awaiting_confirmation = None

        if index >= config.lookback_bars and index + 1 < len(ordered) and position is None:
            long_signal, short_signal, low_extreme, high_extreme = _signals_at(
                ordered,
                index=index,
                config=config,
                last_low=last_low,
                last_high=last_high,
                rsi_values=rsi_values,
                macd_histogram_values=macd_histogram_values,
            )
            if pending is None and awaiting_confirmation is None:
                if long_signal is not None and short_signal is None:
                    if config.require_choch_confirmation:
                        awaiting_confirmation = _awaiting_confirmation_from_signal(
                            ordered,
                            signal=long_signal,
                            choch_lookback_bars=config.choch_lookback_bars,
                        )
                    else:
                        pending = long_signal
                elif short_signal is not None and long_signal is None:
                    if config.require_choch_confirmation:
                        awaiting_confirmation = _awaiting_confirmation_from_signal(
                            ordered,
                            signal=short_signal,
                            choch_lookback_bars=config.choch_lookback_bars,
                        )
                    else:
                        pending = short_signal
            if low_extreme is not None:
                last_low = low_extreme
            if high_extreme is not None:
                last_high = high_extreme

    if position is not None:
        final_bar = ordered[-1]
        exit_price = final_bar.close
        period_return = (
            _signed_return(position.side, position.mark_price, exit_price)
            * position.position_notional_fraction
        )
        exit_cost = cost_model.one_way_cost * position.position_notional_fraction
        returns[-1] += period_return - exit_cost
        total_cost += exit_cost
        gross = (
            _signed_return(position.side, position.entry_price, exit_price)
            * position.position_notional_fraction
        )
        net = gross - cost_model.round_trip_cost * position.position_notional_fraction
        trades.append(
            VolumeDivergenceTrade(
                symbol=symbol,
                side=position.side,
                signal_timestamp=ordered[position.signal_index].timestamp,
                entry_timestamp=ordered[position.entry_index].timestamp,
                exit_timestamp=final_bar.timestamp,
                entry_price=position.entry_price,
                exit_price=exit_price,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
                gross_return=gross,
                net_return_after_costs=net,
                fee_cost=2.0 * cost_model.fee_bps / 10_000.0 * position.position_notional_fraction,
                slippage_cost=2.0 * cost_model.slippage_bps / 10_000.0 * position.position_notional_fraction,
                risk_per_trade_fraction=config.risk_per_trade_fraction,
                position_notional_fraction=position.position_notional_fraction,
                stop_distance_fraction=position.stop_distance_fraction,
                exit_reason="end_of_data",
                previous_extreme_price=position.previous_extreme.price,
                previous_extreme_volume=position.previous_extreme.volume,
                previous_extreme_rsi=position.previous_extreme.rsi,
                previous_extreme_macd_histogram=position.previous_extreme.macd_histogram,
                signal_extreme_price=position.signal_extreme.price,
                signal_extreme_volume=position.signal_extreme.volume,
                signal_extreme_rsi=position.signal_extreme.rsi,
                signal_extreme_macd_histogram=position.signal_extreme.macd_histogram,
                liquidity_sweep_pass=position.liquidity_sweep_pass,
                choch_level=position.choch_level,
                choch_confirmation_timestamp=(
                    ordered[position.choch_confirmation_index].timestamp
                    if position.choch_confirmation_index is not None
                    else None
                ),
            )
        )

    strategy_metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=2.0 * len(trades),
        net_cost=total_cost,
        include_initial_equity=True,
    )
    benchmark_metrics = metrics_from_returns(
        buy_hold,
        annualization_periods=config.annualization_periods,
        turnover=0.0,
        net_cost=0.0,
        include_initial_equity=True,
    )
    scorecard = trade_scorecard([trade.net_return_after_costs for trade in trades])
    return VolumeDivergenceResult(
        symbol=symbol,
        bars=len(ordered),
        config=config,
        cost_model=cost_model,
        returns=tuple(returns),
        buy_hold_returns=buy_hold,
        trades=tuple(trades),
        metrics=asdict(strategy_metrics),
        benchmark_metrics=asdict(benchmark_metrics),
        trade_scorecard=trade_scorecard_to_dict(scorecard),
        discipline={
            "timeframe": "4h expected by caller",
            "t_plus_1_execution": True,
            "lookahead": "current signal bar uses only current and prior bars",
            "entry": "signal bar close decision, next bar open execution",
            "stop_loss": "signal extreme price",
            "take_profit": f"{config.reward_risk:g}:1 reward/risk from entry",
            "position_sizing": (
                f"fixed account risk {config.risk_per_trade_fraction:.4%} per trade"
                if config.risk_per_trade_fraction is not None
                else "full price exposure; no fixed account-risk sizing"
            ),
            "max_position_notional": (
                f"{config.max_position_notional_fraction:.2f}x account equity"
                if config.max_position_notional_fraction is not None
                else "not capped"
            ),
            "rsi_filter": (
                f"RSI({config.rsi_period}) divergence required"
                if config.require_rsi_divergence
                else "not required"
            ),
            "macd_histogram_filter": (
                f"MACD({config.macd_fast_period},{config.macd_slow_period},"
                f"{config.macd_signal_period}) histogram divergence required"
                if config.require_macd_histogram_divergence
                else "not required"
            ),
            "liquidity_sweep_filter": (
                "signal bar must sweep the previous extreme and close back through it"
                if config.require_liquidity_sweep
                else "not required"
            ),
            "choch_confirmation": (
                f"after signal, require close through prior {config.choch_lookback_bars}-bar structure, "
                "then enter next bar open"
                if config.require_choch_confirmation
                else "not required"
            ),
            "same_bar_stop_and_take_profit": "stop_loss_first_conservative",
            "costs": "fee and slippage charged once on entry and once on exit",
        },
    )


def result_to_dict(result: VolumeDivergenceResult, *, include_trades: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": result.symbol,
        "bars": result.bars,
        "config": asdict(result.config),
        "cost_model": asdict(result.cost_model),
        "metrics": result.metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "trade_scorecard": result.trade_scorecard,
        "discipline": result.discipline,
    }
    if include_trades:
        payload["trades"] = [asdict(trade) for trade in result.trades]
    return payload


def _signals_at(
    bars: tuple[VolumeDivergenceBar, ...],
    *,
    index: int,
    config: VolumeDivergenceConfig,
    last_low: _Extreme | None,
    last_high: _Extreme | None,
    rsi_values: tuple[float | None, ...],
    macd_histogram_values: tuple[float | None, ...],
) -> tuple[_PendingEntry | None, _PendingEntry | None, _Extreme | None, _Extreme | None]:
    bar = bars[index]
    prior = bars[index - config.lookback_bars : index]
    new_low = bar.low < min(item.low for item in prior)
    new_high = bar.high > max(item.high for item in prior)
    current_rsi = rsi_values[index]
    current_macd_histogram = macd_histogram_values[index]
    low_extreme = (
        _Extreme(
            index=index,
            price=bar.low,
            volume=bar.volume,
            rsi=current_rsi,
            macd_histogram=current_macd_histogram,
        )
        if new_low
        else None
    )
    high_extreme = (
        _Extreme(
            index=index,
            price=bar.high,
            volume=bar.volume,
            rsi=current_rsi,
            macd_histogram=current_macd_histogram,
        )
        if new_high
        else None
    )
    long_signal: _PendingEntry | None = None
    short_signal: _PendingEntry | None = None
    if config.allow_long and low_extreme is not None and last_low is not None:
        if (
            low_extreme.price < last_low.price
            and low_extreme.volume < last_low.volume
            and _rsi_divergence_pass(
                side="long",
                previous_rsi=last_low.rsi,
                current_rsi=low_extreme.rsi,
                required=config.require_rsi_divergence,
            )
            and _macd_histogram_divergence_pass(
                side="long",
                previous_histogram=last_low.macd_histogram,
                current_histogram=low_extreme.macd_histogram,
                required=config.require_macd_histogram_divergence,
            )
            and _liquidity_sweep_pass(
                side="long",
                bar=bar,
                previous_extreme_price=last_low.price,
                required=config.require_liquidity_sweep,
            )
        ):
            entry_price = bars[index + 1].open
            risk = entry_price - low_extreme.price
            if risk > 0:
                long_signal = _PendingEntry(
                    side="long",
                    signal_index=index,
                    entry_index=index + 1,
                    stop_price=low_extreme.price,
                    take_profit_price=entry_price + config.reward_risk * risk,
                    previous_extreme=last_low,
                    signal_extreme=low_extreme,
                    liquidity_sweep_pass=True,
                    choch_level=None,
                    choch_confirmation_index=None,
                )
    if config.allow_short and high_extreme is not None and last_high is not None:
        if (
            high_extreme.price > last_high.price
            and high_extreme.volume < last_high.volume
            and _rsi_divergence_pass(
                side="short",
                previous_rsi=last_high.rsi,
                current_rsi=high_extreme.rsi,
                required=config.require_rsi_divergence,
            )
            and _macd_histogram_divergence_pass(
                side="short",
                previous_histogram=last_high.macd_histogram,
                current_histogram=high_extreme.macd_histogram,
                required=config.require_macd_histogram_divergence,
            )
            and _liquidity_sweep_pass(
                side="short",
                bar=bar,
                previous_extreme_price=last_high.price,
                required=config.require_liquidity_sweep,
            )
        ):
            entry_price = bars[index + 1].open
            risk = high_extreme.price - entry_price
            if risk > 0:
                short_signal = _PendingEntry(
                    side="short",
                    signal_index=index,
                    entry_index=index + 1,
                    stop_price=high_extreme.price,
                    take_profit_price=entry_price - config.reward_risk * risk,
                    previous_extreme=last_high,
                    signal_extreme=high_extreme,
                    liquidity_sweep_pass=True,
                    choch_level=None,
                    choch_confirmation_index=None,
                )
    return long_signal, short_signal, low_extreme, high_extreme


def _liquidity_sweep_pass(
    *,
    side: Side,
    bar: VolumeDivergenceBar,
    previous_extreme_price: float,
    required: bool,
) -> bool:
    if not required:
        return True
    if side == "long":
        return bar.low < previous_extreme_price and bar.close > previous_extreme_price
    return bar.high > previous_extreme_price and bar.close < previous_extreme_price


def _awaiting_confirmation_from_signal(
    bars: tuple[VolumeDivergenceBar, ...],
    *,
    signal: _PendingEntry,
    choch_lookback_bars: int,
) -> _AwaitingConfirmation:
    start = max(0, signal.signal_index - choch_lookback_bars)
    prior = bars[start : signal.signal_index]
    if not prior:
        raise ValueError("not enough prior bars for CHOCH confirmation")
    choch_level = max(bar.high for bar in prior) if signal.side == "long" else min(bar.low for bar in prior)
    return _AwaitingConfirmation(
        side=signal.side,
        signal_index=signal.signal_index,
        stop_price=signal.stop_price,
        previous_extreme=signal.previous_extreme,
        signal_extreme=signal.signal_extreme,
        liquidity_sweep_pass=signal.liquidity_sweep_pass,
        choch_level=choch_level,
    )


def _pre_entry_stop_breached(setup: _AwaitingConfirmation, bar: VolumeDivergenceBar) -> bool:
    return bar.low <= setup.stop_price if setup.side == "long" else bar.high >= setup.stop_price


def _choch_confirmation_pass(setup: _AwaitingConfirmation, bar: VolumeDivergenceBar) -> bool:
    return bar.close > setup.choch_level if setup.side == "long" else bar.close < setup.choch_level


def _macd_histogram_divergence_pass(
    *,
    side: Side,
    previous_histogram: float | None,
    current_histogram: float | None,
    required: bool,
) -> bool:
    if not required:
        return True
    if previous_histogram is None or current_histogram is None:
        return False
    return current_histogram > previous_histogram if side == "long" else current_histogram < previous_histogram


def _rsi_divergence_pass(
    *,
    side: Side,
    previous_rsi: float | None,
    current_rsi: float | None,
    required: bool,
) -> bool:
    if not required:
        return True
    if previous_rsi is None or current_rsi is None:
        return False
    return current_rsi > previous_rsi if side == "long" else current_rsi < previous_rsi


def _rsi_values(
    bars: tuple[VolumeDivergenceBar, ...],
    *,
    period: int,
) -> tuple[float | None, ...]:
    if period < 1:
        raise ValueError("period must be >= 1")
    values: list[float | None] = [None for _ in bars]
    if len(bars) <= period:
        return tuple(values)
    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, period + 1):
        change = bars[index].close - bars[index - 1].close
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    values[period] = _rsi_from_averages(average_gain, average_loss)
    for index in range(period + 1, len(bars)):
        change = bars[index].close - bars[index - 1].close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        average_gain = ((period - 1) * average_gain + gain) / period
        average_loss = ((period - 1) * average_loss + loss) / period
        values[index] = _rsi_from_averages(average_gain, average_loss)
    return tuple(values)


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    if average_loss == 0.0:
        return 100.0 if average_gain > 0.0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def _macd_histogram_values(
    bars: tuple[VolumeDivergenceBar, ...],
    *,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[float | None, ...]:
    closes = tuple(bar.close for bar in bars)
    fast = _ema_values(closes, period=fast_period)
    slow = _ema_values(closes, period=slow_period)
    macd_line: list[float | None] = []
    for fast_value, slow_value in zip(fast, slow, strict=True):
        if fast_value is None or slow_value is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_value - slow_value)
    signal = _ema_values(tuple(macd_line), period=signal_period)
    histogram: list[float | None] = []
    for macd_value, signal_value in zip(macd_line, signal, strict=True):
        if macd_value is None or signal_value is None:
            histogram.append(None)
        else:
            histogram.append(macd_value - signal_value)
    return tuple(histogram)


def _ema_values(values: tuple[float | None, ...] | tuple[float, ...], *, period: int) -> tuple[float | None, ...]:
    if period < 1:
        raise ValueError("period must be >= 1")
    result: list[float | None] = [None for _ in values]
    valid_count = 0
    seed: list[float] = []
    previous_ema: float | None = None
    multiplier = 2.0 / (period + 1.0)
    for index, raw_value in enumerate(values):
        if raw_value is None:
            continue
        value = float(raw_value)
        if previous_ema is None:
            seed.append(value)
            valid_count += 1
            if valid_count == period:
                previous_ema = sum(seed) / period
                result[index] = previous_ema
        else:
            previous_ema = (value - previous_ema) * multiplier + previous_ema
            result[index] = previous_ema
    return tuple(result)


def _exit_for_bar(position: _OpenPosition, bar: VolumeDivergenceBar) -> tuple[float | None, ExitReason | None]:
    if position.side == "long":
        stop_hit = bar.low <= position.stop_price
        take_profit_hit = bar.high >= position.take_profit_price
    else:
        stop_hit = bar.high >= position.stop_price
        take_profit_hit = bar.low <= position.take_profit_price
    if stop_hit:
        return position.stop_price, "stop_loss"
    if take_profit_hit:
        return position.take_profit_price, "take_profit"
    return None, None


def _risk(side: Side, entry_price: float, stop_price: float) -> float:
    return entry_price - stop_price if side == "long" else stop_price - entry_price


def _stop_distance_fraction(side: Side, entry_price: float, stop_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return _risk(side, entry_price, stop_price) / entry_price


def _position_notional_fraction(
    *,
    stop_distance_fraction: float,
    risk_per_trade_fraction: float | None,
    max_position_notional_fraction: float | None,
) -> float:
    if risk_per_trade_fraction is None:
        raw_position = 1.0
    elif stop_distance_fraction <= 0:
        raw_position = 0.0
    else:
        raw_position = risk_per_trade_fraction / stop_distance_fraction
    if max_position_notional_fraction is None:
        return raw_position
    return min(raw_position, max_position_notional_fraction)


def _signed_return(side: Side, start_price: float, end_price: float) -> float:
    if start_price <= 0:
        return 0.0
    raw = end_price / start_price - 1.0
    return raw if side == "long" else -raw


def _buy_hold_returns(bars: tuple[VolumeDivergenceBar, ...]) -> tuple[float, ...]:
    values = [0.0 for _ in bars]
    for index in range(1, len(bars)):
        previous = bars[index - 1].close
        values[index] = bars[index].close / previous - 1.0 if previous > 0 else 0.0
    return tuple(values)
