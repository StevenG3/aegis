from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CryptoBar:
    timestamp: int
    open: float
    close: float


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    funding_bps_per_period: float = 0.0
    funding_label: str = "N/A for spot long-only"

    @property
    def round_trip_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass(frozen=True)
class TsmomConfig:
    lookbacks: tuple[int, ...] = (30, 60, 90, 120)
    train_bars: int = 730
    test_bars: int = 180
    step_bars: int = 180
    annualization_periods: int = 365
    allow_short: bool = False


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    positive_period_win_rate: float
    annualized_turnover: float
    net_cost: float


@dataclass(frozen=True)
class SimulationResult:
    returns: tuple[float, ...]
    equity: tuple[float, ...]
    positions: tuple[int, ...]
    costs: tuple[float, ...]
    turnover: float
    metrics: BacktestMetrics
    first_trade_index: int
    last_trade_index: int


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    selected_lookback: int
    train_score: float
    strategy_metrics: BacktestMetrics
    buy_hold_metrics: BacktestMetrics
    excess_return: float
    beats_buy_hold_return: bool
    beats_buy_hold_sharpe: bool
    selector_max_bar_seen: int
    first_oos_execution_bar: int


@dataclass(frozen=True)
class WalkForwardReport:
    status: str
    verdict: str
    reason: str
    symbols: tuple[str, ...]
    config: TsmomConfig
    cost_model: CostModel
    windows: tuple[WalkForwardWindow, ...]
    standard_metrics: dict[str, dict[str, float]]
    benchmark_metrics: dict[str, dict[str, float]]
    summary: dict[str, float | int | str | None]
    multiple_testing: dict[str, float | int | str | None]
    safety: dict[str, bool | str]


DEFAULT_TSMOM_CONFIG = TsmomConfig()
DEFAULT_COST_MODEL = CostModel()


def run_crypto_tsmom_walk_forward(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    *,
    config: TsmomConfig = DEFAULT_TSMOM_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> WalkForwardReport:
    if not bars_by_symbol:
        return _insufficient("no symbols supplied", (), config, cost_model)
    symbols = tuple(sorted(bars_by_symbol))
    min_bars = min(len(bars_by_symbol[symbol]) for symbol in symbols)
    min_lookback = max(config.lookbacks)
    required = config.train_bars + config.test_bars + min_lookback + 2
    if min_bars < required:
        return _insufficient(
            f"not enough bars for one walk-forward split: have {min_bars}, need {required}",
            symbols,
            config,
            cost_model,
        )
    windows: list[WalkForwardWindow] = []
    starts = _window_starts(min_bars, config.train_bars, config.test_bars, config.step_bars)
    if not starts:
        return _insufficient("not enough bars for configured windows", symbols, config, cost_model)
    for index, train_start in enumerate(starts):
        train_end = train_start + config.train_bars
        test_start = train_end
        test_end = train_end + config.test_bars
        selected_lookback, train_score = _select_lookback(
            bars_by_symbol,
            symbols,
            lookbacks=config.lookbacks,
            start=train_start,
            end=train_end,
            config=config,
            cost_model=cost_model,
        )
        strategy = _portfolio_strategy_result(
            bars_by_symbol,
            symbols,
            selected_lookback,
            start=test_start,
            end=test_end,
            config=config,
            cost_model=cost_model,
        )
        buy_hold = _portfolio_buy_hold_result(
            bars_by_symbol,
            symbols,
            start=test_start,
            end=test_end,
            cost_model=cost_model,
            annualization_periods=config.annualization_periods,
        )
        windows.append(
            WalkForwardWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                selected_lookback=selected_lookback,
                train_score=train_score,
                strategy_metrics=strategy.metrics,
                buy_hold_metrics=buy_hold.metrics,
                excess_return=strategy.metrics.total_return - buy_hold.metrics.total_return,
                beats_buy_hold_return=strategy.metrics.total_return > buy_hold.metrics.total_return,
                beats_buy_hold_sharpe=strategy.metrics.sharpe > buy_hold.metrics.sharpe,
                selector_max_bar_seen=train_end - 1,
                first_oos_execution_bar=test_start,
            )
        )
    all_strategy_returns = tuple(
        value
        for window in windows
        for value in _portfolio_strategy_result(
            bars_by_symbol,
            symbols,
            window.selected_lookback,
            start=window.test_start,
            end=window.test_end,
            config=config,
            cost_model=cost_model,
        ).returns
    )
    all_buy_hold_returns = tuple(
        value
        for window in windows
        for value in _portfolio_buy_hold_result(
            bars_by_symbol,
            symbols,
            start=window.test_start,
            end=window.test_end,
            cost_model=cost_model,
            annualization_periods=config.annualization_periods,
        ).returns
    )
    strategy_metrics = metrics_from_returns(
        all_strategy_returns,
        annualization_periods=config.annualization_periods,
        turnover=sum(window.strategy_metrics.annualized_turnover for window in windows)
        / len(windows),
        net_cost=sum(window.strategy_metrics.net_cost for window in windows),
    )
    buy_hold_metrics = metrics_from_returns(
        all_buy_hold_returns,
        annualization_periods=config.annualization_periods,
        turnover=(
            sum(window.buy_hold_metrics.annualized_turnover for window in windows)
            / len(windows)
        ),
        net_cost=sum(window.buy_hold_metrics.net_cost for window in windows),
    )
    excess = tuple(window.excess_return for window in windows)
    multiple_testing = fdr_report(
        [
            _static_oos_p_value(bars_by_symbol, symbols, lookback, config, cost_model)
            for lookback in config.lookbacks
        ]
    )
    summary = _summary(windows, excess, strategy_metrics, buy_hold_metrics)
    verdict, reason = _verdict(summary, multiple_testing)
    return WalkForwardReport(
        status="OK",
        verdict=verdict,
        reason=reason,
        symbols=symbols,
        config=config,
        cost_model=cost_model,
        windows=tuple(windows),
        standard_metrics={"tsmom": metrics_to_dict(strategy_metrics)},
        benchmark_metrics={"buy_and_hold": metrics_to_dict(buy_hold_metrics)},
        summary=summary,
        multiple_testing=multiple_testing,
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def simulate_tsmom(
    bars: Sequence[CryptoBar],
    *,
    lookback: int,
    start: int,
    end: int,
    config: TsmomConfig,
    cost_model: CostModel,
) -> SimulationResult:
    returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    prev_position = 0
    turnover = 0.0
    start = max(start, lookback + 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        momentum = bars[decision_index].close / bars[decision_index - lookback].close - 1.0
        if momentum > 0:
            position = 1
        elif config.allow_short and momentum < 0:
            position = -1
        else:
            position = 0
        gross_return = position * (
            bars[execution_index + 1].open / bars[execution_index].open - 1.0
        )
        position_change = abs(position - prev_position)
        trade_cost = position_change * cost_model.round_trip_bps / 10_000.0
        funding_cost = abs(position) * cost_model.funding_bps_per_period / 10_000.0
        net_return = gross_return - trade_cost - funding_cost
        returns.append(net_return)
        positions.append(position)
        costs.append(trade_cost + funding_cost)
        turnover += position_change
        prev_position = position
    metrics = metrics_from_returns(
        tuple(returns),
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return SimulationResult(
        returns=tuple(returns),
        equity=equity_curve(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        turnover=turnover,
        metrics=metrics,
        first_trade_index=start,
        last_trade_index=max(start, end - 1),
    )


def simulate_buy_hold(
    bars: Sequence[CryptoBar],
    *,
    start: int,
    end: int,
    cost_model: CostModel,
    annualization_periods: int,
) -> SimulationResult:
    returns: list[float] = []
    costs: list[float] = []
    start = max(start, 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        trade_cost = cost_model.round_trip_bps / 10_000.0 if execution_index == start else 0.0
        returns.append(
            bars[execution_index + 1].open / bars[execution_index].open
            - 1.0
            - trade_cost
        )
        costs.append(trade_cost)
    metrics = metrics_from_returns(
        tuple(returns),
        annualization_periods=annualization_periods,
        turnover=1.0,
        net_cost=sum(costs),
    )
    return SimulationResult(
        returns=tuple(returns),
        equity=equity_curve(returns),
        positions=tuple(1 for _ in returns),
        costs=tuple(costs),
        turnover=1.0,
        metrics=metrics,
        first_trade_index=start,
        last_trade_index=max(start, end - 1),
    )


def metrics_from_returns(
    returns: Sequence[float],
    *,
    annualization_periods: int,
    turnover: float,
    net_cost: float,
) -> BacktestMetrics:
    if not returns:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, net_cost)
    equity = equity_curve(returns)
    total_return = equity[-1] - 1.0
    max_dd = max_drawdown(equity)
    mean = statistics.fmean(returns)
    stdev = statistics.pstdev(returns)
    downside = [value for value in returns if value < 0]
    downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    years = len(returns) / annualization_periods
    cagr = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    sharpe = mean / stdev * math.sqrt(annualization_periods) if stdev > 0 else 0.0
    sortino = mean / downside_dev * math.sqrt(annualization_periods) if downside_dev > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return BacktestMetrics(
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        positive_period_win_rate=sum(1 for value in returns if value > 0) / len(returns),
        annualized_turnover=turnover / max(years, 1e-9),
        net_cost=net_cost,
    )


def equity_curve(returns: Iterable[float]) -> tuple[float, ...]:
    equity = 1.0
    curve = [equity]
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return tuple(curve)


def max_drawdown(equity: Sequence[float]) -> float:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak else 0.0
        worst = min(worst, drawdown)
    return worst


def sign_test_p_value(excess_returns: Sequence[float]) -> float:
    non_zero = [value for value in excess_returns if value != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0)
    losses = n - wins
    tail_count = min(wins, losses)
    tail = float(sum(math.comb(n, k) for k in range(0, tail_count + 1)) / (2**n))
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 1_000,
    seed: int = 44,
) -> dict[str, float | None]:
    if not values:
        return {"p05": None, "p50": None, "p95": None}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(iterations))
    return {
        "p05": means[int(iterations * 0.05)],
        "p50": means[int(iterations * 0.50)],
        "p95": means[int(iterations * 0.95)],
    }


def benjamini_hochberg(p_values: Sequence[float], *, alpha: float = 0.10) -> list[bool]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    passed = [False for _ in p_values]
    max_rank = -1
    m = len(indexed)
    for rank, (_index, p_value) in enumerate(indexed, start=1):
        if p_value <= alpha * rank / m:
            max_rank = rank
    if max_rank >= 1:
        for _rank, (index, _p_value) in enumerate(indexed, start=1):
            if _rank <= max_rank:
                passed[index] = True
    return passed


def report_to_dict(report: WalkForwardReport) -> dict[str, object]:
    return {
        "status": report.status,
        "verdict": report.verdict,
        "reason": report.reason,
        "symbols": list(report.symbols),
        "config": {
            "lookbacks": list(report.config.lookbacks),
            "train_bars": report.config.train_bars,
            "test_bars": report.config.test_bars,
            "step_bars": report.config.step_bars,
            "annualization_periods": report.config.annualization_periods,
            "allow_short": report.config.allow_short,
        },
        "cost_model": {
            "fee_bps": report.cost_model.fee_bps,
            "slippage_bps": report.cost_model.slippage_bps,
            "funding_bps_per_period": report.cost_model.funding_bps_per_period,
            "funding_label": report.cost_model.funding_label,
        },
        "windows": [
            {
                "index": window.index,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_lookback": window.selected_lookback,
                "train_score": window.train_score,
                "strategy_metrics": metrics_to_dict(window.strategy_metrics),
                "buy_hold_metrics": metrics_to_dict(window.buy_hold_metrics),
                "excess_return": window.excess_return,
                "beats_buy_hold_return": window.beats_buy_hold_return,
                "beats_buy_hold_sharpe": window.beats_buy_hold_sharpe,
                "selector_max_bar_seen": window.selector_max_bar_seen,
                "first_oos_execution_bar": window.first_oos_execution_bar,
            }
            for window in report.windows
        ],
        "standard_metrics": report.standard_metrics,
        "benchmark_metrics": report.benchmark_metrics,
        "summary": report.summary,
        "multiple_testing": report.multiple_testing,
        "safety": report.safety,
    }


def metrics_to_dict(metrics: BacktestMetrics) -> dict[str, float]:
    return {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "positive_period_win_rate": metrics.positive_period_win_rate,
        "annualized_turnover": metrics.annualized_turnover,
        "net_cost": metrics.net_cost,
    }


def _select_lookback(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    symbols: tuple[str, ...],
    *,
    lookbacks: tuple[int, ...],
    start: int,
    end: int,
    config: TsmomConfig,
    cost_model: CostModel,
) -> tuple[int, float]:
    scored: list[tuple[int, float]] = []
    for lookback in lookbacks:
        result = _portfolio_strategy_result(
            bars_by_symbol,
            symbols,
            lookback,
            start=start,
            end=end,
            config=config,
            cost_model=cost_model,
        )
        score = result.metrics.sharpe + result.metrics.calmar
        scored.append((lookback, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return scored[0]


def _portfolio_strategy_result(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    symbols: tuple[str, ...],
    lookback: int,
    *,
    start: int,
    end: int,
    config: TsmomConfig,
    cost_model: CostModel,
) -> SimulationResult:
    per_symbol = [
        simulate_tsmom(
            bars_by_symbol[symbol],
            lookback=lookback,
            start=start,
            end=end,
            config=config,
            cost_model=cost_model,
        )
        for symbol in symbols
    ]
    return _combine_equal_weight(per_symbol, config.annualization_periods)


def _portfolio_buy_hold_result(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    symbols: tuple[str, ...],
    *,
    start: int,
    end: int,
    cost_model: CostModel,
    annualization_periods: int,
) -> SimulationResult:
    per_symbol = [
        simulate_buy_hold(
            bars_by_symbol[symbol],
            start=start,
            end=end,
            cost_model=cost_model,
            annualization_periods=annualization_periods,
        )
        for symbol in symbols
    ]
    return _combine_equal_weight(per_symbol, annualization_periods)


def _combine_equal_weight(
    results: Sequence[SimulationResult], annualization_periods: int
) -> SimulationResult:
    min_len = min((len(result.returns) for result in results), default=0)
    if min_len == 0:
        empty_metrics = metrics_from_returns(
            (),
            annualization_periods=annualization_periods,
            turnover=0.0,
            net_cost=0.0,
        )
        return SimulationResult((), (1.0,), (), (), 0.0, empty_metrics, 0, 0)
    returns = tuple(
        statistics.fmean(result.returns[index] for result in results)
        for index in range(min_len)
    )
    turnover = sum(result.turnover for result in results) / len(results)
    net_cost = sum(sum(result.costs[:min_len]) for result in results) / len(results)
    metrics = metrics_from_returns(
        returns,
        annualization_periods=annualization_periods,
        turnover=turnover,
        net_cost=net_cost,
    )
    return SimulationResult(
        returns=returns,
        equity=equity_curve(returns),
        positions=(),
        costs=(),
        turnover=turnover,
        metrics=metrics,
        first_trade_index=max(result.first_trade_index for result in results),
        last_trade_index=min(result.last_trade_index for result in results),
    )


def _window_starts(
    total_bars: int, train_bars: int, test_bars: int, step_bars: int
) -> list[int]:
    starts: list[int] = []
    start = 0
    while start + train_bars + test_bars <= total_bars:
        starts.append(start)
        start += step_bars
    return starts


def _summary(
    windows: Sequence[WalkForwardWindow],
    excess: Sequence[float],
    strategy_metrics: BacktestMetrics,
    buy_hold_metrics: BacktestMetrics,
) -> dict[str, float | int | str | None]:
    wins = sum(1 for window in windows if window.beats_buy_hold_return)
    sharpe_wins = sum(1 for window in windows if window.beats_buy_hold_sharpe)
    ci = bootstrap_mean_ci(excess)
    return {
        "windows": len(windows),
        "oos_window_win_rate_vs_buy_hold": wins / len(windows) if windows else 0.0,
        "oos_window_sharpe_win_rate_vs_buy_hold": sharpe_wins / len(windows) if windows else 0.0,
        "mean_oos_excess_return": statistics.fmean(excess) if excess else 0.0,
        "median_oos_excess_return": statistics.median(excess) if excess else 0.0,
        "excess_bootstrap_ci_p05": ci["p05"],
        "excess_bootstrap_ci_p50": ci["p50"],
        "excess_bootstrap_ci_p95": ci["p95"],
        "sign_test_p_value": sign_test_p_value(excess),
        "strategy_sharpe": strategy_metrics.sharpe,
        "buy_hold_sharpe": buy_hold_metrics.sharpe,
        "strategy_calmar": strategy_metrics.calmar,
        "buy_hold_calmar": buy_hold_metrics.calmar,
    }


def fdr_report(
    p_values: Sequence[float], *, alpha: float = 0.10
) -> dict[str, float | int | str | None]:
    passed = benjamini_hochberg(p_values, alpha=alpha)
    return {
        "method": "Benjamini-Hochberg",
        "alpha": alpha,
        "tests": len(p_values),
        "discoveries": sum(1 for value in passed if value),
        "min_p_value": min(p_values) if p_values else None,
    }


def _static_oos_p_value(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    symbols: tuple[str, ...],
    lookback: int,
    config: TsmomConfig,
    cost_model: CostModel,
) -> float:
    min_bars = min(len(bars_by_symbol[symbol]) for symbol in symbols)
    starts = _window_starts(min_bars, config.train_bars, config.test_bars, config.step_bars)
    excess = []
    for train_start in starts:
        test_start = train_start + config.train_bars
        test_end = test_start + config.test_bars
        strategy = _portfolio_strategy_result(
            bars_by_symbol,
            symbols,
            lookback,
            start=test_start,
            end=test_end,
            config=config,
            cost_model=cost_model,
        )
        buy_hold = _portfolio_buy_hold_result(
            bars_by_symbol,
            symbols,
            start=test_start,
            end=test_end,
            cost_model=cost_model,
            annualization_periods=config.annualization_periods,
        )
        excess.append(strategy.metrics.total_return - buy_hold.metrics.total_return)
    return sign_test_p_value(excess)


def _verdict(
    summary: dict[str, float | int | str | None],
    multiple_testing: dict[str, float | int | str | None],
) -> tuple[str, str]:
    windows = int(summary["windows"] or 0)
    if windows < 4:
        return "INSUFFICIENT", "fewer than four OOS windows"
    ci_p05 = summary["excess_bootstrap_ci_p05"]
    sign_p = float(summary["sign_test_p_value"] or 1.0)
    win_rate = float(summary["oos_window_win_rate_vs_buy_hold"] or 0.0)
    sharpe_win_rate = float(summary["oos_window_sharpe_win_rate_vs_buy_hold"] or 0.0)
    discoveries = int(multiple_testing["discoveries"] or 0)
    if (
        isinstance(ci_p05, float)
        and ci_p05 > 0
        and sign_p <= 0.10
        and win_rate >= 0.60
        and sharpe_win_rate >= 0.50
        and discoveries > 0
    ):
        return "ROBUST_TREND_EDGE", "OOS excess, sign test, FDR, and risk-adjusted checks passed"
    return "NO_ROBUST_EDGE", "OOS full-cost trend did not clear buy-and-hold robustness gates"


def _insufficient(
    reason: str,
    symbols: tuple[str, ...],
    config: TsmomConfig,
    cost_model: CostModel,
) -> WalkForwardReport:
    return WalkForwardReport(
        status="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=reason,
        symbols=symbols,
        config=config,
        cost_model=cost_model,
        windows=(),
        standard_metrics={},
        benchmark_metrics={},
        summary={"windows": 0},
        multiple_testing={"method": "Benjamini-Hochberg", "tests": 0, "discoveries": 0},
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )
