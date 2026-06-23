from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from aegis.backtest_core import (
    BacktestDiscipline,
    CostModel,
    HypothesisSpec,
    benjamini_hochberg,
    bootstrap_mean_ci,
    equity_curve,
    metrics_from_returns,
    run_backtest,
    sign_test_p_value,
)
from aegis.backtest_core import (
    ReturnMetrics as BacktestMetrics,
)

__all__ = [
    "BacktestMetrics",
    "CostModel",
    "CryptoBar",
    "TsmomConfig",
    "benjamini_hochberg",
    "crypto_tsmom_hypothesis_spec",
    "report_to_dict",
    "run_crypto_tsmom_walk_forward",
    "sign_test_p_value",
    "simulate_buy_hold",
    "simulate_tsmom",
]


@dataclass(frozen=True)
class CryptoBar:
    timestamp: int
    open: float
    close: float


@dataclass(frozen=True)
class TsmomConfig:
    lookbacks: tuple[int, ...] = (30, 60, 90, 120)
    train_bars: int = 730
    test_bars: int = 180
    step_bars: int = 180
    annualization_periods: int = 365
    allow_short: bool = False


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
DEFAULT_COST_MODEL = CostModel(funding_label="N/A for spot long-only")


def run_crypto_tsmom_walk_forward(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    *,
    config: TsmomConfig = DEFAULT_TSMOM_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> WalkForwardReport:
    spec = crypto_tsmom_hypothesis_spec(
        bars_by_symbol,
        config=config,
        cost_model=cost_model,
        runner=lambda: _run_crypto_tsmom_walk_forward_impl(
            bars_by_symbol, config=config, cost_model=cost_model
        ),
    )
    return cast(WalkForwardReport, run_backtest(spec).payload)


def crypto_tsmom_hypothesis_spec(
    bars_by_symbol: dict[str, Sequence[CryptoBar]],
    *,
    config: TsmomConfig = DEFAULT_TSMOM_CONFIG,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    runner: Callable[[], object] | None = None,
) -> HypothesisSpec:
    symbols = tuple(sorted(bars_by_symbol)) or ("<empty>",)
    return HypothesisSpec(
        key="crypto_tsmom_walk_forward",
        hypothesis_type="momentum",
        universe=symbols,
        predeclared_signals=tuple(f"tsmom_{lookback}" for lookback in config.lookbacks),
        params={
            "lookbacks": tuple(config.lookbacks),
            "train_bars": config.train_bars,
            "test_bars": config.test_bars,
            "step_bars": config.step_bars,
            "allow_short": config.allow_short,
        },
        cost_model=cost_model,
        benchmark="buy_and_hold",
        data_source="caller_supplied_crypto_ohlcv_bars",
        trial_count_n=max(1, len(symbols) * len(config.lookbacks)),
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=False,
        ),
        runner=runner,
    )


def _run_crypto_tsmom_walk_forward_impl(
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
        nonpositive_annualized_return=0.0,
    )
    buy_hold_metrics = metrics_from_returns(
        all_buy_hold_returns,
        annualization_periods=config.annualization_periods,
        turnover=(
            sum(window.buy_hold_metrics.annualized_turnover for window in windows)
            / len(windows)
        ),
        net_cost=sum(window.buy_hold_metrics.net_cost for window in windows),
        nonpositive_annualized_return=0.0,
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
        trade_cost = position_change * cost_model.one_way_cost
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
        nonpositive_annualized_return=0.0,
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
        trade_cost = cost_model.one_way_cost if execution_index == start else 0.0
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
        nonpositive_annualized_return=0.0,
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
            nonpositive_annualized_return=0.0,
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
        nonpositive_annualized_return=0.0,
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
        "sign_test_p_value": sign_test_p_value(excess, alternative="two-sided"),
        "strategy_sharpe": strategy_metrics.sharpe,
        "buy_hold_sharpe": buy_hold_metrics.sharpe,
        "strategy_calmar": strategy_metrics.calmar,
        "buy_hold_calmar": buy_hold_metrics.calmar,
    }


def fdr_report(
    p_values: Sequence[float], *, alpha: float = 0.10
) -> dict[str, float | int | str | None]:
    passed = benjamini_hochberg(p_values, alpha=alpha, tie_policy="rank")
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
    return sign_test_p_value(excess, alternative="two-sided")


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
