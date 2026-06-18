from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ComboBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class ComboCostModel:
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    funding_bps_per_period: float = 0.0
    funding_label: str = "N/A for spot long-only; perp funding not used"

    @property
    def one_way_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class ComboSearchConfig:
    train_bars: int = 730
    test_bars: int = 180
    step_bars: int = 180
    locked_oos_fraction: float = 0.30
    annualization_periods: int = 365
    fdr_alpha: float = 0.10
    min_is_folds: int = 3
    top_k_oos: int = 3
    rsi_periods: tuple[int, ...] = (14, 21)
    ma_periods: tuple[int, ...] = (20, 50, 100, 200)
    ma_cross_pairs: tuple[tuple[int, int], ...] = ((20, 50), (50, 100), (50, 200))
    macd_params: tuple[tuple[int, int, int], ...] = ((12, 26, 9),)
    roc_periods: tuple[int, ...] = (20, 60)
    tsmom_periods: tuple[int, ...] = (30, 90)
    atr_periods: tuple[int, ...] = (14, 21)
    bollinger_periods: tuple[int, ...] = (20,)
    realized_vol_periods: tuple[int, ...] = (20, 60)
    volume_z_periods: tuple[int, ...] = (20,)
    obv_periods: tuple[int, ...] = (20,)


@dataclass(frozen=True)
class ComboMetrics:
    annualized_return: float
    total_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    positive_period_win_rate: float
    oos_vs_buy_hold_window_win_rate: float
    annualized_turnover: float
    net_cost: float


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    family: str
    params: tuple[int | float, ...]
    role: str
    evaluate: Callable[..., bool]


@dataclass(frozen=True)
class ComboRule:
    name: str
    operator: str
    indicators: tuple[IndicatorSpec, ...]


@dataclass(frozen=True)
class ComboCandidate:
    symbol: str
    rule: ComboRule

    @property
    def name(self) -> str:
        return f"{self.symbol}::{self.rule.name}"


@dataclass(frozen=True)
class Simulation:
    returns: tuple[float, ...]
    positions: tuple[int, ...]
    costs: tuple[float, ...]
    turnover: float
    first_execution_index: int
    selector_max_index: int
    metrics: ComboMetrics


@dataclass(frozen=True)
class CandidateISResult:
    candidate: ComboCandidate
    fold_excess_returns: tuple[float, ...]
    fold_strategy_returns: tuple[float, ...]
    fold_buy_hold_returns: tuple[float, ...]
    p_value: float
    mean_excess_return: float
    mean_strategy_sharpe: float
    mean_buy_hold_sharpe: float
    selector_max_index: int
    first_oos_execution_index: int


@dataclass(frozen=True)
class ISFoldBenchmark:
    train_end: int
    test_start: int
    test_end: int
    buy_hold: Simulation


@dataclass(frozen=True)
class LockedOOSResult:
    candidate: ComboCandidate
    strategy_metrics: ComboMetrics
    buy_hold_metrics: ComboMetrics
    excess_return: float
    beats_buy_hold_return: bool
    beats_buy_hold_sharpe: bool
    selector_max_index: int
    first_oos_execution_index: int


@dataclass(frozen=True)
class ComboSearchReport:
    status: str
    verdict: str
    reason: str
    symbols: tuple[str, ...]
    config: ComboSearchConfig
    cost_model: ComboCostModel
    indicator_count: int
    rule_count: int
    search_space_n: int
    locked_oos_start: int
    raw_is_survivors: int
    fdr_is_survivors: int
    locked_oos_survivors: int
    selected_for_oos: tuple[str, ...]
    standard_metrics: dict[str, dict[str, float]]
    benchmark_metrics: dict[str, dict[str, float]]
    multiple_testing: dict[str, float | int | str]
    safety: dict[str, bool | str]


DEFAULT_COMBO_CONFIG = ComboSearchConfig()
DEFAULT_COMBO_COST_MODEL = ComboCostModel()


def predeclared_indicators(
    config: ComboSearchConfig = DEFAULT_COMBO_CONFIG,
) -> tuple[IndicatorSpec, ...]:
    indicators: list[IndicatorSpec] = []
    for period in config.rsi_periods:
        indicators.append(
            IndicatorSpec(
                name=f"rsi_{period}_gt_50",
                family="momentum",
                params=(period,),
                role="direction",
                evaluate=lambda bars, i, period=period: _rsi(bars, i, period) > 50.0,
            )
        )
    for period in config.ma_periods:
        indicators.append(
            IndicatorSpec(
                name=f"close_gt_sma_{period}",
                family="trend",
                params=(period,),
                role="direction",
                evaluate=lambda bars, i, period=period: (
                    bars[i].close > _sma([bar.close for bar in bars], i, period)
                ),
            )
        )
    for fast, slow in config.ma_cross_pairs:
        indicators.append(
            IndicatorSpec(
                name=f"sma_{fast}_gt_sma_{slow}",
                family="trend",
                params=(fast, slow),
                role="direction",
                evaluate=lambda bars, i, fast=fast, slow=slow: (
                    _sma([bar.close for bar in bars], i, fast)
                    > _sma([bar.close for bar in bars], i, slow)
                ),
            )
        )
    for fast, slow, signal in config.macd_params:
        indicators.append(
            IndicatorSpec(
                name=f"macd_{fast}_{slow}_{signal}_gt_signal",
                family="trend",
                params=(fast, slow, signal),
                role="direction",
                evaluate=lambda bars, i, fast=fast, slow=slow, signal=signal: (
                    _macd_histogram(bars, i, fast, slow, signal) > 0.0
                ),
            )
        )
    for period in config.roc_periods:
        indicators.append(
            IndicatorSpec(
                name=f"roc_{period}_gt_0",
                family="momentum",
                params=(period,),
                role="direction",
                evaluate=lambda bars, i, period=period: _roc(bars, i, period) > 0.0,
            )
        )
    for period in config.tsmom_periods:
        indicators.append(
            IndicatorSpec(
                name=f"tsmom_{period}_gt_0",
                family="momentum",
                params=(period,),
                role="direction",
                evaluate=lambda bars, i, period=period: (
                    bars[i].close > bars[i - period].close if i >= period else False
                ),
            )
        )
    for period in config.atr_periods:
        indicators.append(
            IndicatorSpec(
                name=f"atr_{period}_below_median",
                family="volatility",
                params=(period,),
                role="filter",
                evaluate=lambda bars, i, period=period: _atr_below_rolling_median(bars, i, period),
            )
        )
    for period in config.bollinger_periods:
        indicators.append(
            IndicatorSpec(
                name=f"bollinger_width_{period}_below_median",
                family="volatility",
                params=(period,),
                role="filter",
                evaluate=lambda bars, i, period=period: _bollinger_below_rolling_median(
                    bars, i, period
                ),
            )
        )
    for period in config.realized_vol_periods:
        indicators.append(
            IndicatorSpec(
                name=f"realized_vol_{period}_below_median",
                family="volatility",
                params=(period,),
                role="filter",
                evaluate=lambda bars, i, period=period: _realized_vol_below_rolling_median(
                    bars, i, period
                ),
            )
        )
    for period in config.volume_z_periods:
        indicators.append(
            IndicatorSpec(
                name=f"volume_z_{period}_gt_0",
                family="volume",
                params=(period,),
                role="filter",
                evaluate=lambda bars, i, period=period: (
                    _zscore([bar.volume for bar in bars], i, period) > 0.0
                ),
            )
        )
    for period in config.obv_periods:
        indicators.append(
            IndicatorSpec(
                name=f"obv_slope_{period}_gt_0",
                family="volume",
                params=(period,),
                role="filter",
                evaluate=lambda bars, i, period=period: _obv_slope(bars, i, period) > 0.0,
            )
        )
    return tuple(indicators)


def predeclared_rules(config: ComboSearchConfig = DEFAULT_COMBO_CONFIG) -> tuple[ComboRule, ...]:
    indicators = predeclared_indicators(config)
    direction = tuple(indicator for indicator in indicators if indicator.role == "direction")
    filters = tuple(indicator for indicator in indicators if indicator.role == "filter")
    rules: list[ComboRule] = []
    for size in (2, 3):
        for combo in itertools.combinations(indicators, size):
            if len({indicator.family for indicator in combo}) < size:
                continue
            rules.append(ComboRule(_rule_name("and", combo), "AND", combo))
            rules.append(ComboRule(_rule_name("or", combo), "OR", combo))
            rules.append(ComboRule(_rule_name("vote", combo), "VOTE", combo))
    for primary in direction:
        for regime_filter in filters:
            rules.append(
                ComboRule(
                    _rule_name("regime", (primary, regime_filter)),
                    "REGIME",
                    (primary, regime_filter),
                )
            )
    return tuple(rules)


def run_combo_indicator_search(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: ComboSearchConfig = DEFAULT_COMBO_CONFIG,
    cost_model: ComboCostModel = DEFAULT_COMBO_COST_MODEL,
) -> ComboSearchReport:
    if not bars_by_symbol:
        return _insufficient("no bars supplied", (), config, cost_model)
    symbols = tuple(sorted(bars_by_symbol))
    min_bars = min(len(bars_by_symbol[symbol]) for symbol in symbols)
    locked_oos_start = int(min_bars * (1.0 - config.locked_oos_fraction))
    if locked_oos_start < config.train_bars + config.test_bars:
        return _insufficient(
            "not enough in-sample bars before locked OOS", symbols, config, cost_model
        )

    indicators = predeclared_indicators(config)
    rules = predeclared_rules(config)
    candidates = tuple(
        ComboCandidate(symbol=symbol, rule=rule) for symbol in symbols for rule in rules
    )
    signal_cache_by_symbol = {
        symbol: _build_signal_cache(bars_by_symbol[symbol], indicators) for symbol in symbols
    }
    is_benchmarks_by_symbol = {
        symbol: _is_fold_benchmarks(
            bars_by_symbol[symbol],
            locked_oos_start=locked_oos_start,
            config=config,
            cost_model=cost_model,
        )
        for symbol in symbols
    }
    search_space_n = len(candidates)
    is_results = tuple(
        result
        for result in (
            _evaluate_candidate_is(
                candidate,
                bars_by_symbol[candidate.symbol],
                signal_cache_by_symbol[candidate.symbol],
                is_benchmarks_by_symbol[candidate.symbol],
                locked_oos_start=locked_oos_start,
                config=config,
                cost_model=cost_model,
            )
            for candidate in candidates
        )
        if result is not None
    )
    if len(is_results) < search_space_n:
        return _insufficient(
            "one or more candidates lacked enough walk-forward folds",
            symbols,
            config,
            cost_model,
            indicator_count=len(indicators),
            rule_count=len(rules),
            search_space_n=search_space_n,
            locked_oos_start=locked_oos_start,
        )

    raw_is_survivors = sum(
        1
        for result in is_results
        if result.mean_excess_return > 0
        and result.mean_strategy_sharpe > result.mean_buy_hold_sharpe
    )
    discoveries = benjamini_hochberg(
        [result.p_value for result in is_results], alpha=config.fdr_alpha
    )
    fdr_results = tuple(
        result for result, keep in zip(is_results, discoveries, strict=True) if keep
    )
    ranked = sorted(
        fdr_results or is_results,
        key=lambda result: (
            result in fdr_results,
            result.mean_excess_return,
            result.mean_strategy_sharpe - result.mean_buy_hold_sharpe,
        ),
        reverse=True,
    )
    selected = tuple(ranked[: config.top_k_oos])
    oos_results = tuple(
        _evaluate_locked_oos(
            result.candidate,
            bars_by_symbol[result.candidate.symbol],
            signal_cache_by_symbol[result.candidate.symbol],
            locked_oos_start=locked_oos_start,
            config=config,
            cost_model=cost_model,
        )
        for result in selected
    )
    locked_oos_survivors = sum(
        1 for result in oos_results if result.beats_buy_hold_return and result.beats_buy_hold_sharpe
    )
    standard_metrics = {
        result.candidate.name: metrics_to_dict(result.strategy_metrics) for result in oos_results
    }
    benchmark_metrics = {
        result.candidate.name: metrics_to_dict(result.buy_hold_metrics) for result in oos_results
    }
    equal_weight_benchmark = _equal_weight_buy_hold_metrics(
        bars_by_symbol,
        start=locked_oos_start,
        end=min_bars - 1,
        config=config,
        cost_model=cost_model,
    )
    benchmark_metrics["equal_weight_buy_hold"] = metrics_to_dict(equal_weight_benchmark)
    verdict, reason = _verdict(
        fdr_is_survivors=len(fdr_results),
        locked_oos_survivors=locked_oos_survivors,
        selected_count=len(selected),
    )
    return ComboSearchReport(
        status="OK",
        verdict=verdict,
        reason=reason,
        symbols=symbols,
        config=config,
        cost_model=cost_model,
        indicator_count=len(indicators),
        rule_count=len(rules),
        search_space_n=search_space_n,
        locked_oos_start=locked_oos_start,
        raw_is_survivors=raw_is_survivors,
        fdr_is_survivors=len(fdr_results),
        locked_oos_survivors=locked_oos_survivors,
        selected_for_oos=tuple(result.candidate.name for result in selected),
        standard_metrics=standard_metrics,
        benchmark_metrics=benchmark_metrics,
        multiple_testing={
            "method": "Benjamini-Hochberg FDR over all predeclared symbol-rule trials",
            "alpha": config.fdr_alpha,
            "trial_count_n": search_space_n,
            "raw_is_survivors": raw_is_survivors,
            "fdr_is_survivors": len(fdr_results),
            "locked_oos_survivors": locked_oos_survivors,
            "min_p_value": min(result.p_value for result in is_results),
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def simulate_rule(
    bars: Sequence[ComboBar],
    rule: ComboRule,
    *,
    start: int,
    end: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> Simulation:
    returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    prev_position = 0
    turnover = 0.0
    start = max(start, _warmup(rule) + 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        position = 1 if _rule_signal(rule, bars, decision_index) else 0
        trade_size = abs(position - prev_position)
        trade_cost = trade_size * cost_model.one_way_cost
        funding_cost = abs(position) * (cost_model.funding_bps_per_period / 10_000.0)
        gross_return = position * (
            bars[execution_index + 1].open / bars[execution_index].open - 1.0
        )
        returns.append(gross_return - trade_cost - funding_cost)
        positions.append(position)
        costs.append(trade_cost + funding_cost)
        turnover += trade_size
        prev_position = position
    if positions and positions[-1] != 0:
        turnover += abs(positions[-1])
        costs[-1] += abs(positions[-1]) * cost_model.one_way_cost
        returns[-1] -= abs(positions[-1]) * cost_model.one_way_cost
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return Simulation(
        returns=tuple(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        turnover=turnover,
        first_execution_index=start,
        selector_max_index=start - 1,
        metrics=metrics,
    )


def simulate_rule_with_cache(
    bars: Sequence[ComboBar],
    rule: ComboRule,
    signal_cache: dict[str, tuple[bool, ...]],
    *,
    start: int,
    end: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> Simulation:
    returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    prev_position = 0
    turnover = 0.0
    start = max(start, _warmup(rule) + 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        position = 1 if _rule_signal_from_cache(rule, signal_cache, decision_index) else 0
        trade_size = abs(position - prev_position)
        trade_cost = trade_size * cost_model.one_way_cost
        funding_cost = abs(position) * (cost_model.funding_bps_per_period / 10_000.0)
        gross_return = position * (
            bars[execution_index + 1].open / bars[execution_index].open - 1.0
        )
        returns.append(gross_return - trade_cost - funding_cost)
        positions.append(position)
        costs.append(trade_cost + funding_cost)
        turnover += trade_size
        prev_position = position
    if positions and positions[-1] != 0:
        turnover += abs(positions[-1])
        costs[-1] += abs(positions[-1]) * cost_model.one_way_cost
        returns[-1] -= abs(positions[-1]) * cost_model.one_way_cost
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return Simulation(
        returns=tuple(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        turnover=turnover,
        first_execution_index=start,
        selector_max_index=start - 1,
        metrics=metrics,
    )


def buy_hold_simulation(
    bars: Sequence[ComboBar],
    *,
    start: int,
    end: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> Simulation:
    returns: list[float] = []
    costs: list[float] = []
    start = max(start, 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        cost = cost_model.one_way_cost if execution_index == start else 0.0
        if execution_index == end - 1:
            cost += cost_model.one_way_cost
        returns.append(bars[execution_index + 1].open / bars[execution_index].open - 1.0 - cost)
        costs.append(cost)
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=2.0 if returns else 0.0,
        net_cost=sum(costs),
    )
    return Simulation(
        returns=tuple(returns),
        positions=tuple(1 for _ in returns),
        costs=tuple(costs),
        turnover=2.0 if returns else 0.0,
        first_execution_index=start,
        selector_max_index=start - 1,
        metrics=metrics,
    )


def metrics_from_returns(
    returns: Sequence[float],
    *,
    annualization_periods: int,
    turnover: float,
    net_cost: float,
    oos_vs_buy_hold_window_win_rate: float = 0.0,
) -> ComboMetrics:
    values = tuple(float(value) for value in returns)
    if not values:
        return ComboMetrics(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, oos_vs_buy_hold_window_win_rate, 0.0, net_cost
        )
    equity = _equity(values)
    total_return = equity[-1] - 1.0
    years = max(len(values) / annualization_periods, 1 / annualization_periods)
    annualized_return = equity[-1] ** (1 / years) - 1.0 if equity[-1] > 0 else -1.0
    max_dd = _max_drawdown(equity)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    sharpe = (mean / stdev) * math.sqrt(annualization_periods) if stdev > 0 else 0.0
    downside = tuple(value for value in values if value < 0)
    downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    sortino = (mean / downside_dev) * math.sqrt(annualization_periods) if downside_dev > 0 else 0.0
    calmar = annualized_return / abs(max_dd) if max_dd < 0 else 0.0
    return ComboMetrics(
        annualized_return=annualized_return,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        positive_period_win_rate=sum(1 for value in values if value > 0) / len(values),
        oos_vs_buy_hold_window_win_rate=oos_vs_buy_hold_window_win_rate,
        annualized_turnover=turnover / years,
        net_cost=net_cost,
    )


def metrics_to_dict(metrics: ComboMetrics) -> dict[str, float]:
    return {
        "annualized_return": metrics.annualized_return,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "positive_period_win_rate": metrics.positive_period_win_rate,
        "oos_vs_buy_hold_window_win_rate": metrics.oos_vs_buy_hold_window_win_rate,
        "annualized_turnover": metrics.annualized_turnover,
        "net_cost": metrics.net_cost,
    }


def benjamini_hochberg(p_values: Sequence[float], *, alpha: float) -> list[bool]:
    m = len(p_values)
    if m == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    threshold_rank = -1
    for rank, (_, p_value) in enumerate(ordered, start=1):
        if p_value <= alpha * rank / m:
            threshold_rank = rank
    keep = [False] * m
    if threshold_rank >= 0:
        cutoff = ordered[threshold_rank - 1][1]
        keep = [p_value <= cutoff for p_value in p_values]
    return keep


def sign_test_p_value(excess_returns: Sequence[float]) -> float:
    non_zero = [value for value in excess_returns if value != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0)
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(wins, n + 1))
    return min(1.0, tail)


def report_to_dict(report: ComboSearchReport) -> dict[str, object]:
    return {
        "status": report.status,
        "verdict": report.verdict,
        "reason": report.reason,
        "symbols": list(report.symbols),
        "indicator_count": report.indicator_count,
        "rule_count": report.rule_count,
        "search_space_n": report.search_space_n,
        "locked_oos_start": report.locked_oos_start,
        "raw_is_survivors": report.raw_is_survivors,
        "fdr_is_survivors": report.fdr_is_survivors,
        "locked_oos_survivors": report.locked_oos_survivors,
        "selected_for_oos": list(report.selected_for_oos),
        "standard_metrics": report.standard_metrics,
        "benchmark_metrics": report.benchmark_metrics,
        "multiple_testing": report.multiple_testing,
        "safety": report.safety,
    }


def _evaluate_candidate_is(
    candidate: ComboCandidate,
    bars: Sequence[ComboBar],
    signal_cache: dict[str, tuple[bool, ...]],
    is_benchmarks: Sequence[ISFoldBenchmark],
    *,
    locked_oos_start: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> CandidateISResult | None:
    fold_excess: list[float] = []
    fold_strategy: list[float] = []
    fold_buy_hold: list[float] = []
    strategy_sharpes: list[float] = []
    buy_hold_sharpes: list[float] = []
    selector_max = -1
    first_oos_execution = locked_oos_start
    for benchmark in is_benchmarks:
        selector_max = max(selector_max, benchmark.train_end - 1)
        strategy = simulate_rule_with_cache(
            bars,
            candidate.rule,
            signal_cache,
            start=benchmark.test_start,
            end=benchmark.test_end,
            config=config,
            cost_model=cost_model,
        )
        buy_hold = benchmark.buy_hold
        fold_excess.append(strategy.metrics.total_return - buy_hold.metrics.total_return)
        fold_strategy.append(strategy.metrics.total_return)
        fold_buy_hold.append(buy_hold.metrics.total_return)
        strategy_sharpes.append(strategy.metrics.sharpe)
        buy_hold_sharpes.append(buy_hold.metrics.sharpe)
    if len(fold_excess) < config.min_is_folds:
        return None
    return CandidateISResult(
        candidate=candidate,
        fold_excess_returns=tuple(fold_excess),
        fold_strategy_returns=tuple(fold_strategy),
        fold_buy_hold_returns=tuple(fold_buy_hold),
        p_value=sign_test_p_value(fold_excess),
        mean_excess_return=statistics.fmean(fold_excess),
        mean_strategy_sharpe=statistics.fmean(strategy_sharpes),
        mean_buy_hold_sharpe=statistics.fmean(buy_hold_sharpes),
        selector_max_index=selector_max,
        first_oos_execution_index=first_oos_execution,
    )


def _is_fold_benchmarks(
    bars: Sequence[ComboBar],
    *,
    locked_oos_start: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> tuple[ISFoldBenchmark, ...]:
    benchmarks: list[ISFoldBenchmark] = []
    starts = range(0, locked_oos_start - config.train_bars - config.test_bars + 1, config.step_bars)
    for train_start in starts:
        train_end = train_start + config.train_bars
        test_start = train_end
        test_end = test_start + config.test_bars
        benchmarks.append(
            ISFoldBenchmark(
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                buy_hold=buy_hold_simulation(
                    bars,
                    start=test_start,
                    end=test_end,
                    config=config,
                    cost_model=cost_model,
                ),
            )
        )
    return tuple(benchmarks)


def _evaluate_locked_oos(
    candidate: ComboCandidate,
    bars: Sequence[ComboBar],
    signal_cache: dict[str, tuple[bool, ...]],
    *,
    locked_oos_start: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> LockedOOSResult:
    strategy = simulate_rule_with_cache(
        bars,
        candidate.rule,
        signal_cache,
        start=locked_oos_start,
        end=len(bars) - 1,
        config=config,
        cost_model=cost_model,
    )
    buy_hold = buy_hold_simulation(
        bars,
        start=locked_oos_start,
        end=len(bars) - 1,
        config=config,
        cost_model=cost_model,
    )
    beats_return = strategy.metrics.total_return > buy_hold.metrics.total_return
    beats_sharpe = strategy.metrics.sharpe > buy_hold.metrics.sharpe
    return LockedOOSResult(
        candidate=candidate,
        strategy_metrics=replace(
            strategy.metrics,
            oos_vs_buy_hold_window_win_rate=1.0 if beats_return else 0.0,
        ),
        buy_hold_metrics=buy_hold.metrics,
        excess_return=strategy.metrics.total_return - buy_hold.metrics.total_return,
        beats_buy_hold_return=beats_return,
        beats_buy_hold_sharpe=beats_sharpe,
        selector_max_index=locked_oos_start - 1,
        first_oos_execution_index=strategy.first_execution_index,
    )


def _equal_weight_buy_hold_metrics(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    start: int,
    end: int,
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
) -> ComboMetrics:
    returns_by_symbol = [
        buy_hold_simulation(
            bars, start=start, end=end, config=config, cost_model=cost_model
        ).returns
        for bars in bars_by_symbol.values()
    ]
    min_len = min((len(values) for values in returns_by_symbol), default=0)
    returns = [
        statistics.fmean(values[index] for values in returns_by_symbol) for index in range(min_len)
    ]
    return metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=2.0 * len(returns_by_symbol),
        net_cost=2.0 * cost_model.one_way_cost * len(returns_by_symbol),
    )


def _rule_signal(rule: ComboRule, bars: Sequence[ComboBar], decision_index: int) -> bool:
    values = [indicator.evaluate(bars, decision_index) for indicator in rule.indicators]
    if rule.operator == "AND":
        return all(values)
    if rule.operator == "OR":
        return any(values)
    if rule.operator == "VOTE":
        return sum(1 for value in values if value) >= math.ceil(len(values) / 2)
    if rule.operator == "REGIME":
        return all(values)
    raise ValueError(f"unknown operator {rule.operator}")


def _rule_signal_from_cache(
    rule: ComboRule,
    signal_cache: dict[str, tuple[bool, ...]],
    decision_index: int,
) -> bool:
    values = [signal_cache[indicator.name][decision_index] for indicator in rule.indicators]
    if rule.operator == "AND":
        return all(values)
    if rule.operator == "OR":
        return any(values)
    if rule.operator == "VOTE":
        return sum(1 for value in values if value) >= math.ceil(len(values) / 2)
    if rule.operator == "REGIME":
        return all(values)
    raise ValueError(f"unknown operator {rule.operator}")


def _build_signal_cache(
    bars: Sequence[ComboBar],
    indicators: Sequence[IndicatorSpec],
) -> dict[str, tuple[bool, ...]]:
    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    volumes = [bar.volume for bar in bars]
    cache: dict[str, tuple[bool, ...]] = {}
    for indicator in indicators:
        period = int(indicator.params[0])
        if indicator.name.startswith("rsi_"):
            rsi_values = _rolling_rsi(closes, period)
            cache[indicator.name] = tuple(value > 50.0 for value in rsi_values)
        elif indicator.name.startswith("close_gt_sma_"):
            mean_values = _rolling_mean(closes, period)
            cache[indicator.name] = tuple(
                _optional_close_gt(closes[index], mean_values[index])
                for index in range(len(closes))
            )
        elif indicator.name.startswith("sma_"):
            fast, slow = (int(value) for value in indicator.params)
            fast_values = _rolling_mean(closes, fast)
            slow_values = _rolling_mean(closes, slow)
            cache[indicator.name] = tuple(
                _optional_gt(fast_values[index], slow_values[index]) for index in range(len(closes))
            )
        elif indicator.name.startswith("macd_"):
            fast, slow, signal = (int(value) for value in indicator.params)
            fast_ema = _ema_series(closes, fast)
            slow_ema = _ema_series(closes, slow)
            macd = [fast_ema[index] - slow_ema[index] for index in range(len(closes))]
            signal_line = _ema_series(macd, signal)
            cache[indicator.name] = tuple(
                macd[index] > signal_line[index] if index >= slow + signal else False
                for index in range(len(closes))
            )
        elif indicator.name.startswith("roc_"):
            cache[indicator.name] = tuple(
                closes[index] / closes[index - period] - 1.0 > 0.0
                if index >= period and closes[index - period] != 0
                else False
                for index in range(len(closes))
            )
        elif indicator.name.startswith("tsmom_"):
            cache[indicator.name] = tuple(
                closes[index] > closes[index - period] if index >= period else False
                for index in range(len(closes))
            )
        elif indicator.name.startswith("atr_"):
            true_ranges = _true_range_series(highs, lows, closes)
            atr_values = _rolling_mean(true_ranges, period)
            cache[indicator.name] = _below_rolling_median(atr_values, period * 3)
        elif indicator.name.startswith("bollinger_width_"):
            widths = _rolling_bollinger_width(closes, period)
            cache[indicator.name] = _below_rolling_median(widths, period * 3)
        elif indicator.name.startswith("realized_vol_"):
            vols = _rolling_realized_vol(closes, period)
            cache[indicator.name] = _below_rolling_median(vols, period * 3)
        elif indicator.name.startswith("volume_z_"):
            zscores = _rolling_zscore(volumes, period)
            cache[indicator.name] = tuple(value is not None and value > 0.0 for value in zscores)
        elif indicator.name.startswith("obv_slope_"):
            obv = _obv_series(closes, volumes)
            cache[indicator.name] = tuple(
                obv[index] - obv[index - period] > 0.0 if index >= period else False
                for index in range(len(closes))
            )
        else:
            cache[indicator.name] = tuple(
                indicator.evaluate(bars, index) for index in range(len(bars))
            )
    return cache


def _warmup(rule: ComboRule) -> int:
    return max(int(max(indicator.params or (1,))) for indicator in rule.indicators)


def _optional_gt(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _optional_close_gt(close: float, threshold: float | None) -> bool:
    return threshold is not None and close > threshold


def _verdict(
    *, fdr_is_survivors: int, locked_oos_survivors: int, selected_count: int
) -> tuple[str, str]:
    if selected_count == 0:
        return "NO_ROBUST_EDGE", "no combination survived IS multiple-testing correction"
    if fdr_is_survivors > 0 and locked_oos_survivors > 0:
        return (
            "INSUFFICIENT",
            "one or more combinations survived the gate; independent OOS confirmation is required",
        )
    return "NO_ROBUST_EDGE", "no combination survived both FDR-corrected IS search and locked OOS"


def _insufficient(
    reason: str,
    symbols: tuple[str, ...],
    config: ComboSearchConfig,
    cost_model: ComboCostModel,
    *,
    indicator_count: int = 0,
    rule_count: int = 0,
    search_space_n: int = 0,
    locked_oos_start: int = 0,
) -> ComboSearchReport:
    return ComboSearchReport(
        status="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=reason,
        symbols=symbols,
        config=config,
        cost_model=cost_model,
        indicator_count=indicator_count,
        rule_count=rule_count,
        search_space_n=search_space_n,
        locked_oos_start=locked_oos_start,
        raw_is_survivors=0,
        fdr_is_survivors=0,
        locked_oos_survivors=0,
        selected_for_oos=(),
        standard_metrics={},
        benchmark_metrics={},
        multiple_testing={"method": "Benjamini-Hochberg FDR", "trial_count_n": search_space_n},
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def _rule_name(operator: str, indicators: Sequence[IndicatorSpec]) -> str:
    return operator + "__" + "__".join(indicator.name for indicator in indicators)


def _sma(values: Sequence[float], index: int, period: int) -> float:
    if index + 1 < period:
        return float("inf")
    return statistics.fmean(values[index - period + 1 : index + 1])


def _rolling_mean(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= period:
            total -= values[index - period]
        if index + 1 >= period:
            result[index] = total / period
    return result


def _rolling_rsi(closes: Sequence[float], period: int) -> list[float]:
    result = [50.0] * len(closes)
    gains: list[float] = [0.0] * len(closes)
    losses: list[float] = [0.0] * len(closes)
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        gains[index] = max(change, 0.0)
        losses[index] = abs(min(change, 0.0))
    gain_mean = _rolling_mean(gains, period)
    loss_mean = _rolling_mean(losses, period)
    for index in range(period, len(closes)):
        avg_gain = gain_mean[index] or 0.0
        avg_loss = loss_mean[index] or 0.0
        if avg_loss == 0:
            result[index] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[index] = 100.0 - (100.0 / (1.0 + rs))
    return result


def _rsi(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for current in range(index - period + 1, index + 1):
        change = bars[current].close - bars[current - 1].close
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = statistics.fmean(gains)
    avg_loss = statistics.fmean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _roc(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period or bars[index - period].close == 0:
        return 0.0
    return bars[index].close / bars[index - period].close - 1.0


def _ema(values: Sequence[float], index: int, period: int) -> float:
    if index < 0:
        return 0.0
    alpha = 2.0 / (period + 1)
    start = max(0, index - period * 4)
    ema = values[start]
    for value in values[start + 1 : index + 1]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def _macd_histogram(
    bars: Sequence[ComboBar], index: int, fast: int, slow: int, signal: int
) -> float:
    closes = [bar.close for bar in bars]
    if index < slow + signal:
        return 0.0
    macd_values = [
        _ema(closes, current, fast) - _ema(closes, current, slow)
        for current in range(max(0, index - signal * 4), index + 1)
    ]
    signal_line = _ema(macd_values, len(macd_values) - 1, signal)
    return macd_values[-1] - signal_line


def _true_range(bars: Sequence[ComboBar], index: int) -> float:
    if index == 0:
        return bars[index].high - bars[index].low
    return max(
        bars[index].high - bars[index].low,
        abs(bars[index].high - bars[index - 1].close),
        abs(bars[index].low - bars[index - 1].close),
    )


def _true_range_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    values: list[float] = []
    for index in range(len(closes)):
        if index == 0:
            values.append(highs[index] - lows[index])
        else:
            values.append(
                max(
                    highs[index] - lows[index],
                    abs(highs[index] - closes[index - 1]),
                    abs(lows[index] - closes[index - 1]),
                )
            )
    return values


def _atr(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index + 1 < period:
        return float("inf")
    return statistics.fmean(
        _true_range(bars, current) for current in range(index - period + 1, index + 1)
    )


def _atr_below_rolling_median(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    median_period = period * 3
    if index + 1 < period + median_period:
        return False
    current_atr = _atr(bars, index, period)
    history = [
        _atr(bars, current, period) for current in range(index - median_period + 1, index + 1)
    ]
    return current_atr <= statistics.median(history)


def _bollinger_width(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index + 1 < period:
        return float("inf")
    closes = [bar.close for bar in bars[index - period + 1 : index + 1]]
    mean = statistics.fmean(closes)
    if mean == 0:
        return float("inf")
    return (4.0 * statistics.pstdev(closes)) / mean


def _bollinger_below_rolling_median(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    median_period = period * 3
    if index + 1 < period + median_period:
        return False
    current_width = _bollinger_width(bars, index, period)
    history = [
        _bollinger_width(bars, current, period)
        for current in range(index - median_period + 1, index + 1)
    ]
    return current_width <= statistics.median(history)


def _realized_vol(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period:
        return float("inf")
    returns = [
        bars[current].close / bars[current - 1].close - 1.0
        for current in range(index - period + 1, index + 1)
    ]
    return statistics.pstdev(returns) if len(returns) > 1 else 0.0


def _realized_vol_below_rolling_median(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    median_period = period * 3
    if index + 1 < period + median_period:
        return False
    current_vol = _realized_vol(bars, index, period)
    history = [
        _realized_vol(bars, current, period)
        for current in range(index - median_period + 1, index + 1)
    ]
    return current_vol <= statistics.median(history)


def _rolling_bollinger_width(closes: Sequence[float], period: int) -> list[float | None]:
    values: list[float | None] = [None] * len(closes)
    for index in range(period - 1, len(closes)):
        window = closes[index - period + 1 : index + 1]
        mean = statistics.fmean(window)
        values[index] = (4.0 * statistics.pstdev(window)) / mean if mean != 0 else None
    return values


def _rolling_realized_vol(closes: Sequence[float], period: int) -> list[float | None]:
    returns = [0.0]
    returns.extend(
        closes[index] / closes[index - 1] - 1.0 if closes[index - 1] != 0 else 0.0
        for index in range(1, len(closes))
    )
    values: list[float | None] = [None] * len(closes)
    for index in range(period, len(closes)):
        window = returns[index - period + 1 : index + 1]
        values[index] = statistics.pstdev(window) if len(window) > 1 else 0.0
    return values


def _rolling_zscore(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        stdev = statistics.pstdev(window)
        result[index] = (values[index] - statistics.fmean(window)) / stdev if stdev else 0.0
    return result


def _below_rolling_median(values: Sequence[float | None], median_period: int) -> tuple[bool, ...]:
    result: list[bool] = []
    for index, value in enumerate(values):
        if value is None or index + 1 < median_period:
            result.append(False)
            continue
        history = [
            historical
            for historical in values[index - median_period + 1 : index + 1]
            if historical is not None
        ]
        result.append(bool(history) and value <= statistics.median(history))
    return tuple(result)


def _obv_series(closes: Sequence[float], volumes: Sequence[float]) -> list[float]:
    values = [0.0] * len(closes)
    for index in range(1, len(closes)):
        if closes[index] > closes[index - 1]:
            values[index] = values[index - 1] + volumes[index]
        elif closes[index] < closes[index - 1]:
            values[index] = values[index - 1] - volumes[index]
        else:
            values[index] = values[index - 1]
    return values


def _zscore(values: Sequence[float], index: int, period: int) -> float:
    if index + 1 < period:
        return 0.0
    window = values[index - period + 1 : index + 1]
    stdev = statistics.pstdev(window)
    if stdev == 0:
        return 0.0
    return (values[index] - statistics.fmean(window)) / stdev


def _obv_slope(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period:
        return 0.0
    obv = 0.0
    previous = 0.0
    for current in range(index - period + 1, index + 1):
        if bars[current].close > bars[current - 1].close:
            obv += bars[current].volume
        elif bars[current].close < bars[current - 1].close:
            obv -= bars[current].volume
        if current == index - period + 1:
            previous = obv
    return obv - previous


def _equity(returns: Sequence[float]) -> tuple[float, ...]:
    equity = 1.0
    values: list[float] = []
    for value in returns:
        equity *= 1.0 + value
        values.append(equity)
    return tuple(values)


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = 1.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd
