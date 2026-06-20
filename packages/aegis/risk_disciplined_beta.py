from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from aegis.backtest_core import (
    BacktestDiscipline,
    HypothesisSpec,
    benjamini_hochberg,
    risk_return_metrics,
    run_backtest,
)
from aegis.backtest_core import (
    paired_block_bootstrap_risk_difference_test as _paired_block_bootstrap_risk_difference_test,
)
from aegis.combo_indicator_search import ComboBar, ComboCostModel

__all__ = [
    "RiskBetaConfig",
    "RiskCandidate",
    "RiskMetrics",
    "_paired_block_bootstrap_risk_difference_test",
    "predeclared_risk_candidates",
    "report_to_dict",
    "risk_beta_hypothesis_spec",
    "risk_metrics",
    "run_risk_disciplined_beta",
]


@dataclass(frozen=True)
class RiskBetaConfig:
    train_bars: int = 730
    test_bars: int = 180
    step_bars: int = 180
    locked_oos_fraction: float = 0.30
    annualization_periods: int = 365
    fdr_alpha: float = 0.10
    min_is_folds: int = 3
    drawdown_reduction_threshold: float = 0.20
    target_vol_tolerance: float = 0.35
    min_oos_fold_pass_rate: float = 2 / 3
    oos_folds: int = 3
    risk_diff_bootstrap_samples: int = 400
    risk_diff_bootstrap_block_bars: int = 30
    risk_diff_ci_alpha: float = 0.05
    risk_diff_random_seed: int = 47


@dataclass(frozen=True)
class RiskCandidate:
    key: str
    method: str
    thesis: str
    symbols: tuple[str, ...]
    target_vol: float
    lookback: int
    max_exposure: float
    rebalance_days: int
    derisk_below_ma200: bool = False


@dataclass(frozen=True)
class RiskMetrics:
    annualized_return: float
    max_drawdown: float
    calmar: float
    sortino: float
    sharpe: float
    realized_volatility: float
    target_volatility: float
    annualized_turnover: float
    net_cost: float
    worst_month: float
    ulcer_index: float


@dataclass(frozen=True)
class RiskSimulation:
    returns: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    costs: tuple[float, ...]
    turnover: float
    metrics: RiskMetrics
    first_execution_index: int


@dataclass(frozen=True)
class RiskScore:
    candidate: RiskCandidate
    fold_scores: tuple[float, ...]
    p_value: float
    selector_max_index: int
    first_oos_execution_index: int


@dataclass(frozen=True)
class RiskResult:
    candidate: RiskCandidate
    metrics: RiskMetrics
    primary_benchmark: RiskMetrics
    equal_weight_benchmark: RiskMetrics
    static_60_40_benchmark: RiskMetrics
    gate_checks: dict[str, bool]
    oos_fold_pass_rate: float
    drawdown_reduction: float
    risk_difference_test: dict[str, float | int | bool | str]
    alpha_significance: dict[str, float | bool | str]
    verdict: str
    reason: str


@dataclass(frozen=True)
class RiskBetaReport:
    status: str
    verdict: str
    reason: str
    candidate_count_n: int
    locked_oos_start: int
    raw_is_survivors: int
    fdr_is_survivors: int
    risk_diff_fdr_survivors: int
    risk_improved_count: int
    insufficient_count: int
    results: dict[str, dict[str, object]]
    benchmarks: dict[str, dict[str, float]]
    multiple_testing: dict[str, float | int | str]
    safety: dict[str, bool | str]


DEFAULT_RISK_CONFIG = RiskBetaConfig()
DEFAULT_RISK_COST_MODEL = ComboCostModel()


def predeclared_risk_candidates() -> tuple[RiskCandidate, ...]:
    candidates: list[RiskCandidate] = []
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        for target in (0.30, 0.40, 0.50):
            for lookback in (20, 30, 60):
                candidates.append(
                    RiskCandidate(
                        key=f"vol_target_{symbol.replace('/', '')}_{int(target * 100)}_{lookback}",
                        method="vol_target",
                        thesis=(
                            "Scale unlevered spot exposure down when lagged realized volatility "
                            "exceeds the predeclared target."
                        ),
                        symbols=(symbol,),
                        target_vol=target,
                        lookback=lookback,
                        max_exposure=1.0,
                        rebalance_days=1,
                    )
                )
                candidates.append(
                    RiskCandidate(
                        key=(
                            f"vol_target_derisk_{symbol.replace('/', '')}_"
                            f"{int(target * 100)}_{lookback}"
                        ),
                        method="vol_target_derisk",
                        thesis=(
                            "Use the same unlevered volatility target, but cut exposure further "
                            "when price is below the 200-day average."
                        ),
                        symbols=(symbol,),
                        target_vol=target,
                        lookback=lookback,
                        max_exposure=1.0,
                        rebalance_days=1,
                        derisk_below_ma200=True,
                    )
                )
    for lookback in (20, 60):
        for rebalance_days in (7, 30):
            for derisk in (False, True):
                suffix = "derisk" if derisk else "plain"
                candidates.append(
                    RiskCandidate(
                        key=f"invvol_basket_{suffix}_{lookback}_{rebalance_days}",
                        method="inverse_vol_basket",
                        thesis=(
                            "Allocate BTC/ETH/SOL by inverse lagged volatility with a cash "
                            "bucket when portfolio volatility exceeds the target."
                        ),
                        symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"),
                        target_vol=0.40,
                        lookback=lookback,
                        max_exposure=1.0,
                        rebalance_days=rebalance_days,
                        derisk_below_ma200=derisk,
                    )
                )
    return tuple(candidates)


def run_risk_disciplined_beta(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: RiskBetaConfig = DEFAULT_RISK_CONFIG,
    cost_model: ComboCostModel = DEFAULT_RISK_COST_MODEL,
) -> RiskBetaReport:
    spec = risk_beta_hypothesis_spec(
        bars_by_symbol,
        config=config,
        cost_model=cost_model,
        runner=lambda: _run_risk_disciplined_beta_impl(
            bars_by_symbol, config=config, cost_model=cost_model
        ),
    )
    return cast(RiskBetaReport, run_backtest(spec).payload)


def risk_beta_hypothesis_spec(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: RiskBetaConfig = DEFAULT_RISK_CONFIG,
    cost_model: ComboCostModel = DEFAULT_RISK_COST_MODEL,
    runner: Callable[[], object] | None = None,
) -> HypothesisSpec:
    symbols = tuple(sorted(bars_by_symbol)) or ("<empty>",)
    candidates = tuple(
        candidate
        for candidate in predeclared_risk_candidates()
        if all(symbol in bars_by_symbol for symbol in candidate.symbols)
    )
    return HypothesisSpec(
        key="risk_disciplined_beta",
        hypothesis_type="risk",
        universe=symbols,
        predeclared_signals=tuple(candidate.key for candidate in candidates),
        params={
            "train_bars": config.train_bars,
            "test_bars": config.test_bars,
            "step_bars": config.step_bars,
            "locked_oos_fraction": config.locked_oos_fraction,
            "drawdown_reduction_threshold": config.drawdown_reduction_threshold,
            "target_vol_tolerance": config.target_vol_tolerance,
        },
        cost_model=cost_model,
        benchmark="primary_buy_hold/equal_weight/static_60_40",
        data_source="caller_supplied_crypto_ohlcv_bars",
        trial_count_n=max(1, len(candidates)),
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


def _run_risk_disciplined_beta_impl(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: RiskBetaConfig = DEFAULT_RISK_CONFIG,
    cost_model: ComboCostModel = DEFAULT_RISK_COST_MODEL,
) -> RiskBetaReport:
    if not bars_by_symbol:
        return _insufficient_report("no bars supplied", config, cost_model)
    min_bars = min(len(bars) for bars in bars_by_symbol.values())
    locked_oos_start = int(min_bars * (1.0 - config.locked_oos_fraction))
    if locked_oos_start < config.train_bars + config.test_bars:
        return _insufficient_report(
            "not enough in-sample bars before locked OOS",
            config,
            cost_model,
            locked_oos_start=locked_oos_start,
        )
    candidates = tuple(
        candidate
        for candidate in predeclared_risk_candidates()
        if all(symbol in bars_by_symbol for symbol in candidate.symbols)
    )
    scores = tuple(
        score
        for score in (
            _evaluate_is(candidate, bars_by_symbol, locked_oos_start, config, cost_model)
            for candidate in candidates
        )
        if score is not None
    )
    if len(scores) < len(candidates):
        return _insufficient_report(
            "one or more candidates lacked enough walk-forward folds",
            config,
            cost_model,
            candidate_count_n=len(candidates),
            locked_oos_start=locked_oos_start,
        )
    alpha_discoveries = benjamini_hochberg(
        [score.p_value for score in scores], alpha=config.fdr_alpha
    )
    alpha_fdr_names = {
        score.candidate.key
        for score, keep in zip(scores, alpha_discoveries, strict=True)
        if keep
    }
    alpha_p_values = {score.candidate.key: score.p_value for score in scores}
    preliminary_results = tuple(
        _locked_oos_result(
            candidate,
            bars_by_symbol,
            locked_oos_start,
            config,
            cost_model,
            alpha_p_value=alpha_p_values[candidate.key],
            alpha_fdr_discovery=candidate.key in alpha_fdr_names,
            risk_diff_fdr_discovery=False,
        )
        for candidate in candidates
    )
    risk_diff_discoveries = benjamini_hochberg(
        [
            float(result.risk_difference_test["p_value"])
            if result.risk_difference_test["valid"]
            else 1.0
            for result in preliminary_results
        ],
        alpha=config.fdr_alpha,
    )
    risk_diff_fdr_names = {
        result.candidate.key
        for result, keep in zip(preliminary_results, risk_diff_discoveries, strict=True)
        if keep
    }
    results = tuple(
        _locked_oos_result(
            candidate,
            bars_by_symbol,
            locked_oos_start,
            config,
            cost_model,
            alpha_p_value=alpha_p_values[candidate.key],
            alpha_fdr_discovery=candidate.key in alpha_fdr_names,
            risk_diff_fdr_discovery=candidate.key in risk_diff_fdr_names,
        )
        for candidate in candidates
    )
    improved_count = sum(1 for result in results if result.verdict == "RISK_IMPROVED")
    insufficient_count = sum(1 for result in results if result.verdict == "INSUFFICIENT")
    raw_is_survivors = sum(1 for score in scores if statistics.fmean(score.fold_scores) > 0)
    verdict, reason = _portfolio_verdict(improved_count, insufficient_count, len(results))
    benchmarks = _report_benchmarks(bars_by_symbol, locked_oos_start, config, cost_model)
    return RiskBetaReport(
        status="OK",
        verdict=verdict,
        reason=reason,
        candidate_count_n=len(candidates),
        locked_oos_start=locked_oos_start,
        raw_is_survivors=raw_is_survivors,
        fdr_is_survivors=len(alpha_fdr_names),
        risk_diff_fdr_survivors=len(risk_diff_fdr_names),
        risk_improved_count=improved_count,
        insufficient_count=insufficient_count,
        results={result.candidate.key: result_to_dict(result) for result in results},
        benchmarks=benchmarks,
        multiple_testing={
            "method": "Benjamini-Hochberg FDR over predeclared risk-difference tests",
            "alpha": config.fdr_alpha,
            "trial_count_n": len(candidates),
            "risk_diff_fdr_survivors": len(risk_diff_fdr_names),
            "risk_diff_test": "paired block bootstrap on locked-OOS strategy-vs-benchmark returns",
            "risk_diff_bootstrap_samples": config.risk_diff_bootstrap_samples,
            "risk_diff_bootstrap_block_bars": config.risk_diff_bootstrap_block_bars,
            "risk_diff_ci_alpha": config.risk_diff_ci_alpha,
            "raw_is_survivors": raw_is_survivors,
            "alpha_fdr_is_survivors_report_only": len(alpha_fdr_names),
            "min_p_value": min(score.p_value for score in scores),
            "alpha_significance_role": "report_only_not_a_risk_gate",
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def simulate_candidate(
    candidate: RiskCandidate,
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    start: int,
    end: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> RiskSimulation:
    symbols = candidate.symbols
    bars = [bars_by_symbol[symbol] for symbol in symbols]
    start = max(start, candidate.lookback + 1, 201 if candidate.derisk_below_ma200 else 1)
    end = min(end, min(len(series) for series in bars) - 1)
    returns: list[float] = []
    weights_by_period: list[tuple[float, ...]] = []
    costs: list[float] = []
    previous_weights = tuple(0.0 for _ in symbols)
    turnover = 0.0
    current_weights = previous_weights
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        if (execution_index - start) % candidate.rebalance_days == 0:
            current_weights = _target_weights(candidate, bars, decision_index, config)
        trade_size = sum(
            abs(weight - previous)
            for weight, previous in zip(current_weights, previous_weights, strict=True)
        )
        trade_cost = trade_size * cost_model.one_way_cost
        period_gross = sum(
            weight * (series[execution_index + 1].open / series[execution_index].open - 1.0)
            for weight, series in zip(current_weights, bars, strict=True)
        )
        returns.append(period_gross - trade_cost)
        weights_by_period.append(current_weights)
        costs.append(trade_cost)
        turnover += trade_size
        previous_weights = current_weights
    if weights_by_period:
        exit_turnover = sum(abs(weight) for weight in previous_weights)
        exit_cost = exit_turnover * cost_model.one_way_cost
        returns[-1] -= exit_cost
        costs[-1] += exit_cost
        turnover += exit_turnover
    metrics = risk_metrics(
        returns,
        annualization_periods=config.annualization_periods,
        target_volatility=candidate.target_vol,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return RiskSimulation(
        returns=tuple(returns),
        weights=tuple(weights_by_period),
        costs=tuple(costs),
        turnover=turnover,
        metrics=metrics,
        first_execution_index=start,
    )


def buy_hold_simulation(
    bars: Sequence[ComboBar],
    *,
    start: int,
    end: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
    allocation: float = 1.0,
    target_volatility: float = 0.0,
) -> RiskSimulation:
    start = max(start, 1)
    end = min(end, len(bars) - 1)
    returns: list[float] = []
    costs: list[float] = []
    for execution_index in range(start, end):
        cost = abs(allocation) * cost_model.one_way_cost if execution_index == start else 0.0
        if execution_index == end - 1:
            cost += abs(allocation) * cost_model.one_way_cost
        returns.append(
            allocation * (bars[execution_index + 1].open / bars[execution_index].open - 1.0) - cost
        )
        costs.append(cost)
    turnover = abs(allocation) * 2 if returns else 0.0
    metrics = risk_metrics(
        returns,
        annualization_periods=config.annualization_periods,
        target_volatility=target_volatility,
        turnover=turnover,
        net_cost=sum(costs),
    )
    weights = tuple((allocation,) for _ in returns)
    return RiskSimulation(
        returns=tuple(returns),
        weights=weights,
        costs=tuple(costs),
        turnover=turnover,
        metrics=metrics,
        first_execution_index=start,
    )


def equal_weight_buy_hold(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    symbols: Sequence[str],
    *,
    start: int,
    end: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> RiskSimulation:
    bars = [bars_by_symbol[symbol] for symbol in symbols]
    start = max(start, 1)
    end = min(end, min(len(series) for series in bars) - 1)
    weight = 1.0 / len(bars)
    returns: list[float] = []
    costs: list[float] = []
    entry_cost = len(bars) * weight * cost_model.one_way_cost
    exit_cost = entry_cost
    for execution_index in range(start, end):
        cost = entry_cost if execution_index == start else 0.0
        if execution_index == end - 1:
            cost += exit_cost
        period_return = statistics.fmean(
            series[execution_index + 1].open / series[execution_index].open - 1.0 for series in bars
        )
        returns.append(period_return - cost)
        costs.append(cost)
    turnover = 2.0 if returns else 0.0
    metrics = risk_metrics(
        returns,
        annualization_periods=config.annualization_periods,
        target_volatility=0.0,
        turnover=turnover,
        net_cost=sum(costs),
    )
    weights = tuple(tuple(weight for _ in bars) for _ in returns)
    return RiskSimulation(
        returns=tuple(returns),
        weights=weights,
        costs=tuple(costs),
        turnover=turnover,
        metrics=metrics,
        first_execution_index=start,
    )


def risk_metrics(
    returns: Sequence[float],
    *,
    annualization_periods: int,
    target_volatility: float,
    turnover: float,
    net_cost: float,
) -> RiskMetrics:
    values = risk_return_metrics(
        returns,
        annualization_periods=annualization_periods,
        target_volatility=target_volatility,
        turnover=turnover,
        net_cost=net_cost,
    )
    return RiskMetrics(
        annualized_return=values["annualized_return"],
        max_drawdown=values["max_drawdown"],
        calmar=values["calmar"],
        sortino=values["sortino"],
        sharpe=values["sharpe"],
        realized_volatility=values["realized_volatility"],
        target_volatility=values["target_volatility"],
        annualized_turnover=values["annualized_turnover"],
        net_cost=values["net_cost"],
        worst_month=values["worst_month"],
        ulcer_index=values["ulcer_index"],
    )


def metrics_to_dict(metrics: RiskMetrics) -> dict[str, float]:
    return {
        "annualized_return": metrics.annualized_return,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sortino": metrics.sortino,
        "sharpe": metrics.sharpe,
        "realized_volatility": metrics.realized_volatility,
        "target_volatility": metrics.target_volatility,
        "annualized_turnover": metrics.annualized_turnover,
        "net_cost": metrics.net_cost,
        "worst_month": metrics.worst_month,
        "ulcer_index": metrics.ulcer_index,
    }


def result_to_dict(result: RiskResult) -> dict[str, object]:
    return {
        "candidate": {
            "key": result.candidate.key,
            "method": result.candidate.method,
            "thesis": result.candidate.thesis,
            "symbols": list(result.candidate.symbols),
            "target_vol": result.candidate.target_vol,
            "lookback": result.candidate.lookback,
            "max_exposure": result.candidate.max_exposure,
            "rebalance_days": result.candidate.rebalance_days,
            "derisk_below_ma200": result.candidate.derisk_below_ma200,
        },
        "metrics": metrics_to_dict(result.metrics),
        "primary_benchmark": metrics_to_dict(result.primary_benchmark),
        "equal_weight_benchmark": metrics_to_dict(result.equal_weight_benchmark),
        "static_60_40_benchmark": metrics_to_dict(result.static_60_40_benchmark),
        "gate_checks": result.gate_checks,
        "oos_fold_pass_rate": result.oos_fold_pass_rate,
        "drawdown_reduction": result.drawdown_reduction,
        "risk_difference_test": result.risk_difference_test,
        "alpha_significance": result.alpha_significance,
        "verdict": result.verdict,
        "reason": result.reason,
    }


def report_to_dict(report: RiskBetaReport) -> dict[str, object]:
    return {
        "status": report.status,
        "verdict": report.verdict,
        "reason": report.reason,
        "candidate_count_n": report.candidate_count_n,
        "locked_oos_start": report.locked_oos_start,
        "raw_is_survivors": report.raw_is_survivors,
        "fdr_is_survivors": report.fdr_is_survivors,
        "risk_diff_fdr_survivors": report.risk_diff_fdr_survivors,
        "risk_improved_count": report.risk_improved_count,
        "insufficient_count": report.insufficient_count,
        "results": report.results,
        "benchmarks": report.benchmarks,
        "multiple_testing": report.multiple_testing,
        "safety": report.safety,
    }


def _target_weights(
    candidate: RiskCandidate,
    bars: Sequence[Sequence[ComboBar]],
    decision_index: int,
    config: RiskBetaConfig,
) -> tuple[float, ...]:
    vols = tuple(
        _realized_vol(series, decision_index, candidate.lookback, config) for series in bars
    )
    if candidate.method.startswith("vol_target"):
        vol = vols[0]
        exposure = candidate.max_exposure if vol <= 0 else candidate.target_vol / vol
        if math.isinf(vol):
            exposure = 0.0
        exposure = min(candidate.max_exposure, max(0.0, exposure))
        if candidate.derisk_below_ma200 and not _above_sma(bars[0], decision_index, 200):
            exposure *= 0.50
        return (exposure,)
    inv = tuple(0.0 if vol <= 0 or math.isinf(vol) else 1.0 / vol for vol in vols)
    inv_sum = sum(inv)
    if inv_sum == 0:
        return tuple(0.0 for _ in bars)
    raw_weights = tuple(value / inv_sum for value in inv)
    estimated_portfolio_vol = math.sqrt(
        sum((weight * vol) ** 2 for weight, vol in zip(raw_weights, vols, strict=True))
    )
    exposure = (
        0.0
        if estimated_portfolio_vol <= 0
        else min(candidate.max_exposure, candidate.target_vol / estimated_portfolio_vol)
    )
    if candidate.derisk_below_ma200:
        risk_on = sum(1 for series in bars if _above_sma(series, decision_index, 200)) / len(bars)
        exposure *= 0.50 + 0.50 * risk_on
    return tuple(weight * exposure for weight in raw_weights)


def _evaluate_is(
    candidate: RiskCandidate,
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    locked_oos_start: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> RiskScore | None:
    fold_scores: list[float] = []
    selector_max = -1
    starts = range(0, locked_oos_start - config.train_bars - config.test_bars + 1, config.step_bars)
    for train_start in starts:
        train_end = train_start + config.train_bars
        test_start = train_end
        test_end = test_start + config.test_bars
        selector_max = max(selector_max, train_end - 1)
        strategy = simulate_candidate(
            candidate,
            bars_by_symbol,
            start=test_start,
            end=test_end,
            config=config,
            cost_model=cost_model,
        )
        benchmark = _primary_benchmark(
            candidate, bars_by_symbol, test_start, test_end, config, cost_model
        )
        fold_scores.append(_risk_improvement_score(strategy.metrics, benchmark.metrics, config))
    if len(fold_scores) < config.min_is_folds:
        return None
    return RiskScore(
        candidate=candidate,
        fold_scores=tuple(fold_scores),
        p_value=_positive_score_sign_p_value(fold_scores),
        selector_max_index=selector_max,
        first_oos_execution_index=locked_oos_start,
    )


def _locked_oos_result(
    candidate: RiskCandidate,
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    locked_oos_start: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
    *,
    alpha_p_value: float,
    alpha_fdr_discovery: bool,
    risk_diff_fdr_discovery: bool,
) -> RiskResult:
    end = min(len(bars_by_symbol[symbol]) for symbol in candidate.symbols) - 1
    strategy = simulate_candidate(
        candidate,
        bars_by_symbol,
        start=locked_oos_start,
        end=end,
        config=config,
        cost_model=cost_model,
    )
    primary = _primary_benchmark(
        candidate, bars_by_symbol, locked_oos_start, end, config, cost_model
    )
    equal_weight = equal_weight_buy_hold(
        bars_by_symbol,
        ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
        start=locked_oos_start,
        end=end,
        config=config,
        cost_model=cost_model,
    )
    static_60_40 = buy_hold_simulation(
        bars_by_symbol["BTC/USDT"],
        start=locked_oos_start,
        end=end,
        config=config,
        cost_model=cost_model,
        allocation=0.60,
    )
    fold_pass_rate = _locked_oos_fold_pass_rate(
        candidate, bars_by_symbol, locked_oos_start, end, config, cost_model
    )
    drawdown_reduction = _drawdown_reduction(strategy.metrics, primary.metrics)
    risk_difference_test = _paired_block_bootstrap_risk_difference_test(
        strategy.returns,
        primary.returns,
        strategy.metrics.target_volatility,
        primary.metrics.target_volatility,
        config,
        candidate.key,
    )
    gate_checks = {
        "drawdown_reduction_ge_20pct": drawdown_reduction >= config.drawdown_reduction_threshold,
        "calmar_gt_buy_hold": strategy.metrics.calmar > primary.metrics.calmar,
        "sortino_gt_buy_hold": strategy.metrics.sortino > primary.metrics.sortino,
        "realized_vol_near_target": _target_vol_close(strategy.metrics, config),
        "net_cost_positive_and_counted": strategy.metrics.net_cost >= 0.0,
        "oos_fold_pass_rate": fold_pass_rate >= config.min_oos_fold_pass_rate,
        "risk_difference_ci_lower_gt_0": bool(risk_difference_test["ci_lower_gt_0"]),
        "risk_difference_fdr_discovery": risk_diff_fdr_discovery,
    }
    verdict, reason = _candidate_verdict(gate_checks)
    return RiskResult(
        candidate=candidate,
        metrics=strategy.metrics,
        primary_benchmark=primary.metrics,
        equal_weight_benchmark=equal_weight.metrics,
        static_60_40_benchmark=static_60_40.metrics,
        gate_checks=gate_checks,
        oos_fold_pass_rate=fold_pass_rate,
        drawdown_reduction=drawdown_reduction,
        risk_difference_test=risk_difference_test,
        alpha_significance={
            "role": "report_only_not_a_risk_gate",
            "p_value": alpha_p_value,
            "fdr_discovery": alpha_fdr_discovery,
        },
        verdict=verdict,
        reason=reason,
    )


def _primary_benchmark(
    candidate: RiskCandidate,
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    start: int,
    end: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> RiskSimulation:
    if len(candidate.symbols) == 1:
        return buy_hold_simulation(
            bars_by_symbol[candidate.symbols[0]],
            start=start,
            end=end,
            config=config,
            cost_model=cost_model,
        )
    return equal_weight_buy_hold(
        bars_by_symbol,
        candidate.symbols,
        start=start,
        end=end,
        config=config,
        cost_model=cost_model,
    )


def _locked_oos_fold_pass_rate(
    candidate: RiskCandidate,
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    start: int,
    end: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> float:
    fold_size = max((end - start) // config.oos_folds, 1)
    passes = 0
    folds = 0
    for fold_start in range(start, end, fold_size):
        fold_end = min(fold_start + fold_size, end)
        if fold_end - fold_start < 30:
            continue
        strategy = simulate_candidate(
            candidate,
            bars_by_symbol,
            start=fold_start,
            end=fold_end,
            config=config,
            cost_model=cost_model,
        )
        benchmark = _primary_benchmark(
            candidate, bars_by_symbol, fold_start, fold_end, config, cost_model
        )
        folds += 1
        if _risk_improvement_score(strategy.metrics, benchmark.metrics, config) > 0:
            passes += 1
    return passes / folds if folds else 0.0


def _risk_improvement_score(
    strategy: RiskMetrics, benchmark: RiskMetrics, config: RiskBetaConfig
) -> float:
    drawdown_ok = _drawdown_reduction(strategy, benchmark) >= config.drawdown_reduction_threshold
    calmar_ok = strategy.calmar > benchmark.calmar
    sortino_ok = strategy.sortino > benchmark.sortino
    vol_ok = _target_vol_close(strategy, config)
    return statistics.fmean([drawdown_ok, calmar_ok, sortino_ok, vol_ok]) - 0.5


def _drawdown_reduction(strategy: RiskMetrics, benchmark: RiskMetrics) -> float:
    benchmark_dd = abs(benchmark.max_drawdown)
    if benchmark_dd == 0:
        return 0.0
    return (benchmark_dd - abs(strategy.max_drawdown)) / benchmark_dd


def _target_vol_close(metrics: RiskMetrics, config: RiskBetaConfig) -> bool:
    if metrics.target_volatility <= 0:
        return True
    return abs(metrics.realized_volatility - metrics.target_volatility) <= (
        metrics.target_volatility * config.target_vol_tolerance
    )


def _candidate_verdict(gate_checks: dict[str, bool]) -> tuple[str, str]:
    failed = [name for name, passed in gate_checks.items() if not passed]
    if failed:
        return "NO_IMPROVEMENT", "failed gates: " + ", ".join(failed)
    return "RISK_IMPROVED", "passed all risk-adjusted hard gates; paper-only candidate"


def _portfolio_verdict(
    improved_count: int, insufficient_count: int, total_count: int
) -> tuple[str, str]:
    if improved_count > 0:
        return "RISK_IMPROVED", "at least one predeclared risk configuration passed all gates"
    if insufficient_count == total_count:
        return "INSUFFICIENT", "all candidates lacked sufficient OOS/fold evidence"
    return "NO_IMPROVEMENT", "no predeclared risk configuration passed all gates"


def _report_benchmarks(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    locked_oos_start: int,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
) -> dict[str, dict[str, float]]:
    end = min(len(bars) for bars in bars_by_symbol.values()) - 1
    benchmarks: dict[str, dict[str, float]] = {}
    for symbol, bars in bars_by_symbol.items():
        benchmarks[f"{symbol}::buy_hold"] = metrics_to_dict(
            buy_hold_simulation(
                bars, start=locked_oos_start, end=end, config=config, cost_model=cost_model
            ).metrics
        )
    benchmarks["equal_weight_btc_eth_sol"] = metrics_to_dict(
        equal_weight_buy_hold(
            bars_by_symbol,
            ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
            start=locked_oos_start,
            end=end,
            config=config,
            cost_model=cost_model,
        ).metrics
    )
    benchmarks["static_60_40_btc_cash"] = metrics_to_dict(
        buy_hold_simulation(
            bars_by_symbol["BTC/USDT"],
            start=locked_oos_start,
            end=end,
            config=config,
            cost_model=cost_model,
            allocation=0.60,
        ).metrics
    )
    return benchmarks


def _positive_score_sign_p_value(scores: Sequence[float]) -> float:
    non_zero = [value for value in scores if value != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0)
    return min(1.0, sum(math.comb(n, k) * (0.5**n) for k in range(wins, n + 1)))


def _realized_vol(
    bars: Sequence[ComboBar], index: int, period: int, config: RiskBetaConfig
) -> float:
    if index < period:
        return math.inf
    returns = [
        bars[current].close / bars[current - 1].close - 1.0
        for current in range(index - period + 1, index + 1)
    ]
    return statistics.pstdev(returns) * math.sqrt(config.annualization_periods)


def _above_sma(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    if index + 1 < period:
        return False
    mean = statistics.fmean(bar.close for bar in bars[index - period + 1 : index + 1])
    return bars[index].close > mean


def _insufficient_report(
    reason: str,
    config: RiskBetaConfig,
    cost_model: ComboCostModel,
    *,
    candidate_count_n: int = 0,
    locked_oos_start: int = 0,
) -> RiskBetaReport:
    return RiskBetaReport(
        status="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=reason,
        candidate_count_n=candidate_count_n,
        locked_oos_start=locked_oos_start,
        raw_is_survivors=0,
        fdr_is_survivors=0,
        risk_diff_fdr_survivors=0,
        risk_improved_count=0,
        insufficient_count=0,
        results={},
        benchmarks={},
        multiple_testing={
            "method": "Benjamini-Hochberg FDR over predeclared risk-difference tests",
            "trial_count_n": candidate_count_n,
            "alpha_significance_role": "report_only_not_a_risk_gate",
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )
