from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from aegis.combo_indicator_search import (
    ComboBar,
    ComboCostModel,
    ComboMetrics,
    ComboSearchConfig,
    benjamini_hochberg,
    buy_hold_simulation,
    metrics_from_returns,
    metrics_to_dict,
    sign_test_p_value,
)
from aegis.combo_scorecard import TradeScorecard, trade_scorecard
from aegis.risk_disciplined_beta import (
    RiskBetaConfig,
    _paired_block_bootstrap_risk_difference_test,
)


@dataclass(frozen=True)
class ExternalContext:
    ethbtc: tuple[ComboBar, ...] = ()
    funding_by_timestamp: dict[int, float] | None = None
    oi_by_timestamp: dict[int, float] | None = None
    futures_volume_by_timestamp: dict[int, float] | None = None
    spot_volume_by_timestamp: dict[int, float] | None = None


@dataclass(frozen=True)
class PriceActionConfig:
    train_bars: int = 1_080
    test_bars: int = 300
    step_bars: int = 300
    locked_oos_fraction: float = 0.30
    annualization_periods: int = 6 * 365
    fdr_alpha: float = 0.10
    min_is_folds: int = 3
    min_trades: int = 10
    lookbacks: tuple[int, ...] = (60, 80)
    risk_rewards: tuple[float, ...] = (1.0, 1.2, 1.5)
    max_holds: tuple[int, ...] = (24, 30)
    sma_window: int = 200
    daily_sma_window: int = 200 * 6
    volume_window: int = 20
    volume_mult: float = 1.2
    min_atr_pct: float = 0.2
    max_atr_pct: float = 6.0
    short_min_atr_pct: float = 2.0
    short_max_atr_pct: float = 2.8
    atr_period: int = 14
    retest_tolerance: float = 0.006
    sweep_tolerance: float = 0.006
    funding_hot_threshold: float = 0.0005
    oi_expansion_threshold: float = 0.10
    risk_diff_bootstrap_samples: int = 400
    risk_diff_bootstrap_block_bars: int = 30
    risk_diff_ci_alpha: float = 0.05


@dataclass(frozen=True)
class PriceActionParams:
    lookback: int
    risk_reward: float
    max_hold: int

    @property
    def key(self) -> str:
        rr = str(self.risk_reward).replace(".", "p")
        return f"lookback_{self.lookback}_rr_{rr}_hold_{self.max_hold}"


@dataclass(frozen=True)
class PriceActionTrade:
    side: str
    setup: str
    signal_index: int
    entry_index: int
    exit_index: int
    entry: float
    stop: float
    target: float
    exit_price: float
    exit_reason: str
    gross_return: float
    net_return: float
    funding_cost: float


@dataclass(frozen=True)
class PriceActionSimulation:
    returns: tuple[float, ...]
    positions: tuple[int, ...]
    costs: tuple[float, ...]
    trades: tuple[PriceActionTrade, ...]
    metrics: ComboMetrics
    trade_scorecard: TradeScorecard
    turnover: float
    first_execution_index: int


@dataclass(frozen=True)
class PriceActionISScore:
    params: PriceActionParams
    fold_excess_returns: tuple[float, ...]
    fold_trade_counts: tuple[int, ...]
    p_value: float


@dataclass(frozen=True)
class PriceActionResult:
    params: PriceActionParams
    strategy_metrics: ComboMetrics
    buy_hold_metrics: ComboMetrics
    trade_scorecard: TradeScorecard
    trade_count: int
    oos_window_win_rate: float
    alpha_p_value: float
    alpha_fdr_discovery: bool
    risk_difference_test: dict[str, float | int | bool | str]
    risk_diff_fdr_discovery: bool
    alpha_gate_checks: dict[str, bool]
    risk_gate_checks: dict[str, bool]
    alpha_verdict: str
    risk_verdict: str
    reason: str


@dataclass(frozen=True)
class PriceActionReport:
    status: str
    verdict: str
    reason: str
    candidate_count_n: int
    locked_oos_start: int
    raw_is_survivors: int
    alpha_fdr_survivors: int
    risk_diff_fdr_survivors: int
    alpha_edge_count: int
    risk_improved_count: int
    insufficient_count: int
    external_coverage: dict[str, float | int | str]
    hermes_reconciliation: dict[str, float | int | str]
    results: dict[str, dict[str, object]]
    multiple_testing: dict[str, float | int | str]
    safety: dict[str, bool | str]


DEFAULT_PRICE_ACTION_CONFIG = PriceActionConfig()
EMPTY_EXTERNAL_CONTEXT = ExternalContext()
DEFAULT_PRICE_ACTION_COST_MODEL = ComboCostModel(
    fee_bps=4.0,
    slippage_bps=4.0,
    funding_bps_per_period=0.0,
    funding_label="short funding debited from funding_by_timestamp when available",
)


def predeclared_price_action_params(
    config: PriceActionConfig = DEFAULT_PRICE_ACTION_CONFIG,
) -> tuple[PriceActionParams, ...]:
    return tuple(
        PriceActionParams(lookback=lookback, risk_reward=rr, max_hold=max_hold)
        for lookback in config.lookbacks
        for rr in config.risk_rewards
        for max_hold in config.max_holds
    )


def run_btc_price_action_reeval(
    bars: Sequence[ComboBar],
    *,
    external: ExternalContext = EMPTY_EXTERNAL_CONTEXT,
    config: PriceActionConfig = DEFAULT_PRICE_ACTION_CONFIG,
    cost_model: ComboCostModel = DEFAULT_PRICE_ACTION_COST_MODEL,
    hermes_total_return_pct: float = 17.93072784238161,
    hermes_buy_hold_pct: float = 134.90259013211917,
) -> PriceActionReport:
    if not bars:
        return _insufficient_report("no BTC bars supplied", config, hermes_total_return_pct)
    locked_oos_start = int(len(bars) * (1.0 - config.locked_oos_fraction))
    if locked_oos_start < config.train_bars + config.test_bars:
        return _insufficient_report(
            "not enough in-sample bars before locked OOS",
            config,
            hermes_total_return_pct,
            locked_oos_start=locked_oos_start,
        )
    params_grid = predeclared_price_action_params(config)
    is_scores = tuple(
        score
        for score in (
            _evaluate_is(params, bars, external, locked_oos_start, config, cost_model)
            for params in params_grid
        )
        if score is not None
    )
    if len(is_scores) < len(params_grid):
        return _insufficient_report(
            "one or more parameter sets lacked enough walk-forward folds",
            config,
            hermes_total_return_pct,
            candidate_count_n=len(params_grid),
            locked_oos_start=locked_oos_start,
        )
    alpha_discoveries = benjamini_hochberg(
        [score.p_value for score in is_scores], alpha=config.fdr_alpha
    )
    alpha_fdr_names = {
        score.params.key for score, keep in zip(is_scores, alpha_discoveries, strict=True) if keep
    }
    alpha_p_values = {score.params.key: score.p_value for score in is_scores}
    preliminary = tuple(
        _locked_oos_result(
            params,
            bars,
            external,
            locked_oos_start,
            config,
            cost_model,
            alpha_p_value=alpha_p_values[params.key],
            alpha_fdr_discovery=params.key in alpha_fdr_names,
            risk_diff_fdr_discovery=False,
        )
        for params in params_grid
    )
    risk_discoveries = benjamini_hochberg(
        [
            float(result.risk_difference_test["p_value"])
            if result.risk_difference_test["valid"]
            else 1.0
            for result in preliminary
        ],
        alpha=config.fdr_alpha,
    )
    risk_fdr_names = {
        result.params.key
        for result, keep in zip(preliminary, risk_discoveries, strict=True)
        if keep
    }
    results = tuple(
        _locked_oos_result(
            params,
            bars,
            external,
            locked_oos_start,
            config,
            cost_model,
            alpha_p_value=alpha_p_values[params.key],
            alpha_fdr_discovery=params.key in alpha_fdr_names,
            risk_diff_fdr_discovery=params.key in risk_fdr_names,
        )
        for params in params_grid
    )
    alpha_edge_count = sum(1 for result in results if result.alpha_verdict == "EDGE_CANDIDATE")
    risk_improved_count = sum(1 for result in results if result.risk_verdict == "RISK_IMPROVED")
    insufficient_count = sum(
        1
        for result in results
        if result.alpha_verdict == "INSUFFICIENT" or result.risk_verdict == "INSUFFICIENT"
    )
    verdict, reason = _portfolio_verdict(
        alpha_edge_count, risk_improved_count, insufficient_count, len(results)
    )
    raw_is_survivors = sum(
        1 for score in is_scores if statistics.fmean(score.fold_excess_returns) > 0
    )
    best_result = max(results, key=lambda result: result.strategy_metrics.total_return)
    return PriceActionReport(
        status="OK",
        verdict=verdict,
        reason=reason,
        candidate_count_n=len(params_grid),
        locked_oos_start=locked_oos_start,
        raw_is_survivors=raw_is_survivors,
        alpha_fdr_survivors=len(alpha_fdr_names),
        risk_diff_fdr_survivors=len(risk_fdr_names),
        alpha_edge_count=alpha_edge_count,
        risk_improved_count=risk_improved_count,
        insufficient_count=insufficient_count,
        external_coverage=external_coverage(external, len(bars)),
        hermes_reconciliation={
            "hermes_in_sample_total_return_pct": hermes_total_return_pct,
            "hermes_buy_hold_pct": hermes_buy_hold_pct,
            "strict_best_locked_oos_total_return_pct": best_result.strategy_metrics.total_return
            * 100.0,
            "strict_best_locked_oos_buy_hold_pct": best_result.buy_hold_metrics.total_return
            * 100.0,
            "strict_minus_hermes_headline_pct": best_result.strategy_metrics.total_return * 100.0
            - hermes_total_return_pct,
        },
        results={result.params.key: result_to_dict(result) for result in results},
        multiple_testing={
            "method": "BH-FDR over predeclared parameter grid",
            "candidate_count_n": len(params_grid),
            "alpha": config.fdr_alpha,
            "raw_is_survivors": raw_is_survivors,
            "alpha_fdr_survivors": len(alpha_fdr_names),
            "risk_diff_fdr_survivors": len(risk_fdr_names),
            "risk_diff_test": "paired block bootstrap reused from risk_disciplined_beta",
            "risk_diff_bootstrap_samples": config.risk_diff_bootstrap_samples,
            "risk_diff_bootstrap_block_bars": config.risk_diff_bootstrap_block_bars,
            "risk_diff_ci_alpha": config.risk_diff_ci_alpha,
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "wallet_or_order_api_used": False,
            "hermes_source_in_public": False,
            "funding": cost_model.funding_label,
        },
    )


def simulate_price_action(
    bars: Sequence[ComboBar],
    params: PriceActionParams,
    *,
    external: ExternalContext = EMPTY_EXTERNAL_CONTEXT,
    start: int,
    end: int,
    config: PriceActionConfig = DEFAULT_PRICE_ACTION_CONFIG,
    cost_model: ComboCostModel = DEFAULT_PRICE_ACTION_COST_MODEL,
) -> PriceActionSimulation:
    start = max(start, _warmup(config, params))
    end = min(end, len(bars) - 1)
    returns = [0.0 for _ in range(max(end - start, 0))]
    positions = [0 for _ in range(max(end - start, 0))]
    costs = [0.0 for _ in range(max(end - start, 0))]
    trades: list[PriceActionTrade] = []
    turnover = 0.0
    index = start
    while index < end:
        signal_index = index - 1
        signal = _price_action_signal(bars, external, signal_index, params, config)
        if signal is None:
            index += 1
            continue
        entry_index = index
        trade = _execute_trade(
            bars,
            external,
            signal,
            entry_index,
            min(end, entry_index + params.max_hold),
            cost_model,
        )
        side = int(signal["side"])
        period = trade.exit_index - start
        if 0 <= period < len(returns):
            returns[period] += trade.net_return
            costs[period] += (
                abs(side) * cost_model.one_way_cost * 2.0 + trade.funding_cost
            )
        position_start = max(entry_index - start, 0)
        position_end = min(trade.exit_index - start + 1, len(positions))
        for pos_index in range(position_start, position_end):
            positions[pos_index] = side
        trades.append(trade)
        turnover += 2.0
        index = max(trade.exit_index + 1, index + 1)
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return PriceActionSimulation(
        returns=tuple(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        trades=tuple(trades),
        metrics=metrics,
        trade_scorecard=trade_scorecard([trade.net_return for trade in trades]),
        turnover=turnover,
        first_execution_index=start,
    )


def external_coverage(external: ExternalContext, bar_count: int) -> dict[str, float | int | str]:
    def coverage(mapping: dict[int, float] | None) -> float:
        if not mapping or bar_count <= 0:
            return 0.0
        return len(mapping) / bar_count

    return {
        "ethbtc_rows": len(external.ethbtc),
        "funding_rows": len(external.funding_by_timestamp or {}),
        "oi_rows": len(external.oi_by_timestamp or {}),
        "futures_volume_rows": len(external.futures_volume_by_timestamp or {}),
        "spot_volume_rows": len(external.spot_volume_by_timestamp or {}),
        "funding_coverage": coverage(external.funding_by_timestamp),
        "oi_coverage": coverage(external.oi_by_timestamp),
        "oi_policy": (
            "no forward fill; use same-timestamp OI, else same-timestamp futures/spot "
            "volume ratio proxy if available"
        ),
    }


def result_to_dict(result: PriceActionResult) -> dict[str, object]:
    return {
        "params": {
            "lookback": result.params.lookback,
            "risk_reward": result.params.risk_reward,
            "max_hold": result.params.max_hold,
        },
        "strategy_metrics": metrics_to_dict(result.strategy_metrics),
        "buy_hold_metrics": metrics_to_dict(result.buy_hold_metrics),
        "trade_scorecard": {
            "total_trades": result.trade_scorecard.total_trades,
            "win_rate": result.trade_scorecard.win_rate,
            "average_win": result.trade_scorecard.average_win,
            "average_loss": result.trade_scorecard.average_loss,
            "win_loss_ratio": result.trade_scorecard.win_loss_ratio,
            "expectancy_per_trade": result.trade_scorecard.expectancy_per_trade,
            "profit_factor": result.trade_scorecard.profit_factor,
            "max_consecutive_losses": result.trade_scorecard.max_consecutive_losses,
        },
        "trade_count": result.trade_count,
        "oos_window_win_rate": result.oos_window_win_rate,
        "alpha_p_value": result.alpha_p_value,
        "alpha_fdr_discovery": result.alpha_fdr_discovery,
        "risk_difference_test": result.risk_difference_test,
        "risk_diff_fdr_discovery": result.risk_diff_fdr_discovery,
        "alpha_gate_checks": result.alpha_gate_checks,
        "risk_gate_checks": result.risk_gate_checks,
        "alpha_verdict": result.alpha_verdict,
        "risk_verdict": result.risk_verdict,
        "reason": result.reason,
    }


def report_to_dict(report: PriceActionReport) -> dict[str, object]:
    return {
        "status": report.status,
        "verdict": report.verdict,
        "reason": report.reason,
        "candidate_count_n": report.candidate_count_n,
        "locked_oos_start": report.locked_oos_start,
        "raw_is_survivors": report.raw_is_survivors,
        "alpha_fdr_survivors": report.alpha_fdr_survivors,
        "risk_diff_fdr_survivors": report.risk_diff_fdr_survivors,
        "alpha_edge_count": report.alpha_edge_count,
        "risk_improved_count": report.risk_improved_count,
        "insufficient_count": report.insufficient_count,
        "external_coverage": report.external_coverage,
        "hermes_reconciliation": report.hermes_reconciliation,
        "results": report.results,
        "multiple_testing": report.multiple_testing,
        "safety": report.safety,
    }


def _evaluate_is(
    params: PriceActionParams,
    bars: Sequence[ComboBar],
    external: ExternalContext,
    locked_oos_start: int,
    config: PriceActionConfig,
    cost_model: ComboCostModel,
) -> PriceActionISScore | None:
    excess: list[float] = []
    trades: list[int] = []
    for train_start in range(
        0,
        locked_oos_start - config.train_bars - config.test_bars + 1,
        config.step_bars,
    ):
        test_start = train_start + config.train_bars
        test_end = test_start + config.test_bars
        strategy = simulate_price_action(
            bars,
            params,
            external=external,
            start=test_start,
            end=test_end,
            config=config,
            cost_model=cost_model,
        )
        benchmark = buy_hold_simulation(
            bars,
            start=test_start,
            end=test_end,
            config=_combo_config(config),
            cost_model=cost_model,
        )
        excess.append(strategy.metrics.total_return - benchmark.metrics.total_return)
        trades.append(len(strategy.trades))
    if len(excess) < config.min_is_folds:
        return None
    return PriceActionISScore(
        params=params,
        fold_excess_returns=tuple(excess),
        fold_trade_counts=tuple(trades),
        p_value=sign_test_p_value(excess),
    )


def _locked_oos_result(
    params: PriceActionParams,
    bars: Sequence[ComboBar],
    external: ExternalContext,
    locked_oos_start: int,
    config: PriceActionConfig,
    cost_model: ComboCostModel,
    *,
    alpha_p_value: float,
    alpha_fdr_discovery: bool,
    risk_diff_fdr_discovery: bool,
) -> PriceActionResult:
    end = len(bars) - 1
    strategy = simulate_price_action(
        bars,
        params,
        external=external,
        start=locked_oos_start,
        end=end,
        config=config,
        cost_model=cost_model,
    )
    benchmark = buy_hold_simulation(
        bars,
        start=locked_oos_start,
        end=end,
        config=_combo_config(config),
        cost_model=cost_model,
    )
    risk_config = RiskBetaConfig(
        annualization_periods=config.annualization_periods,
        fdr_alpha=config.fdr_alpha,
        risk_diff_bootstrap_samples=config.risk_diff_bootstrap_samples,
        risk_diff_bootstrap_block_bars=config.risk_diff_bootstrap_block_bars,
        risk_diff_ci_alpha=config.risk_diff_ci_alpha,
    )
    risk_difference = _paired_block_bootstrap_risk_difference_test(
        strategy.returns,
        benchmark.returns,
        0.0,
        0.0,
        risk_config,
        params.key,
    )
    oos_window_win_rate = _window_win_rate(strategy.returns, benchmark.returns, config.test_bars)
    alpha_checks = {
        "min_trades": len(strategy.trades) >= config.min_trades,
        "total_return_gt_buy_hold": strategy.metrics.total_return > benchmark.metrics.total_return,
        "sharpe_gt_buy_hold": strategy.metrics.sharpe > benchmark.metrics.sharpe,
        "calmar_gt_buy_hold": strategy.metrics.calmar > benchmark.metrics.calmar,
        "alpha_fdr_discovery": alpha_fdr_discovery,
    }
    risk_checks = {
        "min_trades": len(strategy.trades) >= config.min_trades,
        "drawdown_reduction_positive": _drawdown_reduction(strategy.metrics, benchmark.metrics) > 0,
        "calmar_gt_buy_hold": strategy.metrics.calmar > benchmark.metrics.calmar,
        "sortino_gt_buy_hold": strategy.metrics.sortino > benchmark.metrics.sortino,
        "risk_difference_ci_lower_gt_0": bool(risk_difference["ci_lower_gt_0"]),
        "risk_difference_fdr_discovery": risk_diff_fdr_discovery,
    }
    alpha_verdict, alpha_reason = _gate_verdict(alpha_checks, "EDGE_CANDIDATE")
    risk_verdict, risk_reason = _gate_verdict(risk_checks, "RISK_IMPROVED")
    if len(strategy.trades) < config.min_trades:
        alpha_verdict = "INSUFFICIENT"
        risk_verdict = "INSUFFICIENT"
    return PriceActionResult(
        params=params,
        strategy_metrics=strategy.metrics,
        buy_hold_metrics=benchmark.metrics,
        trade_scorecard=strategy.trade_scorecard,
        trade_count=len(strategy.trades),
        oos_window_win_rate=oos_window_win_rate,
        alpha_p_value=alpha_p_value,
        alpha_fdr_discovery=alpha_fdr_discovery,
        risk_difference_test=risk_difference,
        risk_diff_fdr_discovery=risk_diff_fdr_discovery,
        alpha_gate_checks=alpha_checks,
        risk_gate_checks=risk_checks,
        alpha_verdict=alpha_verdict,
        risk_verdict=risk_verdict,
        reason=f"alpha: {alpha_reason}; risk: {risk_reason}",
    )


def _price_action_signal(
    bars: Sequence[ComboBar],
    external: ExternalContext,
    index: int,
    params: PriceActionParams,
    config: PriceActionConfig,
) -> dict[str, int | float | str] | None:
    if index < _warmup(config, params) or index <= params.lookback:
        return None
    bar = bars[index]
    prev = bars[index - 1]
    atr_pct = _atr_pct(bars, index, config.atr_period)
    if atr_pct < config.min_atr_pct or atr_pct > config.max_atr_pct:
        return None
    volume_threshold = _sma([item.volume for item in bars], index - 1, config.volume_window)
    volume_ok = (
        bar.volume > volume_threshold * config.volume_mult
        or prev.volume > volume_threshold * config.volume_mult
    )
    if not volume_ok:
        return None
    high = max(item.high for item in bars[index - params.lookback - 1 : index - 1])
    low = min(item.low for item in bars[index - params.lookback - 1 : index - 1])
    long_regime = _long_regime(bars, external, index, config)
    short_regime = _short_regime(bars, external, index, atr_pct, config)
    if (
        long_regime
        and prev.close > high
        and bar.low <= high * (1.0 + config.retest_tolerance)
        and bar.close > high
    ):
        stop = min(bar.low, high * (1.0 - config.retest_tolerance))
        return _signal(1, "breakout-retest-volume", index, stop, params.risk_reward)
    swept_low = prev.low < low * (1.0 - config.sweep_tolerance) and prev.close > low
    if long_regime and swept_low and bar.close > prev.close:
        stop = min(prev.low, bar.low)
        return _signal(1, "false-breakdown-reclaim", index, stop, params.risk_reward)
    if (
        short_regime
        and prev.close < low
        and bar.high >= low * (1.0 - config.retest_tolerance)
        and bar.close < low
    ):
        stop = max(bar.high, low * (1.0 + config.retest_tolerance))
        return _signal(-1, "breakdown-retest-volume", index, stop, params.risk_reward)
    return None


def _signal(
    side: int,
    setup: str,
    signal_index: int,
    stop: float,
    risk_reward: float,
) -> dict[str, int | float | str]:
    return {
        "side": side,
        "setup": setup,
        "signal_index": signal_index,
        "stop": stop,
        "risk_reward": risk_reward,
    }


def _execute_trade(
    bars: Sequence[ComboBar],
    external: ExternalContext,
    signal: dict[str, int | float | str],
    entry_index: int,
    max_exit_index: int,
    cost_model: ComboCostModel,
) -> PriceActionTrade:
    side = int(signal["side"])
    entry = bars[entry_index].open
    stop = float(signal["stop"])
    risk = abs(entry - stop)
    if risk <= 0:
        stop = entry * (0.99 if side > 0 else 1.01)
        risk = abs(entry - stop)
    target = entry + side * risk * float(signal["risk_reward"])
    exit_price = bars[max_exit_index].close
    exit_index = max_exit_index
    exit_reason = "timeout"
    for index in range(entry_index, max_exit_index + 1):
        bar = bars[index]
        if side > 0:
            stop_hit = bar.low <= stop
            target_hit = bar.high >= target
            if stop_hit:
                exit_price = stop
                exit_index = index
                exit_reason = "stop"
                break
            if target_hit:
                exit_price = target
                exit_index = index
                exit_reason = "target"
                break
        else:
            stop_hit = bar.high >= stop
            target_hit = bar.low <= target
            if stop_hit:
                exit_price = stop
                exit_index = index
                exit_reason = "stop"
                break
            if target_hit:
                exit_price = target
                exit_index = index
                exit_reason = "target"
                break
    gross = side * (exit_price / entry - 1.0)
    funding_cost = _short_funding_cost(external, bars, entry_index, exit_index, side)
    net = gross - cost_model.one_way_cost * 2.0 - funding_cost
    return PriceActionTrade(
        side="long" if side > 0 else "short",
        setup=str(signal["setup"]),
        signal_index=int(signal["signal_index"]),
        entry_index=entry_index,
        exit_index=exit_index,
        entry=entry,
        stop=stop,
        target=target,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_return=gross,
        net_return=net,
        funding_cost=funding_cost,
    )


def _long_regime(
    bars: Sequence[ComboBar],
    external: ExternalContext,
    index: int,
    config: PriceActionConfig,
) -> bool:
    close = bars[index].close
    if close <= _sma([bar.close for bar in bars], index, config.sma_window):
        return False
    if close <= _sma([bar.close for bar in bars], index, config.daily_sma_window):
        return False
    if _ethbtc_weak(external, index, config):
        return False
    funding = _same_timestamp_value(external.funding_by_timestamp, bars[index].timestamp)
    crowded = _crowding_expansion(external, bars, index, config)
    return not (funding is not None and funding > config.funding_hot_threshold and crowded)


def _short_regime(
    bars: Sequence[ComboBar],
    external: ExternalContext,
    index: int,
    atr_pct: float,
    config: PriceActionConfig,
) -> bool:
    close = bars[index].close
    if not (config.short_min_atr_pct <= atr_pct <= config.short_max_atr_pct):
        return False
    return close < _sma([bar.close for bar in bars], index, config.sma_window) and _ethbtc_weak(
        external, index, config
    )


def _ethbtc_weak(external: ExternalContext, index: int, config: PriceActionConfig) -> bool:
    if not external.ethbtc or index >= len(external.ethbtc):
        return False
    closes = [bar.close for bar in external.ethbtc]
    return external.ethbtc[index].close < _sma(closes, index, config.sma_window)


def _crowding_expansion(
    external: ExternalContext,
    bars: Sequence[ComboBar],
    index: int,
    config: PriceActionConfig,
) -> bool:
    timestamp = bars[index].timestamp
    previous_timestamp = bars[max(index - 6, 0)].timestamp
    oi = _same_timestamp_value(external.oi_by_timestamp, timestamp)
    oi_prev = _same_timestamp_value(external.oi_by_timestamp, previous_timestamp)
    if oi is not None and oi_prev is not None and oi_prev != 0.0:
        return (oi / oi_prev - 1.0) > config.oi_expansion_threshold
    future_vol = _same_timestamp_value(external.futures_volume_by_timestamp, timestamp)
    spot_vol = _same_timestamp_value(external.spot_volume_by_timestamp, timestamp)
    future_prev = _same_timestamp_value(external.futures_volume_by_timestamp, previous_timestamp)
    spot_prev = _same_timestamp_value(external.spot_volume_by_timestamp, previous_timestamp)
    if future_vol is None or future_prev is None:
        return False
    if spot_vol is None or spot_prev is None or spot_vol == 0.0 or spot_prev == 0.0:
        return False
    ratio = future_vol / spot_vol
    prev_ratio = future_prev / spot_prev
    return prev_ratio > 0 and ratio / prev_ratio - 1.0 > config.oi_expansion_threshold


def _same_timestamp_value(mapping: dict[int, float] | None, timestamp: int) -> float | None:
    if not mapping:
        return None
    return mapping.get(timestamp)


def _short_funding_cost(
    external: ExternalContext,
    bars: Sequence[ComboBar],
    entry_index: int,
    exit_index: int,
    side: int,
) -> float:
    if side >= 0 or not external.funding_by_timestamp:
        return 0.0
    cost = 0.0
    for index in range(entry_index, exit_index + 1):
        rate = external.funding_by_timestamp.get(bars[index].timestamp)
        if rate is not None:
            cost += max(rate, 0.0)
    return cost


def _warmup(config: PriceActionConfig, params: PriceActionParams) -> int:
    return max(
        config.daily_sma_window,
        config.sma_window,
        params.lookback + 2,
        config.atr_period + 1,
    )


def _atr_pct(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period:
        return 0.0
    trs: list[float] = []
    for current in range(index - period + 1, index + 1):
        prev_close = bars[current - 1].close
        bar = bars[current]
        trs.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
    return statistics.fmean(trs) / bars[index].close * 100.0 if bars[index].close else 0.0


def _sma(values: Sequence[float], index: int, period: int) -> float:
    if index + 1 < period:
        return statistics.fmean(values[: index + 1])
    return statistics.fmean(values[index - period + 1 : index + 1])


def _drawdown_reduction(strategy: ComboMetrics, benchmark: ComboMetrics) -> float:
    benchmark_dd = abs(benchmark.max_drawdown)
    if benchmark_dd == 0:
        return 0.0
    return (benchmark_dd - abs(strategy.max_drawdown)) / benchmark_dd


def _window_win_rate(
    strategy_returns: Sequence[float], benchmark_returns: Sequence[float], window: int
) -> float:
    if not strategy_returns or not benchmark_returns:
        return 0.0
    wins = 0
    total = 0
    limit = min(len(strategy_returns), len(benchmark_returns))
    for start in range(0, limit, window):
        strategy_total = _compound(strategy_returns[start : start + window])
        benchmark_total = _compound(benchmark_returns[start : start + window])
        if strategy_total > benchmark_total:
            wins += 1
        total += 1
    return wins / total if total else 0.0


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _combo_config(config: PriceActionConfig) -> ComboSearchConfig:
    return ComboSearchConfig(
        train_bars=config.train_bars,
        test_bars=config.test_bars,
        step_bars=config.step_bars,
        locked_oos_fraction=config.locked_oos_fraction,
        annualization_periods=config.annualization_periods,
        fdr_alpha=config.fdr_alpha,
        min_is_folds=config.min_is_folds,
    )


def _gate_verdict(gates: dict[str, bool], pass_verdict: str) -> tuple[str, str]:
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        return "NO_EDGE" if pass_verdict == "EDGE_CANDIDATE" else "NO_IMPROVEMENT", (
            "failed gates: " + ", ".join(failed)
        )
    return pass_verdict, "passed all hard gates"


def _portfolio_verdict(
    alpha_edge_count: int,
    risk_improved_count: int,
    insufficient_count: int,
    total_count: int,
) -> tuple[str, str]:
    if alpha_edge_count > 0:
        return "EDGE_CANDIDATE", "at least one configuration beat buy-and-hold alpha gates"
    if risk_improved_count > 0:
        return "RISK_IMPROVED", "at least one configuration passed risk-difference gates"
    if insufficient_count == total_count:
        return "INSUFFICIENT", "all configurations lacked enough locked-OOS trades"
    return "NO_EDGE", "no predeclared configuration passed alpha or risk-improvement gates"


def _insufficient_report(
    reason: str,
    config: PriceActionConfig,
    hermes_total_return_pct: float,
    *,
    candidate_count_n: int = 0,
    locked_oos_start: int = 0,
) -> PriceActionReport:
    return PriceActionReport(
        status="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=reason,
        candidate_count_n=candidate_count_n,
        locked_oos_start=locked_oos_start,
        raw_is_survivors=0,
        alpha_fdr_survivors=0,
        risk_diff_fdr_survivors=0,
        alpha_edge_count=0,
        risk_improved_count=0,
        insufficient_count=0,
        external_coverage={},
        hermes_reconciliation={"hermes_in_sample_total_return_pct": hermes_total_return_pct},
        results={},
        multiple_testing={
            "candidate_count_n": candidate_count_n,
            "alpha": config.fdr_alpha,
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "wallet_or_order_api_used": False,
            "hermes_source_in_public": False,
        },
    )
