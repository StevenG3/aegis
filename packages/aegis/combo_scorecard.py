from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from aegis.backtest_core import (
    BacktestDiscipline,
    HypothesisSpec,
    TradeScorecard,
    run_backtest,
    trade_scorecard,
    trade_scorecard_to_dict,
)
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

__all__ = [
    "ScorecardConfig",
    "TradeScorecard",
    "combo_scorecard_hypothesis_spec",
    "predeclared_scorecard_combos",
    "report_to_dict",
    "run_combo_scorecard",
    "scorecard_to_dict",
    "simulate_combo_with_signals",
    "trade_scorecard",
    "trade_scorecard_to_dict",
]


@dataclass(frozen=True)
class ScorecardConfig:
    train_bars: int = 730
    test_bars: int = 180
    step_bars: int = 180
    locked_oos_fraction: float = 0.30
    annualization_periods: int = 365
    fdr_alpha: float = 0.10
    min_is_folds: int = 3
    min_trades: int = 5
    profit_factor_threshold: float = 1.30


@dataclass(frozen=True)
class PredeclaredCombo:
    key: str
    thesis: str
    params: dict[str, int | float]
    signal: Callable[[Sequence[ComboBar], int], bool]
    warmup: int


@dataclass(frozen=True)
class ScorecardCandidate:
    symbol: str
    combo: PredeclaredCombo

    @property
    def name(self) -> str:
        return f"{self.symbol}::{self.combo.key}"


@dataclass(frozen=True)
class CandidateScorecard:
    candidate: ScorecardCandidate
    trade: TradeScorecard
    metrics: ComboMetrics
    buy_hold_metrics: ComboMetrics
    composite_score: float
    excess_return: float
    beats_buy_hold_return: bool
    beats_buy_hold_sharpe: bool
    oos_window_win_rate: float
    gate_checks: dict[str, bool]
    verdict: str
    reason: str


@dataclass(frozen=True)
class ScorecardSimulation:
    returns: tuple[float, ...]
    positions: tuple[int, ...]
    costs: tuple[float, ...]
    trade_returns: tuple[float, ...]
    turnover: float
    metrics: ComboMetrics
    first_execution_index: int


@dataclass(frozen=True)
class ISScore:
    candidate: ScorecardCandidate
    fold_excess_returns: tuple[float, ...]
    p_value: float
    selector_max_index: int
    first_oos_execution_index: int


@dataclass(frozen=True)
class ComboScorecardReport:
    status: str
    verdict: str
    reason: str
    symbols: tuple[str, ...]
    combo_count: int
    candidate_count_n: int
    locked_oos_start: int
    raw_is_survivors: int
    fdr_is_survivors: int
    go_candidates: int
    insufficient_candidates: int
    scorecards: dict[str, dict[str, object]]
    benchmark_scorecards: dict[str, dict[str, object]]
    multiple_testing: dict[str, float | int | str]
    safety: dict[str, bool | str]


DEFAULT_SCORECARD_CONFIG = ScorecardConfig()
DEFAULT_SCORECARD_COST_MODEL = ComboCostModel()


def predeclared_scorecard_combos() -> tuple[PredeclaredCombo, ...]:
    return (
        PredeclaredCombo(
            key="trend_pullback_ma200_rsi14_30",
            thesis="Long-only mean reversion inside an uptrend: close above 200MA and RSI(14)<30.",
            params={"ma": 200, "rsi": 14, "rsi_entry": 30},
            signal=lambda bars, i: _close_gt_sma(bars, i, 200) and _rsi(bars, i, 14) < 30,
            warmup=200,
        ),
        PredeclaredCombo(
            key="golden_cross_low_vol_50_200_rv60",
            thesis="Trend following with a low-volatility filter to reduce whipsaw.",
            params={"fast_ma": 50, "slow_ma": 200, "realized_vol": 60},
            signal=lambda bars, i: (
                _sma_gt_sma(bars, i, 50, 200) and _realized_vol_below_median(bars, i, 60)
            ),
            warmup=200,
        ),
        PredeclaredCombo(
            key="donchian20_breakout_ma200",
            thesis="Turtle-style breakout only when the long-term trend is positive.",
            params={"donchian_high": 20, "ma": 200},
            signal=lambda bars, i: _donchian_breakout(bars, i, 20) and _close_gt_sma(bars, i, 200),
            warmup=200,
        ),
        PredeclaredCombo(
            key="macd_adx_trend_strength_12_26_9_14_25",
            thesis="MACD momentum is traded only when ADX indicates trend strength.",
            params={"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "adx": 14, "adx_min": 25},
            signal=lambda bars, i: (
                _macd_histogram(bars, i, 12, 26, 9) > 0 and _adx(bars, i, 14) > 25
            ),
            warmup=104,
        ),
        PredeclaredCombo(
            key="bollinger_lower_low_vol_20_2_rv60",
            thesis="Mean reversion at the lower Bollinger band only in a low-volatility regime.",
            params={"bollinger": 20, "stdev": 2, "realized_vol": 60},
            signal=lambda bars, i: (
                _below_bollinger_lower(bars, i, 20, 2.0) and _realized_vol_below_median(bars, i, 60)
            ),
            warmup=180,
        ),
        PredeclaredCombo(
            key="tsmom30_low_vol_rv60",
            thesis=(
                "Positive time-series momentum may work better after filtering out high "
                "volatility."
            ),
            params={"tsmom": 30, "realized_vol": 60},
            signal=lambda bars, i: (
                _tsmom_positive(bars, i, 30) and _realized_vol_below_median(bars, i, 60)
            ),
            warmup=180,
        ),
        PredeclaredCombo(
            key="tsmom90_low_vol_rv60",
            thesis="Longer-horizon time-series momentum with the same low-volatility filter.",
            params={"tsmom": 90, "realized_vol": 60},
            signal=lambda bars, i: (
                _tsmom_positive(bars, i, 90) and _realized_vol_below_median(bars, i, 60)
            ),
            warmup=270,
        ),
        PredeclaredCombo(
            key="roc20_obv_volume_confirmation",
            thesis="Short momentum is traded only when OBV confirms accumulation.",
            params={"roc": 20, "obv_slope": 20},
            signal=lambda bars, i: _roc(bars, i, 20) > 0 and _obv_slope(bars, i, 20) > 0,
            warmup=40,
        ),
    )


def run_combo_scorecard(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: ScorecardConfig = DEFAULT_SCORECARD_CONFIG,
    cost_model: ComboCostModel = DEFAULT_SCORECARD_COST_MODEL,
) -> ComboScorecardReport:
    spec = combo_scorecard_hypothesis_spec(
        bars_by_symbol,
        config=config,
        cost_model=cost_model,
        runner=lambda: _run_combo_scorecard_impl(
            bars_by_symbol, config=config, cost_model=cost_model
        ),
    )
    return cast(ComboScorecardReport, run_backtest(spec).payload)


def combo_scorecard_hypothesis_spec(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: ScorecardConfig = DEFAULT_SCORECARD_CONFIG,
    cost_model: ComboCostModel = DEFAULT_SCORECARD_COST_MODEL,
    runner: Callable[[], object] | None = None,
) -> HypothesisSpec:
    symbols = tuple(sorted(bars_by_symbol)) or ("<empty>",)
    combos = predeclared_scorecard_combos()
    return HypothesisSpec(
        key="combo_scorecard",
        hypothesis_type="combo",
        universe=symbols,
        predeclared_signals=tuple(combo.key for combo in combos),
        params={
            "train_bars": config.train_bars,
            "test_bars": config.test_bars,
            "step_bars": config.step_bars,
            "locked_oos_fraction": config.locked_oos_fraction,
            "min_trades": config.min_trades,
            "profit_factor_threshold": config.profit_factor_threshold,
        },
        cost_model=cost_model,
        benchmark="buy_and_hold",
        data_source="caller_supplied_ohlcv_bars",
        trial_count_n=max(1, len(symbols) * len(combos)),
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


def _run_combo_scorecard_impl(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    *,
    config: ScorecardConfig = DEFAULT_SCORECARD_CONFIG,
    cost_model: ComboCostModel = DEFAULT_SCORECARD_COST_MODEL,
) -> ComboScorecardReport:
    if not bars_by_symbol:
        return _insufficient_report("no bars supplied", (), config, cost_model)
    symbols = tuple(sorted(bars_by_symbol))
    min_bars = min(len(bars_by_symbol[symbol]) for symbol in symbols)
    locked_oos_start = int(min_bars * (1.0 - config.locked_oos_fraction))
    if locked_oos_start < config.train_bars + config.test_bars:
        return _insufficient_report(
            "not enough in-sample bars before locked OOS",
            symbols,
            config,
            cost_model,
            locked_oos_start=locked_oos_start,
        )
    combos = predeclared_scorecard_combos()
    candidates = tuple(
        ScorecardCandidate(symbol=symbol, combo=combo) for symbol in symbols for combo in combos
    )
    signal_cache_by_symbol = {
        symbol: _combo_signal_cache(bars_by_symbol[symbol], combos) for symbol in symbols
    }
    is_scores = tuple(
        result
        for result in (
            _evaluate_is(
                candidate,
                bars_by_symbol[candidate.symbol],
                signal_cache_by_symbol[candidate.symbol][candidate.combo.key],
                locked_oos_start,
                config,
                cost_model,
            )
            for candidate in candidates
        )
        if result is not None
    )
    if len(is_scores) < len(candidates):
        return _insufficient_report(
            "one or more candidates lacked enough walk-forward folds",
            symbols,
            config,
            cost_model,
            combo_count=len(combos),
            candidate_count_n=len(candidates),
            locked_oos_start=locked_oos_start,
        )
    discoveries = benjamini_hochberg([score.p_value for score in is_scores], alpha=config.fdr_alpha)
    fdr_names = {
        score.candidate.name for score, keep in zip(is_scores, discoveries, strict=True) if keep
    }
    scorecards = tuple(
        _locked_oos_scorecard(
            candidate,
            bars_by_symbol[candidate.symbol],
            signal_cache_by_symbol[candidate.symbol][candidate.combo.key],
            locked_oos_start,
            config,
            cost_model,
            fdr_discovery=candidate.name in fdr_names,
        )
        for candidate in candidates
    )
    raw_is_survivors = sum(
        1 for score in is_scores if statistics.fmean(score.fold_excess_returns) > 0
    )
    go_count = sum(1 for scorecard in scorecards if scorecard.verdict == "GO_CANDIDATE")
    insufficient_count = sum(1 for scorecard in scorecards if scorecard.verdict == "INSUFFICIENT")
    verdict, reason = _portfolio_verdict(go_count, insufficient_count, len(scorecards))
    return ComboScorecardReport(
        status="OK",
        verdict=verdict,
        reason=reason,
        symbols=symbols,
        combo_count=len(combos),
        candidate_count_n=len(candidates),
        locked_oos_start=locked_oos_start,
        raw_is_survivors=raw_is_survivors,
        fdr_is_survivors=len(fdr_names),
        go_candidates=go_count,
        insufficient_candidates=insufficient_count,
        scorecards={
            scorecard.candidate.name: scorecard_to_dict(scorecard) for scorecard in scorecards
        },
        benchmark_scorecards=_benchmark_scorecards(
            bars_by_symbol, locked_oos_start, config, cost_model
        ),
        multiple_testing={
            "method": "Benjamini-Hochberg FDR over all predeclared symbol-combo trials",
            "alpha": config.fdr_alpha,
            "trial_count_n": len(candidates),
            "raw_is_survivors": raw_is_survivors,
            "fdr_is_survivors": len(fdr_names),
            "min_p_value": min(score.p_value for score in is_scores),
        },
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def simulate_combo(
    bars: Sequence[ComboBar],
    combo: PredeclaredCombo,
    *,
    start: int,
    end: int,
    config: ScorecardConfig,
    cost_model: ComboCostModel,
) -> ScorecardSimulation:
    returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    trade_returns: list[float] = []
    active_trade_equity = 1.0
    in_trade = False
    prev_position = 0
    turnover = 0.0
    start = max(start, combo.warmup + 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        position = 1 if combo.signal(bars, decision_index) else 0
        trade_size = abs(position - prev_position)
        trade_cost = trade_size * cost_model.one_way_cost
        funding_cost = abs(position) * (cost_model.funding_bps_per_period / 10_000.0)
        gross_return = position * (
            bars[execution_index + 1].open / bars[execution_index].open - 1.0
        )
        period_return = gross_return - trade_cost - funding_cost
        returns.append(period_return)
        positions.append(position)
        costs.append(trade_cost + funding_cost)
        turnover += trade_size
        if prev_position == 0 and position == 1:
            in_trade = True
            active_trade_equity = 1.0
        if in_trade:
            active_trade_equity *= 1.0 + period_return
        if prev_position == 1 and position == 0 and in_trade:
            trade_returns.append(active_trade_equity - 1.0)
            in_trade = False
            active_trade_equity = 1.0
        prev_position = position
    if positions and positions[-1] != 0:
        exit_cost = cost_model.one_way_cost
        turnover += 1.0
        costs[-1] += exit_cost
        returns[-1] -= exit_cost
        if in_trade:
            active_trade_equity *= 1.0 - exit_cost
            trade_returns.append(active_trade_equity - 1.0)
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return ScorecardSimulation(
        returns=tuple(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        trade_returns=tuple(trade_returns),
        turnover=turnover,
        metrics=metrics,
        first_execution_index=start,
    )


def simulate_combo_with_signals(
    bars: Sequence[ComboBar],
    combo: PredeclaredCombo,
    signals: Sequence[bool],
    *,
    start: int,
    end: int,
    config: ScorecardConfig,
    cost_model: ComboCostModel,
) -> ScorecardSimulation:
    returns: list[float] = []
    positions: list[int] = []
    costs: list[float] = []
    trade_returns: list[float] = []
    active_trade_equity = 1.0
    in_trade = False
    prev_position = 0
    turnover = 0.0
    start = max(start, combo.warmup + 1)
    end = min(end, len(bars) - 1)
    for execution_index in range(start, end):
        decision_index = execution_index - 1
        position = 1 if signals[decision_index] else 0
        trade_size = abs(position - prev_position)
        trade_cost = trade_size * cost_model.one_way_cost
        funding_cost = abs(position) * (cost_model.funding_bps_per_period / 10_000.0)
        gross_return = position * (
            bars[execution_index + 1].open / bars[execution_index].open - 1.0
        )
        period_return = gross_return - trade_cost - funding_cost
        returns.append(period_return)
        positions.append(position)
        costs.append(trade_cost + funding_cost)
        turnover += trade_size
        if prev_position == 0 and position == 1:
            in_trade = True
            active_trade_equity = 1.0
        if in_trade:
            active_trade_equity *= 1.0 + period_return
        if prev_position == 1 and position == 0 and in_trade:
            trade_returns.append(active_trade_equity - 1.0)
            in_trade = False
            active_trade_equity = 1.0
        prev_position = position
    if positions and positions[-1] != 0:
        exit_cost = cost_model.one_way_cost
        turnover += 1.0
        costs[-1] += exit_cost
        returns[-1] -= exit_cost
        if in_trade:
            active_trade_equity *= 1.0 - exit_cost
            trade_returns.append(active_trade_equity - 1.0)
    metrics = metrics_from_returns(
        returns,
        annualization_periods=config.annualization_periods,
        turnover=turnover,
        net_cost=sum(costs),
    )
    return ScorecardSimulation(
        returns=tuple(returns),
        positions=tuple(positions),
        costs=tuple(costs),
        trade_returns=tuple(trade_returns),
        turnover=turnover,
        metrics=metrics,
        first_execution_index=start,
    )


def composite_score(trade: TradeScorecard, metrics: ComboMetrics, excess_return: float) -> float:
    expectancy_points = _clip01((trade.expectancy_per_trade + 0.02) / 0.08) * 25.0
    profit_factor_points = _clip01((min(trade.profit_factor, 3.0) - 1.0) / 2.0) * 20.0
    sharpe_points = _clip01((metrics.sharpe + 1.0) / 3.0) * 20.0
    drawdown_points = _clip01(1.0 + metrics.max_drawdown) * 15.0
    excess_points = _clip01((excess_return + 0.25) / 0.75) * 10.0
    trade_count_points = _clip01(trade.total_trades / 20.0) * 10.0
    return round(
        expectancy_points
        + profit_factor_points
        + sharpe_points
        + drawdown_points
        + excess_points
        + trade_count_points,
        4,
    )


def scorecard_to_dict(scorecard: CandidateScorecard) -> dict[str, object]:
    return {
        "combo_key": scorecard.candidate.combo.key,
        "symbol": scorecard.candidate.symbol,
        "thesis": scorecard.candidate.combo.thesis,
        "params": scorecard.candidate.combo.params,
        "trade": trade_scorecard_to_dict(scorecard.trade),
        "metrics": metrics_to_dict(scorecard.metrics),
        "buy_hold_metrics": metrics_to_dict(scorecard.buy_hold_metrics),
        "composite_score": scorecard.composite_score,
        "excess_return": scorecard.excess_return,
        "beats_buy_hold_return": scorecard.beats_buy_hold_return,
        "beats_buy_hold_sharpe": scorecard.beats_buy_hold_sharpe,
        "oos_window_win_rate": scorecard.oos_window_win_rate,
        "gate_checks": scorecard.gate_checks,
        "verdict": scorecard.verdict,
        "reason": scorecard.reason,
    }


def report_to_dict(report: ComboScorecardReport) -> dict[str, object]:
    return {
        "status": report.status,
        "verdict": report.verdict,
        "reason": report.reason,
        "symbols": list(report.symbols),
        "combo_count": report.combo_count,
        "candidate_count_n": report.candidate_count_n,
        "locked_oos_start": report.locked_oos_start,
        "raw_is_survivors": report.raw_is_survivors,
        "fdr_is_survivors": report.fdr_is_survivors,
        "go_candidates": report.go_candidates,
        "insufficient_candidates": report.insufficient_candidates,
        "scorecards": report.scorecards,
        "benchmark_scorecards": report.benchmark_scorecards,
        "multiple_testing": report.multiple_testing,
        "safety": report.safety,
    }


def _evaluate_is(
    candidate: ScorecardCandidate,
    bars: Sequence[ComboBar],
    signals: Sequence[bool],
    locked_oos_start: int,
    config: ScorecardConfig,
    cost_model: ComboCostModel,
) -> ISScore | None:
    fold_excess: list[float] = []
    selector_max = -1
    starts = range(0, locked_oos_start - config.train_bars - config.test_bars + 1, config.step_bars)
    for train_start in starts:
        train_end = train_start + config.train_bars
        test_start = train_end
        test_end = test_start + config.test_bars
        selector_max = max(selector_max, train_end - 1)
        strategy = simulate_combo_with_signals(
            bars,
            candidate.combo,
            signals,
            start=test_start,
            end=test_end,
            config=config,
            cost_model=cost_model,
        )
        buy_hold = buy_hold_simulation(
            bars,
            start=test_start,
            end=test_end,
            config=_search_like_config(config),
            cost_model=cost_model,
        )
        fold_excess.append(strategy.metrics.total_return - buy_hold.metrics.total_return)
    if len(fold_excess) < config.min_is_folds:
        return None
    return ISScore(
        candidate=candidate,
        fold_excess_returns=tuple(fold_excess),
        p_value=sign_test_p_value(fold_excess),
        selector_max_index=selector_max,
        first_oos_execution_index=locked_oos_start,
    )


def _locked_oos_scorecard(
    candidate: ScorecardCandidate,
    bars: Sequence[ComboBar],
    signals: Sequence[bool] | None,
    locked_oos_start: int,
    config: ScorecardConfig,
    cost_model: ComboCostModel,
    *,
    fdr_discovery: bool,
) -> CandidateScorecard:
    if signals is None:
        strategy = simulate_combo(
            bars,
            candidate.combo,
            start=locked_oos_start,
            end=len(bars) - 1,
            config=config,
            cost_model=cost_model,
        )
    else:
        strategy = simulate_combo_with_signals(
            bars,
            candidate.combo,
            signals,
            start=locked_oos_start,
            end=len(bars) - 1,
            config=config,
            cost_model=cost_model,
        )
    buy_hold = buy_hold_simulation(
        bars,
        start=locked_oos_start,
        end=len(bars) - 1,
        config=_search_like_config(config),
        cost_model=cost_model,
    )
    trade = trade_scorecard(strategy.trade_returns)
    excess_return = strategy.metrics.total_return - buy_hold.metrics.total_return
    beats_return = strategy.metrics.total_return > buy_hold.metrics.total_return
    beats_sharpe = strategy.metrics.sharpe > buy_hold.metrics.sharpe
    gate_checks = {
        "expectancy_positive": trade.expectancy_per_trade > 0.0,
        "profit_factor_gt_1_3": trade.profit_factor > config.profit_factor_threshold,
        "beats_buy_hold_return": beats_return,
        "beats_buy_hold_sharpe": beats_sharpe,
        "fdr_discovery": fdr_discovery,
        "min_trades": trade.total_trades >= config.min_trades,
    }
    verdict, reason = _candidate_verdict(gate_checks, trade.total_trades, config)
    metrics = metrics_from_returns(
        strategy.returns,
        annualization_periods=config.annualization_periods,
        turnover=strategy.turnover,
        net_cost=sum(strategy.costs),
        oos_vs_buy_hold_window_win_rate=1.0 if beats_return else 0.0,
    )
    return CandidateScorecard(
        candidate=candidate,
        trade=trade,
        metrics=metrics,
        buy_hold_metrics=buy_hold.metrics,
        composite_score=composite_score(trade, metrics, excess_return),
        excess_return=excess_return,
        beats_buy_hold_return=beats_return,
        beats_buy_hold_sharpe=beats_sharpe,
        oos_window_win_rate=1.0 if beats_return else 0.0,
        gate_checks=gate_checks,
        verdict=verdict,
        reason=reason,
    )


def _candidate_verdict(
    gate_checks: dict[str, bool], total_trades: int, config: ScorecardConfig
) -> tuple[str, str]:
    if total_trades < config.min_trades:
        return "INSUFFICIENT", f"locked OOS has {total_trades} trades; need {config.min_trades}"
    failed = [name for name, passed in gate_checks.items() if not passed]
    if failed:
        return "NO_GO", "failed gates: " + ", ".join(failed)
    return "GO_CANDIDATE", "passed all hard gates; still requires independent OOS review"


def _portfolio_verdict(go_count: int, insufficient_count: int, total_count: int) -> tuple[str, str]:
    if go_count > 0:
        return (
            "GO_CANDIDATE",
            "at least one predeclared combo passed all hard gates; independent OOS required",
        )
    if insufficient_count == total_count:
        return "INSUFFICIENT", "all candidates lacked the minimum locked-OOS trade sample"
    return "NO_GO", "no predeclared combo passed all hard gates"


def _benchmark_scorecards(
    bars_by_symbol: dict[str, Sequence[ComboBar]],
    locked_oos_start: int,
    config: ScorecardConfig,
    cost_model: ComboCostModel,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for symbol, bars in bars_by_symbol.items():
        benchmark = buy_hold_simulation(
            bars,
            start=locked_oos_start,
            end=len(bars) - 1,
            config=_search_like_config(config),
            cost_model=cost_model,
        )
        result[f"{symbol}::buy_hold"] = {
            "trade": trade_scorecard_to_dict(trade_scorecard((benchmark.metrics.total_return,))),
            "metrics": metrics_to_dict(benchmark.metrics),
        }
    return result


def _combo_signal_cache(
    bars: Sequence[ComboBar], combos: Sequence[PredeclaredCombo]
) -> dict[str, tuple[bool, ...]]:
    closes = [bar.close for bar in bars]
    sma50 = _rolling_mean(closes, 50)
    sma200 = _rolling_mean(closes, 200)
    rsi14 = [_rsi(bars, index, 14) for index in range(len(bars))]
    rv60_low = _realized_vol_low_series(bars, 60)
    donchian20 = _donchian_breakout_series(bars, 20)
    macd = _macd_histogram_series(closes, 12, 26, 9)
    adx14 = [_adx(bars, index, 14) for index in range(len(bars))]
    bollinger20_lower = _bollinger_lower_series(closes, 20, 2.0)
    tsmom30 = _tsmom_series(closes, 30)
    tsmom90 = _tsmom_series(closes, 90)
    roc20 = [_roc(bars, index, 20) for index in range(len(bars))]
    obv20 = _obv_slope_series(bars, 20)
    series = {
        "trend_pullback_ma200_rsi14_30": tuple(
            _optional_close_gt(closes[index], sma200[index]) and rsi14[index] < 30
            for index in range(len(bars))
        ),
        "golden_cross_low_vol_50_200_rv60": tuple(
            _optional_gt(sma50[index], sma200[index]) and rv60_low[index]
            for index in range(len(bars))
        ),
        "donchian20_breakout_ma200": tuple(
            donchian20[index] and _optional_close_gt(closes[index], sma200[index])
            for index in range(len(bars))
        ),
        "macd_adx_trend_strength_12_26_9_14_25": tuple(
            macd[index] > 0 and adx14[index] > 25 for index in range(len(bars))
        ),
        "bollinger_lower_low_vol_20_2_rv60": tuple(
            bollinger20_lower[index] and rv60_low[index] for index in range(len(bars))
        ),
        "tsmom30_low_vol_rv60": tuple(
            tsmom30[index] and rv60_low[index] for index in range(len(bars))
        ),
        "tsmom90_low_vol_rv60": tuple(
            tsmom90[index] and rv60_low[index] for index in range(len(bars))
        ),
        "roc20_obv_volume_confirmation": tuple(
            roc20[index] > 0 and obv20[index] > 0 for index in range(len(bars))
        ),
    }
    return {combo.key: series[combo.key] for combo in combos}


def _search_like_config(config: ScorecardConfig) -> ComboSearchConfig:
    return ComboSearchConfig(annualization_periods=config.annualization_periods)


def _insufficient_report(
    reason: str,
    symbols: tuple[str, ...],
    config: ScorecardConfig,
    cost_model: ComboCostModel,
    *,
    combo_count: int = 0,
    candidate_count_n: int = 0,
    locked_oos_start: int = 0,
) -> ComboScorecardReport:
    return ComboScorecardReport(
        status="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=reason,
        symbols=symbols,
        combo_count=combo_count,
        candidate_count_n=candidate_count_n,
        locked_oos_start=locked_oos_start,
        raw_is_survivors=0,
        fdr_is_survivors=0,
        go_candidates=0,
        insufficient_candidates=0,
        scorecards={},
        benchmark_scorecards={},
        multiple_testing={"method": "Benjamini-Hochberg FDR", "trial_count_n": candidate_count_n},
        safety={
            "paper_only": True,
            "live_trading": False,
            "strategy_plugin_registered": False,
            "order_path_added": False,
            "funding": cost_model.funding_label,
        },
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _closes(bars: Sequence[ComboBar]) -> list[float]:
    return [bar.close for bar in bars]


def _sma(values: Sequence[float], index: int, period: int) -> float:
    if index + 1 < period:
        return math.inf
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


def _optional_gt(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _optional_close_gt(close: float, threshold: float | None) -> bool:
    return threshold is not None and close > threshold


def _close_gt_sma(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    closes = _closes(bars)
    return closes[index] > _sma(closes, index, period)


def _sma_gt_sma(bars: Sequence[ComboBar], index: int, fast: int, slow: int) -> bool:
    closes = _closes(bars)
    return _sma(closes, index, fast) > _sma(closes, index, slow)


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


def _tsmom_positive(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    return index >= period and bars[index].close > bars[index - period].close


def _donchian_breakout(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    if index < period:
        return False
    prior_high = max(bar.high for bar in bars[index - period : index])
    return bars[index].close > prior_high


def _ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _macd_histogram_series(
    closes: Sequence[float], fast: int, slow: int, signal: int
) -> list[float]:
    if not closes:
        return []
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd = [left - right for left, right in zip(fast_ema, slow_ema, strict=True)]
    signal_line = _ema(macd, signal)
    return [
        macd[index] - signal_line[index] if index >= slow + signal else 0.0
        for index in range(len(closes))
    ]


def _macd_histogram(
    bars: Sequence[ComboBar], index: int, fast: int, slow: int, signal: int
) -> float:
    if index < slow + signal:
        return 0.0
    closes = _closes(bars[: index + 1])
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd = [left - right for left, right in zip(fast_ema, slow_ema, strict=True)]
    signal_line = _ema(macd, signal)
    return macd[-1] - signal_line[-1]


def _true_range(bars: Sequence[ComboBar], index: int) -> float:
    if index == 0:
        return bars[index].high - bars[index].low
    return max(
        bars[index].high - bars[index].low,
        abs(bars[index].high - bars[index - 1].close),
        abs(bars[index].low - bars[index - 1].close),
    )


def _adx(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period * 2:
        return 0.0
    dx_values: list[float] = []
    for current in range(index - period + 1, index + 1):
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        tr_sum = 0.0
        for inner in range(current - period + 1, current + 1):
            up_move = bars[inner].high - bars[inner - 1].high
            down_move = bars[inner - 1].low - bars[inner].low
            plus_dm_sum += up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm_sum += down_move if down_move > up_move and down_move > 0 else 0.0
            tr_sum += _true_range(bars, inner)
        if tr_sum == 0:
            dx_values.append(0.0)
            continue
        plus_di = 100.0 * plus_dm_sum / tr_sum
        minus_di = 100.0 * minus_dm_sum / tr_sum
        denominator = plus_di + minus_di
        dx_values.append(100.0 * abs(plus_di - minus_di) / denominator if denominator else 0.0)
    return statistics.fmean(dx_values)


def _realized_vol(bars: Sequence[ComboBar], index: int, period: int) -> float:
    if index < period:
        return math.inf
    returns = [
        bars[current].close / bars[current - 1].close - 1.0
        for current in range(index - period + 1, index + 1)
    ]
    return statistics.pstdev(returns) if len(returns) > 1 else 0.0


def _realized_vol_below_median(bars: Sequence[ComboBar], index: int, period: int) -> bool:
    median_period = period * 3
    if index + 1 < period + median_period:
        return False
    current_vol = _realized_vol(bars, index, period)
    history = [
        _realized_vol(bars, current, period)
        for current in range(index - median_period + 1, index + 1)
    ]
    return current_vol <= statistics.median(history)


def _realized_vol_low_series(bars: Sequence[ComboBar], period: int) -> list[bool]:
    vols = [_realized_vol(bars, index, period) for index in range(len(bars))]
    median_period = period * 3
    result: list[bool] = []
    for index, value in enumerate(vols):
        if index + 1 < period + median_period:
            result.append(False)
            continue
        history = vols[index - median_period + 1 : index + 1]
        result.append(value <= statistics.median(history))
    return result


def _below_bollinger_lower(
    bars: Sequence[ComboBar], index: int, period: int, stdevs: float
) -> bool:
    if index + 1 < period:
        return False
    closes = [bar.close for bar in bars[index - period + 1 : index + 1]]
    mean = statistics.fmean(closes)
    lower = mean - stdevs * statistics.pstdev(closes)
    return bars[index].close < lower


def _bollinger_lower_series(
    closes: Sequence[float], period: int, stdevs: float
) -> list[bool]:
    result: list[bool] = [False] * len(closes)
    for index in range(period - 1, len(closes)):
        window = closes[index - period + 1 : index + 1]
        mean = statistics.fmean(window)
        lower = mean - stdevs * statistics.pstdev(window)
        result[index] = closes[index] < lower
    return result


def _tsmom_series(closes: Sequence[float], period: int) -> list[bool]:
    return [
        closes[index] > closes[index - period] if index >= period else False
        for index in range(len(closes))
    ]


def _donchian_breakout_series(bars: Sequence[ComboBar], period: int) -> list[bool]:
    result: list[bool] = [False] * len(bars)
    for index in range(period, len(bars)):
        prior_high = max(bar.high for bar in bars[index - period : index])
        result[index] = bars[index].close > prior_high
    return result


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


def _obv_slope_series(bars: Sequence[ComboBar], period: int) -> list[float]:
    values = [0.0] * len(bars)
    obv = [0.0] * len(bars)
    for index in range(1, len(bars)):
        if bars[index].close > bars[index - 1].close:
            obv[index] = obv[index - 1] + bars[index].volume
        elif bars[index].close < bars[index - 1].close:
            obv[index] = obv[index - 1] - bars[index].volume
        else:
            obv[index] = obv[index - 1]
        if index >= period:
            values[index] = obv[index] - obv[index - period]
    return values
