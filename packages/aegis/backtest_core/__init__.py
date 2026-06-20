from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Literal

BhTiePolicy = Literal["p_value_cutoff", "rank"]
SignAlternative = Literal["greater", "two-sided"]
HypothesisType = Literal[
    "factor",
    "combo",
    "carry",
    "event",
    "momentum",
    "risk",
    "price_action",
    "other",
]
StandardVerdictState = Literal["EDGE", "NO_EDGE", "INSUFFICIENT"]


POSITIVE_VERDICTS = frozenset(
    {
        "EDGE",
        "EDGE_CANDIDATE",
        "GO",
        "ROBUST",
        "ROBUST_CARRY",
        "RISK_IMPROVED",
        "AGENT_EDGE",
        "SUGGESTIVE_NEEDS_PAID_CONFIRM",
    }
)
INSUFFICIENT_VERDICTS = frozenset({"INSUFFICIENT", "INSUFFICIENT_DATA"})


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    funding_bps_per_period: float = 0.0
    funding_label: str = "N/A for spot long-only; perp funding not used"

    @property
    def one_way_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0

    @property
    def round_trip_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass(frozen=True)
class ReturnMetrics:
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
class TradeScorecard:
    total_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    win_loss_ratio: float
    expectancy_per_trade: float
    profit_factor: float
    max_consecutive_losses: int


@dataclass(frozen=True)
class BacktestDiscipline:
    t_plus_1_execution: bool = True
    locked_oos: bool = True
    walk_forward: bool = True
    full_costs: bool = True
    multiple_testing: bool = True
    survivor_ceiling: bool = False


@dataclass(frozen=True)
class StandardVerdict:
    state: StandardVerdictState
    verdict: str
    reason: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    benchmarks: Mapping[str, object] = field(default_factory=dict)
    candidate_count_n: int = 0
    raw_survivors: int = 0
    fdr_survivors: int = 0
    multiple_testing: Mapping[str, object] = field(default_factory=dict)
    safety: Mapping[str, object] = field(default_factory=dict)
    survivor_ceiling_applied: bool = False


@dataclass(frozen=True)
class HypothesisSpec:
    key: str
    hypothesis_type: HypothesisType
    universe: tuple[str, ...]
    predeclared_signals: tuple[str, ...]
    params: Mapping[str, object]
    cost_model: object
    benchmark: str
    data_source: str
    trial_count_n: int
    discipline: BacktestDiscipline = BacktestDiscipline()
    survivor_light: bool = False
    runner: Callable[[], object] | None = field(default=None, compare=False, repr=False)
    verdict_adapter: Callable[[object, HypothesisSpec], StandardVerdict] | None = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class BacktestRun:
    spec: HypothesisSpec
    verdict: StandardVerdict
    payload: object


def run_backtest(spec: HypothesisSpec) -> BacktestRun:
    _validate_hypothesis_spec(spec)
    if spec.runner is None:
        raise ValueError(f"hypothesis spec {spec.key!r} has no runner")
    payload = spec.runner()
    adapter = spec.verdict_adapter or default_verdict_adapter
    verdict = adapter(payload, spec)
    verdict = _normalize_standard_verdict(verdict, spec)
    return BacktestRun(spec=spec, verdict=verdict, payload=payload)


def default_verdict_adapter(payload: object, spec: HypothesisSpec) -> StandardVerdict:
    verdict = str(_payload_get(payload, "verdict", _payload_get(payload, "alpha_verdict", "OK")))
    status = str(_payload_get(payload, "status", "OK"))
    reason = str(_payload_get(payload, "reason", ""))
    multiple_testing = _mapping_or_empty(_payload_get(payload, "multiple_testing", {}))
    safety = _mapping_or_empty(_payload_get(payload, "safety", {}))
    metrics = _mapping_or_empty(
        _payload_get(payload, "standard_metrics", _payload_get(payload, "metrics", {}))
    )
    benchmarks = _mapping_or_empty(
        _payload_get(payload, "benchmark_metrics", _payload_get(payload, "benchmarks", {}))
    )
    candidate_count_n = _int_from_payload(
        payload,
        "candidate_count_n",
        _int_from_mapping(multiple_testing, "candidate_count_n", spec.trial_count_n),
    )
    raw_survivors = _int_from_payload(
        payload,
        "raw_is_survivors",
        _int_from_mapping(
            multiple_testing,
            "raw_survivors",
            _int_from_mapping(multiple_testing, "raw_is_survivors", 0),
        ),
    )
    fdr_survivors = _int_from_payload(
        payload,
        "fdr_is_survivors",
        _int_from_payload(
            payload,
            "alpha_fdr_survivors",
            _int_from_mapping(
                multiple_testing,
                "fdr_survivors",
                _int_from_mapping(multiple_testing, "alpha_fdr_survivors", 0),
            ),
        ),
    )
    return StandardVerdict(
        state=_standard_state(verdict, status),
        verdict=verdict,
        reason=reason,
        metrics=metrics,
        benchmarks=benchmarks,
        candidate_count_n=candidate_count_n,
        raw_survivors=raw_survivors,
        fdr_survivors=fdr_survivors,
        multiple_testing=multiple_testing,
        safety=safety,
        survivor_ceiling_applied=False,
    )


def _validate_hypothesis_spec(spec: HypothesisSpec) -> None:
    if not spec.key:
        raise ValueError("hypothesis spec key is required")
    if not spec.universe:
        raise ValueError(f"hypothesis spec {spec.key!r} must declare a non-empty universe")
    if spec.trial_count_n < 1:
        raise ValueError(f"hypothesis spec {spec.key!r} must count at least one trial")
    if not spec.benchmark:
        raise ValueError(f"hypothesis spec {spec.key!r} must declare a benchmark")
    if not spec.data_source:
        raise ValueError(f"hypothesis spec {spec.key!r} must declare a data source")
    missing = []
    if not spec.discipline.t_plus_1_execution:
        missing.append("t_plus_1_execution")
    if not spec.discipline.locked_oos:
        missing.append("locked_oos")
    if not spec.discipline.walk_forward:
        missing.append("walk_forward")
    if not spec.discipline.full_costs:
        missing.append("full_costs")
    if not spec.discipline.multiple_testing:
        missing.append("multiple_testing")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"hypothesis spec {spec.key!r} failed discipline checks: {joined}")


def _normalize_standard_verdict(verdict: StandardVerdict, spec: HypothesisSpec) -> StandardVerdict:
    candidate_count_n = max(verdict.candidate_count_n, spec.trial_count_n)
    multiple_testing = dict(verdict.multiple_testing)
    multiple_testing.setdefault("candidate_count_n", candidate_count_n)
    multiple_testing.setdefault("hypothesis_trial_count_n", spec.trial_count_n)
    multiple_testing.setdefault("scope", "all predeclared trials in HypothesisSpec")
    if spec.survivor_light and verdict.state == "EDGE" and verdict.verdict != (
        "SUGGESTIVE_NEEDS_PAID_CONFIRM"
    ):
        return replace(
            verdict,
            state="EDGE",
            verdict="SUGGESTIVE_NEEDS_PAID_CONFIRM",
            reason=(
                verdict.reason
                + "; survivor-light data ceiling caps positive verdict below ROBUST"
            ),
            candidate_count_n=candidate_count_n,
            multiple_testing=multiple_testing,
            survivor_ceiling_applied=True,
        )
    return replace(
        verdict,
        candidate_count_n=candidate_count_n,
        multiple_testing=multiple_testing,
        survivor_ceiling_applied=verdict.survivor_ceiling_applied or spec.survivor_light,
    )


def _standard_state(verdict: str, status: str) -> StandardVerdictState:
    if status in INSUFFICIENT_VERDICTS or verdict in INSUFFICIENT_VERDICTS:
        return "INSUFFICIENT"
    if verdict in POSITIVE_VERDICTS:
        return "EDGE"
    return "NO_EDGE"


def _payload_get(payload: object, key: str, default: object) -> object:
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _int_from_payload(payload: object, key: str, default: int) -> int:
    return _coerce_int(_payload_get(payload, key, default), default)


def _int_from_mapping(payload: Mapping[str, object], key: str, default: int) -> int:
    return _coerce_int(payload.get(key, default), default)


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def equity_curve(returns: Iterable[float], *, include_initial: bool = True) -> tuple[float, ...]:
    equity = 1.0
    curve = [equity] if include_initial else []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return tuple(curve)


def max_drawdown(equity: Sequence[float]) -> float:
    peak = 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak else 0.0
        worst = min(worst, drawdown)
    return worst


def metrics_from_returns(
    returns: Sequence[float],
    *,
    annualization_periods: int,
    turnover: float,
    net_cost: float,
    oos_vs_buy_hold_window_win_rate: float = 0.0,
    include_initial_equity: bool = False,
    nonpositive_annualized_return: float = -1.0,
) -> ReturnMetrics:
    values = tuple(float(value) for value in returns)
    if not values:
        return ReturnMetrics(
            annualized_return=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            positive_period_win_rate=0.0,
            oos_vs_buy_hold_window_win_rate=oos_vs_buy_hold_window_win_rate,
            annualized_turnover=0.0,
            net_cost=net_cost,
        )
    equity = equity_curve(values, include_initial=include_initial_equity)
    total_return = equity[-1] - 1.0
    years = max(len(values) / annualization_periods, 1 / annualization_periods)
    annualized_return = (
        equity[-1] ** (1.0 / years) - 1.0
        if equity[-1] > 0
        else nonpositive_annualized_return
    )
    drawdown = max_drawdown(equity)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    sharpe = (mean / stdev) * math.sqrt(annualization_periods) if stdev > 0 else 0.0
    downside = tuple(value for value in values if value < 0)
    downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    sortino = (mean / downside_dev) * math.sqrt(annualization_periods) if downside_dev > 0 else 0.0
    calmar = annualized_return / abs(drawdown) if drawdown < 0 else 0.0
    return ReturnMetrics(
        annualized_return=annualized_return,
        total_return=total_return,
        max_drawdown=drawdown,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        positive_period_win_rate=sum(1 for value in values if value > 0) / len(values),
        oos_vs_buy_hold_window_win_rate=oos_vs_buy_hold_window_win_rate,
        annualized_turnover=turnover / years,
        net_cost=net_cost,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.10,
    tie_policy: BhTiePolicy = "p_value_cutoff",
) -> list[bool]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    passed = [False for _ in p_values]
    m = len(indexed)
    max_rank = -1
    for rank, (_index, p_value) in enumerate(indexed, start=1):
        if p_value <= alpha * rank / m:
            max_rank = rank
    if max_rank < 1:
        return passed
    if tie_policy == "rank":
        for rank, (index, _p_value) in enumerate(indexed, start=1):
            if rank <= max_rank:
                passed[index] = True
        return passed
    cutoff = indexed[max_rank - 1][1]
    return [float(p_value) <= cutoff for p_value in p_values]


def sign_test_p_value(
    excess_returns: Sequence[float],
    *,
    alternative: SignAlternative = "greater",
) -> float:
    non_zero = [value for value in excess_returns if value != 0.0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in non_zero if value > 0)
    if alternative == "greater":
        tail = sum(math.comb(n, k) * 0.5**n for k in range(wins, n + 1))
        return min(1.0, float(tail))
    losses = n - wins
    tail_count = min(wins, losses)
    tail = float(sum(math.comb(n, k) for k in range(0, tail_count + 1)) / (2**n))
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 1_000,
    seed: int = 44,
    empty_value: float | None = None,
    include_iterations: bool = False,
) -> dict[str, float | int | None]:
    if not values:
        result: dict[str, float | int | None] = {
            "p05": empty_value,
            "p50": empty_value,
            "p95": empty_value,
        }
        if include_iterations:
            result["iterations"] = iterations
        return result
    rng = random.Random(seed)
    effective_iterations = max(1, iterations)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(effective_iterations)
    )
    result = {
        "p05": means[int(effective_iterations * 0.05)],
        "p50": means[int(effective_iterations * 0.50)],
        "p95": means[int(effective_iterations * 0.95)],
    }
    if include_iterations:
        result["iterations"] = iterations
    return result


def pbo(
    returns_or_trials: Sequence[Sequence[float]],
    *,
    n_splits: int = 16,
) -> dict[str, float | int | str | list[float] | list[int]]:
    """Estimate Probability of Backtest Overfitting via deterministic CSCV."""
    trials = _validate_pbo_trials(returns_or_trials, n_splits)
    trial_count = len(trials)
    observation_count = len(trials[0])
    split_ranges = _cscv_split_ranges(observation_count, n_splits)
    split_ids = tuple(range(n_splits))
    half = n_splits // 2
    logits: list[float] = []
    oos_rank_percentiles: list[float] = []
    selected_trial_indices: list[int] = []
    for train_splits in combinations(split_ids, half):
        train_set = frozenset(train_splits)
        test_splits = tuple(split for split in split_ids if split not in train_set)
        train_indices = _indices_for_splits(split_ranges, train_splits)
        test_indices = _indices_for_splits(split_ranges, test_splits)
        train_scores = [_sharpe_for_indices(trial, train_indices) for trial in trials]
        selected_index = max(range(trial_count), key=lambda index: (train_scores[index], -index))
        test_scores = [_sharpe_for_indices(trial, test_indices) for trial in trials]
        percentile = _oos_rank_percentile(test_scores, selected_index)
        selected_trial_indices.append(selected_index)
        oos_rank_percentiles.append(percentile)
        logits.append(_logit(percentile))
    pbo_value = sum(1 for value in logits if value < 0.0) / len(logits)
    return {
        "method": "CSCV_PBO",
        "pbo": pbo_value,
        "n_splits": n_splits,
        "split_count": len(logits),
        "trial_count": trial_count,
        "observation_count": observation_count,
        "logits": logits,
        "oos_rank_percentiles": oos_rank_percentiles,
        "selected_trial_indices": selected_trial_indices,
        "dsr_sharpe_threshold": deflated_sharpe_threshold(
            trial_count=trial_count, observations=observation_count
        ),
    }


def _validate_pbo_trials(
    returns_or_trials: Sequence[Sequence[float]], n_splits: int
) -> tuple[tuple[float, ...], ...]:
    if n_splits < 4:
        raise ValueError("n_splits must be at least 4")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even for CSCV")
    if len(returns_or_trials) < 2:
        raise ValueError("PBO requires at least two trials")
    trials = tuple(tuple(float(value) for value in trial) for trial in returns_or_trials)
    observation_count = len(trials[0]) if trials else 0
    if observation_count < n_splits:
        raise ValueError("each trial must have at least n_splits observations")
    for trial in trials:
        if len(trial) != observation_count:
            raise ValueError("all PBO trials must have the same observation count")
        if any(not math.isfinite(value) for value in trial):
            raise ValueError("PBO trial returns must be finite")
    return trials


def _cscv_split_ranges(observation_count: int, n_splits: int) -> tuple[range, ...]:
    base_size = observation_count // n_splits
    remainder = observation_count % n_splits
    ranges: list[range] = []
    start = 0
    for split in range(n_splits):
        size = base_size + (1 if split < remainder else 0)
        stop = start + size
        ranges.append(range(start, stop))
        start = stop
    return tuple(ranges)


def _indices_for_splits(split_ranges: Sequence[range], split_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(index for split_id in split_ids for index in split_ranges[split_id])


def _sharpe_for_indices(values: Sequence[float], indices: Sequence[int]) -> float:
    selected = tuple(values[index] for index in indices)
    if not selected:
        return 0.0
    mean = statistics.fmean(selected)
    stdev = statistics.pstdev(selected) if len(selected) > 1 else 0.0
    if stdev == 0.0:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return 0.0
    return mean / stdev


def _oos_rank_percentile(scores: Sequence[float], selected_index: int) -> float:
    selected_score = scores[selected_index]
    better_count = sum(1 for score in scores if score > selected_score)
    equal_count = sum(1 for score in scores if score == selected_score)
    average_descending_rank = better_count + (equal_count + 1) / 2.0
    rank_from_worst = len(scores) + 1.0 - average_descending_rank
    return rank_from_worst / (len(scores) + 1.0)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


def trade_scorecard(trade_returns: Sequence[float]) -> TradeScorecard:
    trades = tuple(float(value) for value in trade_returns)
    if not trades:
        return TradeScorecard(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    wins = tuple(value for value in trades if value > 0)
    losses = tuple(value for value in trades if value <= 0)
    total_gain = sum(wins)
    total_loss = abs(sum(losses))
    average_win = statistics.fmean(wins) if wins else 0.0
    average_loss = statistics.fmean(losses) if losses else 0.0
    win_rate = len(wins) / len(trades)
    win_loss_ratio = average_win / abs(average_loss) if average_loss < 0 else 0.0
    expectancy = win_rate * average_win - (1.0 - win_rate) * abs(average_loss)
    profit_factor = (
        total_gain / total_loss if total_loss > 0 else (math.inf if total_gain > 0 else 0.0)
    )
    return TradeScorecard(
        total_trades=len(trades),
        win_rate=win_rate,
        average_win=average_win,
        average_loss=average_loss,
        win_loss_ratio=win_loss_ratio,
        expectancy_per_trade=expectancy,
        profit_factor=profit_factor,
        max_consecutive_losses=max_consecutive_losses(trades),
    )


def trade_scorecard_to_dict(scorecard: TradeScorecard) -> dict[str, float | int]:
    return {
        "total_trades": scorecard.total_trades,
        "win_rate": scorecard.win_rate,
        "average_win": scorecard.average_win,
        "average_loss": scorecard.average_loss,
        "win_loss_ratio": scorecard.win_loss_ratio,
        "expectancy_per_trade": scorecard.expectancy_per_trade,
        "profit_factor": scorecard.profit_factor,
        "max_consecutive_losses": scorecard.max_consecutive_losses,
    }


def max_consecutive_losses(trades: Sequence[float]) -> int:
    worst = 0
    current = 0
    for value in trades:
        if value <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def normal_two_sided_p(t_value: float) -> float:
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(math.floor(q * (len(ordered) - 1))), 0), len(ordered) - 1)
    return ordered[index]


def nonpositive_share(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value <= 0) / len(values)


def drawdown_reduction(strategy_max_drawdown: float, benchmark_max_drawdown: float) -> float:
    benchmark_dd = abs(benchmark_max_drawdown)
    if benchmark_dd == 0:
        return 0.0
    return (benchmark_dd - abs(strategy_max_drawdown)) / benchmark_dd


def risk_difference_metrics(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    strategy_target_volatility: float,
    benchmark_target_volatility: float,
    annualization_periods: int,
) -> dict[str, float]:
    strategy = risk_return_metrics(
        strategy_returns,
        annualization_periods=annualization_periods,
        target_volatility=strategy_target_volatility,
        turnover=0.0,
        net_cost=0.0,
    )
    benchmark = risk_return_metrics(
        benchmark_returns,
        annualization_periods=annualization_periods,
        target_volatility=benchmark_target_volatility,
        turnover=0.0,
        net_cost=0.0,
    )
    return {
        "drawdown_reduction": drawdown_reduction(
            strategy["max_drawdown"], benchmark["max_drawdown"]
        ),
        "calmar_diff": strategy["calmar"] - benchmark["calmar"],
        "sortino_diff": strategy["sortino"] - benchmark["sortino"],
    }


def risk_return_metrics(
    returns: Sequence[float],
    *,
    annualization_periods: int,
    target_volatility: float,
    turnover: float,
    net_cost: float,
) -> dict[str, float]:
    values = tuple(float(value) for value in returns)
    if not values:
        return {
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sortino": 0.0,
            "sharpe": 0.0,
            "realized_volatility": 0.0,
            "target_volatility": target_volatility,
            "annualized_turnover": 0.0,
            "net_cost": net_cost,
            "worst_month": 0.0,
            "ulcer_index": 0.0,
        }
    equity = equity_curve(values, include_initial=False)
    years = max(len(values) / annualization_periods, 1 / annualization_periods)
    annualized_return = equity[-1] ** (1.0 / years) - 1.0 if equity[-1] > 0 else -1.0
    drawdown = max_drawdown(equity)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    realized_vol = stdev * math.sqrt(annualization_periods)
    sharpe = (mean / stdev) * math.sqrt(annualization_periods) if stdev > 0 else 0.0
    downside = tuple(value for value in values if value < 0)
    downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    sortino = (mean / downside_dev) * math.sqrt(annualization_periods) if downside_dev > 0 else 0.0
    calmar = annualized_return / abs(drawdown) if drawdown < 0 else 0.0
    return {
        "annualized_return": annualized_return,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "sortino": sortino,
        "sharpe": sharpe,
        "realized_volatility": realized_vol,
        "target_volatility": target_volatility,
        "annualized_turnover": turnover / years,
        "net_cost": net_cost,
        "worst_month": worst_month(values),
        "ulcer_index": ulcer_index(equity),
    }


def paired_block_bootstrap_risk_difference_test(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    strategy_target_volatility: float,
    benchmark_target_volatility: float,
    config: Any,
    candidate_key: str,
) -> dict[str, float | int | bool | str]:
    n = min(len(strategy_returns), len(benchmark_returns))
    block_bars = int(config.risk_diff_bootstrap_block_bars)
    if n < max(30, block_bars):
        return {
            "valid": False,
            "method": "paired_block_bootstrap",
            "reason": "insufficient locked-OOS paired returns for risk-difference bootstrap",
            "sample_count": 0,
            "p_value": 1.0,
            "ci_lower_gt_0": False,
        }
    strategy = tuple(float(value) for value in strategy_returns[:n])
    benchmark = tuple(float(value) for value in benchmark_returns[:n])
    annualization_periods = int(config.annualization_periods)
    observed = risk_difference_metrics(
        strategy,
        benchmark,
        strategy_target_volatility=strategy_target_volatility,
        benchmark_target_volatility=benchmark_target_volatility,
        annualization_periods=annualization_periods,
    )
    rng = random.Random(
        int(config.risk_diff_random_seed) + sum(ord(char) for char in candidate_key)
    )
    sampled_drawdown: list[float] = []
    sampled_calmar: list[float] = []
    sampled_sortino: list[float] = []
    block = min(block_bars, n)
    for _ in range(int(config.risk_diff_bootstrap_samples)):
        indices: list[int] = []
        while len(indices) < n:
            start = rng.randint(0, n - block)
            indices.extend(range(start, start + block))
        indices = indices[:n]
        sample_strategy = tuple(strategy[index] for index in indices)
        sample_benchmark = tuple(benchmark[index] for index in indices)
        sample = risk_difference_metrics(
            sample_strategy,
            sample_benchmark,
            strategy_target_volatility=strategy_target_volatility,
            benchmark_target_volatility=benchmark_target_volatility,
            annualization_periods=annualization_periods,
        )
        sampled_drawdown.append(sample["drawdown_reduction"])
        sampled_calmar.append(sample["calmar_diff"])
        sampled_sortino.append(sample["sortino_diff"])
    ci_alpha = float(config.risk_diff_ci_alpha)
    drawdown_ci_low = quantile(sampled_drawdown, ci_alpha)
    calmar_ci_low = quantile(sampled_calmar, ci_alpha)
    sortino_ci_low = quantile(sampled_sortino, ci_alpha)
    drawdown_tail = nonpositive_share(sampled_drawdown)
    calmar_tail = nonpositive_share(sampled_calmar)
    sortino_tail = nonpositive_share(sampled_sortino)
    p_value = (
        max(drawdown_tail, calmar_tail, sortino_tail)
        if (
            observed["drawdown_reduction"] > 0
            and observed["calmar_diff"] > 0
            and observed["sortino_diff"] > 0
        )
        else 1.0
    )
    ci_lower_gt_0 = drawdown_ci_low > 0 and calmar_ci_low > 0 and sortino_ci_low > 0
    return {
        "valid": True,
        "method": "paired_block_bootstrap",
        "sample_count": int(config.risk_diff_bootstrap_samples),
        "block_bars": block,
        "ci_alpha": ci_alpha,
        "p_value": p_value,
        "ci_lower_gt_0": ci_lower_gt_0,
        "drawdown_reduction": observed["drawdown_reduction"],
        "drawdown_reduction_ci_low": drawdown_ci_low,
        "calmar_diff": observed["calmar_diff"],
        "calmar_diff_ci_low": calmar_ci_low,
        "sortino_diff": observed["sortino_diff"],
        "sortino_diff_ci_low": sortino_ci_low,
    }


def worst_month(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    return min(
        (math.prod(1.0 + value for value in returns[index : index + 30]) - 1.0)
        for index in range(0, len(returns), 30)
    )


def ulcer_index(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = 1.0
    squares: list[float] = []
    for value in equity:
        peak = max(peak, value)
        drawdown_pct = (value / peak - 1.0) * 100.0 if peak else 0.0
        squares.append(drawdown_pct * drawdown_pct)
    return math.sqrt(statistics.fmean(squares)) / 100.0 if squares else 0.0


def deflated_sharpe_threshold(
    *,
    trial_count: int,
    observations: int,
    base_threshold: float = 0.0,
) -> float:
    if trial_count <= 1 or observations <= 1:
        return base_threshold
    return base_threshold + math.sqrt(2.0 * math.log(float(trial_count)) / observations)


def locked_oos_start_index(total_count: int, locked_oos_fraction: float) -> int:
    return int(total_count * (1.0 - locked_oos_fraction))


def survivor_light_verdict(
    *,
    fdr_discovery_count: int,
    insufficient: bool,
    positive_verdict: str,
    no_edge_verdict: str,
    insufficient_verdict: str = "INSUFFICIENT",
) -> str:
    if insufficient:
        return insufficient_verdict
    if fdr_discovery_count > 0:
        return positive_verdict
    return no_edge_verdict
