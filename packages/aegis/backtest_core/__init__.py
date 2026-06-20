from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

BhTiePolicy = Literal["p_value_cutoff", "rank"]
SignAlternative = Literal["greater", "two-sided"]


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
