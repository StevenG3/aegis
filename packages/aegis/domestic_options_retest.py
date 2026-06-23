from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from aegis.backtest_core import (
    CostModel,
    StandardVerdict,
    benjamini_hochberg,
    metrics_from_returns,
    pbo,
    sign_test_p_value,
    trade_scorecard,
)

Mechanism = Literal["commodity_adx_option_buyer", "mo_im_delta_neutral"]
RequirementKey = Literal[
    "pit_option_chain",
    "option_settlement_prices",
    "option_bid_ask_quotes",
    "option_iv_or_price_for_greeks",
    "underlying_history",
    "contract_multiplier_and_tick_rules",
    "fee_schedule",
    "commodity_daily_underlying",
    "mo_intraday_option_quotes",
    "mo_intraday_greeks_or_reprice_inputs",
    "im_intraday_futures_quotes",
    "second_level_timestamps",
    "margin_rules",
    "rebalance_execution_costs",
]


@dataclass(frozen=True)
class DataRequirement:
    key: RequirementKey
    label: str
    reason: str
    min_frequency: str


@dataclass(frozen=True)
class DomesticOptionsConfig:
    adx_thresholds: tuple[float, ...] = (18.0, 22.0, 26.0, 30.0)
    ema_fast_windows: tuple[int, ...] = (10, 13, 20)
    ema_slow_windows: tuple[int, ...] = (26, 34, 55)
    delta_min_values: tuple[float, ...] = (0.25, 0.35, 0.45)
    delta_max_values: tuple[float, ...] = (0.55, 0.65, 0.75)
    hard_stop_values: tuple[float, ...] = (0.30, 0.40, 0.50)
    trail_drawdown_values: tuple[float, ...] = (0.10, 0.15, 0.20)
    atr_stop_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    min_dte_values: tuple[int, ...] = (15, 30, 45)
    iv_drop_stop_values: tuple[float, ...] = (0.04, 0.08, 0.12)
    mo_delta_bands: tuple[float, ...] = (0.10, 0.20, 0.30)
    hedge_check_seconds: tuple[int, ...] = (1, 5, 15)
    mo_iv_discount_values: tuple[float, ...] = (0.00, 0.08, 0.15)
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    annualization_periods: int = 252
    min_trades: int = 30


@dataclass(frozen=True)
class OptionTrade:
    symbol: str
    signal_timestamp: int
    entry_timestamp: int
    exit_timestamp: int
    entry_ask: float
    exit_bid: float
    quantity: int = 1
    multiplier: float = 1.0
    fee: float = 0.0
    slippage: float = 0.0


@dataclass(frozen=True)
class DeltaNeutralTrade:
    symbol: str
    signal_timestamp: int
    entry_timestamp: int
    exit_timestamp: int
    option_entry_ask: float
    option_exit_bid: float
    option_quantity: int
    option_multiplier: float
    hedge_pnl: float
    hedge_fee: float
    hedge_slippage: float
    margin_capital: float
    option_fee: float = 0.0
    option_slippage: float = 0.0


COMMODITY_ADX_REQUIREMENTS: tuple[DataRequirement, ...] = (
    DataRequirement(
        "pit_option_chain",
        "PIT option contract chain",
        "ADX option buyer must select then-listed contracts without delisted-chain leakage.",
        "daily",
    ),
    DataRequirement(
        "option_settlement_prices",
        "option settlement prices",
        "Daily mark-to-market and exits need actual option prices, not underlying proxies.",
        "daily",
    ),
    DataRequirement(
        "option_bid_ask_quotes",
        "option bid/ask quotes",
        "Full-cost buyer backtest must enter at ask and exit at bid.",
        "daily or intraday close snapshot",
    ),
    DataRequirement(
        "option_iv_or_price_for_greeks",
        "IV or inputs to reprice greeks",
        "Delta filter and Gamma/Theta ranking require point-in-time greeks or repricing inputs.",
        "daily",
    ),
    DataRequirement(
        "underlying_history",
        "underlying futures history",
        "EMA/ADX/ATR signals must be computed from the tradable underlying at t or earlier.",
        "daily",
    ),
    DataRequirement(
        "contract_multiplier_and_tick_rules",
        "contract multiplier and tick rules",
        "P&L, slippage, and contract rounding require exchange-specific contract terms.",
        "static/PIT",
    ),
    DataRequirement(
        "fee_schedule",
        "fees",
        "Net EV requires exchange/broker fee assumptions.",
        "PIT schedule or conservative constant",
    ),
)

MO_IM_REQUIREMENTS: tuple[DataRequirement, ...] = (
    *COMMODITY_ADX_REQUIREMENTS,
    DataRequirement(
        "mo_intraday_option_quotes",
        "MO intraday option quotes",
        "The report claims second-level delta hedging; daily bars cannot model hedge triggers.",
        "second or tick",
    ),
    DataRequirement(
        "mo_intraday_greeks_or_reprice_inputs",
        "MO intraday greeks or repricing inputs",
        "Delta drift and rebalance thresholds require point-in-time intraday deltas.",
        "second or tick",
    ),
    DataRequirement(
        "im_intraday_futures_quotes",
        "IM intraday futures quotes",
        "Every hedge rebalance needs executable IM prices and spread/slippage.",
        "second or tick",
    ),
    DataRequirement(
        "second_level_timestamps",
        "aligned second-level timestamps",
        "Option and hedge legs must be synchronized to avoid hidden lookahead.",
        "second",
    ),
    DataRequirement(
        "margin_rules",
        "margin rules",
        "Delta-neutral return denominator must include IM margin capital, not option premium only.",
        "PIT schedule or conservative constant",
    ),
    DataRequirement(
        "rebalance_execution_costs",
        "rebalance execution costs",
        "The mechanism succeeds or fails on high-frequency hedge fee/slippage drag.",
        "per rebalance",
    ),
)


def requirement_keys(mechanism: Mechanism) -> tuple[RequirementKey, ...]:
    requirements = (
        COMMODITY_ADX_REQUIREMENTS
        if mechanism == "commodity_adx_option_buyer"
        else MO_IM_REQUIREMENTS
    )
    return tuple(item.key for item in requirements)


def trial_count(config: DomesticOptionsConfig, mechanism: Mechanism, *, product_count: int) -> int:
    if product_count < 1:
        return 0
    adx_count = (
        len(config.adx_thresholds)
        * sum(
            1 for fast in config.ema_fast_windows for slow in config.ema_slow_windows if fast < slow
        )
        * sum(
            1
            for min_delta in config.delta_min_values
            for max_delta in config.delta_max_values
            if min_delta < max_delta
        )
        * len(config.hard_stop_values)
        * len(config.trail_drawdown_values)
        * len(config.atr_stop_multipliers)
        * len(config.min_dte_values)
        * len(config.iv_drop_stop_values)
        * product_count
    )
    if mechanism == "commodity_adx_option_buyer":
        return adx_count
    return (
        adx_count
        * len(config.mo_delta_bands)
        * len(config.hedge_check_seconds)
        * len(config.mo_iv_discount_values)
    )


def evaluate_data_feasibility(
    available: Mapping[str, bool],
    *,
    product_count: int = 8,
    config: DomesticOptionsConfig | None = None,
) -> dict[str, object]:
    cfg = config or DomesticOptionsConfig()
    mechanisms: dict[str, object] = {}
    for mechanism in ("commodity_adx_option_buyer", "mo_im_delta_neutral"):
        missing = [
            requirement.key
            for requirement in (
                COMMODITY_ADX_REQUIREMENTS
                if mechanism == "commodity_adx_option_buyer"
                else MO_IM_REQUIREMENTS
            )
            if not bool(available.get(requirement.key, False))
        ]
        mechanisms[mechanism] = {
            "verdict": "INSUFFICIENT" if missing else "DATA_READY",
            "missing_requirements": missing,
            "requirement_count": len(requirement_keys(mechanism)),
            "trial_count_n": trial_count(cfg, mechanism, product_count=product_count),
            "requirements": [
                {
                    "key": requirement.key,
                    "label": requirement.label,
                    "reason": requirement.reason,
                    "min_frequency": requirement.min_frequency,
                    "available": bool(available.get(requirement.key, False)),
                }
                for requirement in (
                    COMMODITY_ADX_REQUIREMENTS
                    if mechanism == "commodity_adx_option_buyer"
                    else MO_IM_REQUIREMENTS
                )
            ],
        }
    overall = (
        "DATA_READY"
        if all(
            mechanism_report["verdict"] == "DATA_READY"
            for mechanism_report in mechanisms.values()
            if isinstance(mechanism_report, Mapping)
        )
        else "INSUFFICIENT"
    )
    return {
        "status": overall,
        "verdict": overall,
        "reason": (
            "required PIT option-chain, executable quotes, greeks, intraday hedge, and cost data "
            "are all available"
            if overall == "DATA_READY"
            else "fail-closed: at least one mechanism is missing required PIT/executable data"
        ),
        "mechanisms": mechanisms,
        "source_report_parameters_are_in_sample": True,
        "max_positive_verdict": "SUGGESTIVE",
    }


def option_buyer_return_after_costs(trade: OptionTrade) -> float:
    _validate_t_plus_one(trade.signal_timestamp, trade.entry_timestamp, trade.exit_timestamp)
    if trade.quantity <= 0 or trade.multiplier <= 0:
        raise ValueError("quantity and multiplier must be positive")
    entry_cash = trade.entry_ask * trade.quantity * trade.multiplier
    if entry_cash <= 0:
        raise ValueError("entry ask must create positive premium at risk")
    exit_cash = trade.exit_bid * trade.quantity * trade.multiplier
    total_cost = trade.fee + trade.slippage
    return (exit_cash - entry_cash - total_cost) / entry_cash


def delta_neutral_return_after_costs(trade: DeltaNeutralTrade) -> float:
    _validate_t_plus_one(trade.signal_timestamp, trade.entry_timestamp, trade.exit_timestamp)
    if trade.option_quantity <= 0 or trade.option_multiplier <= 0:
        raise ValueError("option quantity and multiplier must be positive")
    option_entry_cash = trade.option_entry_ask * trade.option_quantity * trade.option_multiplier
    capital = max(float(trade.margin_capital), option_entry_cash)
    if capital <= 0:
        raise ValueError("margin capital must be positive")
    option_exit_cash = trade.option_exit_bid * trade.option_quantity * trade.option_multiplier
    option_pnl = option_exit_cash - option_entry_cash
    total_cost = trade.option_fee + trade.option_slippage + trade.hedge_fee + trade.hedge_slippage
    return (option_pnl + trade.hedge_pnl - total_cost) / capital


def run_synthetic_mechanism_verdict(
    returns_by_trial: Sequence[Sequence[float]],
    *,
    mechanism: Mechanism,
    config: DomesticOptionsConfig | None = None,
    cost_model: CostModel | None = None,
    survivor_light: bool = True,
) -> StandardVerdict:
    cfg = config or DomesticOptionsConfig()
    costs = cost_model or CostModel(
        fee_bps=2.0,
        slippage_bps=5.0,
        funding_label="N/A for listed options",
    )
    candidate_count = len(returns_by_trial)
    if candidate_count == 0:
        return _insufficient_verdict(mechanism, "no predeclared trial returns supplied")
    if any(len(trial) < cfg.min_trades for trial in returns_by_trial):
        return _insufficient_verdict(mechanism, "trial trade count below minimum")
    p_values = [sign_test_p_value(trial, alternative="greater") for trial in returns_by_trial]
    fdr = benjamini_hochberg(p_values, alpha=cfg.fdr_alpha)
    pbo_result: Mapping[str, object]
    try:
        pbo_result = pbo(returns_by_trial, n_splits=cfg.pbo_splits)
    except ValueError as exc:
        return _insufficient_verdict(mechanism, f"invalid PBO: {exc}")
    best_index = max(
        range(candidate_count),
        key=lambda index: statistics.fmean(returns_by_trial[index]),
    )
    best_returns = tuple(float(value) for value in returns_by_trial[best_index])
    metrics = metrics_from_returns(
        best_returns,
        annualization_periods=cfg.annualization_periods,
        turnover=float(len(best_returns)),
        net_cost=costs.one_way_cost * len(best_returns),
    )
    scorecard = trade_scorecard(best_returns)
    fdr_survivors = sum(1 for passed in fdr if passed)
    pbo_value = cast(float, pbo_result["pbo"])
    positive = fdr[best_index] and pbo_value < 0.5 and statistics.fmean(best_returns) > 0.0
    verdict = "SUGGESTIVE" if positive and survivor_light else ("EDGE" if positive else "NO_EDGE")
    return StandardVerdict(
        state="EDGE" if positive else "NO_EDGE",
        verdict=verdict,
        reason=(
            "synthetic trial passes FDR/PBO/EV gates; capped by source in-sample/survivor-light"
            if positive and survivor_light
            else "no synthetic trial survived FDR/PBO/EV gates"
        ),
        metrics={
            "annualized_return": metrics.annualized_return,
            "total_return": metrics.total_return,
            "max_drawdown": metrics.max_drawdown,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "calmar": metrics.calmar,
            "win_rate": scorecard.win_rate,
            "expectancy": scorecard.expectancy_per_trade,
            "profit_factor": scorecard.profit_factor,
            "funding": "N/A for listed options",
        },
        benchmarks={"cash": 0.0, "underlying_buy_hold": "required for real data"},
        candidate_count_n=candidate_count,
        raw_survivors=sum(1 for value in p_values if value < cfg.fdr_alpha),
        fdr_survivors=fdr_survivors,
        multiple_testing={
            "candidate_count_n": candidate_count,
            "fdr_alpha": cfg.fdr_alpha,
            "p_values": p_values,
            "fdr_pass": fdr,
            "pbo": pbo_result,
            "deflated_sharpe_threshold": pbo_result.get("dsr_sharpe_threshold"),
        },
        safety={
            "live_trading": False,
            "broker_gui": False,
            "source_parameters_in_sample": True,
            "max_positive_verdict": "SUGGESTIVE",
        },
        survivor_ceiling_applied=survivor_light,
    )


def _validate_t_plus_one(signal_timestamp: int, entry_timestamp: int, exit_timestamp: int) -> None:
    if entry_timestamp <= signal_timestamp:
        raise ValueError("entry timestamp must be strictly after signal timestamp (t+1)")
    if exit_timestamp <= entry_timestamp:
        raise ValueError("exit timestamp must be strictly after entry timestamp")


def _insufficient_verdict(mechanism: Mechanism, reason: str) -> StandardVerdict:
    return StandardVerdict(
        state="INSUFFICIENT",
        verdict="INSUFFICIENT",
        reason=f"{mechanism}: {reason}",
        data_adequacy="blocked",
        unlock_condition=reason,
        metrics={},
        benchmarks={"cash": 0.0, "underlying_buy_hold": "not evaluated"},
        candidate_count_n=0,
        raw_survivors=0,
        fdr_survivors=0,
        multiple_testing={},
        safety={"live_trading": False, "broker_gui": False},
        survivor_ceiling_applied=True,
    )


def nan_to_none(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): nan_to_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nan_to_none(item) for item in value]
    return value
