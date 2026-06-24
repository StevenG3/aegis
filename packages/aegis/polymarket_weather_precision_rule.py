from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aegis.backtest_core import benjamini_hochberg, pbo, sign_test_p_value

PrecisionDirection = Literal["BUY_YES", "BUY_NO"]
PrecisionLayer = Literal["cheap_yes", "fade_expensive_yes", "tail_yes", "tail_no"]
PrecisionVerdict = Literal["SUGGESTIVE_NEEDS_PAID_CONFIRM", "NO_EDGE", "INSUFFICIENT"]


@dataclass(frozen=True)
class WeatherPrecisionRuleConfig:
    yes_ask_max: tuple[float, ...] = (0.25, 0.30, 0.35)
    expensive_yes_min: tuple[float, ...] = (0.60, 0.65, 0.70)
    tail_yes_ask_max: tuple[float, ...] = (0.10, 0.15, 0.20)
    tail_yes_min: tuple[float, ...] = (0.75, 0.80, 0.85)
    forecast_yes_min: tuple[float, ...] = (0.55, 0.60, 0.65)
    forecast_yes_max: tuple[float, ...] = (0.35, 0.40, 0.45)
    entry_windows: tuple[str, ...] = ("morning_local", "late_morning_local")
    min_observations: int = 30
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    taker_fee_rate: float = 0.0
    slippage_rate: float = 0.005
    survivor_light: bool = True

    @property
    def candidate_count_n(self) -> int:
        cheap_yes = len(self.yes_ask_max) * len(self.forecast_yes_min)
        fade_expensive = len(self.expensive_yes_min) * len(self.forecast_yes_max)
        tail_yes = len(self.tail_yes_ask_max) * len(self.tail_yes_min)
        tail_no = len(self.expensive_yes_min) * len(self.tail_yes_min)
        return len(self.entry_windows) * (cheap_yes + fade_expensive + tail_yes + tail_no)


@dataclass(frozen=True)
class WeatherPrecisionObservation:
    event_slug: str
    market_slug: str
    city: str
    station: str
    decision_ts: int
    forecast_issue_ts: int
    entry_window: str
    forecast_yes_probability: float
    yes_ask: float
    no_ask: float
    actual_yes_won: bool


@dataclass(frozen=True)
class WeatherPrecisionTrade:
    event_slug: str
    market_slug: str
    city: str
    layer: PrecisionLayer
    direction: PrecisionDirection
    entry_window: str
    ask_price: float
    forecast_yes_probability: float
    actual_won: bool
    fee_cost: float
    slippage_cost: float
    net_return: float


def run_weather_precision_rule(
    rows: Sequence[Mapping[str, object]],
    *,
    config: WeatherPrecisionRuleConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = WeatherPrecisionRuleConfig()
    observations, excluded = _observations(rows)
    coverage: dict[str, Any] = {
        "input_rows": len(rows),
        "observations": len(observations),
        "excluded_rows": len(excluded),
        "excluded_reasons": _reason_counts(excluded),
        "cities": sorted({row.city for row in observations}),
        "stations": sorted({row.station for row in observations}),
        "entry_windows": sorted({row.entry_window for row in observations}),
    }
    if len(observations) < config.min_observations:
        return _insufficient(
            f"weather precision observations {len(observations)} < min_observations "
            f"{config.min_observations}",
            coverage,
            config,
        )

    evaluations = _evaluate_all(observations, config)
    valid = [evaluation for evaluation in evaluations if evaluation["trades"]]
    if not valid:
        return _insufficient(
            "no trades passed predeclared fixed-price/weather-confidence thresholds",
            coverage,
            config,
        )

    p_values = [float(evaluation["p_value"]) for evaluation in valid]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(valid, observations, config)
    pbo_valid = bool(pbo_report.get("valid", False))
    pbo_value = _float(pbo_report.get("pbo"), default=1.0)
    survivors = [
        evaluation
        for evaluation, fdr_pass in zip(valid, fdr_flags, strict=True)
        if fdr_pass
        and pbo_valid
        and pbo_value <= 0.20
        and float(evaluation["mean_net_return"]) > 0.0
    ]
    best = max(valid, key=lambda evaluation: float(evaluation["mean_net_return"]))
    verdict: PrecisionVerdict
    if survivors:
        verdict = "SUGGESTIVE_NEEDS_PAID_CONFIRM"
        state = "EDGE"
        reason = (
            "one or more predeclared weather precision thresholds survived BH-FDR, "
            "valid PBO, and positive net-EV gates; survivor and venue caps apply"
        )
    else:
        verdict = "NO_EDGE"
        state = "NO_EDGE"
        reason = "no predeclared weather precision threshold passed net-EV, FDR, and PBO gates"
    return {
        "status": "OK",
        "state": state,
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "longer multi-city point-in-time forecast history with true YES/NO ask/depth "
            "and settlement-aligned stations"
        ),
        "candidate_count_n": config.candidate_count_n,
        "coverage": coverage,
        "standard_metrics": _metrics(best["trades"]),
        "benchmark_metrics": {
            "weather_buy_hold_bid": "requires historical executable bid series; not inferred",
            "well_calibrated_market_edge": 0.0,
        },
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": config.candidate_count_n,
            "tested_candidates": len(valid),
            "fdr_after": sum(1 for flag in fdr_flags if flag),
            "pbo": pbo_report,
            "min_p": min(p_values) if p_values else None,
            "layer_trade_counts": _layer_trade_counts(valid),
            "direction_trade_counts": _direction_trade_counts(valid),
        },
        "best_candidate": {key: value for key, value in best.items() if key != "trades"},
        "safety": _safety(config),
    }


def precision_data_blocked_report(
    *,
    reason: str,
    coverage: Mapping[str, object],
    config: WeatherPrecisionRuleConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = WeatherPrecisionRuleConfig()
    return _insufficient(reason, coverage, config)


def _observations(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[WeatherPrecisionObservation], list[tuple[str, str]]]:
    observations: list[WeatherPrecisionObservation] = []
    excluded: list[tuple[str, str]] = []
    for row in rows:
        reason = _row_exclusion_reason(row)
        if reason is not None:
            excluded.append((_str(row.get("market_slug")) or "", reason))
            continue
        observations.append(
            WeatherPrecisionObservation(
                event_slug=str(row["event_slug"]),
                market_slug=str(row["market_slug"]),
                city=str(row["city"]),
                station=str(row["station"]),
                decision_ts=_required_int(row["decision_ts"]),
                forecast_issue_ts=_required_int(row["forecast_issue_ts"]),
                entry_window=str(row["entry_window"]),
                forecast_yes_probability=_required_float(row["forecast_yes_probability"]),
                yes_ask=_required_float(row["yes_ask"]),
                no_ask=_required_float(row["no_ask"]),
                actual_yes_won=bool(row["actual_yes_won"]),
            )
        )
    return observations, excluded


def _row_exclusion_reason(row: Mapping[str, object]) -> str | None:
    required = (
        "event_slug",
        "market_slug",
        "city",
        "station",
        "decision_ts",
        "forecast_issue_ts",
        "entry_window",
        "forecast_yes_probability",
        "yes_ask",
        "no_ask",
        "actual_yes_won",
    )
    for key in required:
        if key not in row:
            return f"missing_{key}"
    decision_ts = _optional_int(row["decision_ts"])
    issue_ts = _optional_int(row["forecast_issue_ts"])
    probability = _optional_float(row["forecast_yes_probability"])
    yes_ask = _optional_float(row["yes_ask"])
    no_ask = _optional_float(row["no_ask"])
    if (
        decision_ts is None
        or issue_ts is None
        or probability is None
        or yes_ask is None
        or no_ask is None
    ):
        return "parse_error"
    if issue_ts >= decision_ts:
        return "forecast_issue_not_before_decision"
    if not 0.0 <= probability <= 1.0:
        return "forecast_yes_probability_out_of_range"
    if not 0.0 < yes_ask <= 1.0:
        return "yes_ask_out_of_range"
    if not 0.0 < no_ask <= 1.0:
        return "no_ask_out_of_range"
    return None


def _evaluate_all(
    observations: Sequence[WeatherPrecisionObservation],
    config: WeatherPrecisionRuleConfig,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for entry_window in config.entry_windows:
        scoped = [row for row in observations if row.entry_window == entry_window]
        for yes_ask_max in config.yes_ask_max:
            for forecast_min in config.forecast_yes_min:
                evaluations.append(
                    _evaluate(
                        "cheap_yes",
                        "BUY_YES",
                        entry_window,
                        scoped,
                        config,
                        yes_ask_max=yes_ask_max,
                        forecast_min=forecast_min,
                    )
                )
        for expensive_yes_min in config.expensive_yes_min:
            for forecast_max in config.forecast_yes_max:
                evaluations.append(
                    _evaluate(
                        "fade_expensive_yes",
                        "BUY_NO",
                        entry_window,
                        scoped,
                        config,
                        expensive_yes_min=expensive_yes_min,
                        forecast_max=forecast_max,
                    )
                )
        for tail_yes_ask_max in config.tail_yes_ask_max:
            for tail_min in config.tail_yes_min:
                evaluations.append(
                    _evaluate(
                        "tail_yes",
                        "BUY_YES",
                        entry_window,
                        scoped,
                        config,
                        yes_ask_max=tail_yes_ask_max,
                        forecast_min=tail_min,
                    )
                )
        for expensive_yes_min in config.expensive_yes_min:
            for tail_min in config.tail_yes_min:
                evaluations.append(
                    _evaluate(
                        "tail_no",
                        "BUY_NO",
                        entry_window,
                        scoped,
                        config,
                        expensive_yes_min=expensive_yes_min,
                        forecast_max=1.0 - tail_min,
                    )
                )
    return evaluations


def _evaluate(
    layer: PrecisionLayer,
    direction: PrecisionDirection,
    entry_window: str,
    observations: Sequence[WeatherPrecisionObservation],
    config: WeatherPrecisionRuleConfig,
    *,
    yes_ask_max: float | None = None,
    expensive_yes_min: float | None = None,
    forecast_min: float | None = None,
    forecast_max: float | None = None,
) -> dict[str, Any]:
    trades: list[WeatherPrecisionTrade] = []
    for row in observations:
        if yes_ask_max is not None and row.yes_ask > yes_ask_max:
            continue
        if expensive_yes_min is not None and row.yes_ask < expensive_yes_min:
            continue
        if forecast_min is not None and row.forecast_yes_probability < forecast_min:
            continue
        if forecast_max is not None and row.forecast_yes_probability > forecast_max:
            continue
        ask_price = row.yes_ask if direction == "BUY_YES" else row.no_ask
        actual_won = row.actual_yes_won if direction == "BUY_YES" else not row.actual_yes_won
        fee_cost = ask_price * config.taker_fee_rate
        slippage_cost = ask_price * config.slippage_rate
        net_pnl = (
            1.0 - ask_price - fee_cost - slippage_cost
            if actual_won
            else -ask_price - fee_cost - slippage_cost
        )
        trades.append(
            WeatherPrecisionTrade(
                event_slug=row.event_slug,
                market_slug=row.market_slug,
                city=row.city,
                layer=layer,
                direction=direction,
                entry_window=entry_window,
                ask_price=ask_price,
                forecast_yes_probability=row.forecast_yes_probability,
                actual_won=actual_won,
                fee_cost=fee_cost,
                slippage_cost=slippage_cost,
                net_return=net_pnl / max(ask_price, 1e-9),
            )
        )
    wins = sum(1 for trade in trades if trade.actual_won)
    losses = len(trades) - wins
    signs = [1 if trade.net_return > 0 else -1 if trade.net_return < 0 else 0 for trade in trades]
    return {
        "layer": layer,
        "direction": direction,
        "entry_window": entry_window,
        "yes_ask_max": yes_ask_max,
        "expensive_yes_min": expensive_yes_min,
        "forecast_min": forecast_min,
        "forecast_max": forecast_max,
        "trade_count": len(trades),
        "wins": wins,
        "losses": losses,
        "mean_net_return": statistics.fmean(trade.net_return for trade in trades)
        if trades
        else 0.0,
        "total_net_return": sum(trade.net_return for trade in trades),
        "p_value": sign_test_p_value(signs) if trades else 1.0,
        "trades": trades,
    }


def _pbo_report(
    evaluations: Sequence[Mapping[str, Any]],
    observations: Sequence[WeatherPrecisionObservation],
    config: WeatherPrecisionRuleConfig,
) -> Mapping[str, Any]:
    trial_returns: list[list[float]] = []
    ordered_markets = sorted({observation.market_slug for observation in observations})
    for evaluation in evaluations:
        by_market = {
            trade.market_slug: trade.net_return
            for trade in evaluation["trades"]
            if isinstance(trade, WeatherPrecisionTrade)
        }
        trial_returns.append([by_market.get(market, 0.0) for market in ordered_markets])
    if len(trial_returns) < 2 or len(ordered_markets) < config.pbo_splits:
        return {
            "method": "CSCV_PBO",
            "valid": False,
            "reason": "insufficient trials or observations for PBO",
            "trial_count": len(trial_returns),
            "observation_count": len(ordered_markets),
            "n_splits": config.pbo_splits,
        }
    report = pbo(trial_returns, n_splits=config.pbo_splits)
    return {**report, "valid": True}


def _metrics(trades: object) -> Mapping[str, Any]:
    if not isinstance(trades, Sequence) or not trades:
        return {"trades": 0}
    clean = [trade for trade in trades if isinstance(trade, WeatherPrecisionTrade)]
    wins = sum(1 for trade in clean if trade.actual_won)
    losses = len(clean) - wins
    gains = [trade.net_return for trade in clean if trade.net_return > 0]
    losses_values = [-trade.net_return for trade in clean if trade.net_return < 0]
    average_win = statistics.fmean(gains) if gains else 0.0
    average_loss = statistics.fmean(losses_values) if losses_values else 0.0
    return {
        "trades": len(clean),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(clean) if clean else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "win_loss_ratio": average_win / average_loss if average_loss > 0 else None,
        "mean_net_return": statistics.fmean(trade.net_return for trade in clean) if clean else 0.0,
        "total_net_return": sum(trade.net_return for trade in clean),
        "max_drawdown": _max_drawdown([trade.net_return for trade in clean]),
        "mean_ask_price": statistics.fmean(trade.ask_price for trade in clean) if clean else None,
        "mean_fee_cost": statistics.fmean(trade.fee_cost for trade in clean) if clean else None,
        "mean_slippage_cost": statistics.fmean(trade.slippage_cost for trade in clean)
        if clean
        else None,
        "direction_counts": _counts_by(clean, "direction"),
        "layer_counts": _counts_by(clean, "layer"),
    }


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _counts_by(trades: Sequence[WeatherPrecisionTrade], field: str) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        value = str(getattr(trade, field))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _layer_trade_counts(evaluations: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for evaluation in evaluations:
        layer = str(evaluation.get("layer"))
        counts[layer] = counts.get(layer, 0) + int(evaluation.get("trade_count", 0))
    return counts


def _direction_trade_counts(evaluations: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for evaluation in evaluations:
        direction = str(evaluation.get("direction"))
        counts[direction] = counts.get(direction, 0) + int(evaluation.get("trade_count", 0))
    return counts


def _insufficient(
    reason: str,
    coverage: Mapping[str, object],
    config: WeatherPrecisionRuleConfig,
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "state": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": config.candidate_count_n,
        "coverage": dict(coverage),
        "standard_metrics": {},
        "benchmark_metrics": {
            "weather_buy_hold_bid": "not run under data-feasibility gate",
            "well_calibrated_market_edge": 0.0,
        },
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": config.candidate_count_n,
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": "not run under INSUFFICIENT gate"},
        },
        "safety": _safety(config),
    }


def _safety(config: WeatherPrecisionRuleConfig) -> Mapping[str, object]:
    return {
        "read_only": True,
        "wallet_or_order_access": False,
        "live_trading": False,
        "account_access": False,
        "survivor_light_ceiling_required": config.survivor_light,
        "funding": "N/A prediction market",
        "slippage_rate": config.slippage_rate,
    }


def _reason_counts(excluded: Sequence[tuple[str, str]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for _, reason in excluded:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _required_float(value: object) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"expected finite float-compatible value, got {value!r}")
    return parsed


def _float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else default


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"expected int-compatible value, got {value!r}")
    return parsed
