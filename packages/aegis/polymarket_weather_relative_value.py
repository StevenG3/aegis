from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from aegis.backtest_core import benjamini_hochberg, pbo, sign_test_p_value

Verdict = Literal["SUGGESTIVE_NEEDS_PAID_CONFIRM", "NO_EDGE", "INSUFFICIENT"]
TradeDirection = Literal["BUY_YES", "BUY_NO"]

EDGE_THRESHOLDS = (0.05, 0.10, 0.15)
TRADE_DIRECTIONS: tuple[TradeDirection, ...] = ("BUY_YES", "BUY_NO")
TRIAL_COUNT_N = len(EDGE_THRESHOLDS) * len(TRADE_DIRECTIONS)


@dataclass(frozen=True)
class WeatherRelativeValueConfig:
    edge_thresholds: tuple[float, ...] = EDGE_THRESHOLDS
    min_observations: int = 30
    fdr_alpha: float = 0.10
    pbo_splits: int = 4
    taker_fee_rate: float = 0.0
    survivor_light: bool = True


@dataclass(frozen=True)
class TemperatureBucket:
    label: str
    lower_f: int | None
    upper_f: int | None

    def contains(self, value_f: int) -> bool:
        if self.lower_f is not None and value_f < self.lower_f:
            return False
        if self.upper_f is not None and value_f > self.upper_f:
            return False
        return True


@dataclass(frozen=True)
class WeatherRelativeValueObservation:
    event_slug: str
    market_slug: str
    city: str
    station: str
    bucket: TemperatureBucket
    decision_ts: int
    forecast_issue_ts: int
    model_probability: float
    yes_ask: float
    no_ask: float
    yes_bid: float | None
    no_bid: float | None
    actual_won: bool


@dataclass(frozen=True)
class WeatherTrade:
    event_slug: str
    market_slug: str
    direction: TradeDirection
    threshold: float
    model_probability: float
    side_probability: float
    yes_ask: float
    no_ask: float
    ask_price: float
    fee_cost: float
    net_return: float
    actual_won: bool


def run_weather_relative_value_firstpass(
    rows: Sequence[Mapping[str, object]],
    *,
    config: WeatherRelativeValueConfig | None = None,
) -> Mapping[str, Any]:
    if config is None:
        config = WeatherRelativeValueConfig()
    observations, excluded = _observations(rows)
    coverage: dict[str, Any] = {
        "input_rows": len(rows),
        "observations": len(observations),
        "excluded_rows": len(excluded),
        "excluded_reasons": _reason_counts(excluded),
        "cities": sorted({observation.city for observation in observations}),
        "stations": sorted({observation.station for observation in observations}),
    }
    if len(observations) < config.min_observations:
        return _insufficient(
            f"weather relative-value observations {len(observations)} < min_observations "
            f"{config.min_observations}",
            coverage,
            config,
        )

    evaluations = [
        _evaluate_threshold(threshold, direction, observations, config)
        for direction in TRADE_DIRECTIONS
        for threshold in config.edge_thresholds
    ]
    valid = [evaluation for evaluation in evaluations if evaluation["trades"]]
    if not valid:
        return _insufficient(
            "no trades passed predeclared model-vs-ask thresholds",
            coverage,
            config,
        )

    p_values = [float(evaluation["p_value"]) for evaluation in valid]
    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha, tie_policy="rank")
    pbo_report = _pbo_report(valid, observations, config)
    pbo_value = _float(pbo_report.get("pbo"), default=1.0)
    pbo_valid = bool(pbo_report.get("valid", False))
    survivors = [
        evaluation
        for evaluation, fdr_pass in zip(valid, fdr_flags, strict=True)
        if fdr_pass
        and pbo_valid
        and pbo_value <= 0.20
        and float(evaluation["mean_net_return"]) > 0.0
    ]
    best = max(valid, key=lambda evaluation: float(evaluation["mean_net_return"]))
    verdict: Verdict
    if survivors:
        verdict = "SUGGESTIVE_NEEDS_PAID_CONFIRM"
        reason = (
            "one or more weather relative-value thresholds survived BH-FDR and valid PBO; "
            "survivor-light and venue caps apply"
        )
    else:
        verdict = "NO_EDGE"
        reason = (
            "no predeclared weather relative-value threshold passed FDR, valid PBO, and EV gates"
        )
    return {
        "status": "OK",
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": "limited",
        "unlock_condition": (
            "longer multi-city history with true YES/NO ask series, settlement-aligned stations, "
            "and both-sided coverage beyond the initial small sample"
        ),
        "candidate_count_n": len(config.edge_thresholds) * len(TRADE_DIRECTIONS),
        "coverage": coverage,
        "standard_metrics": _metrics(best["trades"]),
        "benchmark_metrics": {
            "well_calibrated_market_edge": 0.0,
            "coin_flip_edge": 0.0,
            "always_favorite": "not_applicable_to_bucket_binary_firstpass",
        },
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": len(config.edge_thresholds) * len(TRADE_DIRECTIONS),
            "tested_candidates": len(valid),
            "fdr_after": sum(1 for flag in fdr_flags if flag),
            "pbo": pbo_report,
            "min_p": min(p_values) if p_values else None,
            "directions": list(TRADE_DIRECTIONS),
            "direction_trade_counts": _direction_trade_counts(valid),
        },
        "best_candidate": {key: value for key, value in best.items() if key != "trades"},
        "safety": _safety(config),
    }


def parse_temperature_bucket(label: str) -> TemperatureBucket | None:
    clean = label.strip().replace("°", "").replace("degrees", "").replace(" ", "")
    below = re.fullmatch(r"(-?\d+)F?orbelow", clean, flags=re.IGNORECASE)
    if below:
        value = int(below.group(1))
        return TemperatureBucket(label=label, lower_f=None, upper_f=value)
    above = re.fullmatch(r"(-?\d+)F?or(higher|above)", clean, flags=re.IGNORECASE)
    if above:
        value = int(above.group(1))
        return TemperatureBucket(label=label, lower_f=value, upper_f=None)
    span = re.fullmatch(r"(-?\d+)-(-?\d+)F?", clean, flags=re.IGNORECASE)
    if span:
        lower = int(span.group(1))
        upper = int(span.group(2))
        if lower > upper:
            return None
        return TemperatureBucket(label=label, lower_f=lower, upper_f=upper)
    return None


def station_from_wunderground_url(value: str) -> str | None:
    try:
        parts = [part for part in urlparse(value).path.split("/") if part]
    except Exception:
        return None
    if len(parts) < 2 or parts[0] != "history" or parts[1] != "daily":
        return None
    station = parts[-1].upper()
    return station if re.fullmatch(r"[A-Z0-9]{3,5}", station) else None


def settlement_source_alignment(event: Mapping[str, object]) -> Mapping[str, Any]:
    markets = event.get("markets")
    if not isinstance(markets, list):
        markets = []
    event_source = _str(event.get("resolutionSource"))
    market_sources = [
        _str(market.get("resolutionSource")) for market in markets if isinstance(market, Mapping)
    ]
    sources = {source for source in [event_source, *market_sources] if source}
    stations = {station for source in sources if (station := station_from_wunderground_url(source))}
    descriptions = [
        _str(event.get("description")) or "",
        *[
            _str(market.get("description")) or ""
            for market in markets
            if isinstance(market, Mapping)
        ],
    ]
    text = "\n".join(descriptions).lower()
    whole_degrees = "whole degrees fahrenheit" in text
    frozen_after_next_day = (
        "first datapoint for the following date" in text
        or "all data for this date has been finalized" in text
    )
    aligned = len(stations) == 1 and whole_degrees and frozen_after_next_day
    return {
        "aligned": aligned,
        "station": next(iter(stations)) if len(stations) == 1 else None,
        "stations": sorted(stations),
        "sources": sorted(sources),
        "whole_degrees_fahrenheit": whole_degrees,
        "frozen_after_next_day_or_finalized": frozen_after_next_day,
    }


def _observations(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[WeatherRelativeValueObservation], list[tuple[str, str]]]:
    observations: list[WeatherRelativeValueObservation] = []
    excluded: list[tuple[str, str]] = []
    for row in rows:
        reason = _row_exclusion_reason(row)
        if reason is not None:
            excluded.append((_str(row.get("market_slug")) or "", reason))
            continue
        bucket = parse_temperature_bucket(str(row["bucket_label"]))
        assert bucket is not None
        observations.append(
            WeatherRelativeValueObservation(
                event_slug=str(row["event_slug"]),
                market_slug=str(row["market_slug"]),
                city=str(row["city"]),
                station=str(row["station"]),
                bucket=bucket,
                decision_ts=_required_int(row["decision_ts"]),
                forecast_issue_ts=_required_int(row["forecast_issue_ts"]),
                model_probability=_required_float(row["model_probability"]),
                yes_ask=_required_float(row["yes_ask"]),
                no_ask=_required_float(row["no_ask"]),
                yes_bid=_optional_float(row.get("yes_bid")),
                no_bid=_optional_float(row.get("no_bid")),
                actual_won=bool(row["actual_won"]),
            )
        )
    return observations, excluded


def _row_exclusion_reason(row: Mapping[str, object]) -> str | None:
    required = (
        "event_slug",
        "market_slug",
        "city",
        "station",
        "bucket_label",
        "decision_ts",
        "forecast_issue_ts",
        "model_probability",
        "yes_ask",
        "no_ask",
        "actual_won",
    )
    for key in required:
        if key not in row:
            return f"missing_{key}"
    decision_ts = _optional_int(row["decision_ts"])
    issue_ts = _optional_int(row["forecast_issue_ts"])
    model_probability = _optional_float(row["model_probability"])
    yes_ask = _optional_float(row["yes_ask"])
    no_ask = _optional_float(row["no_ask"])
    if (
        decision_ts is None
        or issue_ts is None
        or model_probability is None
        or yes_ask is None
        or no_ask is None
    ):
        return "parse_error"
    if issue_ts >= decision_ts:
        return "forecast_issue_not_before_decision"
    if not 0.0 <= model_probability <= 1.0:
        return "model_probability_out_of_range"
    if not 0.0 < yes_ask <= 1.0:
        return "yes_ask_out_of_range"
    if not 0.0 < no_ask <= 1.0:
        return "no_ask_out_of_range"
    if parse_temperature_bucket(str(row["bucket_label"])) is None:
        return "unparseable_temperature_bucket"
    return None


def _evaluate_threshold(
    threshold: float,
    direction: TradeDirection,
    observations: Sequence[WeatherRelativeValueObservation],
    config: WeatherRelativeValueConfig,
) -> dict[str, Any]:
    trades: list[WeatherTrade] = []
    for observation in observations:
        side_probability = (
            observation.model_probability
            if direction == "BUY_YES"
            else 1.0 - observation.model_probability
        )
        ask_price = observation.yes_ask if direction == "BUY_YES" else observation.no_ask
        if side_probability - ask_price < threshold:
            continue
        actual_won = (
            observation.actual_won if direction == "BUY_YES" else not observation.actual_won
        )
        fee_cost = ask_price * config.taker_fee_rate
        net_return = (
            (1.0 - ask_price - fee_cost) if actual_won else (-ask_price - fee_cost)
        ) / max(ask_price, 1e-9)
        trades.append(
            WeatherTrade(
                event_slug=observation.event_slug,
                market_slug=observation.market_slug,
                direction=direction,
                threshold=threshold,
                model_probability=observation.model_probability,
                side_probability=side_probability,
                yes_ask=observation.yes_ask,
                no_ask=observation.no_ask,
                ask_price=ask_price,
                fee_cost=fee_cost,
                net_return=net_return,
                actual_won=actual_won,
            )
        )
    wins = sum(1 for trade in trades if trade.actual_won)
    losses = sum(1 for trade in trades if not trade.actual_won)
    signs = [1 if trade.net_return > 0 else -1 if trade.net_return < 0 else 0 for trade in trades]
    return {
        "direction": direction,
        "threshold": threshold,
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
    observations: Sequence[WeatherRelativeValueObservation],
    config: WeatherRelativeValueConfig,
) -> Mapping[str, Any]:
    trial_returns: list[list[float]] = []
    ordered_markets = sorted({observation.market_slug for observation in observations})
    for evaluation in evaluations:
        by_market = {
            trade.market_slug: trade.net_return
            for trade in evaluation["trades"]
            if isinstance(trade, WeatherTrade)
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
    return {**report, "valid": bool(report.get("valid", False))}


def _metrics(trades: object) -> Mapping[str, Any]:
    if not isinstance(trades, Sequence) or not trades:
        return {"trades": 0}
    clean = [trade for trade in trades if isinstance(trade, WeatherTrade)]
    wins = sum(1 for trade in clean if trade.actual_won)
    losses = len(clean) - wins
    return {
        "trades": len(clean),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(clean) if clean else 0.0,
        "mean_net_return": statistics.fmean(trade.net_return for trade in clean) if clean else 0.0,
        "total_net_return": sum(trade.net_return for trade in clean),
        "mean_yes_ask": statistics.fmean(trade.yes_ask for trade in clean) if clean else None,
        "mean_no_ask": statistics.fmean(trade.no_ask for trade in clean) if clean else None,
        "mean_ask_price": statistics.fmean(trade.ask_price for trade in clean) if clean else None,
        "mean_model_probability": statistics.fmean(trade.model_probability for trade in clean)
        if clean
        else None,
        "direction_counts": {
            direction: sum(1 for trade in clean if trade.direction == direction)
            for direction in TRADE_DIRECTIONS
        },
        "mean_fee_cost": statistics.fmean(trade.fee_cost for trade in clean) if clean else None,
    }


def _direction_trade_counts(evaluations: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {direction: 0 for direction in TRADE_DIRECTIONS}
    for evaluation in evaluations:
        trades = evaluation.get("trades")
        if not isinstance(trades, Sequence):
            continue
        for trade in trades:
            if isinstance(trade, WeatherTrade):
                counts[trade.direction] += 1
    return counts


def _reason_counts(excluded: Sequence[tuple[str, str]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for _, reason in excluded:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _insufficient(
    reason: str,
    coverage: Mapping[str, object],
    config: WeatherRelativeValueConfig,
) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "data_adequacy": "blocked",
        "unlock_condition": reason,
        "candidate_count_n": len(config.edge_thresholds) * len(TRADE_DIRECTIONS),
        "coverage": dict(coverage),
        "standard_metrics": {},
        "benchmark_metrics": {
            "well_calibrated_market_edge": 0.0,
            "coin_flip_edge": 0.0,
        },
        "multiple_testing": {
            "method": "BH-FDR + CSCV_PBO",
            "candidate_count_n": len(config.edge_thresholds) * len(TRADE_DIRECTIONS),
            "fdr_after": 0,
            "pbo": {"valid": False, "reason": "not run under INSUFFICIENT gate"},
        },
        "safety": _safety(config),
    }


def _safety(config: WeatherRelativeValueConfig) -> Mapping[str, object]:
    return {
        "read_only": True,
        "wallet_or_order_access": False,
        "live_trading": False,
        "account_access": False,
        "survivor_light_ceiling_required": config.survivor_light,
        "funding": "N/A prediction market",
    }


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


def _float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    return default


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


def _required_float(value: object) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"expected finite float-compatible value, got {value!r}")
    return parsed


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"expected int-compatible value, got {value!r}")
    return parsed
