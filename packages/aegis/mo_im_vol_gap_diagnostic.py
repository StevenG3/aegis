from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from aegis.backtest_core import benjamini_hochberg, sign_test_p_value

RvMethod = Literal["close_to_close", "parkinson", "garman_klass"]
Regime = Literal["calm", "normal", "crisis", "unknown"]
VolGapVerdict = Literal["GAP_FAVORS_BUYER", "GAP_FAVORS_SELLER", "INSUFFICIENT"]


@dataclass(frozen=True)
class IndexBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class IvObservation:
    timestamp: int
    annualized_iv: float
    source: str
    source_quality: str = "proxy"


@dataclass(frozen=True)
class VolGapConfig:
    horizons: tuple[int, ...] = (20, 40)
    rv_methods: tuple[RvMethod, ...] = ("close_to_close", "parkinson", "garman_klass")
    annualization_periods: int = 252
    fdr_alpha: float = 0.10
    min_samples_per_cell: int = 30
    buyer_win_rate_threshold: float = 0.55
    buyer_mean_variance_gap_threshold: float = 0.0
    calm_trailing_rv: float = 0.15
    crisis_trailing_rv: float = 0.35


@dataclass(frozen=True)
class VolGapRow:
    timestamp: int
    horizon: int
    rv_method: RvMethod
    iv: float
    rv_forward: float
    variance_gap: float
    iv_minus_rv: float
    regime: Regime


@dataclass(frozen=True)
class OptionQuote:
    option_type: Literal["call", "put"]
    strike: float
    bid: float
    ask: float
    expiry: date
    underlying: float
    as_of: date


DEFAULT_CONFIG = VolGapConfig()


def trial_count(config: VolGapConfig = DEFAULT_CONFIG) -> int:
    return len(config.horizons) * len(config.rv_methods)


def run_vol_gap_diagnostic(
    bars: Sequence[IndexBar],
    iv_observations: Sequence[IvObservation],
    *,
    config: VolGapConfig = DEFAULT_CONFIG,
    iv_source: str = "unknown",
) -> dict[str, object]:
    valid_bars = validate_bars(bars)
    valid_iv = validate_iv_observations(iv_observations)
    if not valid_bars["ok"] or not valid_iv["ok"]:
        return _insufficient(
            reason="historical ATM MO IV is unavailable or invalid; forward collection required",
            bars_report=valid_bars,
            iv_report=valid_iv,
            config=config,
            iv_source=iv_source,
        )

    sorted_bars = sorted(bars, key=lambda item: item.timestamp)
    iv_by_timestamp = {item.timestamp: item for item in iv_observations}
    rows: list[VolGapRow] = []
    cell_summaries: list[dict[str, object]] = []
    p_values: list[float] = []

    for horizon in config.horizons:
        if horizon <= 1:
            raise ValueError("horizons must be greater than one trading day")
        for method in config.rv_methods:
            cell_rows = _cell_rows(sorted_bars, iv_by_timestamp, horizon, method, config)
            rows.extend(cell_rows)
            summary = _summarize_cell(cell_rows, horizon=horizon, method=method, config=config)
            cell_summaries.append(summary)
            if cell_rows:
                p_values.append(_float_field(summary, "buyer_sign_test_p_value"))

    fdr_flags = benjamini_hochberg(p_values, alpha=config.fdr_alpha) if p_values else []
    for summary, passed in zip(cell_summaries, fdr_flags, strict=False):
        summary["buyer_bh_fdr_pass"] = passed

    valid_cells = [
        item
        for item in cell_summaries
        if _int_field(item, "sample_count") >= config.min_samples_per_cell
    ]
    buyer_cells = [
        item
        for item in valid_cells
        if bool(item.get("buyer_bh_fdr_pass"))
        and _float_field(item, "mean_variance_gap") > config.buyer_mean_variance_gap_threshold
        and _float_field(item, "buyer_positive_rate") >= config.buyer_win_rate_threshold
    ]
    if not valid_cells:
        verdict: VolGapVerdict = "INSUFFICIENT"
        reason = "not enough aligned IV_t versus forward RV samples"
    elif buyer_cells:
        verdict = "GAP_FAVORS_BUYER"
        reason = (
            "at least one predeclared horizon/RV cell has positive variance gap after BH-FDR; "
            "proxy/no-cost diagnostic caps any positive result at SUGGESTIVE"
        )
    else:
        verdict = "GAP_FAVORS_SELLER"
        reason = (
            "predeclared IV_t versus forward RV cells do not show systematic RV>IV; "
            "the gross variance-risk-premium gap favors short-vol sellers"
        )

    all_gaps = [row.variance_gap for row in rows]
    return {
        "verdict": verdict,
        "reason": reason,
        "candidate_count_n": trial_count(config),
        "raw_survivors": len(
            [
                item
                for item in cell_summaries
                if _float_field(item, "buyer_sign_test_p_value") < config.fdr_alpha
            ]
        ),
        "fdr_survivors": len(buyer_cells),
        "iv_source": iv_source,
        "data_gate": {"bars": valid_bars, "iv": valid_iv},
        "sample_count": len(rows),
        "summary": {
            "mean_variance_gap": _mean_or_none(all_gaps),
            "buyer_positive_rate": _positive_rate(all_gaps),
            "seller_positive_rate": _positive_rate([-value for value in all_gaps]),
            "mean_iv_minus_rv": _mean_or_none([row.iv_minus_rv for row in rows]),
        },
        "cells": cell_summaries,
        "regime": _regime_summary(rows),
        "multiple_testing": {
            "method": "BH-FDR over predeclared horizons x RV methods",
            "alpha": config.fdr_alpha,
            "candidate_count_n": trial_count(config),
            "p_values": p_values,
            "fdr_flags": fdr_flags,
            "pbo": {
                "valid": False,
                "reason": (
                    "not run; this is a one-asset gross IV/RV diagnostic, not a "
                    "parameterized strategy selection"
                ),
            },
        },
        "cost_model": {
            "scope": "gross diagnostic",
            "delta_hedge_costs": "not modeled",
            "interpretation": (
                "If gross RV^2-IV^2 is negative, hedge costs can only worsen long-gamma economics. "
                "If gross gap is positive, a later full hedge-cost backtest is still required."
            ),
            "funding": "N/A for listed index options/futures",
        },
        "safety": {
            "live_trading": False,
            "order_path": False,
            "account_or_gui_access": False,
            "positive_result_ceiling": "SUGGESTIVE",
        },
    }


def validate_bars(bars: Sequence[IndexBar]) -> dict[str, object]:
    if not bars:
        return {"ok": False, "reason": "no index bars"}
    timestamps = [item.timestamp for item in bars]
    if timestamps != sorted(timestamps):
        return {"ok": False, "reason": "bars must be sorted by timestamp"}
    for item in bars:
        if item.open <= 0 or item.high <= 0 or item.low <= 0 or item.close <= 0:
            return {"ok": False, "reason": f"non-positive OHLC at {item.timestamp}"}
        if item.high < max(item.open, item.close) or item.low > min(item.open, item.close):
            return {"ok": False, "reason": f"incoherent OHLC at {item.timestamp}"}
    return {
        "ok": True,
        "count": len(bars),
        "start": timestamps[0],
        "end": timestamps[-1],
    }


def validate_iv_observations(observations: Sequence[IvObservation]) -> dict[str, object]:
    if not observations:
        return {"ok": False, "reason": "no historical IV observations"}
    timestamps = [item.timestamp for item in observations]
    if timestamps != sorted(timestamps):
        return {"ok": False, "reason": "IV observations must be sorted by timestamp"}
    valid = [
        item
        for item in observations
        if math.isfinite(item.annualized_iv) and item.annualized_iv > 0
    ]
    if not valid:
        return {"ok": False, "reason": "historical IV observations contain no positive finite IV"}
    return {
        "ok": True,
        "count": len(valid),
        "start": valid[0].timestamp,
        "end": valid[-1].timestamp,
        "sources": sorted({item.source for item in valid}),
        "source_quality": sorted({item.source_quality for item in valid}),
    }


def implied_volatility_from_option(
    quote: OptionQuote,
    *,
    risk_free_rate: float = 0.02,
    max_iterations: int = 80,
    tolerance: float = 1e-6,
) -> float | None:
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return None
    midpoint = (quote.bid + quote.ask) / 2.0
    time_to_expiry = max((quote.expiry - quote.as_of).days / 365.0, 1.0 / 365.0)
    intrinsic = (
        max(quote.underlying - quote.strike, 0.0)
        if quote.option_type == "call"
        else max(quote.strike - quote.underlying, 0.0)
    )
    if midpoint <= intrinsic:
        return None

    low = 0.0001
    high = 5.0
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        value = _black_scholes_price(
            option_type=quote.option_type,
            spot=quote.underlying,
            strike=quote.strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=mid,
        )
        if abs(value - midpoint) < tolerance:
            return mid
        if value > midpoint:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def atm_iv_snapshot_from_quotes(
    quotes: Sequence[OptionQuote],
    *,
    risk_free_rate: float = 0.02,
) -> dict[str, object]:
    if not quotes:
        return {"ok": False, "reason": "no option quotes"}
    by_expiry = sorted({quote.expiry for quote in quotes})
    chosen_expiry = by_expiry[0]
    expiry_quotes = [quote for quote in quotes if quote.expiry == chosen_expiry]
    chosen_strike = min(expiry_quotes, key=lambda item: abs(item.strike - item.underlying)).strike
    atm_quotes = [quote for quote in expiry_quotes if quote.strike == chosen_strike]
    ivs = [
        value
        for quote in atm_quotes
        if (value := implied_volatility_from_option(quote, risk_free_rate=risk_free_rate))
        is not None
    ]
    if not ivs:
        return {
            "ok": False,
            "reason": "ATM quotes could not be inverted to IV",
            "expiry": chosen_expiry.isoformat(),
            "strike": chosen_strike,
        }
    return {
        "ok": True,
        "annualized_iv": statistics.fmean(ivs),
        "iv_count": len(ivs),
        "expiry": chosen_expiry.isoformat(),
        "strike": chosen_strike,
        "underlying": atm_quotes[0].underlying,
        "as_of": atm_quotes[0].as_of.isoformat(),
        "source_quality": "forward_proxy_from_atm_mid_quotes",
    }


def _cell_rows(
    bars: Sequence[IndexBar],
    iv_by_timestamp: Mapping[int, IvObservation],
    horizon: int,
    method: RvMethod,
    config: VolGapConfig,
) -> list[VolGapRow]:
    rows: list[VolGapRow] = []
    for index in range(len(bars) - horizon):
        bar = bars[index]
        iv = iv_by_timestamp.get(bar.timestamp)
        if iv is None or iv.annualized_iv <= 0 or not math.isfinite(iv.annualized_iv):
            continue
        future = bars[index + 1 : index + horizon + 1]
        rv = _forward_rv(future, method=method, annualization=config.annualization_periods)
        if rv is None:
            continue
        rows.append(
            VolGapRow(
                timestamp=bar.timestamp,
                horizon=horizon,
                rv_method=method,
                iv=iv.annualized_iv,
                rv_forward=rv,
                variance_gap=(rv * rv) - (iv.annualized_iv * iv.annualized_iv),
                iv_minus_rv=iv.annualized_iv - rv,
                regime=_regime_at(bars, index, config),
            )
        )
    return rows


def _forward_rv(
    bars: Sequence[IndexBar],
    *,
    method: RvMethod,
    annualization: int,
) -> float | None:
    if len(bars) < 2:
        return None
    if method == "close_to_close":
        returns = [
            math.log(bars[index].close / bars[index - 1].close)
            for index in range(1, len(bars))
            if bars[index].close > 0 and bars[index - 1].close > 0
        ]
        if len(returns) < 2:
            return None
        return statistics.stdev(returns) * math.sqrt(annualization)
    if method == "parkinson":
        terms = [math.log(item.high / item.low) ** 2 for item in bars if item.low > 0]
        if not terms:
            return None
        variance = statistics.fmean(terms) * annualization / (4.0 * math.log(2.0))
        return math.sqrt(max(variance, 0.0))
    terms = []
    for item in bars:
        if min(item.open, item.high, item.low, item.close) <= 0:
            continue
        hl = math.log(item.high / item.low)
        co = math.log(item.close / item.open)
        terms.append(0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co)
    if not terms:
        return None
    return math.sqrt(max(statistics.fmean(terms) * annualization, 0.0))


def _regime_at(bars: Sequence[IndexBar], index: int, config: VolGapConfig) -> Regime:
    lookback = 20
    if index < lookback:
        return "unknown"
    trailing = bars[index - lookback : index + 1]
    rv = _forward_rv(trailing, method="close_to_close", annualization=config.annualization_periods)
    if rv is None:
        return "unknown"
    if rv < config.calm_trailing_rv:
        return "calm"
    if rv > config.crisis_trailing_rv:
        return "crisis"
    return "normal"


def _summarize_cell(
    rows: Sequence[VolGapRow],
    *,
    horizon: int,
    method: RvMethod,
    config: VolGapConfig,
) -> dict[str, object]:
    gaps = [row.variance_gap for row in rows]
    p_value = sign_test_p_value(gaps, alternative="greater") if gaps else 1.0
    return {
        "horizon": horizon,
        "rv_method": method,
        "sample_count": len(gaps),
        "mean_variance_gap": _mean_or_none(gaps),
        "median_variance_gap": statistics.median(gaps) if gaps else None,
        "buyer_positive_rate": _positive_rate(gaps),
        "seller_positive_rate": _positive_rate([-value for value in gaps]),
        "mean_iv_minus_rv": _mean_or_none([row.iv_minus_rv for row in rows]),
        "buyer_sign_test_p_value": p_value,
        "min_samples_required": config.min_samples_per_cell,
        "buyer_bh_fdr_pass": False,
    }


def _regime_summary(rows: Sequence[VolGapRow]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for regime in ("calm", "normal", "crisis", "unknown"):
        gaps = [row.variance_gap for row in rows if row.regime == regime]
        output[regime] = {
            "sample_count": len(gaps),
            "mean_variance_gap": _mean_or_none(gaps),
            "buyer_positive_rate": _positive_rate(gaps),
        }
    return output


def _insufficient(
    *,
    reason: str,
    bars_report: Mapping[str, object],
    iv_report: Mapping[str, object],
    config: VolGapConfig,
    iv_source: str,
) -> dict[str, object]:
    return {
        "verdict": "INSUFFICIENT",
        "reason": reason,
        "candidate_count_n": trial_count(config),
        "raw_survivors": 0,
        "fdr_survivors": 0,
        "iv_source": iv_source,
        "data_gate": {"bars": dict(bars_report), "iv": dict(iv_report)},
        "multiple_testing": {
            "candidate_count_n": trial_count(config),
            "method": "not run; historical IV gate failed",
            "pbo": {"valid": False, "reason": "not run under INSUFFICIENT data gate"},
        },
        "forward_collection": {
            "required": True,
            "target": "daily ATM MO IV plus 000852 OHLC",
            "private_path": "${AEGIS_STRATEGIES_ROOT}/incubating/olympus74/forward/",
        },
        "cost_model": {
            "scope": "gross diagnostic",
            "delta_hedge_costs": "not modeled",
            "funding": "N/A for listed index options/futures",
        },
        "safety": {
            "live_trading": False,
            "order_path": False,
            "account_or_gui_access": False,
        },
    }


def _black_scholes_price(
    *,
    option_type: Literal["call", "put"],
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * math.exp(
            -risk_free_rate * time_to_expiry
        ) * _normal_cdf(d2)
    return strike * math.exp(-risk_free_rate * time_to_expiry) * _normal_cdf(
        -d2
    ) - spot * _normal_cdf(-d1)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _positive_rate(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value > 0.0) / len(values)


def _float_field(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool):
        raise TypeError(f"{key} must be numeric, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{key} must be numeric")


def _int_field(row: Mapping[str, object], key: str) -> int:
    value = row[key]
    if isinstance(value, bool):
        raise TypeError(f"{key} must be int, got bool")
    if isinstance(value, int):
        return value
    raise TypeError(f"{key} must be int")
