from __future__ import annotations

import math
import statistics
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]

FactorMode = Literal["time_series", "cross_sectional"]
FactorKind = Literal["return", "momentum", "volatility", "volume_zscore", "close_to_sma"]

MIN_CORRELATION_POINTS = 3


def evaluate_ohlcv_factor_ic(
    symbol_frames: dict[str, pd.DataFrame],
    factor_specs: list[dict[str, Any]],
    *,
    label_periods: int = 1,
    groups: int = 5,
    mode: FactorMode = "time_series",
    ic_window: int = 60,
    redundancy_threshold: float = 0.8,
    icir_threshold: float = 0.3,
) -> dict[str, Any]:
    if not symbol_frames:
        return _empty_report("INSUFFICIENT_DATA", "no symbol frames supplied", mode)
    if not factor_specs:
        return _empty_report("INSUFFICIENT_DATA", "no factor specs supplied", mode)
    if label_periods <= 0:
        raise ValueError("label_periods must be positive")
    if groups < 2:
        raise ValueError("groups must be >= 2")
    if ic_window < MIN_CORRELATION_POINTS:
        raise ValueError(f"ic_window must be >= {MIN_CORRELATION_POINTS}")
    if not 0 < redundancy_threshold <= 1:
        raise ValueError("redundancy_threshold must be in (0, 1]")

    panel = _factor_panel(symbol_frames, factor_specs, label_periods)
    factor_names = [
        str(spec.get("name") or _default_factor_name(spec))
        for spec in factor_specs
    ]
    duplicate_names = sorted({name for name in factor_names if factor_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate factor names: {', '.join(duplicate_names)}")

    factors: dict[str, Any] = {}
    for name in factor_names:
        observations = _observations(panel, name)
        rank_ic = _ic_summary(observations, name, mode, ic_window, rank=True)
        pearson_ic = _ic_summary(observations, name, mode, ic_window, rank=False)
        monotonicity = _monotonicity(observations, name, groups, mode)
        autocorrelation = _autocorrelation(observations, name)
        edge = _edge_assessment(rank_ic, monotonicity, icir_threshold)
        factors[name] = {
            "status": "OK" if observations["label"].notna().any() else "INSUFFICIENT_DATA",
            "observations": int(len(observations.dropna(subset=[name, "label"]))),
            "rank_ic": rank_ic,
            "pearson_ic": pearson_ic,
            "monotonicity": monotonicity,
            "autocorrelation": autocorrelation,
            "edge": edge,
        }

    redundancy = _redundancy(panel, factor_names, factors, redundancy_threshold)
    report = {
        "status": "OK",
        "mode": mode,
        "label": {"kind": "future_return", "periods": label_periods},
        "data": {
            "symbols": sorted(symbol_frames),
            "bars_by_symbol": {symbol: int(len(frame)) for symbol, frame in symbol_frames.items()},
        },
        "thresholds": {
            "icir_min_abs": icir_threshold,
            "groups": groups,
            "ic_window": ic_window,
            "redundancy_abs_corr": redundancy_threshold,
        },
        "factors": factors,
        "redundancy": redundancy,
        "disclaimer": "candidates-only factor evaluation; no trading signal or order path",
    }
    report["readable_report"] = render_factor_ic_report(report)
    return report


def render_factor_ic_report(report: dict[str, Any]) -> str:
    label = cast(dict[str, Any], report.get("label", {}))
    lines = [
        "Factor IC evaluation",
        f"mode={report.get('mode')} label={label.get('periods')} periods",
        "candidates-only; no trading signal or order path",
    ]
    factors = report.get("factors")
    if isinstance(factors, dict):
        for name, raw in factors.items():
            if not isinstance(raw, dict):
                continue
            rank_ic = cast(dict[str, Any], raw.get("rank_ic", {}))
            monotonicity = cast(dict[str, Any], raw.get("monotonicity", {}))
            autocorr = cast(dict[str, Any], raw.get("autocorrelation", {}))
            edge = cast(dict[str, Any], raw.get("edge", {}))
            lines.append(
                f"- {name}: rank_ic_mean={_fmt(rank_ic.get('mean'))}, "
                f"rank_icir={_fmt(rank_ic.get('icir'))}, "
                f"t={_fmt(rank_ic.get('t_value'))}, "
                f"ic_positive_share={_fmt(rank_ic.get('positive_share'))}, "
                f"monotonic={monotonicity.get('is_monotonic')}, "
                f"top_bottom={_fmt(monotonicity.get('top_bottom_return'))}, "
                f"autocorr_lag1={_fmt(autocorr.get('lag_1'))}, "
                f"edge={edge.get('has_predictive_power')} ({edge.get('reason')})"
            )
    redundancy = report.get("redundancy")
    if isinstance(redundancy, dict):
        pairs = redundancy.get("high_correlation_pairs")
        lines.append(f"redundant_pairs={len(pairs) if isinstance(pairs, list) else 0}")
        drops = redundancy.get("suggested_drop")
        if isinstance(drops, list) and drops:
            lines.append(f"suggested_drop={', '.join(str(item) for item in drops)}")
    return "\n".join(lines)


def _empty_report(status: str, reason: str, mode: FactorMode) -> dict[str, Any]:
    report = {
        "status": status,
        "mode": mode,
        "reason": reason,
        "factors": {},
        "redundancy": {
            "correlation_matrix": {},
            "high_correlation_pairs": [],
            "suggested_keep": [],
            "suggested_drop": [],
        },
        "disclaimer": "candidates-only factor evaluation; no trading signal or order path",
    }
    report["readable_report"] = render_factor_ic_report(report)
    return report


def _factor_panel(
    symbol_frames: dict[str, pd.DataFrame],
    factor_specs: list[dict[str, Any]],
    label_periods: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, raw_frame in sorted(symbol_frames.items()):
        frame = _normalized_ohlcv(raw_frame)
        data = pd.DataFrame(index=frame.index.copy())
        data["symbol"] = symbol
        data["label"] = frame["Close"].shift(-label_periods) / frame["Close"] - 1
        for spec in factor_specs:
            name = str(spec.get("name") or _default_factor_name(spec))
            data[name] = _factor_series(frame, spec)
        rows.append(data.reset_index(names="timestamp"))
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    return cast(pd.DataFrame, panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True))


def _normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")
    normalized = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    normalized = normalized.sort_index()
    return cast(pd.DataFrame, normalized)


def _factor_series(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    kind = str(spec.get("kind", ""))
    window = _positive_int(spec.get("window"), default=1)
    close = frame["Close"]
    if kind in {"return", "momentum"}:
        return cast(pd.Series, close / close.shift(window) - 1)
    if kind == "volatility":
        return cast(pd.Series, close.pct_change().rolling(window).std())
    if kind == "volume_zscore":
        volume = frame["Volume"]
        mean = volume.rolling(window).mean()
        std = volume.rolling(window).std()
        return cast(pd.Series, (volume - mean) / std.replace(0, pd.NA))
    if kind == "close_to_sma":
        mean = close.rolling(window).mean()
        return cast(pd.Series, close / mean - 1)
    raise ValueError(f"unsupported factor kind: {kind}")


def _default_factor_name(spec: dict[str, Any]) -> str:
    kind = str(spec.get("kind", "factor"))
    window = _positive_int(spec.get("window"), default=1)
    return f"{kind}_{window}"


def _positive_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("window must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("window must be a positive integer") from exc
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        raise ValueError("window must be a positive integer")
    if parsed <= 0:
        raise ValueError("window must be a positive integer")
    return parsed


def _observations(panel: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    columns = ["timestamp", "symbol", factor_name, "label"]
    frame = panel[columns].copy()
    frame = frame.replace([math.inf, -math.inf], pd.NA)
    return cast(pd.DataFrame, frame.dropna(subset=[factor_name, "label"]))


def _ic_summary(
    observations: pd.DataFrame,
    factor_name: str,
    mode: FactorMode,
    ic_window: int,
    *,
    rank: bool,
) -> dict[str, Any]:
    values: list[float]
    if mode == "cross_sectional":
        values = _cross_sectional_ic_values(observations, factor_name, rank)
    else:
        values = _time_series_ic_values(observations, factor_name, ic_window, rank)
    return _series_summary(values)


def _time_series_ic_values(
    observations: pd.DataFrame,
    factor_name: str,
    ic_window: int,
    rank: bool,
) -> list[float]:
    result: list[float] = []
    for _, group in observations.sort_values("timestamp").groupby("symbol"):
        if len(group) < ic_window:
            continue
        for start in range(0, len(group) - ic_window + 1):
            window = group.iloc[start : start + ic_window]
            corr = _corr(window[factor_name], window["label"], rank)
            if corr is not None:
                result.append(corr)
    return result


def _cross_sectional_ic_values(
    observations: pd.DataFrame,
    factor_name: str,
    rank: bool,
) -> list[float]:
    result: list[float] = []
    for _, group in observations.groupby("timestamp"):
        if len(group) < MIN_CORRELATION_POINTS:
            continue
        corr = _corr(group[factor_name], group["label"], rank)
        if corr is not None:
            result.append(corr)
    return result


def _corr(left: pd.Series, right: pd.Series, rank: bool) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < MIN_CORRELATION_POINTS:
        return None
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    if rank:
        left_values = frame["left"].rank(method="average")
        right_values = frame["right"].rank(method="average")
    else:
        left_values = frame["left"]
        right_values = frame["right"]
    value = left_values.corr(right_values, method="pearson")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _series_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "status": "INSUFFICIENT_DATA",
            "n": 0,
            "mean": None,
            "std": None,
            "icir": None,
            "t_value": None,
            "positive_share": None,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    icir = None if std == 0 else mean / std
    t_value = None if std == 0 else mean / (std / math.sqrt(len(values)))
    return {
        "status": "OK",
        "n": len(values),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "icir": None if icir is None else round(icir, 6),
        "t_value": None if t_value is None else round(t_value, 6),
        "positive_share": round(sum(1 for value in values if value > 0) / len(values), 6),
    }


def _monotonicity(
    observations: pd.DataFrame,
    factor_name: str,
    groups: int,
    mode: FactorMode,
) -> dict[str, Any]:
    if mode == "cross_sectional":
        return _cross_sectional_monotonicity(observations, factor_name, groups)
    return _time_series_monotonicity(observations, factor_name, groups)


def _time_series_monotonicity(
    observations: pd.DataFrame,
    factor_name: str,
    groups: int,
) -> dict[str, Any]:
    if len(observations) < groups * 2:
        return _insufficient_monotonicity(groups, "not enough observations for quantile groups")
    labels = _quantile_labels(observations[factor_name], groups)
    if labels is None:
        return _insufficient_monotonicity(groups, "not enough distinct factor values")
    grouped = observations.assign(factor_group=labels)
    return _group_return_summary(grouped, groups)


def _cross_sectional_monotonicity(
    observations: pd.DataFrame,
    factor_name: str,
    groups: int,
) -> dict[str, Any]:
    grouped_rows: list[pd.DataFrame] = []
    for _, period in observations.groupby("timestamp"):
        if len(period) < groups:
            continue
        labels = _quantile_labels(period[factor_name], groups)
        if labels is not None:
            grouped_rows.append(period.assign(factor_group=labels))
    if not grouped_rows:
        return _insufficient_monotonicity(groups, "not enough symbols per period")
    grouped = pd.concat(grouped_rows, axis=0, ignore_index=True)
    return _group_return_summary(grouped, groups)


def _quantile_labels(values: pd.Series, groups: int) -> pd.Series | None:
    try:
        labels = pd.qcut(values, q=groups, labels=False, duplicates="drop")
    except ValueError:
        return None
    if labels.isna().any() or int(labels.max()) + 1 < groups:
        return None
    return cast(pd.Series, labels.astype(int))


def _group_return_summary(grouped: pd.DataFrame, groups: int) -> dict[str, Any]:
    returns: list[float] = []
    counts: list[int] = []
    for group_id in range(groups):
        subset = grouped[grouped["factor_group"] == group_id]
        counts.append(int(len(subset)))
        returns.append(float(subset["label"].mean()))
    increasing = all(returns[index + 1] >= returns[index] for index in range(len(returns) - 1))
    decreasing = all(returns[index + 1] <= returns[index] for index in range(len(returns) - 1))
    direction = "positive" if increasing else "negative" if decreasing else "none"
    top_bottom = returns[-1] - returns[0]
    return {
        "status": "OK",
        "groups": groups,
        "group_mean_forward_returns": [round(value, 6) for value in returns],
        "group_counts": counts,
        "is_monotonic": bool(increasing or decreasing),
        "direction": direction,
        "top_bottom_return": round(top_bottom, 6),
    }


def _insufficient_monotonicity(groups: int, reason: str) -> dict[str, Any]:
    return {
        "status": "INSUFFICIENT_DATA",
        "groups": groups,
        "reason": reason,
        "group_mean_forward_returns": [],
        "group_counts": [],
        "is_monotonic": False,
        "direction": "none",
        "top_bottom_return": None,
    }


def _autocorrelation(observations: pd.DataFrame, factor_name: str) -> dict[str, Any]:
    values: list[float] = []
    for _, group in observations.sort_values("timestamp").groupby("symbol"):
        series = cast(pd.Series, group[factor_name].dropna())
        if len(series) < MIN_CORRELATION_POINTS:
            continue
        corr = _corr(series.iloc[1:], series.shift(1).dropna(), rank=False)
        if corr is not None:
            values.append(corr)
    if not values:
        return {
            "status": "INSUFFICIENT_DATA",
            "lag_1": None,
            "turnover_proxy": None,
            "note": "turnover_proxy is 1 - abs(lag_1_autocorrelation)",
        }
    mean = statistics.fmean(values)
    return {
        "status": "OK",
        "lag_1": round(mean, 6),
        "turnover_proxy": round(max(0.0, min(1.0, 1 - abs(mean))), 6),
        "note": "turnover_proxy is 1 - abs(lag_1_autocorrelation)",
    }


def _edge_assessment(
    rank_ic: dict[str, Any],
    monotonicity: dict[str, Any],
    icir_threshold: float,
) -> dict[str, Any]:
    raw_icir = rank_ic.get("icir")
    icir = float(raw_icir) if isinstance(raw_icir, int | float) else None
    monotonic = monotonicity.get("is_monotonic") is True
    if icir is None:
        return {
            "has_predictive_power": False,
            "reason": "sample insufficient for ICIR; no edge signal",
        }
    if abs(icir) >= icir_threshold and monotonic:
        return {
            "has_predictive_power": True,
            "reason": "rank ICIR meets threshold and grouped returns are monotonic",
        }
    if abs(icir) < icir_threshold and not monotonic:
        reason = "rank ICIR below threshold and grouped returns are not monotonic; no edge signal"
    elif abs(icir) < icir_threshold:
        reason = "rank ICIR below threshold; no edge signal"
    else:
        reason = "grouped returns are not monotonic; no edge signal"
    return {"has_predictive_power": False, "reason": reason}


def _redundancy(
    panel: pd.DataFrame,
    factor_names: list[str],
    factors: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    if len(factor_names) < 2:
        return {
            "correlation_matrix": {
                factor_names[0]: {factor_names[0]: 1.0}
            } if factor_names else {},
            "high_correlation_pairs": [],
            "suggested_keep": list(factor_names),
            "suggested_drop": [],
        }
    matrix_frame = panel[factor_names].replace([math.inf, -math.inf], pd.NA).dropna(how="all")
    corr = matrix_frame.corr(method="spearman")
    pairs: list[dict[str, Any]] = []
    drop: set[str] = set()
    for left_index, left in enumerate(factor_names):
        for right in factor_names[left_index + 1 :]:
            raw_value = corr.loc[left, right]
            if pd.isna(raw_value):
                continue
            value = float(raw_value)
            if abs(value) < threshold:
                continue
            keep, remove = _redundancy_keep_drop(left, right, factors)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_corr": round(value, 6),
                    "suggest_keep": keep,
                    "suggest_drop": remove,
                    "reason": "absolute factor correlation exceeds redundancy threshold",
                }
            )
            drop.add(remove)
    return {
        "correlation_matrix": _rounded_matrix(corr, factor_names),
        "high_correlation_pairs": pairs,
        "suggested_keep": [name for name in factor_names if name not in drop],
        "suggested_drop": sorted(drop),
    }


def _redundancy_keep_drop(
    left: str,
    right: str,
    factors: dict[str, Any],
) -> tuple[str, str]:
    left_score = _abs_icir(factors.get(left))
    right_score = _abs_icir(factors.get(right))
    if right_score > left_score:
        return right, left
    return left, right


def _abs_icir(raw_factor: object) -> float:
    if not isinstance(raw_factor, dict):
        return -1.0
    rank_ic = raw_factor.get("rank_ic")
    if not isinstance(rank_ic, dict):
        return -1.0
    raw_icir = rank_ic.get("icir")
    if isinstance(raw_icir, int | float):
        return abs(float(raw_icir))
    return -1.0


def _rounded_matrix(
    corr: pd.DataFrame, factor_names: list[str]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for left in factor_names:
        row: dict[str, float | None] = {}
        for right in factor_names:
            raw_value = corr.loc[left, right]
            row[right] = None if pd.isna(raw_value) else round(float(raw_value), 6)
        result[left] = row
    return result


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return "NA"
