from __future__ import annotations

from typing import Any, Literal, cast

HealthcheckVerdict = Literal["PASS", "PASS_WITH_WARN", "BLOCK"]
CompetitionStatus = Literal["promote_candidate", "hold", "retire_candidate"]
CompetitionWeights = dict[str, float]


DEFAULT_WEIGHTS: CompetitionWeights = {
    "median_return": 0.47,
    "beat_bh_share": 0.48,
    "sharpe": 0.03,
    "max_dd": 0.02,
}

DEFAULT_PROMOTE_TOP_N = 3
DEFAULT_RETIRE_BOTTOM_STREAK = 3


def rank_strategies(
    entries: list[dict[str, Any]],
    *,
    weights: CompetitionWeights | None = None,
    promote_top_n: int = DEFAULT_PROMOTE_TOP_N,
    retire_bottom_streak: int = DEFAULT_RETIRE_BOTTOM_STREAK,
) -> list[dict[str, Any]]:
    merged_weights = _weights(weights)
    scored = [_scored_entry(entry, merged_weights) for entry in entries]
    ordered = sorted(
        scored,
        key=lambda item: (
            item["healthcheck_verdict"] == "BLOCK",
            -cast(float, item["score"]),
            str(item["strategy"]),
            _params_sort_key(cast(dict[str, Any], item["params"])),
        ),
    )

    leaderboard: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
        row["status"] = _status(
            row,
            rank=index,
            promote_top_n=promote_top_n,
            retire_bottom_streak=retire_bottom_streak,
        )
        row.pop("_bottom_streak", None)
        leaderboard.append(row)
    return leaderboard


def _weights(overrides: CompetitionWeights | None) -> CompetitionWeights:
    return {**DEFAULT_WEIGHTS, **(overrides or {})}


def _scored_entry(
    entry: dict[str, Any],
    weights: CompetitionWeights,
) -> dict[str, Any]:
    metrics = _key_metrics(entry)
    verdict = _healthcheck_verdict(entry)
    return {
        "strategy": str(entry.get("strategy", "")),
        "params": _params(entry),
        "score": round(_score(metrics, weights), 6),
        "rank": 0,
        "healthcheck_verdict": verdict,
        "key_metrics": metrics,
        "status": "hold",
        "_bottom_streak": _entry_bottom_streak(entry),
    }


def _score(metrics: dict[str, Any], weights: CompetitionWeights) -> float:
    median_return = _metric_fraction(metrics, "median_return")
    beat_bh_share = _metric_float(metrics, "beat_bh_share")
    sharpe = _metric_float(metrics, "sharpe")
    max_dd = abs(_metric_fraction(metrics, "max_dd"))
    return (
        weights["median_return"] * median_return
        + weights["beat_bh_share"] * beat_bh_share
        + weights["sharpe"] * sharpe
        - weights["max_dd"] * max_dd
    )


def _status(
    row: dict[str, Any],
    *,
    rank: int,
    promote_top_n: int,
    retire_bottom_streak: int,
) -> CompetitionStatus:
    if row["healthcheck_verdict"] == "BLOCK":
        return "retire_candidate"
    if (
        row["healthcheck_verdict"] == "PASS"
        and promote_top_n > 0
        and rank <= promote_top_n
    ):
        return "promote_candidate"
    if retire_bottom_streak > 0 and _bottom_streak(row) >= retire_bottom_streak:
        return "retire_candidate"
    return "hold"


def _key_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("key_metrics")
    if not isinstance(source, dict):
        source = entry.get("metrics")
    if not isinstance(source, dict):
        source = entry

    return {
        "median_return": _first_float(source, ("median_return", "return_pct_median")),
        "beat_bh_share": _first_float(
            source,
            ("beat_bh_share", "beat_buy_hold_share", "beat_benchmark_share"),
        ),
        "sharpe": _first_float(source, ("sharpe", "sharpe_median", "avg_sharpe")),
        "max_dd": _first_float(source, ("max_dd", "max_drawdown", "max_drawdown_pct")),
        "exit_breakdown": _exit_breakdown(source),
    }


def _healthcheck_verdict(entry: dict[str, Any]) -> HealthcheckVerdict:
    raw = entry.get("healthcheck_verdict")
    if raw is None and isinstance(entry.get("healthcheck"), dict):
        raw = cast(dict[str, Any], entry["healthcheck"]).get("verdict")
    if raw == "PASS":
        return "PASS"
    if raw == "BLOCK":
        return "BLOCK"
    return "PASS_WITH_WARN"


def _params(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry.get("params")
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    request = entry.get("request")
    if isinstance(request, dict) and isinstance(request.get("params"), dict):
        return cast(dict[str, Any], request["params"])
    return {}


def _bottom_streak(row: dict[str, Any]) -> int:
    raw = row.get("_bottom_streak")
    if isinstance(raw, int):
        return raw
    return 0


def _entry_bottom_streak(entry: dict[str, Any]) -> int:
    history = entry.get("history")
    raw: object = None
    if isinstance(history, dict):
        raw = history.get("bottom_streak")
    if raw is None:
        raw = entry.get("bottom_streak")
    if not isinstance(raw, int | float | str):
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _first_float(source: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int | float | str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def _metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _metric_fraction(metrics: dict[str, Any], key: str) -> float:
    value = _metric_float(metrics, key)
    if abs(value) > 1:
        return value / 100
    return value


def _exit_breakdown(source: dict[str, Any]) -> dict[str, int]:
    raw = source.get("exit_breakdown")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, int | float):
            result[str(key)] = int(value)
    return result


def _params_sort_key(params: dict[str, Any]) -> str:
    return repr(sorted((str(key), repr(value)) for key, value in params.items()))
