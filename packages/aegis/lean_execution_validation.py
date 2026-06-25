from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

LeanExecutionState = Literal["EDGE", "NO_EDGE", "INSUFFICIENT"]
LeanExecutionVerdict = Literal[
    "EXECUTION_VALID_PENDING_FINAL_GATE",
    "EXECUTION_FAIL",
    "INSUFFICIENT",
]
DataAdequacy = Literal["adequate", "limited", "blocked"]


@dataclass(frozen=True)
class LeanExecutionGate:
    max_drawdown_limit: float = -0.30
    min_order_count: int = 1
    require_fees: bool = True
    require_slippage: bool = True
    require_benchmark: bool = True


def validate_lean_execution_report(
    *,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    gate: LeanExecutionGate | None = None,
) -> Mapping[str, Any]:
    """Validate a private LEAN backtest report as an execution-stage Aegis gate.

    This deliberately does not run LEAN or place orders. LEAN is treated as an
    execution simulator whose report is ingested after Aegis has already judged
    the research hypothesis.
    """

    active_gate = gate or LeanExecutionGate()
    spec_errors = _validate_spec(spec)
    report_errors = _validate_report(report, active_gate)
    if spec_errors or report_errors:
        reason = "; ".join((*spec_errors, *report_errors))
        return _payload(
            state="INSUFFICIENT",
            verdict="INSUFFICIENT",
            reason=reason,
            spec=spec,
            report=report,
            gate=active_gate,
            data_adequacy="blocked",
            unlock_condition=reason,
        )

    metrics = _metrics(report)
    benchmark = _benchmark(report)
    annualized_return = _required_float(metrics, "annualized_return")
    sharpe = _required_float(metrics, "sharpe")
    max_drawdown = _required_float(metrics, "max_drawdown")
    benchmark_annualized_return = _required_float(benchmark, "annualized_return")
    benchmark_sharpe = _required_float(benchmark, "sharpe")
    order_count = _required_int(report, "order_count")
    fees = _required_float(report, "total_fees")
    slippage = _required_float(report, "total_slippage")
    tail_safe = max_drawdown >= active_gate.max_drawdown_limit
    beats_benchmark = (
        annualized_return > benchmark_annualized_return and sharpe > benchmark_sharpe
    )
    execution_costed = (
        (not active_gate.require_fees or fees > 0.0)
        and (not active_gate.require_slippage or slippage > 0.0)
        and order_count >= active_gate.min_order_count
    )
    if tail_safe and beats_benchmark and execution_costed:
        state: LeanExecutionState = "EDGE"
        verdict: LeanExecutionVerdict = "EXECUTION_VALID_PENDING_FINAL_GATE"
        reason = "LEAN execution report beat benchmark, included costs, and passed tail gate"
    else:
        state = "NO_EDGE"
        verdict = "EXECUTION_FAIL"
        failed = []
        if not tail_safe:
            failed.append("max_drawdown breached execution tail gate")
        if not beats_benchmark:
            failed.append("risk-adjusted result did not beat benchmark")
        if not execution_costed:
            failed.append("execution report missing required cost/order evidence")
        reason = "; ".join(failed)
    adequacy = _data_adequacy(spec, report, state)
    unlock = (
        "N/A"
        if adequacy == "adequate"
        else str(
            spec.get(
                "unlock_condition",
                "live-like paper execution with broker-native fills and production risk controls",
            )
        )
    )
    return _payload(
        state=state,
        verdict=verdict,
        reason=reason,
        spec=spec,
        report=report,
        gate=active_gate,
        data_adequacy=adequacy,
        unlock_condition=unlock,
    )


def _validate_spec(spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if spec.get("engine") != "lean":
        errors.append("spec.engine must be 'lean'")
    if spec.get("mode") not in {"backtest", "paper"}:
        errors.append("spec.mode must be 'backtest' or 'paper'")
    if spec.get("live_trading") is not False:
        errors.append("spec.live_trading must be false")
    if spec.get("read_only") is not True:
        errors.append("spec.read_only must be true")
    if spec.get("source_aegis_state") != "EDGE":
        errors.append("spec.source_aegis_state must be EDGE before LEAN validation")
    if not _non_empty_str(spec.get("id")):
        errors.append("spec.id is required")
    if not _non_empty_str(spec.get("source_hypothesis_id")):
        errors.append("spec.source_hypothesis_id is required")
    if _contains_forbidden_text(spec):
        errors.append("spec contains forbidden live/account credential marker")
    return errors


def _validate_report(report: Mapping[str, Any], gate: LeanExecutionGate) -> list[str]:
    errors: list[str] = []
    if report.get("engine") != "lean":
        errors.append("report.engine must be 'lean'")
    if report.get("live_trading") is not False:
        errors.append("report.live_trading must be false")
    if gate.require_benchmark and not isinstance(report.get("benchmark"), Mapping):
        errors.append("report.benchmark is required")
    if not isinstance(report.get("metrics"), Mapping):
        errors.append("report.metrics is required")
    for key in ("order_count", "total_fees", "total_slippage"):
        if _optional_float(report.get(key)) is None:
            errors.append(f"report.{key} must be numeric")
    metrics = _metrics(report)
    benchmark = _benchmark(report)
    for key in ("annualized_return", "sharpe", "max_drawdown"):
        if _optional_float(metrics.get(key)) is None:
            errors.append(f"report.metrics.{key} must be numeric")
    for key in ("annualized_return", "sharpe"):
        if _optional_float(benchmark.get(key)) is None:
            errors.append(f"report.benchmark.{key} must be numeric")
    if _contains_forbidden_text(report):
        errors.append("report contains forbidden live/account credential marker")
    return errors


def _payload(
    *,
    state: LeanExecutionState,
    verdict: LeanExecutionVerdict,
    reason: str,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    gate: LeanExecutionGate,
    data_adequacy: DataAdequacy,
    unlock_condition: str,
) -> Mapping[str, Any]:
    return {
        "status": "OK" if state != "INSUFFICIENT" else "INSUFFICIENT",
        "state": state,
        "verdict": verdict,
        "reason": reason,
        "data_adequacy": data_adequacy,
        "unlock_condition": unlock_condition,
        "source_hypothesis_id": spec.get("source_hypothesis_id"),
        "engine": "lean",
        "mode": spec.get("mode"),
        "standard_metrics": _metrics(report),
        "benchmark_metrics": _benchmark(report),
        "execution_metrics": {
            "order_count": _optional_int(report.get("order_count")),
            "total_fees": _optional_float(report.get("total_fees")),
            "total_slippage": _optional_float(report.get("total_slippage")),
            "max_drawdown_limit": gate.max_drawdown_limit,
        },
        "safety": {
            "read_only": spec.get("read_only") is True,
            "live_trading": False,
            "wallet_or_order_access": False,
            "account_access": False,
            "broker_connection": False,
        },
    }


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _benchmark(report: Mapping[str, Any]) -> Mapping[str, Any]:
    benchmark = report.get("benchmark")
    return benchmark if isinstance(benchmark, Mapping) else {}


def _data_adequacy(
    spec: Mapping[str, Any], report: Mapping[str, Any], state: LeanExecutionState
) -> DataAdequacy:
    if state == "INSUFFICIENT":
        return "blocked"
    if spec.get("data_adequacy") == "adequate" and report.get("executable_fills") is True:
        return "adequate"
    return "limited"


def _contains_forbidden_text(value: object) -> bool:
    forbidden = (
        "api_secret",
        "password",
        "private_key",
        "wallet",
        "account_number",
        "live trading",
        "placeorder",
    )
    text = repr(value).lower()
    return any(marker in text for marker in forbidden)


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    parsed = _optional_float(mapping.get(key))
    if parsed is None:
        raise ValueError(f"expected numeric {key}")
    return parsed


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    parsed = _optional_int(mapping.get(key))
    if parsed is None:
        raise ValueError(f"expected integer {key}")
    return parsed


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
