#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from aegis.backtest_core import BacktestDiscipline, HypothesisSpec, StandardVerdict, run_backtest
from aegis.microstructure_perp_runner import run_microstructure_perp_from_spec
from aegis.private_paths import PrivatePathError, private_root_from_env, resolve_private_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_NAME = "hypothesis_registry.jsonl"
RESULTS_DIR_NAME = "results"
VALIDATION_ONLY_RUNNER = "validation_only"
NAMED_RUNNERS = {"microstructure_perp": run_microstructure_perp_from_spec}
ALLOWED_TYPES = frozenset(
    {"factor", "combo", "carry", "event", "momentum", "risk", "price_action", "other"}
)
FORBIDDEN_TEXT_MARKERS = (
    "http://",
    "https://",
    "ws://",
    "wss://",
    "trading api",
    "wallet",
    "placeorder",
    "submit_order",
    "account_number",
    "api_secret",
    "password",
)


class HypothesisCliError(ValueError):
    """Raised when a private HypothesisSpec file fails #59 gates."""


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one private HypothesisSpec JSON file, run a registered local runner "
            "when requested, and append the private task registry."
        )
    )
    parser.add_argument("spec_json", type=Path)
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=None,
        help=(
            "Private output dir, default to the spec's "
            "${AEGIS_STRATEGIES_ROOT}/incubating/<task>."
        ),
    )
    try:
        return _run(parser.parse_args(argv))
    except (HypothesisCliError, PrivatePathError, ValueError) as exc:
        print(f"run_hypothesis.py: error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    private_root = private_root_from_env(repo_root=REPO_ROOT)
    spec_path = _private_task_file(args.spec_json, private_root=private_root)
    task_name = _task_name_from_private_path(spec_path, private_root=private_root)
    task_dir = _task_dir(args.private_dir, private_root=private_root, task_name=task_name)
    raw = _load_spec(spec_path)
    _validate_json_contract(raw)
    spec = _build_spec(raw)

    completed_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    result_path = _result_path(task_dir=task_dir, spec_id=spec.key, completed_at=completed_at)
    registry_path = task_dir / REGISTRY_NAME

    run = run_backtest(spec)
    _write_result(result_path, spec_path=spec_path, completed_at=completed_at, run=run)
    global_trial_n = _append_registry(
        registry_path,
        spec_path=spec_path,
        result_path=result_path,
        completed_at=completed_at,
        verdict=run.verdict,
        trial_n=spec.trial_count_n,
        spec_id=spec.key,
    )

    summary = {
        "spec_id": spec.key,
        "verdict": run.verdict.verdict,
        "state": run.verdict.state,
        "data_adequacy": run.verdict.data_adequacy,
        "unlock_condition": run.verdict.unlock_condition,
        "trial_n": spec.trial_count_n,
        "global_trial_n": global_trial_n,
        "result_path": str(result_path),
        "registry_path": str(registry_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _task_dir(value: Path | None, *, private_root: Path, task_name: str) -> Path:
    path = private_root / "incubating" / task_name if value is None else value
    resolved = resolve_private_dir(path, private_root=private_root, repo_root=REPO_ROOT)
    if resolved.relative_to(private_root).parts[:2] != ("incubating", task_name):
        raise HypothesisCliError(
            "run_hypothesis.py may only write under "
            "the same ${AEGIS_STRATEGIES_ROOT}/incubating/<task>/ as the spec"
        )
    return resolved


def _private_task_file(path: Path, *, private_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolve_private_dir(resolved.parent, private_root=private_root, repo_root=REPO_ROOT)
    if not resolved.exists() or not resolved.is_file():
        raise HypothesisCliError(f"spec JSON does not exist: {resolved}")
    return resolved


def _task_name_from_private_path(path: Path, *, private_root: Path) -> str:
    parts = path.relative_to(private_root).parts
    if len(parts) < 3 or parts[0] != "incubating":
        raise HypothesisCliError(
            "spec JSON must live under ${AEGIS_STRATEGIES_ROOT}/incubating/<task>/"
        )
    return parts[1]


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisCliError(f"invalid JSON spec: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HypothesisCliError("spec JSON must be an object")
    return loaded


def _validate_json_contract(raw: Mapping[str, Any]) -> None:
    supported = {
        "id",
        "type",
        "universe",
        "predeclared_signals",
        "params",
        "cost_model",
        "benchmark",
        "data_source",
        "trial_n",
        "survivor_light",
        "trust",
        "discipline",
        "runner",
        "notes",
    }
    for key in supported - {"notes", "runner"}:
        if key not in raw:
            raise HypothesisCliError(f"{key} is required")
    if set(raw) - supported:
        raise HypothesisCliError("spec contains unsupported fields")
    if _required_str(raw, "type") not in ALLOWED_TYPES:
        raise HypothesisCliError("type must be one of the predeclared HypothesisSpec types")
    if not isinstance(raw["survivor_light"], bool):
        raise HypothesisCliError("survivor_light must be boolean")
    _validate_runner(raw.get("runner", VALIDATION_ONLY_RUNNER))
    _validate_trust(_mapping(raw["trust"], "trust"))
    _validate_discipline(_mapping(raw["discipline"], "discipline"), bool(raw["survivor_light"]))
    _reject_forbidden_text(raw, context="spec")


def _validate_trust(trust: Mapping[str, Any]) -> None:
    for key in ("predeclared", "review_gate", "no_live", "read_only"):
        if trust.get(key) is not True:
            raise HypothesisCliError(f"trust.{key} must be true")
    if trust.get("registry_scope") != "private":
        raise HypothesisCliError("trust.registry_scope must be private")
    if trust.get("export_contains_private_spec_data") is not False:
        raise HypothesisCliError("trust.export_contains_private_spec_data must be false")
    if trust.get("live_or_network_required") is not False:
        raise HypothesisCliError("trust.live_or_network_required must be false")


def _validate_discipline(discipline: Mapping[str, Any], survivor_light: bool) -> None:
    for key in (
        "t_plus_1_execution",
        "locked_oos",
        "walk_forward",
        "full_costs",
        "multiple_testing",
    ):
        if discipline.get(key) is not True:
            raise HypothesisCliError(f"discipline.{key} must be true")
    if not isinstance(discipline.get("survivor_ceiling"), bool):
        raise HypothesisCliError("discipline.survivor_ceiling must be boolean")
    if survivor_light and discipline.get("survivor_ceiling") is not True:
        raise HypothesisCliError("survivor_light specs require discipline.survivor_ceiling=true")


def _build_spec(raw: Mapping[str, Any]) -> HypothesisSpec:
    discipline_raw = _mapping(raw["discipline"], "discipline")
    discipline = BacktestDiscipline(
        t_plus_1_execution=True,
        locked_oos=True,
        walk_forward=True,
        full_costs=True,
        multiple_testing=True,
        survivor_ceiling=bool(discipline_raw["survivor_ceiling"]),
    )
    return HypothesisSpec(
        key=_required_str(raw, "id"),
        hypothesis_type=_required_str(raw, "type"),  # type: ignore[arg-type]
        universe=_str_tuple(raw["universe"], "universe"),
        predeclared_signals=_str_tuple(raw["predeclared_signals"], "predeclared_signals"),
        params=_mapping(raw["params"], "params"),
        cost_model=_mapping(raw["cost_model"], "cost_model"),
        benchmark=_required_str(raw, "benchmark"),
        data_source=_required_str(raw, "data_source"),
        trial_count_n=_positive_int(raw["trial_n"], "trial_n"),
        discipline=discipline,
        survivor_light=bool(raw["survivor_light"]),
        runner=_runner_for(raw),
    )


def _validate_runner(raw: object) -> None:
    name = _runner_name(raw)
    if name != VALIDATION_ONLY_RUNNER and name not in NAMED_RUNNERS:
        allowed = ", ".join(sorted((*NAMED_RUNNERS, VALIDATION_ONLY_RUNNER)))
        raise HypothesisCliError(f"runner must be one of: {allowed}")


def _runner_for(raw: Mapping[str, Any]) -> Any:
    name = _runner_name(raw.get("runner", VALIDATION_ONLY_RUNNER))
    if name == VALIDATION_ONLY_RUNNER:
        return lambda: _validation_only_payload(raw)

    def _run_named() -> Mapping[str, Any]:
        return NAMED_RUNNERS[name](_build_spec_without_runner(raw))

    return _run_named


def _build_spec_without_runner(raw: Mapping[str, Any]) -> HypothesisSpec:
    discipline_raw = _mapping(raw["discipline"], "discipline")
    return HypothesisSpec(
        key=_required_str(raw, "id"),
        hypothesis_type=_required_str(raw, "type"),  # type: ignore[arg-type]
        universe=_str_tuple(raw["universe"], "universe"),
        predeclared_signals=_str_tuple(raw["predeclared_signals"], "predeclared_signals"),
        params=_mapping(raw["params"], "params"),
        cost_model=_mapping(raw["cost_model"], "cost_model"),
        benchmark=_required_str(raw, "benchmark"),
        data_source=_required_str(raw, "data_source"),
        trial_count_n=_positive_int(raw["trial_n"], "trial_n"),
        discipline=BacktestDiscipline(
            t_plus_1_execution=True,
            locked_oos=True,
            walk_forward=True,
            full_costs=True,
            multiple_testing=True,
            survivor_ceiling=bool(discipline_raw["survivor_ceiling"]),
        ),
        survivor_light=bool(raw["survivor_light"]),
    )


def _runner_name(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip() or VALIDATION_ONLY_RUNNER
    if isinstance(raw, Mapping):
        value = raw.get("name", VALIDATION_ONLY_RUNNER)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise HypothesisCliError("runner must be a string or an object with a name")


def _validation_only_payload(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": "INSUFFICIENT",
        "verdict": "INSUFFICIENT",
        "reason": (
            "HypothesisSpec passed file-seam discipline validation; no strategy adapter is "
            "connected in #59."
        ),
        "metrics": {},
        "benchmarks": {"declared": raw["benchmark"]},
        "multiple_testing": {
            "candidate_count_n": raw["trial_n"],
            "hypothesis_trial_count_n": raw["trial_n"],
            "scope": "single predeclared spec plus private registry cumulative view",
        },
        "safety": {
            "local_file_only": True,
            "network": False,
            "live": False,
            "read_only": True,
            "registry_scope": "private",
        },
    }


def _result_path(*, task_dir: Path, spec_id: str, completed_at: str) -> Path:
    safe_id = _safe_filename(spec_id)
    safe_time = completed_at.replace(":", "").replace("-", "")
    path = task_dir / RESULTS_DIR_NAME / f"{safe_id}-{safe_time}.json"
    resolve_private_dir(path.parent, repo_root=REPO_ROOT)
    return path


def _write_result(path: Path, *, spec_path: Path, completed_at: str, run: Any) -> None:
    cost_model = run.spec.cost_model if isinstance(run.spec.cost_model, Mapping) else {}
    document = {
        "completed_at": completed_at,
        "spec_path": str(spec_path),
        "spec": {
            "id": run.spec.key,
            "type": run.spec.hypothesis_type,
            "universe": list(run.spec.universe),
            "predeclared_signals": list(run.spec.predeclared_signals),
            "params": dict(run.spec.params),
            "cost_model": dict(cost_model),
            "benchmark": run.spec.benchmark,
            "data_source": run.spec.data_source,
            "trial_n": run.spec.trial_count_n,
            "survivor_light": run.spec.survivor_light,
            "discipline": asdict(run.spec.discipline),
        },
        "verdict": _verdict_to_dict(run.verdict),
        "payload": run.payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_registry(
    path: Path,
    *,
    spec_path: Path,
    result_path: Path,
    completed_at: str,
    verdict: StandardVerdict,
    trial_n: int,
    spec_id: str,
) -> int:
    resolve_private_dir(path.parent, repo_root=REPO_ROOT)
    row = {
        "time": completed_at,
        "spec_id": spec_id,
        "trial_n": trial_n,
        "verdict": verdict.verdict,
        "state": verdict.state,
        "data_adequacy": verdict.data_adequacy,
        "unlock_condition": verdict.unlock_condition,
        "spec_path": str(spec_path),
        "result_path": str(result_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")
    return _global_trial_n(path)


def _global_trial_n(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise HypothesisCliError(
                    f"registry contains invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise HypothesisCliError(f"registry line {line_number} is not an object")
            total += _positive_int(row.get("trial_n"), f"registry line {line_number} trial_n")
    return total


def _verdict_to_dict(verdict: StandardVerdict) -> dict[str, Any]:
    return {
        "state": verdict.state,
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "data_adequacy": verdict.data_adequacy,
        "unlock_condition": verdict.unlock_condition,
        "metrics": dict(verdict.metrics),
        "benchmarks": dict(verdict.benchmarks),
        "candidate_count_n": verdict.candidate_count_n,
        "raw_survivors": verdict.raw_survivors,
        "fdr_survivors": verdict.fdr_survivors,
        "multiple_testing": dict(verdict.multiple_testing),
        "safety": dict(verdict.safety),
        "survivor_ceiling_applied": verdict.survivor_ceiling_applied,
    }


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HypothesisCliError(f"{key} is required")
    return value.strip()


def _str_tuple(raw: Any, key: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise HypothesisCliError(f"{key} must be a list of strings")
    values = tuple(str(value).strip() for value in raw if str(value).strip())
    if not values:
        raise HypothesisCliError(f"{key} must not be empty")
    return values


def _mapping(raw: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise HypothesisCliError(f"{key} must be an object")
    return {str(item_key): item_value for item_key, item_value in raw.items()}


def _positive_int(raw: Any, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise HypothesisCliError(f"{key} must be a positive integer")
    return raw


def _safe_filename(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    cleaned = "-".join(part for part in "".join(chars).split("-") if part)
    return cleaned or "hypothesis"


def _reject_forbidden_text(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_forbidden_text(key, context=context)
            _reject_forbidden_text(item, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden_text(item, context=context)
        return
    if isinstance(value, str):
        lower = value.lower()
        for marker in FORBIDDEN_TEXT_MARKERS:
            if marker in lower:
                raise HypothesisCliError(
                    f"{context} contains forbidden local-only marker: {marker}"
                )


def main() -> NoReturn:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
