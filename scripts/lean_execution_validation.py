#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis.lean_execution_validation import LeanExecutionGate, validate_lean_execution_report
from aegis.private_paths import PrivatePathError, private_root_from_env, resolve_private_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = "lean_results"


class LeanCliError(ValueError):
    """Raised when the LEAN execution validation seam is misused."""


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one private LEAN backtest report and convert it into an Aegis "
            "execution-stage validation verdict. This is offline/read-only and never "
            "runs live trading."
        )
    )
    parser.add_argument("spec_json", type=Path)
    parser.add_argument("lean_report_json", type=Path)
    parser.add_argument("--private-dir", type=Path, default=None)
    parser.add_argument("--max-drawdown-limit", type=float, default=-0.30)
    try:
        return _run(parser.parse_args(argv))
    except (LeanCliError, PrivatePathError, ValueError) as exc:
        print(f"lean_execution_validation.py: error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    private_root = private_root_from_env(repo_root=REPO_ROOT)
    spec_path = _private_file(args.spec_json, private_root=private_root)
    report_path = _private_file(args.lean_report_json, private_root=private_root)
    task_name = _task_name(spec_path, private_root=private_root)
    if _task_name(report_path, private_root=private_root) != task_name:
        raise LeanCliError("spec and LEAN report must live under the same incubating task")
    output_dir = _output_dir(args.private_dir, private_root=private_root, task_name=task_name)
    spec = _load_json_object(spec_path, label="spec")
    report = _load_json_object(report_path, label="report")
    payload = validate_lean_execution_report(
        spec=spec,
        report=report,
        gate=LeanExecutionGate(max_drawdown_limit=float(args.max_drawdown_limit)),
    )
    completed_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    result_name = f"{_slug(str(spec.get('id')))}-{completed_at}.json"
    result_path = output_dir / DEFAULT_RESULTS_DIR / result_name
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "completed_at": completed_at,
                "spec_path": str(spec_path),
                "lean_report_path": str(report_path),
                "verdict": payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "state": payload["state"],
        "verdict": payload["verdict"],
        "data_adequacy": payload["data_adequacy"],
        "result_path": str(result_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _private_file(path: Path, *, private_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolve_private_dir(resolved.parent, private_root=private_root, repo_root=REPO_ROOT)
    if not resolved.exists() or not resolved.is_file():
        raise LeanCliError(f"file does not exist: {resolved}")
    return resolved


def _output_dir(value: Path | None, *, private_root: Path, task_name: str) -> Path:
    path = private_root / "incubating" / task_name if value is None else value
    resolved = resolve_private_dir(path, private_root=private_root, repo_root=REPO_ROOT)
    if resolved.relative_to(private_root).parts[:2] != ("incubating", task_name):
        raise LeanCliError("output dir must stay under the same incubating task")
    return resolved


def _task_name(path: Path, *, private_root: Path) -> str:
    parts = path.relative_to(private_root).parts
    if len(parts) < 3 or parts[0] != "incubating":
        raise LeanCliError("file must live under ${AEGIS_STRATEGIES_ROOT}/incubating/<task>/")
    return parts[1]


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LeanCliError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise LeanCliError(f"{label} JSON must be an object")
    return loaded


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value.strip()]
    cleaned = "-".join(part for part in "".join(chars).split("-") if part)
    return cleaned or "lean-execution"


if __name__ == "__main__":
    raise SystemExit(run_cli())
