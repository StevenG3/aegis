from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def load_script() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "polymarket_relative_value_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "polymarket_relative_value_evidence_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["polymarket_relative_value_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


def test_run_from_cli_requires_private_olympus68(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))

    args = module._parse_args(["--forward-dir", str(forward_dir)])
    base_dir = module.private_dir_from_cli(args.output_dir, default_task=module.DEFAULT_TASK)

    assert base_dir == private_root / "incubating" / "olympus68"


def test_load_forward_rows_fails_empty_source_loudly(tmp_path: Path) -> None:
    module = load_script()

    assert module.load_forward_rows(tmp_path) == []
