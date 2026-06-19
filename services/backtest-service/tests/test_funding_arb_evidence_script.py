from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_evidence_script():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "funding_arb_evidence.py"
    spec = importlib.util.spec_from_file_location("funding_arb_evidence_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["funding_arb_evidence_script"] = module
    spec.loader.exec_module(module)
    return module


def test_run_from_env_accepts_explicit_symbols_and_private_dir(monkeypatch, tmp_path: Path) -> None:
    module = load_evidence_script()
    private_root = tmp_path / "aegis-strategies"
    private_root.mkdir()
    output_dir = private_root / "incubating" / "olympus50"
    monkeypatch.setenv("AEGIS_STRATEGIES_ROOT", str(private_root))
    monkeypatch.setenv("FUNDING_ARB_EVIDENCE_SYMBOLS", "btcusdt, ethusdt")
    monkeypatch.setenv("FUNDING_ARB_EVIDENCE_START", "2024-01-01")
    monkeypatch.setenv("FUNDING_ARB_EVIDENCE_END", "2024-06-01")
    monkeypatch.setenv("FUNDING_ARB_EVIDENCE_OUTPUT_DIR", str(output_dir))

    run = module._run_from_env()

    assert run.symbols == ("BTCUSDT", "ETHUSDT")
    assert run.start == "2024-01-01"
    assert run.end == "2024-06-01"
    assert run.output_dir == output_dir


def test_markdown_reports_tristate_verdict(tmp_path: Path) -> None:
    module = load_evidence_script()
    markdown = module._markdown(
        {
            "generated_at": "2026-06-19T00:00:00+00:00",
            "fetch_failures": [],
            "report": {
                "verdict": "NO_ROBUST_EDGE",
                "reason": "no predeclared grid survived",
                "search_space_n": 8,
                "tested_candidates": 8,
                "fdr": {"discoveries": 0},
                "best_candidate": None,
            },
        },
        tmp_path / "result.json",
    )

    assert "Verdict: `NO_ROBUST_EDGE`" in markdown
    assert "Benchmark is risk-free cash, not buy-and-hold" in markdown
