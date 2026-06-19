from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

from aegis.private_paths import private_dir_from_cli


def _load_orchestrator_app() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    service_dir = root / "services" / "orchestrator"
    sys.path.insert(0, str(service_dir))
    spec = importlib.util.spec_from_file_location("orchestrator_app", service_dir / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load orchestrator app")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an EV shadow replay report.")
    parser.add_argument("--actor", default=None)
    parser.add_argument("--min-ev", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if not args.no_write:
        os.environ["EV_SHADOW_REPORT_DIR"] = str(
            private_dir_from_cli(args.output_dir, default_task="ev-shadow")
        )
    app = _load_orchestrator_app()
    min_ev = Decimal(args.min_ev) if args.min_ev is not None else None
    report = app.build_ev_shadow_report(actor=args.actor, min_ev=min_ev)
    if not args.no_write:
        report["written_files"] = app.write_ev_shadow_report(report)
    if args.text:
        print(report["human_readable"])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
