from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aegis.edgar_pit import (
    PitFundamentalStore,
    SecEdgarClient,
    build_coverage_matrix,
    extract_submission_metadata,
)

META_PATH = Path("incubating") / "edgar_pit_fundamentals.meta.json"
DEFAULT_MATRIX_OUT = (
    Path.home()
    / "apps"
    / "aegis-strategies"
    / "incubating"
    / "olympus37"
    / "matrix.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch EDGAR pilot submissions/companyfacts cache."
    )
    parser.add_argument("--meta", type=Path, default=META_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--coverage-out", type=Path, default=None)
    parser.add_argument(
        "--matrix-out",
        type=Path,
        default=_default_matrix_out(),
        help="Write ticker x concept x fiscal_year coverage matrix.",
    )
    args = parser.parse_args()

    companies = _load_pilot_universe(args.meta)
    if args.limit is not None:
        companies = companies[: args.limit]

    client = SecEdgarClient()
    coverage: list[dict[str, Any]] = []
    combined_store = PitFundamentalStore()
    for company in companies:
        ticker = str(company["ticker"])
        cik = str(company["cik"])
        submissions = client.fetch_submissions(cik, force=args.force)
        submission_metadata = extract_submission_metadata(
            ticker=ticker,
            cik=cik,
            payload=submissions,
            pilot_status=_optional_str(company.get("status")),
        )
        companyfacts = client.fetch_companyfacts(cik, force=args.force)
        store = PitFundamentalStore.from_companyfacts(
            ticker=ticker,
            cik=cik,
            payload=companyfacts,
            company_metadata=submission_metadata,
        )
        combined_store.facts.extend(store.facts)
        combined_store.restatements.extend(store.restatements)
        combined_store.company_metadata.update(store.company_metadata)
        coverage.append(
            {
                "ticker": ticker,
                "cik": cik,
                "status": company.get("status"),
                "entity": submission_metadata.entity_name,
                "formerNames": list(submission_metadata.former_names),
                "tickers": list(submission_metadata.tickers),
                "earliest_filing_date": (
                    submission_metadata.earliest_filing_date.isoformat()
                    if submission_metadata.earliest_filing_date is not None
                    else None
                ),
                "earliest_recent_block_filing_date": (
                    submission_metadata.earliest_recent_block_filing_date.isoformat()
                    if submission_metadata.earliest_recent_block_filing_date is not None
                    else None
                ),
                "earliest_filing_date_source": submission_metadata.earliest_filing_date_source,
                "coverage": store.coverage(),
            }
        )

    if args.matrix_out is not None:
        args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
        args.matrix_out.write_text(
            json.dumps(build_coverage_matrix(combined_store), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.coverage_out is not None:
        args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_out.write_text(
            json.dumps({"companies": coverage}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        print(json.dumps({"companies": coverage}, indent=2, sort_keys=True))
    return 0


def _default_matrix_out() -> Path | None:
    import os

    configured = os.environ.get("AEGIS_EDGAR_MATRIX_OUT")
    if configured is not None:
        if configured == "":
            return None
        return Path(configured)
    return DEFAULT_MATRIX_OUT


def _load_pilot_universe(meta_path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("metadata root must be a JSON object")
    universe = loaded.get("pilot_universe")
    if not isinstance(universe, list):
        raise ValueError("metadata must contain pilot_universe list")
    companies: list[dict[str, Any]] = []
    for company in universe:
        if not isinstance(company, dict):
            raise ValueError("pilot_universe entries must be objects")
        if "ticker" not in company or "cik" not in company:
            raise ValueError("pilot_universe entries must include ticker and cik")
        companies.append(company)
    return companies


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


if __name__ == "__main__":
    raise SystemExit(main())
