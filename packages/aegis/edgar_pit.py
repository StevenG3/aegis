from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

SEC_BASE_URL = "https://data.sec.gov"
DEFAULT_CACHE_ENV = "AEGIS_EDGAR_CACHE_DIR"
DEFAULT_USER_AGENT_ENV = "AEGIS_SEC_USER_AGENT"
DEFAULT_USER_AGENT = "AegisOlympusResearch/0.1 (CONTACT_NOT_SET)"
MAX_SEC_REQUESTS_PER_SECOND = 10.0

TRACKED_CONCEPTS = {
    "AccountsPayableCurrent",
    "AccountsReceivableNetCurrent",
    "Assets",
    "AssetsCurrent",
    "CommonStockSharesOutstanding",
    "CostOfRevenue",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CommercialPaper",
    "CommercialPaperBorrowings",
    "DebtCurrent",
    "DepreciationAndAmortization",
    "DepreciationDepletionAndAmortization",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "EntityCommonStockSharesOutstanding",
    "GrossProfit",
    "Liabilities",
    "LiabilitiesCurrent",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "Revenues",
    "SalesRevenueNet",
    "ShortTermBorrowings",
    "StockholdersEquity",
}

DERIVED_CONCEPTS = {"Ebitda", "FreeCashFlow"}


class DelistingAwarePriceSource(Protocol):
    def get_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> dict[str, Any]:
        ...

    def survivorship_status(self, ticker: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PilotCompany:
    ticker: str
    cik: str
    name: str
    status: str
    sector: str
    price_coverage_expected: str

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)


@dataclass(frozen=True)
class EdgarFact:
    ticker: str
    cik: str
    concept: str
    unit: str
    value: float
    filed: date
    available_on: date
    period_end: date | None
    accession: str | None
    form: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    frame: str | None
    is_restatement: bool = False


@dataclass(frozen=True)
class EdgarCompanyMetadata:
    ticker: str
    cik: str
    entity_name: str | None
    tickers: tuple[str, ...]
    former_names: tuple[str, ...]
    earliest_filing_date: date | None
    earliest_recent_block_filing_date: date | None
    earliest_filing_date_source: str
    pilot_status: str | None


@dataclass(frozen=True)
class EdgarFetchRecord:
    url: str
    cache_path: str
    cache_hit: bool
    status_code: int | None
    payload_bytes: int
    user_agent: str
    requests_per_second: float
    elapsed_seconds: float


@dataclass(frozen=True)
class SubmissionFilingDates:
    earliest_filing_date: date | None
    earliest_recent_block_filing_date: date | None
    earliest_filing_date_source: str


@dataclass
class PitFundamentalStore:
    facts: list[EdgarFact] = field(default_factory=list)
    restatements: list[EdgarFact] = field(default_factory=list)
    company_metadata: dict[str, EdgarCompanyMetadata] = field(default_factory=dict)

    @classmethod
    def from_companyfacts(
        cls,
        *,
        ticker: str,
        cik: str,
        payload: dict[str, Any],
        concepts: set[str] | None = None,
        availability_lag_business_days: int = 1,
        company_metadata: EdgarCompanyMetadata | None = None,
    ) -> PitFundamentalStore:
        selected = concepts or TRACKED_CONCEPTS
        primary_by_period: dict[tuple[str, str, date | None], EdgarFact] = {}
        restatements: list[EdgarFact] = []

        us_gaap = payload.get("facts", {}).get("us-gaap", {})
        if not isinstance(us_gaap, dict):
            return cls(company_metadata=_metadata_map(company_metadata))

        for concept, concept_payload in us_gaap.items():
            if concept not in selected or not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            for unit, unit_facts in units.items():
                if not isinstance(unit_facts, list):
                    continue
                parsed: list[EdgarFact] = []
                for item in unit_facts:
                    if not isinstance(item, dict):
                        continue
                    fact = _parse_companyfact_item(
                        ticker=ticker,
                        cik=cik,
                        concept=concept,
                        unit=unit,
                        item=cast(dict[str, Any], item),
                        availability_lag_business_days=availability_lag_business_days,
                    )
                    if fact is not None:
                        parsed.append(fact)
                parsed.sort(key=lambda fact: (fact.filed, fact.accession or ""))
                for fact in parsed:
                    period_key = (fact.concept, fact.unit, fact.period_end)
                    previous = primary_by_period.get(period_key)
                    if previous is None:
                        primary_by_period[period_key] = fact
                    else:
                        restatements.append(_mark_restatement(fact))

        facts = sorted(primary_by_period.values(), key=lambda fact: (fact.ticker, fact.filed))
        restatements.sort(key=lambda fact: (fact.ticker, fact.filed))
        return cls(
            facts=facts,
            restatements=restatements,
            company_metadata=_metadata_map(company_metadata),
        )

    def as_of(self, ticker: str, as_of_date: date | str) -> dict[str, EdgarFact]:
        query_date = _parse_date(as_of_date)
        ticker_upper = ticker.upper()
        latest: dict[str, EdgarFact] = {}
        for fact in self.facts:
            if fact.ticker.upper() != ticker_upper or fact.available_on > query_date:
                continue
            current = latest.get(fact.concept)
            if current is None or _fact_recency_key(fact) > _fact_recency_key(current):
                latest[fact.concept] = fact
        return latest

    def coverage(self, as_of_date: date | str | None = None) -> dict[str, Any]:
        eligible_facts = self.facts
        if as_of_date is not None:
            query_date = _parse_date(as_of_date)
            eligible_facts = [fact for fact in self.facts if fact.available_on <= query_date]
        tickers = {fact.ticker for fact in eligible_facts}
        concepts = {fact.concept for fact in eligible_facts}
        return {
            "tickers": len(tickers),
            "concepts": len(concepts),
            "facts": len(eligible_facts),
            "restatements_separated": len(self.restatements),
            "restatement_distribution": _restatement_distribution(self.restatements),
            "companies": _company_coverage_rows(
                eligible_facts=eligible_facts,
                metadata_by_ticker=self.company_metadata,
            ),
        }


def derive_ebitda(as_of_result: dict[str, EdgarFact]) -> EdgarFact | None:
    operating_income = as_of_result.get("OperatingIncomeLoss")
    depreciation = as_of_result.get("DepreciationDepletionAndAmortization")
    if depreciation is None:
        depreciation = as_of_result.get("DepreciationAndAmortization")
    return _derive_sum(
        concept="Ebitda",
        left=operating_income,
        right=depreciation,
    )


def derive_fcf(as_of_result: dict[str, EdgarFact]) -> EdgarFact | None:
    operating_cash_flow = as_of_result.get("NetCashProvidedByUsedInOperatingActivities")
    capex = as_of_result.get("PaymentsToAcquirePropertyPlantAndEquipment")
    return _derive_sum(
        concept="FreeCashFlow",
        left=operating_cash_flow,
        right=capex,
        right_multiplier=-1.0,
    )


def derive_net_debt(as_of_result: dict[str, EdgarFact]) -> EdgarFact | None:
    cash = as_of_result.get("CashAndCashEquivalentsAtCarryingValue")
    if cash is None:
        cash = as_of_result.get("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
    if cash is None:
        return None

    debt_atoms = [
        as_of_result.get(concept)
        for concept in (
            "LongTermDebt",
            "LongTermDebtCurrent",
            "DebtCurrent",
            "ShortTermBorrowings",
            "CommercialPaper",
            "CommercialPaperBorrowings",
        )
    ]
    debt = _derive_many(concept="TotalDebt", facts=debt_atoms)
    if debt is None:
        return None
    return _derive_sum(
        concept="NetDebt",
        left=debt,
        right=cash,
        right_multiplier=-1.0,
    )


def extract_submission_metadata(
    *,
    ticker: str,
    cik: str,
    payload: dict[str, Any],
    pilot_status: str | None,
) -> EdgarCompanyMetadata:
    former_names = payload.get("formerNames")
    tickers = payload.get("tickers")
    filing_dates = _extract_submission_filing_dates(payload)
    return EdgarCompanyMetadata(
        ticker=ticker.upper(),
        cik=cik.zfill(10),
        entity_name=_as_optional_str(payload.get("name")),
        tickers=_as_str_tuple(tickers),
        former_names=_extract_former_names(former_names),
        earliest_filing_date=filing_dates.earliest_filing_date,
        earliest_recent_block_filing_date=filing_dates.earliest_recent_block_filing_date,
        earliest_filing_date_source=filing_dates.earliest_filing_date_source,
        pilot_status=pilot_status,
    )


def build_coverage_matrix(
    store: PitFundamentalStore,
    *,
    concepts: set[str] | None = None,
    fiscal_years: set[int] | None = None,
) -> dict[str, Any]:
    selected_concepts = concepts or (TRACKED_CONCEPTS | DERIVED_CONCEPTS)
    years = fiscal_years or {
        fact.fiscal_year for fact in store.facts if fact.fiscal_year is not None
    }
    sorted_years = sorted(year for year in years if year is not None)
    sorted_concepts = sorted(selected_concepts)
    tickers = sorted({fact.ticker for fact in store.facts} | set(store.company_metadata))
    latest_by_year: dict[tuple[str, int, str], EdgarFact] = {}
    for fact in store.facts:
        if fact.fiscal_year is None:
            continue
        key = (fact.ticker, fact.fiscal_year, fact.concept)
        current = latest_by_year.get(key)
        if current is None or _fact_recency_key(fact) > _fact_recency_key(current):
            latest_by_year[key] = fact

    matrix: dict[str, dict[str, dict[str, bool]]] = {}
    for ticker in tickers:
        ticker_rows: dict[str, dict[str, bool]] = {}
        for concept in sorted_concepts:
            concept_years: dict[str, bool] = {}
            for fiscal_year in sorted_years:
                atoms = {
                    atom_concept: fact
                    for (fact_ticker, fact_year, atom_concept), fact in latest_by_year.items()
                    if fact_ticker == ticker and fact_year == fiscal_year
                }
                concept_years[str(fiscal_year)] = _has_concept_for_matrix(concept, atoms)
            ticker_rows[concept] = concept_years
        matrix[ticker] = ticker_rows

    return {
        "tickers": tickers,
        "concepts": sorted_concepts,
        "fiscal_years": sorted_years,
        "matrix": matrix,
    }


class SecEdgarClient:
    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        user_agent: str | None = None,
        requests_per_second: float = MAX_SEC_REQUESTS_PER_SECOND,
        timeout_seconds: float = 20.0,
    ) -> None:
        if requests_per_second <= 0 or requests_per_second > MAX_SEC_REQUESTS_PER_SECOND:
            raise ValueError("SEC requests_per_second must be in (0, 10]")
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(DEFAULT_CACHE_ENV)
            or Path("data") / "edgar-pit"
        )
        self.user_agent = user_agent or os.environ.get(DEFAULT_USER_AGENT_ENV) or DEFAULT_USER_AGENT
        self.requests_per_second = requests_per_second
        self.timeout_seconds = timeout_seconds
        self._last_request_at = 0.0
        self.fetch_records: list[EdgarFetchRecord] = []

    def fetch_submissions(self, cik: str, *, force: bool = False) -> dict[str, Any]:
        padded = cik.zfill(10)
        return self._get_json(
            path=f"/submissions/CIK{padded}.json",
            cache_path=self.cache_dir / "submissions" / f"CIK{padded}.json",
            force=force,
        )

    def fetch_companyfacts(self, cik: str, *, force: bool = False) -> dict[str, Any]:
        padded = cik.zfill(10)
        return self._get_json(
            path=f"/api/xbrl/companyfacts/CIK{padded}.json",
            cache_path=self.cache_dir / "companyfacts" / f"CIK{padded}.json",
            force=force,
        )

    def _get_json(self, *, path: str, cache_path: Path, force: bool) -> dict[str, Any]:
        url = f"{SEC_BASE_URL}{path}"
        if cache_path.exists() and not force:
            started = time.monotonic()
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"cached EDGAR payload is not an object: {cache_path}")
            self.fetch_records.append(
                EdgarFetchRecord(
                    url=url,
                    cache_path=str(cache_path),
                    cache_hit=True,
                    status_code=None,
                    payload_bytes=cache_path.stat().st_size,
                    user_agent=self.user_agent,
                    requests_per_second=self.requests_per_second,
                    elapsed_seconds=time.monotonic() - started,
                )
            )
            return cast(dict[str, Any], loaded)

        self._respect_rate_limit()
        validate_sec_user_agent(self.user_agent)
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        import httpx

        last_error: httpx.HTTPError | None = None
        for attempt in range(3):
            try:
                started = time.monotonic()
                with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    parsed = response.json()
                    if not isinstance(parsed, dict):
                        raise ValueError(f"SEC JSON payload is not an object: {url}")
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(parsed, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    self.fetch_records.append(
                        EdgarFetchRecord(
                            url=url,
                            cache_path=str(cache_path),
                            cache_hit=False,
                            status_code=response.status_code,
                            payload_bytes=len(response.content),
                            user_agent=self.user_agent,
                            requests_per_second=self.requests_per_second,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    )
                    return cast(dict[str, Any], parsed)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2**attempt)
        message = f"failed to fetch SEC EDGAR payload after retries: {url}"
        raise RuntimeError(message) from last_error

    def _respect_rate_limit(self) -> None:
        min_interval = 1.0 / self.requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()


def _parse_companyfact_item(
    *,
    ticker: str,
    cik: str,
    concept: str,
    unit: str,
    item: dict[str, Any],
    availability_lag_business_days: int,
) -> EdgarFact | None:
    filed_raw = item.get("filed")
    value_raw = item.get("val")
    if filed_raw is None or value_raw is None:
        return None
    filed = _parse_date(filed_raw)
    value = float(value_raw)
    return EdgarFact(
        ticker=ticker.upper(),
        cik=cik.zfill(10),
        concept=concept,
        unit=unit,
        value=value,
        filed=filed,
        available_on=add_business_days(filed, availability_lag_business_days),
        period_end=_parse_optional_date(item.get("end")),
        accession=_as_optional_str(item.get("accn")),
        form=_as_optional_str(item.get("form")),
        fiscal_year=_as_optional_int(item.get("fy")),
        fiscal_period=_as_optional_str(item.get("fp")),
        frame=_as_optional_str(item.get("frame")),
    )


def _mark_restatement(fact: EdgarFact) -> EdgarFact:
    return EdgarFact(
        ticker=fact.ticker,
        cik=fact.cik,
        concept=fact.concept,
        unit=fact.unit,
        value=fact.value,
        filed=fact.filed,
        available_on=fact.available_on,
        period_end=fact.period_end,
        accession=fact.accession,
        form=fact.form,
        fiscal_year=fact.fiscal_year,
        fiscal_period=fact.fiscal_period,
        frame=fact.frame,
        is_restatement=True,
    )


def _metadata_map(
    metadata: EdgarCompanyMetadata | None,
) -> dict[str, EdgarCompanyMetadata]:
    if metadata is None:
        return {}
    return {metadata.ticker.upper(): metadata}


def _company_coverage_rows(
    *,
    eligible_facts: list[EdgarFact],
    metadata_by_ticker: dict[str, EdgarCompanyMetadata],
) -> list[dict[str, Any]]:
    facts_by_ticker: dict[str, list[EdgarFact]] = {}
    for fact in eligible_facts:
        facts_by_ticker.setdefault(fact.ticker, []).append(fact)

    tickers = sorted(set(facts_by_ticker) | set(metadata_by_ticker))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        metadata = metadata_by_ticker.get(ticker)
        facts = facts_by_ticker.get(ticker, [])
        earliest_filing_date = (
            metadata.earliest_filing_date
            if metadata is not None
            else min((fact.filed for fact in facts), default=None)
        )
        pilot_status = metadata.pilot_status if metadata is not None else None
        rows.append(
            {
                "ticker": ticker,
                "cik": metadata.cik if metadata is not None else _first_cik(facts),
                "entity": metadata.entity_name if metadata is not None else None,
                "tickers": list(metadata.tickers) if metadata is not None else [ticker],
                "former_names": (
                    list(metadata.former_names) if metadata is not None else []
                ),
                "former_names_count": (
                    len(metadata.former_names) if metadata is not None else 0
                ),
                "earliest_filing_date": (
                    earliest_filing_date.isoformat()
                    if earliest_filing_date is not None
                    else None
                ),
                "earliest_recent_block_filing_date": (
                    metadata.earliest_recent_block_filing_date.isoformat()
                    if metadata is not None
                    and metadata.earliest_recent_block_filing_date is not None
                    else None
                ),
                "earliest_filing_date_source": (
                    metadata.earliest_filing_date_source
                    if metadata is not None
                    else "companyfacts_fallback"
                ),
                "pilot_status": pilot_status,
                "coverage_window_gap": _coverage_window_gap(
                    earliest_filing_date=earliest_filing_date,
                    pilot_status=pilot_status,
                ),
            }
        )
    return rows


def _restatement_distribution(restatements: list[EdgarFact]) -> dict[str, Any]:
    by_concept: dict[str, int] = {}
    by_form: dict[str, int] = {}
    for fact in restatements:
        by_concept[fact.concept] = by_concept.get(fact.concept, 0) + 1
        form = fact.form or "UNKNOWN"
        by_form[form] = by_form.get(form, 0) + 1
    return {
        "by_concept": dict(sorted(by_concept.items())),
        "by_form": dict(sorted(by_form.items())),
    }


def _coverage_window_gap(
    *,
    earliest_filing_date: date | None,
    pilot_status: str | None,
) -> dict[str, Any]:
    pre_event_status = (
        pilot_status is not None
        and any(
            marker in pilot_status
            for marker in ("post_merger", "restructured", "post_spinoff")
        )
    )
    pre_event_history_missing = (
        earliest_filing_date is not None
        and earliest_filing_date > date(2010, 1, 1)
        and pre_event_status
    )
    return {
        "pre_event_history_missing": pre_event_history_missing,
        "reason": (
            "current CIK starts after 2010-01-01 for corporate-action-sensitive pilot status"
            if pre_event_history_missing
            else None
        ),
    }


def _first_cik(facts: list[EdgarFact]) -> str | None:
    if not facts:
        return None
    return facts[0].cik


def _derive_sum(
    *,
    concept: str,
    left: EdgarFact | None,
    right: EdgarFact | None,
    right_multiplier: float = 1.0,
) -> EdgarFact | None:
    if left is None or right is None:
        return None
    if not _facts_are_compatible_for_derivation(left, right):
        return None
    later_filed_fact = left if left.filed >= right.filed else right
    return EdgarFact(
        ticker=left.ticker,
        cik=left.cik,
        concept=concept,
        unit=left.unit,
        value=left.value + (right.value * right_multiplier),
        filed=later_filed_fact.filed,
        available_on=later_filed_fact.available_on,
        period_end=left.period_end,
        accession=None,
        form="DERIVED",
        fiscal_year=left.fiscal_year,
        fiscal_period=left.fiscal_period,
        frame=None,
    )


def _derive_many(*, concept: str, facts: list[EdgarFact | None]) -> EdgarFact | None:
    present = [fact for fact in facts if fact is not None]
    if not present:
        return None
    left: EdgarFact | None = present[0]
    for right in present[1:]:
        left = _derive_sum(concept=concept, left=left, right=right)
        if left is None:
            return None
    if left is None:
        return None
    if left.concept == concept:
        return left
    return EdgarFact(
        ticker=left.ticker,
        cik=left.cik,
        concept=concept,
        unit=left.unit,
        value=left.value,
        filed=left.filed,
        available_on=left.available_on,
        period_end=left.period_end,
        accession=None,
        form="DERIVED",
        fiscal_year=left.fiscal_year,
        fiscal_period=left.fiscal_period,
        frame=None,
    )


def _facts_are_compatible_for_derivation(left: EdgarFact, right: EdgarFact) -> bool:
    return (
        left.ticker == right.ticker
        and left.cik == right.cik
        and left.unit == right.unit
        and left.period_end == right.period_end
        and left.fiscal_year == right.fiscal_year
        and left.fiscal_period == right.fiscal_period
    )


def _has_concept_for_matrix(concept: str, atoms: dict[str, EdgarFact]) -> bool:
    if concept == "Ebitda":
        return derive_ebitda(atoms) is not None
    if concept == "FreeCashFlow":
        return derive_fcf(atoms) is not None
    return concept in atoms


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _extract_former_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
    return tuple(names)


def validate_sec_user_agent(user_agent: str) -> None:
    if "CONTACT_NOT_SET" in user_agent or "example.com" in user_agent:
        raise ValueError(
            "AEGIS_SEC_USER_AGENT must be set to a real project contact before SEC requests"
        )
    if "@" not in user_agent and "http://" not in user_agent and "https://" not in user_agent:
        raise ValueError(
            "AEGIS_SEC_USER_AGENT must include a real email address or contact URL"
        )


def _extract_submission_filing_dates(payload: dict[str, Any]) -> SubmissionFilingDates:
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        return SubmissionFilingDates(
            earliest_filing_date=None,
            earliest_recent_block_filing_date=None,
            earliest_filing_date_source="none",
        )
    recent_earliest = _extract_earliest_recent_block_filing_date(filings)
    files_earliest = _extract_earliest_files_filing_from(filings)
    candidates = [
        item for item in (recent_earliest, files_earliest) if item is not None
    ]
    source = "recent_plus_files" if files_earliest is not None else "recent_only"
    return SubmissionFilingDates(
        earliest_filing_date=min(candidates) if candidates else None,
        earliest_recent_block_filing_date=recent_earliest,
        earliest_filing_date_source=source if candidates else "none",
    )


def _extract_earliest_recent_block_filing_date(filings: dict[str, Any]) -> date | None:
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        return None
    filing_dates = recent.get("filingDate")
    if not isinstance(filing_dates, list):
        return None
    parsed = [
        _parse_date(item)
        for item in filing_dates
        if isinstance(item, str) and _is_iso_date(item)
    ]
    return min(parsed, default=None)


def _extract_earliest_files_filing_from(filings: dict[str, Any]) -> date | None:
    files = filings.get("files")
    if not isinstance(files, list):
        return None
    parsed: list[date] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        filing_from = item.get("filingFrom")
        if isinstance(filing_from, str) and _is_iso_date(filing_from):
            parsed.append(_parse_date(filing_from))
    return min(parsed, default=None)


def _is_iso_date(value: str) -> bool:
    try:
        _parse_date(value)
    except ValueError:
        return False
    return True


def add_business_days(start: date, days: int) -> date:
    if days < 0:
        raise ValueError("days must be non-negative")
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _fact_recency_key(fact: EdgarFact) -> tuple[date, date, str]:
    return (fact.filed, fact.period_end or date.min, fact.accession or "")


def _parse_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return _parse_date(value)


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
