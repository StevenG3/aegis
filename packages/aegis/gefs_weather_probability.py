from __future__ import annotations

import importlib
import math
import re
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, cast
from zoneinfo import ZoneInfo

from aegis.polymarket_weather_relative_value import TemperatureBucket

GEFS_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = tuple(f"gep{member:02d}" for member in range(1, 31))
GEFS_CYCLES_UTC = (0, 6, 12, 18)
GEFS_LEAD_HOURS = tuple(range(6, 73, 6))
USER_AGENT = "aegis-gefs-weather-probability/0.1 read-only"


@dataclass(frozen=True)
class StationSpec:
    code: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True)
class GefsCycle:
    issue_time: datetime
    cycle_hour: int


@dataclass(frozen=True)
class TMaxMessageRef:
    url: str
    index_url: str
    member: str
    lead_hour: int
    byte_start: int
    byte_end: int
    descriptor: str


@dataclass(frozen=True)
class StationForecastSample:
    member: str
    daily_max_f: float
    rounded_daily_max_f: int
    lead_hours_used: tuple[int, ...]


@dataclass(frozen=True)
class BucketProbabilityResult:
    probability: float
    member_count: int
    members_in_bucket: int
    issue_ts: int
    issue_time: str
    cycle_hour: int
    lead_hours: tuple[int, ...]
    interpolation: str
    decoder: str
    self_check: Mapping[str, Any]


@dataclass(frozen=True)
class ArchiveAvailability:
    required_messages: int
    available_messages: int
    missing_messages: int
    missing_examples: tuple[str, ...]


@dataclass
class SampledDecoderValueSelfCheck:
    max_checks: int = 24
    _checks_run: int = field(default=0, init=False)
    _lock: Any = field(default_factory=Lock, init=False, repr=False)

    @property
    def checks_run(self) -> int:
        return self._checks_run

    def __call__(self, message: bytes, station: StationSpec) -> float:
        with self._lock:
            should_check = self._checks_run < self.max_checks
            if should_check:
                self._checks_run += 1
        if should_check:
            return _decode_station_tmax_k_checked(message, station)
        return _decode_station_tmax_k(message, station)


STATIONS: Mapping[str, StationSpec] = {
    "KLGA": StationSpec("KLGA", latitude=40.7769, longitude=-73.8740, timezone="America/New_York"),
    "KMIA": StationSpec("KMIA", latitude=25.7959, longitude=-80.2870, timezone="America/New_York"),
    "KLAX": StationSpec(
        "KLAX",
        latitude=33.9416,
        longitude=-118.4085,
        timezone="America/Los_Angeles",
    ),
}


ByteFetcher = Callable[[str, int, int], bytes]
TextFetcher = Callable[[str], str]
Decoder = Callable[[bytes, StationSpec], float]
DECODER_VALUE_SELF_CHECK_MAX_ERROR_K = 0.05


def latest_gefs_cycle_before(decision_ts: int) -> GefsCycle | None:
    decision = datetime.fromtimestamp(decision_ts, UTC)
    for delta_days in range(0, 4):
        day = (decision - timedelta(days=delta_days)).date()
        for hour in reversed(GEFS_CYCLES_UTC):
            issue = datetime.combine(day, time(hour=hour), tzinfo=UTC)
            if issue < decision:
                return GefsCycle(issue_time=issue, cycle_hour=hour)
    return None


def lead_hours_for_station_day(
    *,
    issue_time: datetime,
    station: StationSpec,
    target_date: date,
    lead_hours: Sequence[int] = GEFS_LEAD_HOURS,
) -> tuple[int, ...]:
    zone = ZoneInfo(station.timezone)
    local_start = datetime.combine(target_date, time.min, tzinfo=zone).astimezone(UTC)
    local_end = datetime.combine(
        target_date + timedelta(days=1),
        time.min,
        tzinfo=zone,
    ).astimezone(UTC)
    selected: list[int] = []
    for lead in lead_hours:
        interval_start = issue_time + timedelta(hours=lead - 6)
        interval_end = issue_time + timedelta(hours=lead)
        if interval_start < local_end and interval_end > local_start:
            selected.append(lead)
    return tuple(selected)


def bucket_probability_from_gefs(
    *,
    station: StationSpec,
    target_date: date,
    decision_ts: int,
    bucket: TemperatureBucket,
    members: Sequence[str] = GEFS_MEMBERS,
    text_fetcher: TextFetcher | None = None,
    byte_fetcher: ByteFetcher | None = None,
    decoder: Decoder | None = None,
    base_url: str = GEFS_BASE_URL,
    max_workers: int = 1,
) -> BucketProbabilityResult:
    samples, cycle, leads = station_daily_samples_from_gefs(
        station=station,
        target_date=target_date,
        decision_ts=decision_ts,
        members=members,
        text_fetcher=text_fetcher,
        byte_fetcher=byte_fetcher,
        decoder=decoder,
        base_url=base_url,
        max_workers=max_workers,
    )
    return bucket_probability_from_samples(
        samples=samples,
        bucket=bucket,
        cycle=cycle,
        leads=leads,
        station=station,
    )


def station_daily_samples_from_gefs(
    *,
    station: StationSpec,
    target_date: date,
    decision_ts: int,
    members: Sequence[str] = GEFS_MEMBERS,
    text_fetcher: TextFetcher | None = None,
    byte_fetcher: ByteFetcher | None = None,
    decoder: Decoder | None = None,
    base_url: str = GEFS_BASE_URL,
    max_workers: int = 1,
) -> tuple[list[StationForecastSample], GefsCycle, tuple[int, ...]]:
    text_fetcher = text_fetcher or _fetch_text
    byte_fetcher = byte_fetcher or _fetch_range
    decoder = decoder or _decode_station_tmax_k_checked
    cycle = latest_gefs_cycle_before(decision_ts)
    if cycle is None:
        raise ValueError("no GEFS cycle strictly before decision timestamp")
    leads = lead_hours_for_station_day(
        issue_time=cycle.issue_time,
        station=station,
        target_date=target_date,
    )
    if not leads:
        raise ValueError("no GEFS lead-hour windows overlap the station local date")

    def sample_member(member: str) -> StationForecastSample:
        values_f: list[float] = []
        used_leads: list[int] = []
        for lead in leads:
            ref = tmax_message_ref(
                issue_time=cycle.issue_time,
                member=member,
                lead_hour=lead,
                text_fetcher=text_fetcher,
                base_url=base_url,
            )
            value_k = decoder(byte_fetcher(ref.url, ref.byte_start, ref.byte_end), station)
            values_f.append(kelvin_to_fahrenheit(value_k))
            used_leads.append(lead)
        daily_max_f = max(values_f)
        return StationForecastSample(
            member=member,
            daily_max_f=daily_max_f,
            rounded_daily_max_f=round_temperature_f(daily_max_f),
            lead_hours_used=tuple(used_leads),
        )

    if max_workers <= 1:
        samples = [sample_member(member) for member in members]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            samples = list(executor.map(sample_member, members))
    return samples, cycle, leads


def bucket_probability_from_samples(
    *,
    samples: Sequence[StationForecastSample],
    bucket: TemperatureBucket,
    cycle: GefsCycle,
    leads: Sequence[int],
    station: StationSpec,
) -> BucketProbabilityResult:
    members_in_bucket = sum(1 for sample in samples if bucket.contains(sample.rounded_daily_max_f))
    self_check = decoder_self_check(samples=samples, station=station)
    if not self_check["passed"]:
        raise ValueError(f"GEFS decoder self-check failed: {self_check}")
    return BucketProbabilityResult(
        probability=members_in_bucket / len(samples) if samples else 0.0,
        member_count=len(samples),
        members_in_bucket=members_in_bucket,
        issue_ts=int(cycle.issue_time.timestamp()),
        issue_time=cycle.issue_time.isoformat(),
        cycle_hour=cycle.cycle_hour,
        lead_hours=tuple(leads),
        interpolation="nearest",
        decoder="eccodes.codes_grib_find_nearest",
        self_check=self_check,
    )


def tmax_message_ref(
    *,
    issue_time: datetime,
    member: str,
    lead_hour: int,
    text_fetcher: TextFetcher | None = None,
    base_url: str = GEFS_BASE_URL,
) -> TMaxMessageRef:
    text_fetcher = text_fetcher or _fetch_text
    if issue_time.tzinfo is None:
        raise ValueError("issue_time must be timezone-aware UTC")
    issue_utc = issue_time.astimezone(UTC)
    ymd = issue_utc.strftime("%Y%m%d")
    cycle = issue_utc.strftime("%H")
    filename = f"{member}.t{cycle}z.pgrb2s.0p25.f{lead_hour:03d}"
    url = f"{base_url}/gefs.{ymd}/{cycle}/atmos/pgrb2sp25/{filename}"
    index_url = f"{url}.idx"
    rows = _parse_idx_rows(text_fetcher(index_url))
    for idx, row in enumerate(rows):
        if row["short_name"] != "TMAX" or row["level"] != "2 m above ground":
            continue
        next_offset = rows[idx + 1]["offset"] if idx + 1 < len(rows) else None
        if next_offset is None:
            raise ValueError(f"cannot determine byte end for final idx row in {index_url}")
        return TMaxMessageRef(
            url=url,
            index_url=index_url,
            member=member,
            lead_hour=lead_hour,
            byte_start=row["offset"],
            byte_end=next_offset - 1,
            descriptor=row["descriptor"],
        )
    raise ValueError(f"TMAX 2 m above ground not found in {index_url}")


def gefs_archive_availability(
    *,
    station: StationSpec,
    target_date: date,
    decision_ts: int,
    members: Sequence[str] = GEFS_MEMBERS,
    text_fetcher: TextFetcher | None = None,
    base_url: str = GEFS_BASE_URL,
    max_missing_examples: int = 10,
) -> ArchiveAvailability:
    text_fetcher = text_fetcher or _fetch_text
    cycle = latest_gefs_cycle_before(decision_ts)
    if cycle is None:
        return ArchiveAvailability(0, 0, 0, ())
    leads = lead_hours_for_station_day(
        issue_time=cycle.issue_time,
        station=station,
        target_date=target_date,
    )
    required = len(members) * len(leads)
    available = 0
    missing: list[str] = []
    for member in members:
        for lead in leads:
            try:
                tmax_message_ref(
                    issue_time=cycle.issue_time,
                    member=member,
                    lead_hour=lead,
                    text_fetcher=text_fetcher,
                    base_url=base_url,
                )
            except Exception:
                if len(missing) < max_missing_examples:
                    missing.append(f"{member}:f{lead:03d}")
            else:
                available += 1
    return ArchiveAvailability(
        required_messages=required,
        available_messages=available,
        missing_messages=required - available,
        missing_examples=tuple(missing),
    )


def kelvin_to_fahrenheit(value_k: float) -> float:
    return (value_k - 273.15) * 9.0 / 5.0 + 32.0


def round_temperature_f(value_f: float) -> int:
    return math.floor(value_f + 0.5)


def decoder_self_check(
    *,
    samples: Sequence[StationForecastSample],
    station: StationSpec,
    actual_temperature_f: int | None = None,
    max_abs_error_f: float = 35.0,
) -> Mapping[str, Any]:
    if not samples:
        return {"passed": False, "reason": "no_samples", "station": station.code}
    values = [sample.daily_max_f for sample in samples]
    min_value = min(values)
    max_value = max(values)
    plausible = -80.0 <= min_value <= 140.0 and -80.0 <= max_value <= 140.0
    payload: dict[str, Any] = {
        "passed": plausible,
        "station": station.code,
        "member_count": len(samples),
        "min_daily_max_f": min_value,
        "max_daily_max_f": max_value,
        "plausible_fahrenheit_range": plausible,
    }
    if actual_temperature_f is not None:
        ensemble_mean = sum(values) / len(values)
        abs_error = abs(ensemble_mean - actual_temperature_f)
        payload["actual_temperature_f"] = actual_temperature_f
        payload["ensemble_mean_f"] = ensemble_mean
        payload["abs_error_f"] = abs_error
        payload["passed"] = plausible and abs_error <= max_abs_error_f
    return payload


def decoder_value_cross_check(
    *,
    primary_value_k: float,
    reference_value_k: float,
    max_abs_error_k: float = DECODER_VALUE_SELF_CHECK_MAX_ERROR_K,
) -> Mapping[str, Any]:
    if not math.isfinite(primary_value_k) or not math.isfinite(reference_value_k):
        return {
            "passed": False,
            "reason": "non_finite_decoder_value",
            "primary_value_k": primary_value_k,
            "reference_value_k": reference_value_k,
        }
    abs_error_k = abs(primary_value_k - reference_value_k)
    plausible = 180.0 <= primary_value_k <= 340.0 and 180.0 <= reference_value_k <= 340.0
    return {
        "passed": plausible and abs_error_k <= max_abs_error_k,
        "primary_value_k": primary_value_k,
        "reference_value_k": reference_value_k,
        "abs_error_k": abs_error_k,
        "max_abs_error_k": max_abs_error_k,
        "plausible_kelvin_range": plausible,
    }


def _parse_idx_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        try:
            row_number = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "row_number": row_number,
                "offset": offset,
                "short_name": parts[3],
                "level": parts[4],
                "descriptor": line,
            }
        )
    return rows


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return cast(bytes, response.read()).decode("utf-8")


def _fetch_range(url: str, byte_start: int, byte_end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": f"bytes={byte_start}-{byte_end}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return cast(bytes, response.read())


def _decode_station_tmax_k_checked(message: bytes, station: StationSpec) -> float:
    primary = _decode_station_tmax_k(message, station)
    reference = _decode_station_tmax_k_grid_scan(message, station)
    check = decoder_value_cross_check(
        primary_value_k=primary,
        reference_value_k=reference,
    )
    if not check["passed"]:
        raise ValueError(f"GEFS decoder value self-check failed: {check}")
    return primary


def _decode_station_tmax_k(message: bytes, station: StationSpec) -> float:
    eccodes = importlib.import_module("eccodes")
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(message)
        handle.flush()
        with Path(handle.name).open("rb") as grib_file:
            gid = eccodes.codes_grib_new_from_file(grib_file)
            if gid is None:
                raise ValueError("eccodes could not read GRIB2 message")
            try:
                short_name = eccodes.codes_get(gid, "shortName")
                units = eccodes.codes_get(gid, "units")
                if short_name != "tmax" or units != "K":
                    raise ValueError(f"unexpected GRIB field {short_name=} {units=}")
                nearest = eccodes.codes_grib_find_nearest(
                    gid,
                    station.latitude,
                    _to_360_longitude(station.longitude),
                )
                value = nearest[0].value
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"non-finite nearest TMAX value {value!r}")
                return float(value)
            finally:
                eccodes.codes_release(gid)


def _decode_station_tmax_k_grid_scan(message: bytes, station: StationSpec) -> float:
    eccodes = importlib.import_module("eccodes")
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(message)
        handle.flush()
        with Path(handle.name).open("rb") as grib_file:
            gid = eccodes.codes_grib_new_from_file(grib_file)
            if gid is None:
                raise ValueError("eccodes could not read GRIB2 message")
            try:
                short_name = eccodes.codes_get(gid, "shortName")
                units = eccodes.codes_get(gid, "units")
                if short_name != "tmax" or units != "K":
                    raise ValueError(f"unexpected GRIB field {short_name=} {units=}")
                latitudes = cast(Sequence[float], eccodes.codes_get_double_array(gid, "latitudes"))
                longitudes = cast(
                    Sequence[float], eccodes.codes_get_double_array(gid, "longitudes")
                )
                values = cast(Sequence[float], eccodes.codes_get_double_array(gid, "values"))
                if not (len(latitudes) == len(longitudes) == len(values)):
                    raise ValueError("GRIB latitude/longitude/value arrays have different lengths")
                target_lon = _to_360_longitude(station.longitude)
                best_value: float | None = None
                best_distance = float("inf")
                for latitude, longitude, value in zip(
                    latitudes,
                    longitudes,
                    values,
                    strict=True,
                ):
                    if not math.isfinite(value):
                        continue
                    distance = (float(latitude) - station.latitude) ** 2 + (
                        _to_360_longitude(float(longitude)) - target_lon
                    ) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_value = float(value)
                if best_value is None:
                    raise ValueError("no finite TMAX value found during grid-scan self-check")
                return best_value
            finally:
                eccodes.codes_release(gid)


def _to_360_longitude(value: float) -> float:
    return value % 360.0


def target_date_from_temperature_slug(slug: str) -> date | None:
    match = re.search(r"-on-([a-z]+)-(\d{1,2})-(\d{4})(?:-|$)", slug)
    if not match:
        return None
    month_name, day_text, year_text = match.groups()
    try:
        month = datetime.strptime(month_name, "%B").month
        return date(int(year_text), month, int(day_text))
    except ValueError:
        return None
