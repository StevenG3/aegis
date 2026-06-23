from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from aegis.gefs_weather_probability import (
    SampledDecoderValueSelfCheck,
    StationSpec,
    bucket_probability_from_gefs,
    decoder_value_cross_check,
    gefs_archive_availability,
    kelvin_to_fahrenheit,
    latest_gefs_cycle_before,
    lead_hours_for_station_day,
    round_temperature_f,
    tmax_message_ref,
)
from aegis.polymarket_weather_relative_value import TemperatureBucket


def _idx_text() -> str:
    return "\n".join(
        [
            "1:0:d=2026050106:TMP:2 m above ground:6 hour fcst:ENS=+1",
            "2:100:d=2026050106:TMAX:2 m above ground:0-6 hour max fcst:ENS=+1",
            "3:250:d=2026050106:TMIN:2 m above ground:0-6 hour min fcst:ENS=+1",
        ]
    )


def test_latest_gefs_cycle_is_strictly_before_decision() -> None:
    decision = int(datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp())
    cycle = latest_gefs_cycle_before(decision)
    assert cycle is not None
    assert cycle.issue_time == datetime(2026, 5, 1, 6, 0, tzinfo=UTC)

    exact_cycle_decision = int(datetime(2026, 5, 1, 6, 0, tzinfo=UTC).timestamp())
    previous = latest_gefs_cycle_before(exact_cycle_decision)
    assert previous is not None
    assert previous.issue_time == datetime(2026, 5, 1, 0, 0, tzinfo=UTC)


def test_lead_hours_overlap_station_local_day() -> None:
    station = StationSpec("KLAX", 33.9416, -118.4085, "America/Los_Angeles")
    leads = lead_hours_for_station_day(
        issue_time=datetime(2026, 5, 1, 6, 0, tzinfo=UTC),
        station=station,
        target_date=date(2026, 5, 1),
        lead_hours=(6, 12, 18, 24, 30, 36),
    )
    assert leads == (6, 12, 18, 24, 30)


def test_tmax_message_ref_uses_idx_byte_range() -> None:
    ref = tmax_message_ref(
        issue_time=datetime(2026, 5, 1, 6, 0, tzinfo=UTC),
        member="gep01",
        lead_hour=6,
        text_fetcher=lambda _: _idx_text(),
        base_url="https://example.test",
    )
    assert ref.byte_start == 100
    assert ref.byte_end == 249
    assert ref.member == "gep01"
    assert ref.lead_hour == 6


def test_bucket_probability_requires_issue_before_decision_and_counts_members() -> None:
    station = StationSpec("KLGA", 40.7769, -73.8740, "America/New_York")
    requested_ranges: list[tuple[str, int, int]] = []

    def byte_fetcher(url: str, start: int, end: int) -> bytes:
        requested_ranges.append((url, start, end))
        return b"synthetic"

    member_values_k = {
        "gep01": 300.0,
        "gep02": 301.0,
        "gep03": 305.0,
    }

    def decoder(_: bytes, station_arg: StationSpec) -> float:
        assert station_arg == station
        member = requested_ranges[-1][0].split("/")[-1].split(".")[0]
        return member_values_k[member]

    decision = int(datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp())
    report = bucket_probability_from_gefs(
        station=station,
        target_date=date(2026, 5, 1),
        decision_ts=decision,
        bucket=TemperatureBucket("80-84°F", lower_f=80, upper_f=84),
        members=("gep01", "gep02", "gep03"),
        text_fetcher=lambda _: _idx_text(),
        byte_fetcher=byte_fetcher,
        decoder=decoder,
        base_url="https://example.test",
    )
    assert report.issue_ts < decision
    assert report.member_count == 3
    assert report.members_in_bucket == 2
    assert report.probability == pytest.approx(2 / 3)
    assert report.interpolation == "nearest"


def test_archive_availability_reports_missing_messages() -> None:
    station = StationSpec("KLGA", 40.7769, -73.8740, "America/New_York")

    def text_fetcher(url: str) -> str:
        if "gep02" in url:
            raise FileNotFoundError(url)
        return _idx_text()

    decision = int(datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp())
    report = gefs_archive_availability(
        station=station,
        target_date=date(2026, 5, 1),
        decision_ts=decision,
        members=("gep01", "gep02"),
        text_fetcher=text_fetcher,
    )
    assert report.required_messages > 0
    assert report.missing_messages > 0
    assert report.missing_examples


def test_temperature_unit_helpers() -> None:
    assert kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert round_temperature_f(84.49) == 84
    assert round_temperature_f(84.50) == 85


def test_decoder_value_cross_check_fails_loud_on_large_disagreement() -> None:
    passed = decoder_value_cross_check(primary_value_k=300.00, reference_value_k=300.02)
    assert passed["passed"] is True

    failed = decoder_value_cross_check(primary_value_k=300.00, reference_value_k=301.00)
    assert failed["passed"] is False
    assert failed["abs_error_k"] == pytest.approx(1.0)


def test_sampled_decoder_value_self_check_tracks_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    station = StationSpec("KLGA", 40.7769, -73.8740, "America/New_York")
    import aegis.gefs_weather_probability as module

    checked_calls = 0
    plain_calls = 0

    def checked(_: bytes, __: StationSpec) -> float:
        nonlocal checked_calls
        checked_calls += 1
        return 300.0

    def plain(_: bytes, __: StationSpec) -> float:
        nonlocal plain_calls
        plain_calls += 1
        return 300.0

    monkeypatch.setattr(module, "_decode_station_tmax_k_checked", checked)
    monkeypatch.setattr(module, "_decode_station_tmax_k", plain)
    decoder = SampledDecoderValueSelfCheck(max_checks=2)

    assert decoder(b"message-1", station) == 300.0
    assert decoder(b"message-2", station) == 300.0
    assert decoder(b"message-3", station) == 300.0
    assert decoder.checks_run == 2
    assert checked_calls == 2
    assert plain_calls == 1
