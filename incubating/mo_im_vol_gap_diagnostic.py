from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from aegis.mo_im_vol_gap_diagnostic import (
    DEFAULT_CONFIG,
    IndexBar,
    IvObservation,
    OptionQuote,
    atm_iv_snapshot_from_quotes,
    run_vol_gap_diagnostic,
)
from aegis.private_paths import private_dir_from_cli

BRIEFING = "CODEX_OLYMPUS_74_MO_IM_VOL_GAP_DIAGNOSTIC"
DEFAULT_TASK = "olympus74"
INDEX_SYMBOL = "sh000852"
INDEX_NAME = "中证1000"
QVIX_SOURCE = "akshare.index_option_1000index_qvix"
INDEX_SOURCE = "akshare.stock_zh_index_daily"


def main() -> int:
    generated_at = datetime.now(timezone.utc)  # noqa: UP017 - host Python can be 3.10.
    output_dir = private_dir_from_cli(
        os.getenv("MO_IM_VOL_GAP_OUTPUT_DIR"),
        default_task=DEFAULT_TASK,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    forward_dir = output_dir / "forward"
    forward_dir.mkdir(parents=True, exist_ok=True)

    fetch_report = _fetch_public_inputs()
    result = run_vol_gap_diagnostic(
        fetch_report["bars"],
        fetch_report["iv_observations"],
        config=DEFAULT_CONFIG,
        iv_source=QVIX_SOURCE,
    )
    forward_snapshot = _attempt_forward_snapshot(forward_dir=forward_dir, generated_at=generated_at)
    payload: dict[str, object] = {
        "generated_at": generated_at.isoformat(),
        "briefing": BRIEFING,
        "ev_newness": (
            "诊断模式: vol-arb/VRP 机制,非方向/技术类。已知逆风是 IV 通常高于 RV; "
            "本轮只测 MO+IM delta-neutral long-gamma/vega 买方的 IV/RV 前提。"
        ),
        "data_availability": fetch_report["availability"],
        "result": result,
        "forward_snapshot": forward_snapshot,
        "public_boundary": (
            "Private evidence only. Public repo contains generic diagnostic code and "
            "synthetic tests; no account, broker GUI, order path, or credentials."
        ),
    }

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"mo-im-vol-gap-diagnostic-{stamp}.json"
    md_path = output_dir / f"mo-im-vol-gap-diagnostic-{stamp}.md"
    json_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload, json_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "candidate_count_n": result.get("candidate_count_n"),
                "fdr_survivors": result.get("fdr_survivors"),
                "historical_iv_positive_rows": fetch_report["availability"].get(
                    "historical_iv_positive_rows"
                ),
                "forward_snapshot_ok": forward_snapshot.get("ok"),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _fetch_public_inputs() -> dict[str, Any]:
    ak = importlib.import_module("akshare")
    bars_report = _fetch_index_bars(ak)
    iv_report = _fetch_qvix_iv(ak)
    positive_iv_rows = _int_value(iv_report["positive_row_count"])
    total_iv_rows = _int_value(iv_report["row_count"])
    return {
        "bars": bars_report["bars"],
        "iv_observations": iv_report["iv_observations"],
        "availability": {
            "index_source": INDEX_SOURCE,
            "index_rows": bars_report["row_count"],
            "index_start": bars_report["start"],
            "index_end": bars_report["end"],
            "historical_iv_source": QVIX_SOURCE,
            "historical_iv_rows": iv_report["row_count"],
            "historical_iv_positive_rows": positive_iv_rows,
            "historical_iv_gate": (
                "available"
                if positive_iv_rows > 0
                else "unavailable_all_null_or_non_positive"
            ),
            "historical_iv_limit": _historical_iv_limit_text(
                positive_iv_rows=positive_iv_rows,
                total_rows=total_iv_rows,
            ),
        },
    }


def _fetch_index_bars(ak: Any) -> dict[str, object]:
    raw = ak.stock_zh_index_daily(symbol=INDEX_SYMBOL)
    rows = cast(list[dict[str, Any]], raw.to_dict("records"))
    bars: list[IndexBar] = []
    for row in rows:
        timestamp = _date_key(row.get("date"))
        bars.append(
            IndexBar(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        )
    bars.sort(key=lambda item: item.timestamp)
    return {
        "bars": bars,
        "row_count": len(bars),
        "start": bars[0].timestamp if bars else None,
        "end": bars[-1].timestamp if bars else None,
    }


def _fetch_qvix_iv(ak: Any) -> dict[str, object]:
    raw = ak.index_option_1000index_qvix()
    rows = cast(list[dict[str, Any]], raw.to_dict("records"))
    observations: list[IvObservation] = []
    for row in rows:
        value = _float_or_none(row.get("close"))
        if value is None or value <= 0.0:
            continue
        observations.append(
            IvObservation(
                timestamp=_date_key(row.get("date")),
                annualized_iv=_normalize_iv(value),
                source=QVIX_SOURCE,
                source_quality="historical_qvix_proxy",
            )
        )
    observations.sort(key=lambda item: item.timestamp)
    return {
        "iv_observations": observations,
        "row_count": len(rows),
        "positive_row_count": len(observations),
    }


def _historical_iv_limit_text(*, positive_iv_rows: int, total_rows: int) -> str:
    if positive_iv_rows > 0:
        return (
            "AKShare exposes a 1000-index QVIX history with partial positive finite closes "
            f"({positive_iv_rows}/{total_rows}). This is an IV proxy rather than PIT ATM chain "
            "IV, so any positive buyer result would remain capped, but it is sufficient for the "
            "requested cheap IV/RV gap diagnostic."
        )
    return (
        "AKShare exposes the 1000-index QVIX table, but current fetch returned no positive "
        "finite IV closes. The diagnostic therefore fails closed and switches to forward ATM "
        "MO IV collection."
    )


def _attempt_forward_snapshot(
    *,
    forward_dir: Path,
    generated_at: datetime,
) -> dict[str, object]:
    try:
        ak = importlib.import_module("akshare")
        underlying = _current_zz1000_spot(ak)
        contract = _nearest_contract(
            cast(Mapping[str, Sequence[str]], ak.option_cffex_zz1000_list_sina())
        )
        raw_chain = ak.option_cffex_zz1000_spot_sina(symbol=contract)
        quotes = _option_quotes_from_chain(
            cast(list[dict[str, Any]], raw_chain.to_dict("records")),
            contract=contract,
            underlying=underlying,
            as_of=generated_at.date(),
        )
        snapshot = atm_iv_snapshot_from_quotes(quotes)
        snapshot = {
            **snapshot,
            "generated_at": generated_at.isoformat(),
            "source": "akshare.option_cffex_zz1000_spot_sina + stock_zh_index_spot_sina",
            "contract": contract,
            "read_only": True,
            "note": (
                "Forward proxy only: ATM IV is inverted from current option mid quotes with "
                "Black-Scholes assumptions. It is for future diagnostics, not a live signal."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        snapshot = {
            "ok": False,
            "generated_at": generated_at.isoformat(),
            "reason": repr(exc),
            "source": "akshare forward MO chain probe",
            "read_only": True,
        }
    path = forward_dir / "mo_atm_iv_snapshots.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(snapshot), ensure_ascii=False, sort_keys=True) + "\n")
    return {**snapshot, "path": str(path)}


def _current_zz1000_spot(ak: Any) -> float:
    raw = ak.stock_zh_index_spot_sina()
    rows = cast(list[dict[str, Any]], raw.to_dict("records"))
    for row in rows:
        if str(row.get("代码")) == INDEX_SYMBOL or str(row.get("名称")) == INDEX_NAME:
            return float(row["最新价"])
    raise ValueError("could not find current CSI 1000 spot in stock_zh_index_spot_sina")


def _nearest_contract(raw: Mapping[str, Sequence[str]]) -> str:
    contracts = list(raw.get("中证1000指数", ()))
    if not contracts:
        raise ValueError("no MO contracts returned by option_cffex_zz1000_list_sina")
    today = date.today()
    dated = [(contract, _expiry_from_mo_contract(contract)) for contract in contracts]
    future = [(contract, expiry) for contract, expiry in dated if expiry >= today]
    if not future:
        raise ValueError("no non-expired MO contracts returned by option list")
    return min(future, key=lambda item: item[1])[0]


def _option_quotes_from_chain(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: str,
    underlying: float,
    as_of: date,
) -> list[OptionQuote]:
    expiry = _expiry_from_mo_contract(contract)
    quotes: list[OptionQuote] = []
    for row in rows:
        strike = float(row["行权价"])
        call_bid = _float_or_none(row.get("看涨合约-买价"))
        call_ask = _float_or_none(row.get("看涨合约-卖价"))
        put_bid = _float_or_none(row.get("看跌合约-买价"))
        put_ask = _float_or_none(row.get("看跌合约-卖价"))
        if call_bid is not None and call_ask is not None:
            quotes.append(
                OptionQuote(
                    option_type="call",
                    strike=strike,
                    bid=call_bid,
                    ask=call_ask,
                    expiry=expiry,
                    underlying=underlying,
                    as_of=as_of,
                )
            )
        if put_bid is not None and put_ask is not None:
            quotes.append(
                OptionQuote(
                    option_type="put",
                    strike=strike,
                    bid=put_bid,
                    ask=put_ask,
                    expiry=expiry,
                    underlying=underlying,
                    as_of=as_of,
                )
            )
    return quotes


def _expiry_from_mo_contract(contract: str) -> date:
    digits = "".join(character for character in contract if character.isdigit())
    if len(digits) < 4:
        raise ValueError(f"cannot parse MO contract expiry from {contract!r}")
    year = 2000 + int(digits[:2])
    month = int(digits[2:4])
    first = date(year, month, 1)
    days_until_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_until_friday)
    return first_friday + timedelta(days=14)


def _date_key(value: object) -> int:
    text = str(value)[:10]
    return int(text.replace("-", ""))


def _float_or_none(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid integer field")
    if isinstance(value, int):
        return value
    raise TypeError(f"expected int field, got {type(value).__name__}")


def _normalize_iv(value: float) -> float:
    return value / 100.0 if value > 3.0 else value


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _markdown(payload: Mapping[str, object], json_path: Path) -> str:
    result = cast(Mapping[str, object], payload["result"])
    availability = cast(Mapping[str, object], payload["data_availability"])
    forward = cast(Mapping[str, object], payload["forward_snapshot"])
    return "\n".join(
        [
            "# CODEX OLYMPUS 74 MO+IM Vol Gap Diagnostic",
            "",
            f"- Verdict: `{result.get('verdict')}`",
            f"- Reason: {result.get('reason')}",
            f"- Historical IV source: `{availability.get('historical_iv_source')}`",
            f"- Historical IV positive rows: `{availability.get('historical_iv_positive_rows')}`",
            f"- Index source: `{availability.get('index_source')}`",
            f"- Index rows: `{availability.get('index_rows')}`",
            f"- Forward snapshot ok: `{forward.get('ok')}`",
            f"- Forward snapshot path: `{forward.get('path')}`",
            f"- JSON: `{json_path}`",
            "",
            "## Interpretation",
            "",
            "Historical ATM MO IV was not available from the free QVIX proxy, so the diagnostic "
            "fails closed as INSUFFICIENT. The script appended a read-only forward ATM-IV proxy "
            "snapshot when the current public option chain was available. Several weeks of these "
            "snapshots are needed before IV_t vs forward RV can be diagnosed.",
            "",
            "## Safety",
            "",
            "- Read-only public data only.",
            "- No GUI, account, order API, live trading, or credentials.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
