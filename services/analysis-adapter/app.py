from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol
from uuid import UUID, uuid4

import httpx
from db import connect, db_path
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="analysis-adapter", version="0.1.0")

TA_BRIDGE_URL = os.getenv("TA_BRIDGE_URL", "http://tradingagents-bridge:18181")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
MARKET_DATA_URL = os.getenv("MARKET_DATA_URL", "http://market-data:8083")
TA_DEFAULT_PROVIDER = os.getenv("TA_DEFAULT_PROVIDER", "deepseek")
LLM_PROVIDER_CHAIN = os.getenv("LLM_PROVIDER_CHAIN", "")
LLM_DEFAULT_MODEL_BY_PROVIDER = {
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}
LLM_COST_PER_1K_USD = {
    "deepseek": float(os.getenv("DEEPSEEK_COST_PER_1K_USD", "0")),
    "anthropic": float(os.getenv("ANTHROPIC_COST_PER_1K_USD", "0")),
    "openai": float(os.getenv("OPENAI_COST_PER_1K_USD", "0")),
}
TA_TIMEOUT_SEC = float(os.getenv("TA_TIMEOUT_SEC", "900"))
SCORECARD_TTL_MIN = int(os.getenv("ANALYSIS_SCORECARD_TTL_MIN", "60"))
JobStatus = Literal["queued", "running", "succeeded", "failed"]
AnalystName = Literal["market", "social", "news", "fundamentals"]

CRYPTO_NAME_TO_SYMBOL = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "binancecoin": "BNB",
    "bnb": "BNB",
    "xrp": "XRP",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
}
QUOTE_ASSETS = ("USDT", "USDC", "USD", "BUSD")
BENCHMARK_FOR_ASSET = {"crypto": "BTCUSDT", "stock": "SPY"}


RESEARCH_RATING_TO_DECISION = {
    "STRONG BUY": "BUY",
    "OUTPERFORM": "BUY",
    "OVERWEIGHT": "BUY",
    "ACCUMULATE": "BUY",
    "BUY": "BUY",
    "MARKET WEIGHT": "HOLD",
    "EQUAL WEIGHT": "HOLD",
    "NEUTRAL": "HOLD",
    "HOLD": "HOLD",
    "REDUCE": "SELL",
    "UNDERWEIGHT": "SELL",
    "UNDERPERFORM": "SELL",
    "SELL": "SELL",
    "STRONG SELL": "SELL",
}


def _decision_from_research_rating(value: str) -> str | None:
    normalized = re.sub(r"[^A-Z ]", " ", value.upper())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for rating, decision in RESEARCH_RATING_TO_DECISION.items():
        if re.search(rf"\b{re.escape(rating)}\b", normalized):
            return decision
    return None


def _normalize_ta_ticker(symbol: str, asset_type: str) -> str:
    """Convert user/trading crypto symbols to TradingAgents-friendly tickers."""
    cleaned = symbol.strip()
    if asset_type != "crypto":
        return cleaned.upper()
    compact = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    if not compact:
        return cleaned
    base = CRYPTO_NAME_TO_SYMBOL.get(cleaned.strip().lower(), compact)
    for quote in QUOTE_ASSETS:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            break
    if base.endswith("USD") and "-" not in cleaned:
        base = base[:-3]
    return f"{base}-USD"


def _canonical_crypto_symbol(symbol: str, asset_type: str) -> str:
    cleaned = symbol.strip()
    if asset_type != "crypto":
        return cleaned.upper()
    compact = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    if not compact:
        return cleaned
    base = CRYPTO_NAME_TO_SYMBOL.get(cleaned.lower(), compact)
    for quote in QUOTE_ASSETS:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            break
    if base.endswith("USD"):
        base = base[:-3]
    return f"{base}USDT"


def _bucket_label_for(heuristic: float) -> str | None:
    for lo, hi in [
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]:
        if lo <= heuristic < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def _calibrated_conviction(heuristic: float, source: str, asset_type: str) -> float:
    bucket = _bucket_label_for(heuristic)
    if bucket is None:
        return heuristic
    try:
        response = httpx.get(
            f"{ORCHESTRATOR_URL}/calibration",
            params={"source": source, "asset_type": asset_type},
            timeout=3.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return heuristic
    items = body.get("items", []) if isinstance(body, dict) else []
    if not isinstance(items, list):
        return heuristic
    for row in items:
        if not isinstance(row, dict) or row.get("heuristic_bucket") != bucket:
            continue
        if int(row.get("sample_count", 0)) < int(body.get("min_samples", 5)):
            return heuristic
        try:
            calibrated = row.get("calibrated_conviction")
            if calibrated is None:
                return heuristic
            return float(str(calibrated))
        except (TypeError, ValueError):
            return heuristic
    return heuristic


def _benchmark_for(asset_type: str) -> str:
    return BENCHMARK_FOR_ASSET.get(asset_type, "SPY")


def _fetch_benchmark_price(symbol: str) -> str | None:
    try:
        response = httpx.get(
            f"{MARKET_DATA_URL.rstrip('/')}/ticker", params={"symbol": symbol}, timeout=3.0
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["price"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _decimal_str(value: object) -> str | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if parsed <= 0:
        return None
    return format(parsed.quantize(Decimal("0.00000001")).normalize(), "f")


def _fetch_symbol_price(symbol: str, asset_type: str) -> Decimal | None:
    try:
        params: dict[str, str] = {"symbol": symbol}
        if asset_type == "stock":
            params["asset_type"] = "stock"
        response = httpx.get(
            f"{MARKET_DATA_URL.rstrip('/')}/ticker",
            params=params,
            timeout=3.0,
        )
        response.raise_for_status()
        payload = response.json()
        price = Decimal(str(payload["price"]))
        return price if price > 0 else None
    except (httpx.HTTPError, KeyError, InvalidOperation, ValueError):
        return None


def _scorecard_price_levels(
    req: AnalyzeRequest,
    raw: dict[str, object],
    action: str,
) -> dict[str, str | None]:
    explicit = {
        "entry_low": _decimal_str(raw.get("entry_low")),
        "entry_high": _decimal_str(raw.get("entry_high")),
        "stop_loss": _decimal_str(raw.get("stop_loss")),
        "take_profit": _decimal_str(raw.get("take_profit")),
    }
    if all(explicit.values()) or action == "hold":
        return explicit

    reference = _decimal_str(raw.get("reference_price"))
    mark = (
        Decimal(reference)
        if reference is not None
        else _fetch_symbol_price(req.symbol, req.asset_type)
    )
    if mark is None:
        return explicit
    if action == "sell":
        stop_loss = mark * Decimal("1.05")
        take_profit = mark * Decimal("0.90")
    else:
        stop_loss = mark * Decimal("0.95")
        take_profit = mark * Decimal("1.10")
    return {
        "entry_low": _decimal_str(mark),
        "entry_high": _decimal_str(mark),
        "stop_loss": _decimal_str(stop_loss),
        "take_profit": _decimal_str(take_profit),
    }


def _default_analysts() -> list[AnalystName]:
    return ["market", "news"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    asset_type: Literal["stock", "crypto"] = "crypto"
    analysts: list[AnalystName] = Field(default_factory=_default_analysts)
    provider: str | None = None
    dry_run: bool = False
    origin: str | None = None
    gate_conviction: str | None = None


class ReflectOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    trade_date: str = Field(min_length=10, max_length=10)
    raw_return: str | float
    alpha_return: str | float
    holding_days: int = Field(ge=0)
    provider: str | None = None
    benchmark_name: str | None = None


AnalyzeRequest.model_rebuild(_types_namespace={"Literal": Literal, "AnalystName": AnalystName})


def _now() -> datetime:
    return datetime.now(UTC)


class LLMGatewayError(RuntimeError):
    """Raised when no configured LLM provider can complete the analysis request."""


class LLMProviderCall(Protocol):
    def __call__(self, provider: str, model: str) -> dict[str, object]: ...


def _csv_items(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _provider_chain(requested_provider: str | None) -> list[str]:
    """Return a de-duplicated provider chain with request/default compatibility first."""
    candidates: list[str] = []
    if requested_provider:
        candidates.append(requested_provider)
    if TA_DEFAULT_PROVIDER:
        candidates.append(TA_DEFAULT_PROVIDER)
    candidates.extend(_csv_items(os.getenv("LLM_PROVIDER_CHAIN", LLM_PROVIDER_CHAIN)))
    if not candidates:
        candidates = ["deepseek"]

    seen: set[str] = set()
    chain: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            chain.append(normalized)
    return chain


def _model_for(provider: str, task_hint: str | None) -> str:
    if task_hint:
        key = f"{provider.upper()}_{re.sub(r'[^A-Za-z0-9]+', '_', task_hint).upper()}_MODEL"
        task_model = os.getenv(key)
        if task_model:
            return task_model
    return os.getenv(
        f"{provider.upper()}_MODEL",
        LLM_DEFAULT_MODEL_BY_PROVIDER.get(provider, f"{provider}-default"),
    )


def _estimate_tokens_from_text(value: object) -> int:
    if value is None:
        return 0
    return max(1, len(str(value)) // 4)


def _extract_token_usage(
    body: dict[str, object], payload: dict[str, object]
) -> tuple[int, int, int]:
    usage = body.get("usage") or body.get("token_usage")
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        if total_tokens > 0:
            return input_tokens, output_tokens, total_tokens

    reports = body.get("reports", {})
    input_tokens = _estimate_tokens_from_text(json.dumps(payload, default=str))
    output_tokens = _estimate_tokens_from_text(
        json.dumps(reports, default=str) if isinstance(reports, dict) else body
    )
    return input_tokens, output_tokens, input_tokens + output_tokens


def _estimated_cost_usd(provider: str, total_tokens: int) -> str:
    per_1k = float(
        os.getenv(
            f"{provider.upper()}_COST_PER_1K_USD",
            str(LLM_COST_PER_1K_USD.get(provider, 0.0)),
        )
    )
    return f"{(total_tokens / 1000.0) * per_1k:.8f}"


def _sanitize_error(exc: BaseException | str) -> str:
    text = str(exc)
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[redacted]", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret)=([^&\s]+)", r"\1=[redacted]", text)
    return _truncate(text, 300)


def _record_llm_call(
    *,
    job_id: str | None,
    task: str,
    provider: str,
    model: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    est_cost_usd: str,
    success: bool,
    error: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            "insert into llm_calls "
            "(call_id, job_id, task, provider, model, latency_ms, input_tokens, "
            "output_tokens, total_tokens, est_cost_usd, success, error, created_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid4()),
                job_id,
                task,
                provider,
                model,
                latency_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                est_cost_usd,
                1 if success else 0,
                error,
                _now().isoformat(),
            ),
        )
        conn.commit()


def _complete_with_fallback(
    *,
    requested_provider: str | None,
    task_hint: str,
    job_id: str | None,
    call_provider: LLMProviderCall,
) -> dict[str, object]:
    errors: list[str] = []
    for provider in _provider_chain(requested_provider):
        model = _model_for(provider, task_hint)
        started = time.monotonic()
        try:
            body = call_provider(provider, model)
            latency_ms = int((time.monotonic() - started) * 1000)
            if not body.get("ok", False):
                raise LLMGatewayError(str(body.get("error", "provider returned ok=false")))
            input_tokens, output_tokens, total_tokens = _extract_token_usage(
                body, {"provider": provider, "model": model, "task": task_hint}
            )
            _record_llm_call(
                job_id=job_id,
                task=task_hint,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                est_cost_usd=_estimated_cost_usd(provider, total_tokens),
                success=True,
            )
            body["provider"] = str(body.get("provider") or provider)
            body["model"] = str(body.get("model") or model)
            body["llm_gateway"] = {
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
                "total_tokens": total_tokens,
                "est_cost_usd": _estimated_cost_usd(provider, total_tokens),
            }
            return body
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - started) * 1000)
            error = _sanitize_error(exc)
            errors.append(f"{provider}: {error}")
            _record_llm_call(
                job_id=job_id,
                task=task_hint,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                est_cost_usd="0.00000000",
                success=False,
                error=error,
            )
    raise LLMGatewayError("All LLM providers failed for analysis task; " + "; ".join(errors))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    db_path()
    return {"status": "ready"}


@app.get("/llm/calls", response_model=None)
def list_llm_calls(
    job_id: str | None = None,
    provider: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if job_id:
        clauses.append("job_id = ?")
        params.append(job_id)
    if provider:
        clauses.append("provider = ?")
        params.append(provider.lower())
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            "select call_id, job_id, task, provider, model, latency_ms, input_tokens, "
            "output_tokens, total_tokens, est_cost_usd, success, error, created_at "
            f"from llm_calls{where} order by created_at desc limit ?",
            [*params, limit],
        ).fetchall()
    return {
        "items": [
            {
                "call_id": row["call_id"],
                "job_id": row["job_id"],
                "task": row["task"],
                "provider": row["provider"],
                "model": row["model"],
                "latency_ms": row["latency_ms"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
                "est_cost_usd": row["est_cost_usd"],
                "success": bool(row["success"]),
                "error": row["error"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }


@app.post("/analyze", response_model=None)
def analyze(req: AnalyzeRequest) -> dict[str, object]:
    req = req.model_copy(update={"symbol": _canonical_crypto_symbol(req.symbol, req.asset_type)})
    job_id = str(uuid4())
    requested_at = _now().isoformat()
    with connect() as conn:
        conn.execute(
            "insert into analysis_jobs "
            "(job_id, actor, symbol, asset_type, requested_at, status) "
            "values (?,?,?,?,?,?)",
            (job_id, req.actor, req.symbol, req.asset_type, requested_at, "queued"),
        )
        conn.commit()
    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, req),
        daemon=True,
        name=f"analysis-{job_id[:8]}",
    )
    worker.start()
    return {"job_id": job_id, "status": "queued", "requested_at": requested_at}


@app.post("/reflect/outcome", response_model=None)
def reflect_outcome(req: ReflectOutcomeRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "ticker": req.ticker,
        "date": req.trade_date,
        "raw_return": float(req.raw_return),
        "alpha_return": float(req.alpha_return),
        "holding_days": req.holding_days,
        "provider": req.provider or TA_DEFAULT_PROVIDER,
    }
    if req.benchmark_name:
        payload["benchmark_name"] = req.benchmark_name
    response = httpx.post(f"{TA_BRIDGE_URL}/reflect", json=payload, timeout=TA_TIMEOUT_SEC)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("TA bridge returned non-dict body")
    return body


@app.get("/jobs/{job_id}", response_model=None)
def get_job(job_id: UUID) -> JSONResponse | dict[str, object]:
    with connect() as conn:
        row = conn.execute(
            "select actor, symbol, asset_type, requested_at, finished_at, "
            "status, scorecard_id, error from analysis_jobs where job_id = ?",
            (str(job_id),),
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"code": "JOB_NOT_FOUND"})
    return {
        "job_id": str(job_id),
        "actor": row["actor"],
        "symbol": row["symbol"],
        "asset_type": row["asset_type"],
        "requested_at": row["requested_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "scorecard_id": row["scorecard_id"],
        "error": row["error"],
    }


@app.get("/jobs", response_model=None)
def list_jobs(
    actor: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    clauses: list[str] = []
    params: list[object] = []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" where " + " and ".join(clauses)) if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"select job_id, actor, symbol, status, scorecard_id, requested_at, finished_at "
            f"from analysis_jobs{where} order by requested_at desc limit ?",
            [*params, limit],
        ).fetchall()
    return {
        "items": [
            {
                "job_id": row["job_id"],
                "actor": row["actor"],
                "symbol": row["symbol"],
                "status": row["status"],
                "scorecard_id": row["scorecard_id"],
                "requested_at": row["requested_at"],
                "finished_at": row["finished_at"],
            }
            for row in rows
        ]
    }


def _run_analysis_job(job_id: str, req: AnalyzeRequest) -> None:
    """Worker thread entrypoint. Never raises."""
    _mark_status(job_id, "running")
    try:
        raw = _stub_ta_response(req) if req.dry_run else _call_ta_bridge(job_id, req)
        try:
            scorecard_payload = _translate_to_scorecard_payload(req, raw)
            scorecard_id = _post_scorecard(scorecard_payload)
            _mark_success(job_id, scorecard_id, raw)
        except Exception as exc:  # noqa: BLE001
            _mark_failure(job_id, str(exc)[:500], raw)
    except Exception as exc:  # noqa: BLE001
        _mark_failure(job_id, str(exc)[:500])


def _call_ta_bridge(job_id: str, req: AnalyzeRequest) -> dict[str, object]:
    base_payload: dict[str, object] = {
        "ticker": _normalize_ta_ticker(req.symbol, req.asset_type),
        "date": _now().strftime("%Y-%m-%d"),
        "asset_type": req.asset_type,
        "analysts": list(req.analysts),
        "dry_run": False,
    }

    def call_provider(provider: str, model: str) -> dict[str, object]:
        payload = {**base_payload, "provider": provider, "model": model}
        response = httpx.post(
            f"{TA_BRIDGE_URL}/analyze",
            json=payload,
            timeout=TA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("TA bridge returned non-dict body")
        return body

    return _complete_with_fallback(
        requested_provider=req.provider,
        task_hint=f"analysis_{req.asset_type}",
        job_id=job_id,
        call_provider=call_provider,
    )


def _stub_ta_response(req: AnalyzeRequest) -> dict[str, object]:
    """Deterministic stub used when dry_run=True. Bypasses real TA."""
    return {
        "ok": True,
        "dry_run": True,
        "ticker": _normalize_ta_ticker(req.symbol, req.asset_type),
        "provider": req.provider or TA_DEFAULT_PROVIDER,
        "reference_price": "100",
        "entry_low": "100",
        "entry_high": "100",
        "stop_loss": "95",
        "take_profit": "110",
        "decision": "BUY",
        "final_trade_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "reports": {
            "market": "(dry-run stub) bullish technicals",
            "sentiment": "(dry-run stub) positive social",
            "news": "(dry-run stub) no negative news",
            "fundamentals": "(dry-run stub) healthy on-chain metrics",
        },
    }


def _translate_to_scorecard_payload(
    req: AnalyzeRequest, raw: dict[str, object]
) -> dict[str, object]:
    decision_text = str(raw.get("decision", ""))
    decision = decision_text.strip().upper()
    if decision not in {"BUY", "HOLD", "SELL"}:
        decision = _decision_from_research_rating(decision_text) or ""
    if decision not in {"BUY", "HOLD", "SELL"}:
        final_text = str(raw.get("final_trade_decision", ""))
        decision = _decision_from_research_rating(final_text) or ""
        for token in ("BUY", "SELL", "HOLD"):
            if decision:
                break
            if f"**{token}**" in final_text.upper():
                decision = token
                break
    if decision not in {"BUY", "HOLD", "SELL"}:
        raise ValueError("Could not extract a Buy/Hold/Sell action from TA output")

    action = decision.lower()
    reports = raw.get("reports", {}) or {}
    if not isinstance(reports, dict):
        reports = {}
    heuristic = _derive_conviction(action, reports)
    calibrated = _calibrated_conviction(
        heuristic=heuristic, source="tradingagents", asset_type=req.asset_type
    )
    conviction = calibrated
    metadata = {f"report_{k}": _truncate(str(v), 2000) for k, v in reports.items() if v}
    metadata["heuristic_conviction"] = f"{heuristic:.4f}"
    metadata["calibrated_conviction"] = f"{calibrated:.4f}"
    metadata["ta_decision"] = decision
    metadata["asset_type"] = req.asset_type
    if req.origin:
        metadata["origin"] = req.origin
    if req.gate_conviction:
        metadata["gate_conviction"] = req.gate_conviction
    metadata["provider"] = str(raw.get("provider", req.provider or TA_DEFAULT_PROVIDER))
    if raw.get("model"):
        metadata["model"] = str(raw["model"])
    gateway_meta = raw.get("llm_gateway")
    if isinstance(gateway_meta, dict):
        metadata["llm_provider"] = str(gateway_meta.get("provider", metadata["provider"]))
        metadata["llm_model"] = str(gateway_meta.get("model", raw.get("model", "")))
        metadata["llm_latency_ms"] = str(gateway_meta.get("latency_ms", ""))
        metadata["llm_total_tokens"] = str(gateway_meta.get("total_tokens", ""))
        metadata["llm_est_cost_usd"] = str(gateway_meta.get("est_cost_usd", ""))
    metadata["ta_date"] = str(raw.get("date") or _now().strftime("%Y-%m-%d"))
    metadata["ta_ticker"] = str(
        raw.get("ticker") or _normalize_ta_ticker(req.symbol, req.asset_type)
    )
    benchmark_symbol = _benchmark_for(req.asset_type)
    if benchmark_symbol.upper() == req.symbol.upper():
        metadata["benchmark_symbol"] = "self"
    else:
        metadata["benchmark_symbol"] = benchmark_symbol
        benchmark_open_price = _fetch_benchmark_price(benchmark_symbol)
        if benchmark_open_price is not None:
            metadata["benchmark_open_price"] = benchmark_open_price

    thesis_parts: list[str] = []
    if isinstance(reports.get("market"), str):
        thesis_parts.append("Market: " + _truncate(str(reports["market"]), 600))
    if isinstance(reports.get("news"), str):
        thesis_parts.append("News: " + _truncate(str(reports["news"]), 600))
    thesis = " | ".join(thesis_parts) or f"TradingAgents decided {decision} for {req.symbol}"
    price_levels = _scorecard_price_levels(req, raw, action)

    return {
        "actor": req.actor,
        "symbol": req.symbol,
        "action": action,
        "source": "tradingagents",
        "conviction": f"{conviction:.4f}",
        "thesis": _truncate(thesis, 3900),
        "entry_low": price_levels["entry_low"],
        "entry_high": price_levels["entry_high"],
        "stop_loss": price_levels["stop_loss"],
        "take_profit": price_levels["take_profit"],
        "time_horizon": "swing",
        "ttl_minutes": SCORECARD_TTL_MIN,
        "metadata": metadata,
        "factors": _factor_signals(action, reports),
    }


def _factor_direction(action: str, report: str) -> str:
    text = report.lower()
    bullish = any(
        token in text
        for token in (
            "bullish",
            "buy",
            "positive",
            "upside",
            "outperform",
            "accumulate",
            "supportive",
        )
    )
    bearish = any(
        token in text
        for token in (
            "bearish",
            "sell",
            "negative",
            "downside",
            "underperform",
            "reduce",
            "risk",
        )
    )
    if bullish == bearish:
        return "neutral"
    if action == "buy":
        return "support" if bullish else "oppose"
    if action == "sell":
        return "support" if bearish else "oppose"
    return "neutral"


def _factor_signals(action: str, reports: dict[str, object]) -> list[dict[str, object]]:
    factors: list[dict[str, object]] = []
    for name in ("market", "social", "news", "fundamentals"):
        report = reports.get(name)
        if not isinstance(report, str) or not report.strip():
            continue
        factors.append(
            {
                "name": name,
                "direction": _factor_direction(action, report),
            }
        )
    return factors


def _derive_conviction(action: str, reports: dict[str, object]) -> float:
    if action == "hold":
        return 0.30
    score = 0.50
    for key in ("market", "social", "news", "fundamentals", "sentiment"):
        value = reports.get(key)
        if isinstance(value, str) and value.strip():
            score += 0.10
    return min(0.90, score)


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def _post_scorecard(payload: dict[str, object]) -> str:
    response = httpx.post(
        f"{ORCHESTRATOR_URL}/scorecards",
        json=payload,
        timeout=10.0,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "scorecard_id" not in body:
        raise ValueError(f"Orchestrator did not return a scorecard_id: {body}")
    return str(body["scorecard_id"])


def _mark_status(job_id: str, status: JobStatus) -> None:
    with connect() as conn:
        conn.execute("update analysis_jobs set status = ? where job_id = ?", (status, job_id))
        conn.commit()


def _mark_success(job_id: str, scorecard_id: str, raw: dict[str, object]) -> None:
    with connect() as conn:
        conn.execute(
            "update analysis_jobs set status = 'succeeded', scorecard_id = ?, "
            "raw_response_json = ?, finished_at = ? where job_id = ?",
            (scorecard_id, json.dumps(raw, default=str), _now().isoformat(), job_id),
        )
        conn.commit()


def _mark_failure(job_id: str, error: str, raw: dict[str, object] | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "update analysis_jobs set status = 'failed', error = ?, "
            "raw_response_json = ?, finished_at = ? where job_id = ?",
            (
                error,
                json.dumps(raw, default=str) if raw is not None else None,
                _now().isoformat(),
                job_id,
            ),
        )
        conn.commit()
