from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

positions_router = APIRouter()
router = APIRouter()

router.add_api_route(
    "/paper/autonomy/observability",
    core.get_paper_autonomy_observability,
    methods=["GET"],
    response_model=None,
)
router.add_api_route(
    "/paper/autonomy/alerts", core.post_paper_autonomy_alerts, methods=["POST"], response_model=None
)
router.add_api_route(
    "/paper/autonomy/digest", core.post_paper_autonomy_digest, methods=["POST"], response_model=None
)
router.add_api_route(
    "/paper/autonomy/readiness",
    core.get_paper_autonomy_readiness,
    methods=["GET"],
    response_model=None,
)
router.add_api_route(
    "/paper/bootstrap/status", core.get_paper_bootstrap_status, methods=["GET"], response_model=None
)
router.add_api_route(
    "/paper/bootstrap/evaluate-guardrails",
    core.post_paper_bootstrap_evaluate_guardrails,
    methods=["POST"],
    response_model=None,
)
router.add_api_route(
    "/paper/bootstrap/resume",
    core.post_paper_bootstrap_resume,
    methods=["POST"],
    response_model=None,
)
positions_router.add_api_route(
    "/paper/positions", core.get_paper_positions, methods=["GET"], response_model=None
)
