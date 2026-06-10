from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()
outcome_detail_router = APIRouter()

router.add_api_route("/scorecards", core.create_scorecard, methods=["POST"], response_model=None)
router.add_api_route(
    "/scorecards/{scorecard_id}", core.get_scorecard, methods=["GET"], response_model=None
)
router.add_api_route("/scorecards", core.list_scorecards, methods=["GET"], response_model=None)
router.add_api_route(
    "/scorecard-outcomes", core.list_outcomes, methods=["GET"], response_model=None
)
router.add_api_route(
    "/scorecard-outcomes/summary", core.outcomes_summary, methods=["GET"], response_model=None
)
outcome_detail_router.add_api_route(
    "/scorecard-outcomes/{outcome_id}", core.get_outcome, methods=["GET"], response_model=None
)
outcome_detail_router.add_api_route(
    "/scorecard-outcomes/{outcome_id}/trailing",
    core.update_outcome_trailing,
    methods=["POST"],
    response_model=None,
)
