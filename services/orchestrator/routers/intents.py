from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()
pre_positions_router = APIRouter()
post_positions_router = APIRouter()

router.add_api_route(
    "/intents/from_nl", core.create_intent_from_nl, methods=["POST"], response_model=None
)
router.add_api_route(
    "/intents/from_scorecard",
    core.create_intent_from_scorecard,
    methods=["POST"],
    response_model=None,
)
router.add_api_route("/intents", core.create_intent, methods=["POST"], response_model=None)
router.add_api_route("/intents", core.list_intents, methods=["GET"])
pre_positions_router.add_api_route(
    "/intents/{intent_id}/confirm", core.confirm_intent, methods=["POST"], response_model=None
)
pre_positions_router.add_api_route(
    "/intents/{intent_id}", core.cancel_intent, methods=["DELETE"], response_model=None
)
post_positions_router.add_api_route(
    "/intents/{intent_id}/refresh", core.refresh_intent, methods=["POST"], response_model=None
)
post_positions_router.add_api_route(
    "/intents/{intent_id}", core.get_intent, methods=["GET"], response_model=None
)
