from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

autonomy_router = APIRouter()
live_unlock_router = APIRouter()

autonomy_router.add_api_route(
    "/autonomy/settings", core.update_autonomy, methods=["POST"], response_model=None
)
autonomy_router.add_api_route(
    "/autonomy/settings", core.get_autonomy_settings, methods=["GET"], response_model=None
)
autonomy_router.add_api_route(
    "/autonomy/today", core.get_autonomy_today, methods=["GET"], response_model=None
)
autonomy_router.add_api_route(
    "/admin/live-autonomy/disable",
    core.disable_live_autonomy,
    methods=["POST"],
    response_model=None,
)
autonomy_router.add_api_route(
    "/admin/live-autonomy/enable",
    core.reenable_live_autonomy,
    methods=["POST"],
    response_model=None,
)
autonomy_router.add_api_route(
    "/live-autonomy/settings", core.update_live_autonomy, methods=["POST"], response_model=None
)
autonomy_router.add_api_route(
    "/live-autonomy/settings", core.get_live_autonomy_settings, methods=["GET"], response_model=None
)
autonomy_router.add_api_route(
    "/live-autonomy/today", core.get_live_autonomy_today, methods=["GET"], response_model=None
)
live_unlock_router.add_api_route(
    "/admin/live-unlock", core.issue_live_unlock, methods=["POST"], response_model=None
)
