from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route(
    "/notifications/subscribe", core.subscribe_notifications, methods=["POST"], response_model=None
)
router.add_api_route(
    "/notifications/unsubscribe",
    core.unsubscribe_notifications,
    methods=["POST"],
    response_model=None,
)
router.add_api_route(
    "/notifications/deliveries",
    core.list_notification_deliveries,
    methods=["GET"],
    response_model=None,
)
