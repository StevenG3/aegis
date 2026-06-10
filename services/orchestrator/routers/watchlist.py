from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route("/watchlist", core.add_watchlist, methods=["POST"], response_model=None)
router.add_api_route("/watchlist", core.list_watchlist, methods=["GET"], response_model=None)
router.add_api_route(
    "/watchlist/{symbol}", core.delete_watchlist, methods=["DELETE"], response_model=None
)
