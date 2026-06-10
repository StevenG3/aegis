from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route(
    "/calibration/recompute", core.recompute_calibration, methods=["POST"], response_model=None
)
router.add_api_route("/calibration", core.get_calibration, methods=["GET"], response_model=None)
