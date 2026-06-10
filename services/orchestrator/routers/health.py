from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route("/healthz", core.healthz, methods=["GET"])
router.add_api_route("/readyz", core.readyz, methods=["GET"])
