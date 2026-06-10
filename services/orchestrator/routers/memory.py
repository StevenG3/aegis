from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route(
    "/memory/record", core.record_memory_endpoint, methods=["POST"], response_model=None
)
router.add_api_route("/memory/recall", core.recall_memory, methods=["GET"], response_model=None)
router.add_api_route(
    "/memory/lineage/{memory_id}", core.memory_lineage, methods=["GET"], response_model=None
)
