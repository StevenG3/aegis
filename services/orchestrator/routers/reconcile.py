from __future__ import annotations

from fastapi import APIRouter

try:
    from .. import core
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]

router = APIRouter()

router.add_api_route(
    "/reconcile/ibkr/apply",
    core.post_reconcile_apply,
    methods=["POST"],
    response_model=core.ReconcileApplyResponse,
)
router.add_api_route(
    "/reconcile/ibkr", core.trigger_ibkr_reconcile, methods=["POST"], response_model=None
)
router.add_api_route(
    "/reconcile/ibkr/latest", core.get_latest_ibkr_reconcile, methods=["GET"], response_model=None
)
