from __future__ import annotations

import sys
import types
from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import Response

try:
    from . import core
    from .routers import (
        calibration_router,
        ev_shadow_router,
        exposure_router,
        factor_attribution_router,
        health_router,
        intent_post_positions_router,
        intent_pre_positions_router,
        intents_router,
        memory_router,
        notifications_router,
        paper_positions_router,
        paper_router,
        pnl_router,
        reconcile_router,
        reflect_router,
        safety_autonomy_router,
        safety_live_unlock_router,
        scorecard_outcome_detail_router,
        scorecards_router,
        watchlist_router,
    )
except ImportError:  # pragma: no cover - legacy direct file loading path
    import core  # type: ignore[import-not-found,no-redef]
    from routers import (  # type: ignore[import-not-found,no-redef]
        calibration_router,
        ev_shadow_router,
        exposure_router,
        factor_attribution_router,
        health_router,
        intent_post_positions_router,
        intent_pre_positions_router,
        intents_router,
        memory_router,
        notifications_router,
        paper_positions_router,
        paper_router,
        pnl_router,
        reconcile_router,
        reflect_router,
        safety_autonomy_router,
        safety_live_unlock_router,
        scorecard_outcome_detail_router,
        scorecards_router,
        watchlist_router,
    )

app = FastAPI(title="orchestrator", version="0.1.0", lifespan=core._lifespan)
app.add_exception_handler(
    RequestValidationError,
    cast(
        Callable[[Request, Exception], Response | Awaitable[Response]],
        core.validation_exception_handler,
    ),
)

# Include routers in the original route order. Safety route extraction happened last,
# but final registration order stays unchanged to preserve the route snapshot exactly.
for router in (
    health_router,
    notifications_router,
    ev_shadow_router,
    calibration_router,
    memory_router,
    safety_autonomy_router,
    paper_router,
    watchlist_router,
    safety_live_unlock_router,
    scorecards_router,
    factor_attribution_router,
    reflect_router,
    scorecard_outcome_detail_router,
    pnl_router,
    intents_router,
    exposure_router,
    intent_pre_positions_router,
    paper_positions_router,
    intent_post_positions_router,
    reconcile_router,
):
    app.include_router(router)


class _AppModule(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        return getattr(core, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"app", "core"} or name.startswith("__"):
            super().__setattr__(name, value)
            return
        if hasattr(core, name):
            setattr(core, name, value)
            return
        super().__setattr__(name, value)


_module = sys.modules.get(__name__)
if _module is not None:
    _module.__class__ = _AppModule
