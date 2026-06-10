from __future__ import annotations

from typing import Any

from fastapi import FastAPI


def application_route_snapshot(app: FastAPI) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = sorted(getattr(route, "methods", []) or [])
        if path in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
            continue
        response_model = getattr(route, "response_model", None)
        snapshot.append(
            {
                "methods": methods,
                "path": path,
                "name": getattr(route, "name", ""),
                "response_model": getattr(response_model, "__name__", None)
                if response_model is not None
                else None,
            }
        )
    return snapshot

