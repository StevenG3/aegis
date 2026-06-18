from __future__ import annotations

from typing import Any

from fastapi import FastAPI


def application_route_snapshot(app: FastAPI) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for route in _expanded_routes(app.routes):
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


def _expanded_routes(routes: list[Any], prefix: str = "") -> list[Any]:
    expanded: list[Any] = []
    for route in routes:
        original_router = getattr(route, "original_router", None)
        include_context = getattr(route, "include_context", None)
        if original_router is not None and include_context is not None:
            child_prefix = f"{prefix}{getattr(include_context, 'prefix', '')}"
            expanded.extend(
                _expanded_routes(list(getattr(original_router, "routes", [])), child_prefix)
            )
            continue
        if prefix and hasattr(route, "path"):
            route.path = f"{prefix}{route.path}"
        expanded.append(route)
    return expanded
