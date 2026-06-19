from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_PRIVATE_ROOT_ENV = "AEGIS_STRATEGIES_ROOT"


class PrivatePathError(ValueError):
    """Raised when a private evidence path would violate the public/private boundary."""


def private_dir_from_cli(
    value: str | Path | None,
    *,
    default_task: str,
    env: Mapping[str, str] | None = None,
    env_var: str = DEFAULT_PRIVATE_ROOT_ENV,
    repo_root: Path | None = None,
) -> Path:
    root = private_root_from_env(env=env, env_var=env_var, repo_root=repo_root)
    path = root / "incubating" / default_task if value is None else Path(value)
    return resolve_private_dir(path, private_root=root, repo_root=repo_root)


def private_path_from_cli(
    value: str | Path | None,
    *,
    default_task: str,
    default_name: str,
    env: Mapping[str, str] | None = None,
    env_var: str = DEFAULT_PRIVATE_ROOT_ENV,
    repo_root: Path | None = None,
) -> Path:
    root = private_root_from_env(env=env, env_var=env_var, repo_root=repo_root)
    path = root / "incubating" / default_task / default_name if value is None else Path(value)
    resolved = _resolve(path)
    _validate_private_path(resolved, private_root=root, repo_root=repo_root)
    return resolved


def private_root_from_env(
    *,
    env: Mapping[str, str] | None = None,
    env_var: str = DEFAULT_PRIVATE_ROOT_ENV,
    repo_root: Path | None = None,
) -> Path:
    environ = os.environ if env is None else env
    raw_root = environ.get(env_var, "").strip()
    if not raw_root:
        raise PrivatePathError(
            f"{env_var} is required. Set it to the private aegis-strategies repository root, "
            "then write evidence under ${AEGIS_STRATEGIES_ROOT}/incubating/<task>/."
        )
    root = _resolve(Path(raw_root))
    if not root.exists() or not root.is_dir():
        raise PrivatePathError(
            f"{env_var} must point to an existing private aegis-strategies directory: {root}"
        )
    public_root = _repo_root(repo_root)
    if _is_relative_to(root, public_root):
        raise PrivatePathError(
            f"{env_var} points inside the public repository; use a separate private repo root."
        )
    return root


def resolve_private_dir(
    value: str | Path,
    *,
    private_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    env_var: str = DEFAULT_PRIVATE_ROOT_ENV,
    repo_root: Path | None = None,
) -> Path:
    root = (
        private_root
        if private_root is not None
        else private_root_from_env(env=env, env_var=env_var, repo_root=repo_root)
    )
    path = _resolve(Path(value))
    _validate_private_path(path, private_root=root, repo_root=repo_root)
    return path


def _validate_private_path(path: Path, *, private_root: Path, repo_root: Path | None) -> None:
    public_root = _repo_root(repo_root)
    if _is_relative_to(path, public_root):
        raise PrivatePathError("private evidence path points inside the public repository")
    if not _is_relative_to(path, private_root):
        raise PrivatePathError(
            "private evidence path must be under "
            "${AEGIS_STRATEGIES_ROOT}/incubating/<task>/, not a temp or scatter directory"
        )
    relative = path.relative_to(private_root)
    if not relative.parts or relative.parts[0] != "incubating" or len(relative.parts) < 2:
        raise PrivatePathError(
            "private evidence path must be under "
            "${AEGIS_STRATEGIES_ROOT}/incubating/<task>/"
        )


def _repo_root(repo_root: Path | None) -> Path:
    return _resolve(repo_root or Path(__file__).resolve().parents[2])


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
