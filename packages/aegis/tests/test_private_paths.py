from __future__ import annotations

from pathlib import Path

import pytest

from aegis.private_paths import (
    PrivatePathError,
    private_dir_from_cli,
    private_path_from_cli,
    resolve_private_dir,
)


def test_private_root_env_is_required(tmp_path: Path) -> None:
    with pytest.raises(PrivatePathError, match="AEGIS_STRATEGIES_ROOT is required"):
        private_dir_from_cli(
            None,
            default_task="olympus52",
            env={},
            repo_root=tmp_path / "public",
        )


def test_private_dir_must_stay_outside_public_repo(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private_root = tmp_path / "private"
    public.mkdir()
    private_root.mkdir()

    with pytest.raises(PrivatePathError, match="inside the public repository"):
        resolve_private_dir(
            public / "incubating" / "olympus52",
            env={"AEGIS_STRATEGIES_ROOT": str(private_root)},
            repo_root=public,
        )


def test_private_dir_must_be_under_private_incubating_root(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    scatter = tmp_path / "aegis-strategies-scatter" / "incubating" / "olympus52"
    public.mkdir()
    private_root.mkdir()

    with pytest.raises(PrivatePathError, match="temp or scatter"):
        resolve_private_dir(
            scatter,
            env={"AEGIS_STRATEGIES_ROOT": str(private_root)},
            repo_root=public,
        )

    with pytest.raises(PrivatePathError, match="incubating"):
        resolve_private_dir(
            private_root / "olympus52",
            env={"AEGIS_STRATEGIES_ROOT": str(private_root)},
            repo_root=public,
        )


def test_private_dir_defaults_to_task_under_incubating(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    public.mkdir()
    private_root.mkdir()

    assert private_dir_from_cli(
        None,
        default_task="olympus52",
        env={"AEGIS_STRATEGIES_ROOT": str(private_root)},
        repo_root=public,
    ) == private_root / "incubating" / "olympus52"


def test_private_file_path_uses_same_guard(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private_root = tmp_path / "aegis-strategies"
    public.mkdir()
    private_root.mkdir()

    assert private_path_from_cli(
        None,
        default_task="olympus37",
        default_name="matrix.json",
        env={"AEGIS_STRATEGIES_ROOT": str(private_root)},
        repo_root=public,
    ) == private_root / "incubating" / "olympus37" / "matrix.json"
