"""Reproducible, path-free environment and model provenance."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing its runtime."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def git_revision(repository: Path) -> str | None:
    """Return the current commit, or ``None`` for an uncommitted repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(repository: Path) -> bool:
    """Report whether tracked or untracked repository content differs from HEAD."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def resolve_model_revision(model_id: str, revision: str | None = None) -> str:
    """Resolve a Hugging Face model reference to an immutable commit SHA."""
    info = HfApi().model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {model_id!r}")
    return info.sha


def capture_provenance(
    repository: Path,
    *,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    """Capture deterministic run metadata without absolute paths or usernames."""
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "project_git_commit": git_revision(repository),
        "project_git_dirty": git_is_dirty(repository),
        "platform": platform.platform(),
        "machine_architecture": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": package_version("mlx"),
        "mlx_lm_version": package_version("mlx-lm"),
        "transformers_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "gliner_version": package_version("gliner"),
        "model_id": model_id,
        "model_revision_sha": model_revision,
        "python_implementation": platform.python_implementation(),
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
