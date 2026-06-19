"""Reproducibility helpers: seeding, git hash, run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nlvswe.config import AppConfig


def set_seeds(seed: int) -> None:
    """Seed python, numpy, and pymc/pytensor RNGs for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import pymc as pm

        if hasattr(pm, "set_rng"):
            pm.set_rng(seed)
    except ImportError:
        pass
    try:
        from pytensor.tensor.random import default_rng

        default_rng(seed)
    except ImportError:
        pass


def git_commit() -> str:
    """Return short git commit hash, or 'nogit' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "nogit"


def run_manifest(
    plan: str,
    scope: str,
    *,
    row_count: int | None = None,
    schema_name: str | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Build a run manifest dict for artifact sidecars."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_plan": plan,
        "scope": scope,
    }
    if config_hash is not None:
        manifest["config_hash"] = config_hash
    if row_count is not None:
        manifest["row_count"] = row_count
    if schema_name is not None:
        manifest["schema_name"] = schema_name
    return manifest


def stable_json_dumps(obj: Any) -> str:
    """Serialize JSON with stable key ordering for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
