"""Run provenance: write a manifest of exactly how a run was produced."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from .. import __version__
from ..config.schema import GlobalConfig
from ..relaxation.base import resolve_device
from .paths import Paths

_TRACKED_PACKAGES = ["pymatgen", "ase", "numpy", "pandas", "scikit-learn",
                     "torch", "chgnet", "mace-torch", "matminer", "duckdb"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _pkg_versions() -> dict[str, str]:
    out = {}
    for p in _TRACKED_PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not-installed"
    return out


def write_run_manifest(g: GlobalConfig, campaign_names: list[str], extra: dict | None = None) -> str:
    """Write outputs/run_manifest.json capturing code/version/config/run params."""
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "matscout_version": __version__,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "device": resolve_device(g.device),
        "seed": g.seed,
        "mp_api_key_present": bool(g.mp_api_key),
        "campaigns": campaign_names,
        "relaxation": g.relaxation.model_dump(),
        "packages": _pkg_versions(),
    }
    if extra:
        manifest.update(extra)
    paths = Paths(g)
    paths.out.mkdir(parents=True, exist_ok=True)
    out = paths.out / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(out)
