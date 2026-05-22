"""Load YAML config files into validated pydantic models."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .schema import CampaignConfig, GlobalConfig

# repo root = three parents up from this file (src/matscout/config/loader.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def load_global_config(path: str | Path | None = None) -> GlobalConfig:
    _load_dotenv(REPO_ROOT / ".env")
    path = Path(path) if path else CONFIG_DIR / "global.yaml"
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = GlobalConfig(**data)
    # env overrides
    if os.environ.get("MP_API_KEY"):
        cfg.mp_api_key = os.environ["MP_API_KEY"]
    if os.environ.get("MATSCOUT_DEVICE"):
        cfg.device = os.environ["MATSCOUT_DEVICE"]
    if os.environ.get("MATSCOUT_CACHE_DIR"):
        cfg.cache_dir = os.environ["MATSCOUT_CACHE_DIR"]
    return cfg


def load_campaign_config(name: str, path: str | Path | None = None) -> CampaignConfig:
    path = Path(path) if path else CONFIG_DIR / "campaigns" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Campaign config not found: {path}. Run `matscout init` or check the name."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", name)
    return CampaignConfig(**data)


def available_campaigns() -> list[str]:
    cdir = CONFIG_DIR / "campaigns"
    if not cdir.exists():
        return []
    return sorted(p.stem for p in cdir.glob("*.yaml"))
