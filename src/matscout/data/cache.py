"""Tiny disk cache for external API calls (keyed by a hash of the request)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, namespace: str, params: dict[str, Any]) -> Path:
        raw = namespace + json.dumps(params, sort_keys=True, default=str)
        h = hashlib.sha1(raw.encode()).hexdigest()[:20]
        return self.dir / f"{namespace}_{h}.json"

    def get(self, namespace: str, params: dict[str, Any]) -> Any | None:
        p = self._key(namespace, params)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def set(self, namespace: str, params: dict[str, Any], value: Any) -> None:
        p = self._key(namespace, params)
        p.write_text(json.dumps(value, default=str), encoding="utf-8")
