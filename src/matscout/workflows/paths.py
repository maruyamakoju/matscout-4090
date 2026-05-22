"""Centralized output / interim path layout."""

from __future__ import annotations

from pathlib import Path

from ..config.schema import GlobalConfig


class Paths:
    def __init__(self, global_cfg: GlobalConfig):
        self.data = Path(global_cfg.data_dir)
        self.out = Path(global_cfg.output_dir)

    def interim(self, campaign: str) -> Path:
        p = self.data / "interim" / campaign
        p.mkdir(parents=True, exist_ok=True)
        return p

    def seeds(self, campaign: str) -> Path:
        return self.interim(campaign) / "seeds.parquet"

    def generated(self, campaign: str) -> Path:
        return self.interim(campaign) / "generated.parquet"

    def relaxed(self, campaign: str) -> Path:
        return self.interim(campaign) / "relaxed.parquet"

    def scored(self, campaign: str) -> Path:
        return self.interim(campaign) / "scored.parquet"

    def ranked(self, campaign: str) -> Path:
        p = self.data / "processed" / campaign
        p.mkdir(parents=True, exist_ok=True)
        return p / "ranked.parquet"

    # ---- combined outputs ----
    def candidates_all(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out / "candidates_all.parquet"

    def candidates_ranked(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out / "candidates_ranked.parquet"

    def top_200(self) -> Path:
        p = self.out / "top_200"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dft_queue(self, campaign: str) -> Path:
        p = self.out / "dft_queue" / campaign
        p.mkdir(parents=True, exist_ok=True)
        return p

    def reports(self) -> Path:
        p = self.out / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figures(self) -> Path:
        p = self.out / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p
