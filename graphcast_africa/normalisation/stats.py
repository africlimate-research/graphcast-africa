"""Load and cache GraphCast normalisation statistics from assets/stats/."""
from __future__ import annotations

import os
from functools import lru_cache

import xarray as xr


@lru_cache(maxsize=4)
def load_stats(assets_dir: str):
    def _load(fname):
        path = os.path.join(assets_dir, "stats", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Stats file not found: {path}\n"
                "Run  python scripts/download_assets.py  first."
            )
        return xr.load_dataset(path).compute()
    return _load("mean_by_level.nc"), _load("stddev_by_level.nc"), _load("diffs_stddev_by_level.nc")
