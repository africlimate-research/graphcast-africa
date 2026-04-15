"""ERA5 retrieval via CDS. Requires ~/.cdsapirc or CDSAPI_* env vars."""
from __future__ import annotations

import hashlib
import json
import logging
import os

import earthkit.data as ekd

from .base import DataSource

LOG = logging.getLogger(__name__)

_CACHE_DIR = os.path.expanduser("~/.cache/graphcast-africa/era5")


def _cache_path(request: dict) -> str:
    """Derive a deterministic cache path from the request dict."""
    key = hashlib.md5(json.dumps(request, sort_keys=True).encode()).hexdigest()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{key}.grib2")


def _cached_cds(dataset: str, request: dict) -> ekd.FieldList:
    path = _cache_path({"dataset": dataset, **request})
    if os.path.exists(path):
        LOG.info("CDS cache hit  -> %s", path)
        return ekd.from_source("file", path)
    LOG.info("CDS cache miss -> downloading to %s", path)
    tmp = path + ".tmp"
    ekd.from_source("cds", dataset, request).save(tmp)
    os.rename(tmp, path)
    return ekd.from_source("file", path)


class CDSSource(DataSource):
    def _load_sfc(self, date, time):
        LOG.info("CDS  SFC  %s %04d", date, time)
        return _cached_cds("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "param": self.param_sfc,
            "date": date, "time": f"{time:04d}", "grid": self.grid,
            "area": self.area, "format": "grib",
        })

    def _load_pl(self, date, time):
        params, levels = self.param_level_pl
        LOG.info("CDS  PL   %s %04d", date, time)
        return _cached_cds("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "param": params,
            "pressure_level": levels, "date": date, "time": f"{time:04d}",
            "grid": self.grid, "area": self.area, "format": "grib",
        })
