"""ERA5 retrieval via CDS. Requires ~/.cdsapirc or CDSAPI_* env vars."""
from __future__ import annotations

import logging

import earthkit.data as ekd

from .base import DataSource

LOG = logging.getLogger(__name__)

class CDSSource(DataSource):
    def _load_sfc(self, date, time):
        LOG.info("CDS  SFC  %s %04d", date, time)
        return ekd.from_source("cds", "reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "param": self.param_sfc,
            "date": date, "time": f"{time:04d}", "grid": self.grid,
            "area": self.area, "format": "grib",
        })

    def _load_pl(self, date, time):
        params, levels = self.param_level_pl
        LOG.info("CDS  PL   %s %04d", date, time)
        return ekd.from_source("cds", "reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "param": params,
            "pressure_level": levels, "date": date, "time": f"{time:04d}",
            "grid": self.grid, "area": self.area, "format": "grib",
        })
