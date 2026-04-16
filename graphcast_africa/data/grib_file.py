"""Load input fields from a local GRIB file."""
from __future__ import annotations

import logging

import earthkit.data as ekd

from .base import DataSource

LOG = logging.getLogger(__name__)

class GRIBFileSource(DataSource):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self._ds  = ekd.from_source("file", path)

    def _load_sfc(self, date, time):
        return self._ds.sel(param=self.param_sfc, date=date, time=time, levtype="sfc")

    def _load_pl(self, date, time):
        params, levels = self.param_level_pl
        return self._ds.sel(param=params, level=levels, date=date, time=time, levtype="pl")
