"""ECMWF Open Data retrieval (free, no account required)."""
from __future__ import annotations

import logging
import os

import earthkit.data as ekd
from multiurl import download

from .base import DataSource

LOG = logging.getLogger(__name__)
_CONSTANTS_URL   = "https://get.ecmwf.int/repository/test-data/ai-models/opendata/constants-0p25.grib2"
_CONSTANTS_CACHE = os.path.expanduser("~/.cache/graphcast-africa/constants-0p25.grib2")

def _ensure_constants():
    os.makedirs(os.path.dirname(_CONSTANTS_CACHE), exist_ok=True)
    if not os.path.exists(_CONSTANTS_CACHE):
        LOG.info("Downloading constants to %s", _CONSTANTS_CACHE)
        download(_CONSTANTS_URL, _CONSTANTS_CACHE + ".tmp")
        os.rename(_CONSTANTS_CACHE + ".tmp", _CONSTANTS_CACHE)
    return _CONSTANTS_CACHE

def _gh_to_z(ds):
    G = 9.80665
    from earthkit.data.indexing.fieldlist import FieldArray
    out = []
    for f in ds:
        if f.metadata("shortName") == "gh":
            out.append(f.copy(values=f.to_numpy() * G, shortName="z"))
        else:
            out.append(f)
    return FieldArray(out)

class OpenDataSource(DataSource):
    area = None  # open data does not support area subsetting

    def _load_sfc(self, date, time):
        LOG.info("OPENDATA  SFC  %s %04d", date, time)
        _const = {"lsm", "z"}
        stream = ekd.from_source("ecmwf-open-data", date=date, time=time, step=0,
                                 param=[p for p in self.param_sfc if p not in _const],
                                 levtype="sfc", resol="0p25")
        constants = ekd.from_source("file", _ensure_constants()).sel(param=list(_const))
        return stream + constants

    def _load_pl(self, date, time):
        params, levels = self.param_level_pl
        LOG.info("OPENDATA  PL   %s %04d", date, time)
        ds = ekd.from_source("ecmwf-open-data", date=date, time=time, step=0,
                              param=["gh" if p == "z" else p for p in params],
                              levtype="pl", levelist=levels, resol="0p25")
        return _gh_to_z(ds)
