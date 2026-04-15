"""ECMWF Open Data retrieval (free, no account required)."""
from __future__ import annotations

import logging
import os

import earthkit.data as ekd
import numpy as np
from multiurl import download

from .base import DataSource

LOG = logging.getLogger(__name__)
_CONSTANTS_URL   = "https://get.ecmwf.int/repository/test-data/ai-models/opendata/constants-0p25.grib2"
_CONSTANTS_CACHE = os.path.expanduser("assets/graphcast-africa/constants-0p25.grib2")

def _ensure_constants():
    os.makedirs(os.path.dirname(_CONSTANTS_CACHE), exist_ok=True)
    if not os.path.exists(_CONSTANTS_CACHE):
        LOG.info("Downloading constants to %s", _CONSTANTS_CACHE)
        download(_CONSTANTS_URL, _CONSTANTS_CACHE + ".tmp")
        os.rename(_CONSTANTS_CACHE + ".tmp", _CONSTANTS_CACHE)
    return _CONSTANTS_CACHE

def _gh_to_z(ds):
    """Convert geopotential height (gh, m) fields to geopotential (z, m2/s2).

    earthkit-data's Field.copy() signature varies across versions; the safest
    portable approach is to write converted fields to a temporary GRIB2 buffer
    using eccodes directly, then reload via earthkit.
    """
    import tempfile

    import eccodes

    G = 9.80665
    out_fields = []
    tmp_paths = []

    for f in ds:
        if f.metadata("shortName") == "gh":
            # Get the raw eccodes message handle and clone it
            msg = f.message()  # bytes of the original GRIB message
            handle = eccodes.codes_new_from_message(msg)
            try:
                # Convert values: geopotential = height * g
                size = eccodes.codes_get_size(handle, "values")
                vals = np.array(eccodes.codes_get_array(handle, "values"), dtype=np.float64)
                vals = vals * G
                eccodes.codes_set_values(handle, vals)
                # Update shortName to z (geopotential)
                eccodes.codes_set(handle, "shortName", "z")
                # Write to a temp file and reload
                tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
                eccodes.codes_write(handle, tmp)
                tmp.flush()
                tmp.close()
                tmp_paths.append(tmp.name)
                out_fields.append(ekd.from_source("file", tmp.name)[0])
            finally:
                eccodes.codes_release(handle)
        else:
            out_fields.append(f)

    from earthkit.data.indexing.fieldlist import FieldArray
    result = FieldArray(out_fields)

    # Clean up temp files after the FieldArray holds references
    import atexit
    for p in tmp_paths:
        atexit.register(os.unlink, p)

    return result

class OpenDataSource(DataSource):
    area = None  # open data does not support area subsetting

    def _load_sfc(self, date, time):
        LOG.info("OPENDATA  SFC  %s %04d", date, time)
        _const = {"lsm", "z"}
        stream = ekd.from_source("ecmwf-open-data", date=date, time=time, step=0,
                                 param=[p for p in self.param_sfc if p not in _const],
                                 levtype="sfc", resol="0p25")
        constants = ekd.from_source("file", _ensure_constants())
        LOG.warning("Constants params found: %s", [f.metadata("shortName") for f in constants])
        return stream + constants

    def _load_pl(self, date, time):
        params, levels = self.param_level_pl
        LOG.info("OPENDATA  PL   %s %04d", date, time)
        ds = ekd.from_source("ecmwf-open-data", date=date, time=time, step=0,
                              param=["gh" if p == "z" else p for p in params],
                              levtype="pl", levelist=levels, resol="0p25")
        return _gh_to_z(ds)
