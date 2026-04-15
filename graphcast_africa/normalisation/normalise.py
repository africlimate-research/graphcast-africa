"""Z-score normalisation and denormalisation helpers."""
from __future__ import annotations

import logging

import xarray as xr

from .stats import load_stats

LOG = logging.getLogger(__name__)

def normalise(ds: xr.Dataset, assets_dir: str) -> xr.Dataset:
    mean, stddev, _ = load_stats(assets_dir)
    out = ds.copy()
    for var in ds.data_vars:
        if var in mean and var in stddev:
            std = stddev[var].where(stddev[var] > 0, other=1.0)
            out[var] = (ds[var] - mean[var]) / std
    return out

def denormalise(ds: xr.Dataset, assets_dir: str) -> xr.Dataset:
    mean, stddev, _ = load_stats(assets_dir)
    out = ds.copy()
    for var in ds.data_vars:
        if var in mean and var in stddev:
            std = stddev[var].where(stddev[var] > 0, other=1.0)
            out[var] = ds[var] * std + mean[var]
    return out

def validate_stats_coverage(ds: xr.Dataset, assets_dir: str) -> None:
    mean, stddev, _ = load_stats(assets_dir)
    for var in ds.data_vars:
        if var not in mean:
            LOG.warning("Variable '%s' has no mean stats — will not be normalised", var)
        if var not in stddev:
            LOG.warning("Variable '%s' has no stddev stats — will not be normalised", var)
