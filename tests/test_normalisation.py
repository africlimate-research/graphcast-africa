"""Phase 2 tests — normalisation layer."""
import os

import pytest
import xarray as xr


def _fake_stats(tmpdir):
    d = os.path.join(str(tmpdir), "stats")
    os.makedirs(d)
    for fname, val in [("mean_by_level.nc", 0.0), ("stddev_by_level.nc", 1.0), ("diffs_stddev_by_level.nc", 0.5)]:
        xr.Dataset({"temperature": xr.DataArray([val], dims=["level"])}).to_netcdf(os.path.join(d, fname))
    return str(tmpdir)

def test_normalise(tmp_path):
    assets = _fake_stats(tmp_path)
    from graphcast_africa.normalisation.stats import load_stats; load_stats.cache_clear()
    from graphcast_africa.normalisation.normalise import normalise
    ds = xr.Dataset({"temperature": xr.DataArray([5.0], dims=["level"])})
    assert float(normalise(ds, assets)["temperature"].values[0]) == pytest.approx(5.0)

def test_roundtrip(tmp_path):
    assets = _fake_stats(tmp_path)
    from graphcast_africa.normalisation.stats import load_stats; load_stats.cache_clear()
    from graphcast_africa.normalisation.normalise import denormalise, normalise
    ds = xr.Dataset({"temperature": xr.DataArray([273.15], dims=["level"])})
    assert float(denormalise(normalise(ds, assets), assets)["temperature"].values[0]) == pytest.approx(273.15, rel=1e-5)

def test_missing_stats_raises(tmp_path):
    from graphcast_africa.normalisation.stats import load_stats; load_stats.cache_clear()
    with pytest.raises(FileNotFoundError, match="Stats file not found"):
        load_stats(str(tmp_path))
