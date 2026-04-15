"""Phase 1 tests — data retrieval layer."""
import pytest


def test_registry_correct_sources():
    from graphcast_africa.data.cds import CDSSource
    from graphcast_africa.data.opendata import OpenDataSource
    from graphcast_africa.data.registry import get_source
    assert isinstance(get_source("cds"), CDSSource)
    assert isinstance(get_source("opendata"), OpenDataSource)

def test_registry_unknown_raises():
    from graphcast_africa.data.registry import get_source
    with pytest.raises(ValueError, match="Unknown source"):
        get_source("mars")

def test_lagged_datetimes():
    from graphcast_africa.data.base import DataSource
    r = DataSource._lagged_datetimes("20240101", "1200")
    assert r[0] == ("20240101", 600)
    assert r[1] == ("20240101", 1200)

def test_lagged_datetimes_crosses_midnight():
    from graphcast_africa.data.base import DataSource
    r = DataSource._lagged_datetimes("20240101", "0000")
    assert r[0] == ("20231231", 1800)
    assert r[1] == ("20240101", 0)

def test_file_source_bad_path_raises():
    from graphcast_africa.data.grib_file import GRIBFileSource
    with pytest.raises(Exception):
        GRIBFileSource(path="/nonexistent/file.grib")
