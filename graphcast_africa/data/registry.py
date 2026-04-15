"""Factory for data sources."""
from __future__ import annotations

from .cds import CDSSource
from .grib_file import GRIBFileSource
from .opendata import OpenDataSource

_SOURCES = {"cds": CDSSource, "opendata": OpenDataSource, "file": GRIBFileSource}

def get_source(name: str, **kwargs):
    if name not in _SOURCES:
        raise ValueError(f"Unknown source '{name}'. Choose from: {list(_SOURCES)}")
    return _SOURCES[name](**kwargs)
