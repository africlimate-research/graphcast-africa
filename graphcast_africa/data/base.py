"""Abstract base class for all data sources."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import earthkit.data as ekd

from graphcast_africa.fields.graphcast_fields import (AREA, GRID,
                                                      PARAM_LEVEL_PL,
                                                      PARAM_SFC)

LOG = logging.getLogger(__name__)

class DataSource(ABC):
    param_sfc      = PARAM_SFC
    param_level_pl = PARAM_LEVEL_PL
    grid           = GRID
    area           = AREA

    def retrieve(self, date: str, time: str) -> tuple:
        datetimes = self._lagged_datetimes(date, time)
        LOG.info("Retrieving fields for %s", datetimes)
        fields_sfc = ekd.from_source("multi", [self._load_sfc(d, t) for d, t in datetimes])
        fields_pl  = ekd.from_source("multi", [self._load_pl(d, t)  for d, t in datetimes])
        return fields_sfc, fields_pl

    @staticmethod
    def _lagged_datetimes(date: str, time: str) -> list[tuple[str, int]]:
        from datetime import datetime, timedelta
        dt   = datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")
        lag6 = dt - timedelta(hours=6)
        return [
            (lag6.strftime("%Y%m%d"), int(lag6.strftime("%H%M"))),
            (dt.strftime("%Y%m%d"),   int(dt.strftime("%H%M"))),
        ]

    @abstractmethod
    def _load_sfc(self, date: str, time: int) -> ekd.FieldList: ...
    @abstractmethod
    def _load_pl(self, date: str, time: int) -> ekd.FieldList: ...
