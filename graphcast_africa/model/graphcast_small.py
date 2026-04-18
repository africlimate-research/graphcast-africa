"""GraphCast small variant — 1° resolution, lighter mesh."""
from __future__ import annotations

from graphcast_africa.fields.graphcast_fields import ASSET_FILES_SMALL
from graphcast_africa.model.graphcast_oper import GraphCastOper


class GraphCastSmall(GraphCastOper):
    """GraphCast small model (ERA5 1979-2015, 1° resolution, mesh 2to5).

    Faster than the operational model, making it well-suited for comparing
    GPU performance across hardware tiers.  Note that input data must be
    provided at 1° resolution (use CDS or a pre-prepared GRIB file).
    """

    _asset_files = ASSET_FILES_SMALL
