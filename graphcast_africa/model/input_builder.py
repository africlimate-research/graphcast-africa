# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Inlined implementation of create_training_xarray from ecmwf-lab/ai-models-graphcast."""
from __future__ import annotations

import datetime
import logging
from collections import defaultdict

import earthkit.data as ekd
import numpy as np
import xarray as xr

from graphcast_africa.fields.graphcast_fields import CF_NAME_PL, CF_NAME_SFC

LOG = logging.getLogger(__name__)


def _forcing_variables_numpy(sample, forcing_variables, dates):
    ds = ekd.from_source(
        "forcings",
        sample,
        date=dates,
        param=forcing_variables,
    )
    return (
        ds.order_by(param=forcing_variables, valid_datetime="ascending")
        .to_numpy(dtype=np.float32)
        .reshape(len(forcing_variables), len(dates), 721, 1440)
    )


def create_training_xarray(
    *,
    fields_sfc,
    fields_pl,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
    timer,
) -> tuple[xr.Dataset, list[datetime.timedelta]]:
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]
    all_datetimes = [start_date + td for td in time_deltas]

    with timer("Creating forcing variables"):
        forcing_numpy = _forcing_variables_numpy(fields_sfc, forcing_variables, all_datetimes)

    with timer("Converting GRIB to xarray"):
        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")

        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        given_datetimes_sfc = set()
        for field in fields_sfc:
            given_datetimes_sfc.add(field.metadata("valid_datetime"))
            sfc[field.metadata("param")].append(field)

        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        given_datetimes_pl = set()
        for field in fields_pl:
            given_datetimes_pl.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        data_vars = {}

        for param, fields in sfc.items():
            if param in ("z", "lsm"):
                data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], fields[0].to_numpy())
                continue
            data = np.stack([f.to_numpy(dtype=np.float32) for f in fields]).reshape(
                1, len(given_datetimes_sfc), len(lat), len(lon)
            )
            data = np.pad(
                data,
                ((0, 0), (0, len(all_datetimes) - len(given_datetimes_sfc)), (0, 0), (0, 0)),
                constant_values=np.nan,
            )
            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

        for param, fields in pl.items():
            data = np.stack([f.to_numpy(dtype=np.float32) for f in fields]).reshape(
                1, len(given_datetimes_pl), len(levels), len(lat), len(lon)
            )
            data = np.pad(
                data,
                ((0, 0), (0, len(all_datetimes) - len(given_datetimes_pl)), (0, 0), (0, 0), (0, 0)),
                constant_values=np.nan,
            )
            data_vars[CF_NAME_PL[param]] = (["batch", "time", "level", "lat", "lon"], data)

        data_vars["toa_incident_solar_radiation"] = (
            ["batch", "time", "lat", "lon"],
            forcing_numpy[0:1, :, :, :],
        )

        training_xarray = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=lon,
                lat=lat,
                time=time_deltas,
                datetime=(("batch", "time"), [all_datetimes]),
                level=sorted(levels),
            ),
        )

    with timer("Reindexing"):
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values))

    if constants:
        x = xr.load_dataset(constants)
        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas
