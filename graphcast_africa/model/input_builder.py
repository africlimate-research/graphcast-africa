# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Inlined implementation of create_training_xarray from ecmwf-lab/ai-models-graphcast.

Adapted from:
  https://github.com/ecmwf-lab/ai-models-graphcast/blob/main/src/ai_models_graphcast/input.py
  (C) Copyright 2023 ECMWF, Apache Licence Version 2.0
"""
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

        # ── Surface fields ──────────────────────────────────────────────────��─
        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        for field in fields_sfc:
            # ← ADD THIS
            sfc[field.metadata("param")].append(field)

        # ── Pressure-level fields ─────────────────────────────────────────────
        # order_by("param", "valid_datetime", "level") → time-outer, level-inner
        # i.e. for each param: [t0_lev0, t0_lev1, ..., t1_lev0, t1_lev1, ...]
        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        pl_datetimes = set()
        for field in fields_pl:
            pl_datetimes.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        # n_times is driven by PL fields — always purely time-varying, no statics
        n_times  = len(pl_datetimes)   # == len(lagged) == 2
        n_levels = len(levels)         # == 13

        LOG.debug("n_times=%d  n_levels=%d  n_all_steps=%d", n_times, n_levels, len(all_datetimes))

        data_vars = {}

        # Surface variables
        for param, fields in sfc.items():
            if param not in CF_NAME_SFC:
                LOG.debug("Skipping unknown SFC param %r", param)
                continue
            if param in ("z", "lsm"):
                # static — no time dimension
                data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], fields[0].to_numpy())
                continue
            # time-varying SFC
            arr  = np.stack([f.to_numpy(dtype=np.float32) for f in fields])
            data = arr[np.newaxis, ...]
            data = np.pad(
                data,
                ((0, 0), (0, len(all_datetimes) - n_times), (0, 0), (0, 0)),
                constant_values=np.nan,
            )
            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

            # Time-varying SFC: stack → (n_times, h, w), add batch → (1, n_times, h, w)
            arr  = np.stack([f.to_numpy(dtype=np.float32) for f in fields])
            data = arr[np.newaxis, ...]
            # pad future (target) timesteps with NaN
            data = np.pad(
                data,
                ((0, 0), (0, len(all_datetimes) - n_times), (0, 0), (0, 0)),
                constant_values=np.nan,
            )
            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

        # Pressure-level variables
        for param, fields in pl.items():
            # stack → (n_times * n_levels, h, w)
            arr  = np.stack([f.to_numpy(dtype=np.float32) for f in fields])
            # reshape → (n_times, n_levels, h, w), add batch → (1, n_times, n_levels, h, w)
            data = arr.reshape(n_times, n_levels, len(lat), len(lon))[np.newaxis, ...]
            # pad future timesteps with NaN
            data = np.pad(
                data,
                ((0, 0), (0, len(all_datetimes) - n_times), (0, 0), (0, 0), (0, 0)),
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
        # GraphCast expects lat ascending (south → north)
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values))

    if constants:
        x = xr.load_dataset(constants)
        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas

def _roll_lon(ds: xr.Dataset) -> xr.Dataset:
    """Roll longitudes from 0–360 to -180–180 if needed."""
    if ds.lon.values.max() > 180:
        ds = ds.assign_coords(lon=(ds.lon.values + 180) % 360 - 180)
        ds = ds.sortby("lon")
    return ds
