"""Standalone GraphCast Operational runner — no ai_models.Model dependency."""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import logging
import os

import xarray as xr

from graphcast_africa.fields.graphcast_fields import (AFRICA_LAT, AFRICA_LON,
                                                      ASSET_FILES, HOUR_STEPS,
                                                      LAGGED_HOURS)

LOG = logging.getLogger(__name__)

class GraphCastOper:
    hour_steps = HOUR_STEPS
    lagged     = LAGGED_HOURS

    def __init__(self, assets_dir: str = "./assets"):
        self.assets_dir = assets_dir
        self._model = self._ckpt = None

    def _asset_path(self, rel): return os.path.join(self.assets_dir, rel)

    def load(self) -> None:
        try:
            import haiku as hk
            import jax
            from graphcast import autoregressive, casting
            from graphcast import checkpoint as gc_ckpt
            from graphcast import graphcast as gc_model
            from graphcast import normalization
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "pip install git+https://github.com/deepmind/graphcast.git") from e

        LOG.info("Loading GraphCast checkpoint ...")
        diffs = xr.load_dataset(self._asset_path(ASSET_FILES[1])).compute()
        mean  = xr.load_dataset(self._asset_path(ASSET_FILES[2])).compute()
        std   = xr.load_dataset(self._asset_path(ASSET_FILES[3])).compute()

        def build(model_config, task_config):
            p = gc_model.GraphCast(model_config, task_config)
            p = casting.Bfloat16Cast(p)
            p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=diffs,
                                                 mean_by_level=mean, stddev_by_level=std)
            p = autoregressive.Predictor(p, gradient_checkpointing=True)
            return p

        @hk.transform_with_state
        def _fwd(model_config, task_config, inputs, targets_template, forcings):
            return build(model_config, task_config)(inputs, targets_template=targets_template,
                                                    forcings=forcings)

        with open(self._asset_path(ASSET_FILES[0]), "rb") as f:
            self._ckpt  = gc_ckpt.load(f, gc_model.CheckPoint)
            self._params = self._ckpt.params
            self._state  = {}
            self._mcfg   = self._ckpt.model_config
            self._tcfg   = self._ckpt.task_config
        LOG.info("Model: %s", self._ckpt.description)
        _apply = jax.jit(functools.partial(_fwd.apply, model_config=self._mcfg, task_config=self._tcfg))
        self._model = lambda **kw: _apply(params=self._params, state=self._state, **kw)[0]

    def run(self, fields_sfc, fields_pl, start_date,
            lead_time_hours: int = 240, subset_africa: bool = True) -> xr.Dataset:
        import jax
        from graphcast import data_utils

        from graphcast_africa.model.input_builder import create_training_xarray

        if self._model is None: self.load()

        LOG.info("Building training xarray ...")
        training_xarray, time_deltas = create_training_xarray(
            fields_sfc=fields_sfc, fields_pl=fields_pl, lagged=self.lagged,
            start_date=start_date, hour_steps=self.hour_steps, lead_time=lead_time_hours,
            forcing_variables=["toa_incident_solar_radiation"], constants=None, timer=_NoOpTimer(),
        )
        gc.collect()

        LOG.info("Extracting inputs / targets / forcings ...")
        input_xr, template, forcings = data_utils.extract_inputs_targets_forcings(
            training_xarray,
            target_lead_times=[f"{int(d.days*24+d.seconds/3600):d}h" for d in time_deltas[len(self.lagged):]],
            **dataclasses.asdict(self._tcfg),
        )

        LOG.info("Running JAX rollout (lead_time=%d h) ...", lead_time_hours)
        output: xr.Dataset = self._model(rng=jax.random.PRNGKey(0), inputs=input_xr,
                                          targets_template=template, forcings=forcings)
        if subset_africa:
            LOG.info("Subsetting to Africa: lat %s..%s, lon %s..%s",
                     AFRICA_LAT[0], AFRICA_LAT[1], AFRICA_LON[0], AFRICA_LON[1])
            output = output.sel(lat=slice(AFRICA_LAT[0], AFRICA_LAT[1]),
                                lon=slice(AFRICA_LON[0], AFRICA_LON[1]))
        return output

class _NoOpTimer:
    def __call__(self, label): return contextlib.nullcontext()
