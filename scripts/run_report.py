# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Summary report generator for GraphCast Africa.

Runs every project check and prints a human-readable summary.
Exit code equals the number of failed/errored checks (0 = all passed).

Usage::

    python scripts/run_report.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

# Ensure the project root is importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

_results: list["CheckResult"] = []


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


def _check(name: str):
    """Decorator: run the wrapped function immediately and record its result."""

    def decorator(fn):
        try:
            fn()
            _results.append(CheckResult(name=name, status=PASS))
        except AssertionError as exc:
            _results.append(CheckResult(name=name, status=FAIL, detail=str(exc)))
        except Exception:
            _results.append(
                CheckResult(name=name, status=ERROR, detail=traceback.format_exc())
            )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------


@_check("data: registry returns correct source types")
def _():
    from graphcast_africa.data.cds import CDSSource
    from graphcast_africa.data.opendata import OpenDataSource
    from graphcast_africa.data.registry import get_source

    assert isinstance(get_source("cds"), CDSSource)
    assert isinstance(get_source("opendata"), OpenDataSource)


@_check("data: unknown source raises ValueError")
def _():
    from graphcast_africa.data.registry import get_source

    try:
        get_source("mars")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "Unknown source" in str(exc)


@_check("data: lagged datetimes at 1200")
def _():
    from graphcast_africa.data.base import DataSource

    r = DataSource._lagged_datetimes("20240101", "1200")
    assert r[0] == ("20240101", 600)
    assert r[1] == ("20240101", 1200)


@_check("data: lagged datetimes crosses midnight")
def _():
    from graphcast_africa.data.base import DataSource

    r = DataSource._lagged_datetimes("20240101", "0000")
    assert r[0] == ("20231231", 1800)
    assert r[1] == ("20240101", 0)


@_check("data: GRIB file source raises on bad path")
def _():
    from graphcast_africa.data.grib_file import GRIBFileSource

    try:
        GRIBFileSource(path="/nonexistent/file.grib")
        raise AssertionError("Expected an exception for a bad path")
    except AssertionError:
        raise
    except Exception:
        pass  # any non-AssertionError exception is the expected outcome


# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------


@_check("benchmark: summarise_latencies computes correctly")
def _():
    from scripts.benchmark_gpu import LatencySummary, summarise_latencies

    summary = summarise_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary == LatencySummary(
        runs=4,
        avg_seconds=2.5,
        min_seconds=1.0,
        max_seconds=4.0,
        p95_seconds=4.0,
    )


@_check("benchmark: summarise_latencies raises on empty input")
def _():
    from scripts.benchmark_gpu import summarise_latencies

    try:
        summarise_latencies([])
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "must not be empty" in str(exc)


@_check("benchmark: _query_gpu_snapshot parses nvidia-smi output")
def _():
    import scripts.benchmark_gpu as bm

    class _Res:
        stdout = "NVIDIA A10, 87, 1024, 23028, 61, 550.54.15\n"

    with patch.object(bm.subprocess, "run", return_value=_Res()):
        result = bm._query_gpu_snapshot()

    assert result == {
        "name": "NVIDIA A10",
        "utilization_gpu_percent": "87",
        "memory_used_mib": "1024",
        "memory_total_mib": "23028",
        "temperature_c": "61",
        "driver_version": "550.54.15",
    }


@_check("benchmark: run_benchmark produces expected payload shape")
def _():
    import scripts.benchmark_gpu as bm
    from scripts.benchmark_gpu import run_benchmark

    class _FakeSource:
        def retrieve(self, *_args, **_kwargs):
            return "fields_sfc", "fields_pl"

    class _FakeModel:
        def __init__(self, assets_dir):
            self.assets_dir = assets_dir

        def load(self):
            return None

        def run(self, **_kwargs):
            return {"ok": True}

    timer_values = iter([0.0, 0.2, 0.2, 0.6, 0.6, 1.0, 1.0, 1.5, 1.5, 2.1])

    args = SimpleNamespace(
        source="opendata",
        file=None,
        date="20240601",
        time="0000",
        lead_time=120,
        assets="./assets",
        warmup_runs=1,
        benchmark_runs=2,
        no_subset=False,
        vars=None,
    )

    with (
        patch.object(bm, "_check_assets", return_value=True),
        patch.object(bm, "_get_source", return_value=_FakeSource()),
        patch.object(bm, "_new_model", _FakeModel),
        patch.object(bm, "_query_gpu_snapshot", return_value=None),
        patch("scripts.benchmark_gpu.time.perf_counter", side_effect=lambda: next(timer_values)),
    ):
        payload = run_benchmark(args)

    timing = payload["timing_seconds"]
    assert math.isclose(timing["data_fetch"], 0.2)
    assert math.isclose(timing["model_load"], 0.4)
    assert math.isclose(timing["first_rollout"], 0.4)
    assert timing["benchmark"]["runs"] == 2
    assert math.isclose(timing["benchmark"]["avg_seconds"], 0.55)
    expected_throughput = args.lead_time / timing["benchmark"]["avg_seconds"]
    assert math.isclose(
        payload["throughput"]["lead_hours_per_second"], expected_throughput
    )


# ---------------------------------------------------------------------------
# Model / assets
# ---------------------------------------------------------------------------


@_check("model: check_assets returns False when files missing")
def _():
    from graphcast_africa.model.assets import check_assets

    with tempfile.TemporaryDirectory() as tmpdir:
        assert check_assets(tmpdir) is False


@_check("model: check_assets returns True when all files present")
def _():
    from graphcast_africa.fields.graphcast_fields import ASSET_FILES
    from graphcast_africa.model.assets import check_assets

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in ASSET_FILES:
            dest = os.path.join(tmpdir, f)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "w").close()
        assert check_assets(tmpdir) is True


@_check("model: load raises without assets")
def _():
    from graphcast_africa.model.graphcast_oper import GraphCastOper

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            GraphCastOper(assets_dir=tmpdir).load()
            raise AssertionError("Expected FileNotFoundError or ModuleNotFoundError")
        except AssertionError:
            raise
        except (FileNotFoundError, ModuleNotFoundError):
            pass


@_check("model: Africa domain constants")
def _():
    from graphcast_africa.fields.graphcast_fields import AFRICA_LAT, AFRICA_LON

    assert AFRICA_LAT == (-40.0, 40.0)
    assert AFRICA_LON == (-20.0, 70.0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _make_stats(tmpdir: str) -> str:
    import xarray as xr

    stats_dir = os.path.join(tmpdir, "stats")
    os.makedirs(stats_dir)
    for fname, val in [
        ("mean_by_level.nc", 0.0),
        ("stddev_by_level.nc", 1.0),
        ("diffs_stddev_by_level.nc", 0.5),
    ]:
        xr.Dataset(
            {"temperature": xr.DataArray([val], dims=["level"])}
        ).to_netcdf(os.path.join(stats_dir, fname))
    return tmpdir


@_check("normalisation: normalise produces expected value")
def _():
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmpdir:
        assets = _make_stats(tmpdir)
        from graphcast_africa.normalisation.stats import load_stats

        load_stats.cache_clear()
        from graphcast_africa.normalisation.normalise import normalise

        ds = xr.Dataset({"temperature": xr.DataArray([5.0], dims=["level"])})
        result = float(normalise(ds, assets)["temperature"].values[0])
        assert math.isclose(result, 5.0, rel_tol=1e-5)


@_check("normalisation: denormalise(normalise(x)) ≈ x (roundtrip)")
def _():
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmpdir:
        assets = _make_stats(tmpdir)
        from graphcast_africa.normalisation.stats import load_stats

        load_stats.cache_clear()
        from graphcast_africa.normalisation.normalise import denormalise, normalise

        ds = xr.Dataset({"temperature": xr.DataArray([273.15], dims=["level"])})
        result = float(
            denormalise(normalise(ds, assets), assets)["temperature"].values[0]
        )
        assert math.isclose(result, 273.15, rel_tol=1e-5)


@_check("normalisation: missing stats raises FileNotFoundError")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        from graphcast_africa.normalisation.stats import load_stats

        load_stats.cache_clear()
        try:
            load_stats(tmpdir)
            raise AssertionError("Expected FileNotFoundError was not raised")
        except AssertionError:
            raise
        except FileNotFoundError as exc:
            assert "Stats file not found" in str(exc)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report() -> int:
    width = 64
    passed = [r for r in _results if r.status == PASS]
    failed = [r for r in _results if r.status == FAIL]
    errored = [r for r in _results if r.status == ERROR]

    icon = {PASS: "✓", FAIL: "✗", ERROR: "!"}

    print()
    print("=" * width)
    print("  GraphCast Africa — Summary Report")
    print(f"  Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * width)
    for r in _results:
        print(f"  {icon.get(r.status, '?')} [{r.status:<5}] {r.name}")
        if r.detail:
            for line in r.detail.strip().splitlines():
                print(f"           {line}")
    print("-" * width)
    print(
        f"  Total: {len(_results)}  |  "
        f"Passed: {len(passed)}  |  "
        f"Failed: {len(failed)}  |  "
        f"Errors: {len(errored)}"
    )
    print("=" * width)
    print()
    return len(failed) + len(errored)


if __name__ == "__main__":
    sys.exit(_print_report())
