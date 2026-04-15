# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""GPU benchmark tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.benchmark_gpu import (
    LatencySummary,
    _query_gpu_snapshot,
    run_benchmark,
    summarise_latencies,
)


def test_summarise_latencies():
    summary = summarise_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary == LatencySummary(
        runs=4,
        avg_seconds=2.5,
        min_seconds=1.0,
        max_seconds=4.0,
        p95_seconds=4.0,
    )


def test_summarise_latencies_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        summarise_latencies([])


def test_query_gpu_snapshot_parses(monkeypatch):
    class _Res:
        stdout = "NVIDIA A10, 87, 1024, 23028, 61, 550.54.15\n"

    def _fake_run(*_args, **_kwargs):
        return _Res()

    monkeypatch.setattr("scripts.benchmark_gpu.subprocess.run", _fake_run)
    assert _query_gpu_snapshot() == {
        "name": "NVIDIA A10",
        "utilization_gpu_percent": "87",
        "memory_used_mib": "1024",
        "memory_total_mib": "23028",
        "temperature_c": "61",
        "driver_version": "550.54.15",
    }


def test_run_benchmark_with_mocks(monkeypatch):
    class _FakeSource:
        def retrieve(self, *_args, **_kwargs):
            return "fields_sfc", "fields_pl"

    class _FakeModel:
        def __init__(self, assets_dir):
            self.assets_dir = assets_dir
            self.calls = 0

        def load(self):
            return None

        def run(self, **_kwargs):
            self.calls += 1
            return {"ok": True}

    timer_values = iter([0.0, 0.2, 0.2, 0.6, 0.6, 1.0, 1.0, 1.5, 1.5, 2.1])

    monkeypatch.setattr("scripts.benchmark_gpu._check_assets", lambda _assets: True)
    monkeypatch.setattr(
        "scripts.benchmark_gpu._get_source", lambda *_args, **_kwargs: _FakeSource()
    )
    monkeypatch.setattr("scripts.benchmark_gpu._new_model", _FakeModel)
    monkeypatch.setattr("scripts.benchmark_gpu._query_gpu_snapshot", lambda: None)
    monkeypatch.setattr(
        "scripts.benchmark_gpu.time.perf_counter", lambda: next(timer_values)
    )

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

    payload = run_benchmark(args)
    assert payload["timing_seconds"]["data_fetch"] == pytest.approx(0.2)
    assert payload["timing_seconds"]["model_load"] == pytest.approx(0.4)
    assert payload["timing_seconds"]["first_rollout"] == pytest.approx(0.4)
    assert payload["timing_seconds"]["benchmark"]["runs"] == 2
    assert payload["timing_seconds"]["benchmark"]["avg_seconds"] == pytest.approx(0.55)
    expected = args.lead_time / payload["timing_seconds"]["benchmark"]["avg_seconds"]
    assert payload["throughput"]["lead_hours_per_second"] == pytest.approx(expected)
