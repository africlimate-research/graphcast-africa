# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""GPU benchmark runner for GraphCast Africa forecasts."""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


@dataclass
class LatencySummary:
    runs: int
    avg_seconds: float
    min_seconds: float
    max_seconds: float
    p95_seconds: float


def summarise_latencies(latencies: list[float]) -> LatencySummary:
    if not latencies:
        raise ValueError("latencies must not be empty")
    ordered = sorted(latencies)
    # Nearest-rank percentile is intentional for small sample benchmark runs.
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return LatencySummary(
        runs=len(latencies),
        avg_seconds=statistics.fmean(latencies),
        min_seconds=ordered[0],
        max_seconds=ordered[-1],
        p95_seconds=ordered[p95_idx],
    )


def _query_gpu_snapshot() -> dict[str, str] | None:
    query = (
        "name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
        "driver_version"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    row = output.stdout.strip().splitlines()
    if not row:
        return None
    parts = [item.strip() for item in row[0].split(",")]
    if len(parts) != 6:
        return None
    return {
        "name": parts[0],
        "utilization_gpu_percent": parts[1],
        "memory_used_mib": parts[2],
        "memory_total_mib": parts[3],
        "temperature_c": parts[4],
        "driver_version": parts[5],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark GraphCast Africa GPU runtime")
    p.add_argument(
        "--source", default="opendata", choices=["cds", "opendata", "file", "gcs"]
    )
    p.add_argument("--file", default=None)
    p.add_argument("--date", required=True)
    p.add_argument("--time", default="0000")
    p.add_argument("--lead-time", default=120, type=int)
    p.add_argument("--assets", default="./assets")
    p.add_argument("--warmup-runs", default=1, type=int)
    p.add_argument("--benchmark-runs", default=3, type=int)
    p.add_argument("--no-subset", action="store_true")
    p.add_argument("--vars", default=None)
    p.add_argument("--json", action="store_true", help="Print benchmark output as JSON")
    return p


def _run_once(
    model,
    *,
    fields_sfc,
    fields_pl,
    start_date: datetime,
    lead_time_hours: int,
    subset_africa: bool,
    variables: list[str] | None,
) -> float:
    t0 = time.perf_counter()
    model.run(
        fields_sfc=fields_sfc,
        fields_pl=fields_pl,
        start_date=start_date,
        lead_time_hours=lead_time_hours,
        subset_africa=subset_africa,
        variables=variables,
    )
    return time.perf_counter() - t0


def _check_assets(path: str) -> bool:
    from graphcast_africa.model.assets import check_assets

    return check_assets(path)


def _get_source(source: str, file_path: str | None):
    from graphcast_africa.data.registry import get_source

    kwargs = {"path": file_path} if source == "file" else {}
    return get_source(source, **kwargs)


def _new_model(assets_dir: str):
    from graphcast_africa.model.graphcast_oper import GraphCastOper

    return GraphCastOper(assets_dir=assets_dir)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.source == "file" and not args.file:
        raise ValueError("--file is required when --source file")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be > 0")
    if not _check_assets(args.assets):
        raise FileNotFoundError(
            f"Assets are missing at {args.assets}. "
            f"Run: python scripts/download_assets.py --assets {args.assets}"
        )

    variables = [v.strip() for v in args.vars.split(",")] if args.vars else None
    start_date = datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H%M").replace(
        tzinfo=UTC
    )

    t_fetch0 = time.perf_counter()
    src = _get_source(args.source, args.file)
    fields_sfc, fields_pl = src.retrieve(args.date, args.time)
    data_fetch_seconds = time.perf_counter() - t_fetch0

    model = _new_model(assets_dir=args.assets)
    t_load0 = time.perf_counter()
    model.load()
    model_load_seconds = time.perf_counter() - t_load0

    warmup_latencies: list[float] = []
    for _ in range(args.warmup_runs):
        warmup_latencies.append(
            _run_once(
                model,
                fields_sfc=fields_sfc,
                fields_pl=fields_pl,
                start_date=start_date,
                lead_time_hours=args.lead_time,
                subset_africa=not args.no_subset,
                variables=variables,
            )
        )

    benchmark_latencies: list[float] = []
    for _ in range(args.benchmark_runs):
        benchmark_latencies.append(
            _run_once(
                model,
                fields_sfc=fields_sfc,
                fields_pl=fields_pl,
                start_date=start_date,
                lead_time_hours=args.lead_time,
                subset_africa=not args.no_subset,
                variables=variables,
            )
        )

    summary = summarise_latencies(benchmark_latencies)
    throughput_lead_hours_per_second = args.lead_time / summary.avg_seconds
    first_rollout_seconds = (
        warmup_latencies[0] if warmup_latencies else benchmark_latencies[0]
    )

    payload: dict[str, object] = {
        "run_config": {
            "source": args.source,
            "date": args.date,
            "time": args.time,
            "lead_time_hours": args.lead_time,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "subset_africa": not args.no_subset,
            "variables": variables,
        },
        "timing_seconds": {
            "data_fetch": data_fetch_seconds,
            "model_load": model_load_seconds,
            "first_rollout": first_rollout_seconds,
            "benchmark": asdict(summary),
        },
        "throughput": {
            "lead_hours_per_second": throughput_lead_hours_per_second,
        },
        "gpu_snapshot": _query_gpu_snapshot(),
    }
    return payload


def main() -> None:
    args = build_parser().parse_args()
    payload = run_benchmark(args)
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        print(output)
        return
    LOG.info(
        "Benchmark complete: avg=%.3fs p95=%.3fs throughput=%.2f lead-hours/s",
        payload["timing_seconds"]["benchmark"]["avg_seconds"],
        payload["timing_seconds"]["benchmark"]["p95_seconds"],
        payload["throughput"]["lead_hours_per_second"],
    )
    LOG.debug("Benchmark details:\n%s", output)


if __name__ == "__main__":
    main()
