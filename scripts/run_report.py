# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""10-day GPU performance report for GraphCast Africa.

Runs a single 10-day (240 h) forecast and prints a concise summary of
forecast timing and GPU hardware metrics.  Intended for comparing
performance across different GPU types, not for model correctness checks.

Usage::

    python scripts/run_report.py --date 20240601 --assets ./assets
    python scripts/run_report.py --date 20240601 --model small --source cds --assets ./assets
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Ensure the project root is importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import UTC, datetime

LEAD_TIME_HOURS = 240  # 10-day forecast
WIDTH = 60


def _query_gpu() -> dict[str, str] | None:
    """Return GPU hardware metrics from nvidia-smi, or None if unavailable."""
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,driver_version"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    row = out.stdout.strip().splitlines()
    if not row:
        return None
    parts = [p.strip() for p in row[0].split(",")]
    if len(parts) != 6:
        return None
    return {
        "name": parts[0],
        "driver_version": parts[5],
        "memory_total_mib": parts[3],
        "memory_used_mib": parts[2],
        "utilization_pct": parts[1],
        "temperature_c": parts[4],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a 10-day GraphCast Africa forecast and print a GPU performance report"
    )
    p.add_argument("--date", required=True, help="Forecast start date, e.g. 20240601")
    p.add_argument("--time", default="0000", help="Start time HHMM (default: 0000)")
    p.add_argument(
        "--source",
        default="opendata",
        choices=["cds", "opendata", "file"],
        help="Input data source (default: opendata)",
    )
    p.add_argument("--file", default=None, help="Path to GRIB file (required when --source file)")
    p.add_argument("--assets", default="./assets", help="Path to model assets directory")
    p.add_argument(
        "--model",
        default="operational",
        choices=["operational", "small"],
        help=(
            "Model variant to run: 'operational' (0.25°, default) or 'small' (1°, faster). "
            "The small model requires 1° input data — use --source cds or --source file."
        ),
    )
    return p


def _print_report(
    *,
    date: str,
    time_str: str,
    source: str,
    assets: str,
    model_variant: str,
    gpu: dict[str, str] | None,
    data_fetch_s: float,
    model_load_s: float,
    forecast_s: float,
) -> None:
    throughput = LEAD_TIME_HOURS / forecast_s if forecast_s > 0 else float("nan")

    print()
    print("=" * WIDTH)
    print("  GraphCast Africa — 10-Day Forecast Performance Report")
    print(f"  Generated : {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * WIDTH)

    print("  Run configuration")
    print(f"    Start         : {date} {time_str}")
    print(f"    Lead time     : {LEAD_TIME_HOURS} h  (10-day forecast)")
    print(f"    Model variant : {model_variant}")
    print(f"    Data source   : {source}")
    print(f"    Assets        : {assets}")

    print()
    print("  Timing")
    print(f"    Data fetch    : {data_fetch_s:>8.2f} s")
    print(f"    Model load    : {model_load_s:>8.2f} s")
    print(f"    Forecast run  : {forecast_s:>8.2f} s")
    print(f"    Throughput    : {throughput:>8.2f} lead-h / s")

    print()
    if gpu:
        print("  GPU")
        print(f"    Name          : {gpu['name']}")
        print(f"    Driver        : {gpu['driver_version']}")
        print(f"    Memory total  : {gpu['memory_total_mib']:>6} MiB")
        print(f"    Memory used   : {gpu['memory_used_mib']:>6} MiB")
        print(f"    Utilization   : {gpu['utilization_pct']:>5} %")
        print(f"    Temperature   : {gpu['temperature_c']:>5} °C")
    else:
        print("  GPU   : not detected (nvidia-smi unavailable)")

    print("=" * WIDTH)
    print()


def main() -> None:
    args = _build_parser().parse_args()

    if args.source == "file" and not args.file:
        sys.exit("error: --file is required when --source file")

    if args.model == "small" and args.source == "opendata":
        sys.exit(
            "error: the small model requires 1° input data; "
            "opendata only provides 0.25°. Use --source cds or --source file."
        )

    # Select model class and matching asset-files list.
    if args.model == "small":
        from graphcast_africa.fields.graphcast_fields import ASSET_FILES_SMALL, GRID_SMALL
        from graphcast_africa.model.graphcast_small import GraphCastSmall as ModelClass
        asset_files = ASSET_FILES_SMALL
        grid = GRID_SMALL
    else:
        from graphcast_africa.fields.graphcast_fields import ASSET_FILES, GRID
        from graphcast_africa.model.graphcast_oper import GraphCastOper as ModelClass
        asset_files = ASSET_FILES
        grid = GRID

    from graphcast_africa.data.registry import get_source
    from graphcast_africa.model.assets import check_assets

    if not check_assets(args.assets, asset_files=asset_files):
        sys.exit(
            f"error: assets missing at {args.assets}. "
            f"Run: python scripts/download_assets.py --model {args.model} --assets {args.assets}"
        )

    start_date = datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H%M").replace(tzinfo=UTC)

    source_kwargs: dict = {}
    if args.source == "file":
        source_kwargs["path"] = args.file
    elif args.model == "small":
        # CDS supports arbitrary grids; pass through so ERA5 is fetched at 1°.
        source_kwargs["grid"] = grid

    src = get_source(args.source, **source_kwargs)

    t0 = time.perf_counter()
    fields_sfc, fields_pl = src.retrieve(args.date, args.time)
    data_fetch_s = time.perf_counter() - t0

    model = ModelClass(assets_dir=args.assets)
    t0 = time.perf_counter()
    model.load()
    model_load_s = time.perf_counter() - t0

    gpu_before = _query_gpu()

    t0 = time.perf_counter()
    model.run(
        fields_sfc=fields_sfc,
        fields_pl=fields_pl,
        start_date=start_date,
        lead_time_hours=LEAD_TIME_HOURS,
        subset_africa=True,
    )
    forecast_s = time.perf_counter() - t0

    gpu_after = _query_gpu()
    # Prefer post-run snapshot (peak utilization more likely captured there).
    gpu = gpu_after or gpu_before

    _print_report(
        date=args.date,
        time_str=args.time,
        source=args.source,
        assets=args.assets,
        model_variant=args.model,
        gpu=gpu,
        data_fetch_s=data_fetch_s,
        model_load_s=model_load_s,
        forecast_s=forecast_s,
    )


if __name__ == "__main__":
    main()

