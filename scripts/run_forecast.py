"""
End-to-end GraphCast forecast for Africa.
Usage:
    python scripts/run_forecast.py --source opendata --date 20240601 --time 0000
    python scripts/run_forecast.py --source cds --date 20230101 --time 1200 --lead-time 240
    python scripts/run_forecast.py --source file --file input.grib --date 20240101 --time 0000
    python scripts/run_forecast.py --source opendata --date 20240601 --time 0000 --vars 2m_temperature,mean_sea_level_pressure
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime

warnings.filterwarnings(
    "ignore",
    message=".*Dataset.dims.*will be changed to return a set.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Skipping gradient checkpointing.*",
    category=UserWarning,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    default="opendata", choices=["cds","opendata","file","gcs"])
    p.add_argument("--file",      default=None)
    p.add_argument("--date",      required=True)
    p.add_argument("--time",      default="0000")
    p.add_argument("--lead-time", default=120, type=int)
    p.add_argument("--output",    default="graphcast")
    p.add_argument("--assets",    default="./assets")
    p.add_argument("--no-subset", action="store_true")
    p.add_argument("--vars",      default=None,
                   help="Comma-separated CF variable names to keep, e.g. "
                        "2m_temperature,mean_sea_level_pressure,total_precipitation_6hr")
    args = p.parse_args()

    if args.source == "file" and not args.file:
        p.error("--file is required when --source file")

    variables = [v.strip() for v in args.vars.split(",")] if args.vars else None

    from graphcast_africa.model.assets import check_assets
    if not check_assets(args.assets):
        LOG.error("Run: python scripts/download_assets.py --assets %s", args.assets)
        raise SystemExit(1)

    from graphcast_africa.data.registry import get_source
    src = get_source(args.source, **( {"path": args.file} if args.source == "file" else {} ))
    LOG.info("Retrieving input fields ...")
    fields_sfc, fields_pl = src.retrieve(args.date, args.time)

    from graphcast_africa.model.graphcast_oper import GraphCastOper
    output = GraphCastOper(assets_dir=args.assets).run(
        fields_sfc=fields_sfc, fields_pl=fields_pl,
        start_date=datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H%M"),
        lead_time_hours=args.lead_time,
        subset_africa=not args.no_subset,
        variables=variables,
    )
    date_label = f"_{args.date}_{datetime.strptime(f'{args.time}', '%H%M').strftime('%H')}Z"
    output.to_netcdf(args.output+f"{date_label}.nc")
    LOG.info("Done. Output -> %s", args.output+f"{date_label}.nc")

if __name__ == "__main__": main()
