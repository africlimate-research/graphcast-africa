"""
Download GraphCast oper weights and stats from the DeepMind GCS bucket.
Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --model small
    python scripts/download_assets.py --assets /data/graphcast --force
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
from graphcast_africa.model.assets import download_assets

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default="./assets")
parser.add_argument("--force", action="store_true")
parser.add_argument(
    "--model",
    default="operational",
    choices=["operational", "small"],
    help="Model variant to download assets for (default: operational)",
)
args = parser.parse_args()

if args.model == "small":
    from graphcast_africa.fields.graphcast_fields import ASSET_FILES_SMALL
    asset_files = ASSET_FILES_SMALL
else:
    asset_files = None  # download_assets defaults to ASSET_FILES

download_assets(assets_dir=args.assets, force=args.force, asset_files=asset_files)
