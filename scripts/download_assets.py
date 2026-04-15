"""
Download GraphCast oper weights and stats from the DeepMind GCS bucket.
Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --assets /data/graphcast --force
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
from graphcast_africa.model.assets import download_assets

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default="./assets")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
download_assets(assets_dir=args.assets, force=args.force)
