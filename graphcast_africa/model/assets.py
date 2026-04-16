"""Download GraphCast weights and stats from the DeepMind GCS bucket."""
from __future__ import annotations

import logging
import os

from multiurl import download

from graphcast_africa.fields.graphcast_fields import ASSET_FILES, GCS_BASE_URL

LOG = logging.getLogger(__name__)

def download_assets(
    assets_dir: str = "./assets",
    force: bool = False,
    asset_files: list[str] | None = None,
) -> None:
    if asset_files is None:
        asset_files = ASSET_FILES
    os.makedirs(assets_dir, exist_ok=True)
    for relative_path in asset_files:
        dest = os.path.join(assets_dir, relative_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and not force:
            LOG.info("Already present, skipping: %s", dest); continue
        url = GCS_BASE_URL.format(file=relative_path)
        LOG.info("Downloading %s -> %s", url, dest)
        download(url, dest + ".tmp")
        os.rename(dest + ".tmp", dest)
    LOG.info("All assets ready in %s", assets_dir)

def check_assets(
    assets_dir: str = "./assets",
    asset_files: list[str] | None = None,
) -> bool:
    if asset_files is None:
        asset_files = ASSET_FILES
    missing = [os.path.join(assets_dir, f) for f in asset_files
               if not os.path.exists(os.path.join(assets_dir, f))]
    if missing:
        LOG.warning("Missing assets:\n  %s", "\n  ".join(missing)); return False
    return True
