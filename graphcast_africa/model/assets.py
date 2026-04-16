"""Download GraphCast weights and stats from the DeepMind GCS bucket."""
from __future__ import annotations

import logging
import os

from graphcast_africa.fields.graphcast_fields import ASSET_FILES, GCS_BASE_URL

LOG = logging.getLogger(__name__)

_GCS_BUCKET = "dm_graphcast"


def _download_blob(blob_path: str, dest: str) -> None:
    """Download one GCS object using the GCS Python client (JSON API).

    The GCS Python client is used in preference to a plain HTTP GET because
    the ``dm_graphcast`` bucket is configured to return 404 for anonymous
    HTTP (XML API) requests to objects that require auth, whereas the JSON
    API correctly surfaces the object for anonymous readers.  This mirrors
    the approach used in the official DeepMind ``graphcast_demo.ipynb``
    notebook (``storage.Client.create_anonymous_client()``).

    Falls back to ``multiurl`` if ``google-cloud-storage`` is not installed.
    """
    try:
        from google.cloud import storage as gcs  # type: ignore[import]

        client = gcs.Client.create_anonymous_client()
        bucket = client.bucket(_GCS_BUCKET)
        blob = bucket.blob(blob_path)
        tmp = dest + ".tmp"
        blob.download_to_filename(tmp)
        os.rename(tmp, dest)
        return
    except ImportError:
        LOG.debug(
            "google-cloud-storage not installed; falling back to HTTP download. "
            "Install it with: pip install google-cloud-storage"
        )
    except Exception as exc:
        LOG.debug("GCS client failed (%s); falling back to HTTP download", exc)

    # HTTP fallback via multiurl
    from multiurl import download  # type: ignore[import]

    url = GCS_BASE_URL.format(file=blob_path)
    LOG.debug("HTTP fallback: %s", url)
    tmp = dest + ".tmp"
    download(url, tmp)
    os.rename(tmp, dest)


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
            LOG.info("Already present, skipping: %s", dest)
            continue
        LOG.info("Downloading gs://%s/%s -> %s", _GCS_BUCKET, relative_path, dest)
        _download_blob(relative_path, dest)
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
