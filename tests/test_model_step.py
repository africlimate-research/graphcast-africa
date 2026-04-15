"""Phase 3 tests — model loading and single step."""
import os

import pytest


def test_check_assets_missing(tmp_path):
    from graphcast_africa.model.assets import check_assets
    assert check_assets(str(tmp_path)) is False

def test_check_assets_present(tmp_path):
    from graphcast_africa.fields.graphcast_fields import ASSET_FILES
    from graphcast_africa.model.assets import check_assets
    for f in ASSET_FILES:
        dest = os.path.join(str(tmp_path), f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "w").close()
    assert check_assets(str(tmp_path)) is True

def test_load_raises_without_assets(tmp_path):
    from graphcast_africa.model.graphcast_oper import GraphCastOper
    with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
        GraphCastOper(assets_dir=str(tmp_path)).load()

def test_africa_domain():
    from graphcast_africa.fields.graphcast_fields import AFRICA_LAT, AFRICA_LON
    assert AFRICA_LAT == (-40.0, 40.0)
    assert AFRICA_LON == (-20.0, 70.0)
