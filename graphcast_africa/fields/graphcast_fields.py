"""
GraphCast Operational v1 field definitions.
Source: deepmind/graphcast repo + ecmwf-lab/ai-models-graphcast model.py
"""
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PARAM_LEVEL_PL = (["t", "z", "u", "v", "w", "q"], PRESSURE_LEVELS)
PARAM_SFC = ["lsm", "2t", "msl", "10u", "10v", "tp", "z"]

CF_NAME_SFC = {
    "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature", "lsm": "land_sea_mask",
    "msl": "mean_sea_level_pressure", "tp": "total_precipitation_6hr",
    "z": "geopotential_at_surface",
}
CF_NAME_PL = {
    "q": "specific_humidity", "t": "temperature",
    "u": "u_component_of_wind", "v": "v_component_of_wind",
    "w": "vertical_velocity", "z": "geopotential",
}

GRID = [0.25, 0.25]
AREA = [90, 0, -90, 360]     # global (N, W, S, E)
AFRICA_LAT = (-40.0, 40.0)   # lat: -40 to 40
AFRICA_LON = (-20.0, 70.0)   # lon: -20 to 70

LAGGED_HOURS = [-6, 0]
HOUR_STEPS   = 6

ASSET_FILES = [
    (
        "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 -"
        " pressure levels 13 - mesh 2to6 - precipitation output only.npz"
    ),
    "stats/diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
]

# GraphCast small: 1° resolution, lighter mesh — faster on modest GPUs.
# The small checkpoint lives under the "graphcast/" prefix in the GCS bucket
# (consistent with the DeepMind demo notebook), unlike the operational model
# which is published directly under "params/".
GRID_SMALL = [1.0, 1.0]
ASSET_FILES_SMALL = [
    (
        "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
        "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
    ),
    "stats/diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
]

GCS_BASE_URL = "https://storage.googleapis.com/dm_graphcast/{file}"
