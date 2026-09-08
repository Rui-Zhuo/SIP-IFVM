"""Central path configuration for SIP-IFVM post-processing scripts.

``GRID_PATH`` is fixed to the grid directory stored on the laptop.  Simulation
data can be selected explicitly from either the partial laptop dataset or the
complete dataset on the external F: drive.

This module only defines paths.  Importing it never creates directories.
"""

from __future__ import annotations

from pathlib import Path


# Repository roots
RESEARCH_ROOT = Path("E:/Research")
DATA_ROOT = RESEARCH_ROOT / "Data" / "SIP-IFVM"
CODE_ROOT = RESEARCH_ROOT / "Program" / "SIP-IFVM"
WORK_ROOT = RESEARCH_ROOT / "Work" / "Coronal_hole_by_SIP"

# Coronal-hole region of interest on the r_index=0 footpoint surface.
# The low/mid-latitude window is defined at CH_REGION_REFERENCE_TIME_HOURS
# and advected with the differential-rotation law in utils.py.
CH_REGION_REFERENCE_TIME_HOURS = 82.10
CH_REGION_NORTH_LATITUDE_MIN_DEG = 60.0
CH_REGION_LOW_MID_LON_MIN_DEG = 0.0
CH_REGION_LOW_MID_LON_MAX_DEG = 100.0
CH_REGION_LOW_MID_LAT_MIN_DEG = -60.0
CH_REGION_LOW_MID_LAT_MAX_DEG = 60.0

# Fixed grid location
GRID_DIR = DATA_ROOT / "grid"
GRID_PATH = GRID_DIR
GRID_FILE = GRID_DIR / "merged_spherical_grid.h5"

# Partial dataset stored on the laptop
LOCAL_DATA_ROOT = DATA_ROOT
LOCAL_MERGED_DIR = LOCAL_DATA_ROOT / "merged"
LOCAL_SOLUTION_DIR = LOCAL_DATA_ROOT / "solutions"
LOCAL_DATA_PATH = LOCAL_MERGED_DIR

# Complete dataset stored on the external drive
FULL_DATA_ROOT = Path("F:/Simulation/SIP-IFVM")
FULL_MERGED_DIR = FULL_DATA_ROOT / "merged"
FULL_SOLUTION_DIR = FULL_DATA_ROOT / "solutions"
FULL_DATA_PATH = FULL_MERGED_DIR

# Research-output directories on the laptop
FIGURE_DIR = WORK_ROOT / "figures"
RESULT_DIR = WORK_ROOT / "data"
NPZ_DIR = WORK_ROOT / "npz"
LOG_DIR = WORK_ROOT / "logs"
SLICE_DIR = WORK_ROOT / "slices"
FIELDLINE_DIR = WORK_ROOT / "fieldlines"
OPEN_CLOSED_DIR = WORK_ROOT / "open_closed"
TRACK_OPEN_DIR = WORK_ROOT / "track_crossings"

# Common output directories on the external drive
FULL_TRACK_OPEN_DIR = FULL_DATA_ROOT / "track_crossings"
FULL_SLICE_DIR = FULL_DATA_ROOT / "slices"
FULL_VECTOR_SLICE_DIR = FULL_DATA_ROOT / "slices_vector"
FULL_OPEN_CLOSED_DIR = FULL_DATA_ROOT / "open_closed"

# Explicit output choices
LOCAL_OUTPUT_PATH = RESULT_DIR
FULL_OUTPUT_PATH = FULL_TRACK_OPEN_DIR

# Safe defaults: partial laptop input and research output under WORK_ROOT.
# Override these in an individual script with the explicit constants above.
DATA_PATH = LOCAL_DATA_PATH
OUTPUT_PATH = LOCAL_OUTPUT_PATH


def local_data_path(*parts: str) -> Path:
    """Return a path inside the partial laptop dataset."""
    return LOCAL_DATA_ROOT.joinpath(*parts)


def full_data_path(*parts: str) -> Path:
    """Return a path inside the complete external-drive dataset."""
    return FULL_DATA_ROOT.joinpath(*parts)


def work_path(*parts: str) -> Path:
    """Return a path inside the research-output directory."""
    return WORK_ROOT.joinpath(*parts)
