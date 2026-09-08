"""
plot_sip_fieldlines_202609021207.py

3-D visualization of the merged SIP-IFVM magnetic field.

The script follows the data convention used by plot_merged_slices.py:

    data["r"]       : (Nr,)
    data["theta"]   : (Ntheta,)  colatitude [rad]
    data["phi"]     : (Nphi,)    longitude [rad]

    data["Br"]      : (Nr, Ntheta, Nphi)
    data["Btheta"]  : (Nr, Ntheta, Nphi)
    data["Bphi"]    : (Nr, Ntheta, Nphi)

Main plot:
    1. Draw the inner spherical shell with the precomputed open/closed map
       and overlay the smoothed Br=0 contour.
    2. Select magnetic-field-line seed points on an independently chosen radial shell.
    3. Build a 3-D PyVista StructuredGrid from the merged spherical grid.
    4. Convert (Br, Btheta, Bphi) -> global Cartesian B = (Bx, By, Bz),
       then trace magnetic field lines.
    5. Plot the topology-colored solar surface, transparent seed shell, and
       classified magnetic field lines.

Important:
    This version reads Br, Btheta, Bphi and converts them to GLOBAL
    Cartesian magnetic-field components internally before streamline
    tracing.

Requirements:
    matplotlib==3.10.8
    numpy==1.26.4
    pyvista==0.46.3

The script assumes that read_merged_data.py is available in the same
environment and provides:

    from read_merged_data import read_merged_physics

Author-facing configuration is in the CONFIGURATION section below.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

from config import (
    FIELDLINE_DIR,
    GRID_FILE,
    LOCAL_MERGED_DIR,
    OPEN_CLOSED_DIR,
)
from read_merged_sip_data import (
    read_merged_physics,
    select_merged_data_files,
    simulation_hours_from_filename,
)

# ======================================================================
# CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# Data-processing mode
# ----------------------------------------------------------------------
#
# "single":
#     Process one specified HDF5 file inside DATA_DIR.
#
# "all":
#     Traverse DATA_DIR and process all files matching:
#         xx_yy_merged_spherical.h5
#
PROCESS_MODE = "single"

DATA_DIR = LOCAL_MERGED_DIR

# Used only when PROCESS_MODE = "single".
SINGLE_DATA_FILE = "84_00_merged_spherical.h5"

OUTPUT_DIR = FIELDLINE_DIR

SAVE_HTML = 1
SAVE_SCREENSHOT = 0
SHOW_PLOT = 0

HTML_FILENAME_FORMAT = (
    "fieldlines_time.{simulation_hours:.2f}.idr.{seed_radial_index}.html"
)

PNG_FILENAME_FORMAT = (
    "fieldlines_time.{simulation_hours:.2f}.idr.{seed_radial_index}.png"
)

# ----------------------------------------------------------------------
# Simulation time reference
# ----------------------------------------------------------------------

# Given:
#     82.00 h -> 2026-04-12 02:00
# therefore:
#      0.00 h -> 2026-04-08 16:00
SIMULATION_START_DATETIME = datetime(
    2026, 4, 8, 16, 0
)

# ----------------------------------------------------------------------
# Solar-surface topology shell
# ----------------------------------------------------------------------
SURFACE_RADIAL_INDEX = 0

OPEN_CLOSED_DATA_DIR = OPEN_CLOSED_DIR
OPEN_CLOSED_FILENAME_FORMAT = "open_closed_time.{simulation_hours:.2f}.npz"
OPEN_CLOSED_MAP_TAG = "rindex0"

OPEN_CLOSED_CMAP = ListedColormap(
    [
        "royalblue",
        "lightgray",
        "firebrick",
    ]
)
OPEN_CLOSED_CLIM = (-1.5, 1.5)
BR_ZERO_CONTOUR_COLOR = "black"
BR_ZERO_CONTOUR_LINE_WIDTH = 2.5
BR_ZERO_CONTOUR_RADIUS_FACTOR = 1.002

SURFACE_OPACITY = 1.0
SHOW_SURFACE_EDGES = False

# ----------------------------------------------------------------------
# Seed-shell selection
# ----------------------------------------------------------------------
SEED_RADIAL_MODE = "index"
SEED_RADIAL_INDEX = 0
SEED_RADIAL = 10.0
SEED_RADIUS_FACTOR = 1.0

# ----------------------------------------------------------------------
# Seed-point selection
# ----------------------------------------------------------------------
# Available:
#   "strong_br"
#   "uniform_latlon"
#   "uniform_latlon_region"
#   "uniform_grid"
#   "latitude_ring"
#   "manual"
#   "line_between_points"
#   "neighbor_rings"
SEED_MODE = "line_between_points"
SEED_MODE_2 = None

N_SEEDS = 100

SEED_BR_PERCENTILE = 80.0
MIN_SEED_ANGULAR_DISTANCE_DEG = 10.0

UNIFORM_N_LAT = 8
UNIFORM_N_LON = 16
UNIFORM_LAT_MIN_DEG = -75.0
UNIFORM_LAT_MAX_DEG = 75.0

REGION_N_LAT = 11
REGION_N_LON = 11
REGION_LAT_MIN_DEG = 30.0
REGION_LAT_MAX_DEG = 50.0
REGION_LON_MIN_DEG = 40.0
REGION_LON_MAX_DEG = 60.0

GRID_SEED_THETA_STEP = 8
GRID_SEED_PHI_STEP = 12

RANDOM_SEED = 12345

RING_LATITUDE_DEG = 0.0
RING_N_SEEDS = 36

# ----------------------------------------------------------------------
# Seed-coordinate time mode: applies to ALL SEED_MODE choices
# ----------------------------------------------------------------------
#
# "direct":
#     Generate seed points using SEED_MODE directly for the CURRENT file.
#
# "diffrot_from_82h":
#     First generate seed points using exactly the same SEED_MODE on the
#     reference data at REFERENCE_SEED_TIME_HOURS, then evolve only their
#     longitudes to the current file time using latitude-dependent
#     differential rotation.
#
# This applies to ALL existing seed modes:
#     strong_br
#     uniform_latlon
#     uniform_latlon_region
#     uniform_grid
#     latitude_ring
#     manual
#     line_between_points
#     neighbor_rings
#
SEED_COORDINATE_TIME_MODE = "direct"

# Reference time/data used when SEED_COORDINATE_TIME_MODE="diffrot_from_82h".
REFERENCE_SEED_TIME_HOURS = 82.00

REFERENCE_SEED_FILE = DATA_DIR / "82_00_merged_spherical.h5"

# Manual mode input coordinates.
# In "direct" mode these are interpreted at the current file time.
# In "diffrot_from_82h" mode these are interpreted at the reference time.
MANUAL_SEEDS = [
    (20.0, 30.0),
    (-20.0, 210.0),
]

# ----------------------------------------------------------------------
# line_between_points mode
# ----------------------------------------------------------------------
# Endpoints are (latitude_deg, longitude_deg).
# Seeds are linearly interpolated in the latitude-longitude plane,
# including both endpoints.
# LINE_ENDPOINT_1 = (47.0, 42.0) # 93.50 h
# LINE_ENDPOINT_2 = (38.0, 47.0) # 93.50 h
LINE_ENDPOINT_1 = (-3.0, 40.0) # 84.00 h
LINE_ENDPOINT_2 = (23.0, 47.0) # 84.00 h
LINE_N_SEEDS = 15

# Besides the original line points, also add copies whose longitudes are
# shifted by these offsets [deg].
LINE_LONGITUDE_OFFSETS_DEG = [1.0, -1.0]

# ----------------------------------------------------------------------
# neighbor_rings mode
# ----------------------------------------------------------------------
# Center is (latitude_deg, longitude_deg).
# Rings are circles in the theta-phi / latitude-longitude plane.
NEIGHBOR_CENTER = (30.0, 45.0)

# Example: two rings at 0.5 deg and 1.0 deg from the center.
NEIGHBOR_RING_RADII_DEG = [0.5, 1.0]

# Example: 30 deg -> 12 points per ring.
NEIGHBOR_RING_AZIMUTH_STEP_DEG = 30.0

# Differential-rotation law [deg/day]:
#     Omega(lat) = A + B sin^2(lat) + C sin^4(lat)
DIFFROT_A_DEG_PER_DAY = 14.713
DIFFROT_B_DEG_PER_DAY = -2.396
DIFFROT_C_DEG_PER_DAY = -1.787

# Sign convention for longitude evolution.
DIFFROT_LONGITUDE_SIGN = +1.0

# ----------------------------------------------------------------------
# Field-line integration
# ----------------------------------------------------------------------
INTEGRATION_DIRECTION = "both"

INITIAL_STEP_LENGTH = 0.20
MIN_STEP_LENGTH = 0.01
MAX_STEP_LENGTH = 0.50
MAX_STEPS = 5000

MAX_LENGTH = None
TERMINAL_SPEED = 1.0e-12

DRAW_FIELDLINES_AS_TUBES = True
FIELDLINE_TUBE_RADIUS_FACTOR = 0.001

FIELDLINE_COLOR = "black"
FIELDLINE_OPACITY = 0.90

SHOW_SEED_POINTS = True
SEED_POINT_COLOR = "gold"
SEED_POINT_SIZE = 10

# ----------------------------------------------------------------------
# Seed-shell visualization
# ----------------------------------------------------------------------
SHOW_SEED_SHELL = True
SEED_SHELL_COLOR = "yellow"
SEED_SHELL_OPACITY = 0.15
SEED_SHELL_SHOW_EDGES = False

# ----------------------------------------------------------------------
# Magnetic-field-line classification colors
# ----------------------------------------------------------------------
OPEN_POSITIVE_COLOR = "red"
OPEN_NEGATIVE_COLOR = "blue"
CLOSED_FIELD_COLOR = "black"

OPEN_OUTER_RADIUS_FRACTION = 0.995
OPEN_DISTANCE_LIMIT_FACTOR = 15.0

# ----------------------------------------------------------------------
# Solar rotation axis
# ----------------------------------------------------------------------
SHOW_ROTATION_AXIS = True
ROTATION_AXIS_COLOR = "dimgray"
ROTATION_AXIS_RADIUS_FACTOR = 1.35
ROTATION_AXIS_LINE_WIDTH = 4.0

ROTATION_ARROW_LENGTH_FACTOR = 0.30
ROTATION_ARROW_COLOR = "dimgray"

NORTH_LABEL = "N"
NORTH_LABEL_FONT_SIZE = 18
NORTH_LABEL_OFFSET_FACTOR = 0.10

# ----------------------------------------------------------------------
# Plot window
# ----------------------------------------------------------------------
BACKGROUND = "white"
WINDOW_SIZE = (1500, 1100)
SHOW_AXES = True
SHOW_GRID_BOUNDS = False


# ======================================================================
# Filename time conversion
# ======================================================================


def simulation_datetime_from_filename(
    filename: Path,
    start_datetime: datetime = SIMULATION_START_DATETIME,
) -> datetime:
    """
    Convert simulation time encoded in filename to calendar datetime.
    """
    simulation_hours = simulation_hours_from_filename(
        filename
    )

    return (
        start_datetime
        + timedelta(hours=simulation_hours)
    )


def format_simulation_datetime(filename: Path) -> str:
    """
    Return the text used in the PyVista annotation.
    """
    simulation_hours = simulation_hours_from_filename(
        filename
    )

    simulation_datetime = simulation_datetime_from_filename(
        filename
    )

    return (
        f"{simulation_datetime:%Y-%m-%d %H:%M}"
        f"  ({simulation_hours:.2f} h)"
    )


# ======================================================================
# Utility functions
# ======================================================================

def validate_data(data):
    """
    Validate the merged-data arrays needed for field-line tracing.
    """
    required = (
        "r",
        "theta",
        "phi",
        "Br",
        "Btheta",
        "Bphi",
    )

    for key in required:
        if key not in data:
            raise KeyError(
                f'Missing "{key}". Available keys: {list(data.keys())}'
            )

    r = np.asarray(data["r"])
    theta = np.asarray(data["theta"])
    phi = np.asarray(data["phi"])

    expected = (len(r), len(theta), len(phi))

    for key in ("Br", "Btheta", "Bphi"):
        arr = np.asarray(data[key])
        if arr.shape != expected:
            raise ValueError(
                f'{key}.shape={arr.shape}, expected {expected}'
            )

    if len(r) < 2:
        raise ValueError(
            "At least two radial layers are required for 3-D streamline tracing."
        )

    if not np.all(np.diff(r) > 0.0):
        print(
            "WARNING: r is not strictly increasing. "
            "Structured-grid interpolation may be unreliable."
        )


def spherical_to_cartesian(r, theta, phi):
    """
    Spherical coordinate -> Cartesian coordinate.

    theta : colatitude [rad]
    phi   : longitude [rad]
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z



def spherical_vector_to_cartesian(
    br,
    btheta,
    bphi,
    theta,
    phi,
):
    """
    Convert spherical vector components (Br, Btheta, Bphi) to
    global Cartesian components (Bx, By, Bz).

    Basis vectors:
        e_r     = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
        e_theta = (cos(theta) cos(phi), cos(theta) sin(phi), -sin(theta))
        e_phi   = (-sin(phi),           cos(phi),            0)

    where theta is colatitude.
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    bx = (
        br * sin_theta * cos_phi
        + btheta * cos_theta * cos_phi
        - bphi * sin_phi
    )

    by = (
        br * sin_theta * sin_phi
        + btheta * cos_theta * sin_phi
        + bphi * cos_phi
    )

    bz = (
        br * cos_theta
        - btheta * sin_theta
    )

    return bx, by, bz


def wrap_phi_axis(phi, *arrays):
    """
    Close the phi=0 / 2*pi seam by appending the first longitude column.

    Parameters
    ----------
    phi : ndarray, shape (Nphi,)

    arrays : ndarray
        Arrays whose last axis is phi:
            (..., Nphi)

    Returns
    -------
    phi_wrap
        Shape (Nphi+1,)

    arrays_wrap
        Each input array with its first phi column appended to the end.
    """
    phi = np.asarray(phi, dtype=float)

    phi_wrap = np.concatenate(
        (phi, [phi[0] + 2.0 * np.pi])
    )

    wrapped = [
        np.concatenate((a, a[..., :1]), axis=-1)
        for a in arrays
    ]

    return phi_wrap, wrapped


# ======================================================================
# PyVista grid construction
# ======================================================================

def build_magnetic_structured_grid(data):
    """
    Build the complete 3-D curvilinear StructuredGrid.

    The phi seam is explicitly closed by duplicating the first longitude
    column at phi + 2*pi.

    This version reads Br, Btheta, Bphi and internally converts them to
    global Cartesian Bx, By, Bz for streamline tracing.

    Returns
    -------
    grid : pyvista.StructuredGrid
        Contains point-data vector array "B".

    wrapped : dict
        Wrapped coordinate/data arrays used to construct the grid.
    """
    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)

    br = np.asarray(data["Br"], dtype=float)
    btheta = np.asarray(data["Btheta"], dtype=float)
    bphi = np.asarray(data["Bphi"], dtype=float)

    phi_wrap, (br_w, btheta_w, bphi_w) = wrap_phi_axis(
        phi,
        br,
        btheta,
        bphi,
    )

    rr, tt, pp = np.meshgrid(
        r,
        theta,
        phi_wrap,
        indexing="ij",
    )

    x, y, z = spherical_to_cartesian(
        rr,
        tt,
        pp,
    )

    bx_w, by_w, bz_w = spherical_vector_to_cartesian(
        br_w,
        btheta_w,
        bphi_w,
        tt,
        pp,
    )

    grid = pv.StructuredGrid(
        x,
        y,
        z,
    )

    b_vectors = np.column_stack(
        (
            bx_w.ravel(order="F"),
            by_w.ravel(order="F"),
            bz_w.ravel(order="F"),
        )
    )

    grid.point_data["B"] = b_vectors
    grid.set_active_vectors("B")

    return grid, {
        "r": r,
        "theta": theta,
        "phi": phi_wrap,
        "x": x,
        "y": y,
        "z": z,
        "Br": br_w,
        "Btheta": btheta_w,
        "Bphi": bphi_w,
        "Bx": bx_w,
        "By": by_w,
        "Bz": bz_w,
    }


def load_open_closed_surface_data(data, simulation_hours):
    """Load and validate the precomputed inner-boundary topology map."""
    filename = OPEN_CLOSED_DATA_DIR / OPEN_CLOSED_FILENAME_FORMAT.format(
        simulation_hours=simulation_hours,
    )

    if not filename.is_file():
        raise FileNotFoundError(
            "Precomputed open/closed result not found: "
            f"{filename}"
        )

    map_key = f"{OPEN_CLOSED_MAP_TAG}_open_closed_map"

    with np.load(filename) as result:
        required = {
            "theta",
            "phi",
            "Br_inner_contour_smooth",
            "inner_radius",
            map_key,
        }
        missing = sorted(required.difference(result.files))

        if missing:
            raise KeyError(
                f"Missing arrays in {filename}: {missing}"
            )

        theta_map = np.asarray(result["theta"], dtype=float)
        phi_map = np.asarray(result["phi"], dtype=float)
        open_closed_map = np.asarray(result[map_key], dtype=np.int8)
        br_contour = np.asarray(
            result["Br_inner_contour_smooth"],
            dtype=float,
        )
        inner_radius = float(result["inner_radius"])

    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    radius = float(np.asarray(data["r"], dtype=float)[SURFACE_RADIAL_INDEX])
    expected_shape = (theta.size, phi.size)

    if open_closed_map.shape != expected_shape:
        raise ValueError(
            f"{map_key}.shape={open_closed_map.shape}, "
            f"expected {expected_shape}."
        )

    if br_contour.shape != expected_shape:
        raise ValueError(
            f"Br_inner_contour_smooth.shape={br_contour.shape}, "
            f"expected {expected_shape}."
        )

    if not np.allclose(theta_map, theta) or not np.allclose(phi_map, phi):
        raise ValueError(
            "The open/closed map angular grid does not match the magnetic data."
        )

    if not np.isclose(inner_radius, radius):
        raise ValueError(
            f"The open/closed map radius ({inner_radius}) does not match "
            f"the displayed surface radius ({radius})."
        )

    if not np.all(np.isfinite(open_closed_map)):
        raise ValueError("The open/closed map contains NaN or Inf values.")

    if not set(np.unique(open_closed_map)).issubset({-1, 0, 1}):
        raise ValueError(
            "The open/closed map contains labels other than -1, 0, and +1."
        )

    if not np.all(np.isfinite(br_contour)):
        raise ValueError("The smoothed Br surface contains NaN or Inf values.")

    return filename, open_closed_map, br_contour


def build_open_closed_surface(data, open_closed_map, br_contour):
    """
    Build the inner spherical surface as a PyVista StructuredGrid
    and attach the open/closed labels and smoothed Br contour values.
    """
    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)

    idx = SURFACE_RADIAL_INDEX

    if idx < 0:
        idx = len(r) + idx

    if idx < 0 or idx >= len(r):
        raise IndexError(
            f"SURFACE_RADIAL_INDEX={SURFACE_RADIAL_INDEX} outside "
            f"0..{len(r)-1}"
        )

    radius = float(r[idx])
    phi_wrap, (map_wrap, br_contour_wrap) = wrap_phi_axis(
        phi,
        open_closed_map,
        br_contour,
    )

    tt, pp = np.meshgrid(
        theta,
        phi_wrap,
        indexing="ij",
    )

    rr = np.full_like(tt, radius)

    x, y, z = spherical_to_cartesian(
        rr,
        tt,
        pp,
    )

    surface = pv.StructuredGrid(
        x,
        y,
        z,
    )

    surface.point_data["OpenClosed"] = map_wrap.ravel(order="F")
    surface.point_data["BrContour"] = br_contour_wrap.ravel(order="F")

    br_zero_contour = surface.contour(
        isosurfaces=[0.0],
        scalars="BrContour",
    )

    if br_zero_contour.n_points > 0:
        br_zero_contour.points *= BR_ZERO_CONTOUR_RADIUS_FACTOR

    return surface, br_zero_contour, radius




def resolve_seed_shell(data):
    """
    Resolve the seed-shell location.

    Returns
    -------
    br_index : int
        Radial index used to read Br for Br-based seed modes.

    seed_radius : float
        Actual radius of seed points / transparent yellow seed shell.

    radial_mode : str
        "index" or "value".
    """
    r = np.asarray(data["r"], dtype=float)

    radial_mode = SEED_RADIAL_MODE.lower()

    if radial_mode == "index":
        idx = SEED_RADIAL_INDEX

        if idx < 0:
            idx = len(r) + idx

        if idx < 0 or idx >= len(r):
            raise IndexError(
                f"SEED_RADIAL_INDEX={SEED_RADIAL_INDEX} outside "
                f"0..{len(r)-1}"
            )

        seed_radius = float(r[idx]) * SEED_RADIUS_FACTOR

        return idx, seed_radius, radial_mode

    if radial_mode == "value":
        seed_radial_value = float(SEED_RADIAL)

        if not np.isfinite(seed_radial_value):
            raise ValueError("SEED_RADIAL must be finite.")

        if (
            seed_radial_value < float(r.min())
            or seed_radial_value > float(r.max())
        ):
            raise ValueError(
                f"SEED_RADIAL={seed_radial_value} is outside "
                f"the radial range [{float(r.min())}, {float(r.max())}]."
            )

        idx = int(
            np.argmin(
                np.abs(r - seed_radial_value)
            )
        )

        seed_radius = seed_radial_value * SEED_RADIUS_FACTOR

        return idx, seed_radius, radial_mode

    raise ValueError(
        f"Unknown SEED_RADIAL_MODE={SEED_RADIAL_MODE!r}. "
        'Use "index" or "value".'
    )


def build_seed_shell_surface(data):
    """
    Build the spherical shell used for seed-point selection.

    This shell is independent of the displayed Br surface at r_index=0.
    It is plotted only as a transparent yellow reference surface.
    """
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)

    idx, radius, radial_mode = resolve_seed_shell(
        data
    )

    phi_wrap = np.concatenate(
        (
            phi,
            [phi[0] + 2.0 * np.pi],
        )
    )

    tt, pp = np.meshgrid(
        theta,
        phi_wrap,
        indexing="ij",
    )

    rr = np.full_like(
        tt,
        radius,
        dtype=float,
    )

    x, y, z = spherical_to_cartesian(
        rr,
        tt,
        pp,
    )

    seed_shell = pv.StructuredGrid(
        x,
        y,
        z,
    )

    return seed_shell, radius


# ======================================================================
# Seed selection
# ======================================================================

def angular_distance(unit_vector, selected_vectors):
    """Angular distance [rad] to selected unit vectors."""
    dots = np.clip(
        selected_vectors @ unit_vector,
        -1.0,
        1.0,
    )
    return np.arccos(dots)


def get_seed_shell(data):
    """
    Return:
        seed_index,
        seed_radius,
        Br on the seed shell.

    This is independent of SURFACE_RADIAL_INDEX.
    """
    br = np.asarray(data["Br"], dtype=float)

    idx, seed_radius, radial_mode = resolve_seed_shell(
        data
    )

    return (
        idx,
        seed_radius,
        br[idx, :, :],
    )


def latlon_to_cartesian(
    radius,
    latitude_deg,
    longitude_deg,
):
    """Latitude/longitude [deg] -> Cartesian point."""
    latitude = np.radians(latitude_deg)
    longitude = (
        np.radians(longitude_deg)
        % (2.0 * np.pi)
    )

    theta = (
        np.pi / 2.0
        - latitude
    )

    x, y, z = spherical_to_cartesian(
        radius,
        theta,
        longitude,
    )

    return np.array(
        [x, y, z],
        dtype=float,
    )

def cartesian_to_latlon(
    point,
):
    """
    Cartesian point -> (latitude_deg, longitude_deg).
    """
    point = np.asarray(
        point,
        dtype=float,
    )

    radius = float(
        np.linalg.norm(point)
    )

    if radius <= 0.0:
        raise ValueError(
            "Cannot convert zero-radius point to latitude/longitude."
        )

    x, y, z = point

    latitude_deg = np.degrees(
        np.arcsin(
            np.clip(
                z / radius,
                -1.0,
                1.0,
            )
        )
    )

    longitude_deg = (
        np.degrees(
            np.arctan2(
                y,
                x,
            )
        )
        % 360.0
    )

    return (
        float(latitude_deg),
        float(longitude_deg),
    )


def differential_rotation_rate_deg_per_day(
    latitude_deg,
):
    """
    Latitude-dependent empirical differential-rotation rate [deg/day].

    Omega(lat) = A + B sin^2(lat) + C sin^4(lat)
    """
    lat_rad = np.radians(
        latitude_deg
    )

    s2 = np.sin(
        lat_rad
    ) ** 2

    return (
        DIFFROT_A_DEG_PER_DAY
        + DIFFROT_B_DEG_PER_DAY * s2
        + DIFFROT_C_DEG_PER_DAY * s2**2
    )


def evolve_seed_longitude_from_reference(
    latitude_deg,
    longitude_deg_at_reference,
    current_simulation_hours,
):
    """
    Evolve one longitude from REFERENCE_SEED_TIME_HOURS to the current
    simulation time.
    """
    delta_t_days = (
        float(current_simulation_hours)
        - float(REFERENCE_SEED_TIME_HOURS)
    ) / 24.0

    omega_lat = differential_rotation_rate_deg_per_day(
        latitude_deg
    )

    delta_lon = (
        DIFFROT_LONGITUDE_SIGN
        * omega_lat
        * delta_t_days
    )

    return (
        float(longitude_deg_at_reference)
        + delta_lon
    ) % 360.0


def rotate_seed_xyz_from_reference(
    reference_xyz,
    current_seed_radius,
    current_simulation_hours,
):
    """
    Apply differential rotation to an arbitrary list of reference seed
    coordinates.

    Latitude is kept fixed. Only longitude is evolved.
    All output points are placed on the current seed shell radius.
    """
    reference_xyz = np.asarray(
        reference_xyz,
        dtype=float,
    )

    rotated = []

    for point in reference_xyz:
        lat_deg, lon_ref_deg = cartesian_to_latlon(
            point
        )

        lon_now_deg = evolve_seed_longitude_from_reference(
            latitude_deg=lat_deg,
            longitude_deg_at_reference=lon_ref_deg,
            current_simulation_hours=current_simulation_hours,
        )

        rotated.append(
            latlon_to_cartesian(
                current_seed_radius,
                lat_deg,
                lon_now_deg,
            )
        )

    return np.asarray(
        rotated,
        dtype=float,
    )
def select_strong_br_seeds(data):
    """
    1. Strong-|Br| seeds on SEED_RADIAL_INDEX, with minimum
       angular separation.
    """
    theta = np.asarray(
        data["theta"],
        dtype=float,
    )
    phi = np.asarray(
        data["phi"],
        dtype=float,
    )

    idx, seed_radius, br_shell = get_seed_shell(
        data
    )

    tt, pp = np.meshgrid(
        theta,
        phi,
        indexing="ij",
    )

    br_flat = br_shell.ravel()
    th_flat = tt.ravel()
    ph_flat = pp.ravel()

    finite = np.isfinite(
        br_flat
    )

    threshold = np.percentile(
        np.abs(br_flat[finite]),
        SEED_BR_PERCENTILE,
    )

    candidates = np.flatnonzero(
        finite
        & (np.abs(br_flat) >= threshold)
    )

    candidates = candidates[
        np.argsort(
            np.abs(br_flat[candidates])
        )[::-1]
    ]

    min_angle = np.radians(
        MIN_SEED_ANGULAR_DISTANCE_DEG
    )

    selected = []

    for ind in candidates:
        th = th_flat[ind]
        ph = ph_flat[ind]

        unit = np.array(
            [
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th),
            ],
            dtype=float,
        )

        if selected:
            if np.min(
                angular_distance(
                    unit,
                    np.asarray(selected),
                )
            ) < min_angle:
                continue

        selected.append(
            unit
        )

        if len(selected) >= N_SEEDS:
            break

    if not selected:
        raise RuntimeError(
            "No strong-Br seed points were selected."
        )

    print(
        "\nStrong-|Br| seeds:"
        f"\n  r_index = {idx}"
        f"\n  radius  = {seed_radius:.8g}"
        f"\n  N       = {len(selected)}"
    )

    return (
        seed_radius
        * np.asarray(selected)
    )


def select_uniform_latlon_seeds(data):
    """
    3. Regular latitude-longitude seed grid.
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    latitudes = np.linspace(
        UNIFORM_LAT_MIN_DEG,
        UNIFORM_LAT_MAX_DEG,
        UNIFORM_N_LAT,
    )

    longitudes = np.linspace(
        0.0,
        360.0,
        UNIFORM_N_LON,
        endpoint=False,
    )

    seeds = [
        latlon_to_cartesian(
            seed_radius,
            lat,
            lon,
        )
        for lat in latitudes
        for lon in longitudes
    ]

    print(
        "\nUniform latitude-longitude seeds:"
        f"\n  r_index = {idx}"
        f"\n  radius  = {seed_radius:.8g}"
        f"\n  N       = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )




def select_uniform_latlon_region_seeds(data):
    """
    4. Regular latitude-longitude seeds only inside a specified region.

    Region limits are controlled by:
        REGION_LAT_MIN_DEG
        REGION_LAT_MAX_DEG
        REGION_LON_MIN_DEG
        REGION_LON_MAX_DEG

    and the number of seed rows/columns by:
        REGION_N_LAT
        REGION_N_LON
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    if REGION_LAT_MIN_DEG > REGION_LAT_MAX_DEG:
        raise ValueError(
            "REGION_LAT_MIN_DEG must be <= REGION_LAT_MAX_DEG."
        )

    if REGION_N_LAT < 1 or REGION_N_LON < 1:
        raise ValueError(
            "REGION_N_LAT and REGION_N_LON must both be >= 1."
        )

    latitudes = np.linspace(
        REGION_LAT_MIN_DEG,
        REGION_LAT_MAX_DEG,
        REGION_N_LAT,
    )

    # Longitude is generated directly over the requested interval.
    # Values outside 0..360 are still valid because
    # latlon_to_cartesian() wraps longitude periodically.
    longitudes = np.linspace(
        REGION_LON_MIN_DEG,
        REGION_LON_MAX_DEG,
        REGION_N_LON,
    )

    seeds = [
        latlon_to_cartesian(
            seed_radius,
            lat,
            lon,
        )
        for lat in latitudes
        for lon in longitudes
    ]

    print(
        "\nUniform latitude-longitude regional seeds:"
        f"\n  r_index      = {idx}"
        f"\n  radius       = {seed_radius:.8g}"
        f"\n  latitude     = "
        f"[{REGION_LAT_MIN_DEG:.2f}, {REGION_LAT_MAX_DEG:.2f}] deg"
        f"\n  longitude    = "
        f"[{REGION_LON_MIN_DEG:.2f}, {REGION_LON_MAX_DEG:.2f}] deg"
        f"\n  N_lat x N_lon= {REGION_N_LAT} x {REGION_N_LON}"
        f"\n  N            = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )


def select_uniform_grid_seeds(data):
    """
    5. Seeds selected directly from the native theta/phi grid,
       every GRID_SEED_THETA_STEP and GRID_SEED_PHI_STEP points.
    """
    theta = np.asarray(
        data["theta"],
        dtype=float,
    )
    phi = np.asarray(
        data["phi"],
        dtype=float,
    )

    idx, seed_radius, _ = get_seed_shell(
        data
    )

    seeds = []

    for j in range(
        0,
        len(theta),
        GRID_SEED_THETA_STEP,
    ):
        for k in range(
            0,
            len(phi),
            GRID_SEED_PHI_STEP,
        ):
            x, y, z = spherical_to_cartesian(
                seed_radius,
                theta[j],
                phi[k],
            )

            seeds.append(
                (float(x), float(y), float(z))
            )

    print(
        "\nUniform native-grid seeds:"
        f"\n  r_index = {idx}"
        f"\n  radius  = {seed_radius:.8g}"
        f"\n  N       = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )


def select_latitude_ring_seeds(data):
    """
    7. Uniform seeds on one latitude ring.
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    longitudes = np.linspace(
        0.0,
        360.0,
        RING_N_SEEDS,
        endpoint=False,
    )

    seeds = [
        latlon_to_cartesian(
            seed_radius,
            RING_LATITUDE_DEG,
            lon,
        )
        for lon in longitudes
    ]

    print(
        "\nLatitude-ring seeds:"
        f"\n  r_index = {idx}"
        f"\n  radius  = {seed_radius:.8g}"
        f"\n  latitude= {RING_LATITUDE_DEG:.2f} deg"
        f"\n  N       = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )



def select_line_between_points_seeds(data):
    """
    Select seeds along the straight line between two specified points
    in the latitude-longitude plane.

    LINE_ENDPOINT_1 / LINE_ENDPOINT_2:
        (latitude_deg, longitude_deg)

    LINE_N_SEEDS:
        total number of points on the central line, including both
        endpoints.

    In addition to those central-line points, this mode also adds
    copies whose longitudes are shifted by the values in
    LINE_LONGITUDE_OFFSETS_DEG.

    Example:
        central line
        + all points shifted by +0.5 deg in longitude
        + all points shifted by -0.5 deg in longitude
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    if LINE_N_SEEDS < 2:
        raise ValueError(
            "LINE_N_SEEDS must be >= 2."
        )

    lat1 = float(LINE_ENDPOINT_1[0])
    lon1 = float(LINE_ENDPOINT_1[1])
    lat2 = float(LINE_ENDPOINT_2[0])
    lon2 = float(LINE_ENDPOINT_2[1])

    if not (
        -90.0 <= lat1 <= 90.0
        and -90.0 <= lat2 <= 90.0
    ):
        raise ValueError(
            "Endpoint latitude must be within [-90, 90] deg."
        )

    offsets_deg = [
        float(value)
        for value in LINE_LONGITUDE_OFFSETS_DEG
    ]

    # Shortest periodic longitude difference.
    dlon = (
        (lon2 - lon1 + 180.0)
        % 360.0
        - 180.0
    )

    fraction = np.linspace(
        0.0,
        1.0,
        LINE_N_SEEDS,
    )

    latitudes = (
        lat1
        + fraction * (lat2 - lat1)
    )

    longitudes_center = (
        lon1
        + fraction * dlon
    ) % 360.0

    seeds = []

    # 1) Original central line.
    for lat, lon in zip(
        latitudes,
        longitudes_center,
    ):
        seeds.append(
            latlon_to_cartesian(
                seed_radius,
                lat,
                lon,
            )
        )

    # 2) Longitude-shifted copies.
    for dlon_offset in offsets_deg:
        longitudes_shifted = (
            longitudes_center
            + dlon_offset
        ) % 360.0

        for lat, lon in zip(
            latitudes,
            longitudes_shifted,
        ):
            seeds.append(
                latlon_to_cartesian(
                    seed_radius,
                    lat,
                    lon,
                )
            )

    print(
        "\nLine-between-points seeds:"
        f"\n  r_index              = {idx}"
        f"\n  radius               = {seed_radius:.8g}"
        f"\n  endpoint 1           = ({lat1:.4f}, {lon1 % 360.0:.4f}) deg"
        f"\n  endpoint 2           = ({lat2:.4f}, {lon2 % 360.0:.4f}) deg"
        f"\n  central-line N       = {LINE_N_SEEDS}"
        f"\n  longitude offsets    = {offsets_deg} deg"
        f"\n  copies per offset    = {LINE_N_SEEDS}"
        f"\n  total N              = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )
def select_neighbor_ring_seeds(data):
    """
    Select seeds on one or more circles around NEIGHBOR_CENTER in the
    theta-phi / latitude-longitude plane.

    For each ring radius R [deg]:

        delta_lon = R * cos(alpha)
        delta_lat = R * sin(alpha)

    with alpha sampled from 0 to 360 deg using
    NEIGHBOR_RING_AZIMUTH_STEP_DEG.

    Example:
        radii = [0.5, 1.0]
        azimuth step = 30 deg

    -> 12 points on each ring, total 24 points.

    This is intentionally a planar theta-phi circle, not a geodesic
    circle on the sphere.
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    center_lat = float(
        NEIGHBOR_CENTER[0]
    )

    center_lon = (
        float(NEIGHBOR_CENTER[1])
        % 360.0
    )

    if not (
        -90.0 <= center_lat <= 90.0
    ):
        raise ValueError(
            "NEIGHBOR_CENTER latitude must be within [-90, 90] deg."
        )

    radii = np.asarray(
        NEIGHBOR_RING_RADII_DEG,
        dtype=float,
    )

    if radii.ndim != 1 or radii.size == 0:
        raise ValueError(
            "NEIGHBOR_RING_RADII_DEG must be a non-empty 1-D list."
        )

    if np.any(
        ~np.isfinite(radii)
    ) or np.any(
        radii <= 0.0
    ):
        raise ValueError(
            "All neighbor-ring radii must be finite and > 0."
        )

    step = float(
        NEIGHBOR_RING_AZIMUTH_STEP_DEG
    )

    if not np.isfinite(step) or step <= 0.0 or step > 360.0:
        raise ValueError(
            "NEIGHBOR_RING_AZIMUTH_STEP_DEG must be in (0, 360]."
        )

    azimuth_deg = np.arange(
        0.0,
        360.0,
        step,
        dtype=float,
    )

    azimuth_rad = np.radians(
        azimuth_deg
    )

    seeds = []

    for ring_radius in radii:

        delta_lon = (
            ring_radius
            * np.cos(azimuth_rad)
        )

        delta_lat = (
            ring_radius
            * np.sin(azimuth_rad)
        )

        ring_lat = (
            center_lat
            + delta_lat
        )

        ring_lon = (
            center_lon
            + delta_lon
        ) % 360.0

        if np.any(
            (ring_lat < -90.0)
            | (ring_lat > 90.0)
        ):
            raise ValueError(
                "A neighbor-ring point leaves the valid latitude range "
                "[-90, 90] deg. Reduce radius or move the center."
            )

        for lat, lon in zip(
            ring_lat,
            ring_lon,
        ):
            seeds.append(
                latlon_to_cartesian(
                    seed_radius,
                    lat,
                    lon,
                )
            )

    print(
        "\nNeighbor-ring seeds:"
        f"\n  r_index      = {idx}"
        f"\n  radius       = {seed_radius:.8g}"
        f"\n  center       = ({center_lat:.4f}, {center_lon:.4f}) deg"
        f"\n  ring radii   = {radii.tolist()} deg"
        f"\n  azimuth step = {step:.4f} deg"
        f"\n  N per ring   = {len(azimuth_deg)}"
        f"\n  N total      = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )


def select_manual_seeds(data):
    """
    8. Manually specified latitude/longitude seeds.

    MANUAL_SEEDS is interpreted on whichever data/time is being used
    to generate the seed set:
        - current data in direct mode
        - reference data in diffrot_from_82h mode
    """
    idx, seed_radius, _ = get_seed_shell(
        data
    )

    seeds = [
        latlon_to_cartesian(
            seed_radius,
            lat,
            lon,
        )
        for lat, lon in MANUAL_SEEDS
    ]

    print(
        "\nManual seeds:"
        f"\n  r_index = {idx}"
        f"\n  radius  = {seed_radius:.8g}"
        f"\n  N       = {len(seeds)}"
    )

    return np.asarray(
        seeds,
        dtype=float,
    )


def _generate_seed_xyz_from_modes(
    data,
):
    """
    Generate seed coordinates from SEED_MODE / SEED_MODE_2 using the
    supplied data, without applying any time evolution.
    """
    seed_functions = {
        "strong_br": select_strong_br_seeds,
        "uniform_latlon": select_uniform_latlon_seeds,
        "uniform_latlon_region": select_uniform_latlon_region_seeds,
        "uniform_grid": select_uniform_grid_seeds,
        "latitude_ring": select_latitude_ring_seeds,
        "manual": select_manual_seeds,
        "line_between_points": select_line_between_points_seeds,
        "neighbor_rings": select_neighbor_ring_seeds,
    }

    def get_mode_xyz(mode):
        mode_key = str(
            mode
        ).lower()

        if mode_key not in seed_functions:
            raise ValueError(
                f"Unknown seed mode={mode!r}. "
                f"Available: {list(seed_functions)}"
            )

        return np.asarray(
            seed_functions[mode_key](
                data
            ),
            dtype=float,
        )

    xyz_list = [
        get_mode_xyz(
            SEED_MODE
        )
    ]

    if SEED_MODE_2 is not None:
        xyz_list.append(
            get_mode_xyz(
                SEED_MODE_2
            )
        )

    return np.vstack(
        xyz_list
    )


def build_seed_source(
    current_data,
    current_simulation_hours,
    reference_data=None,
):
    """
    Create the PyVista seed source.

    SEED_COORDINATE_TIME_MODE = "direct"
        Generate all seed modes directly from current_data.

    SEED_COORDINATE_TIME_MODE = "diffrot_from_82h"
        Generate the selected seed modes on reference_data at 82.00 h,
        then evolve every generated seed longitude to the current time.

    Therefore the time mode applies to ALL seed modes, including
    strong_br / uniform_* / latitude_ring / manual /
    line_between_points / neighbor_rings.
    """
    mode = str(
        SEED_COORDINATE_TIME_MODE
    ).strip().lower()

    current_seed_index, current_seed_radius, _ = resolve_seed_shell(
        current_data
    )

    if mode == "direct":
        xyz = _generate_seed_xyz_from_modes(
            current_data
        )

    elif mode == "diffrot_from_82h":
        if reference_data is None:
            raise ValueError(
                "reference_data is required when "
                'SEED_COORDINATE_TIME_MODE="diffrot_from_82h".'
            )

        reference_xyz = _generate_seed_xyz_from_modes(
            reference_data
        )

        xyz = rotate_seed_xyz_from_reference(
            reference_xyz=reference_xyz,
            current_seed_radius=current_seed_radius,
            current_simulation_hours=current_simulation_hours,
        )

        print(
            "\nDifferential-rotation seed evolution:"
            f"\n  reference time = {REFERENCE_SEED_TIME_HOURS:.2f} h"
            f"\n  current time   = {current_simulation_hours:.2f} h"
            f"\n  seed radius    = {current_seed_radius:.8g}"
            f"\n  total seeds    = {len(xyz)}"
        )

    else:
        raise ValueError(
            f"Unknown SEED_COORDINATE_TIME_MODE={SEED_COORDINATE_TIME_MODE!r}. "
            'Use "direct" or "diffrot_from_82h".'
        )

    print(
        "\nCombined seed source:"
        f"\n  coordinate time mode = {SEED_COORDINATE_TIME_MODE}"
        f"\n  mode 1               = {SEED_MODE}"
        f"\n  mode 2               = {SEED_MODE_2}"
        f"\n  total seeds          = {len(xyz)}"
    )

    return pv.PolyData(
        xyz
    )


# ======================================================================
# Field-line tracing and classification
# ======================================================================

def _truncate_points_at_open_radius(
    points,
    open_radius,
):
    """
    Truncate one streamline branch at the first point whose radius
    reaches or exceeds open_radius.

    The returned branch keeps that first boundary-crossing point so
    the line can still be classified as open.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Streamline points ordered from the seed outward along one
        integration direction.

    open_radius : float
        Radial distance at which tracing is considered open and the
        stored streamline is stopped.
    """
    if points is None:
        return None

    points = np.asarray(
        points,
        dtype=float,
    )

    radii = np.linalg.norm(
        points,
        axis=1,
    )

    hit = np.flatnonzero(
        radii >= open_radius
    )

    if hit.size == 0:
        return points

    stop_index = int(
        hit[0]
    )

    return points[
        : stop_index + 1
    ]


def _trace_one_direction(
    magnetic_grid,
    seed_point,
    direction,
    outer_radius,
    open_radius,
):
    """
    Trace one seed in one integration direction.

    Returns
    -------
    ndarray or None
        Polyline points ordered from the seed toward the integration
        direction. Returns None if no usable streamline is produced.
    """
    max_length = MAX_LENGTH

    if max_length is None:
        max_length = 4.0 * outer_radius

    source = pv.PolyData(
        np.asarray(
            [seed_point],
            dtype=float,
        )
    )

    stream = magnetic_grid.streamlines_from_source(
        source,
        vectors="B",
        integration_direction=direction,
        integrator_type=45,
        initial_step_length=INITIAL_STEP_LENGTH,
        step_unit="cl",
        min_step_length=MIN_STEP_LENGTH,
        max_step_length=MAX_STEP_LENGTH,
        max_steps=MAX_STEPS,
        terminal_speed=TERMINAL_SPEED,
        max_length=max_length,
        compute_vorticity=False,
        interpolator_type="point",
    )

    if stream.n_cells == 0 or stream.n_points < 2:
        return None

    # Normally one seed + one direction gives one polyline.
    # If VTK returns more than one cell, keep the longest one.
    best_points = None
    best_length = -1.0

    for cell_id in range(stream.n_cells):
        cell = stream.get_cell(cell_id)

        ids = np.asarray(
            cell.point_ids,
            dtype=int,
        )

        if len(ids) < 2:
            continue

        pts = stream.points[ids]

        length = float(
            np.sum(
                np.linalg.norm(
                    np.diff(pts, axis=0),
                    axis=1,
                )
            )
        )

        if length > best_length:
            best_length = length
            best_points = np.asarray(
                pts,
                dtype=float,
            )

    if best_points is None:
        return None

    # Ensure the first point is the one nearest the supplied seed.
    d0 = np.linalg.norm(
        best_points[0] - seed_point
    )
    d1 = np.linalg.norm(
        best_points[-1] - seed_point
    )

    if d1 < d0:
        best_points = best_points[::-1]

    best_points = _truncate_points_at_open_radius(
        best_points,
        open_radius,
    )

    return best_points


def _combine_forward_backward(
    seed_point,
    backward_points,
    forward_points,
):
    """
    Combine backward and forward streamline branches into one full line.
    """
    branches = []

    if backward_points is not None:
        # backward_points starts at seed -> backward endpoint
        # reverse it so it becomes endpoint -> seed
        branches.append(
            backward_points[::-1]
        )

    if forward_points is not None:
        if branches:
            # Avoid duplicating the seed point.
            branches.append(
                forward_points[1:]
            )
        else:
            branches.append(
                forward_points
            )

    if not branches:
        return None

    full_points = np.vstack(
        branches
    )

    if len(full_points) < 2:
        return None

    return full_points


def _cartesian_to_theta_phi(point):
    """
    Cartesian point -> (theta, phi), with theta as colatitude.
    """
    x, y, z = point

    radius = np.linalg.norm(
        point
    )

    if radius <= 0.0:
        raise ValueError(
            "Cannot convert zero-radius point to spherical coordinates."
        )

    theta = np.arccos(
        np.clip(
            z / radius,
            -1.0,
            1.0,
        )
    )

    phi = np.arctan2(
        y,
        x,
    ) % (2.0 * np.pi)

    return theta, phi


def _nearest_periodic_phi_index(phi_array, phi_target):
    """
    Nearest phi index with 2*pi periodicity.
    """
    delta = np.angle(
        np.exp(
            1j * (
                phi_array
                - phi_target
            )
        )
    )

    return int(
        np.argmin(
            np.abs(delta)
        )
    )


def get_surface_br_at_point(
    data,
    point,
):
    """
    Obtain Br on the displayed solar surface (r_index=0) nearest to
    a Cartesian footpoint direction.
    """
    theta = np.asarray(
        data["theta"],
        dtype=float,
    )

    phi = np.asarray(
        data["phi"],
        dtype=float,
    )

    br = np.asarray(
        data["Br"],
        dtype=float,
    )

    theta_p, phi_p = _cartesian_to_theta_phi(
        point
    )

    j = int(
        np.argmin(
            np.abs(
                theta
                - theta_p
            )
        )
    )

    k = _nearest_periodic_phi_index(
        phi,
        phi_p,
    )

    return float(
        br[SURFACE_RADIAL_INDEX, j, k]
    )


def get_seed_br_at_point(
    data,
    seed_point,
):
    """
    Return Br corresponding to the seed-point direction.

    For SEED_RADIAL_MODE="index":
        Br is read from r[SEED_RADIAL_INDEX].

    For SEED_RADIAL_MODE="value":
        Br is read from the nearest radial grid layer to SEED_RADIAL,
        consistent with resolve_seed_shell().

    The theta/phi location is chosen by nearest native grid point.
    """
    theta = np.asarray(
        data["theta"],
        dtype=float,
    )

    phi = np.asarray(
        data["phi"],
        dtype=float,
    )

    br = np.asarray(
        data["Br"],
        dtype=float,
    )

    br_index, seed_radius, radial_mode = resolve_seed_shell(
        data
    )

    theta_seed, phi_seed = _cartesian_to_theta_phi(
        seed_point
    )

    j = int(
        np.argmin(
            np.abs(
                theta
                - theta_seed
            )
        )
    )

    k = _nearest_periodic_phi_index(
        phi,
        phi_seed,
    )

    return float(
        br[br_index, j, k]
    )


def classify_field_line(
    data,
    points,
    seed_point,
    outer_radius,
    surface_radius,
):
    """
    Classify one full magnetic field line.

    Open:
        A field line is classified as open if EITHER

        1. at least one endpoint reaches the outer radial boundary, OR
        2. any point along the line reaches radius >
           OPEN_DISTANCE_LIMIT_FACTOR * surface_radius.

        Color is determined from Br at the seed point:
            Br > 0 -> red
            Br < 0 -> blue

    Closed:
        Neither criterion is satisfied.
        Color -> black.

    Returns
    -------
    category : str
        "open_positive", "open_negative", or "closed"

    seed_br : float or None
        Br at the seed point for open lines.
    """
    point_radii = np.linalg.norm(
        points,
        axis=1,
    )

    r_start = float(
        point_radii[0]
    )

    r_end = float(
        point_radii[-1]
    )

    open_threshold = (
        OPEN_OUTER_RADIUS_FRACTION
        * outer_radius
    )

    distance_limit = (
        OPEN_DISTANCE_LIMIT_FACTOR
        * surface_radius
    )

    reaches_outer_boundary = (
        r_start >= open_threshold
        or r_end >= open_threshold
    )

    exceeds_distance_limit = bool(
        np.any(
            point_radii > distance_limit
        )
    )

    is_open = (
        reaches_outer_boundary
        or exceeds_distance_limit
    )

    if not is_open:
        return "closed", None

    # Open-field color is determined by Br at the SEED POINT,
    # not by the projected/photospheric footpoint.
    seed_br = get_seed_br_at_point(
        data,
        seed_point,
    )

    if seed_br >= 0.0:
        return (
            "open_positive",
            seed_br,
        )

    return (
        "open_negative",
        seed_br,
    )


def _make_polyline(points):
    """
    Build one PyVista PolyData polyline from an (N,3) point array.
    """
    points = np.asarray(
        points,
        dtype=float,
    )

    poly = pv.PolyData(
        points
    )

    n = len(points)

    poly.lines = np.hstack(
        (
            [n],
            np.arange(
                n,
                dtype=np.int64,
            ),
        )
    )

    return poly


def trace_and_classify_field_lines(
    magnetic_grid,
    seed_source,
    data,
    outer_radius,
):
    """
    Trace each seed separately, merge its forward/backward branches,
    classify it as open/closed, and split the lines into three groups.

    Returns
    -------
    dict
        {
            "open_positive": list[PolyData],
            "open_negative": list[PolyData],
            "closed": list[PolyData],
        }
    """
    groups = {
        "open_positive": [],
        "open_negative": [],
        "closed": [],
    }

    seeds = np.asarray(
        seed_source.points,
        dtype=float,
    )

    surface_radius = float(
        np.asarray(
            data["r"],
            dtype=float,
        )[SURFACE_RADIAL_INDEX]
    )

    open_radius = (
        OPEN_DISTANCE_LIMIT_FACTOR
        * surface_radius
    )

    print(
        "\nTracing and classifying field lines:"
        f"\n  seeds = {len(seeds)}"
        f"\n  stop radius = {open_radius:.8g} Rs"
    )

    n_failed = 0

    for i, seed in enumerate(
        seeds,
        start=1,
    ):
        forward = _trace_one_direction(
            magnetic_grid,
            seed,
            "forward",
            outer_radius,
            open_radius,
        )

        backward = _trace_one_direction(
            magnetic_grid,
            seed,
            "backward",
            outer_radius,
            open_radius,
        )

        full_points = _combine_forward_backward(
            seed,
            backward,
            forward,
        )

        if full_points is None:
            n_failed += 1
            continue

        category, seed_br = classify_field_line(
            data,
            full_points,
            seed,
            outer_radius,
            surface_radius=surface_radius,
        )

        groups[category].append(
            _make_polyline(
                full_points
            )
        )

        if i % 20 == 0 or i == len(seeds):
            print(
                f"  processed {i}/{len(seeds)}"
            )

    print(
        "\nField-line classification:"
        f"\n  open Br>0 = {len(groups['open_positive'])}"
        f"\n  open Br<0 = {len(groups['open_negative'])}"
        f"\n  closed    = {len(groups['closed'])}"
        f"\n  failed    = {n_failed}"
    )

    return groups


def merge_polydata_list(poly_list):
    """
    Merge a list of independent PolyData line objects.
    """
    if not poly_list:
        return None

    merged = poly_list[0].copy()

    for poly in poly_list[1:]:
        merged = merged.merge(
            poly,
            merge_points=False,
        )

    return merged


# ======================================================================
# Solar rotation axis
# ======================================================================

def add_rotation_axis(plotter, surface_radius):
    """
    Draw the solar rotation axis along the global Z-axis.

    - Full axis from south (-Z) to north (+Z)
    - A clear arrow at the north end pointing toward +Z
    - North pole labeled only with "N"
    - South pole is not labeled
    """
    extent = ROTATION_AXIS_RADIUS_FACTOR * surface_radius

    south = np.array([0.0, 0.0, -extent], dtype=float)
    north = np.array([0.0, 0.0, extent], dtype=float)

    axis_line = pv.Line(south, north)

    plotter.add_mesh(
        axis_line,
        color=ROTATION_AXIS_COLOR,
        line_width=ROTATION_AXIS_LINE_WIDTH,
        name="solar_rotation_axis",
    )

    arrow_length = ROTATION_ARROW_LENGTH_FACTOR * surface_radius

    arrow_start = np.array(
        [0.0, 0.0, extent - arrow_length],
        dtype=float,
    )

    north_arrow = pv.Arrow(
        start=arrow_start,
        direction=(0.0, 0.0, 1.0),
        scale=arrow_length,
    )

    plotter.add_mesh(
        north_arrow,
        color=ROTATION_ARROW_COLOR,
        name="north_rotation_arrow",
    )

    label_z = extent + NORTH_LABEL_OFFSET_FACTOR * surface_radius

    plotter.add_point_labels(
        np.array([[0.0, 0.0, label_z]], dtype=float),
        [NORTH_LABEL],
        point_size=0,
        font_size=NORTH_LABEL_FONT_SIZE,
        text_color=ROTATION_AXIS_COLOR,
        shape=None,
        always_visible=True,
    )


# ======================================================================
# Plot
# ======================================================================

def _add_fieldline_group(
    plotter,
    poly_list,
    color,
    name,
    surface_radius,
):
    """
    Add one field-line category to the PyVista scene.
    """
    merged = merge_polydata_list(
        poly_list
    )

    if merged is None:
        return

    if DRAW_FIELDLINES_AS_TUBES:
        tube_radius = (
            FIELDLINE_TUBE_RADIUS_FACTOR
            * surface_radius
        )

        mesh = merged.tube(
            radius=tube_radius,
        )
    else:
        mesh = merged

    plotter.add_mesh(
        mesh,
        color=color,
        opacity=FIELDLINE_OPACITY,
        name=name,
    )


def add_exported_html_overlays(
    output_html_file,
    title_lines,
):
    """Add HTML-native title and categorical colorbar overlays."""
    output_html_file = Path(output_html_file)
    html_text = output_html_file.read_text(encoding="utf-8")

    title_html = "<br>".join(
        escape(str(line))
        for line in title_lines
    )

    browser_title = escape(
        f"SIP-IFVM magnetic field - {title_lines[1]}"
    )

    style = """
    <style id="sip-fieldline-overlays-style">
      .sip-scene-title {
        position: fixed;
        top: 18px;
        left: 22px;
        z-index: 1000;
        color: #111;
        font-family: Arial, sans-serif;
        font-size: 18px;
        line-height: 1.35;
        pointer-events: none;
        text-shadow: 0 0 3px white, 0 0 3px white;
      }
      .sip-topology-colorbar {
        position: fixed;
        top: 50%;
        right: 28px;
        z-index: 1000;
        transform: translateY(-50%);
        color: #111;
        font-family: Arial, sans-serif;
        font-size: 15px;
        pointer-events: none;
      }
      .sip-colorbar-title {
        margin-bottom: 8px;
        text-align: center;
        font-size: 16px;
      }
      .sip-colorbar-body {
        display: flex;
        align-items: stretch;
      }
      .sip-colorbar-swatches {
        width: 34px;
        height: 240px;
        border: 1px solid #444;
        box-sizing: border-box;
      }
      .sip-colorbar-swatch {
        height: 33.333333%;
      }
      .sip-colorbar-labels {
        display: flex;
        height: 240px;
        flex-direction: column;
        justify-content: space-around;
        margin-left: 9px;
      }
      .sip-colorbar-label {
        white-space: nowrap;
      }
    </style>
    """

    overlay = f"""
    <div class="sip-scene-title">{title_html}</div>
    <div class="sip-topology-colorbar">
      <div class="sip-colorbar-title">Open / closed</div>
      <div class="sip-colorbar-body">
        <div class="sip-colorbar-swatches">
          <div class="sip-colorbar-swatch" style="background:#b22222"></div>
          <div class="sip-colorbar-swatch" style="background:#d3d3d3"></div>
          <div class="sip-colorbar-swatch" style="background:#4169e1"></div>
        </div>
        <div class="sip-colorbar-labels">
          <div class="sip-colorbar-label">Open (+)</div>
          <div class="sip-colorbar-label">Closed</div>
          <div class="sip-colorbar-label">Open (-)</div>
        </div>
      </div>
    </div>
    """

    html_text = html_text.replace(
        "</head>",
        f"<title>{browser_title}</title>\n{style}\n</head>",
        1,
    )
    html_text = html_text.replace(
        "</body>",
        f"{overlay}\n</body>",
        1,
    )

    output_html_file.write_text(
        html_text,
        encoding="utf-8",
    )


def plot_magnetic_configuration(
    surface,
    br_zero_contour,
    seed_shell,
    seed_source,
    field_line_groups,
    surface_radius,
    seed_shell_radius,
    datetime_label,
    output_html_file=None,
    output_png_file=None,
):
    """
    Plot:
        - r_index=0 open/closed sphere with the smoothed Br=0 contour
        - transparent yellow seed shell
        - open field lines with seed-point Br>0 in red
        - open field lines with seed-point Br<0 in blue
        - closed field lines in black
    """
    use_off_screen = bool(
        (not SHOW_PLOT)
        or SAVE_SCREENSHOT
    )

    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
        off_screen=use_off_screen,
    )

    plotter.set_background(
        BACKGROUND,
    )

    # --------------------------------------------------------------
    # Solar surface: open/closed topology on r_index=0
    # --------------------------------------------------------------
    plotter.add_mesh(
        surface,
        scalars="OpenClosed",
        cmap=OPEN_CLOSED_CMAP,
        clim=OPEN_CLOSED_CLIM,
        n_colors=3,
        annotations={
            -1.0: "Open (-)",
            0.0: "Closed",
            1.0: "Open (+)",
        },
        opacity=SURFACE_OPACITY,
        show_edges=SHOW_SURFACE_EDGES,
        smooth_shading=False,
        scalar_bar_args={
            "title": "Open / closed",
            "vertical": True,
            "n_labels": 0,
        },
        name="solar_surface_open_closed",
    )

    if br_zero_contour.n_points > 0:
        plotter.add_mesh(
            br_zero_contour,
            color=BR_ZERO_CONTOUR_COLOR,
            line_width=BR_ZERO_CONTOUR_LINE_WIDTH,
            render_lines_as_tubes=True,
            name="solar_surface_Br_zero_contour",
        )

    # --------------------------------------------------------------
    # Seed shell: transparent yellow reference sphere
    # --------------------------------------------------------------
    if SHOW_SEED_SHELL:
        plotter.add_mesh(
            seed_shell,
            color=SEED_SHELL_COLOR,
            opacity=SEED_SHELL_OPACITY,
            show_edges=SEED_SHELL_SHOW_EDGES,
            smooth_shading=False,
            name="seed_shell",
        )

    # --------------------------------------------------------------
    # Field-line categories
    # --------------------------------------------------------------
    _add_fieldline_group(
        plotter,
        field_line_groups["open_positive"],
        OPEN_POSITIVE_COLOR,
        "open_positive_field_lines",
        surface_radius,
    )

    _add_fieldline_group(
        plotter,
        field_line_groups["open_negative"],
        OPEN_NEGATIVE_COLOR,
        "open_negative_field_lines",
        surface_radius,
    )

    _add_fieldline_group(
        plotter,
        field_line_groups["closed"],
        CLOSED_FIELD_COLOR,
        "closed_field_lines",
        surface_radius,
    )

    # --------------------------------------------------------------
    # Seed points
    # --------------------------------------------------------------
    if SHOW_SEED_POINTS:
        plotter.add_mesh(
            seed_source,
            color=SEED_POINT_COLOR,
            point_size=SEED_POINT_SIZE,
            render_points_as_spheres=True,
            name="seed_points",
        )

    # --------------------------------------------------------------
    # Scene
    # --------------------------------------------------------------
    if SHOW_AXES:
        plotter.add_axes(
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )

    if SHOW_GRID_BOUNDS:
        plotter.show_bounds(
            grid="back",
            location="outer",
            all_edges=True,
        )

    title_lines = [
        "SIP-IFVM magnetic field",
        datetime_label,
        f"sphere shell: r = {surface_radius:.6g} Rs",
        f"seed shell: r = {seed_shell_radius:.6g} Rs",
    ]

    plotter.add_text(
        "\n".join(title_lines),
        position="upper_left",
        font_size=12,
    )

    # Compute camera before adding the long rotation-axis actor, preserving
    # the framing behavior used in the earlier version.
    plotter.view_isometric()

    # --------------------------------------------------------------
    # Solar rotation axis
    # --------------------------------------------------------------
    if SHOW_ROTATION_AXIS:
        add_rotation_axis(
            plotter,
            surface_radius,
        )

    plotter.camera.focal_point = (
        0.0,
        0.0,
        0.0,
    )

    if output_html_file is not None and SAVE_HTML:
        output_html_file = Path(
            output_html_file
        )
        output_html_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        scalar_bar_actor = dict(
            plotter.scalar_bars.items()
        ).get("Open / closed")

        if scalar_bar_actor is not None:
            scalar_bar_actor.SetVisibility(False)

        try:
            plotter.export_html(
                output_html_file
            )
        finally:
            if scalar_bar_actor is not None:
                scalar_bar_actor.SetVisibility(True)

        add_exported_html_overlays(
            output_html_file,
            title_lines,
        )
        print(
            f"Saved HTML:\n  {output_html_file}"
        )

    if output_png_file is not None and SAVE_SCREENSHOT:
        output_png_file = Path(
            output_png_file
        )
        output_png_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plotter.screenshot(
            output_png_file
        )
        print(
            f"Saved screenshot:\n  {output_png_file}"
        )

    if SHOW_PLOT:
        plotter.show()
    else:
        plotter.close()



# ======================================================================
# Main
# ======================================================================

def main():

    data_files = select_merged_data_files(
        data_dir=DATA_DIR,
        process_mode=PROCESS_MODE,
        single_filename=SINGLE_DATA_FILE,
    )

    print(
        f"Processing mode: {PROCESS_MODE}"
        f"\nData directory : {DATA_DIR}"
        f"\nFile count     : {len(data_files)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_data = None

    if str(SEED_COORDINATE_TIME_MODE).strip().lower() == "diffrot_from_82h":
        if not REFERENCE_SEED_FILE.exists():
            raise FileNotFoundError(
                REFERENCE_SEED_FILE
            )

        reference_file_hours = simulation_hours_from_filename(
            REFERENCE_SEED_FILE
        )

        if not np.isclose(
            reference_file_hours,
            REFERENCE_SEED_TIME_HOURS,
            atol=1.0e-8,
        ):
            raise ValueError(
                f"REFERENCE_SEED_FILE encodes {reference_file_hours:.2f} h, "
                f"but REFERENCE_SEED_TIME_HOURS="
                f"{REFERENCE_SEED_TIME_HOURS:.2f} h."
            )

        reference_data = read_merged_physics(
            filename=REFERENCE_SEED_FILE,
            grid_filename=GRID_FILE,
        )

        validate_data(
            reference_data
        )

        print(
            "\nReference seed data:"
            f"\n  file = {REFERENCE_SEED_FILE}"
            f"\n  time = {reference_file_hours:.2f} h"
        )

    for file_index, DATA_FILE in enumerate(
        data_files,
        start=1,
    ):

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"Processing file "
            f"{file_index}/{len(data_files)}:"
            f"\n  {DATA_FILE.name}"
        )

        print(
            "=" * 72
        )

        data = read_merged_physics(
            filename=DATA_FILE,
            grid_filename=GRID_FILE,
        )

        simulation_hours = simulation_hours_from_filename(
            DATA_FILE
        )

        simulation_datetime = simulation_datetime_from_filename(
            DATA_FILE
        )

        datetime_label = format_simulation_datetime(
            DATA_FILE
        )

        print(
            "Simulation time from filename:"
            f"\n  file     = {DATA_FILE.name}"
            f"\n  time     = {simulation_hours:.2f} h"
            f"\n  datetime = {simulation_datetime:%Y-%m-%d %H:%M}"
        )

        validate_data(
            data,
        )

        r = np.asarray(
            data["r"],
            dtype=float,
        )

        print(
            "Merged magnetic data:"
            f"\n  Nr      = {len(data['r'])}"
            f"\n  Ntheta  = {len(data['theta'])}"
            f"\n  Nphi    = {len(data['phi'])}"
            f"\n  r_min   = {r.min():.8g}"
            f"\n  r_max   = {r.max():.8g}"
        )

        (
            open_closed_file,
            open_closed_map,
            br_contour,
        ) = load_open_closed_surface_data(
            data,
            simulation_hours,
        )

        (
            surface,
            br_zero_contour,
            surface_radius,
        ) = build_open_closed_surface(
            data,
            open_closed_map,
            br_contour,
        )

        print(
            "\nDisplayed open/closed surface:"
            f"\n  radial index = {SURFACE_RADIAL_INDEX}"
            f"\n  radius       = {surface_radius:.8g}"
            f"\n  source       = {open_closed_file}"
            f"\n  map tag      = {OPEN_CLOSED_MAP_TAG}"
            f"\n  Br=0 points  = {br_zero_contour.n_points}"
        )

        seed_radial_index, seed_radius, seed_radial_mode = resolve_seed_shell(
            data
        )

        print(
            "\nSeed shell:"
            f"\n  radial mode  = {seed_radial_mode}"
            f"\n  radius       = {seed_radius:.8g}"
            f"\n  Br index     = {seed_radial_index}"
            f"\n  Br radius    = {r[seed_radial_index]:.8g}"
            f"\n  seed mode 1  = {SEED_MODE}"
            f"\n  seed mode 2  = {SEED_MODE_2}"
        )

        print(
            "\nSeed-coordinate time mode:"
            f"\n  mode           = {SEED_COORDINATE_TIME_MODE}"
            f"\n  reference time = {REFERENCE_SEED_TIME_HOURS:.2f} h"
            f"\n  current time   = {simulation_hours:.2f} h"
        )

        magnetic_grid, wrapped = build_magnetic_structured_grid(
            data,
        )

        print(
            "\nPyVista StructuredGrid:"
            f"\n  dimensions = {magnetic_grid.dimensions}"
            f"\n  points     = {magnetic_grid.n_points}"
            f"\n  cells      = {magnetic_grid.n_cells}"
        )

        seed_shell, seed_shell_radius = build_seed_shell_surface(
            data
        )

        seed_source = build_seed_source(
            current_data=data,
            current_simulation_hours=simulation_hours,
            reference_data=reference_data,
        )

        field_line_groups = trace_and_classify_field_lines(
            magnetic_grid,
            seed_source,
            data,
            outer_radius=float(r.max()),
        )

        html_filename = HTML_FILENAME_FORMAT.format(
            simulation_hours=simulation_hours,
            seed_radial_index=seed_radial_index,
        )

        png_filename = PNG_FILENAME_FORMAT.format(
            simulation_hours=simulation_hours,
            seed_radial_index=seed_radial_index,
        )

        output_html_file = OUTPUT_DIR / html_filename
        output_png_file = OUTPUT_DIR / png_filename

        plot_magnetic_configuration(
            surface,
            br_zero_contour,
            seed_shell,
            seed_source,
            field_line_groups,
            surface_radius,
            seed_shell_radius,
            datetime_label,
            output_html_file=output_html_file,
            output_png_file=output_png_file,
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "All selected merged HDF5 files finished."
    )


if __name__ == "__main__":
    main()
