"""
plot_tracked_crossings_connectivity_series.py

Post-process tracked SIP-IFVM open-field footpoints.

Main tasks
----------
1. Read all:
       track_open.time.xx.xx.r0.R0.npz

2. Read an open/closed topology file such as:
       open_closed_time.82.10.npz

3. Inside a configurable longitude/latitude region, find boundary pairs
   where one pixel is closed (0) and the neighboring pixel is open
   (+1 or -1).

4. Randomly select N_ID target positions from both sides of these
   open/closed boundaries.

5. Match those target positions to the closest initial r_index=0
   tracking footpoints and use the corresponding global IDs.

6. For the selected IDs:
       - print longitude / latitude at every time to the terminal
       - plot longitude vs time
       - plot latitude vs time
       - legend displays ID only

7. Save the selected global IDs and their initial r_index=0 and R0
   longitude/latitude to:
       tracked_open_field_id.npz

8. Save separate longitude/latitude time-series figures for the r_index=0
   and R0 footpoints.

Longitude time series can optionally remove the empirical
latitude-dependent differential-rotation drift.

Topology-map plotting is handled separately by:
    plot_tracked_crossings_connectivity_map.py

Outputs: `selected_open_boundary_footpoints_crossings.r.<R0>.npz`,
`open_boundary_footpoint_tracks.crossing-r.<R0>.png`, and
`open_boundary_crossing_tracks.r.<R0>.png`.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

from config import OPEN_CLOSED_DIR, TRACK_OPEN_DIR, WORK_ROOT
from figure_provenance import add_figure_provenance


# ======================================================================
# CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# Tracking files
# ----------------------------------------------------------------------

WORK_DIR = WORK_ROOT
TRACK_DIR = TRACK_OPEN_DIR

R0 = 10.0

# ----------------------------------------------------------------------
# Open / closed reference map used to select IDs
# ----------------------------------------------------------------------

OPEN_CLOSED_FILE = OPEN_CLOSED_DIR / "open_closed_time.82.10.npz"

# Prefix/tag inside the NPZ.
# Example keys:
#   rindex0_open_closed_map
#   rindex0_Br_surface
OPEN_CLOSED_TAG = "rindex0"

# Number of final IDs to select.
N_ID = 30

RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Boundary-search region
# ----------------------------------------------------------------------
#
# Longitude [deg], 0..360.
#
BOUNDARY_LON_MIN_DEG = 20.0
BOUNDARY_LON_MAX_DEG = 80.0

# Latitude [deg], -90..90.
BOUNDARY_LAT_MIN_DEG = -10.0
BOUNDARY_LAT_MAX_DEG = 60.0

# Boundary detection uses direct 4-neighbor adjacency:
# left/right/up/down.
#
# One point must have label 0 and the other label +1 or -1.
#
# When selecting N_ID target points:
# - boundary PAIRS are sampled randomly
# - the chosen target alternates between the closed and open side
#   so both sides of the boundary are represented.
#
# If True, one tracking ID may be used only once.
REQUIRE_UNIQUE_IDS = True

# ----------------------------------------------------------------------
# Differential-rotation correction
# ----------------------------------------------------------------------

DIFFROT_A_DEG_PER_DAY = 14.713
DIFFROT_B_DEG_PER_DAY = -2.396
DIFFROT_C_DEG_PER_DAY = -1.787

# None -> first tracking time.
DIFFROT_REFERENCE_TIME_HOURS = None

DIFFROT_DRIFT_SIGN = +1.0

WRAP_CORRECTED_LONGITUDE = True

# ----------------------------------------------------------------------
# Time-series plot
# ----------------------------------------------------------------------

FIGSIZE = (11, 8)

MARKER = "o"
MARKER_SIZE = 2.0
LINE_WIDTH = 1.2

LONGITUDE_YLIM = (
    0.0,
    90.0,
)

LATITUDE_YLIM = (
    -20.0,
    80.0,
)

LONGITUDE_YTICKS = np.arange(
    0.0,
    91.0,
    15.0,
)

LATITUDE_YTICKS = np.arange(
    -20.0,
    81.0,
    20.0,
)

SHOW_GRID = True
GRID_ALPHA = 0.30

LEGEND_NCOL = 2
LEGEND_FONT_SIZE = 8

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

SAVE_OR_NOT = 1

DPI = 300

TIME_SERIES_OUTPUT_FILE = (
    WORK_DIR
    / f"open_boundary_footpoint_tracks.crossing-r.{R0:g}.png"
)

R0_TIME_SERIES_OUTPUT_FILE = (
    WORK_DIR
    / f"open_boundary_crossing_tracks.r.{R0:g}.png"
)

ID_OUTPUT_FILE = (
    WORK_DIR
    / f"selected_open_boundary_footpoints_crossings.r.{R0:g}.npz"
)


# ======================================================================
# FILE DISCOVERY
# ======================================================================

def discover_track_files(
    track_dir=TRACK_DIR,
    r0=R0,
):
    track_dir = Path(
        track_dir
    )

    if not track_dir.exists():
        raise FileNotFoundError(
            track_dir
        )

    pattern = re.compile(
        r"^track_open\.time\.([0-9]+(?:\.[0-9]+)?)"
        r"\.r0\.([0-9]+(?:\.[0-9]+)?)\.npz$"
    )

    matched = []

    for filename in track_dir.iterdir():

        if not filename.is_file():
            continue

        match = pattern.match(
            filename.name
        )

        if match is None:
            continue

        simulation_hours = float(
            match.group(1)
        )

        file_r0 = float(
            match.group(2)
        )

        if not np.isclose(
            file_r0,
            float(
                r0
            ),
            rtol=0.0,
            atol=1.0e-8,
        ):
            continue

        matched.append(
            (
                simulation_hours,
                filename,
            )
        )

    matched.sort(
        key=lambda item: item[0]
    )

    if not matched:
        raise FileNotFoundError(
            f"No track files for R0={r0:g} were found in:\n"
            f"{track_dir}"
        )

    return matched


# ======================================================================
# READ TRACK NPZ
# ======================================================================

def load_track_npz(
    filename,
):
    required = (
        "id",
        "r0_theta",
        "r0_phi",
        "inner_theta",
        "inner_phi",
    )

    with np.load(
        filename,
        allow_pickle=False,
    ) as f:

        for key in required:

            if key not in f:
                raise KeyError(
                    f'Missing "{key}" in {filename}'
                )

        return {
            key: np.asarray(
                f[
                    key
                ]
            ).copy()
            for key in required
        }


def build_id_to_index(
    ids,
):
    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    return {
        int(seed_id): int(index)
        for index, seed_id in enumerate(
            ids
        )
    }


# ======================================================================
# READ OPEN / CLOSED NPZ
# ======================================================================

def load_open_closed_reference(
    filename=OPEN_CLOSED_FILE,
    tag=OPEN_CLOSED_TAG,
):
    filename = Path(
        filename
    )

    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    map_key = (
        f"{tag}_open_closed_map"
    )

    br_key = (
        f"{tag}_Br_surface"
    )

    required = (
        "theta",
        "phi",
        map_key,
        br_key,
    )

    with np.load(
        filename,
        allow_pickle=False,
    ) as f:

        for key in required:

            if key not in f:
                raise KeyError(
                    f'Missing "{key}" in {filename}. '
                    f"Available keys include: {f.files}"
                )

        theta = np.asarray(
            f[
                "theta"
            ],
            dtype=float,
        )

        phi = np.asarray(
            f[
                "phi"
            ],
            dtype=float,
        )

        open_closed_map = np.asarray(
            f[
                map_key
            ],
            dtype=float,
        )

        Br_surface = np.asarray(
            f[
                br_key
            ],
            dtype=float,
        )

    expected_shape = (
        len(
            theta
        ),
        len(
            phi
        ),
    )

    if open_closed_map.shape != expected_shape:
        raise ValueError(
            f"{map_key}.shape={open_closed_map.shape}, "
            f"expected {expected_shape}"
        )

    if Br_surface.shape != expected_shape:
        raise ValueError(
            f"{br_key}.shape={Br_surface.shape}, "
            f"expected {expected_shape}"
        )

    return (
        theta,
        phi,
        open_closed_map,
        Br_surface,
    )


# ======================================================================
# COORDINATE HELPERS
# ======================================================================

def theta_to_latitude_deg(
    theta_rad,
):
    return (
        90.0
        - np.degrees(
            theta_rad
        )
    )


def phi_to_longitude_deg(
    phi_rad,
):
    return (
        np.degrees(
            phi_rad
        )
        % 360.0
    )


def lon_in_region(
    longitude_deg,
):
    """
    Region test supporting longitude intervals that cross 0 deg.

    Examples:
        30..100
        330..30
    """
    longitude_deg = np.asarray(
        longitude_deg,
        dtype=float,
    ) % 360.0

    lon_min = float(
        BOUNDARY_LON_MIN_DEG
    ) % 360.0

    lon_max = float(
        BOUNDARY_LON_MAX_DEG
    ) % 360.0

    # Treat 0..360 as full longitude.
    span = (
        float(
            BOUNDARY_LON_MAX_DEG
        )
        - float(
            BOUNDARY_LON_MIN_DEG
        )
    )

    if abs(
        span
    ) >= 360.0 - 1.0e-10:
        return np.ones_like(
            longitude_deg,
            dtype=bool,
        )

    if lon_min <= lon_max:

        return (
            (longitude_deg >= lon_min)
            & (longitude_deg <= lon_max)
        )

    return (
        (longitude_deg >= lon_min)
        | (longitude_deg <= lon_max)
    )


def lat_in_region(
    latitude_deg,
):
    latitude_deg = np.asarray(
        latitude_deg,
        dtype=float,
    )

    return (
        (
            latitude_deg
            >= BOUNDARY_LAT_MIN_DEG
        )
        & (
            latitude_deg
            <= BOUNDARY_LAT_MAX_DEG
        )
    )


def spherical_angular_distance(
    lat1_deg,
    lon1_deg,
    lat2_deg,
    lon2_deg,
):
    """
    Great-circle angular distance [rad].
    """
    lat1 = np.radians(
        lat1_deg
    )

    lon1 = np.radians(
        lon1_deg
    )

    lat2 = np.radians(
        lat2_deg
    )

    lon2 = np.radians(
        lon2_deg
    )

    cosang = (
        np.sin(
            lat1
        )
        * np.sin(
            lat2
        )
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.cos(
            lon1
            - lon2
        )
    )

    return np.arccos(
        np.clip(
            cosang,
            -1.0,
            1.0,
        )
    )


# ======================================================================
# FIND OPEN / CLOSED BOUNDARY PAIRS
# ======================================================================

def is_open_label(
    value,
):
    return (
        value
        == 1
        or value
        == -1
    )


def find_boundary_pairs(
    theta,
    phi,
    open_closed_map,
):
    """
    Find adjacent closed/open pixel pairs inside the specified region.

    Returns
    -------
    list of dict
        {
            "closed_j",
            "closed_k",
            "open_j",
            "open_k",
        }

    Longitude adjacency is periodic.
    Theta adjacency is not periodic.
    """
    latitude = theta_to_latitude_deg(
        theta
    )

    longitude = phi_to_longitude_deg(
        phi
    )

    region_theta = lat_in_region(
        latitude
    )

    region_phi = lon_in_region(
        longitude
    )

    nt = len(
        theta
    )

    np_ = len(
        phi
    )

    pairs = []

    # --------------------------------------------------------------
    # Theta-neighbor pairs
    # --------------------------------------------------------------
    for j in range(
        nt - 1
    ):

        j2 = (
            j + 1
        )

        if not (
            region_theta[
                j
            ]
            and region_theta[
                j2
            ]
        ):
            continue

        for k in range(
            np_
        ):

            if not region_phi[
                k
            ]:
                continue

            v1 = int(
                open_closed_map[
                    j,
                    k,
                ]
            )

            v2 = int(
                open_closed_map[
                    j2,
                    k,
                ]
            )

            if (
                v1 == 0
                and is_open_label(
                    v2
                )
            ):
                pairs.append(
                    {
                        "closed_j": j,
                        "closed_k": k,
                        "open_j": j2,
                        "open_k": k,
                    }
                )

            elif (
                v2 == 0
                and is_open_label(
                    v1
                )
            ):
                pairs.append(
                    {
                        "closed_j": j2,
                        "closed_k": k,
                        "open_j": j,
                        "open_k": k,
                    }
                )

    # --------------------------------------------------------------
    # Phi-neighbor pairs, periodic
    # --------------------------------------------------------------
    for j in range(
        nt
    ):

        if not region_theta[
            j
        ]:
            continue

        for k in range(
            np_
        ):

            k2 = (
                k + 1
            ) % np_

            if not (
                region_phi[
                    k
                ]
                and region_phi[
                    k2
                ]
            ):
                continue

            v1 = int(
                open_closed_map[
                    j,
                    k,
                ]
            )

            v2 = int(
                open_closed_map[
                    j,
                    k2,
                ]
            )

            if (
                v1 == 0
                and is_open_label(
                    v2
                )
            ):
                pairs.append(
                    {
                        "closed_j": j,
                        "closed_k": k,
                        "open_j": j,
                        "open_k": k2,
                    }
                )

            elif (
                v2 == 0
                and is_open_label(
                    v1
                )
            ):
                pairs.append(
                    {
                        "closed_j": j,
                        "closed_k": k2,
                        "open_j": j,
                        "open_k": k,
                    }
                )

    if not pairs:
        raise RuntimeError(
            "No open/closed boundary pairs were found inside "
            "the requested longitude/latitude region."
        )

    return pairs


def choose_boundary_target_points(
    theta,
    phi,
    boundary_pairs,
    n_id=N_ID,
    random_seed=RANDOM_SEED,
):
    """
    Randomly choose N_ID target pixels from BOTH sides of the boundary.

    The sampled target alternates:
        closed side, open side, closed side, open side, ...

    Thus, for N_ID=10, approximately half are closed-side pixels and
    half are open-side pixels.
    """
    if len(
        boundary_pairs
    ) < int(
        np.ceil(
            n_id / 2
        )
    ):
        raise ValueError(
            f"Only {len(boundary_pairs)} boundary pairs are available, "
            f"not enough for N_ID={n_id}."
        )

    rng = np.random.default_rng(
        random_seed
    )

    # Sampling with replacement is unnecessary if enough pairs exist.
    replace = (
        len(
            boundary_pairs
        )
        < n_id
    )

    selected_pair_indices = rng.choice(
        len(
            boundary_pairs
        ),
        size=n_id,
        replace=replace,
    )

    targets = []

    for i, pair_index in enumerate(
        selected_pair_indices
    ):

        pair = boundary_pairs[
            int(
                pair_index
            )
        ]

        if i % 2 == 0:

            side = "closed"

            j = pair[
                "closed_j"
            ]

            k = pair[
                "closed_k"
            ]

        else:

            side = "open"

            j = pair[
                "open_j"
            ]

            k = pair[
                "open_k"
            ]

        targets.append(
            {
                "side": side,
                "theta": float(
                    theta[
                        j
                    ]
                ),
                "phi": float(
                    phi[
                        k
                    ]
                ),
                "latitude_deg": float(
                    theta_to_latitude_deg(
                        theta[
                            j
                        ]
                    )
                ),
                "longitude_deg": float(
                    phi_to_longitude_deg(
                        phi[
                            k
                        ]
                    )
                ),
                "j": int(
                    j
                ),
                "k": int(
                    k
                ),
            }
        )

    return targets


# ======================================================================
# MATCH TARGET POINTS TO INITIAL TRACKING IDs
# ======================================================================

def get_initial_tracking_positions(
    first_track,
):
    ids = np.asarray(
        first_track[
            "id"
        ],
        dtype=np.int64,
    )

    theta = np.asarray(
        first_track[
            "inner_theta"
        ],
        dtype=float,
    )

    phi = np.asarray(
        first_track[
            "inner_phi"
        ],
        dtype=float,
    )

    valid = (
        np.isfinite(
            theta
        )
        & np.isfinite(
            phi
        )
    )

    return (
        ids[
            valid
        ],
        theta[
            valid
        ],
        phi[
            valid
        ],
    )


def match_targets_to_ids(
    targets,
    first_track,
):
    """
    Match every selected boundary-side target to the nearest initial
    tracking footpoint on the sphere.

    Great-circle distance is used.

    If REQUIRE_UNIQUE_IDS=True, once an ID is selected it cannot be used
    again for another target.
    """
    (
        available_ids,
        available_theta,
        available_phi,
    ) = get_initial_tracking_positions(
        first_track
    )

    available_lat = theta_to_latitude_deg(
        available_theta
    )

    available_lon = phi_to_longitude_deg(
        available_phi
    )

    used_ids = set()

    selected_ids = []
    match_info = []

    for target in targets:

        distance = spherical_angular_distance(
            available_lat,
            available_lon,
            target[
                "latitude_deg"
            ],
            target[
                "longitude_deg"
            ],
        )

        order = np.argsort(
            distance
        )

        selected_index = None

        for index in order:

            candidate_id = int(
                available_ids[
                    index
                ]
            )

            if (
                REQUIRE_UNIQUE_IDS
                and candidate_id
                in used_ids
            ):
                continue

            selected_index = int(
                index
            )

            break

        if selected_index is None:
            raise RuntimeError(
                "Could not find enough unique tracking IDs for "
                "the selected boundary targets."
            )

        seed_id = int(
            available_ids[
                selected_index
            ]
        )

        used_ids.add(
            seed_id
        )

        selected_ids.append(
            seed_id
        )

        match_info.append(
            {
                "id": seed_id,
                "target_side": target[
                    "side"
                ],
                "target_latitude_deg": target[
                    "latitude_deg"
                ],
                "target_longitude_deg": target[
                    "longitude_deg"
                ],
                "matched_latitude_deg": float(
                    available_lat[
                        selected_index
                    ]
                ),
                "matched_longitude_deg": float(
                    available_lon[
                        selected_index
                    ]
                ),
                "angular_distance_deg": float(
                    np.degrees(
                        distance[
                            selected_index
                        ]
                    )
                ),
            }
        )

    return (
        np.asarray(
            selected_ids,
            dtype=np.int64,
        ),
        match_info,
    )


# ======================================================================
# SAVE SELECTED IDS
# ======================================================================

def save_selected_ids(
    filename,
    selected_ids,
    match_info,
    initial_track,
    reference_time_hours,
):
    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    id_to_index = build_id_to_index(
        initial_track["id"]
    )

    r0_longitude_deg = np.full(
        len(selected_ids),
        np.nan,
        dtype=float,
    )

    r0_latitude_deg = np.full(
        len(selected_ids),
        np.nan,
        dtype=float,
    )

    for iid, seed_id in enumerate(selected_ids):
        index = id_to_index.get(int(seed_id))

        if index is None:
            raise KeyError(
                f"Selected ID {int(seed_id)} is absent from the initial track."
            )

        theta_value = float(initial_track["r0_theta"][index])
        phi_value = float(initial_track["r0_phi"][index])

        if not (np.isfinite(theta_value) and np.isfinite(phi_value)):
            raise ValueError(
                f"Initial R0 coordinates are invalid for ID {int(seed_id)}."
            )

        r0_longitude_deg[iid] = phi_to_longitude_deg(phi_value)
        r0_latitude_deg[iid] = theta_to_latitude_deg(theta_value)

    np.savez_compressed(
        filename,

        id=np.asarray(
            selected_ids,
            dtype=np.int64,
        ),

        reference_time_hours=float(
            reference_time_hours
        ),

        r0=float(
            R0
        ),

        open_closed_file=str(
            OPEN_CLOSED_FILE
        ),

        open_closed_tag=str(
            OPEN_CLOSED_TAG
        ),

        target_side=np.asarray(
            [
                info["target_side"]
                for info in match_info
            ]
        ),

        target_longitude_deg=np.asarray(
            [
                info["target_longitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        target_latitude_deg=np.asarray(
            [
                info["target_latitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        matched_longitude_deg=np.asarray(
            [
                info["matched_longitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        matched_latitude_deg=np.asarray(
            [
                info["matched_latitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        # Initial-time r_index=0 footpoint coordinates for each ID.
        initial_inner_longitude_deg=np.asarray(
            [
                info["matched_longitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        initial_inner_latitude_deg=np.asarray(
            [
                info["matched_latitude_deg"]
                for info in match_info
            ],
            dtype=float,
        ),

        # Initial R0 footpoint coordinates for each selected ID.
        initial_r0_longitude_deg=r0_longitude_deg,

        initial_r0_latitude_deg=r0_latitude_deg,

        angular_distance_deg=np.asarray(
            [
                info["angular_distance_deg"]
                for info in match_info
            ],
            dtype=float,
        ),
    )


# ======================================================================
# DIFFERENTIAL ROTATION
# ======================================================================

def differential_rotation_rate_deg_per_day(
    latitude_deg,
):
    latitude_rad = np.radians(
        latitude_deg
    )

    sin2 = (
        np.sin(
            latitude_rad
        )
        ** 2
    )

    return (
        DIFFROT_A_DEG_PER_DAY
        + DIFFROT_B_DEG_PER_DAY
        * sin2
        + DIFFROT_C_DEG_PER_DAY
        * sin2**2
    )


def remove_differential_rotation_drift(
    longitude_deg,
    latitude_deg,
    simulation_hours,
    reference_time_hours,
):
    longitude_deg = np.asarray(
        longitude_deg,
        dtype=float,
    )

    latitude_deg = np.asarray(
        latitude_deg,
        dtype=float,
    )

    simulation_hours = np.asarray(
        simulation_hours,
        dtype=float,
    )

    omega = differential_rotation_rate_deg_per_day(
        latitude_deg
    )

    delta_t_days = (
        simulation_hours
        - float(
            reference_time_hours
        )
    ) / 24.0

    drift = (
        DIFFROT_DRIFT_SIGN
        * omega
        * delta_t_days
    )

    corrected = (
        longitude_deg
        - drift
    )

    if WRAP_CORRECTED_LONGITUDE:

        finite = np.isfinite(
            corrected
        )

        corrected[
            finite
        ] %= 360.0

    return corrected


# ======================================================================
# READ SELECTED ID TIME SERIES
# ======================================================================

def read_selected_id_series(
    track_files,
    selected_ids,
    surface="inner",
):
    coordinate_keys = {
        "inner": (
            "inner_theta",
            "inner_phi",
        ),
        "r0": (
            "r0_theta",
            "r0_phi",
        ),
    }

    if surface not in coordinate_keys:
        raise ValueError(
            f"Unknown surface={surface!r}; use 'inner' or 'r0'."
        )

    theta_key, phi_key = coordinate_keys[surface]

    times = np.asarray(
        [
            time_hours
            for time_hours, _
            in track_files
        ],
        dtype=float,
    )

    nid = len(
        selected_ids
    )

    ntime = len(
        times
    )

    longitude = np.full(
        (
            nid,
            ntime,
        ),
        np.nan,
        dtype=float,
    )

    latitude = np.full(
        (
            nid,
            ntime,
        ),
        np.nan,
        dtype=float,
    )

    for itime, (
        time_hours,
        filename,
    ) in enumerate(
        track_files
    ):

        track = load_track_npz(
            filename
        )

        id_to_index = build_id_to_index(
            track[
                "id"
            ]
        )

        surface_theta = np.asarray(track[theta_key], dtype=float)
        surface_phi = np.asarray(track[phi_key], dtype=float)

        for iid, seed_id in enumerate(
            selected_ids
        ):

            index = id_to_index.get(
                int(
                    seed_id
                )
            )

            if index is None:
                continue

            theta_value = surface_theta[index]
            phi_value = surface_phi[index]

            if not (
                np.isfinite(
                    theta_value
                )
                and np.isfinite(
                    phi_value
                )
            ):
                continue

            latitude[
                iid,
                itime,
            ] = theta_to_latitude_deg(
                theta_value
            )

            longitude[
                iid,
                itime,
            ] = phi_to_longitude_deg(
                phi_value
            )

    return (
        times,
        longitude,
        latitude,
    )


# ======================================================================
# TERMINAL OUTPUT
# ======================================================================

def print_selected_id_time_series(
    times,
    selected_ids,
    longitude,
    latitude,
):
    """
    Print lon/lat for every selected ID at every saved time.
    """
    print(
        "\n"
        + "=" * 76
    )

    print(
        "Selected ID footpoint coordinates "
        "(r_index=0, before differential-rotation correction)"
    )

    for iid, seed_id in enumerate(
        selected_ids
    ):

        print(
            "\n"
            + "-" * 76
        )

        print(
            f"ID {int(seed_id)}"
        )

        for itime, time_hours in enumerate(
            times
        ):

            lon = longitude[
                iid,
                itime,
            ]

            lat = latitude[
                iid,
                itime,
            ]

            print(
                f"  time={time_hours:8.2f} h | "
                f"lon={lon:10.4f} deg | "
                f"lat={lat:10.4f} deg"
            )


# ======================================================================
# TIME-SERIES PLOT
# ======================================================================

def plot_tracked_footpoints(
    times,
    selected_ids,
    corrected_longitude,
    latitude,
    surface_label="r_index = 0",
):
    fig, (
        ax_lon,
        ax_lat,
    ) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIGSIZE,
        sharex=True,
        constrained_layout=True,
    )

    for iid, seed_id in enumerate(
        selected_ids
    ):

        lon_series = corrected_longitude[
            iid
        ]

        lat_series = latitude[
            iid
        ]

        # Legend now displays ONLY the ID.
        legend_label = (
            f"ID {int(seed_id)}"
        )

        line_lon, = ax_lon.plot(
            times,
            lon_series,
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=legend_label,
        )

        color = line_lon.get_color()

        ax_lat.plot(
            times,
            lat_series,
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=color,
        )

    ax_lon.set_ylabel(
        "Longitude [deg]"
    )

    ax_lon.set_ylim(
        *LONGITUDE_YLIM
    )

    ax_lon.set_yticks(
        LONGITUDE_YTICKS
    )

    ax_lon.set_title(
        f"{surface_label} footpoints: longitude "
        "(differential-rotation drift removed)"
    )

    ax_lat.set_xlabel(
        "Simulation time [h]"
    )

    ax_lat.set_ylabel(
        "Latitude [deg]"
    )

    ax_lat.set_ylim(
        *LATITUDE_YLIM
    )

    ax_lat.set_yticks(
        LATITUDE_YTICKS
    )

    ax_lat.set_title(
        f"{surface_label} footpoints: latitude"
    )

    if SHOW_GRID:

        for ax in (
            ax_lon,
            ax_lat,
        ):

            ax.grid(
                True,
                alpha=GRID_ALPHA,
            )

    ax_lon.legend(
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONT_SIZE,
    )

    return (
        fig,
        (
            ax_lon,
            ax_lat,
        ),
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    # --------------------------------------------------------------
    # 1. Tracking files
    # --------------------------------------------------------------
    track_files = discover_track_files(
        TRACK_DIR,
        R0,
    )

    print(
        "Tracking files:"
        f"\n  directory = {TRACK_DIR}"
        f"\n  R0        = {R0:g} Rs"
        f"\n  count     = {len(track_files)}"
        f"\n  time      = "
        f"{track_files[0][0]:.2f} .. "
        f"{track_files[-1][0]:.2f} h"
    )

    first_track = load_track_npz(
        track_files[
            0
        ][
            1
        ]
    )

    # --------------------------------------------------------------
    # 2. Reference topology
    # --------------------------------------------------------------
    (
        topology_theta,
        topology_phi,
        open_closed_map,
        Br_surface,
    ) = load_open_closed_reference(
        OPEN_CLOSED_FILE,
        OPEN_CLOSED_TAG,
    )

    print(
        "\nReference topology:"
        f"\n  file = {OPEN_CLOSED_FILE}"
        f"\n  tag  = {OPEN_CLOSED_TAG}"
        f"\n  shape= {open_closed_map.shape}"
    )

    # --------------------------------------------------------------
    # 3. Boundary pairs -> random target positions
    # --------------------------------------------------------------
    boundary_pairs = find_boundary_pairs(
        topology_theta,
        topology_phi,
        open_closed_map,
    )

    print(
        "\nBoundary search:"
        f"\n  longitude = "
        f"[{BOUNDARY_LON_MIN_DEG:.2f}, "
        f"{BOUNDARY_LON_MAX_DEG:.2f}] deg"
        f"\n  latitude  = "
        f"[{BOUNDARY_LAT_MIN_DEG:.2f}, "
        f"{BOUNDARY_LAT_MAX_DEG:.2f}] deg"
        f"\n  boundary pairs = {len(boundary_pairs):,}"
    )

    targets = choose_boundary_target_points(
        topology_theta,
        topology_phi,
        boundary_pairs,
        n_id=N_ID,
        random_seed=RANDOM_SEED,
    )

    # --------------------------------------------------------------
    # 4. Match targets to initial tracking IDs
    # --------------------------------------------------------------
    (
        selected_ids,
        match_info,
    ) = match_targets_to_ids(
        targets,
        first_track,
    )

    print(
        "\nSelected IDs matched to boundary-side targets:"
    )

    for info in match_info:

        print(
            f"  ID {info['id']:6d} | "
            f"side={info['target_side']:6s} | "
            f"target(lon,lat)="
            f"({info['target_longitude_deg']:8.3f}, "
            f"{info['target_latitude_deg']:8.3f}) | "
            f"matched(lon,lat)="
            f"({info['matched_longitude_deg']:8.3f}, "
            f"{info['matched_latitude_deg']:8.3f}) | "
            f"distance={info['angular_distance_deg']:.4f} deg"
        )

    save_selected_ids(
        ID_OUTPUT_FILE,
        selected_ids,
        match_info,
        first_track,
        reference_time_hours=track_files[0][0],
    )

    print(
        "\nSaved selected IDs and initial r_index=0/R0 positions:"
        f"\n  {ID_OUTPUT_FILE}"
    )

    # --------------------------------------------------------------
    # 5. Read full time series
    # --------------------------------------------------------------
    (
        times,
        longitude,
        latitude,
    ) = read_selected_id_series(
        track_files,
        selected_ids,
        surface="inner",
    )

    (
        r0_times,
        r0_longitude,
        r0_latitude,
    ) = read_selected_id_series(
        track_files,
        selected_ids,
        surface="r0",
    )

    if not np.array_equal(times, r0_times):
        raise ValueError("Inner and R0 time axes do not match.")

    # print_selected_id_time_series(
    #     times,
    #     selected_ids,
    #     longitude,
    #     latitude,
    # )

    # --------------------------------------------------------------
    # 6. Differential-rotation correction
    # --------------------------------------------------------------
    if DIFFROT_REFERENCE_TIME_HOURS is None:

        reference_time_hours = float(
            times[
                0
            ]
        )

    else:

        reference_time_hours = float(
            DIFFROT_REFERENCE_TIME_HOURS
        )

    corrected_longitude = np.full_like(
        longitude,
        np.nan,
        dtype=float,
    )

    for iid in range(
        len(
            selected_ids
        )
    ):

        corrected_longitude[
            iid
        ] = remove_differential_rotation_drift(
            longitude[
                iid
            ],
            latitude[
                iid
            ],
            times,
            reference_time_hours,
        )

    corrected_r0_longitude = np.full_like(
        r0_longitude,
        np.nan,
        dtype=float,
    )

    for iid in range(len(selected_ids)):
        corrected_r0_longitude[iid] = remove_differential_rotation_drift(
            r0_longitude[iid],
            r0_latitude[iid],
            times,
            reference_time_hours,
        )

    print(
        "\nDifferential-rotation correction:"
        f"\n  reference time = "
        f"{reference_time_hours:.2f} h"
        f"\n  A = {DIFFROT_A_DEG_PER_DAY:.4f} deg/day"
        f"\n  B = {DIFFROT_B_DEG_PER_DAY:.4f} deg/day"
        f"\n  C = {DIFFROT_C_DEG_PER_DAY:.4f} deg/day"
    )

    # --------------------------------------------------------------
    # 7. Time-series plot
    # --------------------------------------------------------------
    (
        fig_series,
        axes_series,
    ) = plot_tracked_footpoints(
        times,
        selected_ids,
        corrected_longitude,
        latitude,
    )

    (
        fig_r0_series,
        axes_r0_series,
    ) = plot_tracked_footpoints(
        times,
        selected_ids,
        corrected_r0_longitude,
        r0_latitude,
        surface_label=f"R0 = {R0:g} Rs",
    )

    # --------------------------------------------------------------
    # 9. Save / show
    # --------------------------------------------------------------
    if SAVE_OR_NOT:

        TIME_SERIES_OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        add_figure_provenance(fig_series, "plot_tracked_crossings_connectivity_series.py")
        fig_series.savefig(
            TIME_SERIES_OUTPUT_FILE,
            dpi=DPI,
            bbox_inches="tight",
        )

        add_figure_provenance(fig_r0_series, "plot_tracked_crossings_connectivity_series.py")
        fig_r0_series.savefig(
            R0_TIME_SERIES_OUTPUT_FILE,
            dpi=DPI,
            bbox_inches="tight",
        )

        print(
            "\nSaved:"
            f"\n  {TIME_SERIES_OUTPUT_FILE}"
            f"\n  {R0_TIME_SERIES_OUTPUT_FILE}"
            f"\n  {ID_OUTPUT_FILE}"
        )

        plt.close(
            fig_series
        )

        plt.close(
            fig_r0_series
        )

    else:

        plt.show()


if __name__ == "__main__":
    main()
