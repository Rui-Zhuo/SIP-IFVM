"""
plot_sip_open_closed_field_202608272304.py

Batch classify open/closed magnetic topology from merged SIP-IFVM HDF5 files.

Compared with the previous version, this script now:

1. Processes all merged HDF5 files in DATA_DIR.
2. Uses three tracing surfaces for each file:
       row 1: r_index = 0
       row 2: r = 5 Rs   (nearest radial layer; configurable by value)
       row 3: r = 10 Rs  (nearest radial layer; configurable by value)
3. Produces a 3 x 2 figure:
       left  column: Br on that tracing surface
       right column: open/closed map on the r_index=0 surface
                     (-1 / 0 / +1), obtained from field lines started
                     at the corresponding tracing surface, smoothed,
                     and overlaid with the smoothed Br=0 contour from
                     r_index=0
4. Saves open-field correspondence for each tracing surface:
       (theta, phi) on tracing surface
       (theta, phi) on r_index=0 footpoint surface
       open polarity label (+1/-1)

The classification itself is:
    +1 : open field, positive Br polarity at the r_index=0 footpoint
    -1 : open field, negative Br polarity at the r_index=0 footpoint
     0 : closed field

Unresolved / unvisited points are treated as 0 before categorical smoothing.

Outputs: `open_closed_time.<t>.npz` topology maps and matching
`open_closed_time.<t>.png` diagnostic figures.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter, convolve

from config import FULL_MERGED_DIR, FULL_OPEN_CLOSED_DIR, GRID_FILE
from figure_provenance import add_figure_provenance
from read_merged_sip_data import (
    read_merged_grid,
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
PROCESS_MODE = "all"

DATA_DIR = FULL_MERGED_DIR / "82d1to132"

# Used only when PROCESS_MODE = "single".
SINGLE_DATA_FILE = "82_00_merged_spherical.h5"

OUTPUT_DIR = FULL_OPEN_CLOSED_DIR / "82d1to132"

# ----------------------------------------------------------------------
# Simulation time reference
# ----------------------------------------------------------------------

SIMULATION_START_DATETIME = datetime(
    2026, 4, 8, 16, 0
)


def simulation_datetime_from_filename(filename: Path) -> datetime:
    """Convert filename simulation time to calendar time."""
    return SIMULATION_START_DATETIME + timedelta(
        hours=simulation_hours_from_filename(filename)
    )


def format_simulation_datetime(filename: Path) -> str:
    """Return the standard datetime label used in figures."""
    hours = simulation_hours_from_filename(filename)
    simulation_datetime = simulation_datetime_from_filename(filename)
    return f"{simulation_datetime:%Y-%m-%d %H:%M} ({hours:.2f}h)"

# ----------------------------------------------------------------------
# Field-line geometry
# ----------------------------------------------------------------------

OPEN_RADIUS = 15.0
INNER_RADIAL_INDEX = 0

TRACING_SURFACES = [
    {
        "tag": "rindex0",
        "mode": "index",
        "value": 0,
        "title": "r_index = 0",
    },
    {
        "tag": "r1",
        "mode": "radius",
        "value": 2.5,
        "title": "r = 2.5 Rs",
    },
    {
        "tag": "r2",
        "mode": "radius",
        "value": 5.0,
        "title": "r = 5.0 Rs",
    },
]

INNER_TRACE_OFFSET_FRACTION = 0.05

# ----------------------------------------------------------------------
# Fast batch tracing
# ----------------------------------------------------------------------

BATCH_SIZE = 4096
STEP_SIZE = 0.03
MAX_STEPS = 6000

INNER_TOL = 2.0e-3
OPEN_TOL = 2.0e-3

THETA_EPS = 1.0e-7
MIN_BMAG = 1.0e-14

# ----------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------

CLASS_SMOOTH_SIZE = 3
CLASS_SMOOTH_PASSES = 2
BR_CONTOUR_SMOOTH_SIGMA = 1.2

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

BR_CMAP = "seismic"

CLASS_CMAP = ListedColormap(
    [
        "royalblue",
        "lightgray",
        "firebrick",
    ]
)

FIGSIZE = (10, 8)
COLORBAR_SHRINK = 0.80
SUPTITLE_FONTSIZE = 13
DPI = 250


# ======================================================================
# FAST VECTORIZED TRILINEAR INTERPOLATION
# ======================================================================

class SphericalBInterpolator:
    """
    Fast vectorized trilinear interpolation of Br, Btheta, Bphi.

    r and theta may be nonuniform.
    phi is periodic.
    """

    def __init__(
        self,
        r,
        theta,
        phi,
        Br,
        Btheta,
        Bphi,
    ):
        self.r = np.asarray(
            r,
            dtype=float,
        )

        self.theta = np.asarray(
            theta,
            dtype=float,
        )

        self.phi = np.asarray(
            phi,
            dtype=float,
        )

        self.Br = np.asarray(
            Br,
            dtype=float,
        )

        self.Bt = np.asarray(
            Btheta,
            dtype=float,
        )

        self.Bp = np.asarray(
            Bphi,
            dtype=float,
        )

        self.nr = len(self.r)
        self.nt = len(self.theta)
        self.np = len(self.phi)

        if self.nr < 2 or self.nt < 2 or self.np < 2:
            raise ValueError(
                "Need at least 2 points in each dimension."
            )

        if not np.all(
            np.diff(self.r) > 0.0
        ):
            raise ValueError(
                "r must be strictly increasing."
            )

        if not np.all(
            np.diff(self.theta) > 0.0
        ):
            raise ValueError(
                "theta must be strictly increasing."
            )

        if not np.all(
            np.diff(self.phi) > 0.0
        ):
            raise ValueError(
                "phi must be strictly increasing."
            )

        self.phi0 = float(
            self.phi[0]
        )

        self.period = 2.0 * np.pi

        dphi = np.diff(
            self.phi
        )

        if not np.allclose(
            dphi,
            dphi[0],
            rtol=1.0e-8,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Fast interpolator currently expects uniformly spaced phi."
            )

        self.dphi = float(
            dphi[0]
        )

    def _indices_weights(
        self,
        states,
    ):
        states = np.asarray(
            states,
            dtype=float,
        )

        rr = states[:, 0]

        tt = np.clip(
            states[:, 1],
            self.theta[0],
            self.theta[-1],
        )

        pp = (
            states[:, 2]
            - self.phi0
        ) % self.period + self.phi0

        ir = np.searchsorted(
            self.r,
            rr,
            side="right",
        ) - 1

        ir = np.clip(
            ir,
            0,
            self.nr - 2,
        )

        r0 = self.r[ir]
        r1 = self.r[ir + 1]
        wr = (
            (rr - r0)
            / (r1 - r0)
        )

        it = np.searchsorted(
            self.theta,
            tt,
            side="right",
        ) - 1

        it = np.clip(
            it,
            0,
            self.nt - 2,
        )

        t0 = self.theta[it]
        t1 = self.theta[it + 1]
        wt = (
            (tt - t0)
            / (t1 - t0)
        )

        phi_pos = (
            (pp - self.phi0)
            / self.dphi
        )

        ip = np.floor(
            phi_pos
        ).astype(np.int64)

        wp = (
            phi_pos
            - np.floor(phi_pos)
        )

        ip0 = (
            ip % self.np
        )

        ip1 = (
            (ip + 1)
            % self.np
        )

        return (
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

    @staticmethod
    def _interp_one(
        field,
        ir,
        it,
        ip0,
        ip1,
        wr,
        wt,
        wp,
    ):
        ir1 = ir + 1
        it1 = it + 1

        c000 = field[ir,  it,  ip0]
        c001 = field[ir,  it,  ip1]
        c010 = field[ir,  it1, ip0]
        c011 = field[ir,  it1, ip1]
        c100 = field[ir1, it,  ip0]
        c101 = field[ir1, it,  ip1]
        c110 = field[ir1, it1, ip0]
        c111 = field[ir1, it1, ip1]

        c00 = c000 * (1.0 - wp) + c001 * wp
        c01 = c010 * (1.0 - wp) + c011 * wp
        c10 = c100 * (1.0 - wp) + c101 * wp
        c11 = c110 * (1.0 - wp) + c111 * wp

        c0 = c00 * (1.0 - wt) + c01 * wt
        c1 = c10 * (1.0 - wt) + c11 * wt

        return c0 * (1.0 - wr) + c1 * wr

    def __call__(
        self,
        states,
    ):
        (
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        ) = self._indices_weights(
            states
        )

        Br = self._interp_one(
            self.Br,
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

        Bt = self._interp_one(
            self.Bt,
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

        Bp = self._interp_one(
            self.Bp,
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

        return np.column_stack(
            (
                Br,
                Bt,
                Bp,
            )
        )


# ======================================================================
# VECTORIZED FIELD-LINE RHS
# ======================================================================

def normalize_states(
    states,
):
    states = np.asarray(
        states,
        dtype=float,
    ).copy()

    states[:, 1] = np.clip(
        states[:, 1],
        THETA_EPS,
        np.pi - THETA_EPS,
    )

    states[:, 2] %= (
        2.0 * np.pi
    )

    return states


def fieldline_rhs_batch(
    states,
    directions,
    interpolator,
):
    """
    Vectorized spherical field-line equations.

    dr/ds     = +/- Br / |B|
    dtheta/ds = +/- Btheta / (r |B|)
    dphi/ds   = +/- Bphi / (r sin(theta) |B|)
    """
    states = normalize_states(
        states
    )

    B = interpolator(
        states
    )

    Br = B[:, 0]
    Bt = B[:, 1]
    Bp = B[:, 2]

    Bmag = np.sqrt(
        Br**2
        + Bt**2
        + Bp**2
    )

    valid = (
        np.all(
            np.isfinite(B),
            axis=1,
        )
        & np.isfinite(Bmag)
        & (Bmag > MIN_BMAG)
        & (states[:, 0] > 0.0)
    )

    rhs = np.full(
        states.shape,
        np.nan,
        dtype=float,
    )

    if not np.any(
        valid
    ):
        return rhs, valid

    rr = states[valid, 0]
    tt = states[valid, 1]

    sin_theta = np.maximum(
        np.sin(tt),
        THETA_EPS,
    )

    scale = (
        directions[valid]
        / Bmag[valid]
    )

    rhs[valid, 0] = (
        scale
        * Br[valid]
    )

    rhs[valid, 1] = (
        scale
        * Bt[valid]
        / rr
    )

    rhs[valid, 2] = (
        scale
        * Bp[valid]
        / (
            rr
            * sin_theta
        )
    )

    return rhs, valid


# ======================================================================
# VECTORIZED RK2
# ======================================================================

def rk2_step_batch(
    states,
    directions,
    interpolator,
    ds,
):
    """
    Midpoint / second-order Runge-Kutta step.
    """
    k1, valid1 = fieldline_rhs_batch(
        states,
        directions,
        interpolator,
    )

    mid = (
        states
        + 0.5
        * ds
        * np.nan_to_num(
            k1,
            nan=0.0,
        )
    )

    mid = normalize_states(
        mid
    )

    k2, valid2 = fieldline_rhs_batch(
        mid,
        directions,
        interpolator,
    )

    valid = (
        valid1
        & valid2
    )

    new_states = (
        states.copy()
    )

    new_states[valid] = (
        states[valid]
        + ds
        * k2[valid]
    )

    new_states = normalize_states(
        new_states
    )

    return new_states, valid


# ======================================================================
# VECTORIZED BOUNDARY CROSSING
# ======================================================================

def interpolate_crossing_batch(
    previous,
    current,
    mask,
    target_r,
):
    p0 = previous[mask]
    p1 = current[mask]

    r0 = p0[:, 0]
    r1 = p1[:, 0]

    denom = (
        r1 - r0
    )

    frac = np.zeros_like(
        r0
    )

    good = (
        np.abs(denom)
        > 1.0e-14
    )

    frac[good] = (
        (target_r - r0[good])
        / denom[good]
    )

    frac = np.clip(
        frac,
        0.0,
        1.0,
    )

    theta = (
        p0[:, 1]
        + frac
        * (
            p1[:, 1]
            - p0[:, 1]
        )
    )

    dphi = np.angle(
        np.exp(
            1j
            * (
                p1[:, 2]
                - p0[:, 2]
            )
        )
    )

    phi = (
        p0[:, 2]
        + frac * dphi
    ) % (
        2.0 * np.pi
    )

    out = np.empty(
        (
            len(theta),
            3,
        ),
        dtype=float,
    )

    out[:, 0] = target_r
    out[:, 1] = theta
    out[:, 2] = phi

    return out


# ======================================================================
# TRACE ONE BATCH OF SEEDS
# ======================================================================

STATUS_ACTIVE = 0
STATUS_INNER = 1
STATUS_OPEN = 2
STATUS_FAILED = 3
STATUS_MAX_STEPS = 4


def trace_seed_batch(
    seeds,
    interpolator,
    r_inner,
    r_grid_max,
):
    """
    Trace +B and -B branches for all seeds simultaneously.
    """
    nseed = len(
        seeds
    )

    states = np.vstack(
        (
            seeds,
            seeds,
        )
    ).astype(
        float,
        copy=True,
    )

    directions = np.concatenate(
        (
            np.ones(
                nseed,
                dtype=float,
            ),
            -np.ones(
                nseed,
                dtype=float,
            ),
        )
    )

    status = np.full(
        2 * nseed,
        STATUS_ACTIVE,
        dtype=np.int8,
    )

    final_states = np.full(
        (
            2 * nseed,
            3,
        ),
        np.nan,
        dtype=float,
    )

    for _ in range(
        MAX_STEPS
    ):
        active_idx = np.flatnonzero(
            status
            == STATUS_ACTIVE
        )

        if active_idx.size == 0:
            break

        previous = (
            states[active_idx].copy()
        )

        new_states, valid = rk2_step_batch(
            previous,
            directions[active_idx],
            interpolator,
            STEP_SIZE,
        )

        invalid_local = (
            ~valid
        )

        if np.any(
            invalid_local
        ):
            bad_global = active_idx[
                invalid_local
            ]

            status[bad_global] = STATUS_FAILED
            final_states[bad_global] = previous[
                invalid_local
            ]

        good_local = valid

        if not np.any(
            good_local
        ):
            continue

        good_global = active_idx[
            good_local
        ]

        prev_good = previous[
            good_local
        ]

        curr_good = new_states[
            good_local
        ]

        states[good_global] = curr_good

        r_now = curr_good[:, 0]

        hit_inner = (
            r_now
            <= r_inner + INNER_TOL
        )

        if np.any(
            hit_inner
        ):
            ids = good_global[
                hit_inner
            ]

            final_states[ids] = interpolate_crossing_batch(
                prev_good,
                curr_good,
                hit_inner,
                r_inner,
            )

            status[ids] = STATUS_INNER

        still = ~hit_inner

        hit_open = (
            still
            & (
                r_now
                >= OPEN_RADIUS
                - OPEN_TOL
            )
        )

        if np.any(
            hit_open
        ):
            ids = good_global[
                hit_open
            ]

            final_states[ids] = interpolate_crossing_batch(
                prev_good,
                curr_good,
                hit_open,
                OPEN_RADIUS,
            )

            status[ids] = STATUS_OPEN

        still = (
            still
            & (~hit_open)
        )

        failed_outer = (
            still
            & (r_now > r_grid_max)
        )

        if np.any(
            failed_outer
        ):
            ids = good_global[
                failed_outer
            ]

            status[ids] = STATUS_FAILED
            final_states[ids] = curr_good[
                failed_outer
            ]

    unresolved = (
        status
        == STATUS_ACTIVE
    )

    if np.any(
        unresolved
    ):
        status[unresolved] = STATUS_MAX_STEPS
        final_states[unresolved] = states[
            unresolved
        ]

    return (
        status[:nseed],
        final_states[:nseed],
        status[nseed:],
        final_states[nseed:],
    )


# ======================================================================
# INDEX / SURFACE HELPERS
# ======================================================================

def nearest_phi_index_vectorized(
    phi_grid,
    phi_values,
):
    phi0 = float(
        phi_grid[0]
    )

    dphi = float(
        phi_grid[1]
        - phi_grid[0]
    )

    nphi = len(
        phi_grid
    )

    idx = np.rint(
        (
            (
                phi_values
                - phi0
            )
            % (
                2.0
                * np.pi
            )
        )
        / dphi
    ).astype(
        np.int64
    )

    return idx % nphi


def nearest_theta_index_vectorized(
    theta_grid,
    theta_values,
):
    idx_right = np.searchsorted(
        theta_grid,
        theta_values,
        side="left",
    )

    idx_right = np.clip(
        idx_right,
        0,
        len(theta_grid) - 1,
    )

    idx_left = np.clip(
        idx_right - 1,
        0,
        len(theta_grid) - 1,
    )

    dl = np.abs(
        theta_values
        - theta_grid[idx_left]
    )

    dr = np.abs(
        theta_values
        - theta_grid[idx_right]
    )

    choose_right = (
        dr < dl
    )

    return np.where(
        choose_right,
        idx_right,
        idx_left,
    )


def map_footpoints_to_indices(
    theta,
    phi,
    footpoints,
):
    j = nearest_theta_index_vectorized(
        theta,
        footpoints[:, 1],
    )

    k = nearest_phi_index_vectorized(
        phi,
        footpoints[:, 2],
    )

    return j, k


def resolve_tracing_surface(
    r,
    spec,
):
    """
    Resolve one tracing-surface specification.

    Returns a dict with:
        tag
        title
        requested_value
        seed_index_plot
        seed_radius_plot
        seed_radius_trace
    """
    mode = spec["mode"]
    value = spec["value"]

    if mode == "index":
        idx = int(value)

        if idx < 0 or idx >= len(r):
            raise IndexError(
                f"Invalid radial index: {idx}"
            )

        seed_radius_plot = float(
            r[idx]
        )

        if idx == INNER_RADIAL_INDEX:
            dr0 = float(
                r[1] - r[0]
            )

            seed_radius_trace = (
                seed_radius_plot
                + INNER_TRACE_OFFSET_FRACTION * dr0
            )
        else:
            seed_radius_trace = seed_radius_plot

        requested_value = float(idx)

    elif mode == "radius":
        requested_value = float(value)

        idx = int(
            np.argmin(
                np.abs(
                    r - requested_value
                )
            )
        )

        seed_radius_plot = float(
            r[idx]
        )

        seed_radius_trace = seed_radius_plot

    else:
        raise ValueError(
            f"Unknown tracing-surface mode: {mode}"
        )

    return {
        "tag": spec["tag"],
        "title": spec["title"],
        "mode": mode,
        "requested_value": requested_value,
        "seed_index_plot": idx,
        "seed_radius_plot": seed_radius_plot,
        "seed_radius_trace": seed_radius_trace,
    }


# ======================================================================
# SMOOTHING
# ======================================================================

def smooth_classification_map(
    values,
    window_size=CLASS_SMOOTH_SIZE,
    passes=CLASS_SMOOTH_PASSES,
):
    """
    Smooth a categorical -1/0/+1 map using local majority voting.

    - NaN is first treated as 0.
    - Longitude is periodic.
    - Theta uses nearest-edge behavior.
    """
    arr = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
    ).astype(np.int8)

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "CLASS_SMOOTH_SIZE must be a positive odd integer."
        )

    kernel = np.ones(
        (window_size, window_size),
        dtype=np.int16,
    )

    for _ in range(int(passes)):
        counts = []

        for label in (-1, 0, 1):
            mask = (
                arr == label
            ).astype(np.int16)

            pad = window_size // 2

            mask_theta = np.pad(
                mask,
                ((pad, pad), (0, 0)),
                mode="edge",
            )

            count_padded = convolve(
                mask_theta,
                kernel,
                mode="wrap",
            )

            count = count_padded[
                pad:pad + arr.shape[0],
                :
            ]

            counts.append(count)

        counts = np.stack(
            counts,
            axis=0,
        )

        labels = np.array(
            [-1, 0, 1],
            dtype=np.int8,
        )

        max_count = np.max(
            counts,
            axis=0,
        )

        new_arr = arr.copy()

        unique_winner = (
            np.sum(
                counts == max_count[None, :, :],
                axis=0,
            )
            == 1
        )

        winner_idx = np.argmax(
            counts,
            axis=0,
        )

        new_arr[unique_winner] = (
            labels[
                winner_idx[unique_winner]
            ]
        )

        arr = new_arr

    return arr


def smooth_br_for_contour(
    Br_surface,
    sigma=BR_CONTOUR_SMOOTH_SIGMA,
):
    """
    Light Gaussian smoothing for the Br=0 contour.

    Theta uses nearest-edge behavior and longitude is periodic.
    """
    Br_surface = np.asarray(
        Br_surface,
        dtype=float,
    )

    pad = max(
        2,
        int(np.ceil(3.0 * sigma)),
    )

    padded = np.pad(
        Br_surface,
        ((pad, pad), (0, 0)),
        mode="edge",
    )

    smoothed = gaussian_filter(
        padded,
        sigma=(sigma, sigma),
        mode="wrap",
    )

    return smoothed[
        pad:pad + Br_surface.shape[0],
        :
    ]


# ======================================================================
# CLASSIFICATION ON ONE TRACING SURFACE
# ======================================================================

def classify_surface(
    r,
    theta,
    phi,
    Br,
    interpolator,
    surface_spec,
):
    """
    Trace field lines from one selected seed surface, but construct the
    open_closed_map ALWAYS on the r_index=0 surface.

    Seed surface
    ------------
    Determined by surface_spec:
      - r_index=0
      - nearest radial layer to a requested radius such as 5 or 10 Rs

    Output map
    ----------
    Always shape (Ntheta, Nphi) on r_index=0.

    Classification
    --------------
    Open:
        one branch reaches r >= OPEN_RADIUS and the other reaches r_index=0.
        The inner-boundary footpoint is labeled +1/-1 according to Br there.

    Closed:
        both branches reach r_index=0 without reaching OPEN_RADIUS.
        Both inner-boundary footpoints are labeled 0.

    Unresolved / unvisited inner-surface pixels remain NaN in the raw map,
    then become 0 and are smoothed by smooth_classification_map().
    """
    surface_info = resolve_tracing_surface(
        r,
        surface_spec,
    )

    seed_index_plot = surface_info[
        "seed_index_plot"
    ]

    seed_radius_plot = surface_info[
        "seed_radius_plot"
    ]

    seed_radius_trace = surface_info[
        "seed_radius_trace"
    ]

    r_inner = float(
        r[INNER_RADIAL_INDEX]
    )

    r_grid_max = float(
        np.max(r)
    )

    if OPEN_RADIUS > r_grid_max:
        raise ValueError(
            f"OPEN_RADIUS={OPEN_RADIUS} exceeds "
            f"grid maximum r={r_grid_max}."
        )

    Br_inner = Br[
        INNER_RADIAL_INDEX
    ]

    # Left-column Br remains the Br distribution on the tracing surface.
    Br_seed = Br[
        seed_index_plot
    ]

    # Full theta-phi seed grid on the selected tracing surface.
    tt, pp = np.meshgrid(
        theta,
        phi,
        indexing="ij",
    )

    seed_theta_all = tt.ravel()
    seed_phi_all = pp.ravel()

    total = len(
        seed_theta_all
    )

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"Tracing surface: {surface_info['title']}"
        f"\n  mode              = {surface_info['mode']}"
        f"\n  requested value   = {surface_info['requested_value']}"
        f"\n  seed radial index = {seed_index_plot}"
        f"\n  seed plot radius  = {seed_radius_plot:.6g}"
        f"\n  seed trace radius = {seed_radius_trace:.6g}"
        f"\n  map radius        = r_index=0, r={r_inner:.6g} Rs"
        f"\n  seed count        = {total:,}"
        f"\n  batch size        = {BATCH_SIZE:,}"
        f"\n  step size         = {STEP_SIZE}"
    )

    # IMPORTANT:
    # This map is always defined on r_index=0, not on the tracing surface.
    inner_surface_map_raw = np.full(
        (
            len(theta),
            len(phi),
        ),
        np.nan,
        dtype=float,
    )

    # Save open-field correspondence between tracing surface and inner surface.
    open_seed_theta = []
    open_seed_phi = []
    open_inner_theta = []
    open_inner_phi = []
    open_label = []

    n_open = 0
    n_closed = 0
    n_unresolved = 0

    nbatch = int(
        np.ceil(
            total
            / BATCH_SIZE
        )
    )

    for ibatch, start in enumerate(
        range(
            0,
            total,
            BATCH_SIZE,
        ),
        start=1,
    ):
        stop = min(
            start + BATCH_SIZE,
            total,
        )

        n = stop - start

        batch_theta = seed_theta_all[
            start:stop
        ]

        batch_phi = seed_phi_all[
            start:stop
        ]

        seeds = np.column_stack(
            (
                np.full(
                    n,
                    seed_radius_trace,
                    dtype=float,
                ),
                batch_theta,
                batch_phi,
            )
        )

        (
            status_plus,
            state_plus,
            status_minus,
            state_minus,
        ) = trace_seed_batch(
            seeds,
            interpolator,
            r_inner,
            r_grid_max,
        )

        resolved_seed = np.zeros(
            n,
            dtype=bool,
        )

        # ----------------------------------------------------------
        # Open: + branch reaches OPEN_RADIUS, - branch reaches inner.
        # ----------------------------------------------------------
        mask = (
            (status_plus == STATUS_OPEN)
            & (status_minus == STATUS_INNER)
        )

        if np.any(
            mask
        ):
            fp_inner = state_minus[
                mask
            ]

            j, k = map_footpoints_to_indices(
                theta,
                phi,
                fp_inner,
            )

            polarity = np.where(
                Br_inner[j, k] >= 0.0,
                1,
                -1,
            ).astype(np.int8)

            # Open label has priority over a previously assigned closed value.
            inner_surface_map_raw[
                j,
                k,
            ] = polarity

            open_seed_theta.append(
                batch_theta[mask]
            )

            open_seed_phi.append(
                batch_phi[mask]
            )

            open_inner_theta.append(
                fp_inner[:, 1]
            )

            open_inner_phi.append(
                fp_inner[:, 2]
            )

            open_label.append(
                polarity
            )

            resolved_seed[mask] = True

            n_open += int(
                np.count_nonzero(mask)
            )

        # ----------------------------------------------------------
        # Open: - branch reaches OPEN_RADIUS, + branch reaches inner.
        # ----------------------------------------------------------
        mask = (
            (status_minus == STATUS_OPEN)
            & (status_plus == STATUS_INNER)
        )

        if np.any(
            mask
        ):
            fp_inner = state_plus[
                mask
            ]

            j, k = map_footpoints_to_indices(
                theta,
                phi,
                fp_inner,
            )

            polarity = np.where(
                Br_inner[j, k] >= 0.0,
                1,
                -1,
            ).astype(np.int8)

            inner_surface_map_raw[
                j,
                k,
            ] = polarity

            open_seed_theta.append(
                batch_theta[mask]
            )

            open_seed_phi.append(
                batch_phi[mask]
            )

            open_inner_theta.append(
                fp_inner[:, 1]
            )

            open_inner_phi.append(
                fp_inner[:, 2]
            )

            open_label.append(
                polarity
            )

            resolved_seed[mask] = True

            n_open += int(
                np.count_nonzero(mask)
            )

        # ----------------------------------------------------------
        # Closed: both branches reach r_index=0.
        #
        # Both footpoints are written to the SAME r_index=0 map.
        # Closed field does not overwrite a pixel already marked open.
        # ----------------------------------------------------------
        closed = (
            (status_plus == STATUS_INNER)
            & (status_minus == STATUS_INNER)
        )

        if np.any(
            closed
        ):
            fp1 = state_plus[
                closed
            ]

            fp2 = state_minus[
                closed
            ]

            for fp_inner in (
                fp1,
                fp2,
            ):
                j, k = map_footpoints_to_indices(
                    theta,
                    phi,
                    fp_inner,
                )

                current = inner_surface_map_raw[
                    j,
                    k,
                ]

                write = (
                    ~np.isfinite(
                        current
                    )
                )

                if np.any(
                    write
                ):
                    inner_surface_map_raw[
                        j[write],
                        k[write],
                    ] = 0

            resolved_seed[closed] = True

            n_closed += int(
                np.count_nonzero(closed)
            )

        n_unresolved += int(
            np.count_nonzero(
                ~resolved_seed
            )
        )

        print(
            f"  batch {ibatch:3d}/{nbatch:3d}: "
            f"seeds {start + 1:,}-{stop:,} | "
            f"open={n_open:,}, "
            f"closed={n_closed:,}, "
            f"unresolved={n_unresolved:,}"
        )

    # NaN/unvisited inner-surface pixels -> 0, then categorical smoothing.
    open_closed_map = smooth_classification_map(
        inner_surface_map_raw
    ).astype(
        np.int8
    )

    if open_seed_theta:
        open_seed_theta = np.concatenate(
            open_seed_theta
        )
        open_seed_phi = np.concatenate(
            open_seed_phi
        )
        open_inner_theta = np.concatenate(
            open_inner_theta
        )
        open_inner_phi = np.concatenate(
            open_inner_phi
        )
        open_label = np.concatenate(
            open_label
        ).astype(np.int8)
    else:
        open_seed_theta = np.empty(
            0,
            dtype=float,
        )
        open_seed_phi = np.empty(
            0,
            dtype=float,
        )
        open_inner_theta = np.empty(
            0,
            dtype=float,
        )
        open_inner_phi = np.empty(
            0,
            dtype=float,
        )
        open_label = np.empty(
            0,
            dtype=np.int8,
        )

    return {
        "tag": surface_info["tag"],
        "title": surface_info["title"],
        "mode": surface_info["mode"],
        "requested_value": surface_info["requested_value"],

        # Seed/tracing-surface information.
        "seed_index_plot": seed_index_plot,
        "seed_radius_plot": seed_radius_plot,
        "seed_radius_trace": seed_radius_trace,
        "Br_surface": Br_seed,

        # Map is ALWAYS on r_index=0.
        "map_radial_index": int(INNER_RADIAL_INDEX),
        "map_radius": r_inner,
        "open_closed_map_raw": inner_surface_map_raw,
        "open_closed_map": open_closed_map,

        # Open-field correspondence:
        # tracing surface -> r_index=0.
        "open_seed_theta": open_seed_theta,
        "open_seed_phi": open_seed_phi,
        "open_inner_theta": open_inner_theta,
        "open_inner_phi": open_inner_phi,
        "open_label": open_label,

        "n_open": int(n_open),
        "n_closed": int(n_closed),
        "n_unresolved": int(n_unresolved),
    }


# ======================================================================
# PLOT
# ======================================================================

def wrap_surface_phi(
    phi,
    values,
):
    phi_wrap = np.concatenate(
        (
            phi,
            [phi[0] + 2.0 * np.pi],
        )
    )

    values_wrap = np.concatenate(
        (
            values,
            values[:, :1],
        ),
        axis=1,
    )

    return (
        phi_wrap,
        values_wrap,
    )


def plot_results(
    theta,
    phi,
    inner_br_contour_smooth,
    surface_results,
    output_png,
    datetime_label=None,
):
    """
    Plot a 3x2 figure.

    Rows correspond to the three tracing surfaces.
    Left  column: Br on the tracing surface.
    Right column: open/closed map ALWAYS on r_index=0, obtained from
                  field lines started at the tracing surface, plus the
                  smoothed Br=0 contour from r_index=0.
    """
    nrows = len(
        surface_results
    )

    phi_wrap_base = np.concatenate(
        (
            phi,
            [phi[0] + 2.0 * np.pi],
        )
    )

    phi_deg = np.degrees(
        phi_wrap_base
    )

    theta_deg = np.degrees(
        theta
    )

    _, inner_contour_wrap = wrap_surface_phi(
        phi,
        inner_br_contour_smooth,
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.array(
            [axes]
        )

    class_norm = BoundaryNorm(
        [
            -1.5,
            -0.5,
            0.5,
            1.5,
        ],
        CLASS_CMAP.N,
    )

    for irow, result in enumerate(
        surface_results
    ):
        ax_left = axes[irow, 0]
        ax_right = axes[irow, 1]

        Br_surface = result[
            "Br_surface"
        ]

        open_closed_map = result[
            "open_closed_map"
        ]

        _, Br_wrap = wrap_surface_phi(
            phi,
            Br_surface,
        )

        _, map_wrap = wrap_surface_phi(
            phi,
            open_closed_map,
        )

        # ----------------------------------------------------------
        # Left column: Br
        # ----------------------------------------------------------
        finite_br = Br_surface[
            np.isfinite(
                Br_surface
            )
        ]

        br_absmax = float(
            np.max(
                np.abs(
                    finite_br
                )
            )
        )

        if br_absmax <= 0.0:
            br_absmax = 1.0

        im_left = ax_left.pcolormesh(
            phi_deg,
            theta_deg,
            Br_wrap,
            shading="auto",
            cmap=BR_CMAP,
            vmin=-br_absmax,
            vmax=br_absmax,
        )

        ax_left.set_xlim(
            0.0,
            360.0,
        )

        ax_left.set_ylim(
            180.0,
            0.0,
        )

        ax_left.set_xlabel(
            "phi [deg]"
        )

        ax_left.set_ylabel(
            "theta [deg]"
        )

        ax_left.set_title(
            f"Br, {result['title']}"
        )

        ax_left.set_aspect(
            "equal",
            adjustable="box",
        )

        cbar_left = fig.colorbar(
            im_left,
            ax=ax_left,
            orientation="vertical",
            pad=0.02,
            shrink=COLORBAR_SHRINK,
        )

        cbar_left.set_label(
            "Br [G]"
        )

        # ----------------------------------------------------------
        # Right column: open / closed map
        # ----------------------------------------------------------
        im_right = ax_right.pcolormesh(
            phi_deg,
            theta_deg,
            map_wrap,
            shading="auto",
            cmap=CLASS_CMAP,
            norm=class_norm,
        )

        ax_right.contour(
            phi_deg,
            theta_deg,
            inner_contour_wrap,
            levels=[0.0],
            colors="black",
            linewidths=1.0,
        )

        ax_right.set_xlim(
            0.0,
            360.0,
        )

        ax_right.set_ylim(
            180.0,
            0.0,
        )

        ax_right.set_xlabel(
            "phi [deg]"
        )

        ax_right.set_ylabel(
            "theta [deg]"
        )

        ax_right.set_title(
            f"Map from {result['title']}"
        )

        ax_right.set_aspect(
            "equal",
            adjustable="box",
        )

        cbar_right = fig.colorbar(
            im_right,
            ax=ax_right,
            orientation="vertical",
            pad=0.02,
            shrink=COLORBAR_SHRINK,
            ticks=[-1, 0, 1],
        )

        cbar_right.ax.set_yticklabels(
            [
                "Open (-)",
                "Closed",
                "Open (+)",
            ]
        )

    if datetime_label is not None:
        fig.suptitle(
            datetime_label,
            fontsize=SUPTITLE_FONTSIZE,
        )

    try:
        fig.subplots_adjust(
            top=0.90 if datetime_label is not None else 0.95
        )
    except Exception:
        pass

    add_figure_provenance(fig, "classify_open_closed_states.py")
    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    return fig, axes


# ======================================================================
# SAVE NPZ
# ======================================================================

def save_results_npz(
    filename,
    theta,
    phi,
    surface_results,
    Br_inner_surface,
    Br_inner_contour_smooth,
    inner_radius,
):
    """
    Save all surface results to one NPZ.

    In particular, for each tracing surface, save:
      - Br_surface
      - open_closed_map_raw on r_index=0
      - open_closed_map on r_index=0
      - open-field correspondence:
            (theta, phi) on tracing surface
            (theta, phi) on r_index=0 footpoint surface
            polarity (+1/-1)
    """
    save_dict = {
        "theta": theta,
        "phi": phi,
        "Br_inner_surface": Br_inner_surface,
        "Br_inner_contour_smooth": Br_inner_contour_smooth,
        "open_radius": float(OPEN_RADIUS),
        "inner_radial_index": int(INNER_RADIAL_INDEX),
        "inner_radius": float(inner_radius),
        "batch_size": int(BATCH_SIZE),
        "step_size": float(STEP_SIZE),
        "class_smooth_size": int(CLASS_SMOOTH_SIZE),
        "class_smooth_passes": int(CLASS_SMOOTH_PASSES),
        "br_contour_smooth_sigma": float(BR_CONTOUR_SMOOTH_SIGMA),
        "surface_tags": np.array(
            [result["tag"] for result in surface_results]
        ),
        "surface_titles": np.array(
            [result["title"] for result in surface_results]
        ),
    }

    for result in surface_results:
        tag = result["tag"]

        save_dict[f"{tag}_mode"] = str(
            result["mode"]
        )

        save_dict[f"{tag}_requested_value"] = float(
            result["requested_value"]
        )

        save_dict[f"{tag}_seed_index_plot"] = int(
            result["seed_index_plot"]
        )

        save_dict[f"{tag}_seed_radius_plot"] = float(
            result["seed_radius_plot"]
        )

        save_dict[f"{tag}_seed_radius_trace"] = float(
            result["seed_radius_trace"]
        )

        save_dict[f"{tag}_map_radial_index"] = int(
            result["map_radial_index"]
        )

        save_dict[f"{tag}_map_radius"] = float(
            result["map_radius"]
        )

        save_dict[f"{tag}_Br_surface"] = result[
            "Br_surface"
        ]

        save_dict[f"{tag}_open_closed_map_raw"] = result[
            "open_closed_map_raw"
        ]

        save_dict[f"{tag}_open_closed_map"] = result[
            "open_closed_map"
        ]

        save_dict[f"{tag}_open_seed_theta"] = result[
            "open_seed_theta"
        ]

        save_dict[f"{tag}_open_seed_phi"] = result[
            "open_seed_phi"
        ]

        save_dict[f"{tag}_open_inner_theta"] = result[
            "open_inner_theta"
        ]

        save_dict[f"{tag}_open_inner_phi"] = result[
            "open_inner_phi"
        ]

        save_dict[f"{tag}_open_label"] = result[
            "open_label"
        ]

        save_dict[f"{tag}_n_open"] = int(
            result["n_open"]
        )

        save_dict[f"{tag}_n_closed"] = int(
            result["n_closed"]
        )

        save_dict[f"{tag}_n_unresolved"] = int(
            result["n_unresolved"]
        )

    np.savez_compressed(
        filename,
        **save_dict,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = read_merged_grid(
        filename=GRID_FILE,
        load_cartesian=False,
        load_selection_maps=False,
    )
    r = np.asarray(grid["r"], dtype=float)
    theta = np.asarray(grid["theta"], dtype=float)
    phi = np.asarray(grid["phi"], dtype=float)

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

    for file_index, data_file in enumerate(
        data_files,
        start=1,
    ):
        print(
            "\n"
            + "=" * 72
        )

        print(
            f"Processing file {file_index}/{len(data_files)}:"
            f"\n  {data_file.name}"
        )

        print(
            "=" * 72
        )

        fields = read_merged_physics(
            filename=data_file,
            grid_filename=GRID_FILE,
            load_component_map=False,
            field_names=("Br", "Btheta", "Bphi"),
        )

        Br = fields["Br"]
        Btheta = fields["Btheta"]
        Bphi = fields["Bphi"]

        expected_shape = (
            len(r),
            len(theta),
            len(phi),
        )

        for name in ("Br", "Btheta", "Bphi"):
            array = fields[name]
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name}.shape={array.shape}, "
                    f"expected {expected_shape}"
                )

        print(
            "Loaded magnetic field:"
            f"\n  data file = {data_file}"
            f"\n  shape     = {expected_shape}"
            f"\n  r range   = {r.min():.6g} .. {r.max():.6g} Rs"
        )

        interpolator = SphericalBInterpolator(
            r,
            theta,
            phi,
            Br,
            Btheta,
            Bphi,
        )

        Br_inner_surface = Br[
            INNER_RADIAL_INDEX
        ]

        Br_inner_contour_smooth = smooth_br_for_contour(
            Br_inner_surface
        )

        surface_results = []

        for surface_spec in TRACING_SURFACES:
            result = classify_surface(
                r,
                theta,
                phi,
                Br,
                interpolator,
                surface_spec,
            )

            surface_results.append(
                result
            )

            print(
                f"\nFinal smoothed classification on {result['title']}:"
            )

            for label in (-1, 0, 1):
                count = int(
                    np.count_nonzero(
                        result["open_closed_map"] == label
                    )
                )

                print(
                    f"  label {label:+d}: {count:,} cells"
                )

            print(
                "  saved open-field correspondences:"
                f" {len(result['open_label']):,}"
            )

        simulation_hours = simulation_hours_from_filename(
            data_file
        )

        datetime_label = format_simulation_datetime(
            data_file
        )

        stem = data_file.stem

        output_npz = (
            OUTPUT_DIR
            / f"open_closed_time.{simulation_hours:.2f}.npz"
        )

        save_results_npz(
            output_npz,
            theta,
            phi,
            surface_results,
            Br_inner_surface,
            Br_inner_contour_smooth,
            r[INNER_RADIAL_INDEX],
        )

        output_png = (
            OUTPUT_DIR
            / f"open_closed_time.{simulation_hours:.2f}.png"
        )

        plot_results(
            theta,
            phi,
            Br_inner_contour_smooth,
            surface_results,
            output_png,
            datetime_label=datetime_label,
        )

        print(
            "\nSaved:"
            f"\n  {output_npz}"
            f"\n  {output_png}"
        )

        plt.close(
            "all"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "All merged HDF5 files finished."
    )


if __name__ == "__main__":
    main()
