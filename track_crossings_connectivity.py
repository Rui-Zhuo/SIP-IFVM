"""
track_sip_open_field_202609021919.py

Track open magnetic-field footpoints through a time sequence of merged
SIP-IFVM HDF5 files.

Workflow
--------
At the initial time T0:

1. Put seeds on the R0 spherical surface using the same theta-phi seed
   grid as plot_sip_open_closed_field.py.
2. Give every surviving OPEN seed a globally unique integer ID.
3. Trace +B and -B with the same spherical equations, vectorized
   trilinear interpolation, and midpoint RK2 scheme as the reference
   open/closed script.
4. Keep only field lines for which one branch reaches r_index=0 and the
   other branch reaches OPEN_RADIUS.
5. Save, for every ID:
       - current R0 crossing theta, phi
       - current r_index=0 footpoint theta, phi
       - predicted r, theta, phi after DT_HOURS
6. The predicted position is computed from Vr, Vtheta, Vphi at the
   current R0 crossing.

At every later time:

1. Read the previous output file.
2. Use its predicted next_r/next_theta/next_phi as the starting location.
3. Trace the current magnetic field from that predicted 3-D location.
4. Re-identify:
       - the field-line crossing on R0
       - the r_index=0 footpoint
5. Use Vr, Vtheta, Vphi at the newly identified R0 crossing to predict
   the seed position DT_HOURS later.
6. Save all arrays with the same global IDs.

If an output NPZ already exists, it is loaded and used directly as the
state for the next time, so the expensive tracing step is skipped.

Invalid / unresolved IDs are retained in the arrays but their positions
are stored as NaN. Their IDs are never reused.

Notes
-----
- r is measured in solar radii (Rs).
- velocities are expected in km/s, consistent with the merged SIP-IFVM
  physical files used by the existing plotting scripts.

Outputs: one `track_open.time.<t>.r0.<R0>.npz` file per time, containing
paired r_index=0 footpoints and R0 crossings for persistent global IDs.
- theta is colatitude [rad].
- phi is longitude [rad], periodic over 2*pi.
"""

from __future__ import annotations

from pathlib import Path
import re

import h5py
import numpy as np

from config import FULL_MERGED_DIR, FULL_TRACK_OPEN_DIR, GRID_FILE


# ======================================================================
# CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# Input / output
# ----------------------------------------------------------------------

DATA_DIR = FULL_MERGED_DIR / "82d1to132"
OUTPUT_DIR = FULL_TRACK_OPEN_DIR

# Initial simulation time [h].
T0_HOURS = 82.10

# Time interval between consecutive files / tracking steps [h].
DT_HOURS = 0.10

# Small tolerance used when selecting files on the requested cadence.
TIME_MATCH_TOL = 1.0e-6

# ----------------------------------------------------------------------
# Tracking surface
# ----------------------------------------------------------------------

# Requested tracking sphere [Rs].
R0 = 10.0

# r_index=0 is the inner footpoint surface.
INNER_RADIAL_INDEX = 0

# Open-field threshold, identical in meaning to the reference code.
OPEN_RADIUS = 15.0

# ----------------------------------------------------------------------
# Initial seed grid
# ----------------------------------------------------------------------

# The reference open/closed code uses the full native theta-phi grid.
# Keep both values equal to 1 for exactly the same seed sampling.
INITIAL_SEED_THETA_STEP = 1
INITIAL_SEED_PHI_STEP = 1

# ----------------------------------------------------------------------
# Magnetic-field-line tracing
# ----------------------------------------------------------------------

BATCH_SIZE = 4096

# Same default RK2 step as the reference open/closed script.
FIELDLINE_STEP_SIZE = 0.03

MAX_FIELDLINE_STEPS = 6000

INNER_TOL = 2.0e-3
OPEN_TOL = 2.0e-3
R0_CROSSING_TOL = 2.0e-3

THETA_EPS = 1.0e-7
MIN_BMAG = 1.0e-14

# ----------------------------------------------------------------------
# Velocity advance
# ----------------------------------------------------------------------

# Solar radius used to convert km/s to Rs/hour.
SOLAR_RADIUS_KM = 696300.0

# The requested workflow uses the current R0 velocity to advance the
# position by DT_HOURS. This is a forward Euler step.
VELOCITY_ADVANCE_METHOD = "euler"

MIN_VALID_SIN_THETA = 1.0e-7

# ----------------------------------------------------------------------
# HDF5 dataset names
# ----------------------------------------------------------------------

BR_NAME = "Br"
BTHETA_NAME = "Btheta"
BPHI_NAME = "Bphi"

VR_NAME = "vr"
VTHETA_NAME = "vtheta"
VPHI_NAME = "vphi"


# ======================================================================
# STATUS CODES
# ======================================================================

STATUS_ACTIVE = 0
STATUS_INNER = 1
STATUS_OPEN = 2
STATUS_FAILED = 3
STATUS_MAX_STEPS = 4

TRACK_INVALID = 0
TRACK_OPEN = 1

# IDs that have already produced a first-NaN warning.
WARNED_NAN_IDS = set()


# ======================================================================
# FILE / TIME HELPERS
# ======================================================================

def simulation_hours_from_filename(
    filename,
):
    """
    xx_yy_merged_spherical.h5 -> xx.yy hours.
    """
    filename = Path(
        filename
    )

    match = re.match(
        r"^(\d+)_([0-9]{2})_merged_spherical\.h5$",
        filename.name,
    )

    if match is None:
        raise ValueError(
            f'Cannot parse simulation time from "{filename.name}".'
        )

    return (
        int(match.group(1))
        + int(match.group(2)) / 100.0
    )


def discover_data_files(
    data_dir=DATA_DIR,
):
    """
    Discover and numerically sort all merged HDF5 files.
    """
    data_dir = Path(
        data_dir
    )

    if not data_dir.exists():
        raise FileNotFoundError(
            data_dir
        )

    pattern = re.compile(
        r"^(\d+)_([0-9]{2})_merged_spherical\.h5$"
    )

    matched = []

    for filename in data_dir.iterdir():

        if not filename.is_file():
            continue

        match = pattern.match(
            filename.name
        )

        if match is None:
            continue

        hours = (
            int(match.group(1))
            + int(match.group(2)) / 100.0
        )

        matched.append(
            (
                hours,
                filename,
            )
        )

    matched.sort(
        key=lambda item: item[0]
    )

    return matched


def select_tracking_files(
    data_dir=DATA_DIR,
    t0_hours=T0_HOURS,
    dt_hours=DT_HOURS,
):
    """
    Select files at T0, T0+DT, T0+2DT, ... from all files in DATA_DIR.

    Files not lying on this cadence are ignored.
    """
    discovered = discover_data_files(
        data_dir
    )

    selected = []

    for hours, filename in discovered:

        if hours < t0_hours - TIME_MATCH_TOL:
            continue

        step_float = (
            (hours - t0_hours)
            / dt_hours
        )

        step_round = int(
            np.rint(
                step_float
            )
        )

        expected = (
            t0_hours
            + step_round * dt_hours
        )

        if abs(
            hours - expected
        ) <= TIME_MATCH_TOL:
            selected.append(
                (
                    hours,
                    filename,
                )
            )

    if not selected:
        raise FileNotFoundError(
            f"No files at or after T0={t0_hours:.2f} h "
            f"on a {dt_hours:.2f} h cadence were found in:\n"
            f"{data_dir}"
        )

    if abs(
        selected[0][0] - t0_hours
    ) > TIME_MATCH_TOL:
        raise FileNotFoundError(
            f"Initial file for T0={t0_hours:.2f} h was not found."
        )

    return selected


def format_r0_for_filename(
    r0,
):
    """
    10.0 -> "10"
    10.5 -> "10.5"
    """
    return f"{float(r0):g}"


def output_filename(
    simulation_hours,
):
    return (
        OUTPUT_DIR
        / (
            f"track_open.time.{simulation_hours:.2f}."
            f"r0.{format_r0_for_filename(R0)}.npz"
        )
    )


# ======================================================================
# READ GRID / DATA
# ======================================================================

def read_grid(
    filename=GRID_FILE,
):
    filename = Path(
        filename
    )

    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    with h5py.File(
        filename,
        "r",
    ) as f:

        r = np.asarray(
            f["r"][...],
            dtype=float,
        )

        theta = np.asarray(
            f["theta"][...],
            dtype=float,
        )

        phi = np.asarray(
            f["phi"][...],
            dtype=float,
        )

    return (
        r,
        theta,
        phi,
    )


def read_fields(
    filename,
):
    """
    Read magnetic field and velocity required by this tracker.
    """
    filename = Path(
        filename
    )

    required = (
        BR_NAME,
        BTHETA_NAME,
        BPHI_NAME,
        VR_NAME,
        VTHETA_NAME,
        VPHI_NAME,
    )

    out = {}

    with h5py.File(
        filename,
        "r",
    ) as f:

        for name in required:

            if name not in f:
                raise KeyError(
                    f'Missing dataset "{name}" in {filename}'
                )

            out[name] = np.asarray(
                f[name][...],
                dtype=float,
            )

    return out


def validate_field_shapes(
    fields,
    r,
    theta,
    phi,
):
    expected = (
        len(r),
        len(theta),
        len(phi),
    )

    for name, values in fields.items():

        if values.shape != expected:
            raise ValueError(
                f"{name}.shape={values.shape}, "
                f"expected {expected}"
            )


# ======================================================================
# FAST VECTORIZED SPHERICAL INTERPOLATOR
# ======================================================================

class SphericalVectorInterpolator:
    """
    Vectorized trilinear interpolation of three spherical components.

    r and theta may be nonuniform.
    phi must be uniform and is treated as periodic.
    """

    def __init__(
        self,
        r,
        theta,
        phi,
        component_r,
        component_theta,
        component_phi,
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

        self.Fr = np.asarray(
            component_r,
            dtype=float,
        )

        self.Ft = np.asarray(
            component_theta,
            dtype=float,
        )

        self.Fp = np.asarray(
            component_phi,
            dtype=float,
        )

        self.nr = len(
            self.r
        )

        self.nt = len(
            self.theta
        )

        self.np = len(
            self.phi
        )

        if not np.all(
            np.diff(
                self.r
            ) > 0.0
        ):
            raise ValueError(
                "r must be strictly increasing."
            )

        if not np.all(
            np.diff(
                self.theta
            ) > 0.0
        ):
            raise ValueError(
                "theta must be strictly increasing."
            )

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
                "phi must be uniformly spaced."
            )

        self.phi0 = float(
            self.phi[0]
        )

        self.dphi = float(
            dphi[0]
        )

        self.period = (
            2.0 * np.pi
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
            (
                states[:, 2]
                - self.phi0
            )
            % self.period
            + self.phi0
        )

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

        r0 = self.r[
            ir
        ]

        r1 = self.r[
            ir + 1
        ]

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

        t0 = self.theta[
            it
        ]

        t1 = self.theta[
            it + 1
        ]

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
        ).astype(
            np.int64
        )

        wp = (
            phi_pos
            - np.floor(
                phi_pos
            )
        )

        ip0 = (
            ip
            % self.np
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
        ir1 = (
            ir + 1
        )

        it1 = (
            it + 1
        )

        c000 = field[
            ir,
            it,
            ip0,
        ]

        c001 = field[
            ir,
            it,
            ip1,
        ]

        c010 = field[
            ir,
            it1,
            ip0,
        ]

        c011 = field[
            ir,
            it1,
            ip1,
        ]

        c100 = field[
            ir1,
            it,
            ip0,
        ]

        c101 = field[
            ir1,
            it,
            ip1,
        ]

        c110 = field[
            ir1,
            it1,
            ip0,
        ]

        c111 = field[
            ir1,
            it1,
            ip1,
        ]

        c00 = (
            c000 * (1.0 - wp)
            + c001 * wp
        )

        c01 = (
            c010 * (1.0 - wp)
            + c011 * wp
        )

        c10 = (
            c100 * (1.0 - wp)
            + c101 * wp
        )

        c11 = (
            c110 * (1.0 - wp)
            + c111 * wp
        )

        c0 = (
            c00 * (1.0 - wt)
            + c01 * wt
        )

        c1 = (
            c10 * (1.0 - wt)
            + c11 * wt
        )

        return (
            c0 * (1.0 - wr)
            + c1 * wr
        )

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

        Fr = self._interp_one(
            self.Fr,
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

        Ft = self._interp_one(
            self.Ft,
            ir,
            it,
            ip0,
            ip1,
            wr,
            wt,
            wp,
        )

        Fp = self._interp_one(
            self.Fp,
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
                Fr,
                Ft,
                Fp,
            )
        )


# ======================================================================
# SPHERICAL STATE HELPERS
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


def interpolate_radial_crossing(
    previous,
    current,
    mask,
    target_r,
):
    """
    Interpolate theta/phi when a traced segment crosses target_r.
    """
    p0 = previous[
        mask
    ]

    p1 = current[
        mask
    ]

    r0 = p0[
        :,
        0,
    ]

    r1 = p1[
        :,
        0,
    ]

    denom = (
        r1 - r0
    )

    frac = np.zeros_like(
        r0
    )

    good = (
        np.abs(
            denom
        ) > 1.0e-14
    )

    frac[
        good
    ] = (
        (
            target_r
            - r0[good]
        )
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

    out[:, 0] = (
        target_r
    )

    out[:, 1] = (
        theta
    )

    out[:, 2] = (
        phi
    )

    return out


# ======================================================================
# MAGNETIC FIELD-LINE INTEGRATION
# ======================================================================

def magnetic_rhs_batch(
    states,
    directions,
    magnetic_interpolator,
):
    """
    Same normalized spherical field-line equations as the reference code:

        dr/ds     = +/- Br / |B|
        dtheta/ds = +/- Btheta / (r |B|)
        dphi/ds   = +/- Bphi / (r sin(theta) |B|)
    """
    states = normalize_states(
        states
    )

    B = magnetic_interpolator(
        states
    )

    Br = B[
        :,
        0,
    ]

    Bt = B[
        :,
        1,
    ]

    Bp = B[
        :,
        2,
    ]

    Bmag = np.sqrt(
        Br**2
        + Bt**2
        + Bp**2
    )

    valid = (
        np.all(
            np.isfinite(
                B
            ),
            axis=1,
        )
        & np.isfinite(
            Bmag
        )
        & (
            Bmag
            > MIN_BMAG
        )
        & (
            states[:, 0]
            > 0.0
        )
    )

    rhs = np.full(
        states.shape,
        np.nan,
        dtype=float,
    )

    if not np.any(
        valid
    ):
        return (
            rhs,
            valid,
        )

    rr = states[
        valid,
        0,
    ]

    tt = states[
        valid,
        1,
    ]

    sin_theta = np.maximum(
        np.sin(
            tt
        ),
        THETA_EPS,
    )

    scale = (
        directions[
            valid
        ]
        / Bmag[
            valid
        ]
    )

    rhs[
        valid,
        0,
    ] = (
        scale
        * Br[
            valid
        ]
    )

    rhs[
        valid,
        1,
    ] = (
        scale
        * Bt[
            valid
        ]
        / rr
    )

    rhs[
        valid,
        2,
    ] = (
        scale
        * Bp[
            valid
        ]
        / (
            rr
            * sin_theta
        )
    )

    return (
        rhs,
        valid,
    )


def rk2_fieldline_step_batch(
    states,
    directions,
    magnetic_interpolator,
):
    """
    Midpoint RK2, matching the reference tracer.
    """
    k1, valid1 = magnetic_rhs_batch(
        states,
        directions,
        magnetic_interpolator,
    )

    mid = (
        states
        + 0.5
        * FIELDLINE_STEP_SIZE
        * np.nan_to_num(
            k1,
            nan=0.0,
        )
    )

    mid = normalize_states(
        mid
    )

    k2, valid2 = magnetic_rhs_batch(
        mid,
        directions,
        magnetic_interpolator,
    )

    valid = (
        valid1
        & valid2
    )

    new_states = (
        states.copy()
    )

    new_states[
        valid
    ] = (
        states[
            valid
        ]
        + FIELDLINE_STEP_SIZE
        * k2[
            valid
        ]
    )

    new_states = normalize_states(
        new_states
    )

    return (
        new_states,
        valid,
    )


def trace_batch_inward(
    seeds,
    magnetic_interpolator,
    r_inner,
    r0,
    r_grid_max,
):
    """
    Trace each seed ONLY in the direction that initially points toward
    smaller radius.

    For every seed:

    1. Interpolate Br at the seed.
    2. Choose integration direction:
           Br > 0  -> direction = -1
           Br < 0  -> direction = +1
       so that dr/ds is initially negative.
    3. Trace only this inward branch.
    4. Record the first crossing of R0, but continue tracing.
    5. Stop when reaching r_index=0, failing, or MAX_FIELDLINE_STEPS.

    Returns
    -------
    status
        Final branch status.

    final_states
        Final state at the inner boundary / failed location.

    r0_crossings
        First R0 crossing for each seed.

    r0_found
        Whether an R0 crossing was identified.
    """
    seeds = np.asarray(
        seeds,
        dtype=float,
    )

    nseed = len(
        seeds
    )

    states = seeds.astype(
        float,
        copy=True,
    )

    # --------------------------------------------------------------
    # Determine the inward integration direction from Br at seed.
    # --------------------------------------------------------------
    B_seed = magnetic_interpolator(
        states
    )

    Br_seed = B_seed[
        :,
        0,
    ]

    directions = np.where(
        Br_seed >= 0.0,
        -1.0,
        +1.0,
    )

    finite_seed = (
        np.all(
            np.isfinite(
                B_seed
            ),
            axis=1,
        )
        & np.isfinite(
            Br_seed
        )
    )

    status = np.full(
        nseed,
        STATUS_ACTIVE,
        dtype=np.int8,
    )

    final_states = np.full(
        (
            nseed,
            3,
        ),
        np.nan,
        dtype=float,
    )

    status[
        ~finite_seed
    ] = STATUS_FAILED

    final_states[
        ~finite_seed
    ] = states[
        ~finite_seed
    ]

    r0_crossings = np.full(
        (
            nseed,
            3,
        ),
        np.nan,
        dtype=float,
    )

    r0_found = np.zeros(
        nseed,
        dtype=bool,
    )

    # If a seed is already on R0, that point is the R0 crossing.
    initially_on_r0 = (
        np.abs(
            states[:, 0]
            - r0
        )
        <= R0_CROSSING_TOL
    )

    initially_on_r0 &= (
        status
        == STATUS_ACTIVE
    )

    if np.any(
        initially_on_r0
    ):
        r0_crossings[
            initially_on_r0
        ] = states[
            initially_on_r0
        ]

        r0_crossings[
            initially_on_r0,
            0,
        ] = r0

        r0_found[
            initially_on_r0
        ] = True

    for _ in range(
        MAX_FIELDLINE_STEPS
    ):
        active_idx = np.flatnonzero(
            status
            == STATUS_ACTIVE
        )

        if active_idx.size == 0:
            break

        previous = states[
            active_idx
        ].copy()

        new_states, valid = rk2_fieldline_step_batch(
            previous,
            directions[
                active_idx
            ],
            magnetic_interpolator,
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

            status[
                bad_global
            ] = STATUS_FAILED

            final_states[
                bad_global
            ] = previous[
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

        states[
            good_global
        ] = curr_good

        # ----------------------------------------------------------
        # Record first R0 crossing; do not stop.
        # ----------------------------------------------------------
        missing_r0 = (
            ~r0_found[
                good_global
            ]
        )

        prev_r = prev_good[
            :,
            0,
        ]

        curr_r = curr_good[
            :,
            0,
        ]

        crossed_r0 = (
            missing_r0
            & (
                (
                    prev_r
                    - r0
                )
                * (
                    curr_r
                    - r0
                )
                <= 0.0
            )
            & (
                np.abs(
                    curr_r
                    - prev_r
                )
                > 1.0e-14
            )
        )

        if np.any(
            crossed_r0
        ):
            ids = good_global[
                crossed_r0
            ]

            r0_crossings[
                ids
            ] = interpolate_radial_crossing(
                prev_good,
                curr_good,
                crossed_r0,
                r0,
            )

            r0_found[
                ids
            ] = True

        # ----------------------------------------------------------
        # Inner boundary: successful inward footpoint.
        # ----------------------------------------------------------
        r_now = curr_good[
            :,
            0,
        ]

        hit_inner = (
            r_now
            <= r_inner
            + INNER_TOL
        )

        if np.any(
            hit_inner
        ):
            ids = good_global[
                hit_inner
            ]

            final_states[
                ids
            ] = interpolate_radial_crossing(
                prev_good,
                curr_good,
                hit_inner,
                r_inner,
            )

            status[
                ids
            ] = STATUS_INNER

        still = (
            ~hit_inner
        )

        # ----------------------------------------------------------
        # If the supposedly inward trace moves beyond the grid, fail.
        # ----------------------------------------------------------
        failed_outer = (
            still
            & (
                r_now
                > r_grid_max
            )
        )

        if np.any(
            failed_outer
        ):
            ids = good_global[
                failed_outer
            ]

            status[
                ids
            ] = STATUS_FAILED

            final_states[
                ids
            ] = curr_good[
                failed_outer
            ]

    unresolved = (
        status
        == STATUS_ACTIVE
    )

    if np.any(
        unresolved
    ):
        status[
            unresolved
        ] = STATUS_MAX_STEPS

        final_states[
            unresolved
        ] = states[
            unresolved
        ]

    return (
        status,
        final_states,
        r0_crossings,
        r0_found,
    )




# ======================================================================
# OPEN-FIELD MAPPING
# ======================================================================

def angular_distance_from_seed(
    candidate,
    seed,
):
    """
    Angular separation on the sphere [rad].
    """
    if not np.all(
        np.isfinite(
            candidate
        )
    ):
        return np.inf

    if not np.all(
        np.isfinite(
            seed
        )
    ):
        return np.inf

    t1 = candidate[
        1
    ]

    p1 = candidate[
        2
    ]

    t2 = seed[
        1
    ]

    p2 = seed[
        2
    ]

    cosang = (
        np.cos(
            t1
        )
        * np.cos(
            t2
        )
        + np.sin(
            t1
        )
        * np.sin(
            t2
        )
        * np.cos(
            p1 - p2
        )
    )

    return float(
        np.arccos(
            np.clip(
                cosang,
                -1.0,
                1.0,
            )
        )
    )


def choose_r0_crossing(
    seed,
    plus_crossing,
    plus_found,
    minus_crossing,
    minus_found,
):
    """
    Select the R0 crossing belonging to this field line.

    Normally only one branch crosses R0 when the predicted seed is
    slightly inside/outside R0. If both crossings are available, choose
    the one angularly closest to the supplied starting position.
    """
    if plus_found and not minus_found:
        return plus_crossing

    if minus_found and not plus_found:
        return minus_crossing

    if plus_found and minus_found:

        dplus = angular_distance_from_seed(
            plus_crossing,
            seed,
        )

        dminus = angular_distance_from_seed(
            minus_crossing,
            seed,
        )

        if dplus <= dminus:
            return plus_crossing

        return minus_crossing

    return np.array(
        [
            np.nan,
            np.nan,
            np.nan,
        ],
        dtype=float,
    )


def map_open_seeds(
    seeds,
    magnetic_interpolator,
    r_inner,
    r0,
    r_grid_max,
):
    """
    For arbitrary 3-D starting locations, trace ONLY inward and find:

        - current crossing on R0
        - current r_index=0 footpoint

    A seed is considered valid when:
        - an R0 crossing is identified, and
        - the inward trace reaches r_index=0.

    Returns fixed-length arrays. Invalid IDs remain NaN.
    """
    seeds = np.asarray(
        seeds,
        dtype=float,
    )

    nseed = len(
        seeds
    )

    current_r0 = np.full(
        (
            nseed,
            3,
        ),
        np.nan,
        dtype=float,
    )

    inner_fp = np.full(
        (
            nseed,
            3,
        ),
        np.nan,
        dtype=float,
    )

    track_status = np.full(
        nseed,
        TRACK_INVALID,
        dtype=np.int8,
    )

    valid_input = np.all(
        np.isfinite(
            seeds
        ),
        axis=1,
    )

    valid_indices = np.flatnonzero(
        valid_input
    )

    if valid_indices.size == 0:
        return (
            current_r0,
            inner_fp,
            track_status,
        )

    for start in range(
        0,
        len(valid_indices),
        BATCH_SIZE,
    ):
        ids = valid_indices[
            start:
            start + BATCH_SIZE
        ]

        batch = seeds[
            ids
        ]

        (
            status,
            final_state,
            r0_crossing,
            r0_found,
        ) = trace_batch_inward(
            batch,
            magnetic_interpolator,
            r_inner,
            r0,
            r_grid_max,
        )

        valid_track = (
            (
                status
                == STATUS_INNER
            )
            & r0_found
        )

        local_valid_ids = np.flatnonzero(
            valid_track
        )

        if local_valid_ids.size == 0:
            continue

        global_valid_ids = ids[
            local_valid_ids
        ]

        current_r0[
            global_valid_ids
        ] = r0_crossing[
            local_valid_ids
        ]

        inner_fp[
            global_valid_ids
        ] = final_state[
            local_valid_ids
        ]

        track_status[
            global_valid_ids
        ] = TRACK_OPEN

    return (
        current_r0,
        inner_fp,
        track_status,
    )




# ======================================================================
# INITIAL SEEDS
# ======================================================================

def build_initial_seed_grid(
    theta,
    phi,
):
    """
    Same native theta-phi seed selection as the reference code.

    With both step values equal to 1, every native theta-phi point on
    R0 is used as an initial seed.
    """
    theta_sel = theta[
        ::INITIAL_SEED_THETA_STEP
    ]

    phi_sel = phi[
        ::INITIAL_SEED_PHI_STEP
    ]

    tt, pp = np.meshgrid(
        theta_sel,
        phi_sel,
        indexing="ij",
    )

    seeds = np.column_stack(
        (
            np.full(
                tt.size,
                R0,
                dtype=float,
            ),
            tt.ravel(),
            pp.ravel(),
        )
    )

    return seeds


# ======================================================================
# VELOCITY ADVANCE
# ======================================================================

def advance_r0_seeds_with_velocity(
    r0_states,
    velocity_interpolator,
):
    """
    Advance current R0 crossings by DT_HOURS using Vr,Vtheta,Vphi.

    For forward Euler:

        dr/dt     = Vr
        dtheta/dt = Vtheta / r
        dphi/dt   = Vphi / (r sin(theta))

    with the km/s -> Rs/hour conversion applied explicitly.
    """
    if str(
        VELOCITY_ADVANCE_METHOD
    ).lower() != "euler":
        raise ValueError(
            "Only VELOCITY_ADVANCE_METHOD='euler' is implemented."
        )

    r0_states = np.asarray(
        r0_states,
        dtype=float,
    )

    next_states = np.full(
        r0_states.shape,
        np.nan,
        dtype=float,
    )

    valid = np.all(
        np.isfinite(
            r0_states
        ),
        axis=1,
    )

    if not np.any(
        valid
    ):
        return next_states

    states = r0_states[
        valid
    ].copy()

    velocity = velocity_interpolator(
        states
    )

    finite_velocity = np.all(
        np.isfinite(
            velocity
        ),
        axis=1,
    )

    good_global = np.flatnonzero(
        valid
    )[
        finite_velocity
    ]

    if good_global.size == 0:
        return next_states

    states_good = states[
        finite_velocity
    ]

    velocity_good = velocity[
        finite_velocity
    ]

    rr = states_good[
        :,
        0,
    ]

    tt = states_good[
        :,
        1,
    ]

    vr = velocity_good[
        :,
        0,
    ]

    vt = velocity_good[
        :,
        1,
    ]

    vp = velocity_good[
        :,
        2,
    ]

    conversion = (
        DT_HOURS
        * 3600.0
        / SOLAR_RADIUS_KM
    )

    sin_theta = np.maximum(
        np.abs(
            np.sin(
                tt
            )
        ),
        MIN_VALID_SIN_THETA,
    )

    new = states_good.copy()

    new[
        :,
        0,
    ] = (
        rr
        + vr
        * conversion
    )

    new[
        :,
        1,
    ] = (
        tt
        + vt
        * conversion
        / rr
    )

    new[
        :,
        2,
    ] = (
        states_good[
            :,
            2,
        ]
        + vp
        * conversion
        / (
            rr
            * sin_theta
        )
    )

    new = normalize_states(
        new
    )

    next_states[
        good_global
    ] = new

    return next_states


# ======================================================================
# FIRST-NaN WARNING
# ======================================================================

def warn_first_nan_for_ids(
    ids,
    r0_states,
    inner_states,
    next_states,
    simulation_hours,
):
    """
    Print a warning only the FIRST time each global ID develops NaN in
    any key tracked position.

    Once an ID has been warned, later NaNs for the same ID are silent.
    """
    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    has_nan = (
        np.any(
            ~np.isfinite(
                r0_states
            ),
            axis=1,
        )
        | np.any(
            ~np.isfinite(
                inner_states
            ),
            axis=1,
        )
        | np.any(
            ~np.isfinite(
                next_states
            ),
            axis=1,
        )
    )

    bad_ids = ids[
        has_nan
    ]

    for seed_id in bad_ids:

        seed_id_int = int(
            seed_id
        )

        if seed_id_int in WARNED_NAN_IDS:
            continue

        WARNED_NAN_IDS.add(
            seed_id_int
        )

        print(
            "WARNING: seed ID first became NaN:"
            f"\n  id   = {seed_id_int}"
            f"\n  time = {simulation_hours:.2f} h"
        )


# ======================================================================
# SAVE / LOAD TRACK FILES
# ======================================================================

def save_track_file(
    filename,
    simulation_hours,
    ids,
    r0_states,
    inner_states,
    next_states,
    track_status,
    initial_seed_count,
):
    """
    Save one tracking step.

    Important arrays:
        id
        r0_theta
        r0_phi
        inner_theta
        inner_phi
        next_r
        next_theta
        next_phi
    """
    filename = Path(
        filename
    )

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        filename,

        simulation_hours=float(
            simulation_hours
        ),

        dt_hours=float(
            DT_HOURS
        ),

        requested_r0=float(
            R0
        ),

        inner_radial_index=int(
            INNER_RADIAL_INDEX
        ),

        open_radius=float(
            OPEN_RADIUS
        ),

        initial_seed_count=int(
            initial_seed_count
        ),

        id=np.asarray(
            ids,
            dtype=np.int64,
        ),

        track_status=np.asarray(
            track_status,
            dtype=np.int8,
        ),

        r0_r=np.asarray(
            r0_states[
                :,
                0,
            ],
            dtype=float,
        ),

        r0_theta=np.asarray(
            r0_states[
                :,
                1,
            ],
            dtype=float,
        ),

        r0_phi=np.asarray(
            r0_states[
                :,
                2,
            ],
            dtype=float,
        ),

        inner_r=np.asarray(
            inner_states[
                :,
                0,
            ],
            dtype=float,
        ),

        inner_theta=np.asarray(
            inner_states[
                :,
                1,
            ],
            dtype=float,
        ),

        inner_phi=np.asarray(
            inner_states[
                :,
                2,
            ],
            dtype=float,
        ),

        next_r=np.asarray(
            next_states[
                :,
                0,
            ],
            dtype=float,
        ),

        next_theta=np.asarray(
            next_states[
                :,
                1,
            ],
            dtype=float,
        ),

        next_phi=np.asarray(
            next_states[
                :,
                2,
            ],
            dtype=float,
        ),
    )


def load_track_file(
    filename,
):
    filename = Path(
        filename
    )

    with np.load(
        filename,
        allow_pickle=False,
    ) as f:

        out = {
            key: f[
                key
            ].copy()
            for key in f.files
        }

    return out


def next_states_from_track(
    track,
):
    return np.column_stack(
        (
            track[
                "next_r"
            ],
            track[
                "next_theta"
            ],
            track[
                "next_phi"
            ],
        )
    )


# ======================================================================
# PROCESS ONE TIME
# ======================================================================

def build_interpolators(
    fields,
    r,
    theta,
    phi,
):
    magnetic_interpolator = SphericalVectorInterpolator(
        r,
        theta,
        phi,
        fields[
            BR_NAME
        ],
        fields[
            BTHETA_NAME
        ],
        fields[
            BPHI_NAME
        ],
    )

    velocity_interpolator = SphericalVectorInterpolator(
        r,
        theta,
        phi,
        fields[
            VR_NAME
        ],
        fields[
            VTHETA_NAME
        ],
        fields[
            VPHI_NAME
        ],
    )

    return (
        magnetic_interpolator,
        velocity_interpolator,
    )


def process_initial_time(
    data_file,
    simulation_hours,
    r,
    theta,
    phi,
):
    """
    Initial step:
      full R0 theta-phi seed grid -> keep only open seeds -> assign IDs.
    """
    fields = read_fields(
        data_file
    )

    validate_field_shapes(
        fields,
        r,
        theta,
        phi,
    )

    (
        magnetic_interpolator,
        velocity_interpolator,
    ) = build_interpolators(
        fields,
        r,
        theta,
        phi,
    )

    initial_seeds = build_initial_seed_grid(
        theta,
        phi,
    )

    initial_seed_count = len(
        initial_seeds
    )

    print(
        "\nInitial seed grid:"
        f"\n  R0               = {R0:.8g} Rs"
        f"\n  theta step       = {INITIAL_SEED_THETA_STEP}"
        f"\n  phi step         = {INITIAL_SEED_PHI_STEP}"
        f"\n  candidate seeds  = {initial_seed_count:,}"
    )

    (
        mapped_r0,
        inner_fp,
        status,
    ) = map_open_seeds(
        initial_seeds,
        magnetic_interpolator,
        r_inner=float(
            r[
                INNER_RADIAL_INDEX
            ]
        ),
        r0=float(
            R0
        ),
        r_grid_max=float(
            r.max()
        ),
    )

    open_mask = (
        status
        == TRACK_OPEN
    )

    # At t0 the R0 footpoint is the original seed itself.
    # map_open_seeds returns the same point because the initial states lie
    # exactly on R0, but using the original array makes this explicit.
    r0_open = initial_seeds[
        open_mask
    ]

    inner_open = inner_fp[
        open_mask
    ]

    ids = np.arange(
        np.count_nonzero(
            open_mask
        ),
        dtype=np.int64,
    )

    status_open = np.full(
        len(
            ids
        ),
        TRACK_OPEN,
        dtype=np.int8,
    )

    next_states = advance_r0_seeds_with_velocity(
        r0_open,
        velocity_interpolator,
    )

    print(
        "Initial open-field mapping:"
        f"\n  open seeds       = {len(ids):,}"
        f"\n  rejected seeds   = "
        f"{initial_seed_count - len(ids):,}"
    )

    return (
        ids,
        r0_open,
        inner_open,
        next_states,
        status_open,
        initial_seed_count,
    )


def process_later_time(
    data_file,
    simulation_hours,
    previous_track,
    r,
    theta,
    phi,
):
    """
    Later step:
      previous next-state -> current field-line R0/inner footpoints ->
      velocity advance to next time.
    """
    fields = read_fields(
        data_file
    )

    validate_field_shapes(
        fields,
        r,
        theta,
        phi,
    )

    (
        magnetic_interpolator,
        velocity_interpolator,
    ) = build_interpolators(
        fields,
        r,
        theta,
        phi,
    )

    ids = np.asarray(
        previous_track[
            "id"
        ],
        dtype=np.int64,
    )

    predicted_states = next_states_from_track(
        previous_track
    )

    (
        r0_states,
        inner_states,
        track_status,
    ) = map_open_seeds(
        predicted_states,
        magnetic_interpolator,
        r_inner=float(
            r[
                INNER_RADIAL_INDEX
            ]
        ),
        r0=float(
            R0
        ),
        r_grid_max=float(
            r.max()
        ),
    )

    next_states = advance_r0_seeds_with_velocity(
        r0_states,
        velocity_interpolator,
    )

    n_open = int(
        np.count_nonzero(
            track_status
            == TRACK_OPEN
        )
    )

    n_total = len(
        ids
    )

    print(
        "Current tracking result:"
        f"\n  active open IDs  = {n_open:,}"
        f"\n  invalid IDs      = {n_total - n_open:,}"
        f"\n  total IDs        = {n_total:,}"
    )

    initial_seed_count = int(
        np.asarray(
            previous_track[
                "initial_seed_count"
            ]
        ).item()
    )

    return (
        ids,
        r0_states,
        inner_states,
        next_states,
        track_status,
        initial_seed_count,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    r, theta, phi = read_grid(
        GRID_FILE
    )

    if not (
        float(
            r.min()
        )
        < R0
        < float(
            r.max()
        )
    ):
        raise ValueError(
            f"R0={R0} Rs is outside the grid radial range "
            f"[{r.min()}, {r.max()}]."
        )

    if OPEN_RADIUS > float(
        r.max()
    ):
        raise ValueError(
            f"OPEN_RADIUS={OPEN_RADIUS} exceeds "
            f"grid maximum r={r.max()}."
        )

    if R0 >= OPEN_RADIUS:
        raise ValueError(
            "R0 must be smaller than OPEN_RADIUS."
        )

    selected_files = select_tracking_files(
        DATA_DIR,
        T0_HOURS,
        DT_HOURS,
    )

    print(
        "Tracking configuration:"
        f"\n  DATA_DIR      = {DATA_DIR}"
        f"\n  OUTPUT_DIR    = {OUTPUT_DIR}"
        f"\n  T0            = {T0_HOURS:.2f} h"
        f"\n  DT            = {DT_HOURS:.2f} h"
        f"\n  R0            = {R0:.8g} Rs"
        f"\n  inner radius  = {r[INNER_RADIAL_INDEX]:.8g} Rs"
        f"\n  open radius   = {OPEN_RADIUS:.8g} Rs"
        f"\n  files         = {len(selected_files)}"
    )

    previous_track = None
    previous_time = None

    for file_index, (
        simulation_hours,
        data_file,
    ) in enumerate(
        selected_files,
        start=1,
    ):

        print(
            "\n"
            + "=" * 76
        )

        print(
            f"Time {file_index}/{len(selected_files)}: "
            f"{simulation_hours:.2f} h"
            f"\n  data = {data_file.name}"
        )

        output_file = output_filename(
            simulation_hours
        )

        # ----------------------------------------------------------
        # Existing output: skip expensive processing and use its
        # next-state as the input for the following time.
        # ----------------------------------------------------------
        if output_file.exists():

            print(
                "Output already exists -> skip tracing:"
                f"\n  {output_file}"
            )

            previous_track = load_track_file(
                output_file
            )

            cached_ids = np.asarray(
                previous_track["id"],
                dtype=np.int64,
            )

            cached_nan = (
                ~np.isfinite(previous_track["r0_theta"])
                | ~np.isfinite(previous_track["r0_phi"])
                | ~np.isfinite(previous_track["inner_theta"])
                | ~np.isfinite(previous_track["inner_phi"])
                | ~np.isfinite(previous_track["next_r"])
                | ~np.isfinite(previous_track["next_theta"])
                | ~np.isfinite(previous_track["next_phi"])
            )

            WARNED_NAN_IDS.update(
                int(seed_id)
                for seed_id in cached_ids[cached_nan]
            )

            saved_time = float(
                np.asarray(
                    previous_track[
                        "simulation_hours"
                    ]
                ).item()
            )

            if abs(
                saved_time
                - simulation_hours
            ) > TIME_MATCH_TOL:
                raise ValueError(
                    f"Saved file time {saved_time:.8f} h does not "
                    f"match expected {simulation_hours:.8f} h."
                )

            previous_time = simulation_hours

            continue

        # ----------------------------------------------------------
        # Missing output: calculate it.
        # ----------------------------------------------------------
        if file_index == 1:

            (
                ids,
                r0_states,
                inner_states,
                next_states,
                track_status,
                initial_seed_count,
            ) = process_initial_time(
                data_file,
                simulation_hours,
                r,
                theta,
                phi,
            )

        else:

            if previous_track is None:
                raise RuntimeError(
                    "Previous tracking state is unavailable."
                )

            if previous_time is None:
                raise RuntimeError(
                    "Previous tracking time is unavailable."
                )

            expected_time = (
                previous_time
                + DT_HOURS
            )

            if abs(
                simulation_hours
                - expected_time
            ) > TIME_MATCH_TOL:
                raise RuntimeError(
                    "Time sequence is not continuous:"
                    f"\n  previous = {previous_time:.8f} h"
                    f"\n  current  = {simulation_hours:.8f} h"
                    f"\n  expected = {expected_time:.8f} h"
                )

            (
                ids,
                r0_states,
                inner_states,
                next_states,
                track_status,
                initial_seed_count,
            ) = process_later_time(
                data_file,
                simulation_hours,
                previous_track,
                r,
                theta,
                phi,
            )

        warn_first_nan_for_ids(
            ids,
            r0_states,
            inner_states,
            next_states,
            simulation_hours,
        )

        save_track_file(
            output_file,
            simulation_hours,
            ids,
            r0_states,
            inner_states,
            next_states,
            track_status,
            initial_seed_count,
        )

        print(
            "Saved:"
            f"\n  {output_file}"
        )

        previous_track = load_track_file(
            output_file
        )

        previous_time = simulation_hours

    print(
        "\n"
        + "=" * 76
    )

    print(
        "Tracking finished."
    )


if __name__ == "__main__":
    main()
