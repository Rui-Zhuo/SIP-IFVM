"""
plot_sip_slices_vector.py

Plot tangential vector components on a selected spherical shell in the
2D theta-phi coordinate plane.

This script is intended for quantities such as:
    vt, vp
    Bt, Bp

Main features
-------------
1. Two processing modes:
    - process one specified file
    - process all merged HDF5 files in a folder

2. Flexible shell selection:
    - by radial_index
    - by physical radius r [Rs] (nearest radial layer is used)

3. Quiver plot in the theta-phi plane:
    - small black arrows
    - configurable arrow skipping to avoid overcrowding

4. Main configuration is concentrated at the top of the file, similar
   to the scalar plotting script.

Notes
-----
For a vector field on a spherical surface, the plotted horizontal
component in the theta-phi plane is, by default, taken as

    U ~ v_phi / sin(theta)

and the vertical component is

    V ~ v_theta

This makes the arrow direction more consistent with the theta-phi
coordinate plane, because along a spherical surface the physical arc
length in the phi direction is proportional to sin(theta).

If you prefer to plot the raw phi component directly, set

    APPLY_PHI_PROJECTION_CORRECTION = False
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import GRID_FILE, LOCAL_MERGED_DIR, SLICE_DIR
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
PROCESS_MODE = "all"

DATA_DIR = LOCAL_MERGED_DIR

# Used only when PROCESS_MODE = "single".
SINGLE_DATA_FILE = "82_00_merged_spherical.h5"

OUTPUT_DIR = SLICE_DIR

SAVE_OR_NOT = 1

# ----------------------------------------------------------------------
# Simulation time reference
# ----------------------------------------------------------------------

# 0.00 h  -> 2026-04-08 16:00
# 82.00 h -> 2026-04-12 02:00
SIMULATION_START_DATETIME = datetime(
    2026, 4, 8, 16, 0
)

# ----------------------------------------------------------------------
# Shell selection
# ----------------------------------------------------------------------

# "index" or "radius"
RADIAL_SELECTION_MODE = "index"

# Used when RADIAL_SELECTION_MODE = "index"
RADIAL_INDEX = 0

# Used when RADIAL_SELECTION_MODE = "radius"
REQUESTED_RADIUS = 5.0

# ----------------------------------------------------------------------
# Arrow plotting
# ----------------------------------------------------------------------

# Plot every STRIDE_THETA / STRIDE_PHI points.
ARROW_STRIDE_THETA = 4
ARROW_STRIDE_PHI = 4

# If True, the plotted x-component is divided by sin(theta), which is
# more appropriate for a theta-phi coordinate-plane representation.
APPLY_PHI_PROJECTION_CORRECTION = True

# If True, arrows are normalized to unit length so that the plot mainly
# shows direction. If False, arrow length also reflects amplitude.
NORMALIZE_ARROWS = False

# Quiver appearance.
ARROW_COLOR = "k"

# Separate arrow scales for the two panels.
# In matplotlib.quiver(), a SMALLER scale gives LONGER arrows.
MAGNETIC_ARROW_SCALE = 1.00
VELOCITY_ARROW_SCALE = 0.50

# Reference-arrow values shown by quiverkey.
# Units follow the physical units in the merged data:
#     magnetic field -> G
#     velocity       -> km/s
MAGNETIC_QUIVER_KEY_VALUE = 10.0
VELOCITY_QUIVER_KEY_VALUE = 10.0

ARROW_WIDTH = 0.0022
ARROW_HEADWIDTH = 3.2
ARROW_HEADLENGTH = 4.2
ARROW_HEADAxisLength = 4.0
ARROW_PIVOT = "mid"

# Background scalar ranges on the selected shell.
BR_SHELL_CLIM = (-50.0, 50.0)
VR_SHELL_CLIM = (-40.0, 40.0)

# ----------------------------------------------------------------------
# Common plotting parameters
# ----------------------------------------------------------------------

FIGSIZE = (10.0, 9.2)
SUPTITLE_FONTSIZE = 13
DPI = 300

# ----------------------------------------------------------------------
# Which vector pairs to plot
# ----------------------------------------------------------------------

PLOT_VELOCITY_TANGENTIAL = True
VELOCITY_THETA_COMPONENT = "vtheta"
VELOCITY_PHI_COMPONENT = "vphi"
VELOCITY_TITLE = "vtheta / vphi"
VELOCITY_FILENAME_PREFIX = "vt_vp"

PLOT_MAGNETIC_TANGENTIAL = True
MAGNETIC_THETA_COMPONENT = "Btheta"
MAGNETIC_PHI_COMPONENT = "Bphi"
MAGNETIC_TITLE = "Btheta / Bphi"
MAGNETIC_FILENAME_PREFIX = "Bt_Bp"


# ======================================================================
# Filename time conversion
# ======================================================================


def simulation_datetime_from_filename(
    filename: Path,
    start_datetime: datetime = SIMULATION_START_DATETIME,
) -> datetime:
    """
    Convert the simulation time encoded in the filename to calendar datetime.
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
    Return figure annotation text.

    Example
    -------
    82_00_merged_spherical.h5
        -> 2026-04-12 02:00  (82.00 h)
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
# Helper functions
# ======================================================================

def resolve_radial_index(
    r,
    mode=RADIAL_SELECTION_MODE,
    radial_index=RADIAL_INDEX,
    requested_radius=REQUESTED_RADIUS,
):
    """
    Resolve the shell index from either radial index or physical radius.
    """
    r = np.asarray(r, dtype=float)
    mode = str(mode).strip().lower()

    if mode == "index":
        idx = int(radial_index)

        if idx < 0:
            idx = len(r) + idx

        if idx < 0 or idx >= len(r):
            raise IndexError(
                f"radial_index={radial_index} is outside "
                f"the valid range 0 to {len(r)-1}."
            )

        return idx, float(r[idx])

    if mode == "radius":
        rr = float(requested_radius)

        idx = int(
            np.argmin(
                np.abs(r - rr)
            )
        )

        return idx, float(r[idx])

    raise ValueError(
        f'Unknown RADIAL_SELECTION_MODE="{mode}". '
        'Use "index" or "radius".'
    )


def radial_label_for_filename(
    actual_index,
    actual_radius,
    mode=RADIAL_SELECTION_MODE,
    requested_radius=REQUESTED_RADIUS,
):
    """
    Build a short radial label for saved filenames.
    """
    mode = str(mode).strip().lower()

    if mode == "index":
        return f"ri.{actual_index}"

    return f"r.{requested_radius:.2f}_actual.{actual_radius:.2f}"


def _prepare_vector_components_for_plot(
    theta_2d,
    theta_component_2d,
    phi_component_2d,
    apply_phi_projection_correction=APPLY_PHI_PROJECTION_CORRECTION,
    normalize_arrows=NORMALIZE_ARROWS,
):
    """
    Convert tangential components on a spherical shell to components
    suitable for a theta-phi coordinate-plane quiver plot.
    """
    theta_2d = np.asarray(theta_2d, dtype=float)
    theta_component_2d = np.asarray(theta_component_2d, dtype=float)
    phi_component_2d = np.asarray(phi_component_2d, dtype=float)

    v_plot = theta_component_2d.copy()

    if apply_phi_projection_correction:
        sin_theta = np.sin(theta_2d)
        sin_theta = np.where(
            np.abs(sin_theta) < 1.0e-8,
            np.nan,
            sin_theta,
        )
        u_plot = phi_component_2d / sin_theta
    else:
        u_plot = phi_component_2d.copy()

    finite = (
        np.isfinite(u_plot)
        & np.isfinite(v_plot)
    )

    u_plot = np.where(
        finite,
        u_plot,
        np.nan,
    )

    v_plot = np.where(
        finite,
        v_plot,
        np.nan,
    )

    if normalize_arrows:
        amp = np.sqrt(
            u_plot**2
            + v_plot**2
        )

        good = (
            np.isfinite(amp)
            & (amp > 0.0)
        )

        u_new = np.full_like(
            u_plot,
            np.nan,
            dtype=float,
        )

        v_new = np.full_like(
            v_plot,
            np.nan,
            dtype=float,
        )

        u_new[good] = (
            u_plot[good]
            / amp[good]
        )

        v_new[good] = (
            v_plot[good]
            / amp[good]
        )

        u_plot = u_new
        v_plot = v_new

    return u_plot, v_plot


# ======================================================================
# Combined two-panel plotting function
# ======================================================================

def plot_shell_vector_combined(
    data,
    radial_selection_mode=RADIAL_SELECTION_MODE,
    radial_index=RADIAL_INDEX,
    requested_radius=REQUESTED_RADIUS,
    stride_theta=ARROW_STRIDE_THETA,
    stride_phi=ARROW_STRIDE_PHI,
    apply_phi_projection_correction=APPLY_PHI_PROJECTION_CORRECTION,
    normalize_arrows=NORMALIZE_ARROWS,
    magnetic_arrow_scale=MAGNETIC_ARROW_SCALE,
    velocity_arrow_scale=VELOCITY_ARROW_SCALE,
    arrow_color=ARROW_COLOR,
    arrow_width=ARROW_WIDTH,
    arrow_headwidth=ARROW_HEADWIDTH,
    arrow_headlength=ARROW_HEADLENGTH,
    arrow_headaxislength=ARROW_HEADAxisLength,
    arrow_pivot=ARROW_PIVOT,
    figsize=FIGSIZE,
    datetime_label=None,
):
    """
    Plot one 2-row x 1-column figure on the selected spherical shell.

    Top panel
    ---------
    Btheta/Bphi quiver over Br pcolormesh background.

    Bottom panel
    ------------
    vtheta/vphi quiver over Vr pcolormesh background.
    """
    required = (
        "r",
        "theta",
        "phi",
        MAGNETIC_THETA_COMPONENT,
        MAGNETIC_PHI_COMPONENT,
        VELOCITY_THETA_COMPONENT,
        VELOCITY_PHI_COMPONENT,
        "Br",
        "vr",
    )

    for key in required:
        if key not in data:
            raise KeyError(
                f'Missing "{key}". '
                f"Available keys: {list(data.keys())}"
            )

    r = np.asarray(
        data["r"],
        dtype=float,
    )

    theta = np.asarray(
        data["theta"],
        dtype=float,
    )

    phi = np.asarray(
        data["phi"],
        dtype=float,
    )

    shell_index, shell_radius = resolve_radial_index(
        r,
        mode=radial_selection_mode,
        radial_index=radial_index,
        requested_radius=requested_radius,
    )

    # --------------------------------------------------------------
    # Scalar backgrounds
    # --------------------------------------------------------------
    br_shell = np.asarray(
        data["Br"][shell_index, :, :],
        dtype=float,
    )

    vr_shell = np.asarray(
        data["vr"][shell_index, :, :],
        dtype=float,
    )

    # --------------------------------------------------------------
    # Tangential magnetic field
    # --------------------------------------------------------------
    btheta_shell = np.asarray(
        data[MAGNETIC_THETA_COMPONENT][shell_index, :, :],
        dtype=float,
    )

    bphi_shell = np.asarray(
        data[MAGNETIC_PHI_COMPONENT][shell_index, :, :],
        dtype=float,
    )

    # --------------------------------------------------------------
    # Tangential velocity
    # --------------------------------------------------------------
    vtheta_shell = np.asarray(
        data[VELOCITY_THETA_COMPONENT][shell_index, :, :],
        dtype=float,
    )

    vphi_shell = np.asarray(
        data[VELOCITY_PHI_COMPONENT][shell_index, :, :],
        dtype=float,
    )

    theta_2d, phi_2d = np.meshgrid(
        theta,
        phi,
        indexing="ij",
    )

    # Keep the same theta-phi vector convention as the original code.
    b_u, b_v = _prepare_vector_components_for_plot(
        theta_2d,
        btheta_shell,
        bphi_shell,
        apply_phi_projection_correction=apply_phi_projection_correction,
        normalize_arrows=normalize_arrows,
    )

    v_u, v_v = _prepare_vector_components_for_plot(
        theta_2d,
        vtheta_shell,
        vphi_shell,
        apply_phi_projection_correction=apply_phi_projection_correction,
        normalize_arrows=normalize_arrows,
    )

    theta_deg = np.degrees(
        theta
    )

    phi_deg = np.degrees(
        phi
    )

    theta_deg_2d, phi_deg_2d = np.meshgrid(
        theta_deg,
        phi_deg,
        indexing="ij",
    )

    stride_theta = max(
        1,
        int(stride_theta),
    )

    stride_phi = max(
        1,
        int(stride_phi),
    )

    theta_q = theta_deg_2d[
        ::stride_theta,
        ::stride_phi,
    ]

    phi_q = phi_deg_2d[
        ::stride_theta,
        ::stride_phi,
    ]

    b_u_q = b_u[
        ::stride_theta,
        ::stride_phi,
    ]

    b_v_q = b_v[
        ::stride_theta,
        ::stride_phi,
    ]

    v_u_q = v_u[
        ::stride_theta,
        ::stride_phi,
    ]

    v_v_q = v_v[
        ::stride_theta,
        ::stride_phi,
    ]

    valid_b = (
        np.isfinite(b_u_q)
        & np.isfinite(b_v_q)
    )

    valid_v = (
        np.isfinite(v_u_q)
        & np.isfinite(v_v_q)
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
    )

    ax_b, ax_v = axes

    # ============================================================
    # Top: Btheta/Bphi + Br
    # ============================================================
    im_b = ax_b.pcolormesh(
        phi_deg_2d,
        theta_deg_2d,
        br_shell,
        shading="auto",
        cmap="seismic",
        vmin=BR_SHELL_CLIM[0],
        vmax=BR_SHELL_CLIM[1],
    )

    q_b = ax_b.quiver(
        phi_q[valid_b],
        theta_q[valid_b],
        b_u_q[valid_b],
        b_v_q[valid_b],
        angles="xy",
        scale_units="xy",
        scale=magnetic_arrow_scale,
        color=arrow_color,
        width=arrow_width,
        headwidth=arrow_headwidth,
        headlength=arrow_headlength,
        headaxislength=arrow_headaxislength,
        pivot=arrow_pivot,
    )

    ax_b.quiverkey(
        q_b,
        X=0.82,
        Y=1.06,
        U=MAGNETIC_QUIVER_KEY_VALUE,
        label=f"{MAGNETIC_QUIVER_KEY_VALUE:g} G",
        labelpos="E",
        coordinates="axes",
    )

    cbar_b = fig.colorbar(
        im_b,
        ax=ax_b,
        orientation="vertical",
        pad=0.02,
        fraction=0.035,
    )

    cbar_b.set_label(
        "Br [G]"
    )

    ax_b.set_title(
        f"{MAGNETIC_TITLE}, r = {shell_radius:.4g} Rs"
    )

    # ============================================================
    # Bottom: vtheta/vphi + Vr
    # ============================================================
    im_v = ax_v.pcolormesh(
        phi_deg_2d,
        theta_deg_2d,
        vr_shell,
        shading="auto",
        cmap="seismic",
        vmin=VR_SHELL_CLIM[0],
        vmax=VR_SHELL_CLIM[1],
    )

    q_v = ax_v.quiver(
        phi_q[valid_v],
        theta_q[valid_v],
        v_u_q[valid_v],
        v_v_q[valid_v],
        angles="xy",
        scale_units="xy",
        scale=velocity_arrow_scale,
        color=arrow_color,
        width=arrow_width,
        headwidth=arrow_headwidth,
        headlength=arrow_headlength,
        headaxislength=arrow_headaxislength,
        pivot=arrow_pivot,
    )

    ax_v.quiverkey(
        q_v,
        X=0.82,
        Y=1.06,
        U=VELOCITY_QUIVER_KEY_VALUE,
        label=f"{VELOCITY_QUIVER_KEY_VALUE:g} km/s",
        labelpos="E",
        coordinates="axes",
    )

    cbar_v = fig.colorbar(
        im_v,
        ax=ax_v,
        orientation="vertical",
        pad=0.02,
        fraction=0.035,
    )

    cbar_v.set_label(
        "Vr [km/s]"
    )

    ax_v.set_title(
        f"{VELOCITY_TITLE}, r = {shell_radius:.4g} Rs"
    )

    # ============================================================
    # Common axis formatting
    # ============================================================
    for ax in axes:
        ax.set_xlabel(
            "phi [deg]"
        )

        ax.set_ylabel(
            "theta [deg]"
        )

        ax.set_xlim(
            0.0,
            360.0,
        )

        ax.set_ylim(
            180.0,
            0.0,
        )

        ax.set_xticks(
            np.arange(
                0.0,
                361.0,
                60.0,
            )
        )

        ax.set_yticks(
            np.arange(
                0.0,
                181.0,
                30.0,
            )
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

    if datetime_label is not None:
        fig.suptitle(
            datetime_label,
            fontsize=SUPTITLE_FONTSIZE,
        )

    fig.subplots_adjust(
        hspace=0.42,
        top=0.93 if datetime_label is not None else 0.97,
        bottom=0.07,
    )

    info = {
        "shell_index": shell_index,
        "shell_radius": shell_radius,
        "radial_selection_mode": radial_selection_mode,
    }

    return fig, axes, info


# ======================================================================
# Main script
# ======================================================================

if __name__ == "__main__":

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

    if SAVE_OR_NOT:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if PLOT_VELOCITY_TANGENTIAL:
            (
                OUTPUT_DIR
                / VELOCITY_FILENAME_PREFIX
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if PLOT_MAGNETIC_TANGENTIAL:
            (
                OUTPUT_DIR
                / MAGNETIC_FILENAME_PREFIX
            ).mkdir(
                parents=True,
                exist_ok=True,
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

        # ----------------------------------------------------------
        # Combined tangential-vector figure
        # ----------------------------------------------------------
        fig, axes, info = plot_shell_vector_combined(
            data,
            radial_selection_mode=RADIAL_SELECTION_MODE,
            radial_index=RADIAL_INDEX,
            requested_radius=REQUESTED_RADIUS,
            stride_theta=ARROW_STRIDE_THETA,
            stride_phi=ARROW_STRIDE_PHI,
            apply_phi_projection_correction=APPLY_PHI_PROJECTION_CORRECTION,
            normalize_arrows=NORMALIZE_ARROWS,
            magnetic_arrow_scale=MAGNETIC_ARROW_SCALE,
            velocity_arrow_scale=VELOCITY_ARROW_SCALE,
            arrow_color=ARROW_COLOR,
            arrow_width=ARROW_WIDTH,
            arrow_headwidth=ARROW_HEADWIDTH,
            arrow_headlength=ARROW_HEADLENGTH,
            arrow_headaxislength=ARROW_HEADAxisLength,
            arrow_pivot=ARROW_PIVOT,
            figsize=FIGSIZE,
            datetime_label=datetime_label,
        )

        print(
            "  Combined vector shell:"
            f" shell_index = {info['shell_index']},"
            f" shell_radius = {info['shell_radius']:.6g} Rs"
        )

        if SAVE_OR_NOT:
            radial_tag = radial_label_for_filename(
                info["shell_index"],
                info["shell_radius"],
                mode=RADIAL_SELECTION_MODE,
                requested_radius=REQUESTED_RADIUS,
            )

            fig.savefig(
                OUTPUT_DIR
                / (
                    f"Bt_Bp_vt_vp_time."
                    f"{simulation_hours:.2f}."
                    f"idr.{radial_tag}.png"
                ),
                dpi=DPI,
                bbox_inches="tight",
            )

            plt.close(
                "all"
            )
        else:
            plt.show()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "All selected merged HDF5 files finished."
    )
