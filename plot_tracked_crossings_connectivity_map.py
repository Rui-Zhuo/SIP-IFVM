"""
plot_open_boundary_footpoints_crossings.py

Plot open/closed topology and tracked ID positions at any requested time.

The selected footpoints/crossings are read from:
    selected_open_boundary_footpoints_crossings.r.{R0}.npz

For every time listed in PLOT_TIME_HOURS, this script reads:
    open_closed_time.{time:.2f}.npz
    track_open.time.{time:.2f}.r0.{R0:g}.npz

and, in "inner" mode, plots:
    - open_closed_map as pcolormesh
    - Br=0 contour
    - selected IDs at their r_index=0 footpoint positions

In "r0" mode, it reads the initial crossing longitude/latitude and plots
them on the Br distribution at R0.

Outputs: `open_boundary_footpoint_map.time.<t>.crossing-r.<R0>.png` in
inner mode, or `open_boundary_crossing_map.time.<t>.r.<R0>.png` in r0 mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from config import (
    GRID_FILE,
    LOCAL_MERGED_DIR,
    OPEN_CLOSED_DIR,
    TRACK_OPEN_DIR,
    WORK_ROOT,
)
from figure_provenance import add_figure_provenance
from read_merged_sip_data import read_merged_physics


# ======================================================================
# CONFIGURATION
# ======================================================================

WORK_DIR = WORK_ROOT
TRACK_DIR = TRACK_OPEN_DIR
R0 = 10.0

ID_FILE = (
    WORK_DIR
    / f"selected_open_boundary_footpoints_crossings.r.{R0:g}.npz"
)

# "inner": preserve the existing r_index=0 topology maps.
# "r0": plot the saved initial R0 positions on Br(r=R0).
PLOT_MODE = "inner"

# Directory containing xx_yy_merged_spherical.h5 files for R0 Br maps.
MERGED_DIR = LOCAL_MERGED_DIR

# Any desired times [h].
# PLOT_TIME_HOURS = [
#     82.10,
#     84.00,
#     85.50,
#     93.50,
#     94.50,
#     97.50,
#     98.50,
#     99.00,
#     101.50,
#     102.50,
# ]
PLOT_TIME_HOURS = [
    99.00,
    101.50,
    102.50,
]


# Prefix/tag in open_closed_time.xx.xx.npz.
OPEN_CLOSED_TAG = "rindex0"

FIGSIZE = (11, 5.5)

CLASS_CMAP = ListedColormap(
    [
        "royalblue",
        "lightgray",
        "firebrick",
    ]
)

CLASS_NORM = BoundaryNorm(
    [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ],
    CLASS_CMAP.N,
)

BR_CMAP = "seismic"
BR_CLIM_PERCENTILE = 99.5

BR_ZERO_CONTOUR_COLOR = "black"
BR_ZERO_CONTOUR_LINEWIDTH = 1.0

ID_POINT_MARKER = "o"
ID_POINT_SIZE = 45

ADD_ID_TEXT = False
ID_TEXT_FONTSIZE = 8

SAVE_OR_NOT = 1
DPI = 300

def output_filename(
    simulation_hours,
):
    return (
        WORK_DIR
        / (
            f"open_boundary_footpoint_map.time.{simulation_hours:.2f}."
            f"crossing-r.{R0:g}.png"
        )
    )


def r0_output_filename(
    simulation_hours,
):
    return (
        WORK_DIR
        / (
            f"open_boundary_crossing_map.time.{simulation_hours:.2f}."
            f"r.{R0:g}.png"
        )
    )


# ======================================================================
# FILE HELPERS
# ======================================================================

def open_closed_filename(
    simulation_hours,
):
    return (
        OPEN_CLOSED_DIR
        / f"open_closed_time.{simulation_hours:.2f}.npz"
    )


def track_filename(
    simulation_hours,
):
    return (
        TRACK_DIR
        / (
            f"track_open.time.{simulation_hours:.2f}."
            f"r0.{R0:g}.npz"
        )
    )


def merged_filename(
    simulation_hours,
):
    time_tag = f"{float(simulation_hours):.2f}".replace(".", "_")
    return MERGED_DIR / f"{time_tag}_merged_spherical.h5"


# ======================================================================
# LOAD SELECTED IDS
# ======================================================================

def load_selected_ids(
    filename=ID_FILE,
):
    filename = Path(
        filename
    )

    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    with np.load(
        filename,
        allow_pickle=False,
    ) as f:
        if "id" not in f:
            raise KeyError(
                f'Missing "id" in {filename}'
            )

        ids = np.asarray(
            f["id"],
            dtype=np.int64,
        ).copy()

    return ids


def load_initial_r0_positions(
    filename=ID_FILE,
):
    """Read selected IDs and their initial R0 longitude/latitude."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    required = (
        "id",
        "initial_r0_longitude_deg",
        "initial_r0_latitude_deg",
        "reference_time_hours",
        "r0",
    )

    with np.load(filename, allow_pickle=False) as f:
        missing = [key for key in required if key not in f]

        if missing:
            raise KeyError(
                f"Missing {missing} in {filename}. Re-run "
                "plot_tracked_crossings_connectivity_series.py to create the R0 coordinates."
            )

        ids = np.asarray(f["id"], dtype=np.int64).copy()
        longitude_deg = np.asarray(
            f["initial_r0_longitude_deg"],
            dtype=float,
        ).copy()
        latitude_deg = np.asarray(
            f["initial_r0_latitude_deg"],
            dtype=float,
        ).copy()
        reference_time_hours = float(f["reference_time_hours"])
        saved_r0 = float(f["r0"])

    if longitude_deg.shape != ids.shape or latitude_deg.shape != ids.shape:
        raise ValueError(
            "Initial R0 longitude/latitude arrays must match the ID array."
        )

    if not np.isclose(saved_r0, R0, rtol=0.0, atol=1.0e-8):
        raise ValueError(
            f"ID file was created for R0={saved_r0:g} Rs, "
            f"but R0={R0:g} Rs is configured."
        )

    if not (
        np.all(np.isfinite(longitude_deg))
        and np.all(np.isfinite(latitude_deg))
    ):
        raise ValueError("Initial R0 coordinates contain NaN or Inf values.")

    return (
        ids,
        longitude_deg % 360.0,
        latitude_deg,
        reference_time_hours,
    )


# ======================================================================
# LOAD OPEN / CLOSED MAP
# ======================================================================

def load_open_closed_map(
    filename,
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
                    f'Missing "{key}" in {filename}'
                )

        theta = np.asarray(
            f["theta"],
            dtype=float,
        )

        phi = np.asarray(
            f["phi"],
            dtype=float,
        )

        open_closed_map = np.asarray(
            f[map_key],
            dtype=float,
        )

        Br_surface = np.asarray(
            f[br_key],
            dtype=float,
        )

    return (
        theta,
        phi,
        open_closed_map,
        Br_surface,
    )


def load_br_at_r0(
    filename,
    r0=R0,
):
    """Load Br and linearly interpolate it to the requested R0 sphere."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    data = read_merged_physics(
        filename=filename,
        grid_filename=GRID_FILE,
    )

    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    br = np.asarray(data["Br"], dtype=float)

    if not (r[0] <= r0 <= r[-1]):
        raise ValueError(
            f"R0={r0:g} Rs is outside [{r[0]:g}, {r[-1]:g}] Rs."
        )

    upper_index = int(np.searchsorted(r, r0, side="left"))

    if upper_index == 0 or np.isclose(r[upper_index], r0):
        br_r0 = br[upper_index]
        lower_radius = float(r[upper_index])
        upper_radius = float(r[upper_index])
    else:
        lower_index = upper_index - 1
        lower_radius = float(r[lower_index])
        upper_radius = float(r[upper_index])
        weight = (float(r0) - lower_radius) / (upper_radius - lower_radius)
        br_r0 = (1.0 - weight) * br[lower_index] + weight * br[upper_index]

    expected_shape = (theta.size, phi.size)

    if br_r0.shape != expected_shape:
        raise ValueError(
            f"Br(R0).shape={br_r0.shape}, expected {expected_shape}."
        )

    if not np.all(np.isfinite(br_r0)):
        raise ValueError("Br(R0) contains NaN or Inf values.")

    return theta, phi, br_r0, lower_radius, upper_radius


def choose_br_clim(
    br_surface,
):
    """Return a symmetric robust color range for Br [G]."""
    finite = np.asarray(br_surface, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        raise ValueError("Br surface has no finite values.")

    limit = float(np.percentile(np.abs(finite), BR_CLIM_PERCENTILE))

    if limit <= 0.0:
        limit = float(np.max(np.abs(finite)))

    return (-limit, limit) if limit > 0.0 else (-1.0, 1.0)


# ======================================================================
# LOAD TRACKED POSITIONS
# ======================================================================

def load_track_positions(
    filename,
):
    filename = Path(
        filename
    )

    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    required = (
        "id",
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

        ids = np.asarray(
            f["id"],
            dtype=np.int64,
        )

        theta = np.asarray(
            f["inner_theta"],
            dtype=float,
        )

        phi = np.asarray(
            f["inner_phi"],
            dtype=float,
        )

    return (
        ids,
        theta,
        phi,
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


def wrap_phi_surface(
    phi,
    values,
):
    phi_wrap = np.concatenate(
        (
            phi,
            [
                phi[0]
                + 2.0
                * np.pi
            ],
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


# ======================================================================
# PLOT
# ======================================================================

def plot_map(
    theta,
    phi,
    open_closed_map,
    Br_surface,
    selected_ids,
    track_ids,
    track_theta,
    track_phi,
    simulation_hours,
):
    phi_wrap, map_wrap = (
        wrap_phi_surface(
            phi,
            open_closed_map,
        )
    )

    _, br_wrap = (
        wrap_phi_surface(
            phi,
            Br_surface,
        )
    )

    longitude_deg = (
        phi_to_longitude_deg(
            phi_wrap
        )
    )

    longitude_deg[-1] = (
        longitude_deg[0]
        + 360.0
    )

    latitude_deg = (
        theta_to_latitude_deg(
            theta
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        longitude_deg,
        latitude_deg,
        map_wrap,
        shading="auto",
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
    )

    ax.contour(
        longitude_deg,
        latitude_deg,
        br_wrap,
        levels=[0.0],
        colors=BR_ZERO_CONTOUR_COLOR,
        linewidths=BR_ZERO_CONTOUR_LINEWIDTH,
    )

    id_to_index = {
        int(seed_id): int(index)
        for index, seed_id
        in enumerate(track_ids)
    }

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ]

    for iid, seed_id in enumerate(
        selected_ids
    ):
        index = id_to_index.get(
            int(seed_id)
        )

        if index is None:
            continue

        theta_value = float(
            track_theta[index]
        )

        phi_value = float(
            track_phi[index]
        )

        if not (
            np.isfinite(theta_value)
            and np.isfinite(phi_value)
        ):
            continue

        lon = float(
            phi_to_longitude_deg(
                phi_value
            )
        )

        lat = float(
            theta_to_latitude_deg(
                theta_value
            )
        )

        color = default_colors[
            iid
            % len(default_colors)
        ]

        ax.scatter(
            lon,
            lat,
            s=ID_POINT_SIZE,
            marker=ID_POINT_MARKER,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )

        if ADD_ID_TEXT:
            ax.text(
                lon,
                lat,
                f" {int(seed_id)}",
                color=color,
                fontsize=ID_TEXT_FONTSIZE,
                va="center",
                zorder=6,
            )

    ax.set_xlim(
        0.0,
        360.0,
    )

    ax.set_ylim(
        -90.0,
        90.0,
    )

    ax.set_xlabel(
        "Longitude [deg]"
    )

    ax.set_ylabel(
        "Latitude [deg]"
    )

    ax.set_title(
        f"Open/closed topology and tracked IDs "
        f"at {simulation_hours:.2f} h"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        ticks=[
            -1,
            0,
            1,
        ],
    )

    cbar.ax.set_yticklabels(
        [
            "Open (-)",
            "Closed",
            "Open (+)",
        ]
    )

    return (
        fig,
        ax,
    )


def plot_r0_map(
    theta,
    phi,
    br_r0,
    selected_ids,
    longitude_deg,
    latitude_deg,
    simulation_hours,
):
    """Plot saved initial R0 positions over Br interpolated to R0."""
    phi_wrap, br_wrap = wrap_phi_surface(phi, br_r0)
    longitude_axis_deg = phi_to_longitude_deg(phi_wrap)
    longitude_axis_deg[-1] = longitude_axis_deg[0] + 360.0
    latitude_axis_deg = theta_to_latitude_deg(theta)
    br_clim = choose_br_clim(br_r0)

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        longitude_axis_deg,
        latitude_axis_deg,
        br_wrap,
        shading="auto",
        cmap=BR_CMAP,
        vmin=br_clim[0],
        vmax=br_clim[1],
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for iid, (seed_id, lon, lat) in enumerate(
        zip(selected_ids, longitude_deg, latitude_deg)
    ):
        color = default_colors[iid % len(default_colors)]

        ax.scatter(
            lon,
            lat,
            s=ID_POINT_SIZE,
            marker=ID_POINT_MARKER,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )

        if ADD_ID_TEXT:
            ax.text(
                lon,
                lat,
                f" {int(seed_id)}",
                color=color,
                fontsize=ID_TEXT_FONTSIZE,
                va="center",
                zorder=6,
            )

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(
        f"Br and initial tracked IDs at R0 = {R0:g} Rs "
        f"({simulation_hours:.2f} h)"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        pad=0.02,
    )
    cbar.set_label("Br [G]")

    return fig, ax


# ======================================================================
# MAIN
# ======================================================================

def main():
    mode = str(PLOT_MODE).strip().lower()

    if mode not in {"inner", "r0"}:
        raise ValueError("PLOT_MODE must be 'inner' or 'r0'.")

    if mode == "r0":
        (
            selected_ids,
            longitude_deg,
            latitude_deg,
            reference_time_hours,
        ) = load_initial_r0_positions(ID_FILE)

        data_file = merged_filename(reference_time_hours)

        (
            theta,
            phi,
            br_r0,
            lower_radius,
            upper_radius,
        ) = load_br_at_r0(data_file, R0)

        print(
            "R0 map:"
            f"\n  ID file        = {ID_FILE}"
            f"\n  merged data    = {data_file}"
            f"\n  reference time = {reference_time_hours:.2f} h"
            f"\n  Br interpolation= [{lower_radius:.8g}, "
            f"{upper_radius:.8g}] Rs -> {R0:.8g} Rs"
            f"\n  IDs            = {selected_ids.tolist()}"
        )

        fig, ax = plot_r0_map(
            theta,
            phi,
            br_r0,
            selected_ids,
            longitude_deg,
            latitude_deg,
            reference_time_hours,
        )

        if SAVE_OR_NOT:
            output_file = r0_output_filename(reference_time_hours)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            add_figure_provenance(fig, "plot_open_boundary_footpoints_crossings.py")
            fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved:\n  {output_file}")
        else:
            plt.show()

        return

    selected_ids = load_selected_ids(ID_FILE)

    plot_times = np.atleast_1d(
        np.asarray(
            PLOT_TIME_HOURS,
            dtype=float,
        )
    )

    if plot_times.size == 0:
        raise ValueError(
            "PLOT_TIME_HOURS is empty."
        )

    print(
        "Selected IDs:"
        f"\n  ID file = {ID_FILE}"
        f"\n  IDs     = {selected_ids.tolist()}"
        f"\n  times   = {plot_times.tolist()}"
    )

    for itime, simulation_hours in enumerate(
        plot_times,
        start=1,
    ):
        simulation_hours = float(
            simulation_hours
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"Processing {itime}/{len(plot_times)}:"
            f"\n  time = {simulation_hours:.2f} h"
        )

        open_closed_file = (
            open_closed_filename(
                simulation_hours
            )
        )

        track_file = (
            track_filename(
                simulation_hours
            )
        )

        (
            theta,
            phi,
            open_closed_map,
            Br_surface,
        ) = load_open_closed_map(
            open_closed_file,
            OPEN_CLOSED_TAG,
        )

        (
            track_ids,
            track_theta,
            track_phi,
        ) = load_track_positions(
            track_file
        )

        print(
            "Input files:"
            f"\n  open/closed = {open_closed_file}"
            f"\n  track       = {track_file}"
        )

        fig, ax = plot_map(
            theta,
            phi,
            open_closed_map,
            Br_surface,
            selected_ids,
            track_ids,
            track_theta,
            track_phi,
            simulation_hours,
        )

        if SAVE_OR_NOT:
            output_file = output_filename(
                simulation_hours
            )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            add_figure_provenance(fig, "plot_open_boundary_footpoints_crossings.py")
            fig.savefig(
                output_file,
                dpi=DPI,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

            print(
                "Saved:"
                f"\n  {output_file}"
            )

    if not SAVE_OR_NOT:
        plt.show()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "All requested times finished."
    )


if __name__ == "__main__":
    main()
