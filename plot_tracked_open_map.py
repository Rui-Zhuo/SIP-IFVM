"""
plot_tracked_open_field_map.py

Plot open/closed topology and tracked ID positions at any requested time.

The selected IDs are read from:
    tracked_open_field_id.npz

For every time listed in PLOT_TIME_HOURS, this script reads:
    open_closed_time.{time:.2f}.npz
    track_open.time.{time:.2f}.r0.{R0:g}.npz

and plots:
    - open_closed_map as pcolormesh
    - Br=0 contour
    - selected IDs at their r_index=0 footpoint positions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from config import OPEN_CLOSED_DIR, TRACK_OPEN_DIR, WORK_ROOT


# ======================================================================
# CONFIGURATION
# ======================================================================

WORK_DIR = WORK_ROOT
TRACK_DIR = TRACK_OPEN_DIR

ID_FILE = (
    WORK_DIR
    / "tracked_open_field_id.npz"
)

R0 = 10.0

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
            f"tracked_open_map.time.{simulation_hours:.2f}."
            f"r0.{R0:g}.png"
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


# ======================================================================
# MAIN
# ======================================================================

def main():
    selected_ids = load_selected_ids(
        ID_FILE
    )

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
