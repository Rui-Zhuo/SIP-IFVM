"""Plot LCT-selected open-boundary footpoints at r_index=0 or R0.

Dependencies: numpy==1.26.4, matplotlib==3.10.8.
The selected LCT footpoints and their R0 crossings are read from
track_footpoints_lct_connectivity.py output.

Outputs: `lct_open_boundary_footpoint_map.time.<t>.r.<R0>.png` or
`lct_open_boundary_crossing_map.time.<t>.r.<R0>.png`, depending on mode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from config import GRID_FILE, LOCAL_MERGED_DIR, OPEN_CLOSED_DIR, WORK_ROOT
from figure_provenance import add_figure_provenance
from read_merged_sip_data import read_merged_physics


# ======================================================================
# CONFIGURATION
# ======================================================================

SELECTION_FILE = WORK_ROOT / "lct_open_boundary" / "selected_lct_open_boundary_footpoints_crossings.r.10.npz"
OUTPUT_DIR = WORK_ROOT / "lct_open_boundary"
MERGED_DIR = LOCAL_MERGED_DIR
R0 = 10.0

# "inner" keeps the existing open/closed map style.
# "r0" plots Br interpolated to R0 and the matched R0 crossings.
PLOT_MODE = "inner"
PLOT_TIME_HOURS = [82.10]

FIGSIZE = (11, 5.5)
CLASS_CMAP = ListedColormap(["royalblue", "lightgray", "firebrick"])
CLASS_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CLASS_CMAP.N)
BR_CMAP = "seismic"
BR_CLIM_PERCENTILE = 99.5
BR_ZERO_CONTOUR_COLOR = "black"
BR_ZERO_CONTOUR_LINEWIDTH = 1.0
ID_POINT_MARKER = "o"
ID_POINT_SIZE = 45
ADD_ID_TEXT = False
ID_TEXT_FONTSIZE = 8
SAVE_OR_NOT = True
DPI = 300


# ======================================================================
# INPUT HELPERS
# ======================================================================

def open_closed_filename(time_hours: float) -> Path:
    return OPEN_CLOSED_DIR / f"open_closed_time.{time_hours:.2f}.npz"


def merged_filename(time_hours: float) -> Path:
    tag = f"{time_hours:.2f}".replace(".", "_")
    return MERGED_DIR / f"{tag}_merged_spherical.h5"


def output_filename(time_hours: float, mode: str) -> Path:
    surface = "footpoint" if mode == "inner" else "crossing"
    return OUTPUT_DIR / f"lct_open_boundary_{surface}_map.time.{time_hours:.2f}.r.{R0:g}.png"


def load_selection(filename: Path) -> dict[str, np.ndarray | float]:
    """Load LCT-selected inner coordinates and matched R0 crossings."""
    required = (
        "time_hours", "lct_seed_index", "inner_longitude_deg", "inner_latitude_deg",
        "r0_longitude_deg", "r0_latitude_deg", "r0_rs",
    )
    with np.load(filename, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {filename}")
        result = {key: np.asarray(data[key]).copy() for key in required}
    result["r0_rs"] = float(result["r0_rs"])
    if not np.isclose(result["r0_rs"], R0, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"Selection file R0={result['r0_rs']:g} differs from configured R0={R0:g}.")
    return result


def selection_time_index(times: np.ndarray, requested_time: float) -> int:
    """Return an exact saved-time index within a small floating tolerance."""
    matches = np.flatnonzero(np.isclose(times, requested_time, rtol=0.0, atol=1.0e-8))
    if matches.size != 1:
        raise KeyError(f"Time {requested_time:.2f} h is absent from the selection file.")
    return int(matches[0])


def wrap_phi(phi: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[phi, phi[0] + 2.0 * np.pi], np.c_[values, values[:, :1]]


def load_inner_map(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the inner open/closed map and Br contour field."""
    with np.load(filename, allow_pickle=False) as data:
        required = ("theta", "phi", "rindex0_open_closed_map", "Br_inner_surface")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {filename}")
        theta = np.asarray(data["theta"], dtype=float)
        phi = np.asarray(data["phi"], dtype=float)
        labels = np.asarray(data["rindex0_open_closed_map"], dtype=float)
        br = np.asarray(data["Br_inner_surface"], dtype=float)
    if labels.shape != (theta.size, phi.size) or br.shape != labels.shape:
        raise ValueError("Inner map shape does not match theta-phi coordinates.")
    return theta, phi, labels, br


def load_br_at_r0(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Br and linearly interpolate its radial coordinate to R0."""
    data = read_merged_physics(filename=filename, grid_filename=GRID_FILE, field_names=["Br"], load_component_map=False)
    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    br = np.asarray(data["Br"], dtype=float)
    if not (r[0] <= R0 <= r[-1]):
        raise ValueError(f"R0={R0:g} Rs is outside the merged radial grid.")
    upper = int(np.searchsorted(r, R0, side="left"))
    if upper == 0 or np.isclose(r[upper], R0):
        br_r0 = br[upper]
    else:
        lower = upper - 1
        weight = (R0 - r[lower]) / (r[upper] - r[lower])
        br_r0 = (1.0 - weight) * br[lower] + weight * br[upper]
    if not np.all(np.isfinite(br_r0)):
        raise ValueError("Br(R0) contains NaN or Inf.")
    return theta, phi, br_r0


def choose_br_clim(br: np.ndarray) -> tuple[float, float]:
    """Return a symmetric robust Br color scale."""
    limit = float(np.nanpercentile(np.abs(br), BR_CLIM_PERCENTILE))
    limit = limit if limit > 0.0 else 1.0
    return -limit, limit


# ======================================================================
# PLOTTING
# ======================================================================

def scatter_selected(ax, ids: np.ndarray, longitude: np.ndarray, latitude: np.ndarray) -> None:
    """Draw selected points with the same color assigned to each LCT seed."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (seed_id, lon, lat) in enumerate(zip(ids, longitude, latitude)):
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        color = colors[index % len(colors)]
        ax.scatter(lon, lat, s=ID_POINT_SIZE, marker=ID_POINT_MARKER, color=color, edgecolors="black", linewidths=0.4, zorder=5)
        if ADD_ID_TEXT:
            ax.text(lon, lat, f" {int(seed_id)}", color=color, fontsize=ID_TEXT_FONTSIZE, va="center", zorder=6)


def format_axes(ax, title: str) -> None:
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)


def plot_inner(theta, phi, labels, br, ids, longitude, latitude, time_hours):
    phi_wrap, labels_wrap = wrap_phi(phi, labels)
    _, br_wrap = wrap_phi(phi, br)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    image = ax.pcolormesh(np.degrees(phi_wrap), 90.0 - np.degrees(theta), labels_wrap, shading="auto", cmap=CLASS_CMAP, norm=CLASS_NORM)
    ax.contour(np.degrees(phi_wrap), 90.0 - np.degrees(theta), br_wrap, levels=[0.0], colors=BR_ZERO_CONTOUR_COLOR, linewidths=BR_ZERO_CONTOUR_LINEWIDTH)
    scatter_selected(ax, ids, longitude, latitude)
    format_axes(ax, f"Open/closed topology and LCT boundary seeds at {time_hours:.2f} h")
    cbar = fig.colorbar(image, ax=ax, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["Open (-)", "Closed", "Open (+)"])
    return fig


def plot_r0(theta, phi, br, ids, longitude, latitude, time_hours):
    phi_wrap, br_wrap = wrap_phi(phi, br)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    clim = choose_br_clim(br)
    image = ax.pcolormesh(np.degrees(phi_wrap), 90.0 - np.degrees(theta), br_wrap, shading="auto", cmap=BR_CMAP, vmin=clim[0], vmax=clim[1])
    ax.contour(np.degrees(phi_wrap), 90.0 - np.degrees(theta), br_wrap, levels=[0.0], colors=BR_ZERO_CONTOUR_COLOR, linewidths=BR_ZERO_CONTOUR_LINEWIDTH)
    scatter_selected(ax, ids, longitude, latitude)
    format_axes(ax, f"Br and matched LCT boundary crossings at R0 = {R0:g} Rs ({time_hours:.2f} h)")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Br [G]")
    return fig


def main() -> None:
    """Plot selected LCT boundary locations in the requested spherical mode."""
    mode = str(PLOT_MODE).strip().lower()
    if mode not in {"inner", "r0"}:
        raise ValueError("PLOT_MODE must be 'inner' or 'r0'.")
    selection = load_selection(SELECTION_FILE)
    times = np.asarray(selection["time_hours"], dtype=float)
    ids = np.asarray(selection["lct_seed_index"], dtype=np.int64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for requested_time in np.asarray(PLOT_TIME_HOURS, dtype=float):
        time_index = selection_time_index(times, float(requested_time))
        if mode == "inner":
            theta, phi, labels, br = load_inner_map(open_closed_filename(float(requested_time)))
            figure = plot_inner(theta, phi, labels, br, ids, selection["inner_longitude_deg"][:, time_index], selection["inner_latitude_deg"][:, time_index], float(requested_time))
        else:
            theta, phi, br = load_br_at_r0(merged_filename(float(requested_time)))
            figure = plot_r0(theta, phi, br, ids, selection["r0_longitude_deg"][:, time_index], selection["r0_latitude_deg"][:, time_index], float(requested_time))
        if SAVE_OR_NOT:
            output_file = output_filename(float(requested_time), mode)
            add_figure_provenance(figure, "plot_tracked_footpoints_lct_connectivity_map.py")
            figure.savefig(output_file, dpi=DPI, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved:\n  {output_file}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
