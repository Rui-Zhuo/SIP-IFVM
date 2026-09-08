"""Plot LCT-tracked footpoint and traced R0-crossing coordinate series.

The input is the all-ID output of ``track_footpoints_lct_connectivity.py``.
By default, 30 unique IDs nearest to randomly selected open/closed-boundary
pixels in the same longitude/latitude search window as the model-track series
are drawn. ``PLOT_IDS`` overrides this selection.

Outputs: `lct_footpoint_tracks.series.r.<R0>.png` and
`lct_crossing_tracks.series.r.<R0>.png`, plus the selected-ID NPZ file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import OPEN_CLOSED_DIR, WORK_ROOT
from figure_provenance import add_figure_provenance


INPUT_DIR = WORK_ROOT / "lct_footpoint_connectivity"
OUTPUT_DIR = INPUT_DIR
R0 = 10.0
PLOT_IDS: list[int] | None = None  # None: random open/closed-boundary selection.
N_PLOT = 30
RANDOM_SEED = 42
BOUNDARY_LON_MIN_DEG = 20.0
BOUNDARY_LON_MAX_DEG = 80.0
BOUNDARY_LAT_MIN_DEG = -10.0
BOUNDARY_LAT_MAX_DEG = 60.0
MARKER = "o"
MARKER_SIZE = 2.0
LINE_WIDTH = 1.0
DPI = 300


def manifest_filename() -> Path:
    return INPUT_DIR / f"lct_footpoint_connectivity_manifest.r.{R0:g}.npz"


def selection_filename() -> Path:
    """Return the reproducible default boundary-ID selection file."""
    return OUTPUT_DIR / f"selected_lct_open_boundary_ids.r.{R0:g}.npz"


def load_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all time files into ID-by-time coordinate arrays."""
    with np.load(manifest_filename(), allow_pickle=False) as manifest:
        times = np.asarray(manifest["time_hours"], dtype=float)
        filenames = [str(item) for item in np.asarray(manifest["filenames"])]
        ids = np.asarray(manifest["id"], dtype=np.int64)
    shape = (ids.size, times.size)
    foot_lon = np.full(shape, np.nan, dtype=float)
    foot_lat = np.full(shape, np.nan, dtype=float)
    cross_lon = np.full(shape, np.nan, dtype=float)
    cross_lat = np.full(shape, np.nan, dtype=float)
    for itime, name in enumerate(filenames):
        with np.load(INPUT_DIR / name, allow_pickle=False) as data:
            file_ids = np.asarray(data["id"], dtype=np.int64)
            if not np.array_equal(file_ids, ids):
                raise ValueError(f"ID ordering differs in {name}")
            foot_lon[:, itime] = np.asarray(data["footpoint_longitude_deg"], dtype=float)
            foot_lat[:, itime] = np.asarray(data["footpoint_latitude_deg"], dtype=float)
            cross_lon[:, itime] = np.asarray(data["crossing_longitude_deg"], dtype=float)
            cross_lat[:, itime] = np.asarray(data["crossing_latitude_deg"], dtype=float)
    return times, ids, foot_lon, foot_lat, cross_lon, cross_lat


def longitude_in_region(longitude: np.ndarray) -> np.ndarray:
    """Return membership in the configured periodic longitude interval."""
    if BOUNDARY_LON_MIN_DEG <= BOUNDARY_LON_MAX_DEG:
        return (longitude >= BOUNDARY_LON_MIN_DEG) & (longitude <= BOUNDARY_LON_MAX_DEG)
    return (longitude >= BOUNDARY_LON_MIN_DEG) | (longitude <= BOUNDARY_LON_MAX_DEG)


def load_initial_open_closed(time_hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the r_index=0 topology map used for default boundary sampling."""
    filename = OPEN_CLOSED_DIR / f"open_closed_time.{time_hours:.2f}.npz"
    with np.load(filename, allow_pickle=False) as data:
        return (
            np.asarray(data["theta"], dtype=float),
            np.asarray(data["phi"], dtype=float),
            np.asarray(data["rindex0_open_closed_map"], dtype=np.int8),
        )


def boundary_targets(theta: np.ndarray, phi: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly select alternating closed/open target pixels at the boundary."""
    latitude = 90.0 - np.degrees(theta)
    longitude = np.degrees(phi) % 360.0
    region = (
        (latitude[:, None] >= BOUNDARY_LAT_MIN_DEG)
        & (latitude[:, None] <= BOUNDARY_LAT_MAX_DEG)
        & longitude_in_region(longitude)[None, :]
    )
    pairs: list[tuple[int, int, int, int]] = []
    for j in range(theta.size - 1):
        for k in range(phi.size):
            if not (region[j, k] and region[j + 1, k]):
                continue
            first, second = labels[j, k], labels[j + 1, k]
            if first == 0 and abs(second) == 1:
                pairs.append((j, k, j + 1, k))
            elif second == 0 and abs(first) == 1:
                pairs.append((j + 1, k, j, k))
    for j in range(theta.size):
        for k in range(phi.size):
            k2 = (k + 1) % phi.size
            if not (region[j, k] and region[j, k2]):
                continue
            first, second = labels[j, k], labels[j, k2]
            if first == 0 and abs(second) == 1:
                pairs.append((j, k, j, k2))
            elif second == 0 and abs(first) == 1:
                pairs.append((j, k2, j, k))
    if len(pairs) < int(np.ceil(N_PLOT / 2.0)):
        raise ValueError(f"Only {len(pairs)} boundary pairs are available; requested {N_PLOT} IDs.")
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(len(pairs), size=N_PLOT, replace=len(pairs) < N_PLOT)
    target_theta = []
    target_phi = []
    for index, pair_index in enumerate(chosen):
        closed_j, closed_k, open_j, open_k = pairs[int(pair_index)]
        j, k = (closed_j, closed_k) if index % 2 == 0 else (open_j, open_k)
        target_theta.append(theta[j])
        target_phi.append(phi[k])
    return np.asarray(target_theta), np.asarray(target_phi)


def angular_distance(theta_a: np.ndarray, phi_a: np.ndarray, theta_b: float, phi_b: float) -> np.ndarray:
    """Return great-circle separation in radians."""
    cosine = np.cos(theta_a) * np.cos(theta_b) + np.sin(theta_a) * np.sin(theta_b) * np.cos(phi_a - phi_b)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def choose_ids(ids: np.ndarray, foot_lon: np.ndarray, foot_lat: np.ndarray, first_time: float) -> np.ndarray:
    """Return requested IDs or default unique IDs nearest boundary targets."""
    lookup = {int(seed_id): index for index, seed_id in enumerate(ids)}
    if PLOT_IDS is not None:
        missing = [seed_id for seed_id in PLOT_IDS if seed_id not in lookup]
        if missing:
            raise KeyError(f"Requested IDs are unavailable: {missing}")
        return np.asarray([lookup[seed_id] for seed_id in PLOT_IDS], dtype=int)
    map_theta, map_phi, labels = load_initial_open_closed(first_time)
    target_theta, target_phi = boundary_targets(map_theta, map_phi, labels)
    available_theta = np.radians(90.0 - foot_lat)
    available_phi = np.radians(foot_lon) % (2.0 * np.pi)
    selected_rows = []
    for theta_value, phi_value in zip(target_theta, target_phi):
        distance = angular_distance(available_theta, available_phi, theta_value, phi_value)
        for candidate in np.argsort(distance):
            candidate = int(candidate)
            if candidate not in selected_rows:
                selected_rows.append(candidate)
                break
    if len(selected_rows) != N_PLOT:
        raise RuntimeError("Could not assign a unique LCT ID to every boundary target.")
    np.savez_compressed(
        selection_filename(),
        id=ids[selected_rows],
        target_theta_rad=target_theta,
        target_phi_rad=target_phi,
        random_seed=RANDOM_SEED,
        reference_time_hours=float(first_time),
        boundary_lon_min_deg=BOUNDARY_LON_MIN_DEG,
        boundary_lon_max_deg=BOUNDARY_LON_MAX_DEG,
        boundary_lat_min_deg=BOUNDARY_LAT_MIN_DEG,
        boundary_lat_max_deg=BOUNDARY_LAT_MAX_DEG,
    )
    return np.asarray(selected_rows, dtype=int)


def make_series_figure(times: np.ndarray, ids: np.ndarray, longitude: np.ndarray, latitude: np.ndarray, title: str) -> plt.Figure:
    """Make a longitude/latitude time-series figure with persistent ID colors."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, seed_id in enumerate(ids):
        color = colors[index % len(colors)]
        axes[0].plot(times, longitude[index], marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=color, label=f"ID {seed_id}")
        axes[1].plot(times, latitude[index], marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=color)
    axes[0].set_title(title)
    axes[0].set_ylabel("Longitude [deg]")
    axes[1].set_ylabel("Latitude [deg]")
    axes[1].set_xlabel("Simulation time [h]")
    axes[0].legend(ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    return figure


def main() -> None:
    """Create LCT footpoint and traced-crossing coordinate time-series plots."""
    times, ids, foot_lon, foot_lat, cross_lon, cross_lat = load_series()
    rows = choose_ids(ids, foot_lon[:, 0], foot_lat[:, 0], float(times[0]))
    selected_ids = ids[rows]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    foot_figure = make_series_figure(times, selected_ids, foot_lon[rows], foot_lat[rows], "LCT-tracked r_index=0 footpoints")
    foot_output = OUTPUT_DIR / f"lct_footpoint_tracks.series.r.{R0:g}.png"
    add_figure_provenance(foot_figure, "plot_tracked_footpoints_lct_connectivity_series.py")
    foot_figure.savefig(foot_output, dpi=DPI, bbox_inches="tight")
    plt.close(foot_figure)
    crossing_figure = make_series_figure(times, selected_ids, cross_lon[rows], cross_lat[rows], f"Traced R0={R0:g} Rs crossings of LCT footpoints")
    crossing_output = OUTPUT_DIR / f"lct_crossing_tracks.series.r.{R0:g}.png"
    add_figure_provenance(crossing_figure, "plot_tracked_footpoints_lct_connectivity_series.py")
    crossing_figure.savefig(crossing_output, dpi=DPI, bbox_inches="tight")
    plt.close(crossing_figure)
    print(f"Saved:\n  {foot_output}\n  {crossing_output}")


if __name__ == "__main__":
    main()
