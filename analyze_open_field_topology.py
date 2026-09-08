"""Measure r_index=0 open-field area, magnetic flux, and boundary length.

Dependencies: numpy==1.26.4, matplotlib==3.10.8.
Br is read in G; surface areas are cm^2, magnetic fluxes are Mx,
and boundary lengths are saved in cm and Mm.

Outputs: one polarity-specific `open_field_topology.<sign>.npz` file and its
matching `open_field_topology.<sign>.png` figure.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from figure_provenance import add_figure_provenance

from config import (
    CH_REGION_LOW_MID_LAT_MAX_DEG,
    CH_REGION_LOW_MID_LAT_MIN_DEG,
    CH_REGION_LOW_MID_LON_MAX_DEG,
    CH_REGION_LOW_MID_LON_MIN_DEG,
    CH_REGION_NORTH_LATITUDE_MIN_DEG,
    CH_REGION_REFERENCE_TIME_HOURS,
    OPEN_CLOSED_DIR,
    WORK_ROOT,
)
from utils import longitude_interval_mask, rotate_longitude_deg


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_DIR = OPEN_CLOSED_DIR
OUTPUT_DIR = WORK_ROOT / "open_flux_area"
OPEN_CLOSED_TAG = "rindex0"
R_SUN_CM = 6.957e10

# Open-field topology label to analyse: -1 for negative, +1 for positive.
OPEN_FIELD_POLARITY = -1

SAVE_FIGURE = True
DPI = 250
RANDOM_SEED = 42


# ======================================================================
# INPUT AND GEOMETRY
# ======================================================================

FILE_PATTERN = re.compile(r"^open_closed_time\.([0-9]+(?:\.[0-9]+)?)\.npz$")


def discover_open_closed_files() -> list[tuple[float, Path]]:
    """Return all topology files ordered by simulation time."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(INPUT_DIR)
    entries = []
    for filename in INPUT_DIR.iterdir():
        match = FILE_PATTERN.fullmatch(filename.name)
        if match is not None:
            entries.append((float(match.group(1)), filename))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(f"No open_closed_time.*.npz files in {INPUT_DIR}")
    return entries


def load_open_closed_surface(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load theta, phi, topology labels, Br [G], and surface radius [Rs]."""
    map_key = f"{OPEN_CLOSED_TAG}_open_closed_map"
    required = ("theta", "phi", map_key, "Br_inner_surface", "inner_radius")
    with np.load(filename, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {filename}")
        theta = np.asarray(data["theta"], dtype=float)
        phi = np.asarray(data["phi"], dtype=float)
        labels = np.asarray(data[map_key], dtype=np.int8)
        br = np.asarray(data["Br_inner_surface"], dtype=float)
        radius_rs = float(data["inner_radius"])

    expected_shape = (theta.size, phi.size)
    if labels.shape != expected_shape or br.shape != expected_shape:
        raise ValueError(f"Topology or Br shape differs from {expected_shape}: {filename}")
    if not np.all(np.isfinite(br)):
        raise ValueError(f"Br contains NaN or Inf: {filename}")
    if not set(np.unique(labels)).issubset({-1, 0, 1}):
        raise ValueError(f"Unexpected open/closed labels: {filename}")
    return theta, phi, labels, br, radius_rs


def theta_cell_edges(theta: np.ndarray) -> np.ndarray:
    """Construct colatitude cell edges, clipped at both poles."""
    if theta.size < 2 or not np.all(np.diff(theta) > 0.0):
        raise ValueError("theta must have at least two increasing samples.")
    edges = np.empty(theta.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    edges[0] = max(0.0, theta[0] - 0.5 * (theta[1] - theta[0]))
    edges[-1] = min(np.pi, theta[-1] + 0.5 * (theta[-1] - theta[-2]))
    return edges


def surface_cell_area(theta: np.ndarray, phi: np.ndarray, radius_rs: float) -> np.ndarray:
    """Return latitude-weighted spherical-cell areas in cm^2."""
    if phi.size < 2 or not np.all(np.diff(phi) > 0.0):
        raise ValueError("phi must have at least two increasing samples.")
    dphi = np.diff(phi)
    phi_step = float(np.mean(dphi))
    if not np.allclose(dphi, phi_step, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("Area calculation currently requires a uniform phi grid.")
    if not np.isclose(phi_step * phi.size, 2.0 * np.pi, rtol=1.0e-4, atol=1.0e-8):
        raise ValueError("phi grid must cover one full 2π longitude period.")
    theta_edges = theta_cell_edges(theta)
    row_area = (radius_rs * R_SUN_CM) ** 2 * dphi.mean() * (
        np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    )
    return np.broadcast_to(row_area[:, None], (theta.size, phi.size)).copy()


def coronal_hole_region_mask(theta: np.ndarray, phi: np.ndarray, time_hours: float) -> np.ndarray:
    """Return the union of the north polar cap and differentially rotated ROI."""
    latitude = 90.0 - np.degrees(np.asarray(theta, dtype=float))
    longitude = np.degrees(np.asarray(phi, dtype=float)) % 360.0
    north_mask = latitude[:, None] >= CH_REGION_NORTH_LATITUDE_MIN_DEG
    rotated_lower = rotate_longitude_deg(
        CH_REGION_LOW_MID_LON_MIN_DEG,
        latitude,
        float(time_hours) - CH_REGION_REFERENCE_TIME_HOURS,
    )
    rotated_upper = rotate_longitude_deg(
        CH_REGION_LOW_MID_LON_MAX_DEG,
        latitude,
        float(time_hours) - CH_REGION_REFERENCE_TIME_HOURS,
    )
    longitude_mask = np.vstack(
        [longitude_interval_mask(longitude, lower, upper) for lower, upper in zip(rotated_lower, rotated_upper)]
    )
    low_mid_mask = (
        (latitude[:, None] >= CH_REGION_LOW_MID_LAT_MIN_DEG)
        & (latitude[:, None] <= CH_REGION_LOW_MID_LAT_MAX_DEG)
        & longitude_mask
    )
    return north_mask | low_mid_mask


def open_closed_boundary_length(
    theta: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    radius_rs: float,
    region_mask: np.ndarray,
    polarity: int,
) -> float:
    """Return selected-polarity/closed interface length inside the analysis region."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)

    expected_shape = (theta.size, phi.size)
    if labels.shape != expected_shape:
        raise ValueError(f"labels.shape={labels.shape}, expected {expected_shape}")
    region_mask = np.asarray(region_mask, dtype=bool)
    if region_mask.shape != expected_shape:
        raise ValueError(f"region_mask.shape={region_mask.shape}, expected {expected_shape}")
    if int(polarity) not in {-1, 1}:
        raise ValueError("polarity must be -1 or +1.")

    if phi.size < 2:
        raise ValueError("phi must contain at least two samples.")

    dphi = np.diff(phi)
    phi_step = float(np.mean(dphi))
    if not np.allclose(dphi, phi_step, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("Boundary calculation requires a uniform phi grid.")
    if not np.isclose(phi_step * phi.size, 2.0 * np.pi, rtol=1.0e-4, atol=1.0e-8):
        raise ValueError("phi grid must cover one full 2π longitude period.")

    theta_edges = theta_cell_edges(theta)
    radius_cm = float(radius_rs) * R_SUN_CM
    selected_open = labels == int(polarity)
    closed_mask = labels == 0
    total_length_cm = 0.0

    # Interfaces between adjacent theta rows. The shared edge runs in phi.
    theta_transition = (
        ((selected_open[:-1, :] & closed_mask[1:, :]) | (closed_mask[:-1, :] & selected_open[1:, :]))
        & region_mask[:-1, :]
        & region_mask[1:, :]
    )
    if np.any(theta_transition):
        interface_theta = theta_edges[1:-1]
        edge_length = radius_cm * np.sin(interface_theta) * phi_step
        counts = np.sum(theta_transition, axis=1, dtype=np.int64)
        total_length_cm += float(np.sum(edge_length * counts, dtype=np.float64))

    # Interfaces between adjacent phi columns. Phi is periodic.
    phi_transition = (
        ((selected_open & np.roll(closed_mask, shift=-1, axis=1)) | (closed_mask & np.roll(selected_open, shift=-1, axis=1)))
        & region_mask
        & np.roll(region_mask, shift=-1, axis=1)
    )
    if np.any(phi_transition):
        theta_width = np.diff(theta_edges)
        edge_length = radius_cm * theta_width
        counts = np.sum(phi_transition, axis=1, dtype=np.int64)
        total_length_cm += float(np.sum(edge_length * counts, dtype=np.float64))

    return total_length_cm


# ======================================================================
# DIAGNOSTICS
# ======================================================================

def region_metrics(mask: np.ndarray, area_cm2: np.ndarray, br_gauss: np.ndarray) -> tuple[float, float, float]:
    """Return selected-region area, signed flux, and unsigned flux."""
    area = float(np.sum(area_cm2[mask], dtype=np.float64))
    signed_flux = float(np.sum(br_gauss[mask] * area_cm2[mask], dtype=np.float64))
    unsigned_flux = float(np.sum(np.abs(br_gauss[mask]) * area_cm2[mask], dtype=np.float64))
    return area, signed_flux, unsigned_flux


def make_figure(times: np.ndarray, area_cm2: np.ndarray, signed_flux_mx: np.ndarray, boundary_length_cm: np.ndarray) -> plt.Figure:
    """Plot one selected-polarity area, signed-flux, and boundary-length series."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    color = "royalblue" if OPEN_FIELD_POLARITY == -1 else "firebrick"
    sign_label = "Open (-)" if OPEN_FIELD_POLARITY == -1 else "Open (+)"
    axes[0].plot(times, area_cm2 / 1.0e21, marker="o", markersize=2.5, linewidth=1.2, color=color)
    axes[1].plot(times, signed_flux_mx / 1.0e21, marker="o", markersize=2.5, linewidth=1.2, color=color)
    axes[2].plot(times, boundary_length_cm / 1.0e8, marker="o", markersize=2.5, linewidth=1.2, color=color)
    axes[0].set_ylabel("Area [$10^{21}$ cm$^2$]")
    axes[1].set_ylabel("Signed flux [$10^{21}$ Mx]")
    axes[2].set_ylabel("Boundary length [Mm]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Simulation time [h]")
    fig.suptitle(f"Coronal-hole ROI: {sign_label} r_index=0 topology evolution")
    return fig


def main() -> None:
    """Measure and save one selected-polarity open-field topology history."""
    if OPEN_FIELD_POLARITY not in {-1, 1}:
        raise ValueError("OPEN_FIELD_POLARITY must be -1 or +1.")
    entries = discover_open_closed_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    times = []
    boundary_length_cm_list = []
    area_cm2_list = []
    signed_flux_mx_list = []
    unsigned_flux_mx_list = []

    reference_theta = None
    reference_phi = None
    reference_radius = None

    for time_hours, filename in entries:
        theta, phi, labels, br, radius_rs = load_open_closed_surface(filename)
        if reference_theta is None:
            reference_theta, reference_phi, reference_radius = theta, phi, radius_rs
        elif not (np.allclose(theta, reference_theta) and np.allclose(phi, reference_phi) and np.isclose(radius_rs, reference_radius)):
            raise ValueError(f"Incompatible spherical grid: {filename}")

        area_cm2 = surface_cell_area(theta, phi, radius_rs)
        analysis_mask = coronal_hole_region_mask(theta, phi, time_hours)
        boundary_length_cm = open_closed_boundary_length(
            theta, phi, labels, radius_rs, analysis_mask, OPEN_FIELD_POLARITY,
        )
        boundary_length_cm_list.append(boundary_length_cm)
        mask = (labels == OPEN_FIELD_POLARITY) & analysis_mask
        area, signed_flux, unsigned_flux = region_metrics(mask, area_cm2, br)
        area_cm2_list.append(area)
        signed_flux_mx_list.append(signed_flux)
        unsigned_flux_mx_list.append(unsigned_flux)
        times.append(time_hours)

    area_cm2 = np.asarray(area_cm2_list, dtype=float)
    signed_flux_mx = np.asarray(signed_flux_mx_list, dtype=float)
    unsigned_flux_mx = np.asarray(unsigned_flux_mx_list, dtype=float)
    boundary_length_cm = np.asarray(boundary_length_cm_list, dtype=float)
    boundary_length_mm = boundary_length_cm / 1.0e8
    polarity_tag = "negative" if OPEN_FIELD_POLARITY == -1 else "positive"
    output_data = OUTPUT_DIR / f"open_field_topology.{polarity_tag}.npz"
    np.savez_compressed(
        output_data,
        time_hours=np.asarray(times, dtype=float),
        theta=reference_theta,
        phi=reference_phi,
        surface_radius_rs=float(reference_radius),
        open_closed_tag=OPEN_CLOSED_TAG,
        open_field_polarity=OPEN_FIELD_POLARITY,
        analysis_region="north_lat_ge_60_union_differentially_rotated_low_mid_roi",
        region_reference_time_hours=CH_REGION_REFERENCE_TIME_HOURS,
        north_latitude_min_deg=CH_REGION_NORTH_LATITUDE_MIN_DEG,
        low_mid_lon_min_deg=CH_REGION_LOW_MID_LON_MIN_DEG,
        low_mid_lon_max_deg=CH_REGION_LOW_MID_LON_MAX_DEG,
        low_mid_lat_min_deg=CH_REGION_LOW_MID_LAT_MIN_DEG,
        low_mid_lat_max_deg=CH_REGION_LOW_MID_LAT_MAX_DEG,
        random_seed=RANDOM_SEED,
        area_cm2=area_cm2,
        signed_magnetic_flux_mx=signed_flux_mx,
        unsigned_magnetic_flux_mx=unsigned_flux_mx,
        boundary_length_cm=boundary_length_cm,
        boundary_length_Mm=boundary_length_mm,
    )

    print(f"Saved data:\n  {output_data}")
    if SAVE_FIGURE:
        times_array = np.asarray(times, dtype=float)

        fig = make_figure(times_array, area_cm2, signed_flux_mx, boundary_length_cm)
        output_figure = OUTPUT_DIR / f"open_field_topology.{polarity_tag}.png"
        add_figure_provenance(fig, "analyze_open_field_topology.py")
        fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure:\n  {output_figure}")


if __name__ == "__main__":
    main()
