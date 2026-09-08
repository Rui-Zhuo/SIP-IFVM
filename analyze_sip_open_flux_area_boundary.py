"""Measure r_index=0 open-field area, magnetic flux, and boundary length.

Dependencies: numpy==1.26.4, matplotlib==3.10.8.
Br is read in G; surface areas are cm^2, magnetic fluxes are Mx,
and boundary lengths are saved in cm and Mm.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from config import OPEN_CLOSED_DIR, WORK_ROOT


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_DIR = OPEN_CLOSED_DIR
OUTPUT_DIR = WORK_ROOT / "open_flux_area"
OPEN_CLOSED_TAG = "rindex0"
R_SUN_CM = 6.957e10

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


def open_closed_boundary_length(
    theta: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    radius_rs: float,
) -> float:
    """Return total boundary length between open (|label|=1) and closed (0)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)

    expected_shape = (theta.size, phi.size)
    if labels.shape != expected_shape:
        raise ValueError(f"labels.shape={labels.shape}, expected {expected_shape}")

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
    open_mask = np.abs(labels) == 1
    total_length_cm = 0.0

    # Interfaces between adjacent theta rows. The shared edge runs in phi.
    theta_transition = open_mask[:-1, :] != open_mask[1:, :]
    if np.any(theta_transition):
        interface_theta = theta_edges[1:-1]
        edge_length = radius_cm * np.sin(interface_theta) * phi_step
        counts = np.sum(theta_transition, axis=1, dtype=np.int64)
        total_length_cm += float(np.sum(edge_length * counts, dtype=np.float64))

    # Interfaces between adjacent phi columns. Phi is periodic.
    phi_transition = open_mask != np.roll(open_mask, shift=-1, axis=1)
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
    """Return area, signed flux, and unsigned flux for one region."""
    area = float(np.sum(area_cm2[mask], dtype=np.float64))
    signed_flux = float(np.sum(br_gauss[mask] * area_cm2[mask], dtype=np.float64))
    unsigned_flux = float(np.sum(np.abs(br_gauss[mask]) * area_cm2[mask], dtype=np.float64))
    return area, signed_flux, unsigned_flux


def make_figure(times: np.ndarray, metrics: dict[str, np.ndarray]) -> plt.Figure:
    """Plot area, unsigned flux, and signed flux for open-field regions."""
    labels = (("positive", "Open (+)"), ("negative", "Open (-)"), ("all", "All open"))
    colors = {"positive": "firebrick", "negative": "royalblue", "all": "black"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    panels = (("area_cm2", "Area [$10^{21}$ cm$^2$]", 1.0e21), ("unsigned_flux_mx", "Unsigned flux [$10^{21}$ Mx]", 1.0e21), ("signed_flux_mx", "Signed flux [$10^{21}$ Mx]", 1.0e21))

    for ax, (metric, ylabel, scale) in zip(axes, panels):
        for key, label in labels:
            ax.plot(times, metrics[f"{key}_{metric}"] / scale, marker="o", markersize=2.5, linewidth=1.2, color=colors[key], label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Simulation time [h]")
    fig.suptitle("r_index=0 open-field area and magnetic-flux evolution")
    return fig


def make_boundary_length_figure(
    times: np.ndarray,
    boundary_length_cm: np.ndarray,
) -> plt.Figure:
    """Plot total open-field boundary length versus simulation time."""
    boundary_length_mm = np.asarray(boundary_length_cm, dtype=float) / 1.0e8
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    ax.plot(times, boundary_length_mm, marker="o", markersize=3.0, linewidth=1.2)
    ax.set_xlabel("Simulation time [h]")
    ax.set_ylabel("Open-field boundary length [Mm]")
    ax.set_title("r_index=0 total open-field boundary length")
    ax.grid(True, alpha=0.3)
    return fig


def main() -> None:
    """Measure and save positive, negative, and total open-field diagnostics."""
    entries = discover_open_closed_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    times = []
    boundary_length_cm_list = []
    metric_lists = {
        f"{region}_{quantity}": []
        for region in ("positive", "negative", "all")
        for quantity in ("area_cm2", "signed_flux_mx", "unsigned_flux_mx")
    }

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
        boundary_length_cm = open_closed_boundary_length(theta, phi, labels, radius_rs)
        boundary_length_cm_list.append(boundary_length_cm)
        masks = {"positive": labels == 1, "negative": labels == -1, "all": np.abs(labels) == 1}
        for region, mask in masks.items():
            area, signed_flux, unsigned_flux = region_metrics(mask, area_cm2, br)
            metric_lists[f"{region}_area_cm2"].append(area)
            metric_lists[f"{region}_signed_flux_mx"].append(signed_flux)
            metric_lists[f"{region}_unsigned_flux_mx"].append(unsigned_flux)
        times.append(time_hours)

    metrics = {key: np.asarray(values, dtype=float) for key, values in metric_lists.items()}
    boundary_length_cm = np.asarray(boundary_length_cm_list, dtype=float)
    boundary_length_mm = boundary_length_cm / 1.0e8
    output_data = OUTPUT_DIR / "open_field_area_flux.rindex0.npz"
    np.savez_compressed(
        output_data,
        time_hours=np.asarray(times, dtype=float),
        theta=reference_theta,
        phi=reference_phi,
        surface_radius_rs=float(reference_radius),
        open_closed_tag=OPEN_CLOSED_TAG,
        random_seed=RANDOM_SEED,
        open_field_boundary_length_cm=boundary_length_cm,
        open_field_boundary_length_Mm=boundary_length_mm,
        **metrics,
    )

    print(f"Saved data:\n  {output_data}")
    if SAVE_FIGURE:
        times_array = np.asarray(times, dtype=float)

        fig = make_figure(times_array, metrics)
        output_figure = OUTPUT_DIR / "open_field_area_flux.rindex0.png"
        fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure:\n  {output_figure}")

        boundary_fig = make_boundary_length_figure(times_array, boundary_length_cm)
        boundary_output_figure = OUTPUT_DIR / "open_field_boundary_length.rindex0.png"
        boundary_fig.savefig(boundary_output_figure, dpi=DPI, bbox_inches="tight")
        plt.close(boundary_fig)
        print(f"Saved boundary-length figure:\n  {boundary_output_figure}")


if __name__ == "__main__":
    main()
