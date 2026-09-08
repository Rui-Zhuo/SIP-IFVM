"""Compute r_index=0 horizontal magnetic-element velocities with LCT.

Dependencies: numpy==1.26.4, scipy>=1.10,<1.15, matplotlib==3.10.8.
The input Br maps are in G.  Output velocities are Vtheta/Vphi [km s^-1],
where positive Vtheta is toward increasing colatitude and positive Vphi is
toward increasing longitude.

Outputs: `lct_footpoint_velocity.time.<t0>_to_<t1>.npz` and, when enabled,
the identically tagged `.png` velocity diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from config import GRID_FILE, LOCAL_MERGED_DIR, WORK_ROOT
from figure_provenance import add_figure_provenance
from read_merged_sip_data import read_merged_physics, select_merged_data_files
from utils import differential_rotation_rate_deg_per_day


# ======================================================================
# CONFIGURATION
# ======================================================================

PROCESS_MODE = "all"
DATA_DIR = LOCAL_MERGED_DIR
SINGLE_DATA_FILE = "82_10_merged_spherical.h5"
OUTPUT_DIR = WORK_ROOT / "lct_velocity"

SURFACE_RADIAL_INDEX = 0
R_SUN_KM = 695700.0

# LCT window and search parameters in grid cells.
LCT_WINDOW_SIGMA_CELLS = 3.0
LCT_SEARCH_THETA_CELLS = 4
LCT_SEARCH_PHI_CELLS = 6
MIN_CORRELATION = 0.25
MIN_TEXTURE_GAUSS = 1.0e-5
POLAR_SIN_THETA_MIN = 0.05

SAVE_FIGURE = True
DPI = 220
RANDOM_SEED = 42


# ======================================================================
# HELPERS
# ======================================================================

def velocity_filename(time_start_hours: float, time_end_hours: float) -> Path:
    return OUTPUT_DIR / (
        f"lct_footpoint_velocity.time.{time_start_hours:.2f}_to_{time_end_hours:.2f}.npz"
    )


def figure_filename(time_start_hours: float, time_end_hours: float) -> Path:
    return OUTPUT_DIR / (
        f"lct_footpoint_velocity.time.{time_start_hours:.2f}_to_{time_end_hours:.2f}.png"
    )


def validate_angular_grid(theta: np.ndarray, phi: np.ndarray) -> tuple[float, float]:
    """Validate a monotonic nearly-uniform theta-phi grid."""
    dtheta = np.diff(theta)
    dphi = np.diff(phi)

    if not (np.all(dtheta > 0.0) and np.all(dphi > 0.0)):
        raise ValueError("theta and phi must be strictly increasing.")

    theta_step = float(np.mean(dtheta))
    phi_step = float(np.mean(dphi))

    if not np.allclose(dtheta, theta_step, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("LCT currently requires a uniform theta grid.")

    if not np.allclose(dphi, phi_step, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("LCT currently requires a uniform phi grid.")

    return theta_step, phi_step


def read_surface_br(filename: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Read the r_index=0 Br map in G."""
    data = read_merged_physics(
        filename=filename,
        grid_filename=GRID_FILE,
        field_names=["Br"],
        load_component_map=False,
    )

    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    br = np.asarray(data["Br"], dtype=float)

    index = SURFACE_RADIAL_INDEX
    if index < 0:
        index += r.size
    if index < 0 or index >= r.size:
        raise IndexError("SURFACE_RADIAL_INDEX is outside the radial grid.")

    br_surface = br[index]
    if br_surface.shape != (theta.size, phi.size):
        raise ValueError("Br surface shape does not match theta-phi grid.")
    if not np.all(np.isfinite(br_surface)):
        raise ValueError(f"Br contains NaN or Inf: {filename}")

    validate_angular_grid(theta, phi)
    return theta, phi, float(r[index]), br_surface


def local_normalized_correlation(
    br_start: np.ndarray,
    br_end: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate theta/phi displacements with local normalized correlation."""
    dtheta, dphi = validate_angular_grid(theta, phi)
    nt, np_ = br_start.shape
    jj, kk = np.indices((nt, np_), dtype=float)
    latitude_deg = 90.0 - np.degrees(theta)
    reference_shift_rad = np.radians(
        differential_rotation_rate_deg_per_day(latitude_deg) * dt_hours / 24.0
    )
    reference_shift_cells = reference_shift_rad / dphi

    best_corr = np.full((nt, np_), -np.inf, dtype=float)
    best_theta_cells = np.zeros((nt, np_), dtype=float)
    best_phi_cells = np.zeros((nt, np_), dtype=float)
    br_end_periodic = np.concatenate((br_end, br_end[:, :1]), axis=1)

    mean_start = gaussian_filter(br_start, sigma=LCT_WINDOW_SIGMA_CELLS, mode=("nearest", "wrap"))
    start_zero_mean = br_start - mean_start
    start_power = gaussian_filter(
        start_zero_mean**2,
        sigma=LCT_WINDOW_SIGMA_CELLS,
        mode=("nearest", "wrap"),
    )

    def correlation_for_candidate(theta_shift_cells, phi_displacement_cells):
        """Evaluate local correlation for scalar or spatially varying shifts."""
        theta_shift_cells = np.broadcast_to(theta_shift_cells, (nt, np_))
        phi_displacement_cells = np.broadcast_to(phi_displacement_cells, (nt, np_))
        sample_theta = np.clip(jj + theta_shift_cells, 0.0, nt - 1.0)
        sample_phi = (kk + phi_displacement_cells) % np_
        shifted_end = map_coordinates(
            br_end_periodic,
            [sample_theta, sample_phi],
            order=1,
            mode="nearest",
        )
        end_mean = gaussian_filter(
            shifted_end,
            sigma=LCT_WINDOW_SIGMA_CELLS,
            mode=("nearest", "wrap"),
        )
        end_zero_mean = shifted_end - end_mean
        end_power = gaussian_filter(
            end_zero_mean**2,
            sigma=LCT_WINDOW_SIGMA_CELLS,
            mode=("nearest", "wrap"),
        )
        cross_power = gaussian_filter(
            start_zero_mean * end_zero_mean,
            sigma=LCT_WINDOW_SIGMA_CELLS,
            mode=("nearest", "wrap"),
        )
        return cross_power / np.sqrt(start_power * end_power)

    theta_offsets = range(-LCT_SEARCH_THETA_CELLS, LCT_SEARCH_THETA_CELLS + 1)
    phi_offsets = range(-LCT_SEARCH_PHI_CELLS, LCT_SEARCH_PHI_CELLS + 1)

    for theta_offset in theta_offsets:
        for phi_offset in phi_offsets:
            candidate_displacement_cells = np.broadcast_to(
                reference_shift_cells[:, None] + phi_offset,
                (nt, np_),
            )
            correlation = correlation_for_candidate(
                theta_offset,
                candidate_displacement_cells,
            )

            improved = correlation > best_corr
            best_corr[improved] = correlation[improved]
            best_theta_cells[improved] = theta_offset
            best_phi_cells[improved] = candidate_displacement_cells[improved]

    # Quadratic sub-pixel refinement around the best integer-cell peak.
    theta_minus = correlation_for_candidate(best_theta_cells - 1.0, best_phi_cells)
    theta_plus = correlation_for_candidate(best_theta_cells + 1.0, best_phi_cells)
    phi_minus = correlation_for_candidate(best_theta_cells, best_phi_cells - 1.0)
    phi_plus = correlation_for_candidate(best_theta_cells, best_phi_cells + 1.0)

    def quadratic_peak_offset(score_minus, score_center, score_plus):
        denominator = score_minus - 2.0 * score_center + score_plus
        offset = np.zeros_like(score_center)
        usable = np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12)
        offset[usable] = 0.5 * (score_minus[usable] - score_plus[usable]) / denominator[usable]
        return np.clip(offset, -0.5, 0.5)

    theta_subpixel = quadratic_peak_offset(theta_minus, best_corr, theta_plus)
    phi_subpixel = quadratic_peak_offset(phi_minus, best_corr, phi_plus)
    theta_at_edge = np.abs(best_theta_cells) >= LCT_SEARCH_THETA_CELLS
    phi_at_edge = np.abs(best_phi_cells - reference_shift_cells[:, None]) >= LCT_SEARCH_PHI_CELLS
    theta_subpixel[theta_at_edge] = 0.0
    phi_subpixel[phi_at_edge] = 0.0

    valid = (
        np.isfinite(best_corr)
        & (best_corr >= MIN_CORRELATION)
        & (start_power >= MIN_TEXTURE_GAUSS)
        & (np.sin(theta)[:, None] >= POLAR_SIN_THETA_MIN)
    )

    displacement_theta = (best_theta_cells + theta_subpixel) * dtheta
    displacement_phi = (best_phi_cells + phi_subpixel) * dphi
    displacement_theta[~valid] = np.nan
    displacement_phi[~valid] = np.nan
    best_corr[~valid] = np.nan

    return displacement_theta, displacement_phi, best_corr, reference_shift_rad


def make_velocity_figure(
    theta: np.ndarray,
    phi: np.ndarray,
    velocity_theta: np.ndarray,
    velocity_phi: np.ndarray,
    correlation: np.ndarray,
    time_start_hours: float,
    time_end_hours: float,
) -> plt.Figure:
    """Plot LCT Vtheta, Vphi, and peak correlation maps."""
    phi_deg = np.degrees(np.r_[phi, phi[0] + 2.0 * np.pi])
    lat_deg = 90.0 - np.degrees(theta)
    fields = [velocity_theta, velocity_phi, correlation]
    titles = ["LCT Vtheta [km s$^{-1}$]", "LCT Vphi [km s$^{-1}$]", "LCT peak correlation"]
    cmaps = ["coolwarm", "coolwarm", "viridis"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        field_wrap = np.c_[field, field[:, :1]]
        if title.startswith("LCT V"):
            limit = float(np.nanpercentile(np.abs(field), 99.0))
            limit = limit if limit > 0.0 else 1.0
            image = ax.pcolormesh(phi_deg, lat_deg, field_wrap, shading="auto", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = ax.pcolormesh(phi_deg, lat_deg, field_wrap, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax, pad=0.01)
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)
        ax.set_ylim(-90.0, 90.0)

    axes[-1].set_xlabel("Longitude [deg]")
    fig.suptitle(f"LCT r_index=0: {time_start_hours:.2f} to {time_end_hours:.2f} h")
    return fig


def main() -> None:
    """Compute and save one LCT map for every adjacent file pair."""
    data_files = select_merged_data_files(DATA_DIR, PROCESS_MODE, SINGLE_DATA_FILE)
    if len(data_files) < 2:
        raise ValueError("At least two merged files are required for LCT.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, (start_file, end_file) in enumerate(zip(data_files[:-1], data_files[1:]), start=1):
        from read_merged_sip_data import simulation_hours_from_filename

        time_start = simulation_hours_from_filename(start_file)
        time_end = simulation_hours_from_filename(end_file)
        dt_hours = time_end - time_start
        if dt_hours <= 0.0:
            raise ValueError("Merged-file times must be strictly increasing.")

        theta, phi, radius_rs, br_start = read_surface_br(start_file)
        theta_end, phi_end, radius_end_rs, br_end = read_surface_br(end_file)
        if not (np.allclose(theta, theta_end) and np.allclose(phi, phi_end) and np.isclose(radius_rs, radius_end_rs)):
            raise ValueError("The two merged files use incompatible r_index=0 grids.")

        displacement_theta, displacement_phi, correlation, reference_shift = local_normalized_correlation(
            br_start, br_end, theta, phi, dt_hours
        )
        radius_km = radius_rs * R_SUN_KM
        dt_seconds = dt_hours * 3600.0
        velocity_theta = radius_km * displacement_theta / dt_seconds
        velocity_phi = radius_km * np.sin(theta)[:, None] * displacement_phi / dt_seconds
        valid = np.isfinite(velocity_theta) & np.isfinite(velocity_phi)

        output_file = velocity_filename(time_start, time_end)
        np.savez_compressed(
            output_file,
            time_start_hours=time_start,
            time_end_hours=time_end,
            dt_hours=dt_hours,
            surface_radial_index=SURFACE_RADIAL_INDEX,
            surface_radius_rs=radius_rs,
            theta=theta,
            phi=phi,
            displacement_theta_rad=displacement_theta.astype(np.float32),
            displacement_phi_rad=displacement_phi.astype(np.float32),
            velocity_theta_km_s=velocity_theta.astype(np.float32),
            velocity_phi_km_s=velocity_phi.astype(np.float32),
            correlation_peak=correlation.astype(np.float32),
            valid_lct=valid,
            reference_phi_shift_rad=reference_shift.astype(np.float32),
            lct_window_sigma_cells=LCT_WINDOW_SIGMA_CELLS,
            lct_search_theta_cells=LCT_SEARCH_THETA_CELLS,
            lct_search_phi_cells=LCT_SEARCH_PHI_CELLS,
            min_correlation=MIN_CORRELATION,
            random_seed=RANDOM_SEED,
        )

        print(
            f"[{index}/{len(data_files) - 1}] {time_start:.2f}->{time_end:.2f} h"
            f" | valid={np.mean(valid):.2%} | saved={output_file}"
        )

        if SAVE_FIGURE:
            fig = make_velocity_figure(theta, phi, velocity_theta, velocity_phi, correlation, time_start, time_end)
            add_figure_provenance(fig, "calculate_lct_surface_velocity.py")
            fig.savefig(figure_filename(time_start, time_end), dpi=DPI, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
