"""Advect all r_index=0 magnetic elements with saved LCT velocity maps.

Dependencies: numpy==1.26.4, scipy>=1.10,<1.15.
The first advance samples the LCT map directly on its native seed grid;
later advances bilinearly interpolate each new velocity map at the advected
positions. Each saved file contains persistent `id`, theta, and phi [rad]
for all seeds; IDs are identical at every saved time.

Outputs: `lct_footpoint_track.time.<t>.npz` for each time and
`lct_footpoint_track_manifest.npz`.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.ndimage import map_coordinates

from config import WORK_ROOT
from utils import differential_rotation_rate_deg_per_day


# ======================================================================
# CONFIGURATION
# ======================================================================

VELOCITY_DIR = WORK_ROOT / "lct_velocity"
OUTPUT_DIR = WORK_ROOT / "lct_surface_tracks"
R_SUN_KM = 695700.0
POLAR_SIN_THETA_MIN = 0.05

RANDOM_SEED = 42


# ======================================================================
# FILE HELPERS
# ======================================================================

VELOCITY_PATTERN = re.compile(
    r"^lct_footpoint_velocity\.time\.([0-9]+(?:\.[0-9]+)?)_to_"
    r"([0-9]+(?:\.[0-9]+)?)\.npz$"
)


def track_filename(time_hours: float) -> Path:
    return OUTPUT_DIR / f"lct_footpoint_track.time.{time_hours:.2f}.npz"


def manifest_filename() -> Path:
    return OUTPUT_DIR / "lct_footpoint_track_manifest.npz"


def discover_velocity_files() -> list[tuple[float, float, Path]]:
    """Return adjacent LCT map files ordered by start time."""
    if not VELOCITY_DIR.is_dir():
        raise FileNotFoundError(VELOCITY_DIR)

    entries = []
    for filename in VELOCITY_DIR.iterdir():
        match = VELOCITY_PATTERN.fullmatch(filename.name)
        if match is not None:
            entries.append((float(match.group(1)), float(match.group(2)), filename))

    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(f"No LCT velocity files found in {VELOCITY_DIR}")

    for previous, current in zip(entries[:-1], entries[1:]):
        if not np.isclose(previous[1], current[0], rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"Non-contiguous velocity maps: {previous[2].name} -> {current[2].name}"
            )

    return entries


def load_velocity_map(filename: Path) -> dict[str, np.ndarray | float]:
    """Read one validated LCT velocity map."""
    required = (
        "time_start_hours", "time_end_hours", "dt_hours", "surface_radius_rs",
        "theta", "phi", "velocity_theta_km_s", "velocity_phi_km_s", "valid_lct",
    )
    with np.load(filename, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {filename}")
        result = {key: np.asarray(data[key]).copy() for key in required}

    theta = np.asarray(result["theta"], dtype=float)
    phi = np.asarray(result["phi"], dtype=float)
    shape = (theta.size, phi.size)
    for key in ("velocity_theta_km_s", "velocity_phi_km_s", "valid_lct"):
        if np.asarray(result[key]).shape != shape:
            raise ValueError(f"{key} shape is incompatible with theta-phi grid.")

    result["theta"] = theta
    result["phi"] = phi
    result["time_start_hours"] = float(result["time_start_hours"])
    result["time_end_hours"] = float(result["time_end_hours"])
    result["dt_hours"] = float(result["dt_hours"])
    result["surface_radius_rs"] = float(result["surface_radius_rs"])
    return result


# ======================================================================
# SPHERICAL ADVECTION
# ======================================================================

def fractional_grid_coordinates(theta: np.ndarray, phi: np.ndarray, grid_theta: np.ndarray, grid_phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical theta-phi points to fractional uniform-grid indices."""
    dtheta = float(np.mean(np.diff(grid_theta)))
    dphi = float(np.mean(np.diff(grid_phi)))
    if not np.allclose(np.diff(grid_theta), dtheta, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("Velocity interpolation requires a uniform theta grid.")
    if not np.allclose(np.diff(grid_phi), dphi, rtol=1.0e-5, atol=1.0e-10):
        raise ValueError("Velocity interpolation requires a uniform phi grid.")
    theta_index = np.clip((theta - grid_theta[0]) / dtheta, 0.0, grid_theta.size - 1.0)
    phi_index = ((phi - grid_phi[0]) / dphi) % grid_phi.size
    return theta_index, phi_index


def interpolate_field(field: np.ndarray, theta: np.ndarray, phi: np.ndarray, grid_theta: np.ndarray, grid_phi: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a scalar field with periodic longitude."""
    theta_index, phi_index = fractional_grid_coordinates(theta, phi, grid_theta, grid_phi)
    periodic_field = np.concatenate((field, field[:, :1]), axis=1)
    return map_coordinates(periodic_field, [theta_index, phi_index], order=1, mode="nearest")


def normalize_spherical_angles(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect colatitude across poles and keep longitude periodic."""
    theta_mod = np.mod(theta, 2.0 * np.pi)
    crossed_south = theta_mod > np.pi
    theta_out = np.where(crossed_south, 2.0 * np.pi - theta_mod, theta_mod)
    phi_out = np.where(crossed_south, phi + np.pi, phi)
    return theta_out, np.mod(phi_out, 2.0 * np.pi)


def advance_positions(theta: np.ndarray, phi: np.ndarray, velocity_theta: np.ndarray, velocity_phi: np.ndarray, valid: np.ndarray, radius_rs: float, dt_hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance positions one interval and use differential rotation as fallback."""
    radius_km = radius_rs * R_SUN_KM
    dt_seconds = dt_hours * 3600.0
    sin_theta = np.sin(theta)
    finite_lct = valid & np.isfinite(velocity_theta) & np.isfinite(velocity_phi)

    delta_theta = np.zeros_like(theta, dtype=float)
    delta_phi = np.radians(
        differential_rotation_rate_deg_per_day(90.0 - np.degrees(theta)) * dt_hours / 24.0
    )

    delta_theta[finite_lct] = velocity_theta[finite_lct] * dt_seconds / radius_km
    safe = finite_lct & (np.abs(sin_theta) >= POLAR_SIN_THETA_MIN)
    delta_phi[safe] = velocity_phi[safe] * dt_seconds / (radius_km * sin_theta[safe])

    theta_next, phi_next = normalize_spherical_angles(theta + delta_theta, phi + delta_phi)
    return theta_next, phi_next, finite_lct


def save_track(time_hours: float, ids: np.ndarray, theta: np.ndarray, phi: np.ndarray, seed_shape: tuple[int, int], step_valid_fraction: float | None) -> Path:
    """Save all Lagrangian r_index=0 positions at one time."""
    output_file = track_filename(time_hours)
    np.savez_compressed(
        output_file,
        time_hours=float(time_hours),
        id=np.asarray(ids, dtype=np.int64),
        seed_shape=np.asarray(seed_shape, dtype=np.int32),
        theta_rad=theta.astype(np.float32),
        phi_rad=phi.astype(np.float32),
        latitude_deg=(90.0 - np.degrees(theta)).astype(np.float32),
        longitude_deg=(np.degrees(phi) % 360.0).astype(np.float32),
        lct_valid_fraction=np.nan if step_valid_fraction is None else float(step_valid_fraction),
        random_seed=RANDOM_SEED,
    )
    return output_file


def main() -> None:
    """Build a complete sequence of r_index=0 LCT trajectories."""
    entries = discover_velocity_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_map = load_velocity_map(entries[0][2])
    grid_theta = np.asarray(first_map["theta"], dtype=float)
    grid_phi = np.asarray(first_map["phi"], dtype=float)
    seed_theta_grid, seed_phi_grid = np.meshgrid(grid_theta, grid_phi, indexing="ij")
    theta = seed_theta_grid.ravel().copy()
    phi = seed_phi_grid.ravel().copy()
    ids = np.arange(theta.size, dtype=np.int64)
    seed_shape = seed_theta_grid.shape

    saved_times = [float(first_map["time_start_hours"])]
    saved_files = [str(save_track(saved_times[0], ids, theta, phi, seed_shape, None).name)]

    for step, (time_start, time_end, filename) in enumerate(entries):
        velocity = load_velocity_map(filename)
        if not (np.isclose(time_start, velocity["time_start_hours"]) and np.isclose(time_end, velocity["time_end_hours"])):
            raise ValueError(f"Time metadata differs from velocity filename: {filename}")
        if not (np.allclose(grid_theta, velocity["theta"]) and np.allclose(grid_phi, velocity["phi"])):
            raise ValueError("All LCT maps must use the same theta-phi grid.")

        if step == 0:
            velocity_theta = np.asarray(velocity["velocity_theta_km_s"], dtype=float).ravel()
            velocity_phi = np.asarray(velocity["velocity_phi_km_s"], dtype=float).ravel()
            valid = np.asarray(velocity["valid_lct"], dtype=bool).ravel()
        else:
            velocity_theta = interpolate_field(np.asarray(velocity["velocity_theta_km_s"], dtype=float), theta, phi, grid_theta, grid_phi)
            velocity_phi = interpolate_field(np.asarray(velocity["velocity_phi_km_s"], dtype=float), theta, phi, grid_theta, grid_phi)
            valid_float = interpolate_field(np.asarray(velocity["valid_lct"], dtype=float), theta, phi, grid_theta, grid_phi)
            valid = valid_float >= 0.5

        theta, phi, used_lct = advance_positions(
            theta, phi, velocity_theta, velocity_phi, valid,
            float(velocity["surface_radius_rs"]), float(velocity["dt_hours"]),
        )
        output_file = save_track(time_end, ids, theta, phi, seed_shape, float(np.mean(used_lct)))
        saved_times.append(time_end)
        saved_files.append(str(output_file.name))
        print(f"[{step + 1}/{len(entries)}] {time_start:.2f}->{time_end:.2f} h | LCT={np.mean(used_lct):.2%} | saved={output_file}")

    np.savez_compressed(
        manifest_filename(),
        times_hours=np.asarray(saved_times, dtype=float),
        filenames=np.asarray(saved_files),
        seed_shape=np.asarray(seed_shape, dtype=np.int32),
        id=ids,
        velocity_directory=str(VELOCITY_DIR),
        random_seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
