"""Trace R0 crossings from all LCT-tracked r_index=0 footpoints.

Each LCT footpoint ID is traced independently at every saved time using the
same spherical RK2 magnetic-field-line equations as
``track_crossings_connectivity.py``. No nearest-neighbour mapping is used.

Outputs: `lct_footpoint_connectivity.time.<t>.r.<R0>.npz` for every time and
`lct_footpoint_connectivity_manifest.r.<R0>.npz`.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from config import FULL_MERGED_DIR, GRID_FILE, WORK_ROOT
from read_merged_sip_data import discover_merged_data_files, simulation_hours_from_filename
import track_crossings_connectivity as fieldline_trace


LCT_TRACK_DIR = WORK_ROOT / "lct_surface_tracks"
DATA_DIR = FULL_MERGED_DIR / "82d1to132"
OUTPUT_DIR = WORK_ROOT / "lct_footpoint_connectivity"
R0 = 10.0
TIME_TOL_HOURS = 1.0e-6
LCT_TRACK_PATTERN = re.compile(r"^lct_footpoint_track\.time\.([0-9]+(?:\.[0-9]+)?)\.npz$")

TRACE_ACTIVE = np.int8(0)
TRACE_REACHED_R0 = np.int8(1)
TRACE_CLOSED_BELOW_R0 = np.int8(2)
TRACE_FAILED = np.int8(3)
TRACE_MAX_STEPS = np.int8(4)


def connectivity_filename(time_hours: float) -> Path:
    """Return the per-time all-ID connectivity filename."""
    return OUTPUT_DIR / f"lct_footpoint_connectivity.time.{time_hours:.2f}.r.{R0:g}.npz"


def manifest_filename() -> Path:
    """Return the all-time connectivity manifest filename."""
    return OUTPUT_DIR / f"lct_footpoint_connectivity_manifest.r.{R0:g}.npz"


def discover_lct_tracks() -> list[tuple[float, Path]]:
    """Return LCT footpoint tracks ordered by time."""
    if not LCT_TRACK_DIR.is_dir():
        raise FileNotFoundError(LCT_TRACK_DIR)
    entries = []
    for filename in LCT_TRACK_DIR.iterdir():
        match = LCT_TRACK_PATTERN.fullmatch(filename.name)
        if match is not None:
            entries.append((float(match.group(1)), filename))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(f"No LCT footpoint tracks in {LCT_TRACK_DIR}")
    return entries


def discover_merged_by_time() -> dict[float, Path]:
    """Map simulation hour to its merged physical-data file."""
    files = discover_merged_data_files(DATA_DIR)
    if not files:
        raise FileNotFoundError(f"No merged files in {DATA_DIR}")
    return {float(simulation_hours_from_filename(filename)): filename for filename in files}


def matching_merged_file(files_by_time: dict[float, Path], time_hours: float) -> Path:
    """Return the merged file matching one LCT time."""
    matches = [time for time in files_by_time if np.isclose(time, time_hours, rtol=0.0, atol=TIME_TOL_HOURS)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one merged file at {time_hours:.8f} h; found {len(matches)}")
    return files_by_time[matches[0]]


def load_lct_footpoints(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load persistent IDs and r_index=0 footpoint coordinates."""
    with np.load(filename, allow_pickle=False) as data:
        required = ("id", "theta_rad", "phi_rad")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {filename}; rerun track_footpoints_lct.py.")
        ids = np.asarray(data["id"], dtype=np.int64)
        theta = np.asarray(data["theta_rad"], dtype=float)
        phi = np.asarray(data["phi_rad"], dtype=float) % (2.0 * np.pi)
    if ids.ndim != 1 or theta.shape != ids.shape or phi.shape != ids.shape:
        raise ValueError(f"Incompatible ID/theta/phi shapes in {filename}")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"LCT IDs are not unique in {filename}")
    return ids, theta, phi


def trace_footpoints_to_r0(
    footpoint_theta: np.ndarray,
    footpoint_phi: np.ndarray,
    magnetic_interpolator,
    r_inner: float,
    r_grid_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace every footpoint outward and linearly interpolate first R0 crossing."""
    nseed = footpoint_theta.size
    launch_radius = float(r_inner) + max(1.0e-8, 1.0e-6 * (R0 - float(r_inner)))
    states = fieldline_trace.normalize_states(
        np.column_stack((np.full(nseed, launch_radius), footpoint_theta, footpoint_phi))
    )
    magnetic_field = magnetic_interpolator(states)
    br = magnetic_field[:, 0]
    finite = np.all(np.isfinite(magnetic_field), axis=1) & np.isfinite(br) & (np.abs(br) > fieldline_trace.MIN_BMAG)
    directions = np.where(br >= 0.0, 1.0, -1.0)
    status = np.full(nseed, TRACE_ACTIVE, dtype=np.int8)
    status[~finite] = TRACE_FAILED
    crossings = np.full((nseed, 3), np.nan, dtype=float)

    for _ in range(fieldline_trace.MAX_FIELDLINE_STEPS):
        active = np.flatnonzero(status == TRACE_ACTIVE)
        if active.size == 0:
            break
        previous = states[active].copy()
        current, valid = fieldline_trace.rk2_fieldline_step_batch(previous, directions[active], magnetic_interpolator)
        if np.any(~valid):
            status[active[~valid]] = TRACE_FAILED
        if not np.any(valid):
            continue
        good = active[valid]
        prev_good = previous[valid]
        curr_good = current[valid]
        states[good] = curr_good
        crossed = (
            ((prev_good[:, 0] - R0) * (curr_good[:, 0] - R0) <= 0.0)
            & (np.abs(curr_good[:, 0] - prev_good[:, 0]) > 1.0e-14)
        )
        if np.any(crossed):
            reached = good[crossed]
            crossings[reached] = fieldline_trace.interpolate_radial_crossing(prev_good, curr_good, crossed, R0)
            status[reached] = TRACE_REACHED_R0
        returned_inner = (~crossed) & (curr_good[:, 0] <= r_inner + fieldline_trace.INNER_TOL)
        if np.any(returned_inner):
            status[good[returned_inner]] = TRACE_CLOSED_BELOW_R0
        escaped = (~crossed) & ~returned_inner & (curr_good[:, 0] > r_grid_max)
        if np.any(escaped):
            status[good[escaped]] = TRACE_FAILED

    status[status == TRACE_ACTIVE] = TRACE_MAX_STEPS
    return crossings, status


def main() -> None:
    """Trace all LCT IDs at each time and save their R0 crossings."""
    lct_tracks = discover_lct_tracks()
    merged_by_time = discover_merged_by_time()
    r, theta_grid, phi_grid = fieldline_trace.read_grid(GRID_FILE)
    r_inner = float(r[fieldline_trace.INNER_RADIAL_INDEX])
    if not (r_inner < R0 < float(np.max(r))):
        raise ValueError(f"R0={R0:g} Rs is outside ({r_inner:g}, {np.max(r):g}) Rs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_ids = None
    saved_files = []
    for index, (time_hours, lct_file) in enumerate(lct_tracks, start=1):
        ids, foot_theta, foot_phi = load_lct_footpoints(lct_file)
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"LCT IDs differ from the initial track: {lct_file}")
        merged_file = matching_merged_file(merged_by_time, time_hours)
        fields = fieldline_trace.read_fields(merged_file)
        fieldline_trace.validate_field_shapes(fields, r, theta_grid, phi_grid)
        magnetic_interpolator, _ = fieldline_trace.build_interpolators(fields, r, theta_grid, phi_grid)
        crossings, trace_status = trace_footpoints_to_r0(
            foot_theta, foot_phi, magnetic_interpolator, r_inner, float(np.max(r)),
        )
        output_file = connectivity_filename(time_hours)
        np.savez_compressed(
            output_file,
            time_hours=float(time_hours),
            id=ids,
            footpoint_theta_rad=foot_theta.astype(np.float32),
            footpoint_phi_rad=foot_phi.astype(np.float32),
            footpoint_longitude_deg=(np.degrees(foot_phi) % 360.0).astype(np.float32),
            footpoint_latitude_deg=(90.0 - np.degrees(foot_theta)).astype(np.float32),
            crossing_r_rs=crossings[:, 0].astype(np.float32),
            crossing_theta_rad=crossings[:, 1].astype(np.float32),
            crossing_phi_rad=crossings[:, 2].astype(np.float32),
            crossing_longitude_deg=(np.degrees(crossings[:, 2]) % 360.0).astype(np.float32),
            crossing_latitude_deg=(90.0 - np.degrees(crossings[:, 1])).astype(np.float32),
            crossing_trace_status=trace_status,
            r0_rs=float(R0),
        )
        saved_files.append(output_file.name)
        valid_fraction = np.mean(trace_status == TRACE_REACHED_R0)
        print(f"[{index}/{len(lct_tracks)}] {time_hours:.2f} h | crossings={valid_fraction:.2%} | saved={output_file}")

    np.savez_compressed(
        manifest_filename(),
        time_hours=np.asarray([time for time, _ in lct_tracks], dtype=float),
        filenames=np.asarray(saved_files),
        id=reference_ids,
        r0_rs=float(R0),
    )


if __name__ == "__main__":
    main()
