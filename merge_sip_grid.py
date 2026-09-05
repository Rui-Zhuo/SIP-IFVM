"""
merge_sip_grid_202608260111.py

Read the six SIP-IFVM grid components and construct one regular global
spherical grid.  The logic follows the Fortran snippets supplied by the user:

1. Read each component's local Cartesian grid (x_local, y_local, z_local).
2. Convert each component to the common/global Cartesian frame.
3. Build a regular target grid (r, theta, phi).
4. For every target point, search all six components and select the component
   whose grid contains the nearest source point.
5. Save the regular spherical grid plus the selected source-component map.

This script does NOT blend overlapping components.  It follows the Fortran
"nearest-component selection" logic.

Requirements:
    numpy
    scipy
    h5py

Default input:
    config.GRID_DIR
        0_00_0Gridskip1.h5
        ...
        0_00_5Gridskip1.h5

Default output:
    config.GRID_FILE
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from scipy.spatial import cKDTree

from config import GRID_DIR


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

GRID_PATTERN = "0_00_{component}Gridskip1.h5"
OUTPUT_FILE = GRID_DIR / "merged_spherical_grid.h5"

# The Fortran post-processing snippet uses N_th_p=80 and N_ph_p=160.
# These are therefore sensible defaults for the global spherical grid.
N_THETA = 180
N_PHI = 360

# Number of target points processed per KD-tree query batch.
QUERY_CHUNK = 200_000


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------

@dataclass
class HDF5Layout:
    """
    Layout needed to convert a raw HDF5 array to canonical form:
        (channel, radial, angular_1, angular_2)

    channel_axis:
        Axis containing x/y/z (or rho/vx/...).

    spatial_perm:
        After moving channel_axis to axis 0, this tuple gives the order of
        the remaining 3 axes as (radial, angular_1, angular_2).
    """
    channel_axis: int
    spatial_perm: Tuple[int, int, int]


@dataclass
class GridComponent:
    component: int
    x_local: np.ndarray
    y_local: np.ndarray
    z_local: np.ndarray
    x_global: np.ndarray
    y_global: np.ndarray
    z_global: np.ndarray
    layout: HDF5Layout
    simulation_time: float | None
    nyy_component: int | None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.x_local.shape

    @property
    def global_xyz_flat(self) -> np.ndarray:
        return np.column_stack(
            (
                self.x_global.ravel(),
                self.y_global.ravel(),
                self.z_global.ravel(),
            )
        )


# ----------------------------------------------------------------------
# HDF5 helpers
# ----------------------------------------------------------------------

def _read_scalar_attr(dataset: h5py.Dataset, name: str):
    """Return an HDF5 attribute as a Python scalar if it exists."""
    if name not in dataset.attrs:
        return None
    value = dataset.attrs[name]
    arr = np.asarray(value)
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _detect_channel_axis(raw: np.ndarray, expected_channels: int) -> int:
    """
    Detect the axis containing the physical-variable/channel dimension.

    This is intentionally robust to the common Fortran-HDF5 situation where
    h5py may show the dataset dimensions in reversed order.
    """
    candidates = [ax for ax, n in enumerate(raw.shape) if n == expected_channels]
    if not candidates:
        raise ValueError(
            f"Cannot find a channel axis of length {expected_channels} "
            f"in raw dataset shape {raw.shape}."
        )

    # Usually the channel axis is either first or last.
    if 0 in candidates:
        return 0
    if raw.ndim - 1 in candidates:
        return raw.ndim - 1

    # Fall back to the first matching axis.
    return candidates[0]


def _detect_radial_spatial_axis(grid_after_channel: np.ndarray) -> int:
    """
    Detect which of the 3 spatial axes is radial.

    grid_after_channel has shape:
        (3, s0, s1, s2)

    The radial axis should show the strongest systematic change in
    radius = sqrt(x^2+y^2+z^2) when the two other dimensions are averaged.
    """
    x, y, z = grid_after_channel
    radius = np.sqrt(x * x + y * y + z * z)

    scores = []
    for spatial_axis in range(3):
        other_axes = tuple(ax for ax in range(3) if ax != spatial_axis)
        profile = np.nanmedian(radius, axis=other_axes)
        if profile.size <= 1:
            score = -np.inf
        else:
            score = float(np.nanstd(profile))
        scores.append(score)

    radial_axis = int(np.nanargmax(scores))
    return radial_axis


def canonicalize_grid_array(raw: np.ndarray) -> Tuple[np.ndarray, HDF5Layout]:
    """
    Convert a raw grid dataset to canonical shape:
        (3, Nr, Na, Nb)

    Returns both the canonical array and the layout description so that
    physics datasets written by the same Fortran code can be reordered
    identically.
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected a 4-D grid dataset, got shape {raw.shape}")

    channel_axis = _detect_channel_axis(raw, expected_channels=3)
    arr = np.moveaxis(raw, channel_axis, 0)

    radial_axis = _detect_radial_spatial_axis(arr)
    other = [ax for ax in range(3) if ax != radial_axis]
    spatial_perm = (radial_axis, other[0], other[1])

    arr = np.transpose(
        arr,
        axes=(0, 1 + spatial_perm[0], 1 + spatial_perm[1], 1 + spatial_perm[2]),
    )

    layout = HDF5Layout(
        channel_axis=channel_axis,
        spatial_perm=spatial_perm,
    )
    return np.asarray(arr, dtype=np.float64), layout


def canonicalize_with_layout(
    raw: np.ndarray,
    layout: HDF5Layout,
    expected_channels: int,
) -> np.ndarray:
    """
    Apply a grid-derived HDF5 layout to another dataset written on the same
    component grid, returning:
        (channel, Nr, Na, Nb)
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected a 4-D dataset, got shape {raw.shape}")

    # Prefer the grid's channel-axis location if compatible.  Otherwise
    # auto-detect because Fortran/HDF5 libraries can differ across builds.
    if raw.shape[layout.channel_axis] == expected_channels:
        channel_axis = layout.channel_axis
    else:
        channel_axis = _detect_channel_axis(raw, expected_channels)

    arr = np.moveaxis(raw, channel_axis, 0)

    arr = np.transpose(
        arr,
        axes=(
            0,
            1 + layout.spatial_perm[0],
            1 + layout.spatial_perm[1],
            1 + layout.spatial_perm[2],
        ),
    )
    return np.asarray(arr, dtype=np.float64)


# ----------------------------------------------------------------------
# Six-component coordinate transformations
# ----------------------------------------------------------------------

def local_to_global_xyz(
    component: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one component's LOCAL Cartesian coordinates to GLOBAL Cartesian.

    This is the inverse of the transformations used in sphB_interp:

        global -> local
        0: ( x,  y,  z)
        1: ( y, -x,  z)
        2: (-x, -y,  z)
        3: (-y,  x,  z)
        4: ( z,  y, -x)
        5: (-z,  y,  x)

    Therefore local -> global is:
        0: ( x',  y',  z')
        1: (-y',  x',  z')
        2: (-x', -y',  z')
        3: ( y', -x',  z')
        4: (-z',  y',  x')
        5: ( z',  y', -x')
    """
    if component == 0:
        return x, y, z
    if component == 1:
        return -y, x, z
    if component == 2:
        return -x, -y, z
    if component == 3:
        return y, -x, z
    if component == 4:
        return -z, y, x
    if component == 5:
        return z, y, -x
    raise ValueError(f"component must be 0..5, got {component}")


def global_to_local_xyz(
    component: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global -> local transform exactly matching sphB_interp."""
    if component == 0:
        return x, y, z
    if component == 1:
        return y, -x, z
    if component == 2:
        return -x, -y, z
    if component == 3:
        return -y, x, z
    if component == 4:
        return z, y, -x
    if component == 5:
        return -z, y, x
    raise ValueError(f"component must be 0..5, got {component}")


def local_vector_to_global(
    component: int,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a vector from the component-local Cartesian frame to the global
    Cartesian frame.  This is the same rotation used for B in sphB_interp.
    """
    return local_to_global_xyz(component, vx, vy, vz)


# ----------------------------------------------------------------------
# Reading the six grid files
# ----------------------------------------------------------------------

def read_grid_component(filename: Path, component: int) -> GridComponent:
    """Read one grid HDF5 file and transform coordinates to the global frame."""
    with h5py.File(filename, "r") as f:
        if "Variables" not in f:
            raise KeyError(f'{filename} does not contain dataset "Variables".')

        dset = f["Variables"]
        raw = dset[...]
        simulation_time = _read_scalar_attr(dset, "simulation_time")
        nyy_component = _read_scalar_attr(dset, "nyy_component")

    arr, layout = canonicalize_grid_array(raw)
    x_local, y_local, z_local = arr

    x_global, y_global, z_global = local_to_global_xyz(
        component, x_local, y_local, z_local
    )

    return GridComponent(
        component=component,
        x_local=x_local,
        y_local=y_local,
        z_local=z_local,
        x_global=x_global,
        y_global=y_global,
        z_global=z_global,
        layout=layout,
        simulation_time=simulation_time,
        nyy_component=None if nyy_component is None else int(nyy_component),
    )


def read_all_grid_components(
    grid_dir: Path = GRID_DIR,
    pattern: str = GRID_PATTERN,
) -> List[GridComponent]:
    """Read components 0..5."""
    components: List[GridComponent] = []

    for component in range(6):
        filename = grid_dir / pattern.format(component=component)
        if not filename.exists():
            raise FileNotFoundError(filename)

        grid = read_grid_component(filename, component)
        print(
            f"[grid {component}] file={filename.name}, "
            f"shape={grid.shape}, "
            f"attr_nyy={grid.nyy_component}, "
            f"time={grid.simulation_time}"
        )
        components.append(grid)

    return components


# ----------------------------------------------------------------------
# Target spherical grid
# ----------------------------------------------------------------------

def infer_radial_shells(components: Sequence[GridComponent]) -> np.ndarray:
    """
    Infer the native radial shells from the six grids.

    For every component and radial index, compute the median radius over its
    two angular dimensions, then average the six components.  This preserves
    the simulation's native radial sampling instead of inventing a new r-grid.
    """
    profiles = []

    for comp in components:
        radius = np.sqrt(
            comp.x_global**2 + comp.y_global**2 + comp.z_global**2
        )
        profile = np.nanmedian(radius, axis=(1, 2))
        profiles.append(profile)

    lengths = {len(p) for p in profiles}
    if len(lengths) != 1:
        raise ValueError(
            "The six components do not have the same number of radial shells: "
            f"{sorted(lengths)}"
        )

    r = np.nanmedian(np.vstack(profiles), axis=0)

    # Ensure increasing radial order.  If the source happens to be reversed,
    # sort the target shells here; source arrays remain unchanged because
    # nearest-neighbour searching is done geometrically.
    r = np.asarray(r, dtype=np.float64)
    r = np.sort(r)

    return r


def make_target_spherical_grid(
    r: np.ndarray,
    n_theta: int = N_THETA,
    n_phi: int = N_PHI,
):
    """
    Return 1-D coordinate axes plus 3-D global Cartesian target coordinates.

    theta is colatitude in [0, pi].
    phi is longitude in [0, 2*pi), endpoint excluded.
    """
    theta = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)

    rr = r[:, None, None]
    tt = theta[None, :, None]
    pp = phi[None, None, :]

    sin_t = np.sin(tt)
    x = rr * sin_t * np.cos(pp)
    y = rr * sin_t * np.sin(pp)
    z = rr * np.cos(tt) * np.ones_like(pp)

    return r, theta, phi, x, y, z


# ----------------------------------------------------------------------
# Component selection
# ----------------------------------------------------------------------

def build_component_trees(
    components: Sequence[GridComponent],
) -> List[cKDTree]:
    """Build one KD-tree per component in the GLOBAL Cartesian frame."""
    return [cKDTree(comp.global_xyz_flat) for comp in components]


def find_best_component(
    target_xyz: np.ndarray,
    trees: Sequence[cKDTree],
    chunk_size: int = QUERY_CHUNK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every target point, choose the component with the nearest source point.

    Returns
    -------
    best_component : int8, shape (N,)
        Values 0..5.

    nearest_flat_index : int64, shape (N,)
        Flat source-grid index inside the selected component.

    nearest_distance : float64, shape (N,)
        Euclidean distance to that source point.

    This reproduces the selection logic of sphB_interp:
        search component 0
        search component 1
        ...
        keep the smallest detmin
    """
    n = target_xyz.shape[0]

    best_component = np.empty(n, dtype=np.int8)
    nearest_flat_index = np.empty(n, dtype=np.int64)
    nearest_distance = np.empty(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        pts = target_xyz[start:stop]
        m = len(pts)

        best_d = np.full(m, np.inf, dtype=np.float64)
        best_c = np.full(m, -1, dtype=np.int8)
        best_i = np.full(m, -1, dtype=np.int64)

        for component, tree in enumerate(trees):
            d, idx = tree.query(pts, k=1, workers=-1)
            use = d < best_d
            best_d[use] = d[use]
            best_c[use] = component
            best_i[use] = idx[use]

        best_component[start:stop] = best_c
        nearest_flat_index[start:stop] = best_i
        nearest_distance[start:stop] = best_d

        print(f"component selection: {stop:,}/{n:,} target points")

    return best_component, nearest_flat_index, nearest_distance


# ----------------------------------------------------------------------
# Save merged spherical grid
# ----------------------------------------------------------------------

def save_merged_grid(
    output_file: Path,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    best_component: np.ndarray,
    nearest_flat_index: np.ndarray,
    nearest_distance: np.ndarray,
    components: Sequence[GridComponent],
):
    """
    Save the STATIC merged spherical grid.

    Compared with the compact 202608260058 version, this version stores a
    more complete static grid description, including both spherical and
    Cartesian coordinates:

        r
        theta
        phi

        x
        y
        z

        x_m
        y_m
        z_m

        source_component
        nearest_source_distance
        component_shapes

    The grid is still saved only once, so the extra storage cost is limited
    to a single file rather than repeated for every simulation time.
    """
    shape = x.shape

    component_map = best_component.reshape(shape)
    distance_map = nearest_distance.reshape(shape)

    component_shapes = np.asarray(
        [c.shape for c in components],
        dtype=np.int64,
    )

    # Coordinates in meters.
    Rs_m = 6.963e8

    x_m = x * Rs_m
    y_m = y * Rs_m
    z_m = z * Rs_m

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        output_file,
        "w",
    ) as f:

        f.attrs["description"] = (
            "Static regular global spherical grid reconstructed from six "
            "SIP-IFVM components. Time-dependent physical variables are "
            "stored separately."
        )

        f.attrs["theta_definition"] = (
            "colatitude, radians, 0..pi"
        )

        f.attrs["phi_definition"] = (
            "longitude, radians, 0..2pi, endpoint excluded"
        )

        f.attrs["component_selection"] = (
            "Choose the component whose source grid contains the nearest point."
        )

        f.attrs["source_grid_pattern"] = GRID_PATTERN
        f.attrs["Rs_m"] = Rs_m

        f.attrs["Nr"] = int(len(r))
        f.attrs["Ntheta"] = int(len(theta))
        f.attrs["Nphi"] = int(len(phi))

        f.create_dataset(
            "r",
            data=r,
        )

        f.create_dataset(
            "theta",
            data=theta,
        )

        f.create_dataset(
            "phi",
            data=phi,
        )

        f["r"].attrs["unit"] = "Rs"
        f["theta"].attrs["unit"] = "rad"
        f["phi"].attrs["unit"] = "rad"

        for name, arr, unit in (
            ("x", x, "Rs"),
            ("y", y, "Rs"),
            ("z", z, "Rs"),
            ("x_m", x_m, "m"),
            ("y_m", y_m, "m"),
            ("z_m", z_m, "m"),
        ):
            dset = f.create_dataset(
                name,
                data=arr,
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["unit"] = unit

        f.create_dataset(
            "source_component",
            data=component_map,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "nearest_source_distance",
            data=distance_map,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "component_shapes",
            data=component_shapes,
        )

    print(
        f"\nSaved static merged spherical grid:\n"
        f"{output_file}"
    )

    print(
        f"target shape = {shape}"
    )

def main():
    components = read_all_grid_components()

    r = infer_radial_shells(components)
    r, theta, phi, x, y, z = make_target_spherical_grid(
        r,
        n_theta=N_THETA,
        n_phi=N_PHI,
    )

    print(f"\nTarget spherical grid: Nr={len(r)}, Ntheta={len(theta)}, Nphi={len(phi)}")

    trees = build_component_trees(components)

    target_xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    best_component, nearest_flat_index, nearest_distance = find_best_component(
        target_xyz,
        trees,
    )

    save_merged_grid(
        OUTPUT_FILE,
        r,
        theta,
        phi,
        x,
        y,
        z,
        best_component,
        nearest_flat_index,
        nearest_distance,
        components,
    )


if __name__ == "__main__":
    main()
