"""
plot_merged_sip_grid_202608260058.py

Use PyVista to plot the SIX ORIGINAL SIP-IFVM grid-component surfaces
on the same spherical shell.

This version:
    - is compatible with the new separated merged-data workflow;
    - still visualizes the SIX ORIGINAL grid components directly;
    - therefore does not need any time-dependent merged physics file;
    - does NOT use scatter points;
    - does NOT use the merged source_component map;
    - reads the six original grid HDF5 files directly;
    - extracts one radial shell from each component;
    - transforms each component from its local Cartesian frame to the
      common/global Cartesian frame;
    - constructs one PyVista StructuredGrid surface per component;
    - plots all six semi-transparent surfaces together so overlap regions
      are visible.

Input files:
    E:/Research/Data/SIP-IFVM/grid/
        0_00_0Gridskip1.h5
        0_00_1Gridskip1.h5
        ...
        0_00_5Gridskip1.h5

Requirements:
    numpy
    h5py
    pyvista
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


# ======================================================================
# Configuration
# ======================================================================

GRID_DIR = Path(r"E:\Research\Data\SIP-IFVM\grid")
GRID_PATTERN = "0_00_{component}Gridskip1.h5"

# Choose ONE method:
#
# 1. Select by radial index:
RADIAL_INDEX = 0
RADIUS = None
#
# 2. Or select by physical/model radius:
# RADIAL_INDEX = None
# RADIUS = 10.0

# Plot appearance
OPACITY = 1
SHOW_EDGES = True
EDGE_OPACITY = 0.35
LINE_WIDTH = 0.7

BACKGROUND = "white"
WINDOW_SIZE = (1400, 1000)
SHOW_AXES = True

# Six visually distinct colors
COMPONENT_COLORS = [
    "royalblue",
    "darkorange",
    "seagreen",
    "crimson",
    "mediumpurple",
    "sienna",
]


# ======================================================================
# HDF5 dimension handling
# ======================================================================

def detect_channel_axis(raw: np.ndarray, expected_channels: int = 3) -> int:
    """
    Detect the axis corresponding to x/y/z.

    Handles both common layouts:
        (3, Nr, Ntheta, Nphi)
    and possible Fortran/HDF5 reversed layouts such as:
        (Nphi, Ntheta, Nr, 3)
    """
    candidates = [
        axis for axis, n in enumerate(raw.shape)
        if n == expected_channels
    ]

    if not candidates:
        raise ValueError(
            f"Cannot find axis of length {expected_channels} "
            f"in dataset shape {raw.shape}"
        )

    if 0 in candidates:
        return 0

    if raw.ndim - 1 in candidates:
        return raw.ndim - 1

    return candidates[0]


def detect_radial_axis(grid_xyz: np.ndarray) -> int:
    """
    Determine which spatial axis is radial.

    Parameters
    ----------
    grid_xyz : ndarray
        Shape (3, s0, s1, s2).

    Returns
    -------
    int
        Spatial radial-axis index: 0, 1, or 2.
    """
    x, y, z = grid_xyz
    radius = np.sqrt(x**2 + y**2 + z**2)

    scores = []

    for axis in range(3):
        other_axes = tuple(i for i in range(3) if i != axis)
        radial_profile = np.nanmedian(radius, axis=other_axes)
        scores.append(float(np.nanstd(radial_profile)))

    return int(np.argmax(scores))


def canonicalize_grid(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw HDF5 grid to canonical shape:

        (3, Nr, Na, Nb)

    where:
        0 -> x_local
        1 -> y_local
        2 -> z_local
    """
    if raw.ndim != 4:
        raise ValueError(
            f"Expected 4-D Variables dataset, got {raw.shape}"
        )

    channel_axis = detect_channel_axis(raw, 3)
    arr = np.moveaxis(raw, channel_axis, 0)

    radial_axis = detect_radial_axis(arr)

    angular_axes = [
        axis for axis in range(3)
        if axis != radial_axis
    ]

    arr = np.transpose(
        arr,
        axes=(
            0,
            1 + radial_axis,
            1 + angular_axes[0],
            1 + angular_axes[1],
        ),
    )

    return np.asarray(arr, dtype=np.float64)


# ======================================================================
# Coordinate transformations
# ======================================================================

def local_to_global_xyz(
    component: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
):
    """
    Transform LOCAL Cartesian coordinates of each original grid component
    into GLOBAL Cartesian coordinates.

    This is the inverse of the transforms used in the supplied Fortran
    sphB_interp routine:

    global -> local:
        0: ( x,  y,  z)
        1: ( y, -x,  z)
        2: (-x, -y,  z)
        3: (-y,  x,  z)
        4: ( z,  y, -x)
        5: (-z,  y,  x)

    Therefore local -> global:
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

    raise ValueError(
        f"component must be between 0 and 5, got {component}"
    )


# ======================================================================
# Read original grid component
# ======================================================================

def read_grid_component(
    filename: Path,
    component: int,
):
    """
    Read one original grid HDF5 component.

    Returns
    -------
    x_local, y_local, z_local : ndarray
        Shape (Nr, Na, Nb)
    """
    if not filename.exists():
        raise FileNotFoundError(filename)

    with h5py.File(filename, "r") as f:
        if "Variables" not in f:
            raise KeyError(
                f'{filename} does not contain "Variables"'
            )

        raw = f["Variables"][...]

        nyy = None
        if "nyy_component" in f["Variables"].attrs:
            nyy = np.asarray(
                f["Variables"].attrs["nyy_component"]
            ).reshape(-1)[0]

    arr = canonicalize_grid(raw)

    x_local = arr[0]
    y_local = arr[1]
    z_local = arr[2]

    print(
        f"[component {component}] "
        f"shape={x_local.shape}, "
        f"attr_nyy={nyy}"
    )

    return x_local, y_local, z_local


# ======================================================================
# Radial-shell selection
# ======================================================================

def radial_profile(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
):
    """
    Return median radius of every native radial shell.
    """
    rr = np.sqrt(x**2 + y**2 + z**2)
    return np.nanmedian(rr, axis=(1, 2))


def choose_radial_index(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radial_index: int | None,
    radius: float | None,
):
    """
    Choose a shell either by explicit array index or by nearest radius.
    """
    profile = radial_profile(x, y, z)

    if radius is not None:
        idx = int(np.argmin(np.abs(profile - radius)))
        return idx, float(profile[idx])

    if radial_index is None:
        raise ValueError(
            "Set either RADIAL_INDEX or RADIUS."
        )

    if radial_index < 0 or radial_index >= len(profile):
        raise IndexError(
            f"RADIAL_INDEX={radial_index} outside "
            f"0..{len(profile)-1}"
        )

    return int(radial_index), float(profile[radial_index])


# ======================================================================
# PyVista mesh construction
# ======================================================================

def make_surface_structured_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
):
    """
    Build a 2-D PyVista StructuredGrid surface from an original
    (Na, Nb) component shell.

    We explicitly use Fortran ordering because StructuredGrid dimensions
    are interpreted in i-j-k order.
    """
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x/y/z shell shapes differ: "
            f"{x.shape}, {y.shape}, {z.shape}"
        )

    na, nb = x.shape

    grid = pv.StructuredGrid()

    points = np.column_stack(
        (
            x.ravel(order="F"),
            y.ravel(order="F"),
            z.ravel(order="F"),
        )
    )

    grid.points = points
    grid.dimensions = (na, nb, 1)

    return grid


# ======================================================================
# Plot
# ======================================================================

def plot_six_original_surfaces(
    surfaces,
    shell_radii,
):
    """
    Plot all six original component surfaces together.

    Since the surfaces are semi-transparent, overlapping portions remain
    visible rather than being reduced to one selected component.
    """
    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
    )

    plotter.set_background(BACKGROUND)

    for comp, surface in enumerate(surfaces):
        plotter.add_mesh(
            surface,
            color=COMPONENT_COLORS[comp],
            opacity=OPACITY,
            show_edges=SHOW_EDGES,
            edge_color=COMPONENT_COLORS[comp],
            line_width=LINE_WIDTH,
            label=f"Component {comp}",
            name=f"component_{comp}",
        )

    # Equal scale
    plotter.set_scale(1.0, 1.0, 1.0)

    if SHOW_AXES:
        plotter.add_axes(
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )

    radius_text = np.mean(shell_radii)

    plotter.add_text(
        "SIP-IFVM six original grid surfaces\n"
        f"mean shell radius = {radius_text:.6g}",
        position="upper_left",
        font_size=12,
    )

    legend_entries = [
        (
            f"Component {i}",
            COMPONENT_COLORS[i],
        )
        for i in range(6)
    ]

    plotter.add_legend(
        labels=legend_entries,
        loc="upper right",
        bcolor="white",
        border=True,
    )

    # Helpful default camera
    plotter.view_isometric()

    plotter.show()


# ======================================================================
# Main
# ======================================================================

def main():
    surfaces = []
    shell_radii = []

    reference_index = None

    for component in range(6):
        filename = GRID_DIR / GRID_PATTERN.format(
            component=component
        )

        x_local, y_local, z_local = read_grid_component(
            filename,
            component,
        )

        idx, shell_radius = choose_radial_index(
            x_local,
            y_local,
            z_local,
            radial_index=RADIAL_INDEX,
            radius=RADIUS,
        )

        if reference_index is None:
            reference_index = idx

        print(
            f"  selected radial index = {idx}, "
            f"median radius = {shell_radius:.8g}"
        )

        # Original LOCAL shell
        x_shell_local = x_local[idx, :, :]
        y_shell_local = y_local[idx, :, :]
        z_shell_local = z_local[idx, :, :]

        # Transform entire ORIGINAL shell to GLOBAL Cartesian coordinates
        x_global, y_global, z_global = local_to_global_xyz(
            component,
            x_shell_local,
            y_shell_local,
            z_shell_local,
        )

        surface = make_surface_structured_grid(
            x_global,
            y_global,
            z_global,
        )

        surface.field_data["component"] = np.array(
            [component],
            dtype=np.int32,
        )

        surfaces.append(surface)
        shell_radii.append(shell_radius)

    shell_radii = np.asarray(shell_radii)

    print("\nSelected shell radii for six components:")
    for i, r in enumerate(shell_radii):
        print(
            f"  component {i}: {r:.10g}"
        )

    print(
        "\nRadius spread among six components: "
        f"{shell_radii.max() - shell_radii.min():.6e}"
    )

    print(
        "\nThe six surfaces are plotted independently. "
        "Therefore overlap regions are preserved and rendered "
        "as multiple semi-transparent surfaces."
    )

    plot_six_original_surfaces(
        surfaces,
        shell_radii,
    )


if __name__ == "__main__":
    main()
