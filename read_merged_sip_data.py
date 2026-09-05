"""
read_merged_sip_data_202608260058.py

Read the new separated SIP-IFVM merged-data structure:

Static grid file:
    config.GRID_FILE

Time-dependent physical-data file:
    config.LOCAL_MERGED_DIR / "82_00_merged_spherical.h5"

The static grid is saved only once and provides:
    r, theta, phi
    source_component
    nearest_source_distance
    component_shapes

Cartesian coordinates x, y, z are reconstructed from r, theta, phi when
requested, so they do not need to be stored repeatedly.

Each time-dependent physical-data file contains only 8 three-dimensional
physical variables:
    n
    P
    vr, vtheta, vphi
    Br, Btheta, Bphi

All 3-D fields use shape:
    (Nr, Ntheta, Nphi)

theta is colatitude in radians:
    theta = 0      : north pole
    theta = pi/2   : equator
    theta = pi     : south pole

phi is longitude in radians:
    0 <= phi < 2*pi
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt

from config import GRID_FILE, LOCAL_MERGED_DIR


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_FILE = LOCAL_MERGED_DIR / "82_00_merged_spherical.h5"

MERGED_DATA_FILENAME_PATTERN = re.compile(
    r"^(\d+)_([0-9]{2})_merged_spherical\.h5$"
)


def simulation_hours_from_filename(filename: Path) -> float:
    """Return simulation time [h] encoded in a merged-data filename."""
    filename = Path(filename)
    match = MERGED_DATA_FILENAME_PATTERN.fullmatch(filename.name)

    if match is None:
        raise ValueError(
            f'Cannot parse simulation time from "{filename.name}". '
            'Expected "xx_yy_merged_spherical.h5".'
        )

    return int(match.group(1)) + int(match.group(2)) / 100.0


def discover_merged_data_files(data_dir: Path) -> list[Path]:
    """Discover merged HDF5 files and sort them by simulation time."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    files = [
        filename
        for filename in data_dir.iterdir()
        if filename.is_file()
        and MERGED_DATA_FILENAME_PATTERN.fullmatch(filename.name)
    ]
    return sorted(files, key=simulation_hours_from_filename)


def select_merged_data_files(
    data_dir: Path,
    process_mode: str,
    single_filename: str | Path,
) -> list[Path]:
    """Select one merged file or every valid merged file in a directory."""
    data_dir = Path(data_dir)
    mode = str(process_mode).strip().lower()

    if mode == "single":
        single_path = Path(single_filename)
        filename = single_path if single_path.is_absolute() else data_dir / single_path

        if not filename.is_file():
            raise FileNotFoundError(filename)
        if MERGED_DATA_FILENAME_PATTERN.fullmatch(filename.name) is None:
            raise ValueError(
                f'Invalid merged-data filename "{filename.name}". '
                'Expected "xx_yy_merged_spherical.h5".'
            )
        return [filename]

    if mode == "all":
        files = discover_merged_data_files(data_dir)
        if not files:
            raise FileNotFoundError(
                "No files matching 'xx_yy_merged_spherical.h5' were found in:\n"
                f"{data_dir}"
            )
        return files

    raise ValueError(
        f"Unknown PROCESS_MODE={process_mode!r}. Use 'single' or 'all'."
    )


# ----------------------------------------------------------------------
# Generic HDF5 helpers
# ----------------------------------------------------------------------

def _python_scalar(value):
    """Convert a scalar-like HDF5/numpy object to a normal Python scalar."""
    arr = np.asarray(value)

    if arr.size == 1:
        return arr.reshape(-1)[0].item()

    return value


def read_hdf5_attributes(h5obj) -> Dict[str, Any]:
    """Return all attributes as a normal Python dictionary."""
    return {
        key: _python_scalar(value)
        for key, value in h5obj.attrs.items()
    }


def print_hdf5_structure(filename: Path) -> None:
    """Print datasets, shapes, dtypes and root attributes."""
    print(f"\nHDF5 file: {filename}")

    with h5py.File(filename, "r") as f:
        attrs = read_hdf5_attributes(f)

        if attrs:
            print("\nRoot attributes:")

            for key, value in attrs.items():
                print(
                    f"  {key}: {value}"
                )

        print("\nDatasets:")

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"  {name:30s} "
                    f"shape={str(obj.shape):20s} "
                    f"dtype={obj.dtype}"
                )

        f.visititems(visitor)


# ----------------------------------------------------------------------
# Coordinate construction
# ----------------------------------------------------------------------

def spherical_grid_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
):
    """
    Construct global Cartesian coordinates from the 1-D spherical axes.

    Returns
    -------
    x, y, z : ndarray
        Shape (Nr, Ntheta, Nphi)
    """
    rr = r[:, None, None]
    tt = theta[None, :, None]
    pp = phi[None, None, :]

    x = (
        rr
        * np.sin(tt)
        * np.cos(pp)
    )

    y = (
        rr
        * np.sin(tt)
        * np.sin(pp)
    )

    z = (
        rr
        * np.cos(tt)
        * np.ones_like(pp)
    )

    return x, y, z


# ----------------------------------------------------------------------
# Read static merged grid
# ----------------------------------------------------------------------

def read_merged_grid(
    filename: Path = GRID_FILE,
    load_cartesian: bool = True,
    load_selection_maps: bool = True,
) -> Dict[str, Any]:
    """
    Read the static merged spherical-grid file.

    Always contains
    ---------------
    r, theta, phi

    Optionally contains / reconstructs
    ----------------------------------
    x, y, z
    source_component
    nearest_source_distance
    component_shapes

    Notes
    -----
    x, y, z are reconstructed from r, theta, phi rather than read from
    the HDF5 file, because the new grid-saving structure intentionally
    does not store these redundant 3-D arrays.
    """
    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    grid: Dict[str, Any] = {}

    with h5py.File(
        filename,
        "r",
    ) as f:

        for name in (
            "r",
            "theta",
            "phi",
        ):
            if name not in f:
                raise KeyError(
                    f'Missing dataset "{name}" in {filename}'
                )

            grid[name] = (
                f[name][...]
            )

        if load_selection_maps:
            for name in (
                "source_component",
                "nearest_source_distance",
                "component_shapes",
            ):
                if name in f:
                    grid[name] = (
                        f[name][...]
                    )

        grid["attrs"] = (
            read_hdf5_attributes(f)
        )

    if load_cartesian:
        x, y, z = (
            spherical_grid_to_cartesian(
                np.asarray(grid["r"]),
                np.asarray(grid["theta"]),
                np.asarray(grid["phi"]),
            )
        )

        grid["x"] = x
        grid["y"] = y
        grid["z"] = z

    return grid


# ----------------------------------------------------------------------
# Read separated physical data + static grid
# ----------------------------------------------------------------------

def read_merged_physics(
    filename: Path = DATA_FILE,
    grid_filename: Path = GRID_FILE,
    load_cartesian_coordinates: bool = False,
    load_component_map: bool = True,
    field_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Read one time-dependent merged physical-data file and combine it with
    the separately stored static spherical grid.

    Parameters
    ----------
    filename
        Time-dependent physical-data file containing only the 8 physical
        variables.

    grid_filename
        Static merged spherical-grid file.

    load_cartesian_coordinates
        If True, reconstruct and return x, y, z from r, theta, phi.

    load_component_map
        If True and source_component exists in the grid file, include it.

    field_names
        Physical fields to load. None loads all fields. Valid names are
        n, nrho_cm-3, rho_cm-3, P, vr, vtheta, vphi, Br, Btheta and Bphi.

    Returns
    -------
    data : dict

    Coordinates from grid file:
        r, theta, phi

    Number density:
        nrho_cm-3
            number density [cm^-3], read from HDF5 dataset "n"

        rho_cm-3
            backward-compatible alias of the same ndarray

    Physical variables:
        P                       [Pa]

        vr, vtheta, vphi        [km/s]

        Br, Btheta, Bphi        [G]
    """
    if not filename.exists():
        raise FileNotFoundError(
            filename
        )

    # Read static grid first.
    grid = read_merged_grid(
        filename=grid_filename,
        load_cartesian=load_cartesian_coordinates,
        load_selection_maps=load_component_map,
    )

    dataset_map = {
        "n": "nrho_cm-3",
        "P": "P",
        "vr": "vr",
        "vtheta": "vtheta",
        "vphi": "vphi",
        "Br": "Br",
        "Btheta": "Btheta",
        "Bphi": "Bphi",
    }

    if field_names is not None:
        aliases = {
            "nrho_cm-3": "n",
            "rho_cm-3": "n",
        }
        requested_datasets = {
            aliases.get(name, name)
            for name in field_names
        }
        unknown = requested_datasets.difference(dataset_map)
        if unknown:
            raise ValueError(
                f"Unknown field_names: {sorted(unknown)}. "
                f"Valid names: {sorted(dataset_map)}"
            )
        dataset_map = {
            dataset_name: output_name
            for dataset_name, output_name in dataset_map.items()
            if dataset_name in requested_datasets
        }

    data: Dict[str, Any] = {
        "r": grid["r"],
        "theta": grid["theta"],
        "phi": grid["phi"],
    }

    if load_cartesian_coordinates:
        for name in (
            "x",
            "y",
            "z",
        ):
            data[name] = (
                grid[name]
            )

    if (
        load_component_map
        and "source_component" in grid
    ):
        data["source_component"] = (
            grid["source_component"]
        )

    with h5py.File(
        filename,
        "r",
    ) as f:

        units = {}

        for dataset_name, output_name in dataset_map.items():

            if dataset_name not in f:
                raise KeyError(
                    f'Missing dataset "{dataset_name}" in {filename}'
                )

            arr = f[
                dataset_name
            ][...]

            data[
                output_name
            ] = arr

            if (
                "unit"
                in f[dataset_name].attrs
            ):
                units[
                    output_name
                ] = _python_scalar(
                    f[dataset_name]
                    .attrs["unit"]
                )

        data["attrs"] = (
            read_hdf5_attributes(f)
        )

        data["units"] = units

    # Backward-compatible alias. This does not duplicate the array in memory.
    if "nrho_cm-3" in data:
        data["rho_cm-3"] = data["nrho_cm-3"]

    if (
        "nrho_cm-3"
        in data["units"]
    ):
        data["units"][
            "rho_cm-3"
        ] = data["units"][
            "nrho_cm-3"
        ]

    # Validate 3-D shapes against static grid.
    expected_shape = (
        len(data["r"]),
        len(data["theta"]),
        len(data["phi"]),
    )

    for name in (
        "nrho_cm-3",
        "P",
        "vr",
        "vtheta",
        "vphi",
        "Br",
        "Btheta",
        "Bphi",
    ):
        if name in data and data[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {data[name].shape}; "
                f"expected {expected_shape} from grid file "
                f"{grid_filename}"
            )

    return data


# ----------------------------------------------------------------------
# Validation and summaries
# ----------------------------------------------------------------------

def validate_merged_files(
    grid: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """
    Check that grid dimensions match the physical data.
    """
    expected_shape = (
        len(grid["r"]),
        len(grid["theta"]),
        len(grid["phi"]),
    )

    for name in (
        "nrho_cm-3",
        "P",
        "vr",
        "vtheta",
        "vphi",
        "Br",
        "Btheta",
        "Bphi",
    ):
        if data[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {data[name].shape}; "
                f"expected {expected_shape}"
            )

    for coord in (
        "r",
        "theta",
        "phi",
    ):
        if not np.array_equal(
            np.asarray(grid[coord]),
            np.asarray(data[coord]),
        ):
            raise ValueError(
                f"{coord} values differ between "
                "grid and combined data."
            )

    print(
        "\nGrid/data consistency check: PASSED"
    )

    print(
        f"Common field shape = {expected_shape}"
    )


def summarize_component_usage(
    source_component: np.ndarray,
) -> None:
    """
    Print how many target points were assigned to each of the six grids.
    """
    print(
        "\nSource-component usage:"
    )

    total = (
        source_component.size
    )

    for component in range(6):
        n = int(
            np.count_nonzero(
                source_component
                == component
            )
        )

        fraction = (
            100.0
            * n
            / total
        )

        print(
            f"  component {component}: "
            f"{n:10,d} points  "
            f"({fraction:6.2f}%)"
        )


def summarize_fields(
    data: Dict[str, Any],
) -> None:
    """
    Print min/max/mean for the 8 stored physical variables.
    """
    names = (
        "nrho_cm-3",
        "P",
        "vr",
        "vtheta",
        "vphi",
        "Br",
        "Btheta",
        "Bphi",
    )

    print(
        "\nField summary:"
    )

    print(
        f"{'field':12s} "
        f"{'min':>16s} "
        f"{'max':>16s} "
        f"{'mean':>16s} "
        f"{'nan':>10s}"
    )

    for name in names:
        a = np.asarray(
            data[name]
        )

        print(
            f"{name:12s} "
            f"{np.nanmin(a):16.8e} "
            f"{np.nanmax(a):16.8e} "
            f"{np.nanmean(a):16.8e} "
            f"{np.isnan(a).sum():10d}"
        )


# ----------------------------------------------------------------------
# Convenient shell extraction
# ----------------------------------------------------------------------

def nearest_radial_index(
    r: np.ndarray,
    radius: float,
) -> int:
    """
    Return radial index whose r value is closest to radius.
    """
    return int(
        np.argmin(
            np.abs(
                r - radius
            )
        )
    )


def extract_shell(
    data: Dict[str, Any],
    radial_index: int | None = None,
    radius: float | None = None,
) -> Dict[str, Any]:
    """
    Extract one r=constant spherical shell.

    Returns 2-D fields with shape:
        (Ntheta, Nphi)
    """
    r = np.asarray(
        data["r"]
    )

    if radial_index is None:

        if radius is None:
            raise ValueError(
                "Specify radial_index or radius."
            )

        radial_index = (
            nearest_radial_index(
                r,
                radius,
            )
        )

    if radial_index < 0:
        radial_index = (
            len(r)
            + radial_index
        )

    if (
        radial_index < 0
        or radial_index >= len(r)
    ):
        raise IndexError(
            f"radial_index={radial_index} "
            f"outside 0..{len(r)-1}"
        )

    shell: Dict[str, Any] = {
        "radial_index": radial_index,
        "radius": r[radial_index],
        "theta": data["theta"],
        "phi": data["phi"],
    }

    for name in (
        "nrho_cm-3",
        "P",
        "vr",
        "vtheta",
        "vphi",
        "Br",
        "Btheta",
        "Bphi",
    ):
        shell[name] = (
            data[name][
                radial_index,
                :,
                :,
            ]
        )

    shell["rho_cm-3"] = (
        shell["nrho_cm-3"]
    )

    if "source_component" in data:
        shell["source_component"] = (
            data["source_component"][
                radial_index,
                :,
                :,
            ]
        )

    return shell


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    grid = read_merged_grid(
        GRID_FILE,
        load_cartesian=False,
        load_selection_maps=True,
    )

    data = read_merged_physics(
        filename=DATA_FILE,
        grid_filename=GRID_FILE,
        load_cartesian_coordinates=False,
        load_component_map=True,
    )

    validate_merged_files(
        grid,
        data,
    )

    # Optional diagnostics:
    #
    # if "source_component" in grid:
    #     summarize_component_usage(
    #         grid["source_component"]
    #     )
    #
    # summarize_fields(data)
    #
    # shell = extract_shell(
    #     data,
    #     radial_index=0,
    # )
    #
    # print(
    #     shell["Br"].shape
    # )

    Br = data["Br"]
    Br_shell = data["Br"][0]

    plt.figure(
        figsize=(8, 4)
    )


if __name__ == "__main__":
    main()
