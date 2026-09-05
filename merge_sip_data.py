"""
merge_sip_data_202608260058.py

Read the six SIP-IFVM solution components, interpolate them onto the regular
global spherical grid, de-normalize the physical variables, and save both
normalized and physical-unit data into one merged HDF5 file.

The six-component interpolation logic follows the earlier Python version:
    1. A target point is assigned to exactly one source component.
    2. Interpolation uses source points only from that selected component.
    3. Because vecRBFinterp3D was not supplied, kNN inverse-distance weighting
       with k=27 is used.

Input normalized variables from OutputHDF5Parallel:
    Variables(1) = rho_norm
    Variables(2) = vx_local_norm
    Variables(3) = vy_local_norm
    Variables(4) = vz_local_norm
    Variables(5) = P_norm
    Variables(6) = Bx_local_norm
    Variables(7) = By_local_norm
    Variables(8) = Bz_local_norm

After merging, the code converts to physical units using the normalization
constants from the supplied Fortran post-processing code.

Physical output:
    n                  number density, cm^-3
    rho_mass           mass density, kg/m^3
    P                  thermal pressure, Pa

    vr, vtheta, vphi   spherical velocity, km/s

    Br, Btheta, Bphi   spherical magnetic field, G

    Bmag               magnetic-field magnitude, G
    PlasmaBeta         dimensionless

The normalized variables are also saved with suffix "_norm".

Important:
    Temperature T cannot be produced from the current merged HDF5 files,
    because OutputHDF5Parallel saves only 8 variables and does not save the
    original usph(5) temperature variable.

Default input:
    E:/Research/Data/SIP-IFVM/output/
        82_00_0skip1.h5
        ...
        82_00_5skip1.h5

Default output:
    E:/Research/Data/SIP-IFVM/output/
        82_00_merged_spherical.h5
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple
import re

import h5py
import numpy as np
from scipy.spatial import cKDTree

from merge_sip_grid import (
    GRID_DIR,
    GRID_PATTERN,
    HDF5Layout,
    GridComponent,
    build_component_trees,
    canonicalize_with_layout,
    local_vector_to_global,
    read_all_grid_components,
)


# ======================================================================
# Configuration
# ======================================================================

# OUTPUT_DIR = Path(r"E:/Research/Data/SIP-IFVM/solutions")
# MERGED_DIR = Path(r"E:/Research/Data/SIP-IFVM/merged")

OUTPUT_DIR = Path(r"F:/Simulation/SIP-IFVM/solutions/132d1to182")
MERGED_DIR = Path(r"F:/Simulation/SIP-IFVM/merged/132d1to182")

PHYSICS_PATTERN = "{time_tag}_{component}skip1.h5"
MERGED_GRID_FILE = GRID_DIR / "merged_spherical_grid.h5"

K_NEIGHBORS = 27
IDW_POWER = 2.0
INTERP_CHUNK = 100_000


# ======================================================================
# SIP-IFVM normalization constants
# ======================================================================

gamma = 5.0 / 3.0

Rs = 6.963e8                    # m
Ts = 1.3e6                      # K
nrho = 1.5e8                    # cm^-3
mp = 1.672e-27                  # kg

Rhos = nrho * 1.0e6 * mp        # kg/m^3

R11 = 1.653e4

Vs = np.sqrt(gamma * R11 * Ts)  # m/s

mu0 = 4.0e-7 * np.pi

Ps = Rhos * Vs**2               # Pa

Bs = np.sqrt(mu0 * Rhos * Vs**2)  # Tesla

muT2Gs = 10.0                   # Pa -> dyn/cm^2 conversion used by Fortran


# ======================================================================
# Read one / all six physical components
# ======================================================================

def read_physics_component(
    filename: Path,
    layout: HDF5Layout,
) -> Tuple[np.ndarray, float | None, int | None]:
    """
    Read one physical-variable HDF5 file and return canonical shape:
        (8, Nr, Na, Nb)
    """
    with h5py.File(filename, "r") as f:
        if "Variables" not in f:
            raise KeyError(
                f'{filename} does not contain dataset "Variables".'
            )

        dset = f["Variables"]
        raw = dset[...]

        time = None
        nyy = None

        if "simulation_time" in dset.attrs:
            arr = np.asarray(dset.attrs["simulation_time"])
            if arr.size == 1:
                time = float(arr.reshape(-1)[0])

        if "nyy_component" in dset.attrs:
            arr = np.asarray(dset.attrs["nyy_component"])
            if arr.size == 1:
                nyy = int(arr.reshape(-1)[0])

    data = canonicalize_with_layout(
        raw,
        layout,
        expected_channels=8,
    )

    return data, time, nyy


def read_all_physics_components(
    grids: Sequence[GridComponent],
    output_dir: Path = OUTPUT_DIR,
    time_tag: str = "",
    pattern: str = PHYSICS_PATTERN,
) -> Tuple[List[np.ndarray], float | None]:
    """
    Read all six physical-variable component files.
    """
    all_data: List[np.ndarray] = []
    times = []

    for component in range(6):
        filename = output_dir / pattern.format(
            time_tag=time_tag,
            component=component,
        )

        if not filename.exists():
            raise FileNotFoundError(filename)

        data, time, nyy = read_physics_component(
            filename,
            grids[component].layout,
        )

        if data.shape[1:] != grids[component].shape:
            raise ValueError(
                f"Grid/data shape mismatch for component {component}: "
                f"grid={grids[component].shape}, "
                f"physics={data.shape[1:]}"
            )

        print(
            f"[physics {component}] "
            f"file={filename.name}, "
            f"shape={data.shape}, "
            f"attr_nyy={nyy}, "
            f"time={time}"
        )

        all_data.append(data)

        if time is not None:
            times.append(time)

    simulation_time = None

    if times:
        simulation_time = float(np.median(times))

        if not np.allclose(
            times,
            simulation_time,
        ):
            print(
                "WARNING: simulation_time attributes "
                "are not identical:",
                times,
            )

    return all_data, simulation_time


# ======================================================================
# Interpolation
# ======================================================================

def idw_interpolate_knn(
    tree: cKDTree,
    source_values: np.ndarray,
    target_xyz: np.ndarray,
    k: int = K_NEIGHBORS,
    power: float = IDW_POWER,
) -> np.ndarray:
    """
    Interpolate source variables to target points using k-nearest-neighbour
    inverse-distance weighting.

    This is currently the replacement for the unavailable vecRBFinterp3D.
    """
    nsource = source_values.shape[1]

    k_eff = min(
        int(k),
        nsource,
    )

    dist, idx = tree.query(
        target_xyz,
        k=k_eff,
        workers=-1,
    )

    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    eps = np.finfo(np.float64).eps

    exact = dist <= (100.0 * eps)

    weights = (
        1.0
        / np.maximum(
            dist,
            eps,
        ) ** power
    )

    rows_with_exact = np.any(
        exact,
        axis=1,
    )

    if np.any(rows_with_exact):
        weights[rows_with_exact] = (
            exact[rows_with_exact]
            .astype(np.float64)
        )

    weights /= np.sum(
        weights,
        axis=1,
        keepdims=True,
    )

    values = source_values[:, idx]

    interpolated = np.sum(
        values * weights[None, :, :],
        axis=2,
    )

    return interpolated


def cartesian_vector_to_spherical(
    theta: np.ndarray,
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert global Cartesian vector components to spherical components.

    theta is colatitude.
    """
    st = np.sin(theta)
    ct = np.cos(theta)

    sp = np.sin(phi)
    cp = np.cos(phi)

    vr = (
        vx * st * cp
        + vy * st * sp
        + vz * ct
    )

    vtheta = (
        vx * ct * cp
        + vy * ct * sp
        - vz * st
    )

    vphi = (
        -vx * sp
        + vy * cp
    )

    return vr, vtheta, vphi


def resample_to_spherical_grid(
    grids: Sequence[GridComponent],
    physics: Sequence[np.ndarray],
    merged_grid_file: Path = MERGED_GRID_FILE,
):
    """
    Interpolate normalized rho, velocity, P, and B from the six components
    onto the common spherical grid.

    Returned quantities are still normalized.
    """
    with h5py.File(
        merged_grid_file,
        "r",
    ) as f:

        r = f["r"][...]
        theta = f["theta"][...]
        phi = f["phi"][...]

        source_component = (
            f["source_component"][...]
        )

    # The static grid file intentionally does not store x/y/z.
    # Reconstruct the target Cartesian coordinates from r/theta/phi.
    rr = r[:, None, None]
    tt_grid = theta[None, :, None]
    pp_grid = phi[None, None, :]

    x = (
        rr
        * np.sin(tt_grid)
        * np.cos(pp_grid)
    )

    y = (
        rr
        * np.sin(tt_grid)
        * np.sin(pp_grid)
    )

    z = (
        rr
        * np.cos(tt_grid)
        * np.ones_like(pp_grid)
    )

    shape = x.shape
    n_target = x.size

    target_xyz = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
            z.ravel(),
        )
    )

    selected_component = (
        source_component.ravel()
    )

    trees = build_component_trees(
        grids
    )

    # channel order:
    # 0 rho
    # 1 vx
    # 2 vy
    # 3 vz
    # 4 P
    # 5 Bx
    # 6 By
    # 7 Bz
    merged = np.full(
        (8, n_target),
        np.nan,
        dtype=np.float64,
    )

    for component in range(6):
        target_idx = np.flatnonzero(
            selected_component == component
        )

        if target_idx.size == 0:
            continue

        print(
            f"\nInterpolating component {component}: "
            f"{target_idx.size:,} target points"
        )

        src = physics[component].reshape(
            8,
            -1,
        )

        for start in range(
            0,
            target_idx.size,
            INTERP_CHUNK,
        ):
            stop = min(
                start + INTERP_CHUNK,
                target_idx.size,
            )

            out_idx = target_idx[
                start:stop
            ]

            pts = target_xyz[
                out_idx
            ]

            local_interp = idw_interpolate_knn(
                trees[component],
                src,
                pts,
                k=K_NEIGHBORS,
                power=IDW_POWER,
            )

            # Scalars
            merged[0, out_idx] = (
                local_interp[0]
            )

            merged[4, out_idx] = (
                local_interp[4]
            )

            # Velocity:
            # component-local Cartesian
            # -> global Cartesian
            vx_g, vy_g, vz_g = (
                local_vector_to_global(
                    component,
                    local_interp[1],
                    local_interp[2],
                    local_interp[3],
                )
            )

            merged[1, out_idx] = vx_g
            merged[2, out_idx] = vy_g
            merged[3, out_idx] = vz_g

            # Magnetic field:
            # component-local Cartesian
            # -> global Cartesian
            bx_g, by_g, bz_g = (
                local_vector_to_global(
                    component,
                    local_interp[5],
                    local_interp[6],
                    local_interp[7],
                )
            )

            merged[5, out_idx] = bx_g
            merged[6, out_idx] = by_g
            merged[7, out_idx] = bz_g

            print(
                f"  component {component}: "
                f"{stop:,}/{target_idx.size:,}"
            )

    if np.isnan(merged).any():
        raise RuntimeError(
            "NaNs remain after interpolation. "
            "Check component selection and input files."
        )

    merged = merged.reshape(
        (8,) + shape
    )

    rho = merged[0]

    vx = merged[1]
    vy = merged[2]
    vz = merged[3]

    pressure = merged[4]

    bx = merged[5]
    by = merged[6]
    bz = merged[7]

    tt = np.broadcast_to(
        theta[None, :, None],
        shape,
    )

    pp = np.broadcast_to(
        phi[None, None, :],
        shape,
    )

    vr, vtheta, vphi = (
        cartesian_vector_to_spherical(
            tt,
            pp,
            vx,
            vy,
            vz,
        )
    )

    br, btheta, bphi = (
        cartesian_vector_to_spherical(
            tt,
            pp,
            bx,
            by,
            bz,
        )
    )

    return {
        "r": r,
        "theta": theta,
        "phi": phi,

        "x": x,
        "y": y,
        "z": z,

        "source_component": source_component,

        "rho": rho,
        "P": pressure,

        "vr": vr,
        "vtheta": vtheta,
        "vphi": vphi,

        "Br": br,
        "Btheta": btheta,
        "Bphi": bphi,
    }


# ======================================================================
# De-normalization
# ======================================================================

def denormalize_merged_data(
    data_norm: dict,
):
    """
    Convert normalized merged quantities to physical units.

    The conversion follows the supplied Fortran formulas.

    Returns
    -------
    data_phys : dict

        n:
            cm^-3

        rho_mass:
            kg/m^3

        P:
            Pa

        velocities:
            km/s

        magnetic fields:
            G

        Bmag:
            G

        PlasmaBeta:
            dimensionless
    """
    data_phys = {}

    # --------------------------------------------------------------
    # Coordinates
    #
    # Keep original normalized coordinates (typically solar radii)
    # and also provide meter coordinates.
    # --------------------------------------------------------------
    for name in (
        "r",
        "theta",
        "phi",
        "x",
        "y",
        "z",
        "source_component",
    ):
        data_phys[name] = data_norm[name]

    data_phys["r_m"] = (
        data_norm["r"] * Rs
    )

    data_phys["x_m"] = (
        data_norm["x"] * Rs
    )

    data_phys["y_m"] = (
        data_norm["y"] * Rs
    )

    data_phys["z_m"] = (
        data_norm["z"] * Rs
    )

    # --------------------------------------------------------------
    # Density
    #
    # Fortran:
    # rho = usph(1) * (Rhos / 1.672E-27) / 1.E6
    #
    # Here we call it n because it is number density.
    # --------------------------------------------------------------
    rho_norm = data_norm["rho"]

    n = (
        rho_norm
        * (Rhos / mp)
        / 1.0e6
    )

    rho_mass = (
        rho_norm
        * Rhos
    )

    data_phys["n"] = n
    data_phys["rho_mass"] = rho_mass

    # --------------------------------------------------------------
    # Velocity
    #
    # Fortran:
    # v = v_norm * Vs / 1000
    # --------------------------------------------------------------
    velocity_scale = (
        Vs / 1000.0
    )

    for name in (
        "vr",
        "vtheta",
        "vphi",
    ):
        data_phys[name] = (
            data_norm[name]
            * velocity_scale
        )

    # --------------------------------------------------------------
    # Pressure
    #
    # Fortran:
    # p_ = usph(11) * Ps
    # --------------------------------------------------------------
    data_phys["P"] = (
        data_norm["P"]
        * Ps
    )

    # --------------------------------------------------------------
    # Magnetic field
    #
    # Fortran:
    # B = B_norm * Bs * 1.E4
    # --------------------------------------------------------------
    magnetic_scale = (
        Bs * 1.0e4
    )

    for name in (
        "Br",
        "Btheta",
        "Bphi",
    ):
        data_phys[name] = (
            data_norm[name]
            * magnetic_scale
        )

    # --------------------------------------------------------------
    # Magnetic-field magnitude
    #
    # Fortran:
    # BEXP2 = Br**2 + Bt**2 + Bp**2
    # dabsB = sqrt(BEXP2)
    # --------------------------------------------------------------
    BEXP2 = (
        data_phys["Br"]**2
        + data_phys["Btheta"]**2
        + data_phys["Bphi"]**2
    )

    data_phys["Bmag"] = (
        np.sqrt(BEXP2)
    )

    # --------------------------------------------------------------
    # Plasma beta
    #
    # Fortran:
    # PlasmaBeta = muT2Gs * p_ * 8*pi / BEXP2
    #
    # p_ is in Pa.
    # muT2Gs = 10 converts Pa -> dyn/cm^2.
    # B is in Gauss.
    # --------------------------------------------------------------
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        data_phys["PlasmaBeta"] = (
            muT2Gs
            * data_phys["P"]
            * 8.0
            * np.pi
            / BEXP2
        )

    return data_phys


# ======================================================================
# Save
# ======================================================================

def save_merged_physics(
    output_file: Path,
    data_norm: dict,
    data_phys: dict,
    simulation_time: float | None,
    time_tag: str,
):
    """
    Save ONLY the 8 time-dependent 3-D physical variables.

    The static grid is stored once in merged_spherical_grid.h5.

    Saved variables
    ---------------
    n
        number density [cm^-3]

    P
        thermal pressure [Pa]

    vr, vtheta, vphi
        spherical velocity components [km/s]

    Br, Btheta, Bphi
        spherical magnetic-field components [G]

    Not saved
    ---------
    r, theta, phi
    x, y, z
    x_m, y_m, z_m
    source_component
    normalized/*
    rho_mass
    Bmag
    PlasmaBeta

    These are either static, redundant, or directly derivable.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        output_file,
        "w",
    ) as f:

        f.attrs["description"] = (
            "SIP-IFVM time-dependent merged physical variables. "
            "Static spherical-grid information is stored separately in "
            "merged_spherical_grid.h5."
        )

        f.attrs["time_tag"] = (
            time_tag
        )

        f.attrs["interpolation"] = (
            f"nearest-component selection + kNN IDW, "
            f"k={K_NEIGHBORS}, power={IDW_POWER}"
        )

        f.attrs["grid_file"] = (
            "merged_spherical_grid.h5"
        )

        f.attrs["theta_definition"] = (
            "colatitude, radians"
        )

        f.attrs["phi_definition"] = (
            "longitude, radians"
        )

        # Keep normalization scales as tiny metadata.
        f.attrs["gamma"] = gamma
        f.attrs["Rs_m"] = Rs
        f.attrs["Ts_K"] = Ts
        f.attrs["nrho_cm-3"] = nrho
        f.attrs["Vs_m_s"] = Vs
        f.attrs["Ps_Pa"] = Ps
        f.attrs["Bs_T"] = Bs
        f.attrs["Bs_G"] = Bs * 1.0e4

        f.attrs["divergence_free_note"] = (
            "The original vecRBFinterp3D implementation was not supplied. "
            "Current IDW interpolation does not enforce div(B)=0."
        )

        if simulation_time is not None:
            f.attrs["simulation_time"] = (
                simulation_time
            )

        physical_units = {
            "n": "cm^-3",
            "P": "Pa",
            "vr": "km/s",
            "vtheta": "km/s",
            "vphi": "km/s",
            "Br": "G",
            "Btheta": "G",
            "Bphi": "G",
        }

        for name, unit in physical_units.items():

            dset = f.create_dataset(
                name,
                data=data_phys[name],
                compression="gzip",
                compression_opts=4,
            )

            dset.attrs["unit"] = (
                unit
            )

    print(
        "\nSaved merged physical data:"
    )

    print(
        output_file
    )


# ======================================================================
# Batch TIME_TAG discovery
# ======================================================================

def discover_time_tags(
    output_dir: Path = OUTPUT_DIR,
) -> list[str]:
    """
    Find all files matching xx_yy_0skip1.h5 and extract xx_yy as TIME_TAG.
    """
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    pattern = re.compile(
        r"^(\d+)_([0-9]{2})_0skip1\.h5$"
    )

    tags = []

    for filename in output_dir.iterdir():
        if not filename.is_file():
            continue

        match = pattern.match(filename.name)
        if match is None:
            continue

        tags.append(
            f"{match.group(1)}_{match.group(2)}"
        )

    return sorted(
        set(tags),
        key=lambda tag: (
            int(tag.split("_")[0]),
            int(tag.split("_")[1]),
        ),
    )


def component_files_for_time_tag(
    time_tag: str,
    output_dir: Path = OUTPUT_DIR,
    pattern: str = PHYSICS_PATTERN,
) -> list[Path]:
    """
    Return the six expected component files for one TIME_TAG.
    """
    return [
        output_dir / pattern.format(
            time_tag=time_tag,
            component=component,
        )
        for component in range(6)
    ]


def check_complete_time_tag(
    time_tag: str,
    output_dir: Path = OUTPUT_DIR,
    pattern: str = PHYSICS_PATTERN,
) -> tuple[bool, list[Path]]:
    """
    Check whether components 0..5 all exist for one TIME_TAG.
    """
    expected = component_files_for_time_tag(
        time_tag,
        output_dir=output_dir,
        pattern=pattern,
    )

    missing = [
        filename
        for filename in expected
        if not filename.exists()
    ]

    return len(missing) == 0, missing


# ======================================================================
# Main
# ======================================================================

def main():

    print("\nNormalization scales:")
    print(f"  Rhos = {Rhos:.8e} kg/m^3")
    print(f"  Vs   = {Vs/1000.0:.8e} km/s")
    print(f"  Ps   = {Ps:.8e} Pa")
    print(f"  Bs   = {Bs*1.0e4:.8e} G")

    if not MERGED_GRID_FILE.exists():
        raise FileNotFoundError(
            f"{MERGED_GRID_FILE}\n"
            "Run merge_sip_grid_202608260058.py first."
        )

    MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Grid is identical for all output times, so read it only once.
    grids = read_all_grid_components(
        grid_dir=GRID_DIR,
        pattern=GRID_PATTERN,
    )

    # Discover all TIME_TAG values from xx_yy_0skip1.h5.
    time_tags = discover_time_tags(
        OUTPUT_DIR
    )

    if not time_tags:
        raise FileNotFoundError(
            "No files matching xx_yy_0skip1.h5 were found in:\n"
            f"{OUTPUT_DIR}"
        )

    print("\nDiscovered TIME_TAG values:")
    for time_tag in time_tags:
        print(f"  {time_tag}")

    print(f"\nTotal TIME_TAG count: {len(time_tags)}")

    completed = []
    skipped = []

    for index, time_tag in enumerate(
        time_tags,
        start=1,
    ):

        print("\n" + "=" * 72)
        print(
            f"Processing TIME_TAG {time_tag} "
            f"({index}/{len(time_tags)})"
        )
        print("=" * 72)

        output_file = (
            MERGED_DIR
            / f"{time_tag}_merged_spherical.h5"
        )

        if output_file.exists():
            print(
                f"Output file already exists, skipping TIME_TAG {time_tag}:"
            )
            print(
                f"  {output_file}"
            )

            skipped.append(time_tag)
            continue

        complete, missing = check_complete_time_tag(
            time_tag,
            output_dir=OUTPUT_DIR,
            pattern=PHYSICS_PATTERN,
        )

        if not complete:
            print(
                f"WARNING: skipping TIME_TAG {time_tag} "
                "because one or more component files are missing."
            )
            for filename in missing:
                print(f"  missing: {filename}")

            skipped.append(time_tag)
            continue

        physics, simulation_time = (
            read_all_physics_components(
                grids,
                output_dir=OUTPUT_DIR,
                time_tag=time_tag,
                pattern=PHYSICS_PATTERN,
            )
        )

        merged_norm = (
            resample_to_spherical_grid(
                grids,
                physics,
                merged_grid_file=MERGED_GRID_FILE,
            )
        )

        merged_phys = (
            denormalize_merged_data(
                merged_norm
            )
        )

        save_merged_physics(
            output_file,
            merged_norm,
            merged_phys,
            simulation_time,
            time_tag=time_tag,
        )

        completed.append(time_tag)

    print("\n" + "=" * 72)
    print("Batch merge finished.")
    print(f"Completed: {len(completed)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Output directory:\n{MERGED_DIR}")

    if skipped:
        print("\nSkipped TIME_TAG values:")
        for time_tag in skipped:
            print(f"  {time_tag}")


if __name__ == "__main__":
    main()
