"""
plot_merged_sip_slices_202608260058.py

Plot one merged SIP-IFVM physical variable in three panels:

1. Selected spherical shell: theta-phi map
2. Equatorial plane: theta = 90 deg, shown in x-y
3. 0-degree meridional plane: phi = 0 deg and 180 deg, shown in x-z

Features
--------
- The spherical shell can be selected by radial index or by radial value.
- The three panels can use independent colorbar ranges:
      shell_clim
      equator_clim
      meridian_clim
- Br defaults to a symmetric linear color scale.
- Density variables default to logarithmic color scale.
  Recognized density names:
      "rho", "n", "density", "number_density"
- Other variables default to linear color scale.
- Each subplot has its own horizontal colorbar below it.
- Cartesian panels use equal aspect ratio.
- No LaTeX $...$ formatting is used.

Notes for log-scale density
---------------------------
LogNorm requires strictly positive values.

If a density clim is given, both limits must be > 0.
If clim is None, the automatic lower/upper limits are determined only
from finite positive values in that slice.

Example
-------
from read_merged_data import read_merged_physics
from plot_merged_slices_2 import plot_merged_slices

data = read_merged_physics()

plot_merged_slices(
    data,
    variable="nrho_cm-3",
    radial_index=0,
    shell_clim=(0.8, 1.0),
    equator_clim=(1e-6, 1e-3),
    meridian_clim=(1e-6, 1e-3),
)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


DENSITY_NAMES = {
    "nrho_cm-3",
    "rho_cm-3",
}


# ======================================================================
# CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# Data-processing mode
# ----------------------------------------------------------------------
#
# "single":
#     Process one specified HDF5 file inside DATA_DIR.
#
# "all":
#     Traverse DATA_DIR and process all files matching:
#         xx_yy_merged_spherical.h5
#
PROCESS_MODE = "all"  # "single" or "all"

DATA_DIR = Path(
    r"E:/Research/Data/SIP-IFVM/merged/"
)
# DATA_DIR = Path(
#     r"F:/Simulation/SIP-IFVM/merged/132d1to182/"
# )

# Used only when PROCESS_MODE = "single".
SINGLE_DATA_FILE = "82_10_merged_spherical.h5"

GRID_FILE = Path(
    r"E:/Research/Data/SIP-IFVM/grid/merged_spherical_grid.h5"
)

SAVE_DIR = Path(
    r"E:/Research/Work/Coronal_hole_by_SIP/slices/"
)
# SAVE_DIR = Path(
#     r"F:/Simulation/SIP-IFVM/slices/132d1to182/"
# )

SAVE_OR_NOT = 1

# ----------------------------------------------------------------------
# Simulation time reference
# ----------------------------------------------------------------------

# 0.00 h  -> 2026-04-08 16:00
# 82.00 h -> 2026-04-12 02:00
SIMULATION_START_DATETIME = datetime(
    2026, 4, 8, 16, 0
)

# ----------------------------------------------------------------------
# Spherical-shell radial selection
# ----------------------------------------------------------------------
#
# "index":
#     use RADIAL_INDEX directly.
#
# "value":
#     use the grid layer whose r is nearest to RADIAL_VALUE.
#
RADIAL_MODE = "index"

RADIAL_INDEX = 0

RADIAL_VALUE = 10.0

# ----------------------------------------------------------------------
# Common plotting parameters
# ----------------------------------------------------------------------

FIGSIZE = (15.5, 5.2)

FIGURE_DPI = 300

# ----------------------------------------------------------------------
# Br
# ----------------------------------------------------------------------

PLOT_BR = True

BR_CMAP = "seismic"

BR_SHELL_CLIM = (-50, 50) # 1.01 Rs
# BR_SHELL_CLIM = (-0.01, 0.01) # 10 Rs
BR_EQUATOR_CLIM = (-0.003, 0.003)
BR_MERIDIAN_CLIM = (-0.003, 0.003)

BR_COLORBAR_LABEL = "Br [G]"

# ----------------------------------------------------------------------
# Radial velocity
# ----------------------------------------------------------------------

PLOT_VR = False

VR_CMAP = "jet"

# VR_SHELL_CLIM = (-40, 40) # 1.01 Rs
VR_SHELL_CLIM = (150, 550) # 10 Rs
VR_EQUATOR_CLIM = (0.0, 800)
VR_MERIDIAN_CLIM = (0.0, 800)

VR_COLORBAR_LABEL = "Vr [km/s]"

# ----------------------------------------------------------------------
# Tangential velocity magnitude
# ----------------------------------------------------------------------

PLOT_VH = True

VH_CMAP = "jet"

# Vh = sqrt(Vtheta^2 + Vphi^2)
# VH_SHELL_CLIM = (0.0, 10.0) # 1.01 Rs
VH_SHELL_CLIM = (0.0, 30.0) # 10 Rs
VH_EQUATOR_CLIM = (0.0, 20.0)
VH_MERIDIAN_CLIM = (0.0, 20.0)

VH_COLORBAR_LABEL = "Vh [km/s]"

# ----------------------------------------------------------------------
# Number density
# ----------------------------------------------------------------------

PLOT_DENSITY = False

DENSITY_CMAP = "jet"

DENSITY_SHELL_CLIM = (1.0e8, 2.0e8)
DENSITY_EQUATOR_CLIM = (1.0e2, 1.0e5)
DENSITY_MERIDIAN_CLIM = (1.0e2, 1.0e5)

DENSITY_COLORBAR_LABEL = "n [cm^-3]"


# ======================================================================
# Data-file discovery
# ======================================================================

def discover_merged_data_files(
    data_dir: Path = DATA_DIR,
) -> list[Path]:
    """
    Find all merged SIP-IFVM HDF5 files in the specified folder.

    Expected filename format:
        xx_yy_merged_spherical.h5

    Files are sorted numerically by simulation time.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    pattern = re.compile(
        r"^(\d+)_([0-9]{2})_merged_spherical\.h5$"
    )

    matched_files = []

    for filename in data_dir.iterdir():
        if not filename.is_file():
            continue

        match = pattern.match(filename.name)

        if match is None:
            continue

        matched_files.append(
            (
                int(match.group(1)),
                int(match.group(2)),
                filename,
            )
        )

    matched_files.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return [
        item[2]
        for item in matched_files
    ]


def select_data_files(
    data_dir=DATA_DIR,
    process_mode=PROCESS_MODE,
    single_filename=SINGLE_DATA_FILE,
):
    """
    Select files according to PROCESS_MODE.

    PROCESS_MODE = "single"
        Process DATA_DIR / SINGLE_DATA_FILE only.

    PROCESS_MODE = "all"
        Process all matching merged HDF5 files in DATA_DIR.
    """
    mode = str(
        process_mode
    ).strip().lower()

    data_dir = Path(
        data_dir
    )

    if mode == "single":
        filename = (
            data_dir
            / single_filename
        )

        if not filename.exists():
            raise FileNotFoundError(
                filename
            )

        return [
            filename
        ]

    if mode == "all":
        files = discover_merged_data_files(
            data_dir
        )

        if not files:
            raise FileNotFoundError(
                "No files matching "
                "'xx_yy_merged_spherical.h5' "
                f"were found in:\n{data_dir}"
            )

        return files

    raise ValueError(
        f"Unknown PROCESS_MODE={process_mode!r}. "
        'Use "single" or "all".'
    )


# ======================================================================
# Filename time conversion
# ======================================================================

def simulation_hours_from_filename(filename: Path) -> float:
    """
    Extract simulation time [hour] from a merged-data filename.

    Examples
    --------
    82_00_merged_spherical.h5  -> 82.00 h
    82_50_merged_spherical.h5  -> 82.50 h
    103_25_merged_spherical.h5 -> 103.25 h
    """
    filename = Path(filename)

    match = re.match(
        r"^(\d+)_([0-9]{2})_merged_spherical.h5$",
        filename.name,
    )

    if match is None:
        raise ValueError(
            f'Cannot parse simulation time from "{filename.name}". '
            'Expected a filename like "82_00_merged_spherical.h5".'
        )

    hours_integer = int(match.group(1))
    hours_fraction = int(match.group(2)) / 100.0

    return hours_integer + hours_fraction


def simulation_datetime_from_filename(
    filename: Path,
    start_datetime: datetime = SIMULATION_START_DATETIME,
) -> datetime:
    """
    Convert the simulation time encoded in the filename to calendar datetime.

    Current reference:
        0.00 h  = 2026-04-08 16:00
        82.00 h = 2026-04-12 02:00
    """
    simulation_hours = simulation_hours_from_filename(
        filename
    )

    return (
        start_datetime
        + timedelta(hours=simulation_hours)
    )


def format_simulation_datetime(filename: Path) -> str:
    """
    Return figure annotation text.

    Example
    -------
    82_00_merged_spherical.h5
        -> 2026-04-12 02:00  (82.00 h)
    """
    simulation_hours = simulation_hours_from_filename(
        filename
    )

    simulation_datetime = simulation_datetime_from_filename(
        filename
    )

    return (
        f"{simulation_datetime:%Y-%m-%d %H:%M}"
        f"  ({simulation_hours:.2f} h)"
    )


def _resolve_radial_index(
    r,
    radial_mode="index",
    radial_index=0,
    radial_value=None,
):
    """
    Resolve the radial layer used for the spherical-shell panel.

    radial_mode="index":
        use radial_index directly.

    radial_mode="value":
        use the grid layer whose r value is nearest to radial_value.
    """
    r = np.asarray(r, dtype=float)
    mode = radial_mode.lower()

    if mode == "index":
        idx = int(radial_index)

        if idx < 0:
            idx = len(r) + idx

        if idx < 0 or idx >= len(r):
            raise IndexError(
                f"radial_index={radial_index} is outside "
                f"the valid range 0 to {len(r)-1}."
            )

        return idx, float(r[idx])

    if mode == "value":
        if radial_value is None:
            raise ValueError(
                'radial_value must be provided when radial_mode="value".'
            )

        target_r = float(radial_value)

        if not np.isfinite(target_r):
            raise ValueError(
                "radial_value must be finite."
            )

        idx = int(
            np.argmin(
                np.abs(r - target_r)
            )
        )

        return idx, float(r[idx])

    raise ValueError(
        f"Unknown radial_mode={radial_mode!r}. "
        'Use "index" or "value".'
    )


def _nearest_phi_index(phi, target):
    """
    Return the index of phi nearest to target, accounting for 2*pi periodicity.
    """
    delta = np.angle(np.exp(1j * (phi - target)))
    return int(np.argmin(np.abs(delta)))


def _is_density_variable(variable):
    """
    Return True when the variable should use log color scale by default.
    """
    return variable.lower() in DENSITY_NAMES


def _get_slice_limits(
    values,
    variable,
    clim=None,
    symmetric=None,
    log_scale=False,
):
    """
    Determine color limits for one plotted slice.

    Parameters
    ----------
    values : ndarray
        Data in the current slice.

    variable : str
        Variable name.

    clim : tuple or None
        (vmin, vmax) for this panel.
        If None, determine limits automatically.

    symmetric : bool or None
        If None:
            Br -> symmetric
            other variables -> not symmetric

    log_scale : bool
        If True, limits must be positive and automatic limits are based
        only on positive finite values.
    """
    values = np.asarray(values)

    if log_scale:
        finite = values[np.isfinite(values) & (values > 0.0)]
    else:
        finite = values[np.isfinite(values)]

    if finite.size == 0:
        if log_scale:
            raise ValueError(
                f'No positive finite values found for log-scale variable "{variable}".'
            )
        raise ValueError(
            f'No finite values found for variable "{variable}" in this slice.'
        )

    if clim is not None:
        if len(clim) != 2:
            raise ValueError(
                "clim must be None or a tuple/list of (vmin, vmax)."
            )

        vmin, vmax = float(clim[0]), float(clim[1])

        if vmin >= vmax:
            raise ValueError(
                f"Invalid clim={clim}: vmin must be smaller than vmax."
            )

        if log_scale and (vmin <= 0.0 or vmax <= 0.0):
            raise ValueError(
                f"For logarithmic color scale, clim must be strictly positive. "
                f"Got clim={clim}."
            )

        return vmin, vmax

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))

    if log_scale:
        return data_min, data_max

    if symmetric is None:
        symmetric = variable.lower() in ("br", "b_r")

    if symmetric:
        absmax = max(abs(data_min), abs(data_max))
        return -absmax, absmax

    return data_min, data_max


def _make_norm(
    values,
    variable,
    clim=None,
    symmetric=None,
    log_scale=False,
):
    """
    Build Normalize or LogNorm for one panel.
    """
    vmin, vmax = _get_slice_limits(
        values,
        variable=variable,
        clim=clim,
        symmetric=symmetric,
        log_scale=log_scale,
    )

    if log_scale:
        return LogNorm(vmin=vmin, vmax=vmax)

    return Normalize(vmin=vmin, vmax=vmax)


def _mask_nonpositive_for_log(values):
    """
    Mask zero and negative values so pcolormesh does not pass invalid
    values into LogNorm.
    """
    values = np.asarray(values)
    return np.ma.masked_less_equal(values, 0.0)


def _add_horizontal_colorbar(
    fig,
    ax,
    mappable,
    label,
    size="5%",
    pad=0.45,
):
    """
    Add one horizontal colorbar below an axis.
    """
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(
        "bottom",
        size=size,
        pad=pad,
    )

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_label(label)

    return cbar


def plot_merged_slices(
    data,
    variable="Br",
    radial_mode="index",
    radial_index=0,
    radial_value=None,
    cmap="jet",
    shell_clim=None,
    equator_clim=None,
    meridian_clim=None,
    symmetric=None,
    log_scale=None,
    figsize=(15.5, 5.2),
    colorbar_label=None,
    title=None,
    datetime_label=None,
):
    """
    Plot one merged SIP-IFVM variable in three views.

    Parameters
    ----------
    data : dict
        Dictionary returned by read_merged_physics().

    variable : str
        Variable name, for example:
            "Br"
            "rho"
            "n"
            "vr"
            "P"
            "Btheta"

    radial_mode : {"index", "value"}
        Selection method for the first theta-phi panel.

        "index":
            choose the shell using radial_index.

        "value":
            choose the grid shell nearest to radial_value.

    radial_index : int
        Used when radial_mode="index".

    radial_value : float or None
        Used when radial_mode="value".
        The nearest available radial grid layer is used.

    cmap : str
        Matplotlib colormap.
        Default: "jet"

    shell_clim : tuple or None
        Colorbar range for spherical-shell panel.

    equator_clim : tuple or None
        Colorbar range for equatorial-plane panel.

    meridian_clim : tuple or None
        Colorbar range for meridional-plane panel.

    symmetric : bool or None
        Used only for linear scale when clim is None.
        If None:
            Br -> symmetric around zero
            others -> slice min/max

    log_scale : bool or None
        If None:
            density variables use log scale
            other variables use linear scale

        Set explicitly to True or False to override.

    figsize : tuple
        Figure size.

    colorbar_label : str or None
        Colorbar label.
        Default: variable name.

    title : str or None
        Optional overall figure title.

    datetime_label : str or None
        Optional simulation date/time annotation. If title is also given,
        the datetime is shown on the line below the title.

    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, axes
        axes = (ax_shell, ax_equator, ax_meridian)
    """

    required = ("r", "theta", "phi", variable)

    for key in required:
        if key not in data:
            raise KeyError(
                f'Missing "{key}". '
                f"Available keys: {list(data.keys())}"
            )

    r = np.asarray(data["r"], dtype=float)
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    field = np.asarray(data[variable], dtype=float)

    expected_shape = (
        len(r),
        len(theta),
        len(phi),
    )

    if field.shape != expected_shape:
        raise ValueError(
            f"{variable}.shape = {field.shape}, "
            f"expected {expected_shape}"
        )

    radial_index, radial_value_resolved = _resolve_radial_index(
        r,
        radial_mode=radial_mode,
        radial_index=radial_index,
        radial_value=radial_value,
    )

    if colorbar_label is None:
        colorbar_label = variable

    if log_scale is None:
        log_scale = _is_density_variable(variable)

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(2.0, 1.0, 1.0),
        wspace=0.32,
    )

    ax_shell = fig.add_subplot(gs[0, 0])
    ax_equator = fig.add_subplot(gs[0, 1])
    ax_meridian = fig.add_subplot(gs[0, 2])

    # ============================================================
    # 1. Selected spherical shell
    # ============================================================
    shell = field[radial_index, :, :]

    norm_shell = _make_norm(
        shell,
        variable=variable,
        clim=shell_clim,
        symmetric=symmetric,
        log_scale=log_scale,
    )

    phi_wrap = np.concatenate(
        (
            phi,
            [phi[0] + 2.0 * np.pi],
        )
    )

    shell_wrap = np.concatenate(
        (
            shell,
            shell[:, :1],
        ),
        axis=1,
    )

    if log_scale:
        shell_wrap = _mask_nonpositive_for_log(shell_wrap)

    im_shell = ax_shell.pcolormesh(
        np.degrees(phi_wrap),
        np.degrees(theta),
        shell_wrap,
        shading="auto",
        cmap=cmap,
        norm=norm_shell,
    )

    ax_shell.set_xlabel("phi [deg]")
    ax_shell.set_ylabel("theta [deg]")

    ax_shell.set_title(
        f"Spherical shell: "
        f"index = {radial_index}, r = {radial_value_resolved:.4g} Rs"
    )

    ax_shell.set_xlim(0.0, 360.0)
    ax_shell.set_ylim(180.0, 0.0)

    ax_shell.set_xticks(
        np.arange(0.0, 361.0, 60.0)
    )

    ax_shell.set_yticks(
        np.arange(0.0, 181.0, 30.0)
    )

    ax_shell.set_aspect(
        "equal",
        adjustable="box",
    )

    _add_horizontal_colorbar(
        fig,
        ax_shell,
        im_shell,
        label=colorbar_label,
    )

    # ============================================================
    # 2. Equatorial plane
    # ============================================================
    j_eq = int(
        np.argmin(
            np.abs(theta - np.pi / 2.0)
        )
    )

    theta_eq = theta[j_eq]

    equatorial = field[:, j_eq, :]

    norm_eq = _make_norm(
        equatorial,
        variable=variable,
        clim=equator_clim,
        symmetric=symmetric,
        log_scale=log_scale,
    )

    equatorial_wrap = np.concatenate(
        (
            equatorial,
            equatorial[:, :1],
        ),
        axis=1,
    )

    if log_scale:
        equatorial_wrap = _mask_nonpositive_for_log(equatorial_wrap)

    rr_eq, pp_eq = np.meshgrid(
        r,
        phi_wrap,
        indexing="ij",
    )

    x_eq = rr_eq * np.cos(pp_eq)
    y_eq = rr_eq * np.sin(pp_eq)

    im_eq = ax_equator.pcolormesh(
        x_eq,
        y_eq,
        equatorial_wrap,
        shading="auto",
        cmap=cmap,
        norm=norm_eq,
    )

    ax_equator.set_xlabel("x")
    ax_equator.set_ylabel("y")

    ax_equator.set_title(
        "Equatorial plane: "
        f"theta = {np.degrees(theta_eq):.2f} deg"
    )

    ax_equator.set_aspect(
        "equal",
        adjustable="box",
    )

    _add_horizontal_colorbar(
        fig,
        ax_equator,
        im_eq,
        label=colorbar_label,
    )

    # ============================================================
    # 3. 0-degree meridional plane
    # ============================================================
    k_phi0 = _nearest_phi_index(
        phi,
        0.0,
    )

    k_phipi = _nearest_phi_index(
        phi,
        np.pi,
    )

    meridian_pos = field[:, :, k_phi0]
    meridian_neg = field[:, :, k_phipi]

    meridian_all = np.concatenate(
        (
            meridian_pos.ravel(),
            meridian_neg.ravel(),
        )
    )

    norm_mer = _make_norm(
        meridian_all,
        variable=variable,
        clim=meridian_clim,
        symmetric=symmetric,
        log_scale=log_scale,
    )

    rr_mer, tt_mer = np.meshgrid(
        r,
        theta,
        indexing="ij",
    )

    x_pos = rr_mer * np.sin(tt_mer)
    x_neg = -rr_mer * np.sin(tt_mer)
    z_mer = rr_mer * np.cos(tt_mer)

    if log_scale:
        meridian_pos_plot = _mask_nonpositive_for_log(meridian_pos)
        meridian_neg_plot = _mask_nonpositive_for_log(meridian_neg)
    else:
        meridian_pos_plot = meridian_pos
        meridian_neg_plot = meridian_neg

    im_mer = ax_meridian.pcolormesh(
        x_pos,
        z_mer,
        meridian_pos_plot,
        shading="auto",
        cmap=cmap,
        norm=norm_mer,
    )

    ax_meridian.pcolormesh(
        x_neg,
        z_mer,
        meridian_neg_plot,
        shading="auto",
        cmap=cmap,
        norm=norm_mer,
    )

    phi0_deg = np.degrees(
        phi[k_phi0]
    ) % 360.0

    phipi_deg = np.degrees(
        phi[k_phipi]
    ) % 360.0

    ax_meridian.set_xlabel("x")
    ax_meridian.set_ylabel("z")

    ax_meridian.set_title(
        "Meridional plane: "
        f"phi = {phi0_deg:.1f} / {phipi_deg:.1f} deg"
    )

    ax_meridian.set_aspect(
        "equal",
        adjustable="box",
    )

    _add_horizontal_colorbar(
        fig,
        ax_meridian,
        im_mer,
        label=colorbar_label,
    )

    # ------------------------------------------------------------
    # Same Cartesian spatial range for the two plane panels.
    # ------------------------------------------------------------
    rmax = float(
        np.max(np.abs(r))
    )

    for ax in (
        ax_equator,
        ax_meridian,
    ):
        ax.set_xlim(
            -rmax,
            rmax,
        )

        ax.set_ylim(
            -rmax,
            rmax,
        )

    # ------------------------------------------------------------
    # Overall annotation: optional title + simulation date/time
    # ------------------------------------------------------------
    if title is not None and datetime_label is not None:
        figure_title = (
            f"{title}\n"
            f"{datetime_label}"
        )
    elif title is not None:
        figure_title = title
    elif datetime_label is not None:
        figure_title = datetime_label
    else:
        figure_title = None

    if figure_title is not None:
        fig.suptitle(
            figure_title,
            fontsize=13,
        )

    fig.subplots_adjust(
        bottom=0.20,
        top=0.84 if figure_title is not None else 0.90,
    )

    return fig, (
        ax_shell,
        ax_equator,
        ax_meridian,
    )


if __name__ == "__main__":

    from read_merged_sip_data import read_merged_physics

    data_files = select_data_files(
        data_dir=DATA_DIR,
        process_mode=PROCESS_MODE,
        single_filename=SINGLE_DATA_FILE,
    )

    print(
        f"Processing mode: {PROCESS_MODE}"
        f"\nData directory : {DATA_DIR}"
        f"\nFile count     : {len(data_files)}"
    )

    if SAVE_OR_NOT:
        SAVE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if PLOT_BR:
            (
                SAVE_DIR
                / "Br"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if PLOT_VR:
            (
                SAVE_DIR
                / "Vr"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if PLOT_VH:
            (
                SAVE_DIR
                / "Vh"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if PLOT_DENSITY:
            (
                SAVE_DIR
                / "n"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    for file_index, DATA_FILE in enumerate(
        data_files,
        start=1,
    ):

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"Processing file "
            f"{file_index}/{len(data_files)}:"
            f"\n  {DATA_FILE.name}"
        )

        print(
            "=" * 72
        )

        data = read_merged_physics(
            filename=DATA_FILE,
            grid_filename=GRID_FILE,
        )

        if PLOT_VH:
            if "vtheta" not in data or "vphi" not in data:
                raise KeyError(
                    'PLOT_VH=True requires "vtheta" and "vphi" in data. '
                    f"Available keys: {list(data.keys())}"
                )

            data["Vh"] = np.sqrt(
                np.asarray(data["vtheta"], dtype=float)**2
                + np.asarray(data["vphi"], dtype=float)**2
            )

        simulation_hours = simulation_hours_from_filename(
            DATA_FILE
        )

        simulation_datetime = simulation_datetime_from_filename(
            DATA_FILE
        )

        datetime_label = format_simulation_datetime(
            DATA_FILE
        )

        print(
            "Simulation time from filename:"
            f"\n  file     = {DATA_FILE.name}"
            f"\n  time     = {simulation_hours:.2f} h"
            f"\n  datetime = {simulation_datetime:%Y-%m-%d %H:%M}"
        )

        radial_index, radial_value_resolved = _resolve_radial_index(
            np.asarray(
                data["r"],
                dtype=float,
            ),
            radial_mode=RADIAL_MODE,
            radial_index=RADIAL_INDEX,
            radial_value=RADIAL_VALUE,
        )

        print(
            "\nSpherical-shell selection:"
            f"\n  mode          = {RADIAL_MODE}"
            f"\n  radial index  = {radial_index}"
            f"\n  radial value  = {radial_value_resolved:.8g} Rs"
        )

        # ----------------------------------------------------------
        # Br
        # ----------------------------------------------------------
        if PLOT_BR:
            plot_merged_slices(
                data,
                variable="Br",
                radial_mode="index",
                radial_index=radial_index,
                cmap=BR_CMAP,
                shell_clim=BR_SHELL_CLIM,
                equator_clim=BR_EQUATOR_CLIM,
                meridian_clim=BR_MERIDIAN_CLIM,
                figsize=FIGSIZE,
                datetime_label=datetime_label,
                colorbar_label=BR_COLORBAR_LABEL,
            )

            if SAVE_OR_NOT:
                plt.savefig(
                    SAVE_DIR
                    / "Br"
                    / (
                        f"Br_slices_time.{simulation_hours:.2f}."
                        f"idr.{radial_index}.png"
                    ),
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )

        # ----------------------------------------------------------
        # Radial velocity
        # ----------------------------------------------------------
        if PLOT_VR:
            plot_merged_slices(
                data,
                variable="vr",
                radial_mode="index",
                radial_index=radial_index,
                cmap=VR_CMAP,
                shell_clim=VR_SHELL_CLIM,
                equator_clim=VR_EQUATOR_CLIM,
                meridian_clim=VR_MERIDIAN_CLIM,
                figsize=FIGSIZE,
                datetime_label=datetime_label,
                colorbar_label=VR_COLORBAR_LABEL,
            )

            if SAVE_OR_NOT:
                plt.savefig(
                    SAVE_DIR
                    / "Vr"
                    / (
                        f"Vr_slices_time.{simulation_hours:.2f}."
                        f"idr.{radial_index}.png"
                    ),
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )

        # ----------------------------------------------------------
        # Tangential velocity magnitude
        # ----------------------------------------------------------
        if PLOT_VH:
            plot_merged_slices(
                data,
                variable="Vh",
                radial_mode="index",
                radial_index=radial_index,
                cmap=VH_CMAP,
                shell_clim=VH_SHELL_CLIM,
                equator_clim=VH_EQUATOR_CLIM,
                meridian_clim=VH_MERIDIAN_CLIM,
                figsize=FIGSIZE,
                datetime_label=datetime_label,
                colorbar_label=VH_COLORBAR_LABEL,
            )

            if SAVE_OR_NOT:
                plt.savefig(
                    SAVE_DIR
                    / "Vh"
                    / (
                        f"Vh_slices_time.{simulation_hours:.2f}."
                        f"idr.{radial_index}.png"
                    ),
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )

        # ----------------------------------------------------------
        # Number density
        # ----------------------------------------------------------
        if PLOT_DENSITY:
            plot_merged_slices(
                data,
                variable="nrho_cm-3",
                radial_mode="index",
                radial_index=radial_index,
                cmap=DENSITY_CMAP,
                shell_clim=DENSITY_SHELL_CLIM,
                equator_clim=DENSITY_EQUATOR_CLIM,
                meridian_clim=DENSITY_MERIDIAN_CLIM,
                figsize=FIGSIZE,
                datetime_label=datetime_label,
                colorbar_label=DENSITY_COLORBAR_LABEL,
            )

            if SAVE_OR_NOT:
                plt.savefig(
                    SAVE_DIR
                    / "n"
                    / (
                        f"n_slices_time.{simulation_hours:.2f}."
                        f"idr.{radial_index}.png"
                    ),
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )

        if SAVE_OR_NOT:
            plt.close(
                "all"
            )
        else:
            plt.show()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "All selected merged HDF5 files finished."
    )
