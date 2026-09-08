"""Shared coordinate and solar differential-rotation utilities.

This module does not generate files.
"""

from __future__ import annotations

import numpy as np


# Snodgrass-like photospheric differential rotation [deg day^-1].
DIFFROT_A_DEG_PER_DAY = 14.713
DIFFROT_B_DEG_PER_DAY = -2.396
DIFFROT_C_DEG_PER_DAY = -1.787


def differential_rotation_rate_deg_per_day(latitude_deg: np.ndarray | float) -> np.ndarray:
    """Return the reference angular rotation rate at heliographic latitude."""
    latitude_rad = np.radians(np.asarray(latitude_deg, dtype=float))
    sin2 = np.sin(latitude_rad) ** 2
    return (
        DIFFROT_A_DEG_PER_DAY
        + DIFFROT_B_DEG_PER_DAY * sin2
        + DIFFROT_C_DEG_PER_DAY * sin2**2
    )


def rotate_longitude_deg(longitude_deg: np.ndarray | float, latitude_deg: np.ndarray | float, delta_hours: float) -> np.ndarray:
    """Advance Carrington-style longitude eastward by differential rotation."""
    longitude = np.asarray(longitude_deg, dtype=float)
    rate = differential_rotation_rate_deg_per_day(latitude_deg)
    return (longitude + rate * float(delta_hours) / 24.0) % 360.0


def longitude_interval_mask(longitude_deg: np.ndarray, lower_deg: float, upper_deg: float) -> np.ndarray:
    """Return inclusive periodic-longitude membership for one interval."""
    longitude = np.asarray(longitude_deg, dtype=float) % 360.0
    lower = float(lower_deg) % 360.0
    upper = float(upper_deg) % 360.0
    if lower <= upper:
        return (longitude >= lower) & (longitude <= upper)
    return (longitude >= lower) | (longitude <= upper)
