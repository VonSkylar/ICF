"""
Random sampling utilities for neutron generation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .geometry import build_orthonormal_frame


def sample_neutron_energy(mean_mev: float = 2.45, std_mev: float = 0.1) -> float:
    """Sample a neutron kinetic energy from a Gaussian distribution.

    Parameters
    ----------
    mean_mev : float, optional
        Mean energy in MeV of the distribution. Defaults to 2.45 MeV.
    std_mev : float, optional
        Standard deviation of the energy distribution in MeV. Defaults to 0.1 MeV.

    Returns
    -------
    float
        A randomly sampled energy in MeV. Negative energies are rejected and resampled.
    """
    energy = np.random.normal(mean_mev, std_mev)
    while energy <= 0.0:
        energy = np.random.normal(mean_mev, std_mev)
    return float(energy)


def sample_isotropic_direction() -> np.ndarray:
    """Generate a random unit vector isotropically distributed on the sphere."""
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([r_xy * math.cos(phi), r_xy * math.sin(phi), z], dtype=float)


def sample_direction_in_cone(
    axis: np.ndarray,
    half_angle_deg: float,
) -> np.ndarray:
    """Sample a unit vector within a cone of half-angle ``half_angle_deg``."""
    axis, u, v = build_orthonormal_frame(axis)
    half_angle_rad = math.radians(half_angle_deg)
    cos_min = math.cos(half_angle_rad)
    cos_theta = (1.0 - cos_min) * np.random.rand() + cos_min
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * np.random.rand()
    local_dir = np.array(
        [
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        ],
        dtype=float,
    )

    return local_dir[0] * u + local_dir[1] * v + local_dir[2] * axis
