"""
Kinematics utilities for neutron energy and scattering calculations.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .constants import NEUTRON_MASS_KG


def energy_to_speed(energy_mev: float) -> float:
    """Convert neutron kinetic energy (MeV) to speed (m/s).

    The relationship between kinetic energy and speed for a non‑relativistic
    particle is E = ½ m v².

    Parameters
    ----------
    energy_mev : float
        Kinetic energy in MeV.

    Returns
    -------
    float
        Speed in metres per second.
    """
    energy_joules = energy_mev * 1.0e6 * 1.602176634e-19
    speed = math.sqrt(2.0 * energy_joules / NEUTRON_MASS_KG)
    return speed


def scatter_energy_elastic(neutron_energy_mev: float, target_mass_ratio: float) -> float:
    """Compute the neutron energy after an elastic scattering event.

    A simple two‑body kinematic model is used whereby the scattering is
    isotropic in the centre‑of‑mass frame.

    Parameters
    ----------
    neutron_energy_mev : float
        Incoming neutron energy in MeV.
    target_mass_ratio : float
        Ratio of the target nucleus mass to the neutron mass (A).

    Returns
    -------
    float
        The outgoing neutron energy in MeV after one elastic collision.
    """
    cos_theta = 2.0 * np.random.rand() - 1.0
    A = target_mass_ratio
    numerator = A * A + 1.0 + 2.0 * A * cos_theta
    denominator = (A + 1.0) * (A + 1.0)
    r = numerator / denominator
    r = max(0.0, min(r, 1.0))
    return float(neutron_energy_mev * r)


def scatter_neutron_elastic_cms_to_lab(
    neutron_energy_mev: float,
    incident_direction: np.ndarray,
    target_mass_ratio: float,
) -> Tuple[float, np.ndarray]:
    """Perform elastic neutron scattering with proper CMS to LAB frame conversion.
    
    This function implements the complete two-body elastic scattering kinematics:
    1. Sample scattering angle θ_cm isotropically in the center-of-mass (CMS) frame
    2. Calculate energy loss from θ_cm using the correct kinematic relation
    3. Convert θ_cm to θ_lab using the proper coordinate transformation
    4. Update the neutron direction in the laboratory (LAB) frame
    
    CRITICAL PHYSICS:
    For hydrogen (A=1), the CMS to LAB transformation ensures that:
    - θ_lab ∈ [0, π/2]: neutron can NEVER backscatter from proton
    
    Transformation formula:
        tan(θ_lab) = sin(θ_cm) / (γ + cos(θ_cm))
    where γ = 1/A
    
    Energy relation:
        E_out / E_in = [A² + 1 + 2A·cos(θ_cm)] / (A + 1)²
    
    Parameters
    ----------
    neutron_energy_mev : float
        Incident neutron kinetic energy in MeV.
    incident_direction : np.ndarray, shape (3,)
        Unit vector of neutron velocity before collision (LAB frame).
    target_mass_ratio : float
        Target nucleus mass / neutron mass (A).
    
    Returns
    -------
    tuple : (energy_out, direction_out)
        energy_out : float
            Neutron energy after scattering (MeV)
        direction_out : np.ndarray
            Unit vector of neutron velocity after scattering (LAB frame)
    """
    A = target_mass_ratio
    gamma = 1.0 / A
    
    # Sample scattering angle in CMS (isotropic)
    cos_theta_cm = 2.0 * np.random.rand() - 1.0
    sin_theta_cm = math.sqrt(max(0.0, 1.0 - cos_theta_cm**2))
    
    # Calculate energy after scattering
    numerator = A * A + 1.0 + 2.0 * A * cos_theta_cm
    denominator = (A + 1.0) * (A + 1.0)
    energy_ratio = numerator / denominator
    energy_ratio = max(0.0, min(energy_ratio, 1.0))
    energy_out = neutron_energy_mev * energy_ratio
    
    # Convert θ_cm to θ_lab
    denominator_angle = gamma + cos_theta_cm
    
    if abs(denominator_angle) < 1e-10:
        theta_lab = math.pi / 2.0
    else:
        theta_lab = math.atan2(sin_theta_cm, denominator_angle)
    
    if theta_lab < 0:
        theta_lab += math.pi
    
    # Sample azimuthal angle φ uniformly
    phi = 2.0 * math.pi * np.random.rand()
    
    # Construct new direction in LAB frame
    incident_dir = np.array(incident_direction, dtype=float)
    incident_dir /= np.linalg.norm(incident_dir)
    
    z_axis = incident_dir
    if abs(z_axis[2]) < 0.9:
        x_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    cos_theta_lab = math.cos(theta_lab)
    sin_theta_lab = math.sin(theta_lab)
    
    direction_local = np.array([
        sin_theta_lab * math.cos(phi),
        sin_theta_lab * math.sin(phi),
        cos_theta_lab
    ], dtype=float)
    
    direction_out = (
        direction_local[0] * x_axis +
        direction_local[1] * y_axis +
        direction_local[2] * z_axis
    )
    
    direction_out /= np.linalg.norm(direction_out)
    
    return energy_out, direction_out
