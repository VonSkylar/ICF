"""
Cross-section data loading and mean free path calculations.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .constants import AVOGADRO_CONSTANT, BARN_TO_M2


# Default macroscopic cross-section data (Energy [MeV] -> macroscopic cross-section [m⁻¹])
# Polyethylene (C₂H₄), density ≈ 0.92 g/cm³
MFP_DATA_PE = np.array([
    [0.1, 15.0],
    [0.5, 13.5],
    [1.0, 10.0],
    [2.45, 16.6],
    [5.0, 17.5],
    [10.0, 18.0],
    [14.1, 17.8],
])

# Aluminium (Al), density ≈ 2.70 g/cm³
MFP_DATA_AL = np.array([
    [0.1, 10.0],
    [0.5, 11.2],
    [1.0, 13.0],
    [2.45, 14.4],
    [5.0, 15.5],
    [10.0, 16.0],
    [14.1, 16.2],
])


def load_mfp_data_from_csv(file_path: str) -> np.ndarray:
    """
    Load cross-section data from a two-column CSV file.

    The file must contain data pairs: [Energy, Cross-Section].
    Energy is assumed to be in eV and will be converted to MeV.

    Parameters
    ----------
    file_path : str
        Path to the CSV file on disk containing the cross section data.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Sorted array of [Energy (MeV), Cross-Section (original units)].
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cross section data file '{file_path}' does not exist.")

    try:
        skip_rows = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    float(line.split(';')[0])
                    break
                except ValueError:
                    skip_rows += 1
                if skip_rows > 3:
                    break

        data = np.loadtxt(
            file_path, 
            delimiter=';', 
            skiprows=skip_rows, 
            usecols=(0, 1),
            dtype=float
        )
        
        # Convert eV to MeV
        data[:, 0] = data[:, 0] * 1e-6 
        
    except Exception as e:
        raise ValueError(f"Could not load and process data from CSV: {e}")

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Processed data must contain exactly two columns: Energy (MeV) and Cross-Section.")

    data = data[data[:, 0].argsort()]
    return data[:, :2]


def calculate_pe_macro_sigma(
    h_micro_data: np.ndarray,
    c_micro_data: np.ndarray,
    density_g_cm3: float = 0.92,
) -> np.ndarray:
    """
    Calculates the macroscopic cross section for Polyethylene (C₂H₄) 
    by combining Hydrogen (H) and Carbon (C) micro-sections.

    Assumes H and C data arrays contain micro-sections (sigma) in BARN.

    Parameters
    ----------
    h_micro_data : np.ndarray
        [Energy (MeV), Sigma_Micro (barn)] data for Hydrogen.
    c_micro_data : np.ndarray
        [Energy (MeV), Sigma_Micro (barn)] data for Carbon.
    density_g_cm3 : float, optional
        Density of polyethylene (g/cm³).

    Returns
    -------
    np.ndarray, shape (N, 2)
        Combined [Energy (MeV), Sigma_Macro_PE (m⁻¹)].
    """
    M_C = 12.011  # g/mol
    M_H = 1.008  # g/mol
    M_PE = 2 * M_C + 4 * M_H  # ≈ 28.05 g/mol (C₂H₄)
    
    N_C_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 2) / M_PE
    N_H_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 4) / M_PE
    
    N_C_m3 = N_C_cm3 * 1e6
    N_H_m3 = N_H_cm3 * 1e6

    all_energies = np.unique(np.concatenate([h_micro_data[:, 0], c_micro_data[:, 0]]))
    
    sigma_h_interp_barn = np.interp(all_energies, h_micro_data[:, 0], h_micro_data[:, 1])
    sigma_c_interp_barn = np.interp(all_energies, c_micro_data[:, 0], c_micro_data[:, 1])
    
    sigma_pe_total_m1 = (
        N_C_m3 * sigma_c_interp_barn * BARN_TO_M2 +
        N_H_m3 * sigma_h_interp_barn * BARN_TO_M2
    )
    
    pe_macro_data = np.stack([all_energies, sigma_pe_total_m1], axis=1)
    
    return pe_macro_data


def get_macro_sigma_at_energy(energy_mev: float, mfp_data: np.ndarray) -> float:
    """Get macroscopic cross-section (Sigma, m⁻¹) at a given neutron energy.
    
    Parameters
    ----------
    energy_mev : float
        Neutron kinetic energy (MeV).
    mfp_data : np.ndarray
        [Energy (MeV), macroscopic cross-section (m⁻¹)] data array.
        
    Returns
    -------
    float
        Macroscopic cross-section Sigma (m⁻¹) at the given energy.
    """
    if energy_mev <= 0.1:
        return 1e-12
    
    energies = mfp_data[:, 0]
    sigmas = mfp_data[:, 1]
    
    if energy_mev < energies.min():
        sigma = sigmas[0]
    elif energy_mev > energies.max():
        sigma = sigmas[-1]
    else:
        sigma = np.interp(energy_mev, energies, sigmas)
    
    return max(sigma, 1e-12)


def get_mfp_energy_dependent(
    energy_mev: float,
    mfp_data: np.ndarray,
) -> float:
    """
    Calculate mean free path (MFP) based on neutron energy using linear interpolation.

    Parameters
    ----------
    energy_mev : float
        Neutron kinetic energy (MeV).
    mfp_data : np.ndarray
        [Energy (MeV), macroscopic cross-section (m⁻¹)] data array.

    Returns
    -------
    float
        Mean free path (m).
    """
    if energy_mev <= 0.1:
        return 1e12

    energies = mfp_data[:, 0]
    sigmas = mfp_data[:, 1]
    
    if energy_mev < energies.min():
        sigma = sigmas[0]
    elif energy_mev > energies.max():
        sigma = sigmas[-1]
    else:
        sigma = np.interp(energy_mev, energies, sigmas)

    if sigma <= 1e-12:
        return 1e12
        
    return 1.0 / sigma
