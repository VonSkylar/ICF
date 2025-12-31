"""
Configuration settings for ICF neutron simulation.

This module contains all configurable parameters that were previously hardcoded.
Users can modify these values to customize the simulation without changing the core code.

Data files (cross-sections, STL models) are now bundled with the package.
Use `icf_simulation.data_paths` to access them:

    from icf_simulation.data_paths import get_cross_section_dir, get_stl_dir
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Package Data Paths (内置数据路径)
# =============================================================================

# Package root directory
_PACKAGE_DIR = Path(__file__).resolve().parent

# Cross-section data directory (包内)
CROSS_SECTION_DIR = _PACKAGE_DIR / "data" / "cross_sections"
AL_CROSS_SECTION_FILE = "Al.csv"
H_CROSS_SECTION_FILE = "H.csv"
C_CROSS_SECTION_FILE = "C.csv"

# STL geometry files directory (包内)
STL_MODEL_DIR = _PACKAGE_DIR / "data" / "stl_models"
SHELL_STL_FILE = "Target_ball_model.STL"
CHANNEL_STL_FILE = "nTOF_without_scintillant.STL"

# Output directories (用户工作目录)
DATA_OUTPUT_DIR = "Data"
FIGURES_OUTPUT_DIR = "Figures"

# Output file names
NEUTRON_DATA_CSV = "neutron_data.csv"
TRAJECTORY_DATA_CSV = "neutron_trajectories.csv"
STL_GEOMETRY_FIGURE = "stl_geometry_visualization.png"
NEUTRON_ANALYSIS_FIGURE_BASE = "neutron_analysis"
TRAJECTORY_FIGURE_BASE = "neutron_trajectories"

# =============================================================================
# Detector Configuration
# =============================================================================

# Detector position (along channel axis, in mm)
DETECTOR_Z_MM = 2900.0

# Detector radius (in mm)
DETECTOR_RADIUS_MM = 105.0

# Channel axis direction (unit vector)
CHANNEL_AXIS = [0.0, 0.0, 1.0]

# =============================================================================
# Simulation Parameters
# =============================================================================

# Number of neutron histories to simulate
DEFAULT_N_NEUTRONS = 100

# Energy cutoff for transport (MeV)
DEFAULT_ENERGY_CUTOFF_MEV = 0.1

# Source cone half-angle (degrees)
DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG = 10

# =============================================================================
# Material Properties
# =============================================================================

# Aluminium properties
AL_DENSITY_G_CM3 = 2.70  # g/cm³
AL_MOLAR_MASS = 26.981  # g/mol
AL_MASS_RATIO = 26.98  # Mass ratio A for scattering

# Polyethylene (C2H4) properties
PE_DENSITY_G_CM3 = 0.92  # g/cm³
PE_MOLAR_MASS = 28.054  # g/mol (for C2H4)
PE_MASS_RATIO = 1.0  # Effective mass ratio (H-dominated)

# Hydrogen properties (in PE)
H_ATOMS_PER_MOLECULE = 4.0
H_MASS_RATIO = 1.0

# Carbon properties (in PE)
C_ATOMS_PER_MOLECULE = 2.0
C_MASS_RATIO = 12.0

# =============================================================================
# Visualization Settings
# =============================================================================

# Maximum number of trajectories to plot in visualizations
MAX_TRAJECTORIES_TO_PLOT = 100

# Plot DPI settings
PLOT_DPI = 300
QUICK_PLOT_DPI = 150

# 3D plot figure size
TRAJECTORY_3D_FIGSIZE = (11, 16)

# 2D projections figure size
TRAJECTORY_2D_FIGSIZE = (16, 10)

# Mesh visualization settings
STL_SHELL_ALPHA = 0.12  # Transparency for shell geometry
STL_SHELL_COLOR = "lightgray"
STL_CHANNEL_ALPHA = 0.2  # Transparency for channel geometry
STL_CHANNEL_COLOR = "lightblue"
DETECTOR_COLOR = "red"
DETECTOR_ALPHA = 0.25

# =============================================================================
# Unit Conversions
# =============================================================================

# STL files are typically in millimeters, convert to meters
MM_TO_M = 1.0e-3
