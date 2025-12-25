"""
ICF Neutron Simulation Package
==============================

This package provides a modular Monte-Carlo simulator for the
time-of-flight (TOF) behaviour of neutrons in an inertial confinement fusion
(ICF) diagnostic.

Modules:
--------
- constants: Physical constants and configuration
- config: Configurable simulation parameters
- data_classes: Data structures (MeshGeometry, DetectorPlane, NeutronRecord)
- stl_utils: STL file loading and processing
- geometry: Geometry processing and ray-mesh intersection
- sampling: Random sampling utilities
- kinematics: Energy and scattering calculations
- cross_section: Cross-section data handling
- transport: Particle transport through materials
- simulation: High-level simulation driver
- visualization: Plotting and visualization
- io_utils: Data import/export utilities
"""

from .constants import *
from . import config
from .data_classes import MeshGeometry, DetectorPlane, NeutronRecord
from .stl_utils import load_stl_mesh
from .geometry import (
    prepare_mesh_geometry,
    ray_mesh_intersection,
    find_exit_with_retry,
    build_orthonormal_frame,
    build_detector_plane_from_mesh,
    build_default_detector_plane,
    build_circular_detector_plane,
    mesh_distance_statistics,
    infer_mesh_axis,
)
from .sampling import (
    sample_neutron_energy,
    sample_isotropic_direction,
    sample_direction_in_cone,
)
from .kinematics import (
    energy_to_speed,
    scatter_energy_elastic,
    scatter_neutron_elastic_cms_to_lab,
)
from .cross_section import (
    load_mfp_data_from_csv,
    calculate_pe_macro_sigma,
    get_macro_sigma_at_energy,
    get_mfp_energy_dependent,
    MFP_DATA_PE,
    MFP_DATA_AL,
)
from .transport import (
    transport_through_slab,
    propagate_through_mesh_material,
    simulate_in_aluminium,
    propagate_to_scintillator,
)
from .simulation import (
    simulate_neutron_history,
    run_simulation,
)
from .visualization import (
    visualize_neutron_data,
    visualize_detector_hits,
    print_statistics,
)
from .io_utils import (
    export_neutron_records_to_csv,
    export_neutron_trajectories_to_csv,
)

__version__ = "1.0.0"
__all__ = [
    # Config module
    "config",
    # Constants
    "AVOGADRO_CONSTANT",
    "BARN_TO_M2",
    "DEBUG",
    "DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG",
    # Data classes
    "MeshGeometry",
    "DetectorPlane", 
    "NeutronRecord",
    # STL utils
    "load_stl_mesh",
    # Geometry
    "prepare_mesh_geometry",
    "ray_mesh_intersection",
    "find_exit_with_retry",
    "build_orthonormal_frame",
    "build_detector_plane_from_mesh",
    "build_default_detector_plane",
    "build_circular_detector_plane",
    "mesh_distance_statistics",
    "infer_mesh_axis",
    # Sampling
    "sample_neutron_energy",
    "sample_isotropic_direction",
    "sample_direction_in_cone",
    # Kinematics
    "energy_to_speed",
    "scatter_energy_elastic",
    "scatter_neutron_elastic_cms_to_lab",
    # Cross section
    "load_mfp_data_from_csv",
    "calculate_pe_macro_sigma",
    "get_macro_sigma_at_energy",
    "get_mfp_energy_dependent",
    "MFP_DATA_PE",
    "MFP_DATA_AL",
    # Transport
    "transport_through_slab",
    "propagate_through_mesh_material",
    "simulate_in_aluminium",
    "propagate_to_scintillator",
    # Simulation
    "simulate_neutron_history",
    "run_simulation",
    # Visualization
    "visualize_neutron_data",
    "visualize_detector_hits",
    "print_statistics",
    # IO
    "export_neutron_records_to_csv",
    "export_neutron_trajectories_to_csv",
]
