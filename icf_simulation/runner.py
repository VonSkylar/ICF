"""
ICF Neutron Simulation Runner Module

This module provides the main simulation runner function that can be called
from scripts or imported directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from . import config
from .config import DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG
from .core.constants import AVOGADRO_CONSTANT, BARN_TO_M2
from .core.stl_utils import load_stl_mesh
from .core.geometry import (
    prepare_mesh_geometry,
    build_circular_detector_plane,
    mesh_distance_statistics,
)
from .core.cross_section import (
    load_mfp_data_from_csv,
    calculate_pe_macro_sigma,
    MFP_DATA_AL,
    MFP_DATA_PE,
)
from .core.simulation import run_simulation
from .plotting import visualize_neutron_data, visualize_detector_hits, print_statistics
from .core.io_utils import export_neutron_records_to_csv, export_neutron_trajectories_to_csv
from .core.data_classes import NeutronRecord


def load_cross_section_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load cross-section data from package-bundled CSV files.
    
    Cross-section data files are bundled with the package in
    icf_simulation/data/cross_sections/
        
    Returns
    -------
    aluminium_mfp_data : np.ndarray
        Aluminium macroscopic cross-section data.
    channel_mfp_data : np.ndarray
        Channel (PE) macroscopic cross-section data.
    h_mfp_data : np.ndarray or None
        Hydrogen macroscopic cross-section data (for nuclide sampling).
    c_mfp_data : np.ndarray or None
        Carbon macroscopic cross-section data (for nuclide sampling).
    """
    # 使用包内路径
    AL_CSV_FILE = config.CROSS_SECTION_DIR / config.AL_CROSS_SECTION_FILE
    H_CSV_FILE = config.CROSS_SECTION_DIR / config.H_CROSS_SECTION_FILE
    C_CSV_FILE = config.CROSS_SECTION_DIR / config.C_CROSS_SECTION_FILE 
    
    # Load Aluminium cross-section data
    try:
        if AL_CSV_FILE.exists():
            al_micro_data = load_mfp_data_from_csv(str(AL_CSV_FILE))
            
            rho_Al_g_cm3 = config.AL_DENSITY_G_CM3
            M_Al = config.AL_MOLAR_MASS
            rho_Al_g_m3 = rho_Al_g_cm3 * 1e6  # g/m³
            N_Al_m3 = (rho_Al_g_m3 / M_Al) * AVOGADRO_CONSTANT  # atoms/m³
            
            aluminium_mfp_data = np.column_stack([
                al_micro_data[:, 0],  # Energy (MeV)
                al_micro_data[:, 1] * N_Al_m3 * BARN_TO_M2  # Σ_Al (m⁻¹)
            ])
            print(f"[info] Loaded Aluminium microscopic cross-sections from {AL_CSV_FILE.name}")
            print(f"[info] Converted to macroscopic: N_Al = {N_Al_m3:.4e} atoms/m^3")
        else:
            aluminium_mfp_data = MFP_DATA_AL
            print("[info] Using default internal MFP data for Aluminium.")
    except Exception as e:
        print(f"[warning] Failed to load/convert Aluminium data. Using default. Error: {e}")
        aluminium_mfp_data = MFP_DATA_AL

    # Load Polyethylene cross-section data (H and C)
    h_mfp_data = None
    c_mfp_data = None
    
    try:
        if H_CSV_FILE.exists() and C_CSV_FILE.exists():
            h_micro_data = load_mfp_data_from_csv(str(H_CSV_FILE))
            c_micro_data = load_mfp_data_from_csv(str(C_CSV_FILE))
            
            pe_data_calculated = calculate_pe_macro_sigma(h_micro_data, c_micro_data)
            
            # Calculate separate macroscopic cross-sections for H and C (for nuclide sampling)
            rho_PE = config.PE_DENSITY_G_CM3 * 1e6  # Convert g/cm³ to g/m³
            M_C2H4 = config.PE_MOLAR_MASS
            N_C = (config.C_ATOMS_PER_MOLECULE / M_C2H4) * rho_PE * AVOGADRO_CONSTANT  # number density of C (m⁻³)
            N_H = (config.H_ATOMS_PER_MOLECULE / M_C2H4) * rho_PE * AVOGADRO_CONSTANT  # number density of H (m⁻³)
            
            h_mfp_data = np.column_stack([
                h_micro_data[:, 0],  # Energy (MeV)
                h_micro_data[:, 1] * N_H * BARN_TO_M2  # Σ_H (m⁻¹)
            ])
            c_mfp_data = np.column_stack([
                c_micro_data[:, 0],  # Energy (MeV)
                c_micro_data[:, 1] * N_C * BARN_TO_M2  # Σ_C (m⁻¹)
            ])
            
            channel_mfp_data = pe_data_calculated
            print(f"[info] Calculated Polyethylene MFP data from H.csv and C.csv.")
            print(f"[info] Nuclide sampling enabled: H (A=1) and C (A=12) separated.")
        else:
            channel_mfp_data = MFP_DATA_PE
            print("[info] Using default internal MFP data for Polyethylene.")
            print("[warning] Nuclide sampling disabled: H and C not separated.")
    except Exception as e:
        print(f"[warning] Failed to calculate/load custom Polyethylene MFP data. Using default. Error: {e}")
        channel_mfp_data = MFP_DATA_PE
    
    return aluminium_mfp_data, channel_mfp_data, h_mfp_data, c_mfp_data


def load_stl_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """Load STL geometry files from package-bundled location.
    
    STL model files are bundled with the package in
    icf_simulation/data/stl_models/
        
    Returns
    -------
    shell_geometry : MeshGeometry
        Prepared shell geometry.
    channel_geometry : MeshGeometry
        Prepared channel geometry.
    """
    stl_dir = config.STL_MODEL_DIR
    
    shell_stl_path = stl_dir / config.SHELL_STL_FILE
    channel_stl_path = stl_dir / config.CHANNEL_STL_FILE
    
    if not shell_stl_path.exists():
        raise FileNotFoundError(f"Shell STL not found: {shell_stl_path}")
    if not channel_stl_path.exists():
        raise FileNotFoundError(f"Channel STL not found: {channel_stl_path}")
    
    print(f"[info] Loading shell STL: {shell_stl_path.name}")
    shell_mesh = load_stl_mesh(str(shell_stl_path))
    
    print(f"[info] Loading channel STL: {channel_stl_path.name}")
    channel_mesh = load_stl_mesh(str(channel_stl_path))
    
    # Convert millimetres to metres
    unit_scale = config.MM_TO_M
    mesh_scaled = shell_mesh * unit_scale
    channel_scaled = channel_mesh * unit_scale
    
    shell_geometry = prepare_mesh_geometry(mesh_scaled)
    channel_geometry = prepare_mesh_geometry(channel_scaled)
    
    mean_radius, max_radius = mesh_distance_statistics(mesh_scaled)
    print(f"[info] Shell thickness will be calculated from STL mesh geometry")
    print(f"[info] STL vertex distances: mean={mean_radius:.4f} m, max={max_radius:.4f} m")
    
    return shell_geometry, channel_geometry


def run_full_simulation(
    output_dir: Optional[Path] = None,
    n_neutrons: Optional[int] = None,
    save_results: bool = True,
    generate_plots: bool = True,
) -> List[NeutronRecord]:
    """Run the complete ICF neutron simulation.
    
    This is the main entry point for running simulations. It handles:
    1. Loading cross-section data (from package-bundled files)
    2. Loading STL geometry (from package-bundled files)
    3. Running the Monte Carlo simulation
    4. Exporting results
    5. Generating visualization plots
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory for output files (Data/, Figures/). If None, uses current working directory.
    n_neutrons : int, optional
        Number of neutrons to simulate. If None, uses config default.
    save_results : bool
        Whether to save results to CSV files.
    generate_plots : bool
        Whether to generate visualization plots.
        
    Returns
    -------
    List[NeutronRecord]
        List of neutron records from simulation.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    if n_neutrons is None:
        n_neutrons = config.DEFAULT_N_NEUTRONS
    
    # Load data (from package-bundled files)
    aluminium_mfp_data, channel_mfp_data, h_mfp_data, c_mfp_data = load_cross_section_data()
    shell_geometry, channel_geometry = load_stl_geometry()
    
    # Configure detector
    channel_axis = np.array(config.CHANNEL_AXIS)
    detector_z_mm = config.DETECTOR_Z_MM
    detector_center_mm = np.array([0.0, 0.0, detector_z_mm])
    detector_radius_mm = config.DETECTOR_RADIUS_MM
    detector_plane = build_circular_detector_plane(detector_center_mm, detector_radius_mm, channel_axis)
    
    print("\n" + "="*70)
    print("COORDINATE SYSTEM CONFIGURATION")
    print("="*70)
    print(f"Channel axis: +Z direction [0, 0, 1]")
    print(f"Detector plane: XY plane (perpendicular to Z axis)")
    print(f"Neutron source: Origin (0, 0, 0)")
    print(f"Detector position: z = {detector_z_mm:.1f} mm = {detector_plane.plane_position:.4f} m")
    print(f"Detector center (3D): {detector_plane.center} m")
    print(f"Detector radius: {detector_radius_mm:.1f} mm = {detector_plane.radius:.4f} m")
    print(f"Source cone half-angle: {DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG}°")
    print("="*70 + "\n")
    
    # Run simulation
    print(f"[info] Starting simulation with {n_neutrons} neutrons...")
    
    neutron_records = run_simulation(
        n_neutrons=n_neutrons,
        shell_thickness=0.0,  # Calculated from STL geometry
        aluminium_mass_ratio=config.AL_MASS_RATIO,
        aluminium_mfp_data=aluminium_mfp_data,
        detector_distance=detector_plane.plane_position,
        detector_side=1.0,
        energy_cutoff_mev=config.DEFAULT_ENERGY_CUTOFF_MEV,
        shell_geometry=shell_geometry,
        channel_geometry=channel_geometry,
        channel_mfp_data=channel_mfp_data,
        channel_mass_ratio=config.PE_MASS_RATIO,
        source_cone_axis=channel_axis,
        detector_plane=detector_plane,
        h_mfp_data=h_mfp_data,
        c_mfp_data=c_mfp_data,
    )
    
    # Print statistics
    print_statistics(neutron_records, n_neutrons)
    
    # Save results
    if save_results and neutron_records:
        csv_filename = str(output_dir / config.DATA_OUTPUT_DIR / config.NEUTRON_DATA_CSV)
        export_neutron_records_to_csv(neutron_records, filename=csv_filename)
        
        trajectory_filename = str(output_dir / config.DATA_OUTPUT_DIR / config.TRAJECTORY_DATA_CSV)
        export_neutron_trajectories_to_csv(neutron_records, filename=trajectory_filename)
    
    # Generate plots
    if generate_plots and neutron_records:
        print("[info] Generating visualizations...")
        save_base = str(output_dir / config.FIGURES_OUTPUT_DIR / config.NEUTRON_ANALYSIS_FIGURE_BASE)
        visualize_neutron_data(neutron_records, save_path=save_base)
        visualize_detector_hits(neutron_records, detector_plane, save_path=save_base)
        print("[info] Visualization complete!")
    
    return neutron_records


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ICF neutron simulation")
    parser.add_argument("-n", "--neutrons", type=int, default=None,
                        help="Number of neutrons to simulate")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to CSV")
    parser.add_argument("--no-plot", action="store_true",
                        help="Don't generate visualization plots")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results and figures")
    
    args = parser.parse_args()
    
    run_full_simulation(
        output_dir=args.output_dir,
        n_neutrons=args.neutrons,
        save_results=not args.no_save,
        generate_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()