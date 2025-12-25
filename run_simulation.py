#!/usr/bin/env python
"""
ICF Neutron Simulation - Main Runner Script
============================================

This script runs the complete ICF neutron simulation using modular components
from the icf_simulation package.

Usage:
    python run_simulation.py

The simulation will:
1. Load cross-section data from CSV files
2. Load STL geometry files
3. Run the Monte Carlo simulation
4. Export results to CSV files
5. Generate visualization plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Import from the modular package
from icf_simulation import (
    # Constants
    DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG,
    BARN_TO_M2,
    # STL loading
    load_stl_mesh,
    # Geometry
    prepare_mesh_geometry,
    build_circular_detector_plane,
    mesh_distance_statistics,
    # Cross-section data
    load_mfp_data_from_csv,
    calculate_pe_macro_sigma,
    MFP_DATA_AL,
    MFP_DATA_PE,
    # Simulation
    run_simulation,
    # Visualization
    visualize_neutron_data,
    visualize_detector_hits,
    print_statistics,
    # IO
    export_neutron_records_to_csv,
    export_neutron_trajectories_to_csv,
)


def main():
    """Main entry point for the ICF neutron simulation."""
    
    base_dir = Path(__file__).resolve().parent

    # =========================================================================
    # 1. Load Cross-Section Data
    # =========================================================================
    AL_CSV_FILE = base_dir / "cross_section_data" / "Al.csv" 
    H_CSV_FILE = base_dir / "cross_section_data" / "H.csv" 
    C_CSV_FILE = base_dir / "cross_section_data" / "C.csv" 
    
    # Load Aluminium cross-section data
    try:
        if AL_CSV_FILE.exists():
            from scipy.constants import Avogadro
            al_micro_data = load_mfp_data_from_csv(str(AL_CSV_FILE))
            
            rho_Al_g_cm3 = 2.70  # g/cm³
            M_Al = 26.981  # g/mol
            rho_Al_g_m3 = rho_Al_g_cm3 * 1e6  # g/m³
            N_Al_m3 = (rho_Al_g_m3 / M_Al) * Avogadro  # atoms/m³
            
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
            from scipy.constants import Avogadro
            
            h_micro_data = load_mfp_data_from_csv(str(H_CSV_FILE))
            c_micro_data = load_mfp_data_from_csv(str(C_CSV_FILE))
            
            pe_data_calculated = calculate_pe_macro_sigma(h_micro_data, c_micro_data)
            
            # Calculate separate macroscopic cross-sections for H and C (for nuclide sampling)
            rho_PE = 0.92e6  # g/m³
            M_C2H4 = 28.054  # g/mol
            N_C = (2.0 / M_C2H4) * rho_PE * Avogadro  # number density of C (m⁻³)
            N_H = (4.0 / M_C2H4) * rho_PE * Avogadro  # number density of H (m⁻³)
            
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

    # =========================================================================
    # 2. Load STL Geometry Files
    # =========================================================================
    stl_file_path: Optional[Path] = None
    for candidate in ("Target ball model.stl", "Target_ball_model.stl"):
        path = base_dir / candidate
        if path.exists():
            stl_file_path = path
            break
    if stl_file_path is None:
        stl_candidates = sorted(base_dir.glob("*.stl"))
        if not stl_candidates:
            raise FileNotFoundError("No STL geometry file found in the current directory.")
        stl_file_path = stl_candidates[0]
        print(f"[info] Using STL file: {stl_file_path.name}")

    shell_mesh = load_stl_mesh(str(stl_file_path))
    channel_mesh_path = base_dir / "nTOF_without_scintillant.STL"
    channel_mesh = load_stl_mesh(str(channel_mesh_path))
    
    # Convert millimetres to metres
    unit_scale = 1.0e-3
    mesh_scaled = shell_mesh * unit_scale
    channel_scaled = channel_mesh * unit_scale
    
    shell_geometry = prepare_mesh_geometry(mesh_scaled)
    channel_geometry = prepare_mesh_geometry(channel_scaled)
    mean_radius, max_radius = mesh_distance_statistics(mesh_scaled)
    
    # =========================================================================
    # 3. Configure Coordinate System and Detector
    # =========================================================================
    # Channel axis direction: [0, 0, 1] (positive Z direction)
    channel_axis = np.array([0.0, 0.0, 1.0])
    
    # Detector configuration
    detector_z_mm = 2900.0  # Distance along channel axis (mm)
    detector_center_mm = np.array([0.0, 0.0, detector_z_mm])
    detector_radius_mm = 105.0  # Detector radius (mm)
    detector_plane = build_circular_detector_plane(detector_center_mm, detector_radius_mm, channel_axis)

    # Shell thickness (calculated from STL mesh geometry)
    shell_thickness = 0.0  # Will be calculated from Target_ball_model.STL geometry

    print(f"[info] Shell thickness will be calculated from STL mesh geometry")
    print(f"[info] STL vertex distances: mean={mean_radius:.4f} m, max={max_radius:.4f} m")
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

    # =========================================================================
    # 4. Run Simulation
    # =========================================================================
    # Simulation parameters (adjust as needed)
    n_neutrons = 100  # Number of neutron histories to simulate
    aluminium_mass_ratio = 26.98  # Mass ratio A for aluminium
    channel_mass_ratio = 1.0  # Mass ratio for polyethylene (dominated by H)

    print(f"[info] Starting simulation with {n_neutrons} neutrons...")
    
    neutron_records = run_simulation(
        n_neutrons=n_neutrons,
        shell_thickness=shell_thickness,
        aluminium_mass_ratio=aluminium_mass_ratio,
        aluminium_mfp_data=aluminium_mfp_data,
        detector_distance=detector_plane.plane_position,
        detector_side=1.0,
        energy_cutoff_mev=0.1,
        shell_geometry=shell_geometry,
        channel_geometry=channel_geometry,
        channel_mfp_data=channel_mfp_data,
        channel_mass_ratio=channel_mass_ratio,
        source_cone_axis=channel_axis,
        detector_plane=detector_plane,
        h_mfp_data=h_mfp_data,
        c_mfp_data=c_mfp_data,
    )

    # =========================================================================
    # 5. Print Statistics and Export Results
    # =========================================================================
    print_statistics(neutron_records, n_neutrons)
    
    # Export neutron data to CSV
    if neutron_records:
        csv_filename = str(base_dir / "Data" / "neutron_data.csv")
        export_neutron_records_to_csv(neutron_records, filename=csv_filename)
        
        # Export trajectory data
        trajectory_filename = str(base_dir / "Data" / "neutron_trajectories.csv")
        export_neutron_trajectories_to_csv(neutron_records, filename=trajectory_filename)
    
    # =========================================================================
    # 6. Generate Visualizations
    # =========================================================================
    if neutron_records:
        print("[info] Generating visualizations...")
        save_base = str(base_dir / "Figures" / "neutron_analysis")
        visualize_neutron_data(neutron_records, save_path=save_base)
        visualize_detector_hits(neutron_records, detector_plane, save_path=save_base)
        print("[info] Visualization complete!")


if __name__ == "__main__":
    main()
