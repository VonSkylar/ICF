"""
High-level simulation driver functions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .constants import DEBUG
from .. import config
from ..config import DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG
from .data_classes import MeshGeometry, DetectorPlane, NeutronRecord
from .geometry import build_default_detector_plane
from .sampling import sample_neutron_energy, sample_isotropic_direction, sample_direction_in_cone
from .transport import simulate_in_aluminium, propagate_through_mesh_material, propagate_to_scintillator, unified_transport
from .cross_section import MFP_DATA_PE, MFP_DATA_AL


def simulate_neutron_history(
    shell_thickness: float,
    aluminium_mfp_data: np.ndarray,
    aluminium_mass_ratio: float,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
    channel_geometry: Optional[MeshGeometry] = None,
    channel_mfp_data: np.ndarray = MFP_DATA_PE,
    channel_mass_ratio: float = 1.0,
    source_cone_axis: Optional[np.ndarray] = None,
    source_cone_half_angle_deg: float = DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG,
    detector_plane: Optional[DetectorPlane] = None,
    h_mfp_data: Optional[np.ndarray] = None,
    c_mfp_data: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[NeutronRecord]]:
    """Simulate the complete history of a single neutron.
    
    This function now uses unified transport that handles overlapping geometries
    (Al shell and PE channel) correctly by tracking which material is hit first
    along the neutron's actual flight path.

    Returns
    -------
    tuple
        Returns (status_string, NeutronRecord) if successful, or (status_string, None)
        if the neutron was lost.
    """
    # 1. Generate initial energy and direction
    E0 = sample_neutron_energy()
    if source_cone_axis is None:
        d0 = sample_isotropic_direction()
    else:
        d0 = sample_direction_in_cone(source_cone_axis, source_cone_half_angle_deg)

    origin = np.zeros(3, dtype=float)
    
    # 2. Setup detector plane
    detector_plane = detector_plane or build_default_detector_plane(detector_distance, detector_side)

    # 3. Use unified transport through all materials
    (cumulative_time, final_energy, final_pos, final_dir, 
     detector_crossing, trajectory_points) = unified_transport(
        position=origin,
        direction=d0,
        energy_mev=E0,
        shell_geometry=shell_geometry,
        channel_geometry=channel_geometry,
        aluminium_mfp_data=aluminium_mfp_data,
        channel_mfp_data=channel_mfp_data,
        aluminium_mass_ratio=aluminium_mass_ratio,
        channel_mass_ratio=channel_mass_ratio,
        energy_cutoff_mev=energy_cutoff_mev,
        detector_plane=detector_plane,
        h_mfp_data=h_mfp_data,
        c_mfp_data=c_mfp_data,
    )
    
    # 4. Determine outcome and create record
    # Convert trajectory format for NeutronRecord
    formatted_trajectory = [(pos, energy, event) for pos, energy, event in trajectory_points]
    
    # Count scattering events in each material
    shell_scatters = sum(1 for _, _, e in trajectory_points if "scatter_Al" in e)
    channel_scatters = sum(1 for _, _, e in trajectory_points if "scatter_PE" in e)
    
    # Find energy after shell and after channel based on trajectory
    # Also find the first exit position from aluminum shell
    energy_after_shell = E0
    energy_after_channel = E0
    first_shell_exit_pos = None
    
    for pos, energy, event in trajectory_points:
        if "exit_Al" in event:
            if first_shell_exit_pos is None:
                first_shell_exit_pos = pos.copy()
            energy_after_shell = energy
        if "enter_PE" in event and first_shell_exit_pos is None:
            # If entering PE without explicit Al exit, use this position
            first_shell_exit_pos = pos.copy()
            energy_after_shell = energy
        if "exit_PE" in event or "detector" in event or "escape" in event:
            energy_after_channel = energy
    
    # If no shell exit was found, use final position
    if first_shell_exit_pos is None:
        first_shell_exit_pos = final_pos.copy()
    
    if detector_crossing is not None:
        crossing_time, crossing_point, crossing_energy = detector_crossing
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=crossing_energy,
            tof=crossing_time,
            exit_position=first_shell_exit_pos,
            detector_hit_position=crossing_point.copy(),
            reached_detector=True,
            energy_after_shell=energy_after_shell,
            energy_after_channel=crossing_energy,
            status="success",
            final_position=crossing_point.copy(),
            final_direction=final_dir.copy(),
            trajectory_points=formatted_trajectory
        )
        return ("success", record)
    
    if final_energy <= energy_cutoff_mev:
        # Lost due to energy cutoff
        # Determine where it was lost based on last trajectory event
        last_event = trajectory_points[-1][2] if trajectory_points else "unknown"
        if "Al" in last_event:
            status = "lost_in_shell"
        elif "PE" in last_event:
            status = "lost_in_channel"
        else:
            status = "lost_low_energy"
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=final_energy,
            tof=cumulative_time,
            exit_position=first_shell_exit_pos,
            detector_hit_position=None,
            reached_detector=False,
            energy_after_shell=energy_after_shell,
            energy_after_channel=energy_after_channel,
            status=status,
            final_position=final_pos.copy(),
            final_direction=final_dir.copy(),
            trajectory_points=formatted_trajectory
        )
        return (status, record)
    
    # Missed detector
    record = NeutronRecord(
        initial_energy=E0,
        final_energy=final_energy,
        tof=cumulative_time,
        exit_position=first_shell_exit_pos,
        detector_hit_position=None,
        reached_detector=False,
        energy_after_shell=energy_after_shell,
        energy_after_channel=energy_after_channel,
        status="missed_detector",
        final_position=final_pos.copy(),
        final_direction=final_dir.copy(),
        trajectory_points=formatted_trajectory
    )
    return ("missed_detector", record)


def run_simulation(
    n_neutrons: int,
    shell_thickness: float,
    aluminium_mass_ratio: float,
    aluminium_mfp_data: np.ndarray = MFP_DATA_AL,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
    channel_geometry: Optional[MeshGeometry] = None,
    channel_mfp_data: np.ndarray = MFP_DATA_PE,
    channel_mass_ratio: float = 1.0,
    source_cone_axis: Optional[np.ndarray] = None,
    source_cone_half_angle_deg: float = DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG,
    detector_plane: Optional[DetectorPlane] = None,
    h_mfp_data: Optional[np.ndarray] = None,
    c_mfp_data: Optional[np.ndarray] = None,
) -> List[NeutronRecord]:
    """Simulate multiple neutrons and return their complete records.

    Parameters
    ----------
    n_neutrons : int
        Number of neutron histories to simulate.

    Returns
    -------
    list of NeutronRecord
        A list containing the complete record for each neutron.
    """
    records: List[NeutronRecord] = []
    
    lost_in_shell = 0
    lost_in_channel = 0
    missed_detector = 0
    sample_miss_records = []
    
    for _ in range(n_neutrons):
        status, data = simulate_neutron_history(
            shell_thickness,
            aluminium_mfp_data,
            aluminium_mass_ratio,
            detector_distance,
            detector_side,
            energy_cutoff_mev,
            shell_geometry=shell_geometry,
            channel_geometry=channel_geometry,
            channel_mfp_data=channel_mfp_data,
            channel_mass_ratio=channel_mass_ratio,
            source_cone_axis=source_cone_axis,
            source_cone_half_angle_deg=source_cone_half_angle_deg,
            detector_plane=detector_plane,
            h_mfp_data=h_mfp_data,
            c_mfp_data=c_mfp_data,
        )
        
        if data is not None:
            records.append(data)
        
        if status == "lost_in_shell":
            lost_in_shell += 1
        elif status == "lost_in_channel":
            lost_in_channel += 1
        elif status == "missed_detector":
            missed_detector += 1
            if len(sample_miss_records) < 10 and data is not None:
                sample_miss_records.append(data)
    
    success_count = sum(1 for r in records if r.reached_detector)
    success_rate = success_count / n_neutrons * 100 if n_neutrons > 0 else 0
    print(f"[debug] Neutron fate statistics:")
    print(f"  - Reached detector: {success_count} ({success_rate:.2f}%)")
    print(f"  - Lost in aluminum shell: {lost_in_shell} ({lost_in_shell/n_neutrons*100:.2f}%)")
    print(f"  - Lost in polyethylene channel: {lost_in_channel} ({lost_in_channel/n_neutrons*100:.2f}%)")
    print(f"  - Missed detector: {missed_detector} ({missed_detector/n_neutrons*100:.2f}%)")
    
    if sample_miss_records and detector_plane is not None:
        print(f"\n[debug] Sample neutrons that missed detector (first 10):")
        for i, rec in enumerate(sample_miss_records):
            print(f"  #{i+1}:")
            print(f"    Final position: {rec.final_position}")
            print(f"    Final direction: {rec.final_direction}")
            print(f"    Energy: {rec.energy_after_channel:.4f} MeV")
            if rec.final_position is not None and rec.final_direction is not None:
                to_detector = detector_plane.center - rec.final_position
                dot_product = np.dot(rec.final_direction, to_detector / np.linalg.norm(to_detector))
                print(f"    Direction toward detector (dot product): {dot_product:.4f}")
                dist_to_center = np.linalg.norm(to_detector)
                print(f"    Distance to detector center: {dist_to_center:.4f} m")
    
    return records
