"""
High-level simulation driver functions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .constants import DEBUG, DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG
from .data_classes import MeshGeometry, DetectorPlane, NeutronRecord
from .geometry import build_default_detector_plane
from .sampling import sample_neutron_energy, sample_isotropic_direction, sample_direction_in_cone
from .transport import simulate_in_aluminium, propagate_through_mesh_material, propagate_to_scintillator
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

    trajectory_points = []
    trajectory_points.append((np.zeros(3), E0, "source"))

    # 2. Transport through the aluminium shell
    detector_plane = detector_plane or build_default_detector_plane(detector_distance, detector_side)

    t_shell, E_after_shell, d_after_shell, pos_after_shell, shell_detector_crossing, shell_trajectory = simulate_in_aluminium(
        d0,
        E0,
        shell_thickness,
        aluminium_mfp_data,
        aluminium_mass_ratio,
        energy_cutoff_mev,
        mesh_geometry=shell_geometry,
        detector_plane=detector_plane,
    )
    
    for pos, energy in shell_trajectory:
        trajectory_points.append((pos, energy, "scatter"))
    
    if shell_detector_crossing is not None:
        crossing_time, crossing_point, crossing_energy = shell_detector_crossing
        trajectory_points.append((pos_after_shell.copy(), E_after_shell, "shell_exit"))
        trajectory_points.append((crossing_point.copy(), crossing_energy, "detector_hit"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=crossing_energy,
            tof=crossing_time,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=crossing_point.copy(),
            reached_detector=True,
            energy_after_shell=crossing_energy,
            energy_after_channel=crossing_energy,
            status="success",
            final_position=crossing_point.copy(),
            final_direction=d_after_shell.copy(),
            trajectory_points=trajectory_points
        )
        return ("success", record)
    
    trajectory_points.append((pos_after_shell.copy(), E_after_shell, "shell_exit"))
    
    if E_after_shell <= energy_cutoff_mev:
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=E_after_shell,
            tof=t_shell,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=None,
            reached_detector=False,
            energy_after_shell=E_after_shell,
            energy_after_channel=0.0,
            status="lost_in_shell",
            final_position=pos_after_shell.copy(),
            final_direction=d_after_shell.copy(),
            trajectory_points=trajectory_points
        )
        return ("lost_in_shell", record)

    # 3. Interaction with the polyethylene channel
    result = propagate_through_mesh_material(
        pos_after_shell,
        d_after_shell,
        E_after_shell,
        channel_geometry,
        channel_mfp_data,
        channel_mass_ratio,
        energy_cutoff_mev,
        detector_plane=detector_plane,
        h_mfp_data=h_mfp_data,
        c_mfp_data=c_mfp_data,
    )
    
    t_channel, E_after_channel, pos_after_channel, d_after_channel, detector_crossing, channel_trajectory = result
    
    for pos, energy in channel_trajectory:
        trajectory_points.append((pos, energy, "scatter"))
    
    trajectory_points.append((pos_after_channel.copy(), E_after_channel, "channel_exit"))
    
    if E_after_channel <= energy_cutoff_mev:
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=E_after_channel,
            tof=t_shell + t_channel,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=None,
            reached_detector=False,
            energy_after_shell=E_after_shell,
            energy_after_channel=E_after_channel,
            status="lost_in_channel",
            final_position=pos_after_channel.copy(),
            final_direction=d_after_channel.copy(),
            trajectory_points=trajectory_points
        )
        return ("lost_in_channel", record)
    
    if detector_crossing is not None:
        crossing_time, crossing_point, crossing_energy = detector_crossing
        total_tof = t_shell + crossing_time
        
        trajectory_points.append((crossing_point.copy(), crossing_energy, "detector_hit"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=crossing_energy,
            tof=total_tof,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=crossing_point.copy(),
            reached_detector=True,
            energy_after_shell=E_after_shell,
            energy_after_channel=crossing_energy,
            status="success",
            final_position=crossing_point.copy(),
            final_direction=d_after_channel.copy(),
            trajectory_points=trajectory_points
        )
        return ("success", record)

    # 4. Check final position relative to detector plane
    pos_proj = np.dot(detector_plane.axis, pos_after_channel - detector_plane.center)
    
    if pos_proj > 0:
        extension_distance = 1.0
        extended_point = pos_after_channel + d_after_channel * extension_distance
        trajectory_points.append((extended_point.copy(), E_after_channel, "miss_extended"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=E_after_channel,
            tof=t_shell + t_channel,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=None,
            reached_detector=False,
            energy_after_shell=E_after_shell,
            energy_after_channel=E_after_channel,
            status="missed_detector",
            final_position=pos_after_channel.copy(),
            final_direction=d_after_channel.copy(),
            trajectory_points=trajectory_points
        )
        return ("missed_detector", record)
    else:
        flight_time, hit_point = propagate_to_scintillator(
            pos_after_channel,
            d_after_channel,
            E_after_channel,
            detector_plane,
            energy_cutoff_mev=energy_cutoff_mev,
        )
        
        if flight_time is None:
            extension_distance = 1.5
            extended_point = pos_after_channel + d_after_channel * extension_distance
            trajectory_points.append((extended_point.copy(), E_after_channel, "miss_extended"))
            
            record = NeutronRecord(
                initial_energy=E0,
                final_energy=E_after_channel,
                tof=t_shell + t_channel,
                exit_position=pos_after_shell.copy(),
                detector_hit_position=None,
                reached_detector=False,
                energy_after_shell=E_after_shell,
                energy_after_channel=E_after_channel,
                status="missed_detector",
                final_position=pos_after_channel.copy(),
                final_direction=d_after_channel.copy(),
                trajectory_points=trajectory_points
            )
            return ("missed_detector", record)

        total_tof = t_shell + t_channel + flight_time
        
        if hit_point is not None:
            trajectory_points.append((hit_point.copy(), E_after_channel, "detector_hit"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=E_after_channel,
            tof=total_tof,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=hit_point.copy() if hit_point is not None else None,
            reached_detector=True,
            energy_after_shell=E_after_shell,
            energy_after_channel=E_after_channel,
            status="success",
            final_position=hit_point.copy() if hit_point is not None else None,
            final_direction=d_after_channel.copy(),
            trajectory_points=trajectory_points
        )
        
        return ("success", record)


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
