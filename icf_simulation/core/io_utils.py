"""
Data import/export utilities for neutron simulation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List

from .data_classes import NeutronRecord


def export_neutron_records_to_csv(records: List[NeutronRecord], filename: str = "neutron_data.csv"):
    """Export neutron records to a CSV file.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of neutron records from simulation.
    filename : str
        Output CSV filename.
    """
    if not records:
        print("[warning] No neutron records to export.")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'neutron_id',
        'status',
        'reached_detector',
        'detector_hit_x_m',
        'detector_hit_y_m',
        'detector_hit_z_m',
        'velocity_magnitude_m_s',
        'velocity_x_m_s',
        'velocity_y_m_s',
        'velocity_z_m_s',
        'direction_x',
        'direction_y',
        'direction_z',
        'final_energy_MeV',
        'initial_energy_MeV',
        'energy_after_shell_MeV',
        'energy_after_channel_MeV',
        'total_flight_time_s',
        'total_flight_time_ns',
        'final_position_x_m',
        'final_position_y_m',
        'final_position_z_m',
        'exit_position_x_m',
        'exit_position_y_m',
        'exit_position_z_m'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for idx, record in enumerate(records, start=1):
            neutron_mass_kg = 1.674927498e-27
            energy_joules = record.final_energy * 1.602176634e-13
            velocity_magnitude = math.sqrt(2 * energy_joules / neutron_mass_kg) if record.final_energy > 0 else 0.0
            
            if record.final_direction is not None:
                velocity_x = velocity_magnitude * record.final_direction[0]
                velocity_y = velocity_magnitude * record.final_direction[1]
                velocity_z = velocity_magnitude * record.final_direction[2]
                dir_x, dir_y, dir_z = record.final_direction[0], record.final_direction[1], record.final_direction[2]
            else:
                velocity_x = velocity_y = velocity_z = 0.0
                dir_x = dir_y = dir_z = 0.0
            
            if record.detector_hit_position is not None:
                hit_x, hit_y, hit_z = record.detector_hit_position[0], record.detector_hit_position[1], record.detector_hit_position[2]
            else:
                hit_x = hit_y = hit_z = None
            
            if record.final_position is not None:
                final_x, final_y, final_z = record.final_position[0], record.final_position[1], record.final_position[2]
            else:
                final_x = final_y = final_z = None
            
            exit_x, exit_y, exit_z = record.exit_position[0], record.exit_position[1], record.exit_position[2]
            
            tof_ns = record.tof * 1e9
            
            row = [
                idx,
                record.status,
                record.reached_detector,
                hit_x,
                hit_y,
                hit_z,
                velocity_magnitude,
                velocity_x,
                velocity_y,
                velocity_z,
                dir_x,
                dir_y,
                dir_z,
                record.final_energy,
                record.initial_energy,
                record.energy_after_shell,
                record.energy_after_channel,
                record.tof,
                tof_ns,
                final_x,
                final_y,
                final_z,
                exit_x,
                exit_y,
                exit_z
            ]
            writer.writerow(row)
    
    print(f"[info] Neutron data exported to {filename}")
    print(f"[info] Total records: {len(records)}")
    print(f"[info] Records that reached detector: {sum(1 for r in records if r.reached_detector)}")


def export_neutron_trajectories_to_csv(records: List[NeutronRecord], filename: str = "neutron_trajectories.csv"):
    """Export neutron trajectory data to a CSV file.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of neutron records from simulation.
    filename : str
        Output CSV filename.
    """
    if not records:
        print("[warning] No neutron records to export.")
        return
    
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'neutron_id',
        'step_id',
        'event_type',
        'position_x_m',
        'position_y_m',
        'position_z_m',
        'energy_MeV'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for neutron_id, record in enumerate(records, start=1):
            if record.trajectory_points is None or len(record.trajectory_points) == 0:
                writer.writerow([neutron_id, 0, "source", 0.0, 0.0, 0.0, record.initial_energy])
                if record.final_position is not None:
                    writer.writerow([
                        neutron_id, 1, "final", 
                        record.final_position[0], record.final_position[1], record.final_position[2],
                        record.final_energy
                    ])
            else:
                for step_id, (position, energy, event_type) in enumerate(record.trajectory_points):
                    writer.writerow([
                        neutron_id,
                        step_id,
                        event_type,
                        position[0],
                        position[1],
                        position[2],
                        energy
                    ])
    
    print(f"[info] Neutron trajectories exported to {filename}")
    print(f"[info] Total neutrons: {len(records)}")
    total_points = sum(len(r.trajectory_points) if r.trajectory_points else 0 for r in records)
    print(f"[info] Total trajectory points: {total_points}")
