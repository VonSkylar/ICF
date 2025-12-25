"""
Particle transport through materials.

This module contains functions for simulating neutron transport through
various materials including aluminum shells and polyethylene channels.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .constants import DEBUG
from .data_classes import MeshGeometry, DetectorPlane
from .geometry import ray_mesh_intersection, find_exit_with_retry
from .sampling import sample_isotropic_direction
from .kinematics import energy_to_speed, scatter_neutron_elastic_cms_to_lab
from .cross_section import get_mfp_energy_dependent, get_macro_sigma_at_energy


def transport_through_slab(
    energy_mev: float,
    slab_thickness: float,
    mfp_data: np.ndarray,
    target_mass_ratio: float,
    energy_cutoff_mev: float,
    initial_direction: Optional[np.ndarray] = None,
    start_position: Optional[np.ndarray] = None,
    detector_plane: Optional[DetectorPlane] = None,
    collect_trajectory: bool = False,
) -> Tuple[float, float, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """Propagate a neutron through a homogeneous slab with multiple scatterings.
    
    NOTE: This function uses an "equivalent optical path" approximation.
    For complex 3D geometries, use propagate_through_mesh_material().
    
    Returns
    -------
    tuple : (cumulative_time, energy, current_dir, detector_crossing, trajectory_points)
    """
    remaining = slab_thickness
    energy = float(energy_mev)
    cumulative_time = 0.0
    detector_crossing = None
    trajectory_points = []

    if initial_direction is not None:
        current_dir = np.array(initial_direction, dtype=float)
        norm = np.linalg.norm(current_dir)
        if norm == 0.0:
            current_dir = sample_isotropic_direction()
        else:
            current_dir /= norm
    else:
        current_dir = sample_isotropic_direction()

    speed = energy_to_speed(energy)

    track_position = (start_position is not None and detector_plane is not None)
    if track_position:
        current_pos = start_position.copy()
        if collect_trajectory:
            trajectory_points.append((current_pos.copy(), energy))
        axis = detector_plane.axis
        center = detector_plane.center
        radius = detector_plane.radius
    else:
        current_pos = start_position.copy() if start_position is not None else np.zeros(3)
        if collect_trajectory:
            trajectory_points.append((current_pos.copy(), energy))
        if detector_plane is not None:
            radius = detector_plane.radius
            axis = detector_plane.axis
            center = detector_plane.center

    mean_free_path = get_mfp_energy_dependent(energy, mfp_data)
    
    if mean_free_path > 100.0 * slab_thickness:
        cumulative_time = slab_thickness / speed
        
        if track_position:
            end_pos = current_pos + slab_thickness * current_dir
            pos_value = np.dot(axis, current_pos - center)
            end_value = np.dot(axis, end_pos - center)
            
            if pos_value * end_value < 0:
                path_vec = end_pos - current_pos
                denominator = np.dot(axis, path_vec)
                if abs(denominator) > 1e-10:
                    s = -pos_value / denominator
                    if 0 <= s <= 1:
                        crossing_point = current_pos + s * path_vec
                        to_crossing = crossing_point - center
                        axial_component = np.dot(to_crossing, axis) * axis
                        radial_vector = to_crossing - axial_component
                        dist_from_center = np.linalg.norm(radial_vector)
                        
                        if dist_from_center <= radius:
                            dist_to_crossing = np.linalg.norm(crossing_point - current_pos)
                            time_to_crossing = dist_to_crossing / speed
                            detector_crossing = (time_to_crossing, crossing_point, energy)
        
        return cumulative_time, energy, current_dir, detector_crossing, trajectory_points

    while energy > energy_cutoff_mev and remaining > 0.0:
        free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
        step = min(free_path, remaining)

        if track_position and detector_crossing is None:
            step_pos = current_pos + step * current_dir
            pos_value = np.dot(axis, current_pos - center)
            step_value = np.dot(axis, step_pos - center)
            
            if pos_value * step_value < 0:
                path_vec = step_pos - current_pos
                denominator = np.dot(axis, path_vec)
                if abs(denominator) > 1e-10:
                    s = -pos_value / denominator
                    if 0 <= s <= 1:
                        crossing_point = current_pos + s * path_vec
                        to_crossing = crossing_point - center
                        axial_component = np.dot(to_crossing, axis) * axis
                        radial_vector = to_crossing - axial_component
                        dist_from_center = np.linalg.norm(radial_vector)
                        
                        if dist_from_center <= radius:
                            dist_to_crossing = np.linalg.norm(crossing_point - current_pos)
                            time_to_crossing = dist_to_crossing / speed
                            detector_crossing = (cumulative_time + time_to_crossing, crossing_point, energy)
                            return cumulative_time + time_to_crossing, energy, current_dir, detector_crossing, trajectory_points
        
        if track_position:
            current_pos += step * current_dir
        else:
            current_pos += step * current_dir
        
        cumulative_time += step / speed
        remaining -= step

        if remaining <= 0.0:
            break
            
        if step >= free_path:
            energy, current_dir = scatter_neutron_elastic_cms_to_lab(
                energy, current_dir, target_mass_ratio
            )
            speed = energy_to_speed(energy)
            mean_free_path = get_mfp_energy_dependent(energy, mfp_data)
            
            if collect_trajectory:
                trajectory_points.append((current_pos.copy(), energy))
    
    return cumulative_time, energy, current_dir, detector_crossing, trajectory_points


def propagate_through_mesh_material(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    geometry: Optional[MeshGeometry],
    mfp_data: np.ndarray,
    target_mass_ratio: float,
    energy_cutoff_mev: float,
    max_segments: int = 20,
    detector_plane: Optional[DetectorPlane] = None,
    h_mfp_data: Optional[np.ndarray] = None,
    c_mfp_data: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """Advance a neutron through a mesh-defined solid material with strict 3D geometry tracking.
    
    This function implements rigorous 3D Monte Carlo transport.
    
    Returns
    -------
    tuple
        (cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points)
    """
    trajectory_points = []
    
    if geometry is None:
        return 0.0, energy_mev, np.array(position, dtype=float), np.array(direction, dtype=float), None, trajectory_points

    current_pos = np.array(position, dtype=float)
    dir_norm = np.array(direction, dtype=float)
    dir_norm /= np.linalg.norm(dir_norm)
    current_energy = energy_mev
    cumulative_time = 0.0
    detector_crossing = None
    max_collisions_per_segment = 1000
    
    for segment_idx in range(max_segments):
        hit = ray_mesh_intersection(current_pos, dir_norm, geometry)
        if hit is None:
            break
        
        entry_dist, entry_point, normal = hit
        
        speed = energy_to_speed(current_energy)
        vacuum_time = entry_dist / speed
        cumulative_time += vacuum_time
        
        current_pos = entry_point.copy()
        trajectory_points.append((current_pos.copy(), current_energy))
        
        collision_count = 0
        
        while current_energy > energy_cutoff_mev and collision_count < max_collisions_per_segment:
            if h_mfp_data is not None and c_mfp_data is not None:
                sigma_H_current = get_macro_sigma_at_energy(current_energy, h_mfp_data)
                sigma_C_current = get_macro_sigma_at_energy(current_energy, c_mfp_data)
                sigma_total_current = sigma_H_current + sigma_C_current
                mean_free_path = 1.0 / sigma_total_current if sigma_total_current > 1e-12 else 1e12
            else:
                mean_free_path = get_mfp_energy_dependent(current_energy, mfp_data)
            
            free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
            
            exit_hit = find_exit_with_retry(current_pos, dir_norm, geometry)
            
            if exit_hit is None:
                if DEBUG:
                    print(f"[CRITICAL] Geometry leak in propagate_through_mesh_material")
                return cumulative_time, energy_cutoff_mev, current_pos, dir_norm, detector_crossing, trajectory_points
            
            distance_to_boundary = exit_hit[0]
            
            if free_path < distance_to_boundary:
                step = free_path
                current_pos += step * dir_norm
                speed = energy_to_speed(current_energy)
                cumulative_time += step / speed
                
                if h_mfp_data is not None and c_mfp_data is not None:
                    if np.random.rand() < (sigma_H_current / sigma_total_current):
                        target_A = 1.0
                    else:
                        target_A = 12.011
                else:
                    target_A = target_mass_ratio
                
                current_energy, dir_norm = scatter_neutron_elastic_cms_to_lab(
                    current_energy, dir_norm, target_A
                )
                
                trajectory_points.append((current_pos.copy(), current_energy))
                collision_count += 1
                
            else:
                step = distance_to_boundary
                current_pos += step * dir_norm
                speed = energy_to_speed(current_energy)
                cumulative_time += step / speed
                break
        
        if collision_count >= max_collisions_per_segment:
            return cumulative_time, energy_cutoff_mev, current_pos, dir_norm, detector_crossing, trajectory_points
        
        if current_energy <= energy_cutoff_mev:
            return cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points
    
    return cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points


def simulate_in_aluminium(
    direction: np.ndarray,
    energy_mev: float,
    shell_thickness: float,
    aluminium_mfp_data: np.ndarray,
    target_mass_ratio: float,
    energy_cutoff_mev: float = 0.1,
    mesh_geometry: Optional[MeshGeometry] = None,
    detector_plane: Optional[DetectorPlane] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """Simulate neutron transport and scattering in an aluminium shell.
    
    Returns
    -------
    tuple
        (time, energy, direction, position, detector_crossing, trajectory_points)
    """
    direction = np.array(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("Direction vector must be non-zero")
    direction = direction / norm

    energy = float(energy_mev)
    speed = energy_to_speed(energy)
    cumulative_time = 0.0
    detector_crossing = None
    trajectory_points = []

    origin = np.zeros(3, dtype=float)

    if mesh_geometry is None:
        slab_time, energy, current_dir, material_crossing, slab_trajectory = transport_through_slab(
            energy,
            shell_thickness,
            aluminium_mfp_data,
            target_mass_ratio,
            energy_cutoff_mev,
            initial_direction=direction,
            start_position=origin,
            detector_plane=detector_plane,
            collect_trajectory=True,
        )
        cumulative_time += slab_time
        position = origin + direction * shell_thickness
        if material_crossing is not None:
            detector_crossing = (material_crossing[0], material_crossing[1], material_crossing[2])
        trajectory_points.extend(slab_trajectory)
        return cumulative_time, energy, current_dir, position, detector_crossing, trajectory_points

    hit_entry = ray_mesh_intersection(origin, direction, mesh_geometry)
    if hit_entry is None:
        typical_shell_radius = 1.2
        exit_position = origin + direction * typical_shell_radius
        vacuum_time = typical_shell_radius / speed
        cumulative_time += vacuum_time
        
        if detector_plane is not None and detector_plane.is_circular:
            center = detector_plane.center
            axis = detector_plane.axis
            pos_value = np.dot(axis, origin - center)
            exit_value = np.dot(axis, exit_position - center)
            
            if pos_value * exit_value < 0:
                path_vec = exit_position - origin
                denominator = np.dot(axis, path_vec)
                if abs(denominator) > 1e-10:
                    s = -pos_value / denominator
                    if 0 <= s <= 1:
                        crossing_point = origin + s * path_vec
                        to_crossing = crossing_point - center
                        axial_component = np.dot(to_crossing, axis) * axis
                        radial_vector = to_crossing - axial_component
                        dist_from_center = np.linalg.norm(radial_vector)
                        if dist_from_center <= detector_plane.radius:
                            dist_to_crossing = np.linalg.norm(crossing_point - origin)
                            time_to_crossing = dist_to_crossing / speed
                            detector_crossing = (time_to_crossing, crossing_point, energy)
        return cumulative_time, energy, direction, exit_position, detector_crossing, trajectory_points

    distance_to_entry, entry_point, entry_normal = hit_entry
    
    if detector_plane is not None and detector_plane.is_circular:
        center = detector_plane.center
        axis = detector_plane.axis
        pos_value = np.dot(axis, origin - center)
        entry_value = np.dot(axis, entry_point - center)
        
        if pos_value * entry_value < 0:
            path_vec = entry_point - origin
            denominator = np.dot(axis, path_vec)
            if abs(denominator) > 1e-10:
                s = -pos_value / denominator
                if 0 <= s <= 1:
                    crossing_point = origin + s * path_vec
                    to_crossing = crossing_point - center
                    axial_component = np.dot(to_crossing, axis) * axis
                    radial_vector = to_crossing - axial_component
                    dist_from_center = np.linalg.norm(radial_vector)
                    if dist_from_center <= detector_plane.radius:
                        dist_to_crossing = np.linalg.norm(crossing_point - origin)
                        time_to_crossing = dist_to_crossing / speed
                        detector_crossing = (time_to_crossing, crossing_point, energy)
    
    cumulative_time += distance_to_entry / speed
    current_pos = entry_point.copy()
    current_dir = direction.copy()
    current_energy = energy
    trajectory_points.append((current_pos.copy(), current_energy))
    
    max_collisions_in_shell = 1000
    collision_count = 0
    
    while current_energy > energy_cutoff_mev and collision_count < max_collisions_in_shell:
        mean_free_path = get_mfp_energy_dependent(current_energy, aluminium_mfp_data)
        free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
        
        exit_hit = find_exit_with_retry(current_pos, current_dir, mesh_geometry)
        
        if exit_hit is None:
            if DEBUG:
                print(f"[CRITICAL] Geometry leak in propagate_shell")
            return cumulative_time, energy_cutoff_mev, current_dir, current_pos, detector_crossing, trajectory_points
        
        distance_to_boundary = exit_hit[0]
        
        if free_path < distance_to_boundary:
            step = free_path
            current_pos += step * current_dir
            speed = energy_to_speed(current_energy)
            cumulative_time += step / speed
            
            current_energy, current_dir = scatter_neutron_elastic_cms_to_lab(
                current_energy, current_dir, target_mass_ratio
            )
            
            trajectory_points.append((current_pos.copy(), current_energy))
            collision_count += 1
            
        else:
            step = distance_to_boundary
            current_pos += step * current_dir
            speed = energy_to_speed(current_energy)
            cumulative_time += step / speed
            break
    
    if collision_count >= max_collisions_in_shell:
        return cumulative_time, energy_cutoff_mev, current_dir, current_pos, detector_crossing, trajectory_points
    
    return cumulative_time, current_energy, current_dir, current_pos, detector_crossing, trajectory_points


def propagate_to_scintillator(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    detector_plane: DetectorPlane,
    energy_cutoff_mev: float = 0.1,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Propagate a neutron from the shell exit to the scintillator.

    Returns
    -------
    tuple of (flight_time, hit_point) or (None, None)
    """
    pos = np.array(position, dtype=float)
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return None, None
    d = d / norm

    axis = detector_plane.axis
    dir_dot = float(np.dot(d, axis))

    if energy_mev <= energy_cutoff_mev:
        return None, None

    speed = energy_to_speed(energy_mev)
    u = detector_plane.u
    v = detector_plane.v
    plane_pos = detector_plane.plane_position

    if detector_plane.is_circular and detector_plane.center is not None:
        center = detector_plane.center
        numerator = float(np.dot(axis, center - pos))
        
        if abs(dir_dot) < 1e-10:
            return None, None
        
        t_total = numerator / dir_dot
        
        if t_total <= 0.0:
            return None, None
        
        if t_total > 1e6:
            return None, None
        
        hit = pos + d * t_total
        
        distance_from_center = np.linalg.norm(hit - center)
        if distance_from_center > detector_plane.radius:
            return None, None
        
        flight_time = t_total / speed
        return flight_time, hit
    
    else:
        t_total = (plane_pos - float(np.dot(pos, axis))) / dir_dot
        if t_total <= 0.0:
            return None, None

        hit = pos + d * t_total
        
        u_coord = float(np.dot(hit, u))
        v_coord = float(np.dot(hit, v))
        if abs(u_coord) > detector_plane.half_u or abs(v_coord) > detector_plane.half_v:
            return None, None

        flight_time = t_total / speed
        return flight_time, hit
