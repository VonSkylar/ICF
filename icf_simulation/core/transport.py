"""
Particle transport through materials.

This module contains functions for simulating neutron transport through
various materials including aluminum shells and polyethylene channels.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .constants import DEBUG
from .data_classes import MeshGeometry, DetectorPlane
from .geometry import ray_mesh_intersection, find_exit_with_retry
from .sampling import sample_isotropic_direction
from .kinematics import energy_to_speed, scatter_neutron_elastic_cms_to_lab
from .cross_section import get_mfp_energy_dependent, get_macro_sigma_at_energy


class MaterialType(Enum):
    """Enum for material types in the simulation."""
    VACUUM = 0
    ALUMINIUM = 1
    POLYETHYLENE = 2


@dataclass
class MaterialConfig:
    """Configuration for a material in the simulation."""
    material_type: MaterialType
    geometry: Optional[MeshGeometry]
    mfp_data: np.ndarray
    mass_ratio: float
    # For PE with H/C separation
    h_mfp_data: Optional[np.ndarray] = None
    c_mfp_data: Optional[np.ndarray] = None


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


def unified_transport(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    shell_geometry: Optional[MeshGeometry],
    channel_geometry: Optional[MeshGeometry],
    aluminium_mfp_data: np.ndarray,
    channel_mfp_data: np.ndarray,
    aluminium_mass_ratio: float,
    channel_mass_ratio: float,
    energy_cutoff_mev: float,
    detector_plane: Optional[DetectorPlane] = None,
    h_mfp_data: Optional[np.ndarray] = None,
    c_mfp_data: Optional[np.ndarray] = None,
    max_segments: int = 100,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float, str]]]:
    """Unified transport through multiple materials (Al shell + PE channel).
    
    This function handles neutron transport through overlapping geometries by:
    1. Finding the nearest intersection with ANY material from current position
    2. Transporting through that material until exit
    3. Repeating until detector is reached or energy falls below cutoff
    
    Parameters
    ----------
    position : np.ndarray
        Starting position (usually origin for source neutrons).
    direction : np.ndarray
        Initial direction unit vector.
    energy_mev : float
        Initial neutron energy in MeV.
    shell_geometry : MeshGeometry or None
        Aluminum shell mesh geometry.
    channel_geometry : MeshGeometry or None
        Polyethylene channel mesh geometry.
    aluminium_mfp_data : np.ndarray
        MFP data for aluminum.
    channel_mfp_data : np.ndarray
        MFP data for polyethylene.
    aluminium_mass_ratio : float
        Mass ratio A for aluminum scattering.
    channel_mass_ratio : float
        Mass ratio A for polyethylene scattering.
    energy_cutoff_mev : float
        Energy threshold below which transport stops.
    detector_plane : DetectorPlane, optional
        Detector geometry for hit detection.
    h_mfp_data : np.ndarray, optional
        Separate H cross-section data for nuclide sampling.
    c_mfp_data : np.ndarray, optional
        Separate C cross-section data for nuclide sampling.
    max_segments : int
        Maximum number of material segments to traverse.
    
    Returns
    -------
    tuple
        (cumulative_time, final_energy, final_position, final_direction, 
         detector_crossing, trajectory_points)
    """
    current_pos = np.array(position, dtype=float)
    current_dir = np.array(direction, dtype=float)
    current_dir /= np.linalg.norm(current_dir)
    current_energy = energy_mev
    cumulative_time = 0.0
    detector_crossing = None
    trajectory_points: List[Tuple[np.ndarray, float, str]] = []
    
    trajectory_points.append((current_pos.copy(), current_energy, "start"))
    
    max_collisions_total = 5000
    total_collisions = 0
    
    for segment_idx in range(max_segments):
        if current_energy <= energy_cutoff_mev:
            break
        
        # Find nearest intersection with either geometry
        shell_hit = None
        channel_hit = None
        
        if shell_geometry is not None:
            shell_hit = ray_mesh_intersection(current_pos, current_dir, shell_geometry)
        
        if channel_geometry is not None:
            channel_hit = ray_mesh_intersection(current_pos, current_dir, channel_geometry)
        
        # Determine which material is hit first (if any)
        next_hit = None
        hit_material = MaterialType.VACUUM
        active_geometry = None
        
        if shell_hit is not None and channel_hit is not None:
            if shell_hit[0] < channel_hit[0]:
                next_hit = shell_hit
                hit_material = MaterialType.ALUMINIUM
                active_geometry = shell_geometry
            else:
                next_hit = channel_hit
                hit_material = MaterialType.POLYETHYLENE
                active_geometry = channel_geometry
        elif shell_hit is not None:
            next_hit = shell_hit
            hit_material = MaterialType.ALUMINIUM
            active_geometry = shell_geometry
        elif channel_hit is not None:
            next_hit = channel_hit
            hit_material = MaterialType.POLYETHYLENE
            active_geometry = channel_geometry
        
        # Check for detector crossing before any material hit
        if detector_plane is not None and detector_crossing is None:
            detector_hit = _check_detector_crossing(
                current_pos, current_dir, current_energy, detector_plane,
                max_distance=next_hit[0] if next_hit else 1e6
            )
            if detector_hit is not None:
                crossing_dist, crossing_point = detector_hit
                speed = energy_to_speed(current_energy)
                crossing_time = cumulative_time + crossing_dist / speed
                detector_crossing = (crossing_time, crossing_point, current_energy)
                trajectory_points.append((crossing_point.copy(), current_energy, "detector_hit"))
                return cumulative_time + crossing_dist / speed, current_energy, crossing_point, current_dir, detector_crossing, trajectory_points
        
        if next_hit is None:
            # No more material intersections - propagate to detector or escape
            if detector_plane is not None:
                flight_time, hit_point = propagate_to_scintillator(
                    current_pos, current_dir, current_energy, detector_plane, energy_cutoff_mev
                )
                if flight_time is not None and hit_point is not None:
                    detector_crossing = (cumulative_time + flight_time, hit_point, current_energy)
                    trajectory_points.append((hit_point.copy(), current_energy, "detector_hit"))
                    return cumulative_time + flight_time, current_energy, hit_point, current_dir, detector_crossing, trajectory_points
            
            # Escape without hitting detector
            escape_dist = 5.0  # 5 meters
            final_pos = current_pos + current_dir * escape_dist
            trajectory_points.append((final_pos.copy(), current_energy, "escape"))
            speed = energy_to_speed(current_energy)
            cumulative_time += escape_dist / speed
            return cumulative_time, current_energy, final_pos, current_dir, detector_crossing, trajectory_points
        
        # Move to entry point of the material
        entry_dist, entry_point, entry_normal = next_hit
        speed = energy_to_speed(current_energy)
        cumulative_time += entry_dist / speed
        current_pos = entry_point.copy()
        
        material_name = "Al" if hit_material == MaterialType.ALUMINIUM else "PE"
        trajectory_points.append((current_pos.copy(), current_energy, f"enter_{material_name}"))
        
        # Transport through this material segment
        if hit_material == MaterialType.ALUMINIUM:
            mfp_data = aluminium_mfp_data
            mass_ratio = aluminium_mass_ratio
            use_nuclide_sampling = False
        else:  # POLYETHYLENE
            mfp_data = channel_mfp_data
            mass_ratio = channel_mass_ratio
            use_nuclide_sampling = (h_mfp_data is not None and c_mfp_data is not None)
        
        # Scatter within this material until exiting
        max_collisions_per_segment = 500
        collision_count = 0
        
        while current_energy > energy_cutoff_mev and collision_count < max_collisions_per_segment:
            if total_collisions >= max_collisions_total:
                trajectory_points.append((current_pos.copy(), current_energy, "max_collisions"))
                return cumulative_time, energy_cutoff_mev, current_pos, current_dir, detector_crossing, trajectory_points
            
            # Calculate mean free path
            if use_nuclide_sampling and hit_material == MaterialType.POLYETHYLENE:
                sigma_H = get_macro_sigma_at_energy(current_energy, h_mfp_data)
                sigma_C = get_macro_sigma_at_energy(current_energy, c_mfp_data)
                sigma_total = sigma_H + sigma_C
                mean_free_path = 1.0 / sigma_total if sigma_total > 1e-12 else 1e12
            else:
                mean_free_path = get_mfp_energy_dependent(current_energy, mfp_data)
            
            # Sample free path
            free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
            
            # Find exit from current material
            exit_hit = find_exit_with_retry(current_pos, current_dir, active_geometry)
            
            if exit_hit is None:
                # Geometry leak - try to recover by moving slightly forward
                if DEBUG:
                    print(f"[WARNING] Geometry leak in {material_name}, attempting recovery")
                current_pos += current_dir * 1e-6
                break
            
            distance_to_boundary = exit_hit[0]
            
            # Check for detector crossing within this segment
            if detector_plane is not None and detector_crossing is None:
                step_dist = min(free_path, distance_to_boundary)
                detector_hit = _check_detector_crossing(
                    current_pos, current_dir, current_energy, detector_plane,
                    max_distance=step_dist
                )
                if detector_hit is not None:
                    crossing_dist, crossing_point = detector_hit
                    speed = energy_to_speed(current_energy)
                    crossing_time = cumulative_time + crossing_dist / speed
                    detector_crossing = (crossing_time, crossing_point, current_energy)
                    trajectory_points.append((crossing_point.copy(), current_energy, "detector_hit"))
                    return cumulative_time + crossing_dist / speed, current_energy, crossing_point, current_dir, detector_crossing, trajectory_points
            
            if free_path < distance_to_boundary:
                # Collision occurs within material
                current_pos += free_path * current_dir
                speed = energy_to_speed(current_energy)
                cumulative_time += free_path / speed
                
                # Determine target nucleus for scattering
                if use_nuclide_sampling and hit_material == MaterialType.POLYETHYLENE:
                    if np.random.rand() < (sigma_H / sigma_total):
                        target_A = 1.0  # Hydrogen
                    else:
                        target_A = 12.011  # Carbon
                else:
                    target_A = mass_ratio
                
                # Perform scattering
                current_energy, current_dir = scatter_neutron_elastic_cms_to_lab(
                    current_energy, current_dir, target_A
                )
                
                trajectory_points.append((current_pos.copy(), current_energy, f"scatter_{material_name}"))
                collision_count += 1
                total_collisions += 1
                
            else:
                # Exit material without collision
                current_pos += distance_to_boundary * current_dir
                speed = energy_to_speed(current_energy)
                cumulative_time += distance_to_boundary / speed
                
                # Small offset to ensure we're outside
                current_pos += current_dir * 1e-9
                
                trajectory_points.append((current_pos.copy(), current_energy, f"exit_{material_name}"))
                break
        
        if collision_count >= max_collisions_per_segment:
            if DEBUG:
                print(f"[WARNING] Max collisions reached in {material_name}")
    
    return cumulative_time, current_energy, current_pos, current_dir, detector_crossing, trajectory_points


def _check_detector_crossing(
    position: np.ndarray,
    direction: np.ndarray,
    energy: float,
    detector_plane: DetectorPlane,
    max_distance: float = 1e6,
) -> Optional[Tuple[float, np.ndarray]]:
    """Check if ray crosses detector plane within max_distance.
    
    Returns
    -------
    tuple of (distance, crossing_point) or None
    """
    axis = detector_plane.axis
    center = detector_plane.center
    
    if center is None:
        return None
    
    # Calculate distance to detector plane
    pos_to_center = center - position
    denominator = np.dot(direction, axis)
    
    if abs(denominator) < 1e-10:
        return None  # Ray parallel to plane
    
    distance = np.dot(pos_to_center, axis) / denominator
    
    if distance <= 0 or distance > max_distance:
        return None
    
    crossing_point = position + direction * distance
    
    # Check if within detector radius
    if detector_plane.is_circular and detector_plane.radius is not None:
        dist_from_center = np.linalg.norm(crossing_point - center)
        if dist_from_center > detector_plane.radius:
            return None
    
    return distance, crossing_point
