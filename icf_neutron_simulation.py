"""
icf_neutron_simulation
======================

This module provides a self-contained Monte-Carlo simulator for the
time-of-flight (TOF) behaviour of neutrons in an inertial confinement fusion
(ICF) diagnostic. It follows, as closely as practicable, the high level
algorithm outlined in the provided project document. Each physical stage of
the neutron's journey is encapsulated in its own function to aid testing and
reuse.

The basic workflow is as follows:

1. **Neutron generation** - Neutrons are produced at the centre of the
   implosion target. Their kinetic energy is sampled from a Gaussian  
   distribution centred on 2.45 MeV (corresponding to DD fusion) and their
   direction is sampled within a configurable cone (or isotropically if no 
   cone is specified).
2. **Transport through the aluminium shell** - The aluminium shell is
   described by an STL mesh. Neutrons undergo a sequence of elastic 
   scatterings in the shell. Free flight distances are drawn from an 
   exponential distribution with energy-dependent mean free path calculated 
   from real cross-section data (ENDF/JANIS). After each scattering the 
   neutron's direction is randomised and its energy is updated using a 
   two-body kinematic model. Once the neutron exits the shell geometry 
   or the energy drops below 0.1 MeV, it leaves the shell or is absorbed.
3. **Flight through the nTOF_without_scintillant channel** - After emerging from the shell, the
   neutron may intersect the polyethylene structures described by nTOF_without_scintillant.STL.
   If it does, Monte-Carlo collisions with the polyethylene nuclei (H and C) 
   are simulated using energy-dependent mean free paths calculated from real 
   cross-section data; otherwise the neutron travels in vacuum to the detector.
4. **Detection** - Neutrons that reach the detector plane and fall within the
   detector aperture (rectangular or circular) are recorded. The total 
   time-of-flight (TOF) from source to detector and the neutron's final 
   energy are stored for analysis. The detector geometry (position, size, 
   shape) is fully configurable.

The functions provided here are intentionally generic: key parameters such as
shell thickness, cross-section data, and material mass ratios are exposed as
arguments. Users can substitute real material data obtained from ENDF or 
JANIS databases without changing the overall structure of the code.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

# Debug flag
DEBUG = False

from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D


AVOGADRO_CONSTANT = 6.02214076e23 
BARN_TO_M2 = 1.0e-28

################################################################################
# STL handling
################################################################################

def load_stl_mesh(file_path: str) -> np.ndarray:
    """Load a mesh from an STL file.

    The STL format can be either ASCII or binary.  This routine attempts to
    detect the format automatically and returns an array of facets, where each
    facet is represented by three 3‑D vertices.  The ordering of the vertices
    follows the order in the file and is not otherwise interpreted.

    Parameters
    ----------
    file_path : str
        Path to the STL file on disk.

    Returns
    -------
    np.ndarray, shape (n_facets, 3, 3)
        An array containing all triangle vertices.  Each facet is of shape
        (3,3) and contains three vertices of (x, y, z) coordinates.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"STL file '{file_path}' does not exist")

    # Read the first 80 bytes to determine whether this is a binary STL.  A
    # binary STL has an 80 byte header followed by an unsigned 32 bit integer
    # specifying the number of triangles.  An ASCII STL normally starts with
    # the word 'solid'.  There are edge cases (e.g. ASCII STLs starting with
    # 'solid' but being binary) but this heuristic is sufficient for most
    # purposes.
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        header = f.read(80)
        count_bytes = f.read(4)
        triangle_count = int.from_bytes(count_bytes, byteorder='little', signed=False) if len(count_bytes) == 4 else 0
    header_str = header.decode(errors='ignore').strip().lower()
    expected_binary_size = 84 + triangle_count * 50 if triangle_count > 0 else None

    def _try_load_binary() -> Optional[np.ndarray]:
        if expected_binary_size is None or expected_binary_size > file_size:
            return None
        if expected_binary_size != file_size:
            # Size mismatch strongly suggests ASCII STL; skip binary parsing.
            return None
        facets_bin: List[np.ndarray] = []
        record_struct = struct.Struct('<12fH')  # normal (3), vertices (9), attribute (H)
        with open(file_path, 'rb') as bf:
            bf.seek(84)
            for _ in range(triangle_count):
                chunk = bf.read(record_struct.size)
                if len(chunk) != record_struct.size:
                    return None
                unpacked = record_struct.unpack(chunk)
                # Store normal and vertices together: [normal_x, normal_y, normal_z, v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z]
                normal = np.array(unpacked[0:3], dtype=float)
                v0 = np.array(unpacked[3:6], dtype=float)
                v1 = np.array(unpacked[6:9], dtype=float)
                v2 = np.array(unpacked[9:12], dtype=float)
                # Stack as [4 x 3]: [normal, v0, v1, v2]
                facets_bin.append(np.stack([normal, v0, v1, v2], axis=0))
        if not facets_bin:
            return None
        return np.stack(facets_bin, axis=0)

    def _try_load_ascii() -> Optional[np.ndarray]:
        facets_ascii: List[np.ndarray] = []
        current_vertices: List[np.ndarray] = []
        current_normal: Optional[np.ndarray] = None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as af:
            for line in af:
                tokens = line.strip().split()
                if not tokens:
                    continue
                keyword = tokens[0].lower()
                if keyword == 'facet' and len(tokens) >= 5 and tokens[1].lower() == 'normal':
                    # Parse normal vector
                    current_normal = np.array(list(map(float, tokens[2:5])), dtype=float)
                    current_vertices = []
                elif keyword == 'vertex' and len(tokens) >= 4:
                    v = np.array(list(map(float, tokens[1:4])), dtype=float)
                    current_vertices.append(v)
                elif keyword == 'endfacet':
                    if len(current_vertices) >= 3:
                        # If normal wasn't parsed, calculate it from vertices
                        if current_normal is None:
                            v0, v1, v2 = current_vertices[0], current_vertices[1], current_vertices[2]
                            edge1 = v1 - v0
                            edge2 = v2 - v0
                            current_normal = np.cross(edge1, edge2)
                            norm = np.linalg.norm(current_normal)
                            if norm > 0:
                                current_normal = current_normal / norm
                            else:
                                current_normal = np.array([0.0, 0.0, 1.0])
                        # Stack as [4 x 3]: [normal, v0, v1, v2]
                        facets_ascii.append(np.stack([current_normal] + current_vertices[:3], axis=0))
                    current_normal = None
                    current_vertices = []
        if not facets_ascii:
            return None
        return np.stack(facets_ascii, axis=0)

    facets = None
    binary_first = True
    if header_str.startswith('solid') and (expected_binary_size is None or expected_binary_size != file_size):
        binary_first = False

    loaders = (_try_load_binary, _try_load_ascii) if binary_first else (_try_load_ascii, _try_load_binary)
    for loader in loaders:
        facets = loader()
        if facets is not None:
            break

    if facets is None:
        raise ValueError(f"No facets were found in '{file_path}' - the file may be corrupt")
    return facets


@dataclass
class MeshGeometry:
    """Preprocessed data for fast(ish) ray intersections with an STL mesh."""

    vertices0: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray
    normals: np.ndarray


@dataclass
class DetectorPlane:
    """Planar detector geometry aligned with an arbitrary axis."""

    axis: np.ndarray
    u: np.ndarray
    v: np.ndarray
    plane_position: float
    half_u: float
    half_v: float
    center: Optional[np.ndarray] = None  # Center position in 3D space (m)
    radius: Optional[float] = None  # Radius for circular detector (m)
    is_circular: bool = False  # Whether detector is circular


@dataclass
class NeutronRecord:
    """Record of a neutron's journey through the simulation."""
    
    initial_energy: float  # MeV
    final_energy: float  # MeV
    tof: float  # seconds
    exit_position: np.ndarray  # 3D position after shell (m)
    detector_hit_position: Optional[np.ndarray]  # 3D position at detector (m)
    reached_detector: bool
    energy_after_shell: float = 0.0  # MeV, energy after aluminum shell
    energy_after_channel: float = 0.0  # MeV, energy after polyethylene channel
    status: str = "unknown"  # Status: success, lost_in_shell, lost_in_channel, missed_detector
    final_position: Optional[np.ndarray] = None  # Final position (m)
    final_direction: Optional[np.ndarray] = None  # Final direction
    trajectory_points: Optional[List[Tuple[np.ndarray, float]]] = None  # List of (position, energy) at each collision


################################################################################
# Simulation Configuration Constants
################################################################################

# Default source cone half-angle (degrees)
DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG = 30


################################################################################
# Mesh Geometry Processing
################################################################################

def prepare_mesh_geometry(mesh: np.ndarray) -> MeshGeometry:
    """Precompute edge vectors and normals for an STL mesh.
    
    Parameters
    ----------
    mesh : np.ndarray
        Mesh data with shape (n_facets, 4, 3) where each facet contains
        [normal, v0, v1, v2], or (n_facets, 3, 3) with just vertices.
    """
    triangles = np.asarray(mesh, dtype=float)
    
    # Check if mesh includes normals (shape is n x 4 x 3) or just vertices (n x 3 x 3)
    if triangles.shape[1] == 4:
        # Mesh includes normals: [normal, v0, v1, v2]
        stl_normals = triangles[:, 0, :]  # Use STL file normals
        v0 = triangles[:, 1, :]
        v1 = triangles[:, 2, :]
        v2 = triangles[:, 3, :]
    else:
        # Legacy format: just vertices [v0, v1, v2]
        v0 = triangles[:, 0, :]
        v1 = triangles[:, 1, :]
        v2 = triangles[:, 2, :]
        # Calculate normals from vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        stl_normals = np.cross(edge1, edge2)
        
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Normalize STL normals
    norms = np.linalg.norm(stl_normals, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    stl_normals = stl_normals / norms
    
    return MeshGeometry(vertices0=v0, edge1=edge1, edge2=edge2, normals=stl_normals)


def ray_mesh_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    geometry: MeshGeometry,
    epsilon: float = 1e-9,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Return the nearest intersection between a ray and the mesh, if any."""
    v0 = geometry.vertices0
    edge1 = geometry.edge1
    edge2 = geometry.edge2
    normals = geometry.normals

    dir_tile = direction[np.newaxis, :]
    pvec = np.cross(dir_tile, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    mask = np.abs(det) > epsilon
    if not np.any(mask):
        return None
    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]

    tvec = origin[np.newaxis, :] - v0
    u = np.zeros_like(det)
    u[mask] = np.einsum("ij,ij->i", tvec[mask], pvec[mask]) * inv_det[mask]
    mask &= (u >= 0.0) & (u <= 1.0)
    if not np.any(mask):
        return None

    qvec = np.cross(tvec, edge1)
    v = np.zeros_like(det)
    dir_repeated = np.broadcast_to(direction, edge1.shape)
    v[mask] = np.einsum("ij,ij->i", qvec[mask], dir_repeated[mask]) * inv_det[mask]
    mask &= (v >= 0.0) & (u + v <= 1.0)
    if not np.any(mask):
        return None

    t = np.zeros_like(det)
    t[mask] = np.einsum("ij,ij->i", edge2[mask], qvec[mask]) * inv_det[mask]
    mask &= t > epsilon
    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    t_hits = t[indices]
    best_idx = indices[np.argmin(t_hits)]
    distance = float(t[best_idx])
    point = origin + distance * direction
    normal = normals[best_idx]
    norm_len = np.linalg.norm(normal)
    if norm_len > 0.0:
        normal = normal / norm_len
    return distance, point, normal


def mesh_distance_statistics(mesh: np.ndarray) -> Tuple[float, float]:
    """Return mean and maximum distance of mesh vertices from the origin."""
    triangles = np.asarray(mesh, dtype=float)
    
    # Extract only vertices, not normals
    if triangles.shape[1] == 4:
        # Mesh format: [normal, v0, v1, v2] - skip the first element (normal)
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
        # Legacy format: [v0, v1, v2] - all are vertices
        vertices = triangles.reshape(-1, 3)
    
    if vertices.size == 0:
        raise ValueError("Mesh does not contain any vertices - cannot compute distances")
    radii = np.linalg.norm(vertices, axis=1)
    return float(np.mean(radii)), float(np.max(radii))


def infer_mesh_axis(mesh: np.ndarray) -> np.ndarray:
    """Infer the dominant axis of an STL mesh via principal component analysis."""
    triangles = np.asarray(mesh, dtype=float)
    
    # Extract only vertices, not normals
    if triangles.shape[1] == 4:
        # Mesh format: [normal, v0, v1, v2] - skip the first element (normal)
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
        # Legacy format: [v0, v1, v2] - all are vertices
        vertices = triangles.reshape(-1, 3)
    
    if vertices.shape[0] < 3:
        raise ValueError("Not enough vertices to infer mesh axis")
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if np.dot(axis, centroid) < 0.0:
        axis = -axis
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def build_orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Axis vector must be non-zero")
    axis /= norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(up, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return axis, u, v


def build_detector_plane_from_mesh(
    mesh: np.ndarray,
    axis: np.ndarray,
    tolerance: float = 0.01,
) -> DetectorPlane:
    triangles = np.asarray(mesh, dtype=float)
    
    # Extract only vertices, not normals
    if triangles.shape[1] == 4:
        # Mesh format: [normal, v0, v1, v2] - skip the first element (normal)
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
        # Legacy format: [v0, v1, v2] - all are vertices
        vertices = triangles.reshape(-1, 3)
    
    axis, u, v = build_orthonormal_frame(axis)
    projections = vertices @ axis
    plane_position = float(projections.max())
    mask = projections > plane_position - tolerance
    subset = vertices[mask] if np.any(mask) else vertices
    u_extent = float(np.max(np.abs(subset @ u)))
    v_extent = float(np.max(np.abs(subset @ v)))
    u_extent = max(u_extent, 1e-6)
    v_extent = max(v_extent, 1e-6)
    return DetectorPlane(axis, u, v, plane_position, u_extent, v_extent)


def build_default_detector_plane(distance: float, side: float) -> DetectorPlane:
    axis, u, v = build_orthonormal_frame(np.array([0.0, 0.0, 1.0]))
    half = side / 2.0
    return DetectorPlane(axis, u, v, distance, half, half)


def build_circular_detector_plane(
    center_mm: np.ndarray,
    radius_mm: float,
    axis: np.ndarray,
) -> DetectorPlane:
    """Build a circular detector plane with specified center and radius.
    
    Parameters
    ----------
    center_mm : np.ndarray
        Center position in millimeters [x, y, z].
    radius_mm : float
        Radius of the circular detector in millimeters.
    axis : np.ndarray
        Direction axis (will be normalized).
    
    Returns
    -------
    DetectorPlane
        Detector plane geometry for circular detector.
    """
    # Convert to meters
    center_m = center_mm * 1.0e-3
    radius_m = radius_mm * 1.0e-3
    
    # Build orthonormal frame
    axis_norm, u, v = build_orthonormal_frame(axis)
    
    # Calculate plane position as projection of center onto axis
    # This represents the distance from origin along the axis direction
    plane_position = float(np.dot(center_m, axis_norm))
    
    return DetectorPlane(
        axis=axis_norm,
        u=u,
        v=v,
        plane_position=plane_position,
        half_u=radius_m,
        half_v=radius_m,
        center=center_m,
        radius=radius_m,
        is_circular=True
    )


def transport_through_slab(
    energy_mev: float,
    slab_thickness: float,
    mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    target_mass_ratio: float,
    energy_cutoff_mev: float,
    initial_direction: Optional[np.ndarray] = None,
    start_position: Optional[np.ndarray] = None,
    detector_plane: Optional[DetectorPlane] = None,
    collect_trajectory: bool = False,
) -> Tuple[float, float, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """Propagate a neutron through a homogeneous slab with multiple scatterings.
    
    Now tracks position and checks for detector crossing after each free flight.
    Also collects trajectory points if collect_trajectory=True.
    
    Returns
    -------
    tuple : (cumulative_time, energy, current_dir, detector_crossing, trajectory_points)
        trajectory_points is a list of (position, energy) tuples at each collision
    """
    remaining = slab_thickness
    energy = float(energy_mev)
    cumulative_time = 0.0
    detector_crossing = None
    trajectory_points = []  # List of (position, energy) at each collision

    # 初始化方向和速度
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

    # Initialize position tracking
    track_position = (start_position is not None and detector_plane is not None)
    if track_position:
        current_pos = start_position.copy()
        # Record starting position if collecting trajectory
        if collect_trajectory:
            trajectory_points.append((current_pos.copy(), energy))
        
        axis = detector_plane.axis
        center = detector_plane.center
        radius = detector_plane.radius
        
        # Calculate initial position value
        pos_value_init = np.dot(axis, current_pos - center)
        if DEBUG:
            print(f"[debug TRACK] Position tracking enabled, start={current_pos}, pos_value_init={pos_value_init:.4f}")
    else:
        current_pos = start_position.copy() if start_position is not None else np.zeros(3)
        # Record starting position if collecting trajectory
        if collect_trajectory:
            trajectory_points.append((current_pos.copy(), energy))
        
        if DEBUG:
            print(f"[debug NO TRACK] Position tracking disabled: start_position={start_position is not None}, detector_plane={detector_plane is not None}")
        if detector_plane is not None:
            radius = detector_plane.radius
            axis = detector_plane.axis
            center = detector_plane.center

    # --- 在循环开始前计算当前的MFP ---
    mean_free_path = get_mfp_energy_dependent(energy,mfp_data)
    
    # If MFP is extremely large (> 100 * slab_thickness), treat as no collision
    # This avoids numerical issues with log() for very large MFP values
    if mean_free_path > 100.0 * slab_thickness:
        # Direct flight through the slab without collision
        cumulative_time = slab_thickness / speed
        
        # Check if this direct flight crosses detector
        if track_position:
            end_pos = current_pos + slab_thickness * current_dir
            pos_value = np.dot(axis, current_pos - center)
            end_value = np.dot(axis, end_pos - center)
            
            if pos_value * end_value < 0:
                # Crossing occurred
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

        # Check detector crossing BEFORE this flight step
        if track_position and detector_crossing is None:
            step_pos = current_pos + step * current_dir
            pos_value = np.dot(axis, current_pos - center)
            step_value = np.dot(axis, step_pos - center)
            
            if DEBUG and abs(pos_value + 2.9) < 0.5:  # Near detector region
                print(f"[debug STEP] pos_value={pos_value:.4f}, step_value={step_value:.4f}, step={step:.4f}m, remaining={remaining:.4f}m")
            
            # Check if we cross the detector plane during this step
            if pos_value * step_value < 0:
                if DEBUG:
                    print(f"[debug CROSS CHECK] pos_value={pos_value:.4f}, step_value={step_value:.4f}")
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
                            # Detector hit!
                            dist_to_crossing = np.linalg.norm(crossing_point - current_pos)
                            time_to_crossing = dist_to_crossing / speed
                            detector_crossing = (cumulative_time + time_to_crossing, crossing_point, energy)
                            # Return immediately
                            return cumulative_time + time_to_crossing, energy, current_dir, detector_crossing
        
        # Update position
        if track_position:
            current_pos += step * current_dir
        else:
            current_pos += step * current_dir
        
        cumulative_time += step / speed
        remaining -= step

        # # If we've exhausted the slab thickness, exit
        # if step < free_path:  
        #     energy = scatter_energy_elastic(energy, target_mass_ratio)
        #     current_dir = sample_isotropic_direction()
        #     speed = energy_to_speed(energy)
        #     mean_free_path = get_mfp_energy_dependent(energy, mfp_data)
        # else:
        #     break

        # If we've exhausted the slab thickness, exit
        if remaining <= 0.0:
            break
            
        # Otherwise, a collision occurred (step >= free_path)
        if step >= free_path:
            # Collision occurred before reaching boundary
            energy = scatter_energy_elastic(energy, target_mass_ratio)
            current_dir = sample_isotropic_direction()
            speed = energy_to_speed(energy)
            mean_free_path = get_mfp_energy_dependent(energy, mfp_data)
            
            # Record collision position and energy
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
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[Tuple[float, np.ndarray]], List[Tuple[np.ndarray, float]]]:
    """Advance a neutron through a mesh-defined solid material.
    
    This function properly handles segmented/discontinuous geometries by
    detecting each entry-exit pair separately and only simulating collisions
    within actual material, not empty space between segments.
    
    Parameters
    ----------
    max_segments : int
        Maximum number of material segments to traverse before giving up.
        Prevents infinite loops in pathological cases.
    detector_plane : DetectorPlane, optional
        If provided, checks if neutron crosses detector plane during transport.
        
    Returns
    -------
    tuple
        (cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points)
        detector_crossing is None if no crossing, or (time_to_detector, hit_point) if crossed.
        trajectory_points is a list of (position, energy) tuples at each collision.
    """
    trajectory_points = []  # Collect trajectory points from all segments
    
    if geometry is None:
        return 0.0, energy_mev, np.array(position, dtype=float), np.array(direction, dtype=float), None, trajectory_points

    current_pos = np.array(position, dtype=float)
    dir_norm = np.array(direction, dtype=float)
    dir_norm /= np.linalg.norm(dir_norm)
    current_energy = energy_mev
    cumulative_time = 0.0
    detector_crossing = None
    
    # Process each material segment separately
    for segment_idx in range(max_segments):
        # Debug at start of segment
        if DEBUG and detector_plane is not None and segment_idx == 0:
            pass  # Silence
            #proj = np.dot(detector_plane.axis, current_pos - detector_plane.center)
            #print(f"[debug start seg {segment_idx}] pos_proj={proj:.4f}m")
        
        # Check if we hit any geometry from current position
        hit = ray_mesh_intersection(current_pos, dir_norm, geometry)
        if hit is None:
            # No more intersections - neutron is now in free space
            break
        
        entry_dist, entry_point, normal = hit
        
        # Time to reach this segment (vacuum flight)
        speed = energy_to_speed(current_energy)
        vacuum_time = entry_dist / speed
        
        # Save time BEFORE entering this segment
        time_before_segment = cumulative_time + vacuum_time
        
        # Move slightly inside the material to find exit point
        inside_origin = entry_point + dir_norm * 1e-6
        exit_hit = ray_mesh_intersection(inside_origin, dir_norm, geometry)
        
        if exit_hit is None:
            # No exit found - neutron is lost inside material
            cumulative_time += vacuum_time
            return cumulative_time, energy_cutoff_mev, inside_origin, dir_norm, detector_crossing, trajectory_points
        
        # Calculate path length through THIS segment only
        path_inside = exit_hit[0]
        
        # Save energy BEFORE transport through segment
        energy_before_segment = current_energy
        
        # Simulate collisions within this segment
        slab_time, energy_after, direction_after, material_detector_crossing, segment_trajectory = transport_through_slab(
            current_energy,
            path_inside,
            mfp_data,
            target_mass_ratio,
            energy_cutoff_mev,
            initial_direction=dir_norm,
            start_position=entry_point,
            detector_plane=detector_plane,
            collect_trajectory=True,  # 收集完整轨迹
        )
        cumulative_time += vacuum_time + slab_time
        current_energy = energy_after
        # Collect trajectory points from this segment
        trajectory_points.extend(segment_trajectory)
        
        # If detector was hit inside material, record it
        if material_detector_crossing is not None and detector_crossing is None:
            # material_detector_crossing is (time_from_entry, position, energy)
            # Need to add time_before_segment to get absolute time
            crossing_time = time_before_segment + material_detector_crossing[0]
            detector_crossing = (crossing_time, material_detector_crossing[1], material_detector_crossing[2])
            if DEBUG:
                print(f"[debug MATERIAL HIT] Detector hit inside material at seg{segment_idx}, crossing_time={crossing_time*1e9:.1f}ns, energy={material_detector_crossing[2]:.3f}MeV")
        
        # Check if neutron was absorbed
        if current_energy <= energy_cutoff_mev:
            # Exit point is along the original trajectory (before final scatter)
            exit_point = inside_origin + dir_norm * path_inside
            return cumulative_time, current_energy, exit_point, direction_after, detector_crossing, trajectory_points
        
        # Move to exit point of this segment, ready for next iteration
        # The exit point is along the ORIGINAL direction (geometric path through mesh)
        exit_point = inside_origin + dir_norm * path_inside
        
        # Check if detector plane was crossed during vacuum flight to this segment
        # or during the path through this segment
        if detector_plane is not None and detector_plane.is_circular and detector_crossing is None:
            # Check both the vacuum flight and the material path
            start_for_check = position if segment_idx == 0 else current_pos
            
            # We need to check if crossing happened:
            # 1. During vacuum flight from start_for_check to entry_point (uses energy_before_segment)
            # 2. During material transit from entry_point to exit_point (needs energy interpolation)
            # 3. Special case: if we START at or very close to the detector plane
            
            center = detector_plane.center
            axis = detector_plane.axis
            
            # Check if start position is already at/on the detector plane
            pos_value = np.dot(axis, start_for_check - center)
            
            if DEBUG:
                print(f"[debug] START seg{segment_idx}: pos_value={pos_value:.6f}m")
            
            # If we're starting very close to the plane (within 1mm), check if we're within aperture
            if abs(pos_value) < 0.001:  # 1mm tolerance
                dist_from_center = np.linalg.norm(start_for_check - center - pos_value * axis)
                if dist_from_center <= detector_plane.radius:
                    # We're starting on the detector! Record this as a crossing
                    if DEBUG:
                        print(f"[debug ON PLANE] Starting on detector at dist={dist_from_center*1000:.1f}mm")
                    detector_crossing = (time_before_segment, start_for_check.copy(), energy_before_segment)
                    # Don't continue to other checks
                else:
                    if DEBUG:
                        print(f"[debug NEAR PLANE] Starting near plane but outside radius: {dist_from_center*1000:.1f}mm > {detector_plane.radius*1000:.1f}mm")
            
            if detector_crossing is None:
                # Check if entry point itself is on/near the detector plane
                entry_value = np.dot(axis, entry_point - center)
                
                if abs(entry_value) < 0.001:  # Entry point within 1mm of detector plane
                    # Calculate radial distance from axis
                    to_entry = entry_point - center
                    entry_axial = np.dot(to_entry, axis) * axis
                    entry_radial = to_entry - entry_axial
                    entry_dist = np.linalg.norm(entry_radial)
                    
                    if entry_dist <= detector_plane.radius:
                        # Entry point is on detector!
                        detector_crossing = (time_before_segment, entry_point.copy(), energy_before_segment)
                
                if detector_crossing is None:
                    # Check vacuum flight to entry_point
                    crossed_in_vacuum = pos_value * entry_value < 0
                    
                    if DEBUG and segment_idx == 0:
                        print(f"[debug VAC CHECK] pos={pos_value:.4f}, entry={entry_value:.4f}, crossed={crossed_in_vacuum}")
                    
                    # Then check material segment
                    exit_value = np.dot(axis, exit_point - center)
                    crossed_in_material = entry_value * exit_value < 0
                
                if DEBUG and segment_idx == 0 and abs(pos_value + 2.9) < 0.01 and not crossed_in_vacuum:
                    print(f"[debug NO VAC CROSS] pos={pos_value:.4f}, entry={entry_value:.4f}, exit={exit_value:.4f}, mat={crossed_in_material}")
                
                if crossed_in_vacuum:
                    # Crossing happened during vacuum flight before entering material
                    path_vec = entry_point - start_for_check
                    denominator = np.dot(axis, path_vec)
                    
                    if abs(denominator) > 1e-10:
                        s = -pos_value / denominator
                        
                        if 0 <= s <= 1:
                            crossing_point = start_for_check + s * path_vec
                            # Calculate perpendicular distance from detector axis
                            to_crossing = crossing_point - center
                            axial_component = np.dot(to_crossing, axis) * axis
                            radial_vector = to_crossing - axial_component
                            dist_from_center = np.linalg.norm(radial_vector)
                            
                            if dist_from_center <= detector_plane.radius:
                                # Energy during vacuum flight is constant (energy_before_segment)
                                dist_to_crossing = np.linalg.norm(crossing_point - start_for_check)
                                time_to_crossing = dist_to_crossing / energy_to_speed(energy_before_segment)
                                # Use time at start of vacuum flight (time_before_segment - vacuum_time)
                                crossing_time = (time_before_segment - vacuum_time) + time_to_crossing
                                detector_crossing = (crossing_time, crossing_point, energy_before_segment)
                                if DEBUG:
                                    print(f"[debug VAC HIT] dist={dist_from_center*1000:.1f}mm, time={crossing_time*1e9:.1f}ns")
                            else:
                                if DEBUG and segment_idx == 0:
                                    print(f"[debug VAC MISS] dist={dist_from_center*1000:.1f}mm > radius={detector_plane.radius*1000:.1f}mm")
                                crossing_time = (time_before_segment - vacuum_time) + time_to_crossing
                                detector_crossing = (crossing_time, crossing_point, energy_before_segment)
                            
                elif crossed_in_material:
                    # Crossing happened inside material segment
                    # Need to estimate energy at crossing point
                    path_vec = exit_point - entry_point
                    denominator = np.dot(axis, path_vec)
                    
                    if abs(denominator) > 1e-10:
                        s = -entry_value / denominator
                        
                        if 0 <= s <= 1:
                            crossing_point = entry_point + s * path_vec
                            # Calculate perpendicular distance from detector axis
                            to_crossing = crossing_point - center
                            axial_component = np.dot(to_crossing, axis) * axis
                            radial_vector = to_crossing - axial_component
                            dist_from_center = np.linalg.norm(radial_vector)
                            
                            if dist_from_center <= detector_plane.radius:
                                # Estimate energy at crossing: linear interpolation based on position
                                # This is approximate but better than using end energy
                                energy_at_crossing = energy_before_segment + s * (energy_after - energy_before_segment)
                                
                                # Time calculation: proportion of slab_time based on position
                                time_to_crossing = s * slab_time
                                crossing_time = time_before_segment + time_to_crossing
                                detector_crossing = (crossing_time, crossing_point, energy_at_crossing)
        
        # If detector was crossed, stop propagation immediately
        # We don't want to continue into the scintillator volume
        if detector_crossing is not None:
            # Return current state at detector crossing
            return cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points
        
        # Also check if we've passed the detector plane without crossing it
        # This means the neutron missed the detector aperture or is traveling through continuous material
        if detector_plane is not None:
            # Calculate projection of current position onto detector axis
            # Positive projection means beyond the detector (towards scintillator end)
            # Negative projection means before detector (towards source)
            current_proj = np.dot(detector_plane.axis, current_pos - detector_plane.center)
            
            # If projection is positive, we've passed the detector plane
            if current_proj > 0:
                # We've gone past the detector without hitting it via the crossing detection
                # This means we're in continuous material spanning the detector plane
                if DEBUG:
                    to_point = current_pos - detector_plane.center
                    parallel_component = np.dot(to_point, detector_plane.axis) * detector_plane.axis
                    perpendicular_vec = to_point - parallel_component
                    perp_dist = np.linalg.norm(perpendicular_vec)
                    print(f"[debug] Passed detector: proj={current_proj:.4f}m, perp={perp_dist*1000:.1f}mm")
                
                # Stop propagation - we've passed the detector region
                return cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points
        
        # Update direction for next segment (after scattering)
        dir_norm = direction_after / np.linalg.norm(direction_after)
        
        # Move slightly outside the material along the NEW direction
        current_pos = exit_point + dir_norm * 1e-5
        
        # Debug: show position after each segment
        if DEBUG and detector_plane is not None and segment_idx < 3:
            proj = np.dot(detector_plane.axis, current_pos - detector_plane.center)
            print(f"[debug seg {segment_idx}] pos_proj={proj:.4f}m, E={current_energy:.4f}")
    
    # Exited all material segments successfully
    return cumulative_time, current_energy, current_pos, dir_norm, detector_crossing, trajectory_points


################################################################################
# Sampling utilities
################################################################################

def sample_neutron_energy(mean_mev: float = 2.45, std_mev: float = 0.1) -> float:
    """Sample a neutron kinetic energy from a Gaussian distribution.

    Parameters
    ----------
    mean_mev : float, optional
        Mean energy in MeV of the distribution.  Defaults to 2.45 MeV.
    std_mev : float, optional
        Standard deviation of the energy distribution in MeV.  Defaults to
        0.1 MeV.  Modify this value to control the spectral width.

    Returns
    -------
    float
        A randomly sampled energy in MeV.  Negative energies are rejected and
        resampled.
    """
    energy = np.random.normal(mean_mev, std_mev)
    # Resample to avoid unphysical negative energies
    while energy <= 0.0:  
        energy = np.random.normal(mean_mev, std_mev)
    return float(energy)


def sample_isotropic_direction() -> np.ndarray:
    """Generate a random unit vector isotropically distributed on the sphere."""
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([r_xy * math.cos(phi), r_xy * math.sin(phi), z], dtype=float)


def sample_direction_in_cone(
    axis: np.ndarray,
    half_angle_deg: float,
) -> np.ndarray:
    """Sample a unit vector within a cone of half-angle ``half_angle_deg``."""
    axis, u, v = build_orthonormal_frame(axis)
    half_angle_rad = math.radians(half_angle_deg)
    cos_min = math.cos(half_angle_rad)
    cos_theta = (1.0 - cos_min) * np.random.rand() + cos_min
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * np.random.rand()
    local_dir = np.array(
        [
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        ],
        dtype=float,
    )

    return local_dir[0] * u + local_dir[1] * v + local_dir[2] * axis


################################################################################
# Kinematics utilities
################################################################################

def energy_to_speed(energy_mev: float) -> float:
    """Convert neutron kinetic energy (MeV) to speed (m/s).

    The relationship between kinetic energy and speed for a non‑relativistic
    particle is

        E = ½ m v²

    where E is the kinetic energy in Joules, m is the neutron mass and v is
    the speed.  The neutron mass is taken as 1.67492749804×10⁻²⁷ kg.  This
    conversion is adequate for energies up to a few MeV where the speed
    remains well below the speed of light.  For extremely high energies
    relativistic corrections would be necessary.

    Parameters
    ----------
    energy_mev : float
        Kinetic energy in MeV.

    Returns
    -------
    float
        Speed in metres per second.
    """
    neutron_mass = 1.67492749804e-27  # kg
    # Convert MeV to Joules: 1 eV = 1.602176634×10⁻¹⁹ J
    energy_joules = energy_mev * 1.0e6 * 1.602176634e-19
    # v = sqrt(2E/m)
    speed = math.sqrt(2.0 * energy_joules / neutron_mass)
    return speed


def scatter_energy_elastic(neutron_energy_mev: float, target_mass_ratio: float) -> float:
    """Compute the neutron energy after an elastic scattering event.

    A simple two‑body kinematic model is used whereby the scattering is
    isotropic in the centre‑of‑mass frame.  For a neutron of energy E
    scattering off a stationary nucleus of mass ratio A (= target mass /
    neutron mass) the fractional energy retention r is

        r = [ (A - 1)² + 2(A + 1) cos θ + 1 ] / (A + 1)²

    where θ is a random angle uniformly distributed in [0, π].  After the
    collision the neutron energy becomes E= r E.  This formula ignores
    details such as angular distributions that vary with energy and nuclear
    structure, but provides a reasonable first approximation.

    Parameters
    ----------
    neutron_energy_mev : float
        Incoming neutron energy in MeV.
    target_mass_ratio : float
        Ratio of the target nucleus mass to the neutron mass (A).  For
        aluminium, A ≈ 26.98.  For hydrogen in a plastic scintillator, A = 1.

    Returns
    -------
    float
        The outgoing neutron energy in MeV after one elastic collision.
    """
    # Draw a scattering angle uniformly in cosine space (isotropic in CMS)
    cos_theta = 2.0 * np.random.rand() - 1.0
    A = target_mass_ratio
    # Compute energy retention fraction r
    numerator = (A - 1.0) * (A - 1.0) + 2.0 * (A + 1.0) * cos_theta + 1.0
    denominator = (A + 1.0) * (A + 1.0)
    r = numerator / denominator
    # Physical constraint: elastic scattering cannot increase energy
    # Ensure 0 ≤ r ≤ 1
    r = max(0.0, min(r, 1.0))
    return float(neutron_energy_mev * r)


################################################################################
# Data Loading Utilities
################################################################################

def load_mfp_data_from_csv(file_path: str) -> np.ndarray:
    """
    Load macro-scopic cross section data from a two-column CSV file.

    The file must contain data pairs: [Energy (MeV), Sigma_Macro (m⁻¹)].
    The function handles conversion and ensures the data is sorted by energy.

    Parameters
    ----------
    file_path : str
        Path to the CSV file on disk containing the cross section data.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Sorted array of [Energy, Sigma_Macro].
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cross section data file '{file_path}' does not exist.")

    try:
        # NOTE: Updated to handle ';' delimiter and skip potential header lines.
        # We assume the energy is in the first column (index 0) 
        # and the macroscopic cross section (Sigma_Macro) is in the second column (index 1), 
        # based on the header structure provided.
        # We load only columns 0 and 1.
        
        # Determine the number of header rows to skip (assuming non-numeric lines)
        skip_rows = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Try to parse the first value as float. If it fails, it's likely a header.
                    float(line.split(';')[0])
                    break
                except ValueError:
                    skip_rows += 1
                if skip_rows > 3: # Avoid infinite loop on corrupt files
                    break

        data = np.loadtxt(
            file_path, 
            delimiter=';', 
            skiprows=skip_rows, 
            usecols=(0, 1), # 使用第0列(Energy)和第1列(Sigma_Macro)
            dtype=float
        )
        
        # JANIS/ENDF data often uses large energy units (e.g., eV). 
        # Assuming the first column is in eV, we convert it to MeV for consistency.
        # This conversion step is critical if the source data is not already in MeV.
        data[:, 0] = data[:, 0] * 1e-6 
        
    except Exception as e:
        raise ValueError(f"Could not load and process data from CSV: {e}")

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Processed data must contain exactly two columns: Energy (MeV) and Macro-Sigma (m⁻¹).")

    # Ensure data is sorted by Energy for correct interpolation
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
    
    PE molar mass (C₂H₄): M_PE = 2 * M_C + 4 * M_H ≈ 28.05 g/mol.
    
    Macro-section Sigma = N_i * sigma_i(E).

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
    
    # 1. PE 物理常数
    M_C = 12.011  # g/mol
    M_H = 1.008  # g/mol
    M_PE = 2 * M_C + 4 * M_H  # ≈ 28.05 g/mol (C₂H₄)
    
    # 2. 计算原子核数密度 N_i (m⁻³)
    # N_i = (rho * N_A * n_i) / M_PE
    rho_kg_m3 = density_g_cm3 * 1000 # kg/m³ (或保留 g/cm³, 最终单位调整)
    
    # 我们使用 M_PE (g/mol) 和 rho (g/cm³) 进行计算，最终转换为 m⁻³
    # N_i (cm⁻³) = (rho_g_cm3 * N_A * n_i) / M_PE
    # N_i (m⁻³) = N_i (cm⁻³) * 1e6
    
    N_C_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 2) / M_PE
    N_H_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 4) / M_PE
    
    N_C_m3 = N_C_cm3 * 1e6
    N_H_m3 = N_H_cm3 * 1e6

    # 3. 创建统一的能量网格
    all_energies = np.unique(np.concatenate([h_micro_data[:, 0], c_micro_data[:, 0]]))
    
    # 4. 插值微观截面(Sigma_Micro) 到统一网格
    sigma_h_interp_barn = np.interp(all_energies, h_micro_data[:, 0], h_micro_data[:, 1])
    sigma_c_interp_barn = np.interp(all_energies, c_micro_data[:, 0], c_micro_data[:, 1])
    
    # 5. 计算最终的宏观截面 Sigma_Macro (m⁻¹)
    # Sigma_Macro (m⁻¹) = N_i (m⁻³) * sigma_i (m²)
    # Sigma_Macro (m⁻¹) = N_i (m⁻³) * sigma_i (barn) * BARN_TO_M2
    
    sigma_pe_total_m1 = (
        N_C_m3 * sigma_c_interp_barn * BARN_TO_M2 +
        N_H_m3 * sigma_h_interp_barn * BARN_TO_M2
    )
    
    # 6. 组成新的数据数组
    pe_macro_data = np.stack([all_energies, sigma_pe_total_m1], axis=1)
    
    return pe_macro_data


################################################################################
# Energy-dependent MFP utility (UPDATED FUNCTION)
################################################################################

# 示例数据结构: 预设的宏观截面数据（能量 [MeV] -> 宏观截面 [m⁻¹]）
# 注意：这些数据需要根据材料密度和真实核数据计算得到。
# 聚乙烯(C₂H₄) 密度 ≈ 0.92 g/cm³)
MFP_DATA_PE = np.array([
    [0.1, 15.0],  # 0.1 MeV 时的宏观截面 (m⁻¹)
    [0.5, 13.5],
    [1.0, 10.0],
    [2.45, 16.6], # 2.45 MeV 基准值
    [5.0, 17.5],
    [10.0, 18.0],
    [14.1, 17.8], # D-T 中子能量
])

# 铝(Al, 密度 ≈ 2.70 g/cm³)
MFP_DATA_AL = np.array([
    [0.1, 10.0],
    [0.5, 11.2],
    [1.0, 13.0],
    [2.45, 14.4], # 2.45 MeV 基准值
    [5.0, 15.5],
    [10.0, 16.0],
    [14.1, 16.2],
])


def get_mfp_energy_dependent(
    energy_mev: float,
    mfp_data: np.ndarray,
) -> float:
    """
    根据中子能量计算平均自由程（MFP），使用线性插值。

    参数
    ----------
    energy_mev : float
        中子的当前动能(MeV)。
    mfp_data : np.ndarray
        预设的 [能量 (MeV), 宏观截面 (m⁻¹)] 数据对数组。

    返回
    -------
    float
        新的平均自由程(m)。
    """
    if energy_mev <= 0.1:
        return 1e12  # 能量为零，视为停止（无限大 MFP，但实际会被截止）

    energies = mfp_data[:, 0]
    sigmas = mfp_data[:, 1]
    
    # 确保能量在插值范围内，否则使用边界值（常数外插）
    if energy_mev < energies.min():
        sigma = sigmas[0]
    elif energy_mev > energies.max():
        sigma = sigmas[-1]
    else:
        # 使用线性插值计算宏观截面 sigma (m⁻¹)
        sigma = np.interp(energy_mev, energies, sigmas)

    # MFP = 1 / Sigma。确保 Sigma 不为零。
    if sigma <= 1e-12:
        return 1e12  # 如果宏观截面为零，MFP 视为无限大
        
    # 返回 MFP (m)
    return 1.0 / sigma


################################################################################
# Transport through the aluminium shell
################################################################################
def simulate_in_aluminium(
    direction: np.ndarray,
    energy_mev: float,
    shell_thickness: float,
    aluminium_mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    target_mass_ratio: float,
    energy_cutoff_mev: float = 0.1,
    mesh_geometry: Optional[MeshGeometry] = None,
    detector_plane: Optional[DetectorPlane] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[Tuple[float, np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """Simulate neutron transport and scattering in an aluminium shell.

    When ``mesh_geometry`` is provided the neutron first propagates in vacuum
    from the origin (target centre) until it intersects the STL surface.  If
    no intersection occurs the neutron leaves the target without touching the
    aluminium shell.  Otherwise the intersection point is treated as the inner
    surface of the shell and the neutron traverses a material slab of
    thickness ``shell_thickness``.  Collisions are sampled from an exponential
    free-path distribution.  This treatment captures the presence of holes in
    the shell: directions that miss the mesh experience no aluminium
    interactions.
    
    Returns
    -------
    tuple
        (time, energy, direction, position, detector_crossing, trajectory_points)
        trajectory_points is a list of (position, energy) tuples at each collision

    Parameters
    ----------
    direction : np.ndarray, shape (3,)
        Initial direction unit vector.  It need not be normalised on entry.
    energy_mev : float
        Initial neutron kinetic energy in MeV.
    shell_thickness : float
        Thickness of the aluminium shell in metres.
    mean_free_path : float
        Mean free path for neutron collisions in aluminium in metres.
    target_mass_ratio : float
        Ratio A of the aluminium nucleus mass to the neutron mass.
    energy_cutoff_mev : float, optional
        Energies below this threshold are considered lost; the neutron is
        assumed to be absorbed.
    mesh_geometry : MeshGeometry, optional
        Preprocessed STL mesh that defines where aluminium is present.

    Returns
    -------
    tuple of (time, energy, direction, position)
        * **time** - the total time spent from the target centre to the aluminium
          exit point (s).
        * **energy** - the neutron energy after leaving the shell (MeV).
        * **direction** - the final direction unit vector when exiting the shell.
        * **position** - 3-D position (m) of the exit point relative to the origin.
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
    trajectory_points = []  # Collect trajectory points in shell

    # If a mesh is supplied, determine whether the neutron encounters aluminium.
    origin = np.zeros(3, dtype=float)

    if mesh_geometry is None:
        # Simple slab approximation when no geometry is supplied.
        slab_time, energy, current_dir, material_crossing, slab_trajectory = transport_through_slab(
            energy,
            shell_thickness,
            aluminium_mfp_data, # 传入数据数组
            target_mass_ratio,
            energy_cutoff_mev,
            initial_direction=direction,
            start_position=origin,
            detector_plane=detector_plane,
            collect_trajectory=True,  # 收集完整轨迹
        )
        cumulative_time += slab_time
        position = origin + direction * shell_thickness
        if material_crossing is not None:
            detector_crossing = (material_crossing[0], material_crossing[1], material_crossing[2])
        # Collect trajectory points from slab
        trajectory_points.extend(slab_trajectory)
        return cumulative_time, energy, current_dir, position, detector_crossing, trajectory_points

    # Detailed treatment using STL geometry
    # Find first intersection (entry point into aluminum shell)
    hit_entry = ray_mesh_intersection(origin, direction, mesh_geometry)
    if hit_entry is None:
        # Trajectory leaves through an opening without touching aluminium.
        # The neutron flies through a hole in the shell.
        # We need to advance the neutron to approximately where the shell would be.
        # Use shell_thickness as an estimate of the radial distance to the outer surface
        typical_shell_radius = 1.2  # meters, from STL analysis (mean vertex distance)
        exit_position = origin + direction * typical_shell_radius
        vacuum_time = typical_shell_radius / speed
        cumulative_time += vacuum_time
        
        # Still need to check if detector was crossed during this vacuum flight!
        if detector_plane is not None and detector_plane.is_circular:
            center = detector_plane.center
            axis = detector_plane.axis
            pos_value = np.dot(axis, origin - center)
            exit_value = np.dot(axis, exit_position - center)
            
            # Check if crosses detector plane
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
    
    # Check if detector is crossed during vacuum flight from origin to entry point
    if detector_plane is not None and detector_plane.is_circular:
        center = detector_plane.center
        axis = detector_plane.axis
        pos_value = np.dot(axis, origin - center)
        entry_value = np.dot(axis, entry_point - center)
        
        # Check if crosses detector plane
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
                        if DEBUG:
                            print(f"[debug SHELL-ENTRY VAC HIT] Detector hit in vacuum before shell entry")
    
    # Advance to entry point
    cumulative_time += distance_to_entry / speed
    position = entry_point.copy()
    
    # Find second intersection (exit point from aluminum shell)
    # Start slightly inside the mesh to avoid numerical issues
    epsilon = 1e-6  # Small offset to ensure we're inside the shell
    search_origin = position + direction * epsilon
    hit_exit = ray_mesh_intersection(search_origin, direction, mesh_geometry)
    
    if hit_exit is None:
        # This shouldn't happen for a closed mesh
        # The neutron has entered but cannot find an exit - likely a mesh issue
        # Treat as absorbed/lost in the shell
        return cumulative_time, energy_cutoff_mev - 0.01, direction, position, detector_crossing, trajectory_points
    
    distance_to_exit, mesh_exit_point, exit_normal = hit_exit
    # Calculate actual path length through the mesh
    path_length_along_direction = distance_to_exit + epsilon
    
    # The actual exit point from the mesh
    # mesh_exit_point is relative to search_origin, need to convert to absolute position
    actual_exit_point = search_origin + direction * distance_to_exit
    
    # Slab simulation inside the mesh material
    # We call transport_through_slab to simulate the interaction in the slab material
    slab_time, energy_out, direction_out, material_crossing, slab_trajectory = transport_through_slab(
        energy,  # Use current energy, not initial energy_mev
        path_length_along_direction, # Actual path length through the mesh
        aluminium_mfp_data, # 传入数据数组
        target_mass_ratio,  # Use the function parameter, not undefined aluminium_mass_ratio
        energy_cutoff_mev,
        initial_direction=direction,
        start_position=position,
        detector_plane=detector_plane,
        collect_trajectory=True,  # 收集完整轨迹
    )
    
    # Check if detector was crossed inside aluminum
    if material_crossing is not None and detector_crossing is None:
        crossing_time_in_slab = material_crossing[0]
        detector_crossing = (cumulative_time + crossing_time_in_slab, material_crossing[1], material_crossing[2])
        if DEBUG:
            print(f"[debug SHELL MATERIAL HIT] Detector hit inside aluminum shell")
    
    # Update time and return the actual exit point
    cumulative_time += slab_time
    
    # Collect trajectory points from slab
    trajectory_points.extend(slab_trajectory)
    
    return cumulative_time, energy_out, direction_out, actual_exit_point, detector_crossing, trajectory_points


################################################################################
# Flight from shell to scintillator
################################################################################

def propagate_to_scintillator(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    detector_plane: DetectorPlane,
    energy_cutoff_mev: float = 0.1,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Propagate a neutron from the shell exit to the scintillator.

    The scintillator is modelled as a planar rectangle oriented along
    ``detector_plane.axis``.  Once the neutron exits the polyethylene channel
    it travels in vacuum straight to this plane; a hit is recorded only if the
    intersection falls within the detector area.

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Starting position in metres (typically the shell exit point).
    direction : np.ndarray, shape (3,)
        Unit vector giving the direction of motion leaving the aluminium shell.
    energy_mev : float
        Kinetic energy in MeV when leaving the aluminium shell.
    detector_plane : DetectorPlane
        Geometry describing the scintillator plane.
    energy_cutoff_mev : float, optional
        Energies below this threshold are considered lost.  If a neutron has
        too little energy to travel the full distance, it is discarded.

    Returns
    -------
    tuple of (flight_time, hit_point) or (None, None)
        If the neutron reaches the scintillator, ``flight_time`` is the time
        (s) spent in flight from the shell exit to the scintillator, and
        ``hit_point`` is the 3‑D position on the detector plane where it
        arrives.  If the neutron misses the detector, both return values are
        ``None``.
    """
    # Normalise direction to ensure it is a unit vector
    pos = np.array(position, dtype=float)
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return None, None
    d = d / norm

    # Check if neutron is travelling towards the detector plane
    # For circular detector with explicit center, we need different logic
    axis = detector_plane.axis
    dir_dot = float(np.dot(d, axis))
    
    # For debugging: we'll allow checking both directions
    # The key is whether the neutron will intersect the detector plane at all

    if energy_mev <= energy_cutoff_mev:
        return None, None

    speed = energy_to_speed(energy_mev)
    u = detector_plane.u
    v = detector_plane.v
    plane_pos = detector_plane.plane_position

    # For circular detector with explicit center, calculate intersection differently
    if detector_plane.is_circular and detector_plane.center is not None:
        # Find where ray intersects the plane perpendicular to axis through detector center
        center = detector_plane.center
        # Plane equation: axis · (point - center) = 0
        # Ray: point = pos + t * d
        # axis · (pos + t*d - center) = 0
        # axis · pos + t * axis · d - axis · center = 0
        # t = (axis · center - axis · pos) / (axis · d)
        numerator = float(np.dot(axis, center - pos))
        
        # Check if ray is parallel to plane (dir_dot ≈ 0)
        if abs(dir_dot) < 1e-10:
            return None, None
        
        t_total = numerator / dir_dot
        
        # Only accept forward intersections (t > 0)
        # t > 0 means the intersection point is ahead of current position
        if t_total <= 0.0:
            return None, None
        
        if t_total > 1e6:  # Sanity check: too far away
            return None, None
        
        hit = pos + d * t_total
        
        # Check distance from detector center
        distance_from_center = np.linalg.norm(hit - center)
        if distance_from_center > detector_plane.radius:
            return None, None
        
        flight_time = t_total / speed
        return flight_time, hit
    
    else:
        # Original rectangular detector logic
        t_total = (plane_pos - float(np.dot(pos, axis))) / dir_dot
        if t_total <= 0.0:
            return None, None

        hit = pos + d * t_total
        
        # For rectangular detector, check u and v coordinates
        u_coord = float(np.dot(hit, u))
        v_coord = float(np.dot(hit, v))
        if abs(u_coord) > detector_plane.half_u or abs(v_coord) > detector_plane.half_v:
            return None, None

        flight_time = t_total / speed
        return flight_time, hit


################################################################################
# Detector arrival (scintillator interaction removed)
################################################################################
# Note: Scintillator interaction simulation has been removed.
# The simulation now ends when neutrons reach the detector surface.


################################################################################
# High level simulation driver
################################################################################

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
) -> Optional[float]:
    """Simulate the complete history of a single neutron.

    This function ties together all the individual stages: generation at the
    target centre, transport through the aluminium shell, interaction with the
    polyethylene channel, and flight to the detector surface.  If the
    neutron reaches the detector, the function returns its total time‑of‑flight
    (source to detector).  Otherwise it returns ``None`` to indicate that the
    neutron was lost or absorbed.

    Parameters
    ----------
    shell_thickness : float
        Aluminium shell thickness (m).
    aluminium_mfp_data : np.ndarray
        Mean free path data array for neutron collisions in aluminium (m).
    aluminium_mass_ratio : float
        Aluminium nucleus mass divided by neutron mass (A ≈ 26.98).
    scintillator_thickness : float
        Thickness of the scintillator (m).
    scintillator_mfp_data : np.ndarray
        Mean free path data array for neutron collisions in the scintillator (m).
    scintillator_mass_ratio : float
        Mass ratio of the dominant scattering nucleus in the scintillator.
    detector_distance : float, optional
        Distance from the target to the scintillator (m).  Used only when
        ``detector_plane`` is ``None``.  Defaults to 16 m.
    detector_side : float, optional
        Side length of the square scintillator (m) for the fallback detector
        geometry.  Defaults to 1 m.
    energy_cutoff_mev : float, optional
        Energy threshold below which neutrons are considered absorbed.
    shell_geometry : MeshGeometry, optional
        Preprocessed STL data.  When provided the simulation respects the shell
        openings described by the mesh.  If ``None``, a spherical approximation
        is used.
    channel_geometry : MeshGeometry, optional
        STL geometry describing the polyethylene channel.
    channel_mfp : float, optional
        Mean free path in polyethylene (m).  Defaults to 0.0602 m.
    channel_mass_ratio : float, optional
        Mass ratio of the dominant scattering nucleus in polyethylene.
    source_cone_axis : np.ndarray, optional
        Axis of the emission cone.  When ``None`` the emission is isotropic.
    source_cone_half_angle_deg : float, optional
        Half-angle (degrees) of the emission cone.  Defaults to 15°.
    detector_plane : DetectorPlane, optional
        Explicit detector geometry.  When ``None`` a square detector located
        ``detector_distance`` metres along ``+z`` is used.

    Returns
    -------
    tuple or None
        Returns (status_string, NeutronRecord) if successful, or (status_string, None)
        if the neutron was lost. Status can be 'success', 'lost_in_shell', 
        'lost_in_channel', or 'missed_detector'.
    """
    # 1. Generate initial energy and direction
    E0 = sample_neutron_energy()
    if source_cone_axis is None:
        d0 = sample_isotropic_direction()
    else:
        d0 = sample_direction_in_cone(source_cone_axis, source_cone_half_angle_deg)

    # Initialize trajectory points list: (position, energy, event_type)
    trajectory_points = []
    # Add starting point at origin
    trajectory_points.append((np.zeros(3), E0, "source"))

    # 2. Transport through the aluminium shell
    detector_plane = detector_plane or build_default_detector_plane(detector_distance, detector_side)

    t_shell, E_after_shell, d_after_shell, pos_after_shell, shell_detector_crossing, shell_trajectory = simulate_in_aluminium(
        d0,
        E0,
        shell_thickness,
        aluminium_mfp_data, # 传入数据数组
        aluminium_mass_ratio,
        energy_cutoff_mev,
        mesh_geometry=shell_geometry,
        detector_plane=detector_plane,
    )
    # Merge shell trajectory points into main trajectory with event type
    for pos, energy in shell_trajectory:
        trajectory_points.append((pos, energy, "scatter"))
    # Check if detector was hit during shell transport
    if shell_detector_crossing is not None:
        crossing_time, crossing_point, crossing_energy = shell_detector_crossing
        # Add shell exit and detector hit to trajectory
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
        if DEBUG:
            print(f"[debug] Neutron hit detector during shell phase at t={crossing_time*1e9:.1f}ns")
        return ("success", record)
    
    # Add shell exit point to trajectory
    trajectory_points.append((pos_after_shell.copy(), E_after_shell, "shell_exit"))
    
    # If energy is below cutoff the neutron was absorbed in the shell
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
    E_before_channel = E_after_shell  # Store for debugging
    result = propagate_through_mesh_material(
        pos_after_shell,
        d_after_shell,
        E_after_shell,
        channel_geometry,
        channel_mfp_data, # 传入数据数组
        channel_mass_ratio,
        energy_cutoff_mev,
        detector_plane=detector_plane,  # Pass detector plane for crossing detection
    )
    
    # Unpack result with detector crossing info and trajectory points
    t_channel, E_after_channel, pos_after_channel, d_after_channel, detector_crossing, channel_trajectory = result
    # Merge channel trajectory points into main trajectory with event type
    for pos, energy in channel_trajectory:
        trajectory_points.append((pos, energy, "scatter"))
    
    # Debug output
    if DEBUG and detector_crossing is None:
        pos_proj = np.dot(detector_plane.axis, pos_after_channel - detector_plane.center)
        # Calculate perpendicular distance from detector axis
        to_point = pos_after_channel - detector_plane.center
        parallel_component = np.dot(to_point, detector_plane.axis) * detector_plane.axis
        perpendicular_vec = to_point - parallel_component
        perp_dist = np.linalg.norm(perpendicular_vec)
        print(f"[debug] Lost: E={E_after_channel:.4f}, pos_proj={pos_proj:.4f}m, perp_dist={perp_dist*1000:.1f}mm")
    
    # Add channel exit point to trajectory
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
    
    # Check if detector was crossed inside channel
    if detector_crossing is not None:
        crossing_time, crossing_point, crossing_energy = detector_crossing
        total_tof = t_shell + crossing_time
        
        # Add detector hit to trajectory
        trajectory_points.append((crossing_point.copy(), crossing_energy, "detector_hit"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=crossing_energy,  # Energy AT detector crossing
            tof=total_tof,
            exit_position=pos_after_shell.copy(),
            detector_hit_position=crossing_point.copy(),
            reached_detector=True,
            energy_after_shell=E_after_shell,
            energy_after_channel=crossing_energy,  # Use crossing energy, not end-of-channel energy
            status="success",
            final_position=crossing_point.copy(),
            final_direction=d_after_channel.copy(),
            trajectory_points=trajectory_points
        )
        return ("success", record)

    # 4. Check final position relative to detector plane
    # If neutron exited all materials without hitting detector, determine if it missed or never reached
    pos_proj = np.dot(detector_plane.axis, pos_after_channel - detector_plane.center)
    
    if pos_proj > 0:
        # Neutron has passed beyond the detector plane without hitting it
        # This is a confirmed miss
        # Add an extension point to show continued trajectory (1m beyond current position)
        extension_distance = 1.0  # meters
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
        # Neutron is still before the detector plane
        # Try vacuum flight to detector (no more material interactions)
        flight_time, hit_point = propagate_to_scintillator(
            pos_after_channel,
            d_after_channel,
            E_after_channel,
            detector_plane,
            energy_cutoff_mev=energy_cutoff_mev,
        )
        
        if flight_time is None:
            # Neutron won't reach detector even in vacuum flight
            # Add an extension point to show continued trajectory
            extension_distance = 1.5  # meters - extend to show it's flying away
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

        # At this point, neutron has reached the detector surface via vacuum flight
        # Record the energy at detector entrance (E_after_channel) and total TOF
        # Total TOF = time through shell + time through channel + flight time to detector
        total_tof = t_shell + t_channel + flight_time
        
        # Add detector hit to trajectory
        if hit_point is not None:
            trajectory_points.append((hit_point.copy(), E_after_channel, "detector_hit"))
        
        record = NeutronRecord(
            initial_energy=E0,
            final_energy=E_after_channel,  # Energy when reaching detector
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
) -> List[NeutronRecord]:
    """Simulate multiple neutrons and return their complete records.

    This function repeatedly calls :func:`simulate_neutron_history` and
    collects the returned records for neutrons that reach the detector surface.
    It provides a simple entry point for generating a TOF spectrum and analyzing
    neutron behavior.

    Parameters
    ----------
    n_neutrons : int
        Number of neutron histories to simulate.
    All other parameters
        These are passed directly to :func:`simulate_neutron_history`.

    Returns
    -------
    list of NeutronRecord
        A list containing the complete record for each neutron that reached the
        detector surface.  Neutrons that were lost or absorbed do not appear in
        this list.
    """
    records: List[NeutronRecord] = []
    
    # Diagnostics
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
        )
        
        # Record ALL neutrons
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
    
    # Print diagnostics
    success_count = sum(1 for r in records if r.reached_detector)
    success_rate = success_count / n_neutrons * 100 if n_neutrons > 0 else 0
    print(f"[debug] Neutron fate statistics:")
    print(f"  - Reached detector: {success_count} ({success_rate:.2f}%)")
    print(f"  - Lost in aluminum shell: {lost_in_shell} ({lost_in_shell/n_neutrons*100:.2f}%)")
    print(f"  - Lost in polyethylene channel: {lost_in_channel} ({lost_in_channel/n_neutrons*100:.2f}%)")
    print(f"  - Missed detector: {missed_detector} ({missed_detector/n_neutrons*100:.2f}%)")
    
    if sample_miss_records:
        print(f"\n[debug] Sample neutrons that missed detector (first 10):")
        for i, rec in enumerate(sample_miss_records):
            print(f"  #{i+1}:")
            print(f"    Final position: {rec.final_position}")
            print(f"    Final direction: {rec.final_direction}")
            print(f"    Energy: {rec.energy_after_channel:.4f} MeV")
            # Calculate if neutron direction points toward detector
            if rec.final_position is not None and rec.final_direction is not None:
                to_detector = detector_plane.center - rec.final_position
                dot_product = np.dot(rec.final_direction, to_detector / np.linalg.norm(to_detector))
                print(f"    Direction toward detector (dot product): {dot_product:.4f}")
                # Calculate distance from final position to detector center
                dist_to_center = np.linalg.norm(to_detector)
                print(f"    Distance to detector center: {dist_to_center:.4f} m")
    
    return records


################################################################################
# Data Export utilities
################################################################################

def export_neutron_records_to_csv(records: List[NeutronRecord], filename: str = "neutron_data.csv"):
    """Export neutron records to a CSV file.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of neutron records from simulation.
    filename : str
        Output CSV filename.
    """
    import csv
    
    if not records:
        print("[warning] No neutron records to export.")
        return
    
    # Define CSV headers
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
    
    # Open file and write data
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for idx, record in enumerate(records, start=1):
            # Calculate velocity from energy (E = 0.5 * m * v^2)
            # For neutrons: m_n = 1.674927498e-27 kg
            # E (Joules) = E (MeV) * 1.602176634e-13
            neutron_mass_kg = 1.674927498e-27
            energy_joules = record.final_energy * 1.602176634e-13
            velocity_magnitude = math.sqrt(2 * energy_joules / neutron_mass_kg) if record.final_energy > 0 else 0.0
            
            # Calculate velocity components from direction and magnitude
            if record.final_direction is not None:
                velocity_x = velocity_magnitude * record.final_direction[0]
                velocity_y = velocity_magnitude * record.final_direction[1]
                velocity_z = velocity_magnitude * record.final_direction[2]
                dir_x, dir_y, dir_z = record.final_direction[0], record.final_direction[1], record.final_direction[2]
            else:
                velocity_x = velocity_y = velocity_z = 0.0
                dir_x = dir_y = dir_z = 0.0
            
            # Detector hit position
            if record.detector_hit_position is not None:
                hit_x, hit_y, hit_z = record.detector_hit_position[0], record.detector_hit_position[1], record.detector_hit_position[2]
            else:
                hit_x = hit_y = hit_z = None
            
            # Final position
            if record.final_position is not None:
                final_x, final_y, final_z = record.final_position[0], record.final_position[1], record.final_position[2]
            else:
                final_x = final_y = final_z = None
            
            # Exit position (position after shell)
            exit_x, exit_y, exit_z = record.exit_position[0], record.exit_position[1], record.exit_position[2]
            
            # Convert TOF to nanoseconds for convenience
            tof_ns = record.tof * 1e9
            
            # Write row
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
    
    This exports the complete trajectory of each neutron including all collision points.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of neutron records from simulation.
    filename : str
        Output CSV filename.
    """
    import csv
    
    if not records:
        print("[warning] No neutron records to export.")
        return
    
    # Create output directory if it doesn't exist
    from pathlib import Path
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV headers
    headers = [
        'neutron_id',
        'step_id',
        'event_type',
        'position_x_m',
        'position_y_m',
        'position_z_m',
        'energy_MeV'
    ]
    
    # Open file and write data
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for neutron_id, record in enumerate(records, start=1):
            if record.trajectory_points is None or len(record.trajectory_points) == 0:
                # If no trajectory data, write at least the start and end points
                writer.writerow([neutron_id, 0, "source", 0.0, 0.0, 0.0, record.initial_energy])
                if record.final_position is not None:
                    writer.writerow([
                        neutron_id, 1, "final", 
                        record.final_position[0], record.final_position[1], record.final_position[2],
                        record.final_energy
                    ])
            else:
                # Write all trajectory points
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


################################################################################
# Visualization utilities
################################################################################

def visualize_neutron_data(records: List[NeutronRecord], save_path: Optional[str] = None):
    """Create comprehensive visualizations of neutron simulation results.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of ALL neutron records from simulation.
    save_path : str, optional
        Base path for saving figures. If None, figures are displayed but not saved.
    """
    if not records:
        print("[warning] No neutron records to visualize.")
        return
    
    # Extract data from ALL neutrons
    initial_energies = np.array([r.initial_energy for r in records])
    final_energies = np.array([r.final_energy for r in records])
    tofs = np.array([r.tof for r in records])
    energy_loss = initial_energies - final_energies
    
    # Color by status
    status_colors = []
    for r in records:
        if r.status == "success":
            status_colors.append('green')
        elif r.status == "lost_in_channel":
            status_colors.append('red')
        elif r.status == "lost_in_shell":
            status_colors.append('orange')
        else:
            status_colors.append('gray')
    status_colors = np.array(status_colors)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Energy histogram
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.hist(initial_energies, bins=50, alpha=0.7, label='Initial Energy', color='blue')
    ax1.hist(final_energies, bins=50, alpha=0.7, label='Final Energy', color='red')
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Count')
    ax1.set_title('All Neutron Energy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy loss
    ax2 = fig.add_subplot(3, 3, 2)
    energy_loss = initial_energies - final_energies
    ax2.hist(energy_loss, bins=50, color='green', alpha=0.7)
    ax2.set_xlabel('Energy Loss (MeV)')
    ax2.set_ylabel('Count')
    ax2.set_title('Energy Loss Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. TOF histogram
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(tofs * 1e9, bins=50, color='purple', alpha=0.7)  # Convert to ns
    ax3.set_xlabel('Time of Flight (ns)')
    ax3.set_ylabel('Count')
    ax3.set_title('TOF Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Initial vs Final Energy scatter
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(initial_energies, final_energies, alpha=0.5, s=30, c=tofs*1e9, cmap='viridis')
    ax4.plot([0, 3], [0, 3], 'r--', alpha=0.5, label='No energy loss')
    ax4.set_xlabel('Initial Energy (MeV)')
    ax4.set_ylabel('Final Energy (MeV)')
    ax4.set_title('Initial vs Final Energy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('TOF (ns)')
    
    # 5. Energy loss distribution
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.hist(energy_loss, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(energy_loss), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(energy_loss):.2f} MeV')
    ax5.set_xlabel('Energy Loss (MeV)')
    ax5.set_ylabel('Count')
    ax5.set_title('Energy Loss Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Initial Energy vs Energy Loss
    ax6 = fig.add_subplot(3, 3, 6)
    scatter = ax6.scatter(initial_energies, energy_loss, c=tofs*1e9, cmap='coolwarm', s=30, alpha=0.6)
    ax6.set_xlabel('Initial Energy (MeV)')
    ax6.set_ylabel('Energy Loss (MeV)')
    ax6.set_title('Initial Energy vs Energy Loss')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='TOF (ns)')
    
    # 7. Energy vs TOF
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.scatter(initial_energies, tofs * 1e9, alpha=0.5, s=20, label='Initial')
    ax7.scatter(final_energies, tofs * 1e9, alpha=0.5, s=20, label='Final')
    ax7.set_xlabel('Energy (MeV)')
    ax7.set_ylabel('TOF (ns)')
    ax7.set_title('Energy vs Time of Flight')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Energy Retention vs TOF
    ax8 = fig.add_subplot(3, 3, 8)
    retention = final_energies / initial_energies
    scatter = ax8.scatter(tofs * 1e9, retention, c=initial_energies, cmap='viridis', s=30, alpha=0.6)
    ax8.set_xlabel('TOF (ns)')
    ax8.set_ylabel('Energy Retention Fraction')
    ax8.set_title('Energy Retention vs TOF')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, label='Initial Energy (MeV)')
    
    # 9. Energy retention
    ax9 = fig.add_subplot(3, 3, 9)
    retention = final_energies / initial_energies
    ax9.hist(retention, bins=50, color='brown', alpha=0.7)
    ax9.set_xlabel('Energy Retention Fraction')
    ax9.set_ylabel('Count')
    ax9.set_title('Energy Retention (Final/Initial)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_comprehensive.png", dpi=300, bbox_inches='tight')
        print(f"[info] Saved comprehensive visualization to {save_path}_comprehensive.png")
    
    plt.show()


def visualize_detector_hits(records: List[NeutronRecord], detector_plane: DetectorPlane, 
                           save_path: Optional[str] = None):
    """Visualize neutron hit positions on the detector plane.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of neutron records from simulation.
    detector_plane : DetectorPlane
        Detector geometry information.
    save_path : str, optional
        Path for saving the figure.
    """
    # Filter records with detector hits
    hit_records = [r for r in records if r.detector_hit_position is not None]
    
    if not hit_records:
        print("[warning] No detector hits to visualize.")
        return
    
    hit_positions = np.array([r.detector_hit_position for r in hit_records])
    energies = np.array([r.final_energy for r in hit_records])
    tofs = np.array([r.tof for r in hit_records])
    
    # Project onto detector plane coordinates
    u_coords = np.array([np.dot(pos, detector_plane.u) for pos in hit_positions])
    v_coords = np.array([np.dot(pos, detector_plane.v) for pos in hit_positions])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hit map colored by energy
    ax1 = axes[0]
    scatter1 = ax1.scatter(u_coords, v_coords, c=energies, cmap='viridis', s=50, alpha=0.6)
    
    # Draw detector boundary
    if detector_plane.is_circular:
        from matplotlib.patches import Circle
        circle = Circle((0, 0), detector_plane.radius, fill=False, edgecolor='red', linewidth=2, label='Detector boundary')
        ax1.add_patch(circle)
    else:
        ax1.add_patch(Rectangle((-detector_plane.half_u, -detector_plane.half_v),
                                2*detector_plane.half_u, 2*detector_plane.half_v,
                                fill=False, edgecolor='red', linewidth=2, label='Detector boundary'))
    
    ax1.set_xlabel('U coordinate (m)')
    ax1.set_ylabel('V coordinate (m)')
    ax1.set_title('Detector Hit Map (colored by final energy)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Final Energy (MeV)')
    
    # Hit map colored by TOF
    ax2 = axes[1]
    scatter2 = ax2.scatter(u_coords, v_coords, c=tofs*1e9, cmap='plasma', s=50, alpha=0.6)
    
    # Draw detector boundary
    if detector_plane.is_circular:
        from matplotlib.patches import Circle
        circle = Circle((0, 0), detector_plane.radius, fill=False, edgecolor='red', linewidth=2, label='Detector boundary')
        ax2.add_patch(circle)
    else:
        ax2.add_patch(Rectangle((-detector_plane.half_u, -detector_plane.half_v),
                                2*detector_plane.half_u, 2*detector_plane.half_v,
                                fill=False, edgecolor='red', linewidth=2, label='Detector boundary'))
    
    ax2.set_xlabel('U coordinate (m)')
    ax2.set_ylabel('V coordinate (m)')
    ax2.set_title('Detector Hit Map (colored by TOF)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='TOF (ns)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_detector_hits.png", dpi=300, bbox_inches='tight')
        print(f"[info] Saved detector hit visualization to {save_path}_detector_hits.png")
    
    plt.show()


def print_statistics(records: List[NeutronRecord], n_total: int):
    """Print statistical summary of simulation results.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of ALL neutron records from simulation.
    n_total : int
        Total number of neutrons simulated.
    """
    if not records:
        print(f"\n[Statistics] No neutron records to display.")
        return
    
    # Separate successful and all records
    successful_records = [r for r in records if r.reached_detector]
    
    print("\n" + "="*60)
    print("ALL NEUTRON STATISTICS")
    print("="*60)
    print(f"Total neutrons simulated: {n_total}")
    print(f"Neutrons reaching detector: {len(successful_records)} ({100*len(successful_records)/n_total:.2f}%)")
    print(f"Neutrons lost in shell: {sum(1 for r in records if r.status == 'lost_in_shell')} ({100*sum(1 for r in records if r.status == 'lost_in_shell')/n_total:.2f}%)")
    print(f"Neutrons lost in channel: {sum(1 for r in records if r.status == 'lost_in_channel')} ({100*sum(1 for r in records if r.status == 'lost_in_channel')/n_total:.2f}%)")
    print(f"Neutrons missed detector: {sum(1 for r in records if r.status == 'missed_detector')} ({100*sum(1 for r in records if r.status == 'missed_detector')/n_total:.2f}%)")
    print()
    
    # Statistics for ALL neutrons
    initial_energies = np.array([r.initial_energy for r in records])
    initial_energies = np.array([r.initial_energy for r in records])
    final_energies = np.array([r.final_energy for r in records])
    energy_after_shell = np.array([r.energy_after_shell for r in records])
    energy_after_channel = np.array([r.energy_after_channel for r in records])
    tofs = np.array([r.tof for r in records])
    
    # Calculate energy losses at each stage
    loss_in_shell = initial_energies - energy_after_shell
    loss_in_channel = energy_after_shell - energy_after_channel
    print()
    print("Initial Energy (MeV):")
    print(f"  Mean: {np.mean(initial_energies):.4f}, Std: {np.std(initial_energies):.4f}")
    print(f"  Range: [{np.min(initial_energies):.4f}, {np.max(initial_energies):.4f}]")
    print()
    print("Energy Loss in Aluminum Shell (MeV):")
    print(f"  Mean: {np.mean(loss_in_shell):.4f}, Std: {np.std(loss_in_shell):.4f}")
    print(f"  Range: [{np.min(loss_in_shell):.4f}, {np.max(loss_in_shell):.4f}]")
    print()
    print("Energy Loss in Polyethylene Channel (MeV):")
    print(f"  Mean: {np.mean(loss_in_channel):.4f}, Std: {np.std(loss_in_channel):.4f}")
    print(f"  Range: [{np.min(loss_in_channel):.4f}, {np.max(loss_in_channel):.4f}]")
    print()
    print("Final Energy at Detector (MeV):")
    print(f"  Mean: {np.mean(final_energies):.4f}, Std: {np.std(final_energies):.4f}")
    print(f"  Range: [{np.min(final_energies):.4f}, {np.max(final_energies):.4f}]")
    print()
    print("Total Energy Loss (MeV):")
    energy_loss = initial_energies - final_energies
    print(f"  Mean: {np.mean(energy_loss):.4f}, Std: {np.std(energy_loss):.4f}")
    print(f"  Range: [{np.min(energy_loss):.4f}, {np.max(energy_loss):.4f}]")
    print()
    print("Time of Flight (ns):")
    print(f"  Mean: {np.mean(tofs)*1e9:.4f}, Std: {np.std(tofs)*1e9:.4f}")
    print(f"  Range: [{np.min(tofs)*1e9:.4f}, {np.max(tofs)*1e9:.4f}]")
    print("="*60 + "\n")


########################################################################################################
#################################################### main code ####################################################
########################################################################################################
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    AL_CSV_FILE = base_dir / "Al.csv" 
    H_CSV_FILE = base_dir / "H.csv" 
    C_CSV_FILE = base_dir / "C.csv" 
    
    # --- KERNEL SETUP ---
    # 1. 尝试加载用户提供的铝截面数据
    try:
        if AL_CSV_FILE.exists():
            # 使用新的加载函数加载真实数据
            aluminium_mfp_data = load_mfp_data_from_csv(str(AL_CSV_FILE))
            print(f"[info] Successfully loaded energy-dependent MFP data for Aluminium from {AL_CSV_FILE.name}")
        else:
            # 如果文件不存在，则退回使用占位符
            aluminium_mfp_data = MFP_DATA_AL
            print("[info] Using default internal MFP data for Aluminium.")
    except Exception as e:
        print(f"[warning] Failed to load custom Aluminium MFP data. Using default. Error: {e}")
        aluminium_mfp_data = MFP_DATA_AL

    # --- 2. 聚乙烯数据计算(用于 Channel) ---
    try:
        if H_CSV_FILE.exists() and C_CSV_FILE.exists():
            # 加载 H 和 C 的微观截面数据
            h_micro_data = load_mfp_data_from_csv(str(H_CSV_FILE))
            c_micro_data = load_mfp_data_from_csv(str(C_CSV_FILE))
            
            # 计算聚乙烯的组合宏观截面
            pe_data_calculated = calculate_pe_macro_sigma(h_micro_data, c_micro_data)
            
            channel_mfp_data = pe_data_calculated
            print(f"[info] Calculated Polyethylene MFP data from H.csv and C.csv.")
        else:
            # 如果文件不存在，则退回使用占位符
            channel_mfp_data = MFP_DATA_PE
            print("[info] Using default internal MFP data for Polyethylene.")
    except Exception as e:
        print(f"[warning] Failed to calculate/load custom Polyethylene MFP data. Using default. Error: {e}")
        channel_mfp_data = MFP_DATA_PE

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
    unit_scale = 1.0e-3  # Convert millimetres to metres
    mesh_scaled = shell_mesh * unit_scale
    channel_scaled = channel_mesh * unit_scale
    shell_geometry = prepare_mesh_geometry(mesh_scaled)
    channel_geometry = prepare_mesh_geometry(channel_scaled)
    mean_radius, max_radius = mesh_distance_statistics(mesh_scaled)
    
    # =========================================================================
    # COORDINATE SYSTEM CONFIGURATION (Updated)
    # =========================================================================
    # The STL files have been rotated to a new coordinate system:
    # - Polyethylene channel (nTOF_without_scintillant) is aligned along the +Z axis
    # - Detector (scintillator) plane is perpendicular to Z axis (in XY plane)
    # - Neutron source is at the origin (0, 0, 0)
    # - Neutrons travel in the +Z direction through the channel to the detector
    # =========================================================================
    
    # Channel axis direction: [0, 0, 1] (positive Z direction)
    channel_axis = np.array([0.0, 0.0, 1.0])
    
    # Build circular detector with updated coordinates
    # Detector is in XY plane at z = detector_z_mm
    # Centered at (0, 0, z) in the new coordinate system
    # Distance calculated from original geometry: sqrt(2679.25^2 + 1109.78^2) ≈ 2900 mm
    detector_z_mm = 2900.0  # Distance along channel axis (mm)
    detector_center_mm = np.array([0.0, 0.0, detector_z_mm])
    detector_radius_mm = 105.0  # Detector radius (mm)
    detector_plane = build_circular_detector_plane(detector_center_mm, detector_radius_mm, channel_axis)

    # Shell thickness is now calculated from actual STL mesh geometry
    # The value below is only used as a fallback if mesh intersection fails
    shell_thickness = 0.0  # Will be calculated from Target_ball_model.STL geometry

    print(f"[info] Shell thickness will be calculated from STL mesh geometry")
    print(
        f"[info] STL vertex distances: mean={mean_radius:.4f} m, "
        f"max={max_radius:.4f} m"
    )
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

    # Simulation parameters (adjust as needed)
    n_neutrons = 100  # Number of neutron histories to simulate
    aluminium_mass_ratio = 26.98  # Mass ratio A for aluminium
    channel_mass_ratio = 1.0  # Mass ratio for polyethylene (dominated by H)

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
    )

    # Print statistics
    print_statistics(neutron_records, n_neutrons)
    
    # Export neutron data to CSV
    if neutron_records:
        csv_filename = str(base_dir / "Data" / "neutron_data.csv")
        export_neutron_records_to_csv(neutron_records, filename=csv_filename)
        
        # Export trajectory data
        trajectory_filename = str(base_dir / "Data" / "neutron_trajectories.csv")
        export_neutron_trajectories_to_csv(neutron_records, filename=trajectory_filename)
    
    # Create visualizations
    if neutron_records:
        print("[info] Generating visualizations...")
        save_base = str(base_dir / "Figures" / "neutron_analysis")
        visualize_neutron_data(neutron_records, save_path= save_base)
        visualize_detector_hits(neutron_records, detector_plane, save_path= save_base)
        print("[info] Visualization complete!")
