"""
Geometry processing and ray-mesh intersection utilities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import DEBUG, GEOMETRY_LEAK_STATS
from .data_classes import MeshGeometry, DetectorPlane


def prepare_mesh_geometry(mesh: np.ndarray) -> MeshGeometry:
    """Precompute edge vectors and normals for an STL mesh.
    
    Parameters
    ----------
    mesh : np.ndarray
        Mesh data with shape (n_facets, 4, 3) where each facet contains
        [normal, v0, v1, v2], or (n_facets, 3, 3) with just vertices.
    """
    triangles = np.asarray(mesh, dtype=float)
    
    if triangles.shape[1] == 4:
        stl_normals = triangles[:, 0, :]
        v0 = triangles[:, 1, :]
        v1 = triangles[:, 2, :]
        v2 = triangles[:, 3, :]
    else:
        v0 = triangles[:, 0, :]
        v1 = triangles[:, 1, :]
        v2 = triangles[:, 2, :]
        edge1 = v1 - v0
        edge2 = v2 - v0
        stl_normals = np.cross(edge1, edge2)
        
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    norms = np.linalg.norm(stl_normals, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
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


def find_exit_with_retry(
    position: np.ndarray,
    direction: np.ndarray,
    geometry: MeshGeometry,
    max_retries: int = 5,
    base_offset: float = 1e-9,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Robustly find mesh exit point with geometry leak prevention.
    
    This function implements a fault-tolerant strategy to handle the common
    "Geometry Leak" problem in Monte Carlo particle transport.
    """
    global GEOMETRY_LEAK_STATS
    GEOMETRY_LEAK_STATS['total_queries'] += 1
    
    for retry in range(max_retries):
        offset = base_offset * (10 ** retry)
        search_origin = position + direction * offset
        
        exit_hit = ray_mesh_intersection(search_origin, direction, geometry)
        
        if exit_hit is not None:
            distance_from_search = exit_hit[0]
            total_distance = distance_from_search + offset
            exit_point = exit_hit[1]
            normal = exit_hit[2]
            
            if total_distance > 0:
                if retry > 0:
                    GEOMETRY_LEAK_STATS['retry_success'] += 1
                    if DEBUG:
                        print(f"[Geometry] Exit found on retry {retry}, offset={offset*1e9:.2f} nm")
                return (total_distance, exit_point, normal)
    
    opposite_hit = ray_mesh_intersection(position, -direction, geometry)
    
    if opposite_hit is None:
        GEOMETRY_LEAK_STATS['outside_detections'] += 1
        if DEBUG:
            print(f"[Geometry] Particle appears to be outside mesh, continuing forward")
        return (1e-6, position + direction * 1e-6, direction)
    
    for large_offset in [1e-6, 1e-5, 1e-4]:
        search_origin = position + direction * large_offset
        exit_hit = ray_mesh_intersection(search_origin, direction, geometry)
        
        if exit_hit is not None:
            GEOMETRY_LEAK_STATS['retry_success'] += 1
            if DEBUG:
                print(f"[Geometry Leak Warning] Exit found only with large offset: {large_offset*1e6:.2f} μm")
            total_distance = exit_hit[0] + large_offset
            if total_distance > 0:
                return (total_distance, exit_hit[1], exit_hit[2])
    
    GEOMETRY_LEAK_STATS['retry_failures'] += 1
    if DEBUG:
        print(f"[Geometry Leak ERROR] Failed to find exit after {max_retries} retries")
        print(f"  Position: {position}")
        print(f"  Direction: {direction}")
    
    return None


def mesh_distance_statistics(mesh: np.ndarray) -> Tuple[float, float]:
    """Return mean and maximum distance of mesh vertices from the origin."""
    triangles = np.asarray(mesh, dtype=float)
    
    if triangles.shape[1] == 4:
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
        vertices = triangles.reshape(-1, 3)
    
    if vertices.size == 0:
        raise ValueError("Mesh does not contain any vertices - cannot compute distances")
    radii = np.linalg.norm(vertices, axis=1)
    return float(np.mean(radii)), float(np.max(radii))


def infer_mesh_axis(mesh: np.ndarray) -> np.ndarray:
    """Infer the dominant axis of an STL mesh via principal component analysis."""
    triangles = np.asarray(mesh, dtype=float)
    
    if triangles.shape[1] == 4:
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
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
    """Build an orthonormal coordinate frame from a given axis."""
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
    """Build a detector plane from mesh geometry."""
    triangles = np.asarray(mesh, dtype=float)
    
    if triangles.shape[1] == 4:
        vertices = triangles[:, 1:4, :].reshape(-1, 3)
    else:
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
    """Build a default square detector plane."""
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
    center_m = center_mm * 1.0e-3
    radius_m = radius_mm * 1.0e-3
    
    axis_norm, u, v = build_orthonormal_frame(axis)
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
