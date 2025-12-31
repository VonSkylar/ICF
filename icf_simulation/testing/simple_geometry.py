"""
Simple Geometry Generators for Testing and Debugging
=====================================================

This module provides simplified analytical geometry models that can be used
as drop-in replacements for complex STL meshes. These simple geometries are
useful for:
- Debugging ray-tracing and intersection algorithms
- Verifying simulation physics without geometry complexity
- Performance testing
- Educational demonstrations

Each simple geometry function returns mesh data in the same format as
load_stl_mesh(), so they can be used interchangeably.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def create_simple_sphere(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 50.0,
    subdivisions: int = 3
) -> np.ndarray:
    """Create a simple spherical shell mesh using icosphere subdivision.
    
    Parameters
    ----------
    center : tuple[float, float, float], optional
        Center coordinates (x, y, z) in millimeters, by default (0, 0, 0)
    radius : float, optional
        Sphere radius in millimeters, by default 50.0
    subdivisions : int, optional
        Number of icosphere subdivisions (0-4), by default 3
        0: 20 faces (icosahedron)
        1: 80 faces
        2: 320 faces
        3: 1280 faces (good balance)
        4: 5120 faces (fine detail)
    
    Returns
    -------
    np.ndarray, shape (n_facets, 4, 3)
        Mesh in STL format: [normal, v0, v1, v2] for each facet
    """
    # Create icosahedron base
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    vertices = np.array(vertices, dtype=float)
    vertices = vertices / np.linalg.norm(vertices[0])
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        
        def get_midpoint(i: int, j: int) -> int:
            key = tuple(sorted([i, j]))
            if key not in edge_midpoints:
                v1, v2 = vertices[i], vertices[j]
                mid = (v1 + v2) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_midpoints[key] = len(vertices)
                vertices_list.append(mid)
            return edge_midpoints[key]
        
        vertices_list = list(vertices)
        
        for face in faces:
            v0, v1, v2 = face
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            
            new_faces.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c]
            ])
        
        vertices = np.array(vertices_list)
        faces = new_faces
    
    # Scale and translate vertices
    vertices = vertices * radius + np.array(center)
    
    # Compute facets with proper normals
    facets = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute face normal from edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        else:
            # Fallback: use average vertex normal (pointing outward from center)
            center_vec = np.array(center)
            normal = (v0 + v1 + v2) / 3.0 - center_vec
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
        
        facets.append(np.stack([normal, v0, v1, v2], axis=0))
    
    return np.stack(facets, axis=0)


def create_simple_tube(
    start_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    end_point: tuple[float, float, float] = (0.0, 0.0, 3000.0),
    inner_radius: float = 100.0,
    outer_radius: float = 110.0,
    n_segments: int = 32,
    close_ends: bool = True
) -> np.ndarray:
    """Create a simple cylindrical tube mesh.
    
    Parameters
    ----------
    start_point : tuple[float, float, float], optional
        Starting point (x, y, z) in millimeters, by default (0, 0, 0)
    end_point : tuple[float, float, float], optional
        Ending point (x, y, z) in millimeters, by default (0, 0, 3000)
    inner_radius : float, optional
        Inner radius in millimeters, by default 100.0
    outer_radius : float, optional
        Outer radius in millimeters, by default 110.0
    n_segments : int, optional
        Number of angular segments, by default 32
    close_ends : bool, optional
        Whether to close the tube ends, by default True
    
    Returns
    -------
    np.ndarray, shape (n_facets, 4, 3)
        Mesh in STL format: [normal, v0, v1, v2] for each facet
    """
    start = np.array(start_point, dtype=float)
    end = np.array(end_point, dtype=float)
    
    # Compute tube axis
    axis = end - start
    length = np.linalg.norm(axis)
    if length == 0:
        raise ValueError("start_point and end_point must be different")
    axis = axis / length
    
    # Create perpendicular basis vectors
    if abs(axis[2]) < 0.99:
        perp1 = np.cross(axis, [0, 0, 1])
    else:
        perp1 = np.cross(axis, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    
    # Generate circles at start and end
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    facets = []
    
    # Outer surface
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        
        # Vertices at start
        p0_start = start + outer_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
        p1_start = start + outer_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
        
        # Vertices at end
        p0_end = end + outer_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
        p1_end = end + outer_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
        
        # Two triangles forming a quad
        edge1 = p1_start - p0_start
        edge2 = p0_end - p0_start
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        facets.append(np.stack([normal, p0_start, p1_start, p0_end], axis=0))
        
        edge1 = p0_end - p1_start
        edge2 = p1_end - p1_start
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        facets.append(np.stack([normal, p1_start, p1_end, p0_end], axis=0))
    
    # Inner surface
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        
        # Vertices at start
        p0_start = start + inner_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
        p1_start = start + inner_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
        
        # Vertices at end
        p0_end = end + inner_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
        p1_end = end + inner_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
        
        # Two triangles (reversed winding for inner surface)
        edge1 = p0_end - p0_start
        edge2 = p1_start - p0_start
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        facets.append(np.stack([normal, p0_start, p0_end, p1_start], axis=0))
        
        edge1 = p1_end - p1_start
        edge2 = p0_end - p1_start
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        facets.append(np.stack([normal, p1_start, p1_end, p0_end], axis=0))
    
    # Close ends if requested
    if close_ends:
        # Start end (annulus)
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            
            p_outer = start + outer_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
            p_outer_next = start + outer_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
            p_inner = start + inner_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
            p_inner_next = start + inner_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
            
            # Two triangles forming annulus segment (face inward)
            edge1 = p_inner - p_outer
            edge2 = p_outer_next - p_outer
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            facets.append(np.stack([normal, p_outer, p_inner, p_outer_next], axis=0))
            
            edge1 = p_outer_next - p_inner
            edge2 = p_inner_next - p_inner
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            facets.append(np.stack([normal, p_inner, p_inner_next, p_outer_next], axis=0))
        
        # End end (annulus)
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            
            p_outer = end + outer_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
            p_outer_next = end + outer_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
            p_inner = end + inner_radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
            p_inner_next = end + inner_radius * (cos_angles[i_next] * perp1 + sin_angles[i_next] * perp2)
            
            # Two triangles forming annulus segment (face outward)
            edge1 = p_outer_next - p_outer
            edge2 = p_inner - p_outer
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            facets.append(np.stack([normal, p_outer, p_outer_next, p_inner], axis=0))
            
            edge1 = p_inner_next - p_inner
            edge2 = p_outer_next - p_inner
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            facets.append(np.stack([normal, p_inner, p_inner_next, p_outer_next], axis=0))
    
    return np.stack(facets, axis=0)


def create_simple_box(
    center: tuple[float, float, float] = (0.0, 0.0, 1500.0),
    size: tuple[float, float, float] = (200.0, 200.0, 3000.0)
) -> np.ndarray:
    """Create a simple rectangular box mesh.
    
    Parameters
    ----------
    center : tuple[float, float, float], optional
        Box center (x, y, z) in millimeters, by default (0, 0, 1500)
    size : tuple[float, float, float], optional
        Box dimensions (width, height, depth) in millimeters, by default (200, 200, 3000)
    
    Returns
    -------
    np.ndarray, shape (12, 4, 3)
        Mesh in STL format with 12 triangular facets (2 per box face)
    """
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    
    # 8 vertices of the box
    vertices = np.array([
        [cx - hx, cy - hy, cz - hz],  # 0: left-bottom-front
        [cx + hx, cy - hy, cz - hz],  # 1: right-bottom-front
        [cx + hx, cy + hy, cz - hz],  # 2: right-top-front
        [cx - hx, cy + hy, cz - hz],  # 3: left-top-front
        [cx - hx, cy - hy, cz + hz],  # 4: left-bottom-back
        [cx + hx, cy - hy, cz + hz],  # 5: right-bottom-back
        [cx + hx, cy + hy, cz + hz],  # 6: right-top-back
        [cx - hx, cy + hy, cz + hz],  # 7: left-top-back
    ])
    
    # 12 triangular faces (2 per box face)
    # Each face defined by 3 vertex indices
    faces = [
        # Front face (z = cz - hz)
        [0, 1, 2], [0, 2, 3],
        # Back face (z = cz + hz)
        [5, 4, 7], [5, 7, 6],
        # Left face (x = cx - hx)
        [4, 0, 3], [4, 3, 7],
        # Right face (x = cx + hx)
        [1, 5, 6], [1, 6, 2],
        # Bottom face (y = cy - hy)
        [4, 5, 1], [4, 1, 0],
        # Top face (y = cy + hy)
        [3, 2, 6], [3, 6, 7],
    ]
    
    facets = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        facets.append(np.stack([normal, v0, v1, v2], axis=0))
    
    return np.stack(facets, axis=0)


def create_simple_icf_geometry(
    geometry_type: Literal['minimal', 'standard', 'detailed'] = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """Create a complete simplified ICF geometry (shell + channel).
    
    This is a convenience function that creates a complete ICF-like geometry
    with appropriate dimensions matching the actual system. Use this for
    quick testing of the entire simulation chain.
    
    Parameters
    ----------
    geometry_type : {'minimal', 'standard', 'detailed'}, optional
        Level of mesh detail:
        - 'minimal': Coarse mesh for fast testing (few hundred faces)
        - 'standard': Medium mesh for debugging (few thousand faces)
        - 'detailed': Fine mesh approaching STL quality (many thousand faces)
    
    Returns
    -------
    shell_mesh : np.ndarray
        Spherical shell mesh (target ball)
    channel_mesh : np.ndarray
        Cylindrical channel mesh (nTOF detector channel)
    """
    subdivision_levels = {'minimal': 1, 'standard': 2, 'detailed': 3}
    segment_counts = {'minimal': 16, 'standard': 32, 'detailed': 64}
    
    subdivisions = subdivision_levels[geometry_type]
    n_segments = segment_counts[geometry_type]
    
    # Create spherical target shell
    # Typical ICF target: ~50 mm diameter, center at origin
    shell_mesh = create_simple_sphere(
        center=(0.0, 0.0, 0.0),
        radius=25.0,  # 50 mm diameter
        subdivisions=subdivisions
    )
    
    # Create cylindrical detector channel
    # Typical channel: inner radius ~100 mm, extends to ~3000 mm
    channel_mesh = create_simple_tube(
        start_point=(0.0, 0.0, 0.0),
        end_point=(0.0, 0.0, 3000.0),
        inner_radius=100.0,
        outer_radius=110.0,
        n_segments=n_segments,
        close_ends=True
    )
    
    return shell_mesh, channel_mesh


def print_mesh_info(mesh: np.ndarray, name: str = "Mesh") -> None:
    """Print diagnostic information about a mesh.
    
    Parameters
    ----------
    mesh : np.ndarray
        Mesh data in STL format
    name : str, optional
        Name to display, by default "Mesh"
    """
    n_facets = mesh.shape[0]
    
    if mesh.shape[1] == 4:
        vertices = mesh[:, 1:4, :].reshape(-1, 3)
    else:
        vertices = mesh.reshape(-1, 3)
    
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2
    
    print(f"\n{name} Information:")
    print(f"  Number of facets: {n_facets}")
    print(f"  Bounding box min: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) mm")
    print(f"  Bounding box max: ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}) mm")
    print(f"  Bounding box size: ({bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}) mm")
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm")
