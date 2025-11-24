"""
icf_neutron_simulation
======================

This module provides a simple, self‑contained Monte‑Carlo simulator for the
time‑of‑flight (TOF) behaviour of neutrons in an inertial confinement fusion
(ICF) diagnostic.  It follows, as closely as practicable, the high level
algorithm outlined in the provided project document.  Each physical stage of
the neutron’s journey is encapsulated in its own function to aid testing and
reuse.

The basic workflow is as follows:

1. **Neutron generation** – Neutrons are produced at the centre of the
   implosion target.  Their kinetic energy is sampled from a Gaussian  
   distribution centred on 2.45 MeV (corresponding to DD fusion) and their
   direction is sampled isotropically in three dimensions.
2. **Transport through the aluminium shell** – The aluminium shell is
   described by an STL mesh (or, if a mesh is not available, approximated by
   a spherical shell of known thickness).  Neutrons undergo a sequence of  
   elastic scatterings in the shell.  Free flight distances are drawn from
   an exponential distribution with a specified mean free path.  After each
   scattering the neutron’s direction is randomised and its energy is
   updated using a simple two‑body kinematic model.  Once the cumulative
   radial distance travelled exceeds the shell thickness or the energy drops
   below 0.1 MeV the neutron leaves the shell or is absorbed, respectively.
3. **Flight through the nTOF channel** – After emerging from the shell, the
   neutron may intersect the polyethylene structures described by ``nTOF.STL``.
   If it does, Monte-Carlo collisions with the polyethylene nuclei are simulated
   using the supplied mean free path; otherwise the neutron travels in vacuum to
   the scintillator plane 16 m away.
4. **Energy deposition in the scintillator** – Neutrons that reach the
   scintillator undergo further collisions in the scintillator medium until
   they either deposit most of their energy (E < 0.1 MeV) or escape.

The functions provided here are intentionally generic: key parameters such as
shell thickness, mean free paths and material mass ratios are exposed as
arguments.  Users can therefore substitute real material data (for example
obtained from ENDF or JANIS) without changing the overall structure of the
code.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# 阿伏伽德罗常数 (mol⁻¹)
AVOGADRO_CONSTANT = 6.02214076e23 
# barn 到 m² 的转换 (1 barn = 1e-28 m²)
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
                v0 = np.array(unpacked[3:6], dtype=float)
                v1 = np.array(unpacked[6:9], dtype=float)
                v2 = np.array(unpacked[9:12], dtype=float)
                facets_bin.append(np.stack([v0, v1, v2], axis=0))
        if not facets_bin:
            return None
        return np.stack(facets_bin, axis=0)

    def _try_load_ascii() -> Optional[np.ndarray]:
        facets_ascii: List[np.ndarray] = []
        current_vertices: List[np.ndarray] = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as af:
            for line in af:
                tokens = line.strip().split()
                if not tokens:
                    continue
                keyword = tokens[0].lower()
                if keyword == 'facet':
                    current_vertices = []
                elif keyword == 'vertex' and len(tokens) >= 4:
                    v = np.array(list(map(float, tokens[1:4])), dtype=float)
                    current_vertices.append(v)
                elif keyword == 'endfacet':
                    if len(current_vertices) >= 3:
                        facets_ascii.append(np.stack(current_vertices[:3], axis=0))
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


def prepare_mesh_geometry(mesh: np.ndarray) -> MeshGeometry:
    """Precompute edge vectors and normals for an STL mesh."""
    triangles = np.asarray(mesh, dtype=float)
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    normals = np.cross(edge1, edge2)
    return MeshGeometry(vertices0=v0, edge1=edge1, edge2=edge2, normals=normals)


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
    pts = np.asarray(mesh, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        raise ValueError("Mesh does not contain any vertices - cannot compute distances")
    radii = np.linalg.norm(pts, axis=1)
    return float(np.mean(radii)), float(np.max(radii))


def infer_mesh_axis(mesh: np.ndarray) -> np.ndarray:
    """Infer the dominant axis of an STL mesh via principal component analysis."""
    pts = np.asarray(mesh, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("Not enough vertices to infer mesh axis")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
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
    pts = np.asarray(mesh, dtype=float).reshape(-1, 3)
    axis, u, v = build_orthonormal_frame(axis)
    projections = pts @ axis
    plane_position = float(projections.max())
    mask = projections > plane_position - tolerance
    subset = pts[mask] if np.any(mask) else pts
    u_extent = float(np.max(np.abs(subset @ u)))
    v_extent = float(np.max(np.abs(subset @ v)))
    u_extent = max(u_extent, 1e-6)
    v_extent = max(v_extent, 1e-6)
    return DetectorPlane(axis, u, v, plane_position, u_extent, v_extent)


def build_default_detector_plane(distance: float, side: float) -> DetectorPlane:
    axis, u, v = build_orthonormal_frame(np.array([0.0, 0.0, 1.0]))
    half = side / 2.0
    return DetectorPlane(axis, u, v, distance, half, half)


def transport_through_slab(
    energy_mev: float,
    slab_thickness: float,
    mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    target_mass_ratio: float,
    energy_cutoff_mev: float,
    initial_direction: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Propagate a neutron through a homogeneous slab with multiple scatterings."""
    remaining = slab_thickness
    energy = float(energy_mev)
    cumulative_time = 0.0

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

    # --- 在循环开始前计算当前的 MFP ---
    mean_free_path = get_mfp_energy_dependent(energy,mfp_data)


    while energy > energy_cutoff_mev and remaining > 0.0:
        free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
        step = min(free_path, remaining)
        cumulative_time += step / speed
        remaining -= step

        if step < free_path:
            energy = scatter_energy_elastic(energy, target_mass_ratio)
            current_dir = sample_isotropic_direction()
            speed = energy_to_speed(energy)
            mean_free_path = get_mfp_energy_dependent(energy, mfp_data)
        else:
            break
    return cumulative_time, energy, current_dir


def propagate_through_mesh_material(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    geometry: Optional[MeshGeometry],
    mean_free_path: float,
    target_mass_ratio: float,
    energy_cutoff_mev: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Advance a neutron through a mesh-defined solid material."""
    if geometry is None:
        return 0.0, energy_mev, np.array(position, dtype=float), np.array(direction, dtype=float)

    origin = np.array(position, dtype=float)
    dir_norm = np.array(direction, dtype=float)
    dir_norm /= np.linalg.norm(dir_norm)
    first_hit = ray_mesh_intersection(origin, dir_norm, geometry)
    if first_hit is None:
        return 0.0, energy_mev, origin, dir_norm

    entry_dist, entry_point, normal = first_hit
    speed = energy_to_speed(energy_mev)
    cumulative_time = entry_dist / speed
    inside_origin = entry_point + dir_norm * 1e-6
    exit_hit = ray_mesh_intersection(inside_origin, dir_norm, geometry)
    if exit_hit is None:
        return cumulative_time, energy_cutoff_mev, inside_origin, dir_norm
    path_inside = exit_hit[0]
    exit_point = inside_origin + dir_norm * path_inside

    slab_time, energy_out, direction_out = transport_through_slab(
        energy_mev,
        path_inside,
        mean_free_path,
        target_mass_ratio,
        energy_cutoff_mev,
        initial_direction=dir_norm,
    )
    cumulative_time += slab_time
    return cumulative_time, energy_out, exit_point, direction_out


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

        r = [ (A − 1)² + 2(A + 1) cos θ + 1 ] / (A + 1)²

    where θ is a random angle uniformly distributed in [0, π].  After the
    collision the neutron energy becomes E′ = r E.  This formula ignores
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
    # Energy cannot be negative; ensure r≥0
    r = max(r, 0.0)
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
    rho_kg_m3 = density_g_cm3 * 1000 # kg/m³ (或保持 g/cm³, 最终单位调整)
    
    # 我们使用 M_PE (g/mol) 和 rho (g/cm³) 进行计算，最终转换为 m⁻³
    # N_i (cm⁻³) = (rho_g_cm3 * N_A * n_i) / M_PE
    # N_i (m⁻³) = N_i (cm⁻³) * 1e6
    
    N_C_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 2) / M_PE
    N_H_cm3 = (density_g_cm3 * AVOGADRO_CONSTANT * 4) / M_PE
    
    N_C_m3 = N_C_cm3 * 1e6
    N_H_m3 = N_H_cm3 * 1e6

    # 3. 创建统一的能量网格
    all_energies = np.unique(np.concatenate([h_micro_data[:, 0], c_micro_data[:, 0]]))
    
    # 4. 插值微观截面 (Sigma_Micro) 到统一网格
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
# 聚乙烯 (C₂H₄, 密度 ≈ 0.92 g/cm³)
MFP_DATA_PE = np.array([
    [0.1, 15.0],  # 0.1 MeV 时的宏观截面 (m⁻¹)
    [0.5, 13.5],
    [1.0, 10.0],
    [2.45, 16.6], # 2.45 MeV 基准点
    [5.0, 17.5],
    [10.0, 18.0],
    [14.1, 17.8], # D-T 中子能量
])

# 铝 (Al, 密度 ≈ 2.70 g/cm³)
MFP_DATA_AL = np.array([
    [0.1, 10.0],
    [0.5, 11.2],
    [1.0, 13.0],
    [2.45, 14.4], # 2.45 MeV 基准点
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
        中子的当前动能 (MeV)。
    mfp_data : np.ndarray
        预设的 [能量 (MeV), 宏观截面 (m⁻¹)] 数据对数组。

    返回
    -------
    float
        新的平均自由程 (m)。
    """
    if energy_mev <= 0.0:
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
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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
        * **time** – the total time spent from the target centre to the aluminium
          exit point (s).
        * **energy** – the neutron energy after leaving the shell (MeV).
        * **direction** – the final direction unit vector when exiting the shell.
        * **position** – 3‑D position (m) of the exit point relative to the origin.
    """
    direction = np.array(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("Direction vector must be non-zero")
    direction = direction / norm

    energy = float(energy_mev)
    speed = energy_to_speed(energy)
    cumulative_time = 0.0

    # If a mesh is supplied, determine whether the neutron encounters aluminium.
    origin = np.zeros(3, dtype=float)

    if mesh_geometry is None:
        # Simple slab approximation when no geometry is supplied.
        slab_time, energy, current_dir = transport_through_slab(
            energy,
            shell_thickness,
            aluminium_mfp_data, # 传入数据数组
            target_mass_ratio,
            energy_cutoff_mev,
            initial_direction=direction,
        )
        cumulative_time += slab_time
        position = origin + direction * shell_thickness
        return cumulative_time, energy, current_dir, position

    # Detailed treatment using STL geometry
    hit = ray_mesh_intersection(origin, direction, mesh_geometry)
    if hit is None:
        # Trajectory leaves through an opening without touching aluminium.
        return cumulative_time, energy, direction, origin.copy()

    distance_to_outer, outer_point, normal = hit
    normal_len = np.linalg.norm(normal)
    if normal_len > 0.0:
        normal = normal / normal_len
    cos_incident = float(np.dot(direction, normal))

    # Ensure neutron is travelling towards the slab (positive dot product with normal)
    if cos_incident <= 0.0:
        normal = -normal
        cos_incident = -cos_incident
    cos_incident = max(cos_incident, 1e-6)

    path_length_along_direction = shell_thickness / cos_incident
    distance_to_entry = max(distance_to_outer - path_length_along_direction, 0.0)
    cumulative_time += distance_to_entry / speed
    position = origin + direction * distance_to_entry
    
    
    # Slab simulation inside the mesh material
    # We call transport_through_slab to simulate the interaction in the slab material
    slab_time, energy_out, direction_out = transport_through_slab(
        energy_mev,
        path_length_along_direction, # 使用沿方向的路径长度作为 Slab 厚度
        aluminium_mfp_data, # 传入数据数组
        aluminium_mass_ratio,
        energy_cutoff_mev,
        initial_direction=direction,
    )
    
    # Compute exit point based on initial direction and total path length
    cumulative_time += slab_time
    exit_point = position + direction * path_length_along_direction
    
    return cumulative_time, energy_out, direction_out, exit_point


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

    # A neutron must be travelling towards the detector (positive z)
    if d[2] <= 0.0:
        return None, None

    if energy_mev <= energy_cutoff_mev:
        return None, None

    speed = energy_to_speed(energy_mev)
    axis = detector_plane.axis
    u = detector_plane.u
    v = detector_plane.v
    plane_pos = detector_plane.plane_position

    dir_dot = float(np.dot(d, axis))
    if dir_dot <= 0.0:
        return None, None

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


################################################################################
# Energy deposition in the scintillator
################################################################################

def simulate_in_scintillator(
    energy_mev: float,
    scintillator_thickness: float,
    scintillator_mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    target_mass_ratio: float,
    energy_cutoff_mev: float = 0.1,
) -> Tuple[float, float]:
    """Simulate neutron slowing down and energy deposition in the scintillator.

    The scintillator is modelled as a homogeneous slab of finite thickness.
    Within the slab, neutrons undergo a series of elastic scatterings with
    hydrogen (or other) nuclei.  The free flight distances are sampled from
    an exponential distribution with mean ``mean_free_path``.  After each
    collision the neutron’s energy is updated using the kinematics in
    :func:`scatter_energy_elastic`.  The simulation ends when either the
    neutron’s cumulative path length exceeds ``scintillator_thickness`` or
    its energy falls below ``energy_cutoff_mev``.  Any energy below the
    cutoff is assumed to be deposited immediately and the neutron is lost.

    Parameters
    ----------
    energy_mev : float
        The neutron energy on entering the scintillator.
    scintillator_thickness : float
        Thickness of the scintillator slab in metres.
    mean_free_path : float
        Mean free path for neutron scattering in the scintillator material.
        For organic scintillators (rich in hydrogen) this might be several
        centimetres.
    target_mass_ratio : float
        Mass ratio A of the dominant scattering nucleus in the scintillator to
        the neutron mass.  For hydrogen A=1; for carbon A≈12.
    energy_cutoff_mev : float, optional
        Energies below this threshold are treated as full energy deposition.
        Defaults to 0.1 MeV.

    Returns
    -------
    tuple of (time, final_energy)
        * **time** – the time (s) spent inside the scintillator before the
          neutron is absorbed or exits.
        * **final_energy** – the neutron energy (MeV) on leaving the
          scintillator.  If this is below the cutoff, the neutron is
          considered absorbed.
    """
    energy = float(energy_mev)
    cumulative_distance = 0.0
    cumulative_time = 0.0
    speed = energy_to_speed(energy)
    mean_free_path = get_mfp_energy_dependent(energy,scintillator_mfp_data,)

    while energy > energy_cutoff_mev and cumulative_distance < scintillator_thickness:
        free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
        remaining = scintillator_thickness - cumulative_distance
        step = min(free_path, remaining)
        cumulative_time += step / speed
        cumulative_distance += step

        if cumulative_distance >= scintillator_thickness:
            break
        # Scatter – energy update and isotropic direction.  For simplicity the
        # direction within the scintillator is irrelevant to the time since
        # the mean free path is isotropic; if one wanted to compute spatial
        # escape probabilities the direction would need to be tracked.

        # 碰撞发生
        energy = scatter_energy_elastic(energy, target_mass_ratio)
        speed = energy_to_speed(energy)
        mean_free_path = get_mfp_energy_dependent(energy,scintillator_mfp_data)

    return cumulative_time, energy


################################################################################
# High level simulation driver
################################################################################

def simulate_neutron_history(
    shell_thickness: float,
    aluminium_mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    aluminium_mass_ratio: float,
    scintillator_thickness: float,
    scintillator_mfp_data: np.ndarray, # 修改: 传入 MFP 数据数组
    scintillator_mass_ratio: float,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
    channel_geometry: Optional[MeshGeometry] = None,
    channel_mfp_data: np.ndarray = MFP_DATA_PE, # 修改: 传入 MFP 数据数组
    channel_mass_ratio: float = 1.0,
    source_cone_axis: Optional[np.ndarray] = None,
    source_cone_half_angle_deg: float = 15.0,
    detector_plane: Optional[DetectorPlane] = None,
) -> Optional[float]:
    """Simulate the complete history of a single neutron.

    This function ties together all the individual stages: generation at the
    target centre, transport through the aluminium shell, interaction with the
    polyethylene channel, and final slowing down in the scintillator.  If the
    neutron survives to deposit energy in the scintillator, the function
    returns its total time‑of‑flight (source to detector).  Otherwise it
    returns ``None`` to indicate that the neutron was lost or absorbed.

    Parameters
    ----------
    shell_thickness : float
        Aluminium shell thickness (m).
    aluminium_mfp : float
        Mean free path for neutron collisions in aluminium (m).
    aluminium_mass_ratio : float
        Aluminium nucleus mass divided by neutron mass (A≈26.98).
    scintillator_thickness : float
        Thickness of the scintillator (m).
    scintillator_mfp : float
        Mean free path for neutron collisions in the scintillator (m).
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
    float or None
        The total time‑of‑flight in seconds if the neutron deposits energy in
        the scintillator, otherwise ``None``.
    """
    # 1. Generate initial energy and direction
    E0 = sample_neutron_energy()
    if source_cone_axis is None:
        d0 = sample_isotropic_direction()
    else:
        d0 = sample_direction_in_cone(source_cone_axis, source_cone_half_angle_deg)

    # 2. Transport through the aluminium shell
    detector_plane = detector_plane or build_default_detector_plane(detector_distance, detector_side)

    t_shell, E_after_shell, d_after_shell, pos_after_shell = simulate_in_aluminium(
        d0,
        E0,
        shell_thickness,
        aluminium_mfp_data, # 传入数据数组
        aluminium_mass_ratio,
        energy_cutoff_mev,
        mesh_geometry=shell_geometry,
    )
    # If energy is below cutoff the neutron was absorbed in the shell
    if E_after_shell <= energy_cutoff_mev:
        return None

    # 3. Interaction with the polyethylene channel
    t_channel, E_after_channel, pos_after_channel, d_after_channel = propagate_through_mesh_material(
        pos_after_shell,
        d_after_shell,
        E_after_shell,
        channel_geometry,
        channel_mfp_data, # 传入数据数组
        channel_mass_ratio,
        energy_cutoff_mev,
    )
    if E_after_channel <= energy_cutoff_mev:
        return None

    # 4. Flight to the scintillator
    flight_time, hit_point = propagate_to_scintillator(
        pos_after_channel,
        d_after_channel,
        E_after_channel,
        detector_plane,
        energy_cutoff_mev=energy_cutoff_mev,
    )
    if flight_time is None:
        return None

    # 5. Energy deposition in the scintillator
    t_scin, E_after_scin = simulate_in_scintillator(
        E_after_channel,
        scintillator_thickness,
        scintillator_mfp_data, # 传入数据数组
        scintillator_mass_ratio,
        energy_cutoff_mev,
    )
    # The neutron is considered to contribute to the signal regardless of
    # whether it is absorbed or escapes from the scintillator; the TOF is
    # measured at the moment of first interaction.
    return t_shell + t_channel + flight_time + t_scin


def run_simulation(
    n_neutrons: int,
    shell_thickness: float,
    aluminium_mass_ratio: float,
    scintillator_thickness: float,
    scintillator_mass_ratio: float,
    scintillator_mfp_data: np.ndarray = MFP_DATA_PE,
    aluminium_mfp_data: np.ndarray = MFP_DATA_AL,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
    channel_geometry: Optional[MeshGeometry] = None,
    channel_mfp_data: np.ndarray = MFP_DATA_PE,
    channel_mass_ratio: float = 1.0,
    source_cone_axis: Optional[np.ndarray] = None,
    source_cone_half_angle_deg: float = 15.0,
    detector_plane: Optional[DetectorPlane] = None,
) -> List[float]:
    """Simulate multiple neutrons and return their time‑of‑flight values.

    This function repeatedly calls :func:`simulate_neutron_history` and
    collects the returned times for neutrons that deposit energy in the
    scintillator.  It provides a simple entry point for generating a TOF
    spectrum.

    Parameters
    ----------
    n_neutrons : int
        Number of neutron histories to simulate.
    All other parameters
        These are passed directly to :func:`simulate_neutron_history`.

    Returns
    -------
    list of float
        A list containing the time‑of‑flight for each neutron that reached the
        scintillator.  Neutrons that were lost or absorbed do not appear in
        this list.
    """
    tof_values: List[float] = []
    for _ in range(n_neutrons):
        tof = simulate_neutron_history(
            shell_thickness,
            aluminium_mfp_data,
            aluminium_mass_ratio,
            scintillator_thickness,
            scintillator_mfp_data,
            scintillator_mass_ratio,
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
        if tof is not None:
            tof_values.append(tof)
    return tof_values


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

    # --- 2. 聚乙烯数据计算 (用于 Channel 和 Scintillator) ---
    try:
        if H_CSV_FILE.exists() and C_CSV_FILE.exists():
            # 加载 H 和 C 的微观截面数据 (假设 use_cols=(0, 2) 对应 [Energy, Micro_Sigma (barn)])
            h_micro_data = load_mfp_data_from_csv(str(H_CSV_FILE))
            c_micro_data = load_mfp_data_from_csv(str(C_CSV_FILE))
            
            # 计算聚乙烯的组合宏观截面
            pe_data_calculated = calculate_pe_macro_sigma(h_micro_data, c_micro_data)
            
            channel_mfp_data = pe_data_calculated
            scintillator_mfp_data = pe_data_calculated
            print(f"[info] Calculated Polyethylene MFP data from H.csv and C.csv.")
        else:
            # 如果文件不存在，则退回使用占位符
            channel_mfp_data = MFP_DATA_PE
            scintillator_mfp_data = MFP_DATA_PE
            print("[info] Using default internal MFP data for Polyethylene.")
    except Exception as e:
        print(f"[warning] Failed to calculate/load custom Polyethylene MFP data. Using default. Error: {e}")
        channel_mfp_data = MFP_DATA_PE
        scintillator_mfp_data = MFP_DATA_PE

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
    channel_mesh_path = base_dir / "nTOF.STL"
    channel_mesh = load_stl_mesh(str(channel_mesh_path))
    unit_scale = 1.0e-3  # Convert millimetres to metres
    mesh_scaled = shell_mesh * unit_scale
    channel_scaled = channel_mesh * unit_scale
    shell_geometry = prepare_mesh_geometry(mesh_scaled)
    channel_geometry = prepare_mesh_geometry(channel_scaled)
    mean_radius, max_radius = mesh_distance_statistics(mesh_scaled)
    channel_axis = infer_mesh_axis(channel_scaled)
    detector_plane = build_detector_plane_from_mesh(channel_scaled, channel_axis)

    shell_thickness = 0.08  # metres (given design thickness)

    print(f"[info] Using specified shell thickness of {shell_thickness:.3f} m")
    print(
        f"[info] STL vertex distances: mean={mean_radius:.4f} m, "
        f"max={max_radius:.4f} m"
    )
    print(f"[info] Inferred channel axis: {channel_axis}")
    print(
        "[info] Detector plane: position={:.4f} m, half-sizes=({:.4f}, {:.4f}) m".format(
            detector_plane.plane_position,
            detector_plane.half_u,
            detector_plane.half_v,
        )
    )

    # Simulation parameters (adjust as needed)
    n_neutrons = 3000
    #aluminium_mfp = 0.0694  # 6.94 cm mean free path in aluminium
    scintillator_thickness = 0.05  # Scintillator thickness (m)
    #scintillator_mfp = 0.01  # Mean free path in the scintillator (m)
    aluminium_mass_ratio = 26.98  # Mass ratio A for aluminium
    scintillator_mass_ratio = 1.0  # Dominant scattering nucleus (hydrogen)
    channel_mfp = 0.0602  # 6.02 cm in polyethylene
    channel_mass_ratio = 1.0

    tof_list = run_simulation(
        n_neutrons=n_neutrons,
        shell_thickness=shell_thickness,
        aluminium_mfp_data=MFP_DATA_AL,
        aluminium_mass_ratio=aluminium_mass_ratio,
        scintillator_thickness=scintillator_thickness,
        scintillator_mass_ratio=scintillator_mass_ratio,
        detector_distance=16.0,
        detector_side=1.0,
        energy_cutoff_mev=0.1,
        shell_geometry=shell_geometry,
        channel_geometry=channel_geometry,
        channel_mfp_data=MFP_DATA_PE,
        channel_mass_ratio=channel_mass_ratio,
        source_cone_axis=channel_axis,
        detector_plane=detector_plane,
    )

    print(f"Simulated {len(tof_list)} neutrons reaching the scintillator out of {n_neutrons}")
    if tof_list:
        print(f"Mean time of flight: {np.mean(tof_list):.3e} s")
