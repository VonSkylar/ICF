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
3. **Flight to the scintillator** – After emerging from the shell, the
   neutron travels in (effectively) vacuum to a scintillator located
   16 m away.  A square PVC collimating channel of length 16 m and
   cross-section 1 m × 1 m sits between the target and the detector.  A
   neutron whose trajectory intersects the channel walls is lost with a
   high probability (default 99 %); only those that remain within the
   aperture reach the scintillator.
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
    """Generate a random unit vector isotropically distributed on the sphere.

    The method uses the fact that the cosine of the polar angle is uniformly
    distributed in [−1, 1] and the azimuth is uniformly distributed in
    [0, 2π].

    Returns
    -------
    np.ndarray, shape (3,)
        A unit vector pointing in a random direction.
    """
    z = 2.0 * np.random.rand() - 1.0  # cos(theta) uniformly in [−1,1]
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z], dtype=float)


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
# Transport through the aluminium shell
################################################################################

def simulate_in_aluminium(
    direction: np.ndarray,
    energy_mev: float,
    shell_thickness: float,
    mean_free_path: float,
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
        effective_path = shell_thickness
        current_dir = direction.copy()
        position = origin.copy()
        position += current_dir * effective_path
        remaining_path = effective_path
        while energy > energy_cutoff_mev and remaining_path > 0.0:
            free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
            step_length = min(free_path, remaining_path)
            cumulative_time += step_length / speed
            remaining_path -= step_length
            if step_length < free_path:
                energy = scatter_energy_elastic(energy, target_mass_ratio)
                current_dir = sample_isotropic_direction()
                speed = energy_to_speed(energy)
            else:
                break
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
    if cos_incident <= 0.0:
        normal = -normal
        cos_incident = -cos_incident
    cos_incident = max(cos_incident, 1e-6)

    path_length_along_direction = shell_thickness / cos_incident
    distance_to_entry = max(distance_to_outer - path_length_along_direction, 0.0)
    cumulative_time += distance_to_entry / speed
    position = origin + direction * distance_to_entry
    depth = 0.0  # distance travelled along +normal from inner surface

    EPS = 1e-9
    position += direction * EPS
    depth += EPS * cos_incident

    current_dir = direction.copy()
    exit_position = position.copy()

    while energy > energy_cutoff_mev:
        free_path = -mean_free_path * math.log(max(1e-12, np.random.rand()))
        dot = float(np.dot(current_dir, normal))
        travel = free_path
        boundary = None

        if dot > 1e-9:
            dist_outer = (shell_thickness - depth) / dot
            dist_outer = max(dist_outer, 0.0)
            if dist_outer <= free_path:
                travel = dist_outer
                boundary = "outer"
        elif dot < -1e-9:
            dist_inner = -depth / dot
            dist_inner = max(dist_inner, 0.0)
            if dist_inner <= free_path:
                travel = dist_inner
                boundary = "inner"

        if travel > 0.0:
            position = position + current_dir * travel
            cumulative_time += travel / speed
            depth += travel * dot
            depth = min(max(depth, 0.0), shell_thickness)

        if boundary == "outer":
            exit_position = position.copy()
            return cumulative_time, energy, current_dir, exit_position
        if boundary == "inner":
            # Returned to the cavity; treat as leaving the shell inward.
            return cumulative_time, energy, current_dir, position

        # Collision within aluminium
        energy = scatter_energy_elastic(energy, target_mass_ratio)
        current_dir = sample_isotropic_direction()
        speed = energy_to_speed(energy)

    return cumulative_time, energy, current_dir, position


################################################################################
# Flight from shell to scintillator
################################################################################

def propagate_to_scintillator(
    position: np.ndarray,
    direction: np.ndarray,
    energy_mev: float,
    distance_to_detector: float = 16.0,
    detector_side: float = 1.0,
    collimator_absorption: float = 0.99,
    energy_cutoff_mev: float = 0.1,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Propagate a neutron from the shell exit to the scintillator.

    The scintillator is modelled as a square detector of side length
    ``detector_side`` located a distance ``distance_to_detector`` metres away
    along the +z axis from the source centre.  Between the source and detector
    is a square PVC collimating channel of the same cross-section.  A neutron
    starting from ``position`` must travel entirely within the channel aperture
    to reach the scintillator; should it strike the channel wall it is absorbed
    with probability ``collimator_absorption`` (default 99 %).  Rare survivors
    are assumed to continue along their original trajectory.

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Starting position in metres (typically the shell exit point).
    direction : np.ndarray, shape (3,)
        Unit vector giving the direction of motion leaving the aluminium shell.
    energy_mev : float
        Kinetic energy in MeV when leaving the aluminium shell.
    distance_to_detector : float, optional
        Distance from the neutron source (shell centre) to the scintillator
        surface in metres.  Defaults to 16 m.
    detector_side : float, optional
        Side length of the square scintillator in metres.  Defaults to 1 m.
    collimator_absorption : float, optional
        Probability that a neutron striking the collimator is absorbed.  The
        default value of 0.99 corresponds to a 99 % absorption probability.
    energy_cutoff_mev : float, optional
        Energies below this threshold are considered lost.  If a neutron has
        too little energy to travel the full distance, it is discarded.

    Returns
    -------
    tuple of (flight_time, hit_point) or (None, None)
        If the neutron reaches the scintillator, ``flight_time`` is the time
        (s) spent in flight from the shell exit to the scintillator, and
        ``hit_point`` is the 3‑D position on the detector plane where it
        arrives.  If the neutron misses the detector or is absorbed in the
        collimator, both return values are ``None``.
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
    half_size = detector_side / 2.0

    flight_time = 0.0

    # Move to the start of the collimator (z = 0) if the neutron exits behind it.
    if pos[2] < 0.0:
        t_entry = (0.0 - pos[2]) / d[2]
        if t_entry < 0.0:
            return None, None
        flight_time += t_entry / speed
        pos = pos + d * t_entry

    # Check if the entry point lies within the channel aperture.
    if abs(pos[0]) > half_size or abs(pos[1]) > half_size:
        if np.random.rand() < collimator_absorption:
            return None, None
        pos[0] = max(min(pos[0], half_size), -half_size)
        pos[1] = max(min(pos[1], half_size), -half_size)

    # Determine potential wall intersections along the remaining path.
    def _wall_hit_z(component: float, slope: float) -> Optional[float]:
        if abs(slope) < 1e-12:
            return None
        hits: List[float] = []
        for sign in (-1.0, 1.0):
            numerator = sign * half_size - component
            z_hit = pos[2] + numerator / slope
            if z_hit > pos[2] + 1e-9 and z_hit < distance_to_detector - 1e-9:
                hits.append(z_hit)
        if not hits:
            return None
        return min(hits)

    slope_x = d[0] / d[2]
    slope_y = d[1] / d[2]
    wall_candidates = [z for z in (_wall_hit_z(pos[0], slope_x), _wall_hit_z(pos[1], slope_y)) if z is not None]
    z_wall = min(wall_candidates) if wall_candidates else None

    if z_wall is not None:
        distance_to_wall = (z_wall - pos[2]) / d[2]
        if distance_to_wall > 0.0:
            flight_time += distance_to_wall / speed
            pos = pos + d * distance_to_wall
        if np.random.rand() < collimator_absorption:
            return None, None

    # Propagate to the detector plane (z = distance_to_detector).
    t_total = (distance_to_detector - pos[2]) / d[2]
    if t_total <= 0.0:
        return None, None
    hit = pos + d * t_total
    if abs(hit[0]) > half_size or abs(hit[1]) > half_size:
        return None, None

    flight_time += t_total / speed
    return flight_time, hit


################################################################################
# Energy deposition in the scintillator
################################################################################

def simulate_in_scintillator(
    energy_mev: float,
    scintillator_thickness: float,
    mean_free_path: float,
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
        energy = scatter_energy_elastic(energy, target_mass_ratio)
        speed = energy_to_speed(energy)
    return cumulative_time, energy


################################################################################
# High level simulation driver
################################################################################

def simulate_neutron_history(
    shell_thickness: float,
    aluminium_mfp: float,
    aluminium_mass_ratio: float,
    scintillator_thickness: float,
    scintillator_mfp: float,
    scintillator_mass_ratio: float,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    collimator_absorption: float = 0.99,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
) -> Optional[float]:
    """Simulate the complete history of a single neutron.

    This function ties together all the individual stages: generation at the
    target centre, transport through the aluminium shell, propagation through
    the collimator, and final slowing down in the scintillator.  If the
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
        Distance from the target to the scintillator (m).  Defaults to 16 m.
    detector_side : float, optional
        Side length of the square scintillator (m).  Defaults to 1 m.
    collimator_absorption : float, optional
        Fraction of neutrons absorbed in the collimating channel.  Defaults
        to 0.99 (99 % absorption).
    energy_cutoff_mev : float, optional
        Energy threshold below which neutrons are considered absorbed.
    shell_geometry : MeshGeometry, optional
        Preprocessed STL data.  When provided the simulation respects the shell
        openings described by the mesh.  If ``None``, a spherical approximation
        is used.

    Returns
    -------
    float or None
        The total time‑of‑flight in seconds if the neutron deposits energy in
        the scintillator, otherwise ``None``.
    """
    # 1. Generate initial energy and direction
    E0 = sample_neutron_energy()
    d0 = sample_isotropic_direction()

    # 2. Transport through the aluminium shell
    t_shell, E_after_shell, d_after_shell, pos_after_shell = simulate_in_aluminium(
        d0,
        E0,
        shell_thickness,
        aluminium_mfp,
        aluminium_mass_ratio,
        energy_cutoff_mev,
        mesh_geometry=shell_geometry,
    )
    # If energy is below cutoff the neutron was absorbed in the shell
    if E_after_shell <= energy_cutoff_mev:
        return None

    # 3. Flight to the scintillator with collimator losses
    flight_time, hit_point = propagate_to_scintillator(
        pos_after_shell,
        d_after_shell,
        E_after_shell,
        distance_to_detector=detector_distance,
        detector_side=detector_side,
        collimator_absorption=collimator_absorption,
        energy_cutoff_mev=energy_cutoff_mev,
    )
    if flight_time is None:
        return None

    # 4. Energy deposition in the scintillator
    t_scin, E_after_scin = simulate_in_scintillator(
        E_after_shell,
        scintillator_thickness,
        scintillator_mfp,
        scintillator_mass_ratio,
        energy_cutoff_mev,
    )
    # The neutron is considered to contribute to the signal regardless of
    # whether it is absorbed or escapes from the scintillator; the TOF is
    # measured at the moment of first interaction.
    return t_shell + flight_time + t_scin


def run_simulation(
    n_neutrons: int,
    shell_thickness: float,
    aluminium_mfp: float,
    aluminium_mass_ratio: float,
    scintillator_thickness: float,
    scintillator_mfp: float,
    scintillator_mass_ratio: float,
    detector_distance: float = 16.0,
    detector_side: float = 1.0,
    collimator_absorption: float = 0.99,
    energy_cutoff_mev: float = 0.1,
    shell_geometry: Optional[MeshGeometry] = None,
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
            aluminium_mfp,
            aluminium_mass_ratio,
            scintillator_thickness,
            scintillator_mfp,
            scintillator_mass_ratio,
            detector_distance,
            detector_side,
            collimator_absorption,
            energy_cutoff_mev,
            shell_geometry=shell_geometry,
        )
        if tof is not None:
            tof_values.append(tof)
    return tof_values


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
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
    unit_scale = 1.0e-3  # Convert millimetres to metres
    mesh_scaled = shell_mesh * unit_scale
    shell_geometry = prepare_mesh_geometry(mesh_scaled)
    mean_radius, max_radius = mesh_distance_statistics(mesh_scaled)

    shell_thickness = 0.08  # metres (given design thickness)

    print(f"[info] Using specified shell thickness of {shell_thickness:.3f} m")
    print(
        f"[info] STL vertex distances: mean={mean_radius:.4f} m, "
        f"max={max_radius:.4f} m"
    )

    # Simulation parameters (adjust as needed)
    n_neutrons = 1000
    aluminium_mfp = 0.002  # Mean free path in aluminium (m)
    scintillator_thickness = 0.05  # Scintillator thickness (m)
    scintillator_mfp = 0.01  # Mean free path in the scintillator (m)
    aluminium_mass_ratio = 26.98  # Mass ratio A for aluminium
    scintillator_mass_ratio = 1.0  # Dominant scattering nucleus (hydrogen)

    tof_list = run_simulation(
        n_neutrons=n_neutrons,
        shell_thickness=shell_thickness,
        aluminium_mfp=aluminium_mfp,
        aluminium_mass_ratio=aluminium_mass_ratio,
        scintillator_thickness=scintillator_thickness,
        scintillator_mfp=scintillator_mfp,
        scintillator_mass_ratio=scintillator_mass_ratio,
        detector_distance=16.0,
        detector_side=1.0,
        collimator_absorption=0.99,
        energy_cutoff_mev=0.1,
        shell_geometry=shell_geometry,
    )

    print(f"Simulated {len(tof_list)} neutrons reaching the scintillator out of {n_neutrons}")
    if tof_list:
        print(f"Mean time of flight: {np.mean(tof_list):.3e} s")
    for tof in tof_list:
        print(f"{tof:.6e}")
