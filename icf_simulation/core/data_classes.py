"""
Data classes for the ICF neutron simulation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
    """Record of a neutron's journey through the simulation.
    
    Attributes
    ----------
    initial_energy : float
        Initial neutron energy at source (MeV).
    final_energy : float
        Final neutron energy at end of simulation (MeV).
    tof : float
        Total time of flight (seconds).
    exit_position : np.ndarray
        Position where neutron FIRST exited the aluminum shell (m).
        This is useful for analyzing shell transmission.
    detector_hit_position : np.ndarray or None
        Position where neutron hit the detector (m), or None if missed.
    reached_detector : bool
        Whether the neutron reached the detector.
    energy_after_shell : float
        Energy after first exiting aluminum shell (MeV).
    energy_after_channel : float
        Energy after exiting polyethylene channel (MeV).
    status : str
        Final status: 'success', 'lost_in_shell', 'lost_in_channel', 'missed_detector'.
    final_position : np.ndarray or None
        Final position at end of simulation (m). Different from exit_position
        as neutron may travel further after exiting shell.
    final_direction : np.ndarray or None
        Final direction unit vector.
    trajectory_points : list or None
        List of (position, energy, event_type) tuples for trajectory visualization.
    """
    
    initial_energy: float  # MeV
    final_energy: float  # MeV
    tof: float  # seconds
    exit_position: np.ndarray  # First exit from Al shell (m)
    detector_hit_position: Optional[np.ndarray]  # 3D position at detector (m)
    reached_detector: bool
    energy_after_shell: float = 0.0  # MeV, energy after aluminum shell
    energy_after_channel: float = 0.0  # MeV, energy after polyethylene channel
    status: str = "unknown"  # Status: success, lost_in_shell, lost_in_channel, missed_detector
    final_position: Optional[np.ndarray] = None  # Final position at simulation end (m)
    final_direction: Optional[np.ndarray] = None  # Final direction
    trajectory_points: Optional[List[Tuple[np.ndarray, float, str]]] = None  # List of (position, energy, event_type)
