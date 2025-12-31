"""
Comparison utilities for testing simple vs real geometry.

This module provides functions to compare simulation results between
simple analytical geometry and real STL geometry to help diagnose issues.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np


def load_statistics(csv_file: Union[str, Path], file_type: str = 'auto') -> Optional[dict]:
    """Load statistics from simulation output files.
    
    Parameters
    ----------
    csv_file : str or Path
        Path to CSV file (neutron data or trajectory data).
    file_type : str
        Type of file: 'neutron', 'trajectory', or 'auto' (detect from content).
        
    Returns
    -------
    stats : dict or None
        Dictionary of statistics, or None if file doesn't exist.
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        return None
    
    # Auto-detect file type from filename
    if file_type == 'auto':
        if 'trajectory' in csv_file.name.lower():
            file_type = 'trajectory'
        else:
            file_type = 'neutron'
    
    if file_type == 'trajectory':
        return _load_trajectory_statistics(csv_file)
    else:
        return _load_neutron_statistics(csv_file)


def _load_trajectory_statistics(csv_file: Path) -> dict:
    """Load statistics from trajectory data."""
    trajectories = defaultdict(list)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neutron_id = int(row['neutron_id'])
            position = np.array([
                float(row['position_x_m']),
                float(row['position_y_m']),
                float(row['position_z_m'])
            ])
            energy = float(row['energy_MeV'])
            event_type = row['event_type']
            
            trajectories[neutron_id].append({
                'position': position,
                'energy': energy,
                'event_type': event_type
            })
    
    # Compute statistics
    stats = {
        'n_trajectories': len(trajectories),
        'avg_steps': np.mean([len(traj) for traj in trajectories.values()]),
        'max_steps': max([len(traj) for traj in trajectories.values()]),
        'min_steps': min([len(traj) for traj in trajectories.values()]),
    }
    
    # Count event types
    event_counts = defaultdict(int)
    for traj in trajectories.values():
        for step in traj:
            event_counts[step['event_type']] += 1
    stats['events'] = dict(event_counts)
    
    # Compute path lengths
    path_lengths = []
    for traj in trajectories.values():
        if len(traj) > 1:
            positions = np.array([step['position'] for step in traj])
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            path_lengths.append(np.sum(distances))
    
    if path_lengths:
        stats['avg_path_length_m'] = np.mean(path_lengths)
        stats['max_path_length_m'] = np.max(path_lengths)
        stats['min_path_length_m'] = np.min(path_lengths)
    
    # Final positions
    final_z_positions = []
    for traj in trajectories.values():
        if traj:
            final_z_positions.append(traj[-1]['position'][2])
    
    if final_z_positions:
        stats['avg_final_z_m'] = np.mean(final_z_positions)
        stats['detector_arrivals'] = sum(1 for z in final_z_positions if z >= 2.8)
    
    return stats


def _load_neutron_statistics(csv_file: Path) -> dict:
    """Load statistics from neutron data."""
    data = {
        'detected': 0,
        'leaked': 0,
        'cutoff': 0,
        'total': 0,
    }
    
    initial_energies = []
    final_energies = []
    tofs = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['total'] += 1
            
            fate = row['fate']
            if fate == 'detected':
                data['detected'] += 1
            elif fate == 'leaked':
                data['leaked'] += 1
            elif fate == 'energy_cutoff':
                data['cutoff'] += 1
            
            initial_energies.append(float(row['initial_energy_MeV']))
            final_energies.append(float(row['final_energy_MeV']))
            tofs.append(float(row['tof_ns']))
    
    if initial_energies:
        data['avg_initial_energy'] = np.mean(initial_energies)
        data['avg_final_energy'] = np.mean(final_energies)
        data['avg_tof_ns'] = np.mean(tofs)
    
    return data


def print_comparison(
    stats1: dict,
    stats2: dict,
    label1: str = "Simple",
    label2: str = "STL"
) -> None:
    """Print side-by-side comparison of statistics.
    
    Parameters
    ----------
    stats1 : dict
        Statistics from first simulation.
    stats2 : dict
        Statistics from second simulation.
    label1 : str
        Label for first set.
    label2 : str
        Label for second set.
    """
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Statistic':<35} {label1:>15} {label2:>15}")
    print("-" * 70)
    
    # Common keys
    for key in ['total', 'detected', 'leaked', 'cutoff']:
        if key in stats1 and key in stats2:
            v1, v2 = stats1[key], stats2[key]
            print(f"{key.capitalize():<35} {v1:>15} {v2:>15}")
    
    # Float values
    for key in ['avg_initial_energy', 'avg_final_energy', 'avg_tof_ns',
                'avg_steps', 'avg_path_length_m', 'avg_final_z_m']:
        if key in stats1 and key in stats2:
            v1, v2 = stats1[key], stats2[key]
            print(f"{key:<35} {v1:>15.3f} {v2:>15.3f}")
    
    # Detection efficiency
    if 'total' in stats1 and 'detected' in stats1:
        eff1 = (stats1['detected'] / stats1['total'] * 100) if stats1['total'] > 0 else 0
        eff2 = (stats2['detected'] / stats2['total'] * 100) if stats2['total'] > 0 else 0
        print(f"{'Detection efficiency (%)':<35} {eff1:>15.1f} {eff2:>15.1f}")


def compare_geometries(
    simple_neutron_csv: Optional[Union[str, Path]] = None,
    simple_trajectory_csv: Optional[Union[str, Path]] = None,
    stl_neutron_csv: Optional[Union[str, Path]] = None,
    stl_trajectory_csv: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    simple_suffix: str = '_standard',
    verbose: bool = True
) -> Tuple[dict, dict]:
    """Compare simulation results between simple and STL geometry.
    
    Parameters
    ----------
    simple_neutron_csv : str or Path, optional
        Path to simple geometry neutron data.
    simple_trajectory_csv : str or Path, optional
        Path to simple geometry trajectory data.
    stl_neutron_csv : str or Path, optional
        Path to STL geometry neutron data.
    stl_trajectory_csv : str or Path, optional
        Path to STL geometry trajectory data.
    data_dir : str or Path, optional
        Base data directory. If provided, uses default filenames.
    simple_suffix : str
        Suffix for simple geometry files.
    verbose : bool
        Whether to print comparison.
        
    Returns
    -------
    simple_stats : dict
        Combined statistics for simple geometry.
    stl_stats : dict
        Combined statistics for STL geometry.
        
    Example
    -------
    >>> from icf_simulation.testing import compare_geometries
    >>> simple_stats, stl_stats = compare_geometries(data_dir='Data/')
    """
    # Determine file paths
    if data_dir is not None:
        data_dir = Path(data_dir)
        simple_neutron_csv = data_dir / f"neutron_data{simple_suffix}.csv"
        simple_trajectory_csv = data_dir / f"neutron_trajectories{simple_suffix}.csv"
        stl_neutron_csv = data_dir / "neutron_data.csv"
        stl_trajectory_csv = data_dir / "neutron_trajectories.csv"
    
    # Load statistics
    simple_stats = {}
    stl_stats = {}
    
    if simple_neutron_csv:
        simple_stats['neutron'] = load_statistics(simple_neutron_csv, 'neutron')
    if simple_trajectory_csv:
        simple_stats['trajectory'] = load_statistics(simple_trajectory_csv, 'trajectory')
    if stl_neutron_csv:
        stl_stats['neutron'] = load_statistics(stl_neutron_csv, 'neutron')
    if stl_trajectory_csv:
        stl_stats['trajectory'] = load_statistics(stl_trajectory_csv, 'trajectory')
    
    # Print comparison
    if verbose:
        if simple_stats.get('neutron') and stl_stats.get('neutron'):
            print("\n--- Neutron Data Comparison ---")
            print_comparison(simple_stats['neutron'], stl_stats['neutron'],
                           f"Simple{simple_suffix}", "Real-STL")
        
        if simple_stats.get('trajectory') and stl_stats.get('trajectory'):
            print("\n--- Trajectory Data Comparison ---")
            print_comparison(simple_stats['trajectory'], stl_stats['trajectory'],
                           f"Simple{simple_suffix}", "Real-STL")
        
        # Diagnostic analysis
        _print_diagnostic_analysis(simple_stats, stl_stats)
    
    return simple_stats, stl_stats


def _print_diagnostic_analysis(simple_stats: dict, stl_stats: dict) -> None:
    """Print diagnostic analysis and suggestions."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    s_n = simple_stats.get('neutron', {})
    r_n = stl_stats.get('neutron', {})
    
    if not s_n or not r_n:
        print("[info] Insufficient data for detailed analysis")
        return
    
    # Check detection efficiency difference
    det_eff_simple = (s_n.get('detected', 0) / s_n.get('total', 1) * 100)
    det_eff_stl = (r_n.get('detected', 0) / r_n.get('total', 1) * 100)
    det_diff = abs(det_eff_stl - det_eff_simple)
    
    if det_diff > 10:
        print(f"\n⚠ LARGE DETECTION EFFICIENCY DIFFERENCE: {det_diff:.1f}%")
        print("Possible causes:")
        print("  1. STL geometry has unexpected barriers or openings")
        print("  2. STL mesh normals may be inverted")
        print("  3. Channel geometry differs significantly from simple model")
        print("  4. Ray-mesh intersection issues with complex STL surfaces")
    elif det_diff > 5:
        print(f"\n⚠ MODERATE DETECTION EFFICIENCY DIFFERENCE: {det_diff:.1f}%")
        print("This is within expected range due to geometry differences.")
    else:
        print(f"\n✓ Detection efficiency difference is small: {det_diff:.1f}%")
        print("Geometry appears to be functioning correctly.")
    
    # Check leakage
    leak_simple = s_n.get('leaked', 0)
    leak_stl = r_n.get('leaked', 0)
    leak_diff = abs(leak_stl - leak_simple)
    
    if s_n.get('total', 0) > 0 and leak_diff > s_n['total'] * 0.15:
        print(f"\n⚠ LARGE LEAKAGE DIFFERENCE: {leak_diff} neutrons")
        print("  The STL geometry may have gaps or different boundaries.")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("\n1. Visualize trajectories with both geometries")
    print("2. Check STL file quality (manifold, normals, watertight)")
    print("3. Verify coordinate systems and units match")
