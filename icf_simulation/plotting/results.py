"""
Simulation results visualization.

This module provides visualization functions for neutron simulation results,
including energy distributions, detector hits, and statistical summaries.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from ..core.data_classes import NeutronRecord, DetectorPlane


def visualize_neutron_data(records: List[NeutronRecord], save_path: Optional[str] = None):
    """Create comprehensive visualizations of neutron simulation results.
    
    Parameters
    ----------
    records : List[NeutronRecord]
        List of ALL neutron records from simulation.
    save_path : str, optional
        Base path for saving figures.
    """
    if not records:
        print("[warning] No neutron records to visualize.")
        return
    
    initial_energies = np.array([r.initial_energy for r in records])
    final_energies = np.array([r.final_energy for r in records])
    tofs = np.array([r.tof for r in records])
    energy_loss = initial_energies - final_energies
    retention = final_energies / initial_energies
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Energy histogram (top-left)
    ax1 = axes[0, 0]
    ax1.hist(initial_energies, bins=50, alpha=0.7, label='Initial Energy', color='blue')
    ax1.hist(final_energies, bins=50, alpha=0.7, label='Final Energy', color='red')
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Count')
    ax1.set_title('Energy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy retention distribution (top-right)
    ax2 = axes[0, 1]
    ax2.hist(retention, bins=50, color='teal', alpha=0.7, )
    ax2.axvline(np.mean(retention), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(retention):.3f}')
    ax2.axvline(np.median(retention), color='blue', linestyle=':', linewidth=2,
                label=f'Median: {np.median(retention):.3f}')
    ax2.set_xlabel('Energy Retention (Final/Initial)')
    ax2.set_ylabel('Count')
    ax2.set_title('Energy Retention Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. TOF vs Energy (bottom-left)
    ax3 = axes[1, 0]
    ax3.scatter(tofs * 1e9, initial_energies, alpha=0.5, s=30, label='Initial Energy', color='blue')
    ax3.scatter(tofs * 1e9, final_energies, alpha=0.5, s=30, label='Final Energy', color='red')
    ax3.set_xlabel('Time of Flight (ns)')
    ax3.set_ylabel('Energy (MeV)')
    ax3.set_title('TOF vs Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. TOF histogram (bottom-right)
    ax4 = axes[1, 1]
    ax4.hist(tofs * 1e9, bins=50, color='purple', alpha=0.7)
    ax4.axvline(np.mean(tofs)*1e9, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(tofs)*1e9:.2f} ns')
    ax4.axvline(np.median(tofs)*1e9, color='blue', linestyle=':', linewidth=2,
                label=f'Median: {np.median(tofs)*1e9:.2f} ns')
    ax4.set_xlabel('Time of Flight (ns)')
    ax4.set_ylabel('Count')
    ax4.set_title('TOF Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
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
    hit_records = [r for r in records if r.detector_hit_position is not None]
    
    if not hit_records:
        print("[warning] No detector hits to visualize.")
        return
    
    hit_positions = np.array([r.detector_hit_position for r in hit_records])
    energies = np.array([r.final_energy for r in hit_records])
    tofs = np.array([r.tof for r in hit_records])
    
    u_coords = np.array([np.dot(pos, detector_plane.u) for pos in hit_positions])
    v_coords = np.array([np.dot(pos, detector_plane.v) for pos in hit_positions])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hit map colored by energy
    ax1 = axes[0]
    scatter1 = ax1.scatter(u_coords, v_coords, c=energies, cmap='viridis', s=50, alpha=0.6)
    
    if detector_plane.is_circular:
        circle = Circle((0, 0), detector_plane.radius, fill=False, edgecolor='red', 
                        linewidth=2, label='Detector boundary')
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
    
    if detector_plane.is_circular:
        circle = Circle((0, 0), detector_plane.radius, fill=False, edgecolor='red', 
                        linewidth=2, label='Detector boundary')
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
    
    successful_records = [r for r in records if r.reached_detector]
    
    print("\n" + "="*60)
    print("ALL NEUTRON STATISTICS")
    print("="*60)
    print(f"Total neutrons simulated: {n_total}")
    print(f"Neutrons reaching detector: {len(successful_records)} ({100*len(successful_records)/n_total:.2f}%)")
    print(f"Neutrons lost in shell: {sum(1 for r in records if r.status == 'lost_in_shell')} "
          f"({100*sum(1 for r in records if r.status == 'lost_in_shell')/n_total:.2f}%)")
    print(f"Neutrons lost in channel: {sum(1 for r in records if r.status == 'lost_in_channel')} "
          f"({100*sum(1 for r in records if r.status == 'lost_in_channel')/n_total:.2f}%)")
    print(f"Neutrons missed detector: {sum(1 for r in records if r.status == 'missed_detector')} "
          f"({100*sum(1 for r in records if r.status == 'missed_detector')/n_total:.2f}%)")
    print()
    
    initial_energies = np.array([r.initial_energy for r in records])
    final_energies = np.array([r.final_energy for r in records])
    energy_after_shell = np.array([r.energy_after_shell for r in records])
    energy_after_channel = np.array([r.energy_after_channel for r in records])
    tofs = np.array([r.tof for r in records])
    
    loss_in_shell = initial_energies - energy_after_shell
    loss_in_channel = energy_after_shell - energy_after_channel
    
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
