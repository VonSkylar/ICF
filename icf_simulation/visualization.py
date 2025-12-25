"""
Visualization utilities for neutron simulation results.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from .data_classes import NeutronRecord, DetectorPlane


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
    ax2.hist(energy_loss, bins=50, color='green', alpha=0.7)
    ax2.set_xlabel('Energy Loss (MeV)')
    ax2.set_ylabel('Count')
    ax2.set_title('Energy Loss Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. TOF histogram
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(tofs * 1e9, bins=50, color='purple', alpha=0.7)
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
    ax5.axvline(np.mean(energy_loss), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(energy_loss):.2f} MeV')
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
