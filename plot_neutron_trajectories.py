"""
Neutron Trajectory Visualization
=================================

This script visualizes complete neutron trajectories from the ICF simulation,
showing the path from the origin through all collisions to the final position,
with color representing energy variation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import csv
from pathlib import Path
from collections import defaultdict

# Import from icf_simulation package
from icf_simulation import load_stl_mesh


def load_neutron_trajectory_data(csv_file):
    """Load neutron trajectory data from CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to the neutron trajectory data CSV file.
        
    Returns
    -------
    dict
        Dictionary mapping neutron_id to list of trajectory points.
        Each point is a dict with keys: step_id, event_type, position, energy
    """
    trajectories = defaultdict(list)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neutron_id = int(row['neutron_id'])
            step_id = int(row['step_id'])
            event_type = row['event_type']
            position = np.array([
                float(row['position_x_m']),
                float(row['position_y_m']),
                float(row['position_z_m'])
            ])
            energy = float(row['energy_MeV'])
            
            trajectories[neutron_id].append({
                'step_id': step_id,
                'event_type': event_type,
                'position': position,
                'energy': energy
            })
    
    # Sort each trajectory by step_id
    for neutron_id in trajectories:
        trajectories[neutron_id].sort(key=lambda x: x['step_id'])
    
    return dict(trajectories)


def plot_neutron_trajectories_3d(trajectories, max_trajectories=100, save_path=None, 
                                  show_geometry=True, stl_dir=None):
    """Plot neutron trajectories in 3D with energy-based coloring.
    
    Parameters
    ----------
    trajectories : dict
        Dictionary of trajectory data from load_neutron_trajectory_data.
    max_trajectories : int
        Maximum number of trajectories to plot.
    save_path : str, optional
        Path to save the figure. If None, figure is displayed.
    show_geometry : bool
        Whether to show aluminum shell, channel, and detector geometry.
    stl_dir : str or Path, optional
        Directory containing STL files. If None, uses script directory.
    """
    fig = plt.figure(figsize=(11,16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select trajectories to plot first to get data range
    neutron_ids = sorted(trajectories.keys())[:max_trajectories]
    
    # Find energy range for colormap
    all_energies = []
    all_positions = []
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        all_energies.extend([point['energy'] for point in traj])
        all_positions.extend([point['position'] for point in traj])
    
    if not all_energies:
        print("[warning] No trajectory data to plot.")
        return
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    norm = Normalize(vmin=energy_min, vmax=energy_max)
    cmap = plt.cm.plasma  # Yellow (high energy) to purple (low energy)
    
    # Get position ranges for axis limits
    all_positions = np.array(all_positions)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    
    # Expand limits slightly based on trajectory range
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    margin_x = max(0.1, x_range * 0.15)  # 15% margin or at least 0.1m
    margin_y = max(0.1, y_range * 0.15)
    margin_z = max(0.2, z_range * 0.1)   # 10% margin or at least 0.2m
    
    x_min, x_max = x_min - margin_x, x_max + margin_x
    y_min, y_max = y_min - margin_y, y_max + margin_y
    z_min, z_max = max(-0.2, z_min - margin_z), z_max + margin_z
    
    # Load and display geometry if requested
    if show_geometry:
        if stl_dir is None:
            stl_dir = Path(__file__).resolve().parent
        else:
            stl_dir = Path(stl_dir)
        
        unit_scale = 1.0e-3  # Convert mm to m
        
        # Load and plot aluminum shell
        shell_file = None
        for candidate in ["Target ball model.stl", "Target_ball_model.stl"]:
            path = stl_dir / candidate
            if path.exists():
                shell_file = path
                break
        
        if shell_file and shell_file.exists():
            try:
                print(f"[info] Loading aluminum shell from {shell_file.name}...")
                shell_mesh = load_stl_mesh(str(shell_file))
                # Handle new format: (n, 4, 3) with normals -> extract vertices only
                if shell_mesh.shape[1] == 4:
                    shell_mesh = shell_mesh[:, 1:, :]  # Shape becomes (n, 3, 3)
                shell_mesh_scaled = shell_mesh * unit_scale
                
                num_triangles = len(shell_mesh_scaled)
                # Use all triangles for complete geometry visualization
                shell_triangles = [triangle for triangle in shell_mesh_scaled]
                
                # Plot shell with very low transparency to avoid blocking trajectories
                shell_collection = Poly3DCollection(shell_triangles, alpha=0.12, 
                                                   facecolor='lightgray', edgecolor='none', 
                                                   linewidths=0)
                ax.add_collection3d(shell_collection)
                print(f"[info] Aluminum shell loaded: {num_triangles} triangles displayed")
            except Exception as e:
                print(f"[warning] Could not load aluminum shell: {e}")
        
        # Load and plot polyethylene channel
        channel_file = stl_dir / "nTOF_without_scintillant.STL"
        if channel_file.exists():
            try:
                print(f"[info] Loading polyethylene channel from {channel_file.name}...")
                channel_mesh = load_stl_mesh(str(channel_file))
                # Handle new format: (n, 4, 3) with normals -> extract vertices only
                if channel_mesh.shape[1] == 4:
                    channel_mesh = channel_mesh[:, 1:, :]  # Shape becomes (n, 3, 3)
                channel_mesh_scaled = channel_mesh * unit_scale
                
                num_triangles = len(channel_mesh_scaled)
                # Use all triangles for complete geometry visualization
                channel_triangles = [triangle for triangle in channel_mesh_scaled]
                
                # Plot channel with low transparency
                channel_collection = Poly3DCollection(channel_triangles, alpha=0.2, 
                                                     facecolor='lightblue', edgecolor='none', 
                                                     linewidths=0)
                ax.add_collection3d(channel_collection)
                print(f"[info] Polyethylene channel loaded: {num_triangles} triangles displayed")
            except Exception as e:
                print(f"[warning] Could not load polyethylene channel: {e}")
        
        # Draw detector plane (circular, at z=2.9m, radius=105mm)
        detector_z = 2.9  # meters
        detector_radius = 0.105  # meters
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = detector_radius * np.cos(theta)
        y_circle = detector_radius * np.sin(theta)
        z_circle = np.full_like(x_circle, detector_z)
        
        # Draw detector circle outline
        ax.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2.5, alpha=1.0, zorder=10)
        
        # Draw detector disk (filled circle)
        from matplotlib.patches import Circle
        from mpl_toolkits.mplot3d import art3d
        circle = Circle((0, 0), detector_radius, color='red', alpha=0.25, zorder=5)
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=detector_z, zdir="z")
    
    # Plot each trajectory
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        
        if len(traj) < 2:
            continue
        
        # Extract positions and energies
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        
        # Plot trajectory segments with color gradient
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            mid_energy = (energies[i] + energies[i + 1]) / 2
            
            color = cmap(norm(mid_energy))
            
            ax.plot([start_pos[0], end_pos[0]],
                   [start_pos[1], end_pos[1]],
                   [start_pos[2], end_pos[2]],
                   color=color, linewidth=1.0, alpha=0.6)
        
        # Mark special points
        for point in traj:
            if point['event_type'] == 'source':
                ax.scatter(*point['position'], c='green', marker='o', s=30, alpha=0.8)
            elif point['event_type'] == 'detector_hit':
                ax.scatter(*point['position'], c='red', marker='*', s=50, alpha=0.8)
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0, shrink=0.5, aspect=30)
    cbar.set_label('Neutron Energy (MeV)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title(f'Neutron Trajectories (n={len(neutron_ids)})', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, label='Source (Origin)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=10, label='Detector Hit'),
        Line2D([0], [0], color=cmap(norm(energy_max)), linewidth=2, 
               label=f'High Energy (~{energy_max:.2f} MeV)'),
        Line2D([0], [0], color=cmap(norm(energy_min)), linewidth=2, 
               label=f'Low Energy (~{energy_min:.2f} MeV)')
    ]
    
    if show_geometry:
        legend_elements.extend([
            Line2D([0], [0], color='gray', linewidth=4, alpha=0.3, 
                   label='Aluminum Shell'),
            Line2D([0], [0], color='cyan', linewidth=4, alpha=0.3, 
                   label='Polyethylene Channel'),
            Line2D([0], [0], color='red', linewidth=2, 
                   label='Detector Plane')
        ])
    
    ax.legend(handles=legend_elements, 
          loc='upper left', 
          bbox_to_anchor=(-0.1, 1.0), # 第一个参数负值表示向左移出
          fontsize=9, 
          ncol=1,       # 如果图例太长，可以改为 ncol=2
          frameon=True) # 加上边框可以增加辨识度
    
    # Set explicit axis limits to ensure geometry is visible
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 3)
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 2])  # Emphasize Z dimension
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        output_path = Path(save_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_path}_3d.png', dpi=300, bbox_inches='tight')
        print(f"[info] Saved 3D trajectory plot to {save_path}_3d.png")
    else:
        plt.show()
    
    plt.close()


def plot_trajectories_2d_projections(trajectories, max_trajectories=100, save_path=None):
    """Plot 2D projections of neutron trajectories.
    
    Parameters
    ----------
    trajectories : dict
        Dictionary of trajectory data from load_neutron_trajectory_data.
    max_trajectories : int
        Maximum number of trajectories to plot.
    save_path : str, optional
        Path to save the figure. If None, figure is displayed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Select trajectories to plot
    neutron_ids = sorted(trajectories.keys())[:max_trajectories]
    
    # Find energy range for colormap
    all_energies = []
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        all_energies.extend([point['energy'] for point in traj])
    
    if not all_energies:
        print("[warning] No trajectory data to plot.")
        return
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    norm = Normalize(vmin=energy_min, vmax=energy_max)
    cmap = plt.cm.plasma
    
    # XY projection (top-left)
    ax_xy = axes[0, 0]
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        if len(traj) < 2:
            continue
        
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        
        for i in range(len(positions) - 1):
            mid_energy = (energies[i] + energies[i + 1]) / 2
            color = cmap(norm(mid_energy))
            ax_xy.plot([positions[i][0], positions[i+1][0]],
                      [positions[i][1], positions[i+1][1]],
                      color=color, linewidth=1.0, alpha=0.5)
    
    ax_xy.set_xlabel('X Position (m)', fontsize=11)
    ax_xy.set_ylabel('Y Position (m)', fontsize=11)
    ax_xy.set_title('XY Projection (Detector Plane)', fontsize=12, fontweight='bold')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal')
    
    # XZ projection (top-right)
    ax_xz = axes[0, 1]
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        if len(traj) < 2:
            continue
        
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        
        for i in range(len(positions) - 1):
            mid_energy = (energies[i] + energies[i + 1]) / 2
            color = cmap(norm(mid_energy))
            ax_xz.plot([positions[i][0], positions[i+1][0]],
                      [positions[i][2], positions[i+1][2]],
                      color=color, linewidth=1.0, alpha=0.5)
    
    ax_xz.set_xlabel('X Position (m)', fontsize=11)
    ax_xz.set_ylabel('Z Position (m)', fontsize=11)
    ax_xz.set_title('XZ Projection (Side View)', fontsize=12, fontweight='bold')
    ax_xz.grid(True, alpha=0.3)
    
    # YZ projection (bottom-left)
    ax_yz = axes[1, 0]
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        if len(traj) < 2:
            continue
        
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        
        for i in range(len(positions) - 1):
            mid_energy = (energies[i] + energies[i + 1]) / 2
            color = cmap(norm(mid_energy))
            ax_yz.plot([positions[i][1], positions[i+1][1]],
                      [positions[i][2], positions[i+1][2]],
                      color=color, linewidth=1.0, alpha=0.5)
    
    ax_yz.set_xlabel('Y Position (m)', fontsize=11)
    ax_yz.set_ylabel('Z Position (m)', fontsize=11)
    ax_yz.set_title('YZ Projection (Side View)', fontsize=12, fontweight='bold')
    ax_yz.grid(True, alpha=0.3)
    
    # Energy vs Z (bottom-right)
    ax_ez = axes[1, 1]
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        if len(traj) < 2:
            continue
        
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        z_coords = positions[:, 2]
        
        # Determine if neutron hit detector
        hit_detector = any(p['event_type'] == 'detector_hit' for p in traj)
        color = 'red' if hit_detector else 'gray'
        alpha = 0.6 if hit_detector else 0.2
        
        ax_ez.plot(z_coords, energies, color=color, linewidth=1.0, alpha=alpha)
    
    ax_ez.set_xlabel('Z Position (m)', fontsize=11)
    ax_ez.set_ylabel('Energy (MeV)', fontsize=11)
    ax_ez.set_title('Energy vs Z Position', fontsize=12, fontweight='bold')
    ax_ez.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, shrink=0.8, aspect=40)
    cbar.set_label('Neutron Energy (MeV)', fontsize=12)
    
    plt.suptitle(f'Neutron Trajectory Projections (n={len(neutron_ids)})', 
                 fontsize=14, fontweight='bold', y=0.995)
    # plt.tight_layout()
    
    if save_path:
        output_path = Path(save_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_path}_2d_projections.png', dpi=300, bbox_inches='tight')
        print(f"[info] Saved 2D projection plots to {save_path}_2d_projections.png")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Default paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "Data"
    figures_dir = script_dir / "Figures"
    
    # Load trajectory data
    trajectory_file = data_dir / "neutron_trajectories.csv"
    
    if not trajectory_file.exists():
        print(f"[error] Trajectory file not found: {trajectory_file}")
        print("[info] Please run icf_neutron_simulation.py first to generate trajectory data.")
    else:
        print(f"[info] Loading neutron trajectory data from {trajectory_file}")
        trajectories = load_neutron_trajectory_data(str(trajectory_file))
        
        print(f"[info] Loaded {len(trajectories)} neutron trajectories")
        
        # Create visualizations
        save_base = str(figures_dir / "neutron_trajectories")
        
        print("[info] Creating 3D trajectory plot...")
        plot_neutron_trajectories_3d(trajectories, max_trajectories=100, save_path=save_base)
        
        print("[info] Creating 2D projection plots...")
        plot_trajectories_2d_projections(trajectories, max_trajectories=100, save_path=save_base)
        
        print("[info] Visualization complete!")
