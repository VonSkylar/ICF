"""
Neutron trajectory visualization module.

This module provides functions and classes for visualizing neutron trajectories
from ICF simulations, including 3D plots, 2D projections, and energy analysis.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d

from .. import config
from ..core.stl_utils import load_stl_mesh


# Type alias for trajectory data
TrajectoryDict = Dict[int, List[dict]]


def load_trajectory_data(csv_file: Union[str, Path]) -> TrajectoryDict:
    """Load neutron trajectory data from CSV file.
    
    Parameters
    ----------
    csv_file : str or Path
        Path to the neutron trajectory data CSV file.
        
    Returns
    -------
    dict
        Dictionary mapping neutron_id to list of trajectory points.
        Each point is a dict with keys: step_id, event_type, position, energy
        
    Example
    -------
    >>> trajectories = load_trajectory_data('Data/neutron_trajectories.csv')
    >>> print(f"Loaded {len(trajectories)} trajectories")
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


def plot_trajectories_3d(
    trajectories: TrajectoryDict,
    max_trajectories: int = 100,
    save_path: Optional[str] = None,
    show_geometry: bool = True,
    geometry_source: Literal['stl', 'simple', 'none'] = 'stl',
    simple_geometry_type: str = 'standard',
    stl_dir: Optional[Union[str, Path]] = None,
    figsize: tuple = (11, 16),
    dpi: int = 300,
    show: bool = False
) -> Optional[plt.Figure]:
    """Plot neutron trajectories in 3D with energy-based coloring.
    
    Parameters
    ----------
    trajectories : dict
        Dictionary of trajectory data from load_trajectory_data.
    max_trajectories : int
        Maximum number of trajectories to plot.
    save_path : str, optional
        Path to save the figure. If None and show=False, returns figure.
    show_geometry : bool
        Whether to show geometry (shell, channel, detector).
    geometry_source : {'stl', 'simple', 'none'}
        Source of geometry: 'stl' for real STL files, 'simple' for
        analytical geometry, 'none' to hide geometry.
    simple_geometry_type : str
        Detail level for simple geometry: 'minimal', 'standard', 'detailed'.
    stl_dir : str or Path, optional
        Directory containing STL files. If None, uses default from config.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    show : bool
        Whether to display the figure interactively.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if save_path is None and show is False.
        
    Example
    -------
    >>> trajectories = load_trajectory_data('Data/neutron_trajectories.csv')
    >>> plot_trajectories_3d(trajectories, max_trajectories=50, save_path='Figures/traj')
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Select trajectories to plot
    neutron_ids = sorted(trajectories.keys())[:max_trajectories]
    
    # Collect data for colormap and axis limits
    all_energies = []
    all_positions = []
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        all_energies.extend([point['energy'] for point in traj])
        all_positions.extend([point['position'] for point in traj])
    
    if not all_energies:
        print("[warning] No trajectory data to plot.")
        plt.close(fig)
        return None
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    norm = Normalize(vmin=energy_min, vmax=energy_max)
    cmap = plt.cm.plasma
    
    # Load and display geometry
    if show_geometry and geometry_source != 'none':
        _add_geometry_to_plot(ax, geometry_source, simple_geometry_type, stl_dir)
        _add_detector_plane(ax)
    
    # Plot trajectories
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        
        if len(traj) < 2:
            continue
        
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
            Line2D([0], [0], color='gray', linewidth=4, alpha=0.3, label='Aluminum Shell'),
            Line2D([0], [0], color='cyan', linewidth=4, alpha=0.3, label='Polyethylene Channel'),
            Line2D([0], [0], color='red', linewidth=2, label='Detector Plane')
        ])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 1.0),
              fontsize=9, ncol=1, frameon=True)
    
    # Set axis limits and aspect
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 3)
    ax.set_box_aspect([1, 1, 2])
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        output_path = Path(save_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_path}_3d.png', dpi=dpi, bbox_inches='tight')
        print(f"[info] Saved 3D trajectory plot to {save_path}_3d.png")
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def plot_trajectories_2d(
    trajectories: TrajectoryDict,
    max_trajectories: int = 100,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    dpi: int = 300,
    show: bool = False
) -> Optional[plt.Figure]:
    """Plot 2D projections of neutron trajectories.
    
    Creates a 2x2 grid showing XY, XZ, YZ projections and energy vs Z.
    
    Parameters
    ----------
    trajectories : dict
        Dictionary of trajectory data from load_trajectory_data.
    max_trajectories : int
        Maximum number of trajectories to plot.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    show : bool
        Whether to display the figure interactively.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if save_path is None and show is False.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    neutron_ids = sorted(trajectories.keys())[:max_trajectories]
    
    # Collect energy range
    all_energies = []
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        all_energies.extend([point['energy'] for point in traj])
    
    if not all_energies:
        print("[warning] No trajectory data to plot.")
        plt.close(fig)
        return None
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    norm = Normalize(vmin=energy_min, vmax=energy_max)
    cmap = plt.cm.plasma
    
    # Plot each projection
    projections = [
        (axes[0, 0], 0, 1, 'X Position (m)', 'Y Position (m)', 'XY Projection (Detector Plane)', True),
        (axes[0, 1], 0, 2, 'X Position (m)', 'Z Position (m)', 'XZ Projection (Side View)', False),
        (axes[1, 0], 1, 2, 'Y Position (m)', 'Z Position (m)', 'YZ Projection (Side View)', False),
    ]
    
    for ax, idx1, idx2, xlabel, ylabel, title, equal_aspect in projections:
        for neutron_id in neutron_ids:
            traj = trajectories[neutron_id]
            if len(traj) < 2:
                continue
            
            positions = np.array([point['position'] for point in traj])
            energies = np.array([point['energy'] for point in traj])
            
            for i in range(len(positions) - 1):
                mid_energy = (energies[i] + energies[i + 1]) / 2
                color = cmap(norm(mid_energy))
                ax.plot([positions[i][idx1], positions[i+1][idx1]],
                       [positions[i][idx2], positions[i+1][idx2]],
                       color=color, linewidth=1.0, alpha=0.5)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if equal_aspect:
            ax.set_aspect('equal')
    
    # Energy vs Z (bottom-right)
    ax_ez = axes[1, 1]
    for neutron_id in neutron_ids:
        traj = trajectories[neutron_id]
        if len(traj) < 2:
            continue
        
        positions = np.array([point['position'] for point in traj])
        energies = np.array([point['energy'] for point in traj])
        z_coords = positions[:, 2]
        
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
    
    if save_path:
        output_path = Path(save_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_path}_2d_projections.png', dpi=dpi, bbox_inches='tight')
        print(f"[info] Saved 2D projection plots to {save_path}_2d_projections.png")
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def _add_geometry_to_plot(
    ax,
    geometry_source: str,
    simple_geometry_type: str,
    stl_dir: Optional[Path]
) -> None:
    """Add geometry visualization to a 3D axis."""
    
    if geometry_source == 'simple':
        from ..testing import create_simple_icf_geometry
        shell_mesh, channel_mesh = create_simple_icf_geometry(simple_geometry_type)
        # Convert mm to m
        shell_mesh_m = shell_mesh.copy()
        channel_mesh_m = channel_mesh.copy()
        shell_mesh_m[:, 1:, :] *= config.MM_TO_M
        channel_mesh_m[:, 1:, :] *= config.MM_TO_M
        
        _plot_mesh(ax, shell_mesh_m, color=config.STL_SHELL_COLOR, alpha=config.STL_SHELL_ALPHA)
        _plot_mesh(ax, channel_mesh_m, color=config.STL_CHANNEL_COLOR, alpha=config.STL_CHANNEL_ALPHA)
        
    elif geometry_source == 'stl':
        if stl_dir is None:
            # 使用包内置的STL模型目录
            stl_dir = config.STL_MODEL_DIR
        else:
            stl_dir = Path(stl_dir)
        
        stl_geometries = [
            (config.SHELL_STL_FILE, "Aluminum shell", config.STL_SHELL_ALPHA, config.STL_SHELL_COLOR),
            (config.CHANNEL_STL_FILE, "Polyethylene channel", config.STL_CHANNEL_ALPHA, config.STL_CHANNEL_COLOR),
        ]
        
        for filename, label, alpha, color in stl_geometries:
            stl_path = stl_dir / filename
            
            if not stl_path.exists():
                print(f"[warning] {label} STL not found at {stl_path}, skipping.")
                continue
            
            try:
                print(f"[info] Loading {label} from {filename}...")
                mesh = load_stl_mesh(str(stl_path))
                mesh_scaled = mesh.copy()
                mesh_scaled[:, 1:, :] *= config.MM_TO_M
                _plot_mesh(ax, mesh_scaled, color=color, alpha=alpha)
                print(f"[info] {label} loaded: {len(mesh)} triangles")
            except Exception as e:
                print(f"[warning] Could not load {label}: {e}")


def _plot_mesh(ax, mesh: np.ndarray, color: str, alpha: float) -> None:
    """Plot a mesh on a 3D axis."""
    if mesh.shape[1] == 4:
        vertices = mesh[:, 1:, :]
    else:
        vertices = mesh
    
    triangles = [triangle for triangle in vertices]
    collection = Poly3DCollection(triangles, alpha=alpha, facecolor=color,
                                  edgecolor='none', linewidths=0)
    ax.add_collection3d(collection)


def _add_detector_plane(ax) -> None:
    """Add detector plane visualization to a 3D axis."""
    detector_z = config.DETECTOR_Z_MM * config.MM_TO_M
    detector_radius = config.DETECTOR_RADIUS_MM * config.MM_TO_M
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = detector_radius * np.cos(theta)
    y_circle = detector_radius * np.sin(theta)
    z_circle = np.full_like(x_circle, detector_z)
    
    ax.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2.5, alpha=1.0, zorder=10)
    
    circle = Circle((0, 0), detector_radius, color='red', alpha=0.25, zorder=5)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=detector_z, zdir="z")


class TrajectoryPlotter:
    """Class-based interface for trajectory plotting with persistent settings.
    
    This class provides an object-oriented approach to trajectory visualization,
    allowing you to configure settings once and reuse them.
    
    Example
    -------
    >>> plotter = TrajectoryPlotter(geometry_source='simple', max_trajectories=50)
    >>> plotter.load('Data/neutron_trajectories.csv')
    >>> plotter.plot_3d(save_path='Figures/trajectories')
    >>> plotter.plot_2d(save_path='Figures/trajectories')
    """
    
    def __init__(
        self,
        max_trajectories: int = 100,
        geometry_source: Literal['stl', 'simple', 'none'] = 'stl',
        simple_geometry_type: str = 'standard',
        stl_dir: Optional[Union[str, Path]] = None,
        figsize_3d: tuple = (11, 16),
        figsize_2d: tuple = (16, 10),
        dpi: int = 300
    ):
        """Initialize the trajectory plotter.
        
        Parameters
        ----------
        max_trajectories : int
            Default maximum number of trajectories to plot.
        geometry_source : {'stl', 'simple', 'none'}
            Default geometry source.
        simple_geometry_type : str
            Default simple geometry detail level.
        stl_dir : str or Path, optional
            Directory containing STL files.
        figsize_3d : tuple
            Default figure size for 3D plots.
        figsize_2d : tuple
            Default figure size for 2D plots.
        dpi : int
            Default DPI for saved figures.
        """
        self.max_trajectories = max_trajectories
        self.geometry_source = geometry_source
        self.simple_geometry_type = simple_geometry_type
        self.stl_dir = stl_dir
        self.figsize_3d = figsize_3d
        self.figsize_2d = figsize_2d
        self.dpi = dpi
        self.trajectories: Optional[TrajectoryDict] = None
    
    def load(self, csv_file: Union[str, Path]) -> 'TrajectoryPlotter':
        """Load trajectory data from CSV file.
        
        Parameters
        ----------
        csv_file : str or Path
            Path to trajectory CSV file.
            
        Returns
        -------
        self : TrajectoryPlotter
            Returns self for method chaining.
        """
        self.trajectories = load_trajectory_data(csv_file)
        print(f"[info] Loaded {len(self.trajectories)} trajectories")
        return self
    
    def plot_3d(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> Optional[plt.Figure]:
        """Plot 3D trajectories.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        show : bool
            Whether to display interactively.
        **kwargs
            Override default settings.
            
        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        if self.trajectories is None:
            raise ValueError("No trajectory data loaded. Call load() first.")
        
        return plot_trajectories_3d(
            self.trajectories,
            max_trajectories=kwargs.get('max_trajectories', self.max_trajectories),
            save_path=save_path,
            show_geometry=kwargs.get('show_geometry', True),
            geometry_source=kwargs.get('geometry_source', self.geometry_source),
            simple_geometry_type=kwargs.get('simple_geometry_type', self.simple_geometry_type),
            stl_dir=kwargs.get('stl_dir', self.stl_dir),
            figsize=kwargs.get('figsize', self.figsize_3d),
            dpi=kwargs.get('dpi', self.dpi),
            show=show
        )
    
    def plot_2d(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> Optional[plt.Figure]:
        """Plot 2D projections.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        show : bool
            Whether to display interactively.
        **kwargs
            Override default settings.
            
        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        if self.trajectories is None:
            raise ValueError("No trajectory data loaded. Call load() first.")
        
        return plot_trajectories_2d(
            self.trajectories,
            max_trajectories=kwargs.get('max_trajectories', self.max_trajectories),
            save_path=save_path,
            figsize=kwargs.get('figsize', self.figsize_2d),
            dpi=kwargs.get('dpi', self.dpi),
            show=show
        )
    
    def plot_all(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> None:
        """Create both 3D and 2D plots.
        
        Parameters
        ----------
        save_path : str, optional
            Base path for saving figures.
        show : bool
            Whether to display interactively.
        **kwargs
            Override default settings.
        """
        self.plot_3d(save_path=save_path, show=show, **kwargs)
        self.plot_2d(save_path=save_path, show=show, **kwargs)
