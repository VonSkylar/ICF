"""
STL geometry visualization module.

This module provides functions and classes for visualizing STL geometry files
used in the ICF simulation, including mesh inspection and comparison tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .. import config
from ..core.stl_utils import load_stl_mesh


def plot_stl_geometry(
    stl_dir: Optional[Union[str, Path]] = None,
    shell_only: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """Plot STL geometry files for inspection.
    
    Parameters
    ----------
    stl_dir : str or Path, optional
        Directory containing STL files. If None, uses default from config.
    shell_only : bool
        If True, only plot the shell (skip channel).
    save_path : str, optional
        Path to save the figure. If None, uses default from config.
    show : bool
        Whether to display the plot interactively.
    dpi : int
        Resolution for saved figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if show=False and save_path is None.
        
    Example
    -------
    >>> from icf_simulation.plotting import plot_stl_geometry
    >>> plot_stl_geometry()  # Uses default paths
    >>> plot_stl_geometry(stl_dir='custom/stl/path', shell_only=True)
    """
    # Determine STL directory (默认使用包内置的STL模型目录)
    if stl_dir is None:
        stl_dir = config.STL_MODEL_DIR
    else:
        stl_dir = Path(stl_dir)
    
    # Define STL files to load
    stl_files = [
        (config.SHELL_STL_FILE, "Shell", (0.2, 0.5, 0.9, 0.35)),
    ]
    
    if not shell_only:
        stl_files.append(
            (config.CHANNEL_STL_FILE, "Channel", (0.9, 0.35, 0.2, 0.28))
        )
    
    # Load meshes
    meshes = []
    info_lines = []
    
    for filename, label, color in stl_files:
        stl_path = stl_dir / filename
        
        if not stl_path.exists():
            print(f"[warning] {label} STL not found at {stl_path}, skipping.")
            continue
        
        mesh = _load_mesh_mm(stl_path)
        mean_r, max_r = _mesh_distance_stats(mesh)
        
        info = f"{filename} | mean radius={mean_r:.2f} mm, max radius={max_r:.2f} mm"
        print(info)
        info_lines.append(info)
        
        meshes.append((mesh, filename, color))
    
    if not meshes:
        raise FileNotFoundError(f"No STL files found in {stl_dir}")
    
    # Create plot
    fig = _plot_meshes(meshes, "\n".join(info_lines))
    
    # Save if path provided
    if save_path is None:
        save_path = Path(__file__).resolve().parents[2] / config.FIGURES_OUTPUT_DIR / config.STL_GEOMETRY_FIGURE
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=dpi)
    print(f"\n✓ STL geometry plot saved to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        return fig


def plot_mesh_wireframe(
    ax,
    mesh: np.ndarray,
    color: str = 'gray',
    alpha: float = 0.2,
    label: Optional[str] = None
) -> Optional[plt.Rectangle]:
    """Plot mesh as wireframe on existing 3D axis.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        3D axis to plot on.
    mesh : np.ndarray
        Mesh data in STL format (n, 4, 3) or (n, 3, 3).
    color : str
        Face color for mesh.
    alpha : float
        Transparency level.
    label : str, optional
        Label for legend.
        
    Returns
    -------
    handle : matplotlib.patches.Patch or None
        Legend handle if label provided.
    """
    if mesh.shape[1] == 4:
        vertices = mesh[:, 1:4, :]
    else:
        vertices = mesh
    
    collection = Poly3DCollection(
        vertices,
        alpha=alpha,
        facecolors=color,
        edgecolors='black',
        linewidths=0.1
    )
    ax.add_collection3d(collection)
    
    if label:
        from matplotlib.patches import Patch
        return Patch(facecolor=color, alpha=alpha, label=label)
    return None


def _load_mesh_mm(stl_path: Path) -> np.ndarray:
    """Load STL mesh and extract vertices in mm."""
    mesh = load_stl_mesh(str(stl_path))
    if mesh.size == 0:
        raise ValueError("STL mesh is empty")
    # Handle format: (n, 4, 3) with normals -> extract vertices only
    if mesh.shape[1] == 4:
        mesh = mesh[:, 1:, :]
    return mesh


def _mesh_distance_stats(mesh: np.ndarray) -> Tuple[float, float]:
    """Calculate mesh distance statistics from origin."""
    pts = mesh.reshape(-1, 3)
    radii = np.linalg.norm(pts, axis=1)
    return float(np.mean(radii)), float(np.max(radii))


def _set_axes_equal(ax) -> None:
    """Set equal aspect ratio for 3D axes."""
    limits = np.array(
        [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()],
        dtype=float,
    )
    spans = limits[:, 1] - limits[:, 0]
    centres = np.mean(limits, axis=1)
    max_span = max(spans)
    half = max_span / 2.0
    ax.set_xlim3d(centres[0] - half, centres[0] + half)
    ax.set_ylim3d(centres[1] - half, centres[1] + half)
    ax.set_zlim3d(centres[2] - half, centres[2] + half)


def _plot_meshes(
    meshes: List[Tuple[np.ndarray, str, tuple]],
    title: str
) -> plt.Figure:
    """Internal function to create mesh plot."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter([0.0], [0.0], [0.0], color="red", s=40, label="Origin")
    
    all_points = []
    for mesh, label, facecolor in meshes:
        collection = Poly3DCollection(
            mesh,
            facecolor=facecolor,
            edgecolor=(0.1, 0.1, 0.1, 0.1),
            linewidths=0.2,
        )
        collection.set_label(label)
        ax.add_collection3d(collection)
        all_points.append(mesh.reshape(-1, 3))
    
    pts = np.concatenate(all_points, axis=0)
    max_range = np.linalg.norm(pts, axis=1).max()
    for setter in (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d):
        setter(-max_range, max_range)
    _set_axes_equal(ax)
    
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    return fig


class GeometryViewer:
    """Class-based interface for geometry visualization.
    
    Example
    -------
    >>> viewer = GeometryViewer()
    >>> viewer.load_stl()  # Load default STL files
    >>> viewer.plot()
    >>> 
    >>> # Or with simple geometry
    >>> viewer.load_simple('standard')
    >>> viewer.plot()
    """
    
    def __init__(self, stl_dir: Optional[Union[str, Path]] = None):
        """Initialize the geometry viewer.
        
        Parameters
        ----------
        stl_dir : str or Path, optional
            Directory containing STL files. If None, uses package-bundled STL models.
        """
        if stl_dir is None:
            # 使用包内置的STL模型目录
            self.stl_dir = config.STL_MODEL_DIR
        else:
            self.stl_dir = Path(stl_dir)
        
        self.shell_mesh: Optional[np.ndarray] = None
        self.channel_mesh: Optional[np.ndarray] = None
        self.geometry_source: str = 'none'
    
    def load_stl(self, shell_only: bool = False) -> 'GeometryViewer':
        """Load STL geometry files.
        
        Parameters
        ----------
        shell_only : bool
            If True, only load shell (skip channel).
            
        Returns
        -------
        self : GeometryViewer
            Returns self for method chaining.
        """
        shell_path = self.stl_dir / config.SHELL_STL_FILE
        if shell_path.exists():
            self.shell_mesh = _load_mesh_mm(shell_path)
            print(f"[info] Loaded shell: {len(self.shell_mesh)} triangles")
        else:
            print(f"[warning] Shell STL not found: {shell_path}")
        
        if not shell_only:
            channel_path = self.stl_dir / config.CHANNEL_STL_FILE
            if channel_path.exists():
                self.channel_mesh = _load_mesh_mm(channel_path)
                print(f"[info] Loaded channel: {len(self.channel_mesh)} triangles")
            else:
                print(f"[warning] Channel STL not found: {channel_path}")
        
        self.geometry_source = 'stl'
        return self
    
    def load_simple(self, geometry_type: str = 'standard') -> 'GeometryViewer':
        """Load simple analytical geometry.
        
        Parameters
        ----------
        geometry_type : str
            Detail level: 'minimal', 'standard', 'detailed'.
            
        Returns
        -------
        self : GeometryViewer
            Returns self for method chaining.
        """
        from ..testing import create_simple_icf_geometry
        shell, channel = create_simple_icf_geometry(geometry_type)
        
        # Extract vertices (without normals)
        self.shell_mesh = shell[:, 1:, :] if shell.shape[1] == 4 else shell
        self.channel_mesh = channel[:, 1:, :] if channel.shape[1] == 4 else channel
        
        print(f"[info] Created simple {geometry_type} geometry")
        print(f"[info] Shell: {len(self.shell_mesh)} triangles")
        print(f"[info] Channel: {len(self.channel_mesh)} triangles")
        
        self.geometry_source = 'simple'
        return self
    
    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: int = 150
    ) -> Optional[plt.Figure]:
        """Plot the loaded geometry.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        show : bool
            Whether to display interactively.
        dpi : int
            Resolution for saved figure.
            
        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        meshes = []
        
        if self.shell_mesh is not None:
            meshes.append((self.shell_mesh, "Shell", (0.2, 0.5, 0.9, 0.35)))
        
        if self.channel_mesh is not None:
            meshes.append((self.channel_mesh, "Channel", (0.9, 0.35, 0.2, 0.28)))
        
        if not meshes:
            raise ValueError("No geometry loaded. Call load_stl() or load_simple() first.")
        
        title = f"Geometry ({self.geometry_source})"
        fig = _plot_meshes(meshes, title)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi)
            print(f"[info] Saved to {save_path}")
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def get_mesh_info(self) -> dict:
        """Get information about loaded meshes.
        
        Returns
        -------
        info : dict
            Dictionary with mesh statistics.
        """
        info = {'source': self.geometry_source}
        
        if self.shell_mesh is not None:
            mean_r, max_r = _mesh_distance_stats(self.shell_mesh)
            info['shell'] = {
                'n_triangles': len(self.shell_mesh),
                'mean_radius_mm': mean_r,
                'max_radius_mm': max_r
            }
        
        if self.channel_mesh is not None:
            mean_r, max_r = _mesh_distance_stats(self.channel_mesh)
            info['channel'] = {
                'n_triangles': len(self.channel_mesh),
                'mean_radius_mm': mean_r,
                'max_radius_mm': max_r
            }
        
        return info
