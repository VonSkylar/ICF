"""
Plotting subpackage for ICF neutron simulation visualization.

This subpackage provides comprehensive visualization tools for:
- 3D and 2D trajectory plotting
- STL geometry visualization
- Simulation result analysis plots

Example usage:
    from icf_simulation.plotting import plot_trajectories_3d, load_trajectory_data
    
    trajectories = load_trajectory_data('Data/neutron_trajectories.csv')
    plot_trajectories_3d(trajectories, max_trajectories=50)
    
    # Visualize simulation results
    from icf_simulation.plotting import visualize_neutron_data, print_statistics
    visualize_neutron_data(records, save_path='Figures/results')
"""

from .trajectories import (
    load_trajectory_data,
    plot_trajectories_3d,
    plot_trajectories_2d,
    TrajectoryPlotter,
)

from .geometry_viewer import (
    plot_stl_geometry,
    plot_mesh_wireframe,
    GeometryViewer,
)

from .results import (
    visualize_neutron_data,
    visualize_detector_hits,
    print_statistics,
)

__all__ = [
    # Trajectory plotting
    "load_trajectory_data",
    "plot_trajectories_3d",
    "plot_trajectories_2d",
    "TrajectoryPlotter",
    # Geometry visualization
    "plot_stl_geometry",
    "plot_mesh_wireframe",
    "GeometryViewer",
    # Simulation results
    "visualize_neutron_data",
    "visualize_detector_hits",
    "print_statistics",
]
