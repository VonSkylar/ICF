"""Visualise the target STL geometry to verify orientation and scale.

This utility reuses the STL loader from ``icf_neutron_simulation`` and renders
the mesh with Matplotlib.  It recentres the geometry on the fitted sphere
centre so that visual inspection is easier when the original CAD export is not
aligned with the origin.

Usage
-----
    python plot_stl_geometry.py --stl Target_ball_model.stl --scale 1e-4

The default scale converts STL units that are stored in 0.1 mm to metres.  If
your STL uses millimetres or metres, adjust ``--scale`` accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from icf_neutron_simulation import load_stl_mesh, mesh_radius_bounds


def _set_axes_equal(ax: plt.Axes) -> None:
    """Make 3D axes have equal scale for all directions."""
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


def load_and_prepare_mesh(stl_path: Path, scale: float) -> Tuple[np.ndarray, float, float]:
    """Load STL, fit radii, and return centred & scaled mesh data."""
    mesh = load_stl_mesh(str(stl_path))
    inner_raw, outer_raw, centre = mesh_radius_bounds(mesh)
    mesh_centered = mesh - centre[None, None, :]
    mesh_scaled = mesh_centered * scale
    inner_radius = inner_raw * scale
    outer_radius = outer_raw * scale
    return mesh_scaled, inner_radius, outer_radius


def plot_mesh(mesh: np.ndarray, info: str) -> None:
    """Render the STL mesh with Matplotlib."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    collection = Poly3DCollection(
        mesh,
        facecolor=(0.2, 0.5, 0.9, 0.35),
        edgecolor=(0.1, 0.1, 0.1, 0.1),
        linewidths=0.2,
    )
    ax.add_collection3d(collection)

    # Plot the origin (fitted centre)
    ax.scatter([0.0], [0.0], [0.0], color="red", s=40, label="Fitted centre")

    # Set limits based on mesh extents
    pts = mesh.reshape(-1, 3)
    max_range = np.linalg.norm(pts, axis=1).max()
    for setter in (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d):
        setter(-max_range, max_range)
    _set_axes_equal(ax)

    ax.set_xlabel("x (scaled units)")
    ax.set_ylabel("y (scaled units)")
    ax.set_zlabel("z (scaled units)")
    ax.set_title(f"STL Geometry: {info}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stl",
        type=Path,
        default=Path("Target_ball_model.stl"),
        help="Path to the STL file to visualise.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0e-4,
        help="Scale factor applied to STL coordinates (default converts 0.1 mm to metres).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stl.exists():
        raise FileNotFoundError(f"Cannot find STL file at {args.stl}")

    mesh_scaled, inner_radius, outer_radius = load_and_prepare_mesh(args.stl, args.scale)
    thickness = outer_radius - inner_radius
    info = (
        f"{args.stl.name} | inner={inner_radius:.4f}, outer={outer_radius:.4f}, "
        f"thickness={thickness:.4f} (scaled units)"
    )
    print(info)
    plot_mesh(mesh_scaled, info)


if __name__ == "__main__":
    main()
