"""Visualise the target STL geometry in millimetres relative to the origin."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from icf_simulation import load_stl_mesh, config


def _set_axes_equal(ax: plt.Axes) -> None:
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


def load_mesh_mm(stl_path: Path) -> np.ndarray:
    mesh = load_stl_mesh(str(stl_path))
    if mesh.size == 0:
        raise ValueError("STL mesh is empty")
    # Handle new format: (n, 4, 3) with normals -> extract vertices only
    if mesh.shape[1] == 4:
        # Format is [normal, v0, v1, v2], extract vertices
        mesh = mesh[:, 1:, :]  # Shape becomes (n, 3, 3)
    return mesh


def mesh_distance_stats_mm(mesh: np.ndarray) -> tuple[float, float]:
    pts = mesh.reshape(-1, 3)
    radii = np.linalg.norm(pts, axis=1)
    return float(np.mean(radii)), float(np.max(radii))


def plot_meshes_mm(
    meshes: list[tuple[np.ndarray, str, tuple[float, float, float, float]]],
    title: str,
) -> None:
    if not meshes:
        raise ValueError("At least one mesh is required for plotting")

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
    save_path = Path(config.FIGURES_OUTPUT_DIR) / config.STL_GEOMETRY_FIGURE
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=config.QUICK_PLOT_DPI)
    print(f"\n✓ STL geometry plot saved to {save_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stl-dir",
        type=Path,
        default=None,
        help="Directory containing STL files (default: STL_model).",
    )
    parser.add_argument(
        "--shell-only",
        action="store_true",
        help="Only plot the shell STL (skip channel overlay).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Determine STL directory
    base_dir = Path(__file__).resolve().parent
    stl_dir = args.stl_dir if args.stl_dir else base_dir / config.STL_MODEL_DIR
    
    # Define STL files to load
    stl_files_to_load = [
        (config.SHELL_STL_FILE, "Shell", (0.2, 0.5, 0.9, 0.35)),
    ]
    
    if not args.shell_only:
        stl_files_to_load.append(
            (config.CHANNEL_STL_FILE, "Channel", (0.9, 0.35, 0.2, 0.28))
        )
    
    # Load and process all STL files
    meshes_to_plot = []
    info_lines = []
    
    for filename, label, color in stl_files_to_load:
        stl_path = stl_dir / filename
        
        if not stl_path.exists():
            print(f"[warning] {label} STL not found at {stl_path}, skipping.")
            continue
        
        # Load mesh
        mesh_mm = load_mesh_mm(stl_path)
        mean_r, max_r = mesh_distance_stats_mm(mesh_mm)
        
        # Print and store info
        info = f"{filename} | mean radius={mean_r:.2f} mm, max radius={max_r:.2f} mm"
        print(info)
        info_lines.append(info)
        
        # Add to plot list
        meshes_to_plot.append((mesh_mm, filename, color))
    
    if not meshes_to_plot:
        raise FileNotFoundError(f"No STL files found in {stl_dir}")

    plot_meshes_mm(meshes_to_plot, "\n".join(info_lines))


if __name__ == "__main__":
    main()
