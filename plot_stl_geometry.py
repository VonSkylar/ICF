"""Visualise the target STL geometry in millimetres relative to the origin."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from icf_simulation import load_stl_mesh


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
    plt.savefig('Figures/stl_geometry_visualization.png', dpi=150)
    print("\n✓ STL geometry plot displayed.")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stl",
        type=Path,
        default=Path("Target_ball_model.stl"),
        help="Path to the primary STL file to visualise.",
    )
    parser.add_argument(
        "--nTOF_without_scintillant",
        type=Path,
        default=None,
        help="Path to the nTOF_without_scintillant STL file to overlay (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stl.exists():
        raise FileNotFoundError(f"Cannot find STL file at {args.stl}")

    mesh_mm = load_mesh_mm(args.stl)
    mean_r, max_r = mesh_distance_stats_mm(mesh_mm)
    primary_info = (
        f"{args.stl.name} | mean radius={mean_r:.2f} mm, "
        f"max radius={max_r:.2f} mm"
    )
    print(primary_info)

    meshes_to_plot: list[tuple[np.ndarray, str, tuple[float, float, float, float]]] = [
        (mesh_mm, args.stl.name, (0.2, 0.5, 0.9, 0.35)),
    ]
    info_lines = [primary_info]

    # 尝试自动查找 nTOF_without_scintillant.STL，但优先使用命令行参数
    nTOF_without_scintillant_path = args.nTOF_without_scintillant if args.nTOF_without_scintillant else args.stl.with_name("nTOF_without_scintillant.STL")
    if nTOF_without_scintillant_path and nTOF_without_scintillant_path.exists():
        nTOF_without_scintillant_mesh_mm = load_mesh_mm(nTOF_without_scintillant_path)
        nTOF_without_scintillant_mean, nTOF_without_scintillant_max = mesh_distance_stats_mm(nTOF_without_scintillant_mesh_mm)
        nTOF_without_scintillant_info = (
            f"{nTOF_without_scintillant_path.name} | mean radius={nTOF_without_scintillant_mean:.2f} mm, "
            f"max radius={nTOF_without_scintillant_max:.2f} mm"
        )
        print(nTOF_without_scintillant_info)
        info_lines.append(nTOF_without_scintillant_info)
        meshes_to_plot.append((nTOF_without_scintillant_mesh_mm, nTOF_without_scintillant_path.name, (0.9, 0.35, 0.2, 0.28)))
    else:
        print(f"nTOF_without_scintillant STL not found at {nTOF_without_scintillant_path}, skipping overlay.")

    plot_meshes_mm(meshes_to_plot, "\n".join(info_lines))


if __name__ == "__main__":
    main()
