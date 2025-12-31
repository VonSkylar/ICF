#!/usr/bin/env python
"""
STL Geometry Visualization Script

This script visualizes the STL geometry files used in the ICF simulation.
STL files are bundled with the icf_simulation package.

Usage:
    python plot_stl_geometry.py
    python plot_stl_geometry.py --shell-only
"""

from pathlib import Path
import sys
import argparse

# 添加项目根目录到路径
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from icf_simulation import config
from icf_simulation.plotting import GeometryViewer


def main():
    """脚本入口点"""
    parser = argparse.ArgumentParser(description="Visualize STL geometry")
    parser.add_argument("--stl-dir", type=Path, default=None,
                        help="Directory containing STL files (default: use package-bundled files)")
    parser.add_argument("--shell-only", action="store_true",
                        help="Only plot the shell STL (skip channel overlay)")
    
    args = parser.parse_args()
    
    # 创建几何查看器（如未指定 stl_dir，将使用包内置的 STL 模型）
    viewer = GeometryViewer(stl_dir=args.stl_dir)
    viewer.load_stl(shell_only=args.shell_only)
    
    # 打印网格信息
    info = viewer.get_mesh_info()
    if 'shell' in info:
        print(f"{config.SHELL_STL_FILE} | mean radius={info['shell']['mean_radius_mm']:.2f} mm, "
              f"max radius={info['shell']['max_radius_mm']:.2f} mm")
    if 'channel' in info:
        print(f"{config.CHANNEL_STL_FILE} | mean radius={info['channel']['mean_radius_mm']:.2f} mm, "
              f"max radius={info['channel']['max_radius_mm']:.2f} mm")
    
    # 保存并显示
    save_path = project_dir / config.FIGURES_OUTPUT_DIR / config.STL_GEOMETRY_FIGURE
    viewer.plot(save_path=str(save_path), dpi=config.QUICK_PLOT_DPI)
    print(f"\n✓ STL geometry plot saved to {save_path}")


if __name__ == "__main__":
    main()
