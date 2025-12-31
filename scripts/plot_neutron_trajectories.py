#!/usr/bin/env python
"""
Neutron Trajectory Visualization Script

This script visualizes neutron trajectories from the ICF simulation.

Usage:
    python plot_neutron_trajectories.py
    python plot_neutron_trajectories.py --max-trajectories 50
    python plot_neutron_trajectories.py --no-geometry
"""

from pathlib import Path
import sys
import argparse

# 添加项目根目录到路径
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from icf_simulation import config
from icf_simulation.plotting import (
    load_trajectory_data,
    plot_trajectories_3d,
    plot_trajectories_2d,
)


def main():
    """脚本入口点"""
    parser = argparse.ArgumentParser(description="Plot neutron trajectories")
    parser.add_argument("--max-trajectories", "-n", type=int, 
                        default=config.MAX_TRAJECTORIES_TO_PLOT,
                        help="Maximum number of trajectories to plot")
    parser.add_argument("--no-geometry", action="store_true",
                        help="Don't show STL geometry overlay")
    parser.add_argument("--data-file", type=Path, default=None,
                        help="Path to trajectory data CSV file")
    
    args = parser.parse_args()
    
    # 确定数据文件路径
    data_dir = project_dir / "Data"
    figures_dir = project_dir / "Figures"
    
    if args.data_file:
        trajectory_file = args.data_file
    else:
        trajectory_file = data_dir / config.TRAJECTORY_DATA_CSV
    
    if not trajectory_file.exists():
        print(f"[error] Trajectory file not found: {trajectory_file}")
        print("[info] Please run run_simulation.py first to generate trajectory data.")
        sys.exit(1)
    
    print(f"[info] Loading neutron trajectory data from {trajectory_file}")
    trajectories = load_trajectory_data(str(trajectory_file))
    print(f"[info] Loaded {len(trajectories)} neutron trajectories")
    
    # STL 几何现在来自包内置目录，不需要显式指定
    stl_dir = None  # 使用包内置的 STL 模型
    
    # 创建可视化
    save_base = str(figures_dir / config.TRAJECTORY_FIGURE_BASE)
    
    print("[info] Creating 3D trajectory plot...")
    plot_trajectories_3d(
        trajectories, 
        max_trajectories=args.max_trajectories,
        save_path=save_base,
        show_geometry=not args.no_geometry,
        stl_dir=stl_dir
    )
    
    print("[info] Creating 2D projection plots...")
    plot_trajectories_2d(
        trajectories, 
        max_trajectories=args.max_trajectories,
        save_path=save_base
    )
    
    print("[info] Visualization complete!")


if __name__ == "__main__":
    main()