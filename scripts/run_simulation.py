#!/usr/bin/env python
"""
ICF Neutron Simulation - Main Runner Script

This script runs the complete ICF neutron simulation.

Usage:
    python run_simulation.py
    python run_simulation.py -n 1000
    python run_simulation.py --no-plot

Cross-section data and STL models are bundled with the icf_simulation package.
Output files (Data/, Figures/) will be saved in the current working directory
or in the specified output directory.
"""

from pathlib import Path
import sys

# 添加项目根目录到路径（确保可以导入 icf_simulation）
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from icf_simulation.runner import run_full_simulation, main as runner_main


def main():
    """脚本入口点"""
    if len(sys.argv) > 1:
        # 如果有命令行参数，使用 argparse 处理
        runner_main()
    else:
        # 默认运行 - 输出到项目目录
        run_full_simulation(output_dir=project_dir)


if __name__ == "__main__":
    main()