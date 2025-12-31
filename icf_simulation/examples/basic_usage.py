"""
ICF 模拟包基本使用示例

这个示例展示了如何使用 icf_simulation 包的基本功能。
"""

import numpy as np
import matplotlib.pyplot as plt

# 导入主要模块
from icf_simulation import (
    # 数据类
    NeutronRecord,
    MeshGeometry,
    DetectorPlane,
    
    # 常数
    NEUTRON_MASS_KG,
    
    # 功能函数
    energy_to_speed,
    sample_isotropic_direction,
    sample_direction_in_cone,
)

# 导入子包
from icf_simulation.testing import (
    create_simple_sphere,
    create_simple_icf_geometry,
    print_mesh_info,
    run_quick_test,
)

from icf_simulation.plotting import (
    TrajectoryPlotter,
    GeometryViewer,
)


def example_basic_physics():
    """基本物理量计算示例"""
    print("=" * 60)
    print("基本物理量计算示例")
    print("=" * 60)
    
    # 2.45 MeV 中子速度
    energy_mev = 2.45
    speed = energy_to_speed(energy_mev)
    print(f"\n{energy_mev} MeV 中子速度: {speed:.3e} m/s")
    print(f"中子质量: {NEUTRON_MASS_KG:.6e} kg")
    
    # 各向同性方向抽样
    print("\n各向同性方向抽样 (5个样本):")
    for i in range(5):
        direction = sample_isotropic_direction()
        print(f"  样本 {i+1}: [{direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}]")
    
    # 锥内方向抽样
    axis = np.array([0.0, 0.0, 1.0])
    half_angle_deg = 10.0
    print(f"\n锥内方向抽样 (轴=[0,0,1], 半角={half_angle_deg}°, 5个样本):")
    for i in range(5):
        direction = sample_direction_in_cone(axis, half_angle_deg)
        angle = np.arccos(np.dot(direction, axis)) * 180 / np.pi
        print(f"  样本 {i+1}: 与轴夹角 = {angle:.2f}°")


def example_simple_geometry():
    """简化几何创建示例"""
    print("\n" + "=" * 60)
    print("简化几何创建示例")
    print("=" * 60)
    
    # 创建单个球体
    sphere = create_simple_sphere(
        center=(0, 0, 0),
        radius=1.0,
        subdivisions=2
    )
    print_mesh_info(sphere, "测试球体")
    
    # 创建ICF几何
    print("\n创建完整ICF几何:")
    shell, channel = create_simple_icf_geometry(geometry_type='standard')
    print_mesh_info(shell, "ICF壳层")
    print_mesh_info(channel, "ICF通道")


def example_validation():
    """验证测试示例"""
    print("\n" + "=" * 60)
    print("几何模块验证")
    print("=" * 60)
    
    run_quick_test()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("ICF 模拟包基本使用示例")
    print("=" * 60)
    
    # 运行各个示例
    example_basic_physics()
    example_simple_geometry()
    example_validation()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
