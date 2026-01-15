"""
ICF 中子输运模拟包

该包提供惯性约束聚变(ICF)诊断中的中子输运模拟功能。

子包结构:
-----------
- core: 核心模块（物理常数、截面数据、几何处理、输运计算等）
- plotting: 可视化工具（轨迹绘图、几何查看器等）
- testing: 测试工具（简化几何、验证、对比等）
- data: 内置数据文件（截面数据、STL模型）
- examples: 示例代码

顶层模块:
-----------
- config: 模拟配置参数
- data_paths: 包内数据文件路径工具
- runner: 高级模拟运行器

核心模块 (icf_simulation.core):
-----------
- constants: 物理常数
- cross_section: 截面数据处理
- data_classes: 数据结构定义
- geometry: 几何处理
- io_utils: 输入输出工具
- kinematics: 运动学计算
- sampling: 抽样方法
- simulation: 模拟主逻辑
- stl_utils: STL文件处理
- transport: 输运计算

使用示例:
-----------
>>> from icf_simulation import MeshGeometry, DetectorPlane, NeutronRecord
>>> from icf_simulation.plotting import TrajectoryPlotter
>>> from icf_simulation.testing import create_simple_sphere

>>> # 使用简单几何进行测试
>>> from icf_simulation.testing import run_quick_test
>>> run_quick_test()

>>> # 访问包内置数据文件
>>> from icf_simulation import data_paths
>>> cross_section_dir = data_paths.get_cross_section_dir()
>>> stl_dir = data_paths.get_stl_dir()

>>> # 直接访问核心模块
>>> from icf_simulation.core import geometry, transport
"""

# 版本信息
__version__ = '0.2.0'
__author__ = 'ICF Simulation Team'

# ============================================================================
# 核心模块导入 (从 core 子包)
# ============================================================================

# 导入 core 子包
from . import core

# 常数
from .core.constants import (
    AVOGADRO_CONSTANT,
    BARN_TO_M2,
    NEUTRON_MASS_KG,
    DEBUG,
    reset_geometry_leak_stats,
    print_geometry_leak_stats,
)

# 用户可配置参数 (从 config 导入)
from .config import DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG

# 数据类
from .core.data_classes import (
    MeshGeometry,
    DetectorPlane,
    NeutronRecord,
)

# 截面数据
from .core.cross_section import (
    load_mfp_data_from_csv,
    calculate_pe_macro_sigma,
    get_macro_sigma_at_energy,
    get_mfp_energy_dependent,
    MFP_DATA_AL,
    MFP_DATA_PE,
)

# 几何处理
from .core.geometry import (
    prepare_mesh_geometry,
    ray_mesh_intersection,
    find_exit_with_retry,
    mesh_distance_statistics,
    infer_mesh_axis,
    build_orthonormal_frame,
    build_detector_plane_from_mesh,
    build_default_detector_plane,
    build_circular_detector_plane,
)

# IO工具
from .core.io_utils import (
    export_neutron_records_to_csv,
    export_neutron_trajectories_to_csv,
)

# 运动学
from .core.kinematics import (
    energy_to_speed,
    scatter_energy_elastic,
    scatter_neutron_elastic_cms_to_lab,
)

# 抽样
from .core.sampling import (
    sample_neutron_energy,
    sample_isotropic_direction,
    sample_direction_in_cone,
)

# 模拟
from .core.simulation import (
    simulate_neutron_history,
    run_simulation,
)

# 高级运行器
from .runner import (
    run_full_simulation,
    load_cross_section_data,
    load_stl_geometry,
)

# STL工具
from .core.stl_utils import load_stl_mesh

# 输运
from .core.transport import (
    transport_through_slab,
    propagate_through_mesh_material,
    simulate_in_aluminium,
    propagate_to_scintillator,
)

# 数据路径工具
from . import data_paths

# ============================================================================
# 子包导入
# ============================================================================

from . import plotting
from . import testing

# 可视化 (从 plotting 子包导入)
from .plotting.results import (
    visualize_neutron_data,
    visualize_detector_hits,
    print_statistics,
)

# 从子包中导出常用函数以方便使用
from .testing.simple_geometry import (
    create_simple_sphere,
    create_simple_tube,
    create_simple_box,
    create_simple_icf_geometry,
    print_mesh_info,
)

from .plotting.trajectories import (
    load_trajectory_data,
    plot_trajectories_3d,
    plot_trajectories_2d,
    TrajectoryPlotter,
)

from .plotting.geometry_viewer import (
    plot_stl_geometry,
    plot_mesh_wireframe,
    GeometryViewer,
)

# ============================================================================
# 公开API
# ============================================================================

__all__ = [
    # 版本
    '__version__',
    '__author__',
    
    # 子包
    'core',
    'plotting',
    'testing',
    'data_paths',
    
    # 常数
    'AVOGADRO_CONSTANT',
    'BARN_TO_M2',
    'NEUTRON_MASS_KG',
    'DEBUG',
    'DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG',
    'reset_geometry_leak_stats',
    'print_geometry_leak_stats',
    
    # 数据类
    'MeshGeometry',
    'DetectorPlane',
    'NeutronRecord',
    
    # 截面数据
    'load_mfp_data_from_csv',
    'calculate_pe_macro_sigma',
    'get_macro_sigma_at_energy',
    'get_mfp_energy_dependent',
    'MFP_DATA_AL',
    'MFP_DATA_PE',
    
    # 几何处理
    'prepare_mesh_geometry',
    'ray_mesh_intersection',
    'find_exit_with_retry',
    'mesh_distance_statistics',
    'infer_mesh_axis',
    'build_orthonormal_frame',
    'build_detector_plane_from_mesh',
    'build_default_detector_plane',
    'build_circular_detector_plane',
    
    # IO工具
    'export_neutron_records_to_csv',
    'export_neutron_trajectories_to_csv',
    
    # 运动学
    'energy_to_speed',
    'scatter_energy_elastic',
    'scatter_neutron_elastic_cms_to_lab',
    
    # 抽样
    'sample_neutron_energy',
    'sample_isotropic_direction',
    'sample_direction_in_cone',
    
    # 模拟
    'simulate_neutron_history',
    'run_simulation',
    'run_full_simulation',
    'load_cross_section_data',
    'load_stl_geometry',
    
    # STL工具
    'load_stl_mesh',
    
    # 输运
    'transport_through_slab',
    'propagate_through_mesh_material',
    'simulate_in_aluminium',
    'propagate_to_scintillator',
    
    # 可视化
    'visualize_neutron_data',
    'visualize_detector_hits',
    'print_statistics',
    
    # 简化几何（从testing子包导出）
    'create_simple_sphere',
    'create_simple_tube',
    'create_simple_box',
    'create_simple_icf_geometry',
    'print_mesh_info',
    
    # 轨迹绘图（从plotting子包导出）
    'load_trajectory_data',
    'plot_trajectories_3d',
    'plot_trajectories_2d',
    'TrajectoryPlotter',
    
    # 几何查看器（从plotting子包导出）
    'plot_stl_geometry',
    'plot_mesh_wireframe',
    'GeometryViewer',
]
