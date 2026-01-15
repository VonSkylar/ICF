"""
ICF 模拟核心模块

该子包包含模拟的核心功能模块：
- constants: 物理常数和调试标志
- cross_section: 截面数据处理
- data_classes: 数据结构定义（MeshGeometry, DetectorPlane, NeutronRecord）
- geometry: 几何处理和射线追踪
- io_utils: 输入输出工具
- kinematics: 运动学计算
- sampling: 抽样方法
- simulation: 模拟主逻辑
- stl_utils: STL文件处理
- transport: 输运计算
"""

# 常数
from .constants import (
    AVOGADRO_CONSTANT,
    BARN_TO_M2,
    NEUTRON_MASS_KG,
    DEBUG,
    reset_geometry_leak_stats,
    print_geometry_leak_stats,
)

# 数据类
from .data_classes import (
    MeshGeometry,
    DetectorPlane,
    NeutronRecord,
)

# 截面数据
from .cross_section import (
    load_mfp_data_from_csv,
    calculate_pe_macro_sigma,
    get_macro_sigma_at_energy,
    get_mfp_energy_dependent,
    MFP_DATA_AL,
    MFP_DATA_PE,
)

# 几何处理
from .geometry import (
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
from .io_utils import (
    export_neutron_records_to_csv,
    export_neutron_trajectories_to_csv,
)

# 运动学
from .kinematics import (
    energy_to_speed,
    scatter_energy_elastic,
    scatter_neutron_elastic_cms_to_lab,
)

# 抽样
from .sampling import (
    sample_neutron_energy,
    sample_isotropic_direction,
    sample_direction_in_cone,
)

# 模拟
from .simulation import (
    simulate_neutron_history,
    run_simulation,
)

# STL工具
from .stl_utils import load_stl_mesh

# 输运
from .transport import (
    transport_through_slab,
    propagate_through_mesh_material,
    simulate_in_aluminium,
    propagate_to_scintillator,
    unified_transport,
    MaterialType,
    MaterialConfig,
)

__all__ = [
    # 常数
    'AVOGADRO_CONSTANT',
    'BARN_TO_M2',
    'NEUTRON_MASS_KG',
    'DEBUG',
    'reset_geometry_leak_stats',
    'print_geometry_leak_stats',
    # 数据类
    'MeshGeometry',
    'DetectorPlane',
    'NeutronRecord',
    # 截面
    'load_mfp_data_from_csv',
    'calculate_pe_macro_sigma',
    'get_macro_sigma_at_energy',
    'get_mfp_energy_dependent',
    'MFP_DATA_AL',
    'MFP_DATA_PE',
    # 几何
    'prepare_mesh_geometry',
    'ray_mesh_intersection',
    'find_exit_with_retry',
    'mesh_distance_statistics',
    'infer_mesh_axis',
    'build_orthonormal_frame',
    'build_detector_plane_from_mesh',
    'build_default_detector_plane',
    'build_circular_detector_plane',
    # IO
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
    # STL
    'load_stl_mesh',
    # 输运
    'transport_through_slab',
    'propagate_through_mesh_material',
    'simulate_in_aluminium',
    'propagate_to_scintillator',
    'unified_transport',
    'MaterialType',
    'MaterialConfig',
]
