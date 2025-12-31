# ICF 中子输运模拟包

惯性约束聚变(ICF)诊断中子输运蒙特卡罗模拟工具包。

## 安装

```bash
# 开发模式安装
pip install -e .

# 或者安装开发依赖
pip install -e ".[dev]"
```

## 包结构

```
icf_simulation/
├── __init__.py           # 主包入口
├── config.py             # 配置参数
├── data_paths.py         # 包内数据路径工具
├── runner.py             # 高级模拟运行器
├── core/                 # 核心模块子包
│   ├── __init__.py
│   ├── constants.py      # 物理常数
│   ├── cross_section.py  # 截面数据处理
│   ├── data_classes.py   # 数据结构定义
│   ├── geometry.py       # 几何处理
│   ├── io_utils.py       # 输入输出工具
│   ├── kinematics.py     # 运动学计算
│   ├── sampling.py       # 抽样方法
│   ├── simulation.py     # 模拟主逻辑
│   ├── stl_utils.py      # STL文件处理
│   └── transport.py      # 输运计算
├── data/                 # 内置数据文件
│   ├── cross_sections/   # 截面数据 (Al.csv, H.csv, C.csv)
│   └── stl_models/       # STL几何模型
├── examples/             # 示例代码
│   └── basic_usage.py
├── plotting/             # 可视化子包
│   ├── __init__.py
│   ├── trajectories.py   # 轨迹绘图
│   ├── geometry_viewer.py # 几何查看器
│   └── results.py        # 模拟结果可视化
└── testing/              # 测试工具子包
    ├── __init__.py
    ├── simple_geometry.py # 简化几何生成
    ├── validation.py      # 验证工具
    └── comparison.py      # 对比工具
```

## 快速开始

### 基本使用

```python
from icf_simulation import run_full_simulation

# 运行完整模拟（使用包内置的截面数据和STL几何）
records = run_full_simulation(n_neutrons=1000)
```

### 访问包内置数据

```python
from icf_simulation import data_paths

# 获取截面数据目录
cross_section_dir = data_paths.get_cross_section_dir()

# 获取STL模型目录
stl_dir = data_paths.get_stl_dir()

# 获取特定截面文件
al_csv = data_paths.get_cross_section_file('Al')

# 列出所有可用的截面数据
available_elements = data_paths.list_cross_section_files()
```

### 使用简化几何进行测试

```python
from icf_simulation.testing import (
    create_simple_sphere, 
    create_simple_icf_geometry,
    run_quick_test
)

# 快速验证几何模块
run_quick_test()

# 创建简化的ICF几何
shell_mesh, channel_mesh = create_simple_icf_geometry(detail_level='standard')

# 创建单个球体
sphere = create_simple_sphere(center=[0, 0, 0], radius=1.0, n_subdivisions=2)
```

### 轨迹可视化

```python
from icf_simulation.plotting import TrajectoryPlotter, load_trajectory_data

# 加载轨迹数据
data = load_trajectory_data("Data/neutron_trajectories.csv")

# 使用绘图器
plotter = TrajectoryPlotter(data)
plotter.plot_3d()
plotter.plot_2d_projections()
```

### 几何可视化

```python
from icf_simulation.plotting import GeometryViewer

# 创建几何查看器（默认使用包内置的STL模型）
viewer = GeometryViewer()
viewer.load_stl()  # 加载包内置的STL模型
viewer.plot()
```

## 脚本工具

脚本位于 `scripts/` 目录：

```bash
# 运行模拟
python scripts/run_simulation.py

# 绘制轨迹图
python scripts/plot_neutron_trajectories.py

# 查看STL几何
python scripts/plot_stl_geometry.py
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_geometry.py

# 带覆盖率报告
pytest --cov=icf_simulation
```

## 许可证

MIT License
