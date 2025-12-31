"""
简化几何模块的单元测试
"""

import numpy as np
import pytest

from icf_simulation.testing import (
    create_simple_sphere,
    create_simple_tube,
    create_simple_box,
    create_simple_icf_geometry,
    validate_mesh,
)


class TestSimpleSphere:
    """测试简化球体生成"""
    
    def test_sphere_creation(self):
        """测试基本球体创建"""
        mesh = create_simple_sphere(center=(0, 0, 0), radius=1.0)
        assert mesh is not None
        assert len(mesh) > 0
    
    def test_sphere_center(self):
        """测试球体中心位置"""
        center = (1.0, 2.0, 3.0)
        mesh = create_simple_sphere(center=center, radius=1.0, subdivisions=1)
        
        # 计算网格中心
        mesh_center = np.mean(mesh.reshape(-1, 3), axis=0)
        np.testing.assert_array_almost_equal(mesh_center, center, decimal=1)
    
    def test_sphere_radius(self):
        """测试球体半径"""
        radius = 2.5
        mesh = create_simple_sphere(center=(0, 0, 0), radius=radius, subdivisions=2)
        
        # 检查所有顶点到中心的距离
        vertices = mesh.reshape(-1, 3)
        distances = np.linalg.norm(vertices, axis=1)
        np.testing.assert_array_almost_equal(distances, radius, decimal=5)
    
    def test_sphere_validation(self):
        """测试球体网格验证"""
        mesh = create_simple_sphere(center=(0, 0, 0), radius=1.0)
        is_valid, message = validate_mesh(mesh)
        assert is_valid, f"Sphere validation failed: {message}"


class TestSimpleTube:
    """测试简化圆管生成"""
    
    def test_tube_creation(self):
        """测试基本圆管创建"""
        mesh = create_simple_tube(
            start_point=(0, 0, 0),
            end_point=(0, 0, 10),
            inner_radius=1.0,
            outer_radius=1.5
        )
        assert mesh is not None
        assert len(mesh) > 0
    
    def test_tube_length(self):
        """测试圆管长度"""
        start = (0, 0, 0)
        end = (0, 0, 5)
        mesh = create_simple_tube(
            start_point=start,
            end_point=end,
            inner_radius=0.5,
            outer_radius=1.0
        )
        
        # 检查z方向范围
        z_coords = mesh.reshape(-1, 3)[:, 2]
        z_range = np.max(z_coords) - np.min(z_coords)
        assert abs(z_range - 5.0) < 0.1
    
    def test_tube_validation(self):
        """测试圆管网格验证"""
        mesh = create_simple_tube(
            start_point=(0, 0, 0),
            end_point=(0, 0, 10),
            inner_radius=1.0,
            outer_radius=2.0
        )
        is_valid, message = validate_mesh(mesh)
        assert is_valid, f"Tube validation failed: {message}"


class TestSimpleBox:
    """测试简化盒子生成"""
    
    def test_box_creation(self):
        """测试基本盒子创建"""
        mesh = create_simple_box(center=(0, 0, 0), size=(1, 2, 3))
        assert mesh is not None
        assert len(mesh) == 12  # 6 faces * 2 triangles
    
    def test_box_size(self):
        """测试盒子尺寸"""
        size = (2.0, 4.0, 6.0)
        mesh = create_simple_box(center=(0, 0, 0), size=size)
        
        # 检查各方向范围
        vertices = mesh.reshape(-1, 3)
        for i, s in enumerate(size):
            coord_range = np.max(vertices[:, i]) - np.min(vertices[:, i])
            assert abs(coord_range - s) < 1e-10
    
    def test_box_validation(self):
        """测试盒子网格验证"""
        mesh = create_simple_box(center=(0, 0, 0), size=(1, 1, 1))
        is_valid, message = validate_mesh(mesh)
        assert is_valid, f"Box validation failed: {message}"


class TestICFGeometry:
    """测试ICF几何生成"""
    
    @pytest.mark.parametrize("geometry_type", ["minimal", "standard", "detailed"])
    def test_icf_geometry_creation(self, geometry_type):
        """测试不同精度级别的ICF几何创建"""
        shell, channel = create_simple_icf_geometry(geometry_type=geometry_type)
        
        assert shell is not None
        assert channel is not None
        assert len(shell) > 0
        assert len(channel) > 0
    
    def test_icf_geometry_validation(self):
        """测试ICF几何验证"""
        shell, channel = create_simple_icf_geometry(geometry_type='standard')
        
        is_valid_shell, msg_shell = validate_mesh(shell)
        is_valid_channel, msg_channel = validate_mesh(channel)
        
        assert is_valid_shell, f"Shell validation failed: {msg_shell}"
        assert is_valid_channel, f"Channel validation failed: {msg_channel}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
