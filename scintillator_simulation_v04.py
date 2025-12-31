'''Created by Shen on 2025.12.10
   to simulate neutron after it enter scintillator'''

# v02 删除所有读取的stl文件，改为硬编码闪烁体和接收器，增加与铅层相互作用，简单假设：中子在铅中要么完全不损失能量，要么被吸收
# v03 修改产生中子的逻辑，按照双指数衰减公式;增加进度条和时间,修改光子到达时间直方图y轴为对数坐标
# v04 修改光子最大路程为e指数衰减分布

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import os
import struct
import time  # 引入时间库
from tqdm import tqdm  # 引入进度条库

# ==============================================================================
# I. 核心物理与数值常数 (假设单位: m, s, MeV)
# ==============================================================================
AVOGADRO_CONSTANT = 6.02214076e+23  # 阿伏伽德罗常数 (mol⁻¹)
BARN_TO_M2 = 1e-28                   # 截面单位转换: 1 barn = 10⁻²⁸ m²
M_N = 1.67493e-27                    # 中子质量 (kg)
J_PER_MEV = 1.60218e-13              # 能量单位转换: 1 MeV = 1.602e-13 J
SPEED_OF_LIGHT_SCINT = 2.0e8         # 闪烁体中的光速 (m/s) (简化值)

# ==============================================================================
# II. 几何体数据结构与算法
# ==============================================================================

class AnalyticCylinderGeometry:
    """
    解析圆柱体几何体 (Cylinder)，圆心在 Z 轴上。
    底面中心在 (0, 0, 0)，高 H=0.1m，半径 R=0.1m。
    """
    def __init__(self):
        self.R = 0.1  # 半径 (m)
        self.R_sq = self.R**2
        self.H = 0.1  # 高度 (m)
        
        # 边界框 (m)
        self.bounds_min = np.array([-self.R, -self.R, 0.0])
        self.bounds_max = np.array([ self.R,  self.R, self.H])

    def is_inside(self, point: np.ndarray) -> bool:
        """检查点是否在圆柱体内部。"""
        x, y, z = point
        # 径向检查: x² + y² <= R²
        radial_check = (x*x + y*y) <= self.R_sq
        # 轴向检查: 0 <= z <= H
        axial_check = (z >= 0.0) and (z <= self.H)
        
        return radial_check and axial_check

    def get_distance_to_boundary(self, position: np.ndarray, direction: np.ndarray) -> float:
        """
        计算从当前位置沿方向到最近几何边界的距离 (m)。
        (射线-圆柱体求交)
        """
        x0, y0, z0 = position
        dx, dy, dz = direction
        
        t_hits = []
        epsilon = 1e-9

        # --- A. 弯曲侧面 (Curved Surface) ---
        
        # Ax t² + Bx t + C = 0
        A = dx*dx + dy*dy
        B = 2.0 * (x0*dx + y0*dy)
        C = x0*x0 + y0*y0 - self.R_sq
        
        if A > epsilon:
            discriminant = B*B - 4*A*C
            if discriminant >= 0.0:
                # 存在实数解
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-B - sqrt_disc) / (2.0 * A)
                t2 = (-B + sqrt_disc) / (2.0 * A)
                
                for t in [t1, t2]:
                    if t > epsilon:
                        # 检查击中点是否在圆柱高度范围内 (0 <= z <= H)
                        z_hit = z0 + t * dz
                        if z_hit >= -epsilon and z_hit <= self.H + epsilon:
                            t_hits.append(t)

        # --- B. 顶部和底部平面 (Endcaps) ---
        
        if np.abs(dz) > epsilon:
            # 底部平面 (z=0)
            t_base = -z0 / dz
            # 顶部平面 (z=H)
            t_top = (self.H - z0) / dz
            
            for t, plane_z in [(t_base, 0.0), (t_top, self.H)]:
                if t > epsilon:
                    # 检查击中点是否在圆盘半径内 (x² + y² <= R²)
                    x_hit = x0 + t * dx
                    y_hit = y0 + t * dy
                    if (x_hit*x_hit + y_hit*y_hit) <= self.R_sq + epsilon:
                        t_hits.append(t)

        if not t_hits:
            return 1e12 # 没有有效击中
            
        return np.min(t_hits)

    def get_boundary_normal(self, hit_point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """计算击中点处的边界法向量。"""
        x, y, z = hit_point
        epsilon = 1e-6
        
        # 1. 底部和顶部 (Planar faces)
        if np.abs(z - 0.0) < epsilon:
            # 底部 (z=0)，法线指向 -Z
            return np.array([0.0, 0.0, -1.0])
        if np.abs(z - self.H) < epsilon:
            # 顶部 (z=H)，法线指向 +Z
            return np.array([0.0, 0.0, 1.0])

        # 2. 弯曲侧面 (Curved side)
        # 法线是径向向量 (x/R, y/R, 0)
        # 确保不会除以零，虽然理论上击中点在 x² + y² = R² 上
        norm = np.sqrt(x*x + y*y)
        if norm < epsilon:
             # 在中轴线处击中侧面 (理论上不太可能，除非 R=0)
             # 返回指向远离入射方向的任意法线
             return -direction
        
        return np.array([x/norm, y/norm, 0.0])


class LeadShieldingGeometry:
    """
    铅屏蔽层：外部为边长 40cm 的立方体，中心挖去直径 20cm, 高 10cm 的圆柱。
    """
    def __init__(self, scint_geom: AnalyticCylinderGeometry):
        self.scint = scint_geom
        # 立方体边界 (m)
        self.bounds_min = np.array([-0.20, -0.20, -0.15])
        self.bounds_max = np.array([ 0.20,  0.20,  0.25])

    def is_inside_lead(self, point: np.ndarray) -> bool:
        """检查点是否在铅层实体内部"""
        in_cube = np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max)
        in_hole = self.scint.is_inside(point)
        return in_cube and not in_hole

    def get_distance_to_lead_boundary(self, pos: np.ndarray, direction: np.ndarray) -> float:
        """计算到铅层边界（内外表面）的距离"""
        # 到外部立方体表面的距离 (复用之前的 Box 逻辑)
        t_cube = self._get_box_distance(pos, direction)
        # 到内部圆柱孔表面的距离 (复用圆柱逻辑)
        t_hole = self.scint.get_distance_to_boundary(pos, direction)
        return min(t_cube, t_hole)

    def _get_box_distance(self, pos, direction):
        # 简化的射线-长方体求交逻辑
        inv_d = 1.0 / (direction + 1e-12)
        t1 = (self.bounds_min - pos) * inv_d
        t2 = (self.bounds_max - pos) * inv_d
        t_exit = np.min(np.maximum(t1, t2))
        return t_exit if t_exit > 0 else 1e12
    
# ==============================================================================
# III. 输入文件加载函数
# ==============================================================================

def load_cross_section_data(h_file: str, c_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载 H 和 C 的微观截面数据 [Energy (MeV), Sigma_Micro (barn)]。"""
    try:
        c_data = pd.read_csv(c_file, sep=';', skiprows=3, header=None).values
        h_data = pd.read_csv(h_file, sep=';', skiprows=3, header=None).values
        # ------------------------
        # 确保数据格式为 [E, Sigma]
        if h_data.shape[1] < 2 or c_data.shape[1] < 2:
             raise ValueError("CSV 文件格式错误，需要至少两列 (能量和截面)。")
        print(f"✅ 成功加载 H ({len(h_data)}点) 和 C ({len(c_data)}点) 截面数据。")
        return h_data[:, :2], c_data[:, :2]
    except Exception as e:
        raise IOError(f"加载截面文件失败: {e}")

def load_neutron_initial_data(file_path: str) -> pd.DataFrame:
    """加载中子初始状态数据。"""
    try:
        df = pd.read_csv(file_path)
        # 假设 CSV 包含 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', 'energy_mev', 'time_s'
        required_cols = ['detector_hit_x_m', 'detector_hit_y_m', 'detector_hit_z_m', 
                         'direction_x', 'direction_y', 'direction_z', 'final_energy_MeV', 'time_s']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 缺少必需的列: {required_cols}")
        print(f"✅ 成功加载 {len(df)} 个中子的初始数据。")
        return df
    except Exception as e:
        raise IOError(f"加载中子初始数据文件失败: {e}")

# ==============================================================================
# IV. 物理与截面计算函数
# ==============================================================================

def energy_to_speed(energy_mev: float) -> float:
    """中子能量 (MeV) 转换为速度 (m/s)。"""
    energy_joules = energy_mev * J_PER_MEV
    if energy_joules <= 0:
        return 0.0
    return np.sqrt(2 * energy_joules / M_N)


def calculate_eb_macro_sigma(
    h_micro_data: np.ndarray,
    c_micro_data: np.ndarray,
    density_g_cm3: float = 0.867, # 乙苯 C₈H₁₀ 密度 (g/cm³)
) -> np.ndarray:
    """计算乙苯 (C₈H₁₀) 的宏观截面 [Energy (MeV), Sigma_Macro (m⁻¹)]。"""
    
    M_C, M_H = 12.011, 1.008  # g/mol
    M_EB = 8 * M_C + 10 * M_H
    
    # 原子核数密度 N_i (m⁻³)
    N_C_m3 = (density_g_cm3 * AVOGADRO_CONSTANT * 8 / M_EB) * 1e6
    N_H_m3 = (density_g_cm3 * AVOGADRO_CONSTANT * 10 / M_EB) * 1e6

    # 统一能量网格并插值微观截面
    all_energies = np.unique(np.concatenate([h_micro_data[:, 0], c_micro_data[:, 0]]))
    sigma_h_interp_barn = np.interp(all_energies, h_micro_data[:, 0], h_micro_data[:, 1], left=0, right=0)
    sigma_c_interp_barn = np.interp(all_energies, c_micro_data[:, 0], c_micro_data[:, 1], left=0, right=0)
    
    # 计算宏观截面 Sigma_Macro (m⁻¹)
    sigma_eb_total_m1 = (
        N_C_m3 * sigma_c_interp_barn * BARN_TO_M2 +
        N_H_m3 * sigma_h_interp_barn * BARN_TO_M2
    )
    
    return np.stack([all_energies, sigma_eb_total_m1], axis=1)


def calculate_pb_macro_sigma(pb_micro_data: np.ndarray, density_g_cm3: float = 11.34) -> np.ndarray:
    """计算铅(Pb)的宏观截面。铅的摩尔质量约为 207.2 g/mol。"""
    M_Pb = 207.2
    # 原子核数密度 N_Pb (m⁻³)
    N_Pb_m3 = (density_g_cm3 * AVOGADRO_CONSTANT / M_Pb) * 1e6
    # Sigma_macro = N * sigma_micro * 10^-28
    sigma_pb_m1 = N_Pb_m3 * pb_micro_data[:, 1] * BARN_TO_M2
    return np.stack([pb_micro_data[:, 0], sigma_pb_m1], axis=1)


def get_mfp_energy_dependent(
    energy_mev: float,
    macro_sigma_data: np.ndarray,
) -> float:
    """
    根据中子能量计算平均自由程 (MFP)，使用线性插值。
    """
    if energy_mev <= 0.1:
        return 1e12 

    energies = macro_sigma_data[:, 0]
    sigmas = macro_sigma_data[:, 1]
    
    # 线性插值计算宏观截面 sigma (m⁻¹)
    sigma = np.interp(energy_mev, energies, sigmas, left=sigmas[0], right=sigmas[-1])

    if sigma <= 1e-12:
        return 1e12 # 如果宏观截面为零，MFP 视为无限大
        
    # 返回 MFP (m)
    return 1.0 / sigma

# ==============================================================================
# V. 核心模拟逻辑与光子输运
# ==============================================================================

class Receiver:
    """
    光子接收器模型：四个边长为 S 的正方形，距离闪烁体中心 D 放置在轴向平面上。
    所有单位均为米 (m)。
    """
    
    def __init__(self, scint_center: np.ndarray):
        # 几何常数 (m)
        self.S = 0.100       # 边长 S = 100 mm -> 0.100 m
        D = np.sqrt(0.0075)  # 距离 D ≈ 86.6 mm -> 0.0866 m
        self.half_S = self.S / 2.0
        
        # 闪烁体中心 (作为参考点)
        C_x, C_y, C_z = scint_center
        
        # 定义四个接收器中心 P_i 和它们所在平面的法向量 N_i
        self.detectors: List[Tuple[np.ndarray, np.ndarray, str]] = [
            # 1. -X 方向: 法线 -X. Square in YZ plane.
            (np.array([C_x - D, C_y, C_z]), np.array([-1.0, 0.0, 0.0]), "-X"),
            # 2. +X 方向: 法线 +X. Square in YZ plane.
            (np.array([C_x + D, C_y, C_z]), np.array([ 1.0, 0.0, 0.0]), "+X"),
            # 3. -Y 方向: 法线 -Y. Square in XZ plane.
            (np.array([C_x, C_y - D, C_z]), np.array([ 0.0, -1.0, 0.0]), "-Y"),
            # 4. +Y 方向: 法线 +Y. Square in XZ plane.
            (np.array([C_x, C_y + D, C_z]), np.array([ 0.0,  1.0, 0.0]), "+Y"),
        ]

    def check_absorption(self, hit_point: np.ndarray, boundary_normal: np.ndarray) -> bool:
        """检查光子是否被任何一个接收器正方形区域吸收。"""
        
        for center, normal, _ in self.detectors:
            
            diff = hit_point - center
            half_S = self.half_S
            
            # The square is defined by two dimensions perpendicular to the normal.
            
            # 1. X-axis detectors (Normal: [±1, 0, 0]) -> Check Y and Z
            if np.abs(normal[0]) > 0.9: 
                # Check Y axis: |Y_diff| <= S/2 and Z axis: |Z_diff| <= S/2
                if np.abs(diff[1]) <= half_S and np.abs(diff[2]) <= half_S:
                    return True
            
            # 2. Y-axis detectors (Normal: [0, ±1, 0]) -> Check X and Z
            elif np.abs(normal[1]) > 0.9: 
                # Check X axis: |X_diff| <= S/2 and Z axis: |Z_diff| <= S/2
                if np.abs(diff[0]) <= half_S and np.abs(diff[2]) <= half_S:
                    return True
                
        return False

def sample_isotropic_direction() -> np.ndarray:
    """在 3D 空间中均匀采样一个随机单位方向向量。"""
    # 使用球坐标随机采样
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = costheta
    return np.array([x, y, z])



def simulate_full_transport(
    initial_pos: np.ndarray,
    initial_dir: np.ndarray,
    initial_energy_mev: float,
    start_time: float,
    scint_geometry: 'AnalyticCylinderGeometry',
    lead_geometry: 'LeadShieldingGeometry',
    eb_macro_sigma: np.ndarray,
    pb_macro_sigma: np.ndarray,
    light_yield_per_mev: float,
    receiver: 'Receiver',
    tau_r_ns: float = 0.5,        # 闪烁上升时间 (ns)
    tau_d_ns: float = 2.1,        # 闪烁衰减时间 (ns)
    energy_cutoff_mev: float = 0.1,
    photon_reflection_prob: float = 0.95,
    pb_absorption_prob: float = 0.05,
    photon_attenuation_length: float = 1.0, # 假设衰减长度为 1.0 米 (100cm)
) -> Dict[float, int]:
    """
    全过程模拟：追踪中子在闪烁体和铅层中的运动，并基于双指数分布产生光子。
    """
    # 单位转换：ns -> s
    tau_r = tau_r_ns * 1e-9
    tau_d = tau_d_ns * 1e-9
    v_light = 2.0e8  # 闪烁体中的光速 (m/s)

    # 初始化中子状态
    n_pos = np.array(initial_pos, dtype=float)
    n_dir = np.array(initial_dir, dtype=float) / np.linalg.norm(initial_dir)
    n_energy = initial_energy_mev
    n_time = start_time
    
    received_photons_by_time: Dict[float, int] = {}
    photons_to_track: List[Tuple[np.ndarray, np.ndarray, float]] = []
    
    # --- 1. 中子输运主循环 ---
    while n_energy > energy_cutoff_mev:
        # 1.1 环境判定
        in_scint = scint_geometry.is_inside(n_pos)
        in_lead = lead_geometry.is_inside_lead(n_pos)
        
        if in_scint:
            current_sigma_data = eb_macro_sigma
            medium_type = "SCINT"
        elif in_lead:
            current_sigma_data = pb_macro_sigma
            medium_type = "LEAD"
        else:
            # 真空区域处理
            d_to_scint = scint_geometry.get_distance_to_boundary(n_pos, n_dir)
            d_to_lead = lead_geometry.get_distance_to_lead_boundary(n_pos, n_dir)
            d_vacuum = min(d_to_scint, d_to_lead)
            
            if d_vacuum > 2.0: # 逃逸判定
                break
                
            v_n = energy_to_speed(n_energy)
            n_pos += n_dir * (d_vacuum + 1e-7)
            n_time += d_vacuum / v_n
            continue

        # 1.2 采样碰撞
        mfp = get_mfp_energy_dependent(n_energy, current_sigma_data)
        d_coll = -mfp * np.log(np.random.rand() + 1e-12)
        
        # 计算到当前介质边界距离
        if medium_type == "SCINT":
            d_bound = scint_geometry.get_distance_to_boundary(n_pos, n_dir)
        else:
            d_bound = lead_geometry.get_distance_to_lead_boundary(n_pos, n_dir)
            
        if d_coll < d_bound:
            # 发生碰撞事件
            v_n = energy_to_speed(n_energy)
            n_pos += n_dir * d_coll
            n_time += d_coll / v_n
            
            if medium_type == "SCINT":
                # --- 闪烁体物理逻辑 ---
                energy_loss_ratio = np.random.uniform(0.1, 0.5) 
                dep_energy = n_energy * energy_loss_ratio
                n_energy -= dep_energy
                
                # 基于双指数分布产生光子
                num_photons = int(dep_energy * light_yield_per_mev)
                for _ in range(num_photons):
                    # 抽样时间延迟：t_delay = t_rise + t_decay (卷积效应)
                    t_rise_comp = -tau_r * np.log(np.random.rand() + 1e-12)
                    t_decay_comp = -tau_d * np.log(np.random.rand() + 1e-12)
                    t_delay = t_rise_comp + t_decay_comp
                    
                    photon_emission_time = n_time + t_delay
                    photons_to_track.append((np.copy(n_pos), sample_isotropic_direction(), photon_emission_time))
                
                n_dir = sample_isotropic_direction()
            else:
                # --- 铅屏蔽层逻辑 ---
                if np.random.rand() < pb_absorption_prob:
                    break # 中子被铅吸收
                else:
                    n_dir = sample_isotropic_direction() # 弹性散射，能量不损失
        else:
            # 穿过介质边界
            v_n = energy_to_speed(n_energy)
            n_pos += n_dir * (d_bound + 1e-7)
            n_time += d_bound / v_n

    # --- 2. 光子输运循环 (射线追踪) ---
    for p_pos, p_dir, p_time in photons_to_track:
        curr_p_pos, curr_p_dir, curr_p_time = p_pos, p_dir, p_time
        
        # --- 核心修改：抽样光子的最大飞行距离 (e指数分布) ---
        # 这里的 photon_attenuation_length 决定了光子在被吸收前能跑多远
        max_flight_distance = -photon_attenuation_length * np.log(np.random.rand() + 1e-12)
        accumulated_distance = 0.0

        # 限制循环次数以防万一（防止在极端情况下陷入无限反射），但判断逻辑改为距离
        for _ in range(200): 
            d_to_bound = scint_geometry.get_distance_to_boundary(curr_p_pos, curr_p_dir)
            
            # 检查：如果飞行到下一个边界的距离超过了光子的剩余寿命
            if accumulated_distance + d_to_bound > max_flight_distance:
                # 光子在到达边界前就在介质中被吸收了
                # 即使没被吸收完，如果距离异常大（如 1e12），这里也会截断
                break

            # 光子飞行到边界
            curr_p_pos += curr_p_dir * d_to_bound
            curr_p_time += d_to_bound / v_light
            accumulated_distance += d_to_bound
            
            # 判定边界相互作用
            if not scint_geometry.is_inside(curr_p_pos):
                normal = scint_geometry.get_boundary_normal(curr_p_pos, curr_p_dir)
                
                # 接收器吸收检查
                if receiver.check_absorption(curr_p_pos, normal):
                    t_bin = round(curr_p_time, 10)
                    received_photons_by_time[t_bin] = received_photons_by_time.get(t_bin, 0) + 1
                    break
                
                # 边界反射
                if np.random.rand() < photon_reflection_prob:
                    curr_p_dir = curr_p_dir - 2 * np.dot(curr_p_dir, normal) * normal
                    curr_p_pos += curr_p_dir * 1e-7 # 避开边界
                else:
                    # 未发生反射（被边界吸收或透射）
                    break
            else:
                # 浮点误差处理
                curr_p_pos += curr_p_dir * 1e-7
                
    return received_photons_by_time

# ==============================================================================
# VI. 绘图函数
# ==============================================================================

def plot_photon_time_histogram(
    final_output: Dict[float, int],
    num_neutrons: int,  # 传入中子个数用于命名
    base_filename: str = "hist",
    dpi: int = 300
) -> None:
    """绘制光子到达时间直方图，使用对数Y轴并自动处理文件名冲突。"""
    
    if not final_output:
        print("Warning: No photons received, cannot draw histogram.")
        return

    # --- 1. 处理文件名逻辑 ---
    # 初始目标文件名: hist_1000.png
    filename = f"{base_filename}_{num_neutrons}.png"
    
    # 检查重名并增加后缀 (如 hist_1000_01.png, hist_1000_02.png)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{num_neutrons}_{counter:02d}.png"
        counter += 1

    # --- 2. 准备数据 ---
    photon_arrival_times = []
    for arrival_time, count in final_output.items():
        photon_arrival_times.extend([arrival_time] * count)
    
    time_data_ns = np.array(photon_arrival_times) * 1e9
    
    # 防止因极个别离群值（如 1e12）导致内存溢出，限制时间显示范围
    # 建议只显示前 500ns 的数据，或者根据实际情况调整
    p99 = np.percentile(time_data_ns, 99.9)
    filtered_data = time_data_ns[time_data_ns < p99 + 10]

    bin_width_ns = 1.0 
    time_range_ns = filtered_data.max() - filtered_data.min()
    # 限制最大 bin 数量为 1000，防止 MemoryError
    num_bins = min(1000, max(50, int(time_range_ns / bin_width_ns))) if time_range_ns > 0 else 50

    # --- 3. 绘图 ---
    plt.figure(figsize=(12, 6))
    
    plt.hist(
        filtered_data,
        bins=num_bins,
        color='skyblue',
        edgecolor='black',
        linewidth=0.5,
        log=True  # 也可以在 hist 内部直接设置对数
    )
    
    # 明确设置 Y 轴为对数坐标
    plt.yscale('log')
    
    plt.title(f"Photon Arrival Time Distribution (Neutrons: {num_neutrons})", fontsize=16)
    plt.xlabel("Arrival Time (ns)", fontsize=14)
    plt.ylabel(f"Photons per Bin (Log Scale, Width: {bin_width_ns:.2f} ns)", fontsize=14)
    
    plt.grid(axis='y', linestyle='--', alpha=0.4, which='both') # which='both' 显示对数细分网格
    plt.ticklabel_format(axis='x', style='plain')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close() # 释放内存
    print(f"✅ Histogram saved as: {filename}")

# ==============================================================================
# VII. 主程序执行
# ==============================================================================



def generate_pb_micro_data():
    """生成铅(Pb)的中子微观总截面数据 [MeV, barn]"""
    # 能量点从 0.1 MeV 到 20 MeV
    energies = np.linspace(0.1, 20.0, 100)
    
    # 铅的总截面简化物理模型: 
    # 基准约 5.4 barn，在低能区随能量指数增加
    # 公式: sigma(E) = 5.4 + 5.8 * exp(-E / 0.8) + 波动项
    sigmas = 5.4 + 5.8 * np.exp(-energies / 0.8) + 0.1 * np.cos(energies * 0.5)
    
    return np.stack([energies, sigmas], axis=1)

# 在 main 中使用
pb_micro_data = generate_pb_micro_data()

def main_simulation(
    neutron_file: str = "neutron_data_1000.csv",
    h_file: str = "H.csv",
    c_file: str = "C.csv",
    light_yield: float = 1000.0,
):
    """主程序：加载数据，运行模拟，并显示进度条。"""
    
    # 记录模拟开始的时间
    start_wall_time = time.time()

    # --- 1. 加载所有输入数据 ---
    try:
        neutron_data = load_neutron_initial_data(neutron_file)
        h_micro, c_micro = load_cross_section_data(h_file, c_file)
    except Exception as e:
        print(f"Fatal Error: File loading failed: {e}")
        return

    # --- 2. 预处理 ---
    print("--- Pre-processing ---")
    scint_center_m = np.array([0.0, 0.0, 0.05])
    eb_sigma_data = calculate_eb_macro_sigma(h_micro, c_micro)
    pb_sigma_data = calculate_pb_macro_sigma(pb_micro_data)
    scint_geometry = AnalyticCylinderGeometry()
    lead_geometry = LeadShieldingGeometry(scint_geometry)
    receiver = Receiver(scint_center=scint_center_m)
    print("✅ Geometry and cross-sections initialized.")

    # --- 3. 运行蒙特卡洛模拟 (添加进度条) ---
    num_neutrons = len(neutron_data)
    print(f"--- Running Simulation for {num_neutrons} Neutrons ---")
    
    aggregated_photons: Dict[float, int] = {}
    
    # 使用 tqdm 包裹迭代器，desc 设置进度条前的描述文字
    for _, row in tqdm(neutron_data.iterrows(), total=num_neutrons, desc="Simulating Neutrons"):
        initial_pos = np.array([row['detector_hit_x_m'], row['detector_hit_y_m'], row['detector_hit_z_m']])
        initial_dir = np.array([row['direction_x'], row['direction_y'], row['direction_z']])
        initial_e = row['final_energy_MeV']
        initial_time = row.get('total_flight_time_s', 0.0)
        
        initial_dir = initial_dir / np.linalg.norm(initial_dir)
        
        # 运行单个中子模拟
        photon_output = simulate_full_transport(
            initial_pos, initial_dir, initial_e, initial_time,
            scint_geometry, lead_geometry, eb_sigma_data, pb_sigma_data, 
            light_yield, receiver
        )
        
        for t, count in photon_output.items():
            aggregated_photons[t] = aggregated_photons.get(t, 0) + count

    # 计算总耗时
    end_wall_time = time.time()
    total_duration = end_wall_time - start_wall_time

    print(f"\n--- Simulation Completed ---")
    print(f"Total Photons Received: {sum(aggregated_photons.values())}")
    print(f"Total Time Spent: {total_duration:.2f} seconds")  # 打印最终花费的时间

    # --- 4. 绘图 ---
    num_neutrons = len(neutron_data)
    plot_photon_time_histogram(aggregated_photons, num_neutrons)


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # ⚠️ 启动前请确保以下文件存在于脚本目录中：
    # 1. initial_neutrons.csv (包含 pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_mev)
    # 2. R100xH100_mesh.stl (STL 文件)
    # 3. H.csv, C.csv (微观截面数据)
    # -------------------------------------------------------------------------
    
    # 示例运行参数（请根据您的实际需求修改）
    main_simulation()