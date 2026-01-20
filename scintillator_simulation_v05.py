'''Created by Shen on 2025.12.10
   to simulate neutron after it enter scintillator'''

# v02 删除所有读取的stl文件，改为硬编码闪烁体和接收器，增加与铅层相互作用，简单假设：中子在铅中要么完全不损失能量，要么被吸收
# v03 修改产生中子的逻辑，按照双指数衰减公式;增加进度条和时间,修改光子到达时间直方图y轴为对数坐标
# v04 修改光子最大路程为e指数衰减分布
# v05 改绘图逻辑，现在有一张线性小图，反转y轴

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import os
import struct
import time  # 引入时间库
from tqdm import tqdm  # 引入进度条库
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    def __init__(self, R=0.1, H=0.1, S=0.1):
        """
        R: 原圆柱半径 (m)
        H: 圆柱高度 (m)
        S: 接收器界面的边长 (m)，用于计算切面深度
        """
        self.R = R
        self.H = H
        self.S = S
        
        # 计算圆心到切平面的距离 d
        # 使得切面在圆柱侧面形成的弦长刚好为 S
        self.d = np.sqrt(self.R**2 - (self.S/2)**2)
        
        # 定义 4 个切面的法线和位置
        # 这 4 个面现在是闪烁体边界的一部分
        self.faces = [
            {'norm': np.array([1, 0, 0]),  'val': self.d,  'axis': 0}, # x = d
            {'norm': np.array([-1, 0, 0]), 'val': -self.d, 'axis': 0}, # x = -d
            {'norm': np.array([0, 1, 0]),  'val': self.d,  'axis': 1}, # y = d
            {'norm': np.array([0, -1, 0]), 'val': -self.d, 'axis': 1}  # y = -d
        ]

    def is_inside(self, pos):
        """判定点是否在修改后的闪烁体内部（圆柱体减去4个切面外的区域）"""
        eps = 1e-9
        # 1. 检查高度范围
        if not (-eps <= pos[2] <= self.H + eps):
            return False
        # 2. 检查基础圆柱径向范围
        if (pos[0]**2 + pos[1]**2) > self.R**2 + eps:
            return False
        # 3. 检查 4 个平面的边界限制 (必须在所有平面定义的内部)
        if pos[0] > self.d + eps or pos[0] < -self.d - eps:
            return False
        if pos[1] > self.d + eps or pos[1] < -self.d - eps:
            return False
        return True

    def get_distance_to_boundary(self, pos, direction):
        """计算从当前点沿方向到最近边界（含切面）的距离"""
        distances = []
        eps = 1e-10
        
        # --- A. 上下底面 (z=0, z=H) ---
        if abs(direction[2]) > eps:
            t0 = (0 - pos[2]) / direction[2]
            if t0 > eps: distances.append(t0)
            tH = (self.H - pos[2]) / direction[2]
            if tH > eps: distances.append(tH)
            
        # --- B. 圆柱曲面 (x^2 + y^2 = R^2) ---
        a = direction[0]**2 + direction[1]**2
        if a > eps:
            b = 2 * (pos[0]*direction[0] + pos[1]*direction[1])
            c = pos[0]**2 + pos[1]**2 - self.R**2
            delta = b**2 - 4*a*c
            if delta >= 0:
                sq_delta = np.sqrt(delta)
                for t in [(-b - sq_delta)/(2*a), (-b + sq_delta)/(2*a)]:
                    if t > eps: distances.append(t)
                    
        # --- C. 4 个切面 (x=±d, y=±d) ---
        for face in self.faces:
            if abs(direction[face['axis']]) > eps:
                t = (face['val'] - pos[face['axis']]) / direction[face['axis']]
                if t > eps: distances.append(t)
        
        # 筛选合法交点：交点必须满足几何体的所有约束
        valid_hits = []
        for t in distances:
            hit_pos = pos + t * direction
            # 必须满足所有边界条件 (带容差)
            if (-eps <= hit_pos[2] <= self.H + eps and 
                hit_pos[0]**2 + hit_pos[1]**2 <= self.R**2 + eps and
                -self.d - eps <= hit_pos[0] <= self.d + eps and
                -self.d - eps <= hit_pos[1] <= self.d + eps):
                valid_hits.append(t)
        
        return min(valid_hits) if valid_hits else 1e10

    def get_boundary_normal(self, pos, direction):
        """获取碰撞点的法向量"""
        eps = 1e-6
        # 1. 检查底面和顶面
        if abs(pos[2] - 0) < eps: return np.array([0, 0, -1])
        if abs(pos[2] - self.H) < eps: return np.array([0, 0, 1])
        
        # 2. 检查 4 个切面 (这些面现在是优先判定的平面边界)
        for face in self.faces:
            if abs(pos[face['axis']] - face['val']) < eps:
                return face['norm']
                
        # 3. 否则为圆柱曲面
        norm = np.array([pos[0], pos[1], 0.0])
        norm /= (np.linalg.norm(norm) + 1e-12)
        return norm


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
    def __init__(self, scint_geo):
        """
        接收器初始化
        """
        self.r_disk = 0.00455  # 接收器半径 4.55 mm
        self.scint_geo = scint_geo
        self.detectors = []
        
        # 自动根据闪烁体的 4 个切面计算中心点
        for face in scint_geo.faces:
            center = face['norm'] * scint_geo.d 
            center[2] = scint_geo.H / 2.0  # 设在高度中点 0.05m
            
            self.detectors.append({
                'center': center, 
                'norm': face['norm']
            })

    def check_absorption(self, pos, norm):
        """
        检查光子是否被圆形接收器吸收
        pos: 光子碰撞位置
        norm: 碰撞面的法向量
        """
        for det in self.detectors:
            # 1. 判定法线方向是否一致 (夹角余弦 > 0.9)
            # 只有撞击到对应的切面，才可能被该面上的接收器吸收
            if np.dot(norm, det['norm']) > 0.9:
                # 2. 计算碰撞点到接收器中心点的距离
                dist = np.linalg.norm(pos - det['center'])
                # 3. 如果距离小于接收器半径，则判定为吸收
                if dist <= self.r_disk:
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
            
        collision_count = 0    
        if d_coll < d_bound:
            collision_count += 1
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
    # --- 修复版光子输运循环 ---
    for p_pos, p_dir, p_time in photons_to_track:
        curr_p_pos, curr_p_dir, curr_p_time = p_pos, p_dir, p_time
        max_flight_distance = -photon_attenuation_length * np.log(np.random.rand() + 1e-12)
        accumulated_distance = 0.0

        for _ in range(300): 
            d_to_bound = scint_geometry.get_distance_to_boundary(curr_p_pos, curr_p_dir)
            
            # --- 关键修复 1：防止 1e10 导致的误删 ---
            if d_to_bound > 2.0: # 如果找不到边界，说明光子可能已经逃逸或出错
                break

            # --- 关键修复 2：检查寿命 ---
            if accumulated_distance + d_to_bound > max_flight_distance:
                break

            # 移动到边界
            curr_p_pos += curr_p_dir * d_to_bound
            curr_p_time += d_to_bound / v_light
            accumulated_distance += d_to_bound
            
            # 获取法线并进行判定
            # 注意：即便 is_inside 返回 True，由于 eps 存在，我们也认为在边界上
            normal = scint_geometry.get_boundary_normal(curr_p_pos, curr_p_dir)
            
            # --- 关键修复 3：放宽接收器判定 ---
            # 建议暂时跳过 check_absorption，直接判定只要撞到侧面就接收，用于排查
            if receiver.check_absorption(curr_p_pos, normal):
                t_bin = round(curr_p_time, 10)
                received_photons_by_time[t_bin] = received_photons_by_time.get(t_bin, 0) + 1
                break
            
            # 反射逻辑
            if np.random.rand() < photon_reflection_prob:
                # 镜面反射
                curr_p_dir = curr_p_dir - 2 * np.dot(curr_p_dir, normal) * normal
                curr_p_dir /= np.linalg.norm(curr_p_dir)
                curr_p_pos += curr_p_dir * 1e-6 # 强制向内部推行，避免卡在面外
            else:
                break

    #print(f"This neutron collided {collision_count} times in Scintillator")
    return received_photons_by_time

# ==============================================================================
# VI. 绘图函数
# ==============================================================================

def plot_photon_time_histogram_v2(
    final_output: dict,
    num_neutrons: int,
    base_filename: str = "hist",
    dpi: int = 300
) -> None:
    """
    绘制专业版光子到达时间分布图：
    1. Y轴反转并使用对数坐标 (主图)
    2. 坐标单位为 a.u. (Arbitrary Unit)
    3. 右下角包含线性坐标的小图
    4. 自动处理文件名冲突
    """
    
    if not final_output:
        print("Warning: No photons received, cannot draw histogram.")
        return

    # --- 1. 处理文件名逻辑 (保留并优化) ---
    filename = f"{base_filename}_{num_neutrons}.png"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{num_neutrons}_{counter:02d}.png"
        counter += 1

    data_filename = filename.replace(f"{base_filename}_", "").replace(".png", ".txt")

    # --- 2. 准备数据 ---
    photon_arrival_times = []
    for arrival_time, count in final_output.items():
        photon_arrival_times.extend([arrival_time] * count)
    
    try:
        # np.savetxt 可以高效地将单列数据保存为文本
        np.savetxt(data_filename, photon_arrival_times, fmt='%.10e', header='Arrival_Time_s')
        print(f"📄 Data saved as: {data_filename}")
    except Exception as e:
        print(f"Error saving data file: {e}")

    time_data_ns = np.array(photon_arrival_times) * 1e9
    
    # 过滤异常值，保留主要波形区
    p99 = np.percentile(time_data_ns, 99.5)
    filtered_data = time_data_ns[time_data_ns < p99 + 20]
    
    # 预计算直方图数据用于手动绘图
    bin_width = 1.0
    bins = np.arange(filtered_data.min(), filtered_data.max() + bin_width, bin_width)
    counts, bin_edges = np.histogram(filtered_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 3. 绘图逻辑 ---
    plt.rcParams['font.weight'] = 'bold'
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # A. 主图绘制 (对数 Y 轴)
    
    # 使用 step 阶梯图模拟多道分析仪效果
    ax1.step(bin_centers, counts, where='mid', color='dodgerblue', lw=1.5, label='Log Distribution')
    ax1.fill_between(bin_centers, counts, step="mid", color='skyblue', alpha=0.3)
    
    ax1.set_yscale('log')
    # 反转 Y 轴：设定 ylim，将最大值放在下方，最小值（接近1）放在上方
    # 注意：对数坐标不能设为0，这里设为 0.8 以便显示计数为1的柱子
    ax1.set_ylim(np.max(counts) * 2, 0.8) 
    
    ax1.set_title(f"nToF Spectrum (T=5keV, Neutrons=$10^8$)", fontsize=22, fontweight='bold', pad=20)
    ax1.set_xlabel("Time (ns)", fontsize=18, fontweight='bold')
    ax1.set_ylabel("Intensity (a.u.)", fontsize=18, fontweight='bold')

    ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7)
    ax1.tick_params(axis='both', which='minor', width=1.5, length=4)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')

    ax1.grid(True, which='both', linestyle='--', alpha=0.5, lw=1.2)

    # B. 右下角添加嵌入小图 (线性坐标)
    # loc=4 表示 lower right
    ax_ins = inset_axes(ax1, width="40%", height="40%", loc=4, bbox_to_anchor=(-0.02, 0.06, 1, 1), # 这里的 0.05 就是向上偏移量
                    bbox_transform=ax1.transAxes,
                    borderpad=0)
    
    ax_ins.step(bin_centers, counts, where='mid', color='crimson', lw=2)
    ax_ins.fill_between(bin_centers, counts, step="mid", color='orange', alpha=0.3)
    
    # 小图设置：仅显示前 50ns 左右的特征峰
    ax_ins.set_xlim(filtered_data.min(), filtered_data.min() + 50)
    # 小图也反转 Y 轴（线性）
    ax_ins.set_ylim(np.max(counts) * 1.1, 0) 
    
    ax_ins.set_title("Linear Scale Detail", fontsize=14, fontweight='bold', color='darkred')
    #ax_ins.set_xlabel("ns", fontsize=12, fontweight='bold')

    ax_ins.grid(True, which='major', linestyle=':', alpha=0.8, lw=1.2)
    
    ax_ins.tick_params(axis='both', labelsize=10, width=1.5)
    for label in ax_ins.get_xticklabels() + ax_ins.get_yticklabels():
        label.set_fontweight('bold')

    # --- 4. 保存与退出 ---
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, transparent=True)
    plt.show()
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
    neutron_file: str = "neutron_data_32800.csv",
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
    eb_sigma_data = calculate_eb_macro_sigma(h_micro, c_micro)
    pb_sigma_data = calculate_pb_macro_sigma(pb_micro_data)
    scint_geometry = AnalyticCylinderGeometry()
    lead_geometry = LeadShieldingGeometry(scint_geometry)
    receiver = Receiver(scint_geometry)
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
        
        # 在调用 simulate_full_transport 之前
        #print(f"\n--- DEBUG NEUTRON ---")
        #print(f"Initial Pos: {initial_pos}")
        #print(f"In Scint: {scint_geometry.is_inside(initial_pos)}")
        #print(f"MFP (at {initial_e} MeV): {get_mfp_energy_dependent(initial_e, eb_sigma_data)} m")

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
    plot_photon_time_histogram_v2(aggregated_photons, num_neutrons)


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # ⚠️ 启动前请确保以下文件存在于脚本目录中：
    # 1. initial_neutrons.csv (包含 pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_mev)
    # 2. R100xH100_mesh.stl (STL 文件)
    # 3. H.csv, C.csv (微观截面数据)
    # -------------------------------------------------------------------------
    
    # 示例运行参数（请根据您的实际需求修改）
    main_simulation()