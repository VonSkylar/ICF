'''Created by Shen on 2025.12.24
   to plot neutron and photon tracks after it enter scintillator'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple
import math
import pandas as pd


from scintillator_simulation_v04 import AnalyticCylinderGeometry, LeadShieldingGeometry, Receiver, get_mfp_energy_dependent, sample_isotropic_direction, energy_to_speed

# --- 在函数外部定义物理常数 ---
RHO = 0.867          # 乙苯密度 g/cm3
M_ALCOHOL = 106.16   # 分子量
NA = 0.6022          # 阿伏伽德罗常数 (已缩放，配合 Barn 单位)

# 计算分子密度因子 (atoms/(b*cm))
# 1 Barn = 10^-24 cm^2, 所以 NA * 10^-24 相互抵消简化
N_FACTOR = (RHO * 0.6022) / M_ALCOHOL

RHO_PB = 11.34        # g/cm3
M_PB = 207.2          # g/mol
NA = 0.6022           # 阿伏伽德罗常数 (缩放因子)

# 铅的原子密度因子 (atoms/(b*cm))
N_FACTOR_PB = (RHO_PB * NA) / M_PB  # 约为 0.033

def load_cross_section(filename):
    """
    读取 CSV 文件并生成能量单位为 MeV 的截面数组。
    
    参数:
    filename : str - 文件路径 (H.csv 或 C.csv)
    
    返回:
    np.ndarray - 形状为 (N, 2) 的数组，第一列为 Energy (MeV)，第二列为 Sigma (Barns)
    """
    # 1. 读取文件，使用 ';' 分隔符，并跳过前 3 行表头
    df = pd.read_csv(filename, sep=';', skiprows=3, header=None, names=['Energy', 'Sigma'])
    
    # 2. 将能量单位从 eV 转换为 MeV
    df['Energy'] = df['Energy'] / 1e6
    
    # 3. 转换为 numpy 数组并返回
    return df.values


def scatter_neutron_elastic_cms_to_lab(
    neutron_energy_mev: float,
    incident_direction: np.ndarray,
    target_mass_ratio: float,
) -> Tuple[float, np.ndarray]:
    """Perform elastic neutron scattering with proper CMS to LAB frame conversion.
    
    This function implements the complete two-body elastic scattering kinematics:
    1. Sample scattering angle θ_cm isotropically in the center-of-mass (CMS) frame
    2. Calculate energy loss from θ_cm using the correct kinematic relation
    3. Convert θ_cm to θ_lab using the proper coordinate transformation
    4. Update the neutron direction in the laboratory (LAB) frame
    
    CRITICAL PHYSICS:
    For hydrogen (A=1), the CMS to LAB transformation ensures that:
    - θ_lab ∈ [0, π/2]: neutron can NEVER backscatter from proton
    
    Transformation formula:
        tan(θ_lab) = sin(θ_cm) / (γ + cos(θ_cm))
    where γ = 1/A
    
    Energy relation:
        E_out / E_in = [A² + 1 + 2A·cos(θ_cm)] / (A + 1)²
    
    Parameters
    ----------
    neutron_energy_mev : float
        Incident neutron kinetic energy in MeV.
    incident_direction : np.ndarray, shape (3,)
        Unit vector of neutron velocity before collision (LAB frame).
    target_mass_ratio : float
        Target nucleus mass / neutron mass (A).
    
    Returns
    -------
    tuple : (energy_out, direction_out)
        energy_out : float
            Neutron energy after scattering (MeV)
        direction_out : np.ndarray
            Unit vector of neutron velocity after scattering (LAB frame)
    """
    A = target_mass_ratio
    gamma = 1.0 / A
    
    # Sample scattering angle in CMS (isotropic)
    cos_theta_cm = 2.0 * np.random.rand() - 1.0
    sin_theta_cm = math.sqrt(max(0.0, 1.0 - cos_theta_cm**2))
    
    # Calculate energy after scattering
    numerator = A * A + 1.0 + 2.0 * A * cos_theta_cm
    denominator = (A + 1.0) * (A + 1.0)
    energy_ratio = numerator / denominator
    energy_ratio = max(0.0, min(energy_ratio, 1.0))
    energy_out = neutron_energy_mev * energy_ratio
    
    # Convert θ_cm to θ_lab
    denominator_angle = gamma + cos_theta_cm
    
    if abs(denominator_angle) < 1e-10:
        theta_lab = math.pi / 2.0
    else:
        theta_lab = math.atan2(sin_theta_cm, denominator_angle)
    
    if theta_lab < 0:
        theta_lab += math.pi
    
    # Sample azimuthal angle φ uniformly
    phi = 2.0 * math.pi * np.random.rand()
    
    # Construct new direction in LAB frame
    incident_dir = np.array(incident_direction, dtype=float)
    incident_dir /= np.linalg.norm(incident_dir)
    
    z_axis = incident_dir
    if abs(z_axis[2]) < 0.9:
        x_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    cos_theta_lab = math.cos(theta_lab)
    sin_theta_lab = math.sin(theta_lab)
    
    direction_local = np.array([
        sin_theta_lab * math.cos(phi),
        sin_theta_lab * math.sin(phi),
        cos_theta_lab
    ], dtype=float)
    
    direction_out = (
        direction_local[0] * x_axis +
        direction_local[1] * y_axis +
        direction_local[2] * z_axis
    )
    
    direction_out /= np.linalg.norm(direction_out)
    
    return energy_out, direction_out


# --------------------------------------------------------------------------
# 1. 轨迹追踪版模拟函数 (仅追踪一个中子)
# --------------------------------------------------------------------------

def track_real_event(
    initial_pos: np.ndarray,
    initial_dir: np.ndarray,
    initial_energy: float,
    scint_geo,
    lead_geo,
    h_sigma_data: np.ndarray,
    c_sigma_data: np.ndarray,
    pb_sigma_data: np.ndarray,
    receiver,
    light_yield_per_mev: float = 10.0,
    photon_atten_len: float = 0.5
) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[dict]]:
    """
    返回:
        n_path: 中子轨迹点
        p_paths: 光子轨迹点列表
        collision_events: 碰撞事件列表，包含 [{'pos': pos, 'type': 'H'/'C'/'Pb'}, ...]
    """
    n_path = [np.copy(initial_pos)]
    p_paths = []
    collision_events = [] # 新增：记录碰撞点类型
    
    n_pos = np.copy(initial_pos)
    n_dir = initial_dir / np.linalg.norm(initial_dir)
    n_energy = initial_energy
    
    NC, NH = 8, 10
    
    while n_energy > 0.01:
        in_scint = scint_geo.is_inside(n_pos)
        in_lead = lead_geo.is_inside_lead(n_pos)
        
        if in_scint:
            sig_h = np.interp(n_energy, h_sigma_data[:, 0], h_sigma_data[:, 1])
            sig_c = np.interp(n_energy, c_sigma_data[:, 0], c_sigma_data[:, 1])

            # 计算真实的宏观截面 Sigma (单位: cm^-1)
            # NC=8, NH=10
            sigma_macro = N_FACTOR * (sig_h * 10 + sig_c * 8)
            
            # 计算平均自由程 MFP (单位: cm)
            mfp_cm = 1.0 / sigma_macro
            
            # 转换为米 (m)，因为几何定义通常用米
            mfp = mfp_cm / 100.0

            w_h, w_total = sig_h * 10, (sig_h * 10 + sig_c * 8)
            
            if np.random.rand() < (w_h / w_total):
                target_A, medium_type = 1.0, "H"
            else:
                target_A, medium_type = 12.0, "C"
            mfp = 1.0 / (w_total * 10) 
            
        elif in_lead:
            # 1. 插值获取当前能量下的铅微观总截面 (单位: Barn)
            sig_pb = np.interp(n_energy, pb_sigma_data[:, 0], pb_sigma_data[:, 1])
            
            # 2. 计算真实宏观截面 Sigma_Pb (单位: cm^-1)
            # Sigma = n * sigma
            sigma_macro_pb = N_FACTOR_PB * sig_pb
            
            # 3. 计算平均自由程 MFP (单位: cm)
            # 铅在高能区截面约 5-10 Barn，算得 MFP 约为 3-6 cm
            mfp_cm = 1.0 / (sigma_macro_pb + 1e-12)
            
            # 4. 转换为米 (m)，适配 3D 几何坐标系
            mfp = mfp_cm / 100.0
            
            # 5. 设置散射目标属性
            target_A = 207.2
            medium_type = "Pb"
            
        else:
            # 真空/空气区逻辑
            d_s = scint_geo.get_distance_to_boundary(n_pos, n_dir)
            d_l = lead_geo.get_distance_to_lead_boundary(n_pos, n_dir)
            d_v = min(d_s, d_l)
            if d_v > 2.0: break
            n_pos += n_dir * (d_v + 1e-6)
            n_path.append(np.copy(n_pos))
            continue

        # --- 2. 抽样碰撞位置 ---
        d_coll = -mfp * np.log(np.random.rand() + 1e-12)
        d_bound = scint_geo.get_distance_to_boundary(n_pos, n_dir) if "H" in medium_type or "C" in medium_type else \
                  lead_geo.get_distance_to_lead_boundary(n_pos, n_dir)
        
        if d_coll < d_bound:
            # 发生碰撞
            n_pos += n_dir * d_coll
            n_path.append(np.copy(n_pos))
            
            # --- 记录碰撞事件 ---
            collision_events.append({'pos': np.copy(n_pos), 'type': medium_type})

            # --- 3. 弹性散射逻辑 (CMS -> LAB) ---
            # 使用用户提供的 scatter_neutron_elastic_cms_to_lab 函数
            n_energy_new, n_dir_new = scatter_neutron_elastic_cms_to_lab(n_energy, n_dir, target_A)
            
            dep_energy = n_energy - n_energy_new
            n_energy, n_dir = n_energy_new, n_dir_new
            
            # --- 4. 产生光子 (仅限闪烁体内部) ---
            if medium_type in ["H", "C"] and dep_energy > 0:
                # 考虑猝灭效应 (Quenching): 碳核反冲效率通常较低
                q_factor = 1.0 if medium_type == "H" else 0.2
                num_p = int(dep_energy * q_factor * light_yield_per_mev)
                
                for _ in range(min(num_p, 15)): # 可视化限制线段数量
                    p_path = [np.copy(n_pos)]
                    cp_pos, cp_dir = np.copy(n_pos), sample_isotropic_direction()
                    max_p_dist = -photon_atten_len * np.log(np.random.rand() + 1e-12)
                    acc_dist = 0.0
                    
                    for _ in range(20): # 限制反射次数
                        dp_bound = scint_geo.get_distance_to_boundary(cp_pos, cp_dir)
                        if acc_dist + dp_bound > max_p_dist:
                            p_path.append(cp_pos + cp_dir * (max_p_dist - acc_dist))
                            break
                        
                        cp_pos += cp_dir * dp_bound
                        acc_dist += dp_bound
                        p_path.append(np.copy(cp_pos))
                        
                        # 边界反射与接收判定
                        norm = scint_geo.get_boundary_normal(cp_pos, cp_dir)
                        if receiver.check_absorption(cp_pos, norm): break
                        
                        # 反射方向并推回内部
                        cp_dir = cp_dir - 2.0 * np.dot(cp_dir, norm) * norm
                        cp_dir /= np.linalg.norm(cp_dir)
                        cp_pos += cp_dir * 1e-5
                    
                    p_paths.append(p_path)
        else:
            # 越过边界
            n_pos += n_dir * (d_bound + 1e-6)
            n_path.append(np.copy(n_pos))

    return n_path, p_paths, collision_events
# --------------------------------------------------------------------------
# 2. 绘图逻辑
# --------------------------------------------------------------------------
def draw_dynamic_geometry(ax, scint_geo, lead_geo, receiver):
    """根据类定义动态绘制几何体"""
    
    # --- 1. 动态绘制圆柱体 (闪烁体) ---
    R = scint_geo.R
    H = scint_geo.H
    z = np.linspace(0, H, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = R * np.cos(theta_grid)
    y_grid = R * np.sin(theta_grid)
    
    ax.plot_surface(x_grid, y_grid, z_grid, color='cyan', alpha=0.1, shade=False)

    # --- 2. 动态绘制铅屏蔽层 (立方体) ---
    b_min = lead_geo.bounds_min
    b_max = lead_geo.bounds_max
    
    # 动态构建 8 个顶点
    v = [
        [b_min[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_min[2]],
        [b_max[0], b_max[1], b_min[2]], [b_min[0], b_max[1], b_min[2]],
        [b_min[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_max[2]],
        [b_max[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_max[2]]
    ]
    
    faces = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], # 底, 顶
        [v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]], # 侧1, 2
        [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]]  # 侧3, 4
    ]
    
    poly3d = Poly3DCollection(faces, facecolors='gray', linewidths=0.5, edgecolors='black', alpha=0.03)
    ax.add_collection3d(poly3d)

    # --- 3. 动态绘制接收器 (正方形) ---
    S = receiver.S
    half_S = S / 2.0
    for center, normal, _ in receiver.detectors:
        # 寻找正方形的局部坐标系向量
        v_a = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
        v_a /= np.linalg.norm(v_a)
        v_b = np.cross(normal, v_a)
        
        corners = [
            center + half_S*v_a + half_S*v_b,
            center - half_S*v_a + half_S*v_b,
            center - half_S*v_a - half_S*v_b,
            center + half_S*v_a - half_S*v_b
        ]
        rec_poly = Poly3DCollection([corners], facecolors='orange', alpha=0.6, edgecolors='darkorange')
        ax.add_collection3d(rec_poly)


def visualize_3d_event(show_lead: bool = False, show_receivers: bool = True):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 实例化对象（这将读取你在 simulation 文件中的所有默认值）
    scint_geo = AnalyticCylinderGeometry()
    lead_geo = LeadShieldingGeometry(scint_geo)
    receiver = Receiver(np.array([0.0, 0.0, 0.05]))

    # --- 2. 绘制闪烁体 (圆柱体) ---
    R, H = scint_geo.R, scint_geo.H
    z = np.linspace(0, H, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    ax.plot_surface(R*np.cos(theta_grid), R*np.sin(theta_grid), z_grid, 
                    color='cyan', alpha=0.1, shade=False)
    
    # --- 3. 【新增】根据参数决定是否绘制铅层 ---
    if show_lead:
        b_min, b_max = lead_geo.bounds_min, lead_geo.bounds_max
        v = [
            [b_min[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]], [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_max[2]]
        ]
        faces = [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]],
            [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]]
        ]
        poly3d = Poly3DCollection(faces, facecolors='gray', linewidths=0.5, edgecolors='black', alpha=0.03)
        ax.add_collection3d(poly3d) 


    h_sigma = load_cross_section('H.csv')
    c_sigma = load_cross_section('C.csv')
    pb_sigma = np.array([[0.1, 30.0], [20.0, 15.0]])

    # --- 4. 【新增】显示接收器部分 ---
    if show_receivers:
        for det in receiver.detectors:
            center = det['center']
            norm = det['norm']
            radius = 0.00455
            
            # 生成圆盘的参数方程
            theta = np.linspace(0, 2*np.pi, 50)
            # 根据法线方向确定圆盘所在的平面
            if abs(norm[0]) > 0.9: # X方向的面
                y = center[1] + radius * np.cos(theta)
                z = center[2] + radius * np.sin(theta)
                x = np.full_like(y, center[0])
            elif abs(norm[1]) > 0.9: # Y方向的面
                x = center[0] + radius * np.cos(theta)
                z = center[2] + radius * np.sin(theta)  
                y = np.full_like(x, center[1])
                
            ax.plot(x, y, z, color='red', lw=2) # 画圆边框
            ax.plot_surface(np.array([x]), np.array([y]), np.array([z]), color='orange', alpha=0.8) # 填充颜色

    # 3. 运行真实轨迹模拟
    # 初始条件：从中子管位置发射
    n_path, p_paths, collision_events = track_real_event(
    initial_pos=np.array([0.0, 0.0, -0.05]), 
    initial_dir=np.array([0.0, 0.0, 1.0]),
    initial_energy=14.1,
    scint_geo=scint_geo,
    lead_geo=lead_geo,
    h_sigma_data=h_sigma,     # 修改点：传入氢微观数据
    c_sigma_data=c_sigma,     # 修改点：传入碳微观数据
    pb_sigma_data=pb_sigma,   # 修改点：传入铅微观数据
    receiver=receiver,
    light_yield_per_mev=5.0,  # 保持较小值以防 3D 界面卡顿
    photon_atten_len=1.0      # 增加衰减长度参数，可根据需要调整
    )

    # 4. 绘制轨迹
    # 中子：粗黑虚线
    n_pts = np.array(n_path)
    ax.plot(n_pts[:,0], n_pts[:,1], n_pts[:,2], 'k', linewidth=2, label='Neutron Path', alpha=0.8)
    

    # 光子：细红线
    for i, p_pts in enumerate(p_paths):
        p_pts = np.array(p_pts)
        ax.plot(p_pts[:,0], p_pts[:,1], p_pts[:,2], color='red', linewidth=0.5, alpha=0.3)
    
    color_map = {
        'H': 'green',   # 氢：绿色
        'C': 'yellow',  # 碳：黄色
        'Pb': 'black'   # 铅：黑色
    }
    for event in collision_events:
        pos = event['pos']
        target = event['type']
        ax.scatter(pos[0], pos[1], pos[2], 
                color=color_map[target], 
                s=30, edgecolors='white', linewidth=0.5,
                label=f'Scatter on {target}' if target not in [h.get_label() for h in ax.collections] else "")
    
    # 5. 设置坐标轴与视角
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    if show_lead:
        # 如果显示铅层，视野大一点
        ax.set_xlim(lead_geo.bounds_min[0]-0.05, lead_geo.bounds_max[0]+0.05)
        ax.set_ylim(lead_geo.bounds_min[1]-0.05, lead_geo.bounds_max[1]+0.05)
        ax.set_zlim(lead_geo.bounds_min[2]-0.05, lead_geo.bounds_max[2]+0.05)
    else:
        # 如果关闭铅层，视野聚焦在闪烁体
        margin = 0.01
        z_center = 0.05
        limit = scint_geo.R + margin  # 0.12
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(z_center - limit, z_center + limit)

    # --- 设置坐标轴标签 ---
    ax.set_xlabel('X Axis (meters)', fontsize=12, color='red')
    ax.set_ylabel('Y Axis (meters)', fontsize=12, color='green')
    ax.set_zlabel('Z Axis (meters)', fontsize=12, color='blue')

    ax.view_init(elev=30, azim=45)

    ax.set_box_aspect([1, 1, 1]) 

    plt.title("Synced 3D Scintillator Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_3d_event(show_lead=False)