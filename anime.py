'''Created by Shen on 2025.12.24
   to plot neutron and photon tracks after it enter scintillator'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple


from scintillator_simulation_v04 import AnalyticCylinderGeometry, LeadShieldingGeometry, Receiver, get_mfp_energy_dependent, sample_isotropic_direction, energy_to_speed

# --------------------------------------------------------------------------
# 1. 轨迹追踪版模拟函数 (仅追踪一个中子)
# --------------------------------------------------------------------------

def track_real_event(
    initial_pos: np.ndarray,
    initial_dir: np.ndarray,
    initial_energy: float,
    scint_geo,
    lead_geo,
    eb_macro_sigma,
    pb_macro_sigma,
    receiver,
    light_yield_per_mev: float = 10.0,
    photon_atten_len: float = 1.0
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    n_path = [np.copy(initial_pos)]
    p_paths = []
    
    n_pos = np.copy(initial_pos)
    n_dir = initial_dir / np.linalg.norm(initial_dir)
    n_energy = initial_energy
    
    # --- 中子输运 (保持真实物理逻辑) ---
    while n_energy > 0.1:
        in_scint = scint_geo.is_inside(n_pos)
        in_lead = lead_geo.is_inside_lead(n_pos)
        
        # 确定中子当前的 MFP
        if in_scint:
            mfp = get_mfp_energy_dependent(n_energy, eb_macro_sigma)
            medium = "SCINT"
        elif in_lead:
            mfp = get_mfp_energy_dependent(n_energy, pb_macro_sigma)
            medium = "LEAD"
        else:
            # 真空/空气区逻辑
            d_s = scint_geo.get_distance_to_boundary(n_pos, n_dir)
            d_l = lead_geo.get_distance_to_lead_boundary(n_pos, n_dir)
            d_v = min(d_s, d_l)
            if d_v > 2.0: break
            n_pos += n_dir * (d_v + 1e-6) # 步进稍微大一点跨过边界
            n_path.append(np.copy(n_pos))
            continue

        d_coll = -mfp * np.log(np.random.rand() + 1e-12)
        d_bound = scint_geo.get_distance_to_boundary(n_pos, n_dir) if in_scint else \
                  lead_geo.get_distance_to_lead_boundary(n_pos, n_dir)
        
        if d_coll < d_bound:
            n_pos += n_dir * d_coll
            n_path.append(np.copy(n_pos))
            
            if medium == "SCINT":
                dep_energy = n_energy * np.random.uniform(0.1, 0.5)
                n_energy -= dep_energy
                n_dir = sample_isotropic_direction()
                
                # --- 产生并追踪光子 ---
                num_p = int(dep_energy * light_yield_per_mev)
                for _ in range(min(num_p, 15)): 
                    p_path = [np.copy(n_pos)]
                    cp_pos = np.copy(n_pos)
                    cp_dir = sample_isotropic_direction()
                    max_p_dist = -photon_atten_len * np.log(np.random.rand() + 1e-12)
                    acc_dist = 0.0
                    
                    # 增加最大反射次数，以便观察轨迹
                    for _ in range(30): 
                        dp_bound = scint_geo.get_distance_to_boundary(cp_pos, cp_dir)
                        
                        # 行程结束判定
                        if acc_dist + dp_bound > max_p_dist:
                            p_path.append(cp_pos + cp_dir * (max_p_dist - acc_dist))
                            break
                        
                        # 1. 移动到边界
                        cp_pos += cp_dir * dp_bound
                        acc_dist += dp_bound
                        p_path.append(np.copy(cp_pos))
                        
                        # 2. 边界处理逻辑 (此时 cp_pos 理论上在边界上)
                        # 我们先计算法线
                        norm = scint_geo.get_boundary_normal(cp_pos, cp_dir)
                        
                        # 3. 检查是否被接收器吸收
                        if receiver.check_absorption(cp_pos, norm):
                            break # 终止该光子，它被吸收了
                        
                        # 4. 执行反射 (这里假设 100% 反射以便观察)
                        # 计算反射方向：v_ref = v - 2*(v·n)*n
                        cp_dir = cp_dir - 2.0 * np.dot(cp_dir, norm) * norm
                        cp_dir /= np.linalg.norm(cp_dir)
                        
                        # ⚠️ 关键修复：向反射方向步进一小段距离，确保回到圆柱体内
                        cp_pos += cp_dir * 1e-5 
                        
                    p_paths.append(p_path)
            else:
                n_dir = sample_isotropic_direction()
        else:
            # 穿过边界
            n_pos += n_dir * (d_bound + 1e-6)
            n_path.append(np.copy(n_pos))

    return n_path, p_paths
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


    eb_sigma = np.array([[0.1, 50.0], [20.0, 5.0]]) 
    pb_sigma = np.array([[0.1, 30.0], [20.0, 15.0]])

    # --- 4. 【新增】显示接收器部分 ---
    if show_receivers:
        S = receiver.S  # 读取定义的正方形边长 (0.1m)
        half_S = S / 2.0
        for item in receiver.detectors:
            center, normal = item[0], item[1] # 解包中心和法线
            
            # 计算正方形的局部坐标系向量 (v_a, v_b)
            # 找到一个不与法线平行的参考向量
            ref = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            v_a = np.cross(normal, ref)
            v_a /= np.linalg.norm(v_a)
            v_b = np.cross(normal, v_a)
            
            # 计算正方形的四个顶点
            corners = [
                center + half_S * v_a + half_S * v_b,
                center - half_S * v_a + half_S * v_b,
                center - half_S * v_a - half_S * v_b,
                center + half_S * v_a - half_S * v_b
            ]
            
            # 绘制橙色半透明面板
            rec_poly = Poly3DCollection([corners], facecolors='orange', alpha=0.4, edgecolors='red', linewidths=1)
            ax.add_collection3d(rec_poly)
            # 标记中心点
            #ax.scatter(center[0], center[1], center[2], color='red', s=10)

    # 3. 运行真实轨迹模拟
    # 初始条件：从中子管位置发射
    n_path, p_paths = track_real_event(
        initial_pos=np.array([0.0, 0.0, -0.05]), 
        initial_dir=np.array([0.0, 0.0, 1.0]),
        initial_energy=14.1,
        scint_geo=scint_geo,
        lead_geo=lead_geo,
        eb_macro_sigma=eb_sigma,
        pb_macro_sigma=pb_sigma,
        receiver=receiver,
        light_yield_per_mev=5.0 # 设置小一点，避免画面太乱
    )

    # 4. 绘制轨迹
    # 中子：粗黑虚线
    n_pts = np.array(n_path)
    ax.plot(n_pts[:,0], n_pts[:,1], n_pts[:,2], 'k', linewidth=2, label='Neutron Path', alpha=0.8)
    ax.scatter(n_pts[:,0], n_pts[:,1], n_pts[:,2], color='blue', s=10) # 碰撞点

    # 光子：细红线
    for i, p_pts in enumerate(p_paths):
        p_pts = np.array(p_pts)
        ax.plot(p_pts[:,0], p_pts[:,1], p_pts[:,2], color='red', linewidth=0.5, alpha=0.3)

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

    ax.view_init(elev=30, azim=50)

    ax.set_box_aspect([1, 1, 1]) 

    plt.title("Synced 3D Scintillator Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_3d_event(show_lead=False)