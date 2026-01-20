# 生成用于中子闪烁体模拟的初始数据neutron_data.csv

import numpy as np
import pandas as pd

# 设置模拟的中子数量
num_neutrons = 32800

# 1. 设定停止位置 (z = 0, x^2 + y^2 <= 0.1^2)
# 在半径为 0.1m 的圆内均匀采样
r = 0.1 * np.sqrt(np.random.uniform(0, 1, num_neutrons))
theta = np.random.uniform(0, 2 * np.pi, num_neutrons)

pos_x = r * np.cos(theta)
pos_y = r * np.sin(theta)
pos_z = np.zeros(num_neutrons)

# 2. 设定起始点和计算方向向量
# 起始点 (0, 0, -2.9)
start_pos = np.array([0, 0, -2.9])
dist_z = 2.9  # z方向位移

# 计算方向向量 (dir = stop_pos - start_pos) 并归一化
dx = pos_x - start_pos[0]
dy = pos_y - start_pos[1]
dz = pos_z - start_pos[2]
norm = np.sqrt(dx**2 + dy**2 + dz**2)

dir_x = dx / norm
dir_y = dy / norm
dir_z = dz / norm

# 3. 能量分布：以 2.45MeV 为中心，0.1MeV 为半宽 (FWHM) 的高斯分布
# FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
fwhm = 184.47e-3  # 0.05834 MeV
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
energy_mev = np.random.normal(2.45, sigma, num_neutrons)

# 4. 计算飞行时间 (time_s)
# 中子质量 m = 1.6749e-27 kg, 1 eV = 1.602e-19 J
m_n = 1.67492749804e-27 
joules_per_mev = 1.60218e-13

# 速度 v = sqrt(2 * E / m)
velocity = np.sqrt(2 * energy_mev * joules_per_mev / m_n)
# 时间 t = 距离 / 速度
time_s = norm / velocity

# 5. 构建 DataFrame 并保存
df = pd.DataFrame({
    'detector_hit_x_m': pos_x,
    'detector_hit_y_m': pos_y,
    'detector_hit_z_m': pos_z,
    'direction_x': dir_x,
    'direction_y': dir_y,
    'direction_z': dir_z,
    'final_energy_MeV': energy_mev,
    'time_s': time_s
})

df.to_csv(f'neutron_data_{num_neutrons}.csv', index=False)
print(f"成功生成包含 {num_neutrons} 条数据的 neutron_data_{num_neutrons}.csv")

