import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x = [64, 128, 256, 512, 1024, 2048]
y1 = [62.47, 62.53, 62.55, 62.48, 62.48, 62.47]
y2 = [60.97, 61.03, 61.05, 61.05, 61.07, 61.06]
y3 = [62.25, 62.31, 62.36, 62.44, 62.47, 62.47]
y4 = [62.92, 62.88, 62.87, 62.95, 62.94, 62.95]

# 假设标准差
std1 = [0.24, 0.21, 0.17, 0.15, 0.13, 0.13]
std2 = [0.24, 0.21, 0.18, 0.12, 0.11, 0.11]
std3 = [0.26, 0.22, 0.16, 0.15, 0.10, 0.10]
std4 = [0.25, 0.20, 0.19, 0.10, 0.13, 0.11]

# 生成误差条数据
plt.style.use('seaborn')
# 设置字体
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 12}
plt.rc('font', **font)  # 设置全局字体
# 创建图形
plt.figure(figsize=(5, 3.5))

# 绘制每条折线
plt.plot(x, y1, marker='D', label='C4', color='#E87F8F')  # 蓝色
plt.plot(x, y2, marker='o', label='Wikipedia', color='#E0B800',markersize=8)  # 青色
plt.plot(x, y3, marker='s', label='Slimapjama', color='#6E90B3')  # 粉色，带误差条
plt.plot(x, y4, marker='*', label='DCLM', color='#66A996',markersize=12)  # 紫色
plt.fill_between(x, np.array(y1) - np.array(std1), np.array(y1) + np.array(std1),
                 color='#E87F8F', alpha=0.3)
plt.fill_between(x, np.array(y2) - np.array(std2), np.array(y2) + np.array(std2),
                 color='#E0B800', alpha=0.3)
plt.fill_between(x, np.array(y3) - np.array(std3), np.array(y3) + np.array(std3),
                 color='#6E90B3', alpha=0.3)
plt.fill_between(x, np.array(y4) - np.array(std4), np.array(y4) + np.array(std4),
                 color='#66A996', alpha=0.3)  # 使用阴影表示标准差

# 设置轴的范围
plt.ylim(60, 64)
plt.xscale('log')  # 保持x轴对数尺度
plt.xticks(x, ['64', '128', '256', '512', '1024', '2048'])
# 设置标签和标题
# plt.xlabel('Number of Calibration Examples', fontsize=14, fontdict={'family': 'Times New Roman', 'weight': 'bold'})
plt.ylabel('Average accuracy', fontsize=14, fontdict={'family': 'Times New Roman', 'weight': 'bold'})

plt.xticks(fontsize=12)  # 设置x轴刻度大小
plt.yticks(fontsize=12)  # 设置y轴刻度大小

# 添加图例
plt.legend(fontsize='medium', frameon=True, facecolor='white', edgecolor='#D3D3D3', loc='upper center', ncol=2, columnspacing=1)


# 展示图表
plt.tight_layout()
plt.show()
