import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
# 数据
group1 = [72.16]  # 第一组数据
group2 = [72.21]  # 第二组数据
group3 = [72.19]  # 第三组数据
group4 = [72.21]  # 第四组数据

std_group1 = [0.07]
std_group2 = [0.06]
std_group3 = [0.07]
std_group4 = [0.07]

# 设置柱子的位置和宽度
barWidth = 0.06
spacing = 0.002  # 设置柱子之间的间距
# 设置每组数据的柱子位置
r1 = np.arange(len(group1))  # 第一组柱子的x轴位置
r2 = [x + barWidth+spacing for x in r1]  # 第二组柱子的x轴位置，紧挨着第一组
r3 = [x + barWidth+spacing for x in r2]
r4 = [x + barWidth+spacing for x in r3]

# 设置样式
plt.style.use('seaborn-white')

# 设置字体
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)  # 设置全局字体

# 设置画布像素
plt.figure(figsize=(4,4))

# 绘制柱状图#F4A8A8
plt.bar(r1, group1, color='#F6BEBE', width=barWidth, label='C4', hatch='/', edgecolor='white', linewidth=0)
plt.bar(r2, group2, color='#EEE8AA', width=barWidth, label='Wikipedia', hatch='-', edgecolor='white', linewidth=0)
plt.bar(r3, group3, color='#C8D9EB', width=barWidth, label='Slimpajama', hatch='\\', edgecolor='white', linewidth=0)
plt.bar(r4, group4, color='#C4E3D9', width=barWidth, label='DCLM', hatch='|', edgecolor='white', linewidth=0)

def draw_error_bars(x, y, yerr, color):
    for xi, yi, yierr in zip(x, y, yerr):
        # plt.plot([xi, xi], [yi - yierr, yi + yierr], color=color, linestyle='--', linewidth=1.5)  # 绘制虚线主线
        plt.plot([xi - 0.015, xi + 0.015], [yi + yierr, yi + yierr], color=color, linestyle='--', linewidth=1.5)  # 绘制误差条顶部虚线端点
        plt.plot([xi - 0.015, xi + 0.015], [yi - yierr, yi - yierr], color=color, linestyle='--', linewidth=1.5)  # 绘制误差条底部虚线端点

draw_error_bars(r1, group1, std_group1, color='#E87F8F')
draw_error_bars(r2, group2, std_group2, color='#E0B800')
draw_error_bars(r3, group3, std_group3, color='#6E90B3')
draw_error_bars(r4, group4, std_group4, color='#66A996')

plt.axhline(y=71.96, color='#A0A0A0', linestyle='--', linewidth=2)

plt.errorbar(r1, group1, yerr=std_group1, fmt='D', capsize=0, color='#E87F8F', markersize=6, elinewidth=2, capthick=0)
plt.errorbar(r2, group2, yerr=std_group2, fmt='o', capsize=0, color='#E0B800', markersize=7, elinewidth=2, capthick=0)
plt.errorbar(r3, group3, yerr=std_group3, fmt='s', capsize=0, color='#6E90B3', markersize=7, elinewidth=2, capthick=0)
plt.errorbar(r4, group4, yerr=std_group4, fmt='*', capsize=0, color='#66A996', markersize=10, elinewidth=2, capthick=0)

plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# 调整 y 轴范围
plt.ylim(71, 72.60)
plt.xlim(-0.05, r4[0] + 0.05)  # 控制两侧空白
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# 设置整个图表的边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('#A0A0A0')  # 设置边框颜色
    spine.set_linewidth(2)  # 设置边框粗细

# 添加图例
plt.legend(fontsize='medium', frameon=True, facecolor='white', edgecolor='#D3D3D3', loc='upper center', ncol=2, columnspacing=1)

# 显示图表
plt.show()

