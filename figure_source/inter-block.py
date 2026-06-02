import matplotlib.pyplot as plt
import numpy as np

# 设置超大字体以确保清晰度 - 扩大2倍
plt.rcParams.update({
    'font.size': 80,  # 从40扩大到80
    'axes.labelsize': 48,  # 从24扩大到48
    'axes.titlesize': 52,  # 从26扩大到52
    'xtick.labelsize': 44,  # 从22扩大到44
    'ytick.labelsize': 44,  # 从22扩大到44
    'legend.fontsize': 44,  # 从22扩大到44
    'figure.titlesize': 52,  # 从26扩大到52
    'figure.dpi': 100
})

# 数据
applications = ['mm4', 'mm4', 'con4', 'con4', 'bi4', 'bi4', 'mr', 'mtn', 'mtn']
methods = ['Accel-sim', 'sync', 'unorder']
colors = ['#1f77b4', '#ff7f0e', '#d62728']

# 您的数据
data = np.array([
    [1.30, 3.63, 43.79],      # mm
    [1.29, 3.98, 43.76],      # mm_u
    [2.72, 7.57, 27.81],      # con
    [2.46, 7.85, 22.28],      # con_u
    [27.78, 33.33, 55.56],    # bi
    [23.11, 31.60, 75.00],    # bi_u
    [98.80, 100.00, 100.00],   #rp
    [79.42, 100.00, 100.00],  # rp_u
    [85.91, 100.00, 100.00],  # mtt
    [78.35, 100.00, 100.00],  # mtt_u
])

# 分离数据
# not uniform 数据（不带_u后缀）
not_uniform_apps = ['mm_4', 'con_4', 'bi_4', 'mtx', 'rp']
not_uniform_indices = [1, 3, 5, 9, 7]  # 对应的索引
uniform_data = data[not_uniform_indices]

# uniform 数据（带_u后缀）
uniform_apps = ['mm_4', 'con_4', 'bi_4', 'mtx', 'rp']
uniform_indices = [0, 2, 4, 8, 6]  # 对应的索引 (mr是索引6)
not_uniform_data = data[uniform_indices]

# 创建图表，包含两个子图，调整子图间距
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

# 左边子图 - not uniform
x1 = np.arange(len(not_uniform_apps))
width = 0.25

bars1_1 = ax1.bar(x1 - width, not_uniform_data[:, 0], width,
                  label=methods[0], color=colors[0],
                  edgecolor='black', linewidth=2.4, alpha=0.9)  # 线宽也相应增加
bars1_2 = ax1.bar(x1, not_uniform_data[:, 1], width,
                  label=methods[1], color=colors[1],
                  edgecolor='black', linewidth=2.4, alpha=0.9)
bars1_3 = ax1.bar(x1 + width, not_uniform_data[:, 2], width,
                  label=methods[2], color=colors[2],
                  edgecolor='black', linewidth=2.4, alpha=0.9)

# 移除原来的标题，添加(a)标签，稍微缩小字体，并将位置下移
ax1.text(0.5, -0.25, '(a) without padding', transform=ax1.transAxes,
         fontsize=44, fontweight='bold', ha='center', va='top')  # 将-0.15改为-0.25，增加距离
ax1.set_ylabel('Miss Rate', fontsize=48, fontweight='bold')  # 保持48
ax1.set_xticks(x1)
# 加粗kernel名称
ax1.set_xticklabels(not_uniform_apps, rotation=45, ha='right', fontsize=40, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_yticks(np.arange(0, 101, 20))
ax1.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)], fontsize=44)  # 保持44

# 保留图表边框
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)

# 添加图例（左边子图），字体稍微缩小
legend1 = ax1.legend(frameon=False, loc='upper left', ncol=1, fontsize=40)  # 从48缩小到40

# 右边子图 - uniform
x2 = np.arange(len(uniform_apps))
width = 0.25

bars2_1 = ax2.bar(x2 - width, uniform_data[:, 0], width,
                  label=methods[0], color=colors[0],
                  edgecolor='black', linewidth=2.4, alpha=0.9)
bars2_2 = ax2.bar(x2, uniform_data[:, 1], width,
                  label=methods[1], color=colors[1],
                  edgecolor='black', linewidth=2.4, alpha=0.9)
bars2_3 = ax2.bar(x2 + width, uniform_data[:, 2], width,
                  label=methods[2], color=colors[2],
                  edgecolor='black', linewidth=2.4, alpha=0.9)

# 移除原来的标题，添加(b)标签，稍微缩小字体，并将位置下移
ax2.text(0.5, -0.25, '(b) with padding', transform=ax2.transAxes,
         fontsize=44, fontweight='bold', ha='center', va='top')  # 将-0.15改为-0.25，增加距离
# 去掉右边子图的y轴标题
ax2.set_ylabel('')  # 设置为空字符串
ax2.set_xticks(x2)
# 加粗kernel名称
ax2.set_xticklabels(uniform_apps, rotation=45, ha='right', fontsize=40, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.set_yticks(np.arange(0, 101, 20))
# 去掉右边子图的y轴刻度标签
ax2.set_yticklabels([])  # 设置为空列表，不显示刻度标签

# 保留图表边框
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)

# 添加图例（右边子图），字体稍微缩小
legend2 = ax2.legend(frameon=False, loc='upper left', ncol=1, fontsize=40)  # 从48缩小到40

# 调整布局，减少子图之间的空白
plt.tight_layout(pad=1.0)  # 减少pad值从3.0到1.0
plt.subplots_adjust(bottom=0.25, wspace=0.1)  # 增加bottom值从0.15到0.25，为标签留出更多空间

# 保存为超高分辨率PDF和PNG
plt.savefig('miss_rate_comparison_subplots_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('miss_rate_comparison_subplots_v2.png', dpi=300, bbox_inches='tight')

print("图表已保存为 'miss_rate_comparison_subplots_v2.pdf' 和 'miss_rate_comparison_subplots_v2.png'")