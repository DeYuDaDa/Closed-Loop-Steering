import json
import numpy as np
import matplotlib.pyplot as plt

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')

# 1. 加载数据 (替换为你的真实路径)
with open(r"F:\academic\dynamic-spherical-result\MATH500_4k-045\experiment_results_fixed.json", 'r', encoding='utf-8') as f:
    data_json = json.load(f)

base_dict = {p['id']: p for p in data_json["Baseline"]["per_problem"]}
dyn_probs = data_json["Dynamic_Spherical"]["per_problem"]

# 2. 筛选 10 个完美的逆转 Case
target_cases = []
for p in dyn_probs:
    pid = p.get('id')
    if pid in base_dict and not base_dict[pid]['correct'] and p.get('correct') and p.get('intervention_start') is not None:
        target_cases.append(p)
        if len(target_cases) >= 10:
            break

# 3. 初始化 5x2 画布
fig, axes = plt.subplots(5, 2, figsize=(24, 25))
axes = axes.flatten()

L_max, rho = 36, 0.85
DTR_THRESHOLD = int(np.ceil(L_max * rho))

for idx, ax in enumerate(axes):
    if idx >= len(target_cases):
        ax.axis('off')
        continue
        
    sample = target_cases[idx]
    t0 = sample['intervention_start']
    window_start = max(0, t0 - 15)
    window_end = min(len(sample['ema_trajectory']), len(sample['dtr_trajectory']), t0 + 25)
    x_axis = np.arange(window_start - t0, window_end - t0)
    
    ema_window = sample['ema_trajectory'][window_start:window_end]
    alpha_window = sample['alpha_trajectory'][window_start:window_end]
    dtr_depth_window = np.array(sample['dtr_trajectory'][window_start:window_end])
    
    # 绘制 EMA
    ax.plot(x_axis, ema_window, color='#d62728', linewidth=2.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    # 绘制 DTR 剖面
    y_max_ema = max(ema_window) * 1.2 if max(ema_window) > 0 else 1.0
    ax.set_ylim(0, y_max_ema)
    scaling_factor = (y_max_ema * 0.3) / L_max 
    scaled_depth = dtr_depth_window * scaling_factor
    scaled_threshold = DTR_THRESHOLD * scaling_factor
    
    ax.fill_between(x_axis, 0, scaled_depth, color='#2ca02c', alpha=0.25, step='mid')
    ax.step(x_axis, scaled_depth, where='mid', color='#2ca02c', linewidth=1.5)
    ax.axhline(y=scaled_threshold, color='green', linestyle=':', linewidth=1.5)
    
    # 绘制 Alpha
    ax_twin = ax.twinx()
    ax_twin.plot(x_axis, alpha_window, color='#1f77b4', linewidth=2.5)
    ax_twin.set_ylim(0, max(alpha_window) * 1.2 if max(alpha_window) > 0 else 1.0)
    
    ax.set_title(f"Case: {sample.get('id')}", fontsize=12, fontweight='bold')
    if idx >= 8: # 只在最底下一排显示 X 轴标签
        ax.set_xlabel("Tokens relative to T0", fontsize=10)

plt.tight_layout()
plt.savefig("10_cases_grid.png", dpi=300)
print("10_cases_grid.png saved!")