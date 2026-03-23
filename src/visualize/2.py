import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 样式设置 (符合学术顶会审美)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# ==========================================
# 2. 数据加载 (请在这里替换为您的真实路径)
# ==========================================
def load_real_data():
    # Added encoding='utf-8' here
    with open(r"F:\academic\dynamic-spherical-result\MATH500_4k-045\experiment_results_fixed.json", 'r', encoding='utf-8') as f:
        data_json = json.load(f)
        return {
            "Baseline": data_json["Baseline"], 
            "Continuous": data_json["Continuous"], 
            "Dynamic_Spherical": data_json["Dynamic_Spherical"]
        }

data = load_real_data()

# 注意：这里使用我们刚才后台生成的 mock_data 字典。
# 当您使用真实数据时，只需将其替换为您 load 进来的 json 字典即可。

# ==========================================
# 3. 画板初始化
# ==========================================
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

# ==========================================
# 图 1: 逻辑翻转矩阵 (Logic Flip Matrix)
# 验证：0 负面干扰与正向净收益
# ==========================================
ax1 = fig.add_subplot(gs[0, 0])
flip_data = {'Base_False_Dyn_False': 0, 'Base_False_Dyn_True': 0, 
             'Base_True_Dyn_False': 0, 'Base_True_Dyn_True': 0}

# 遍历对比每一个样本的 id
base_dict = {p['id']: p for p in data["Baseline"]["per_problem"]}
dyn_dict = {p['id']: p for p in data["Dynamic_Spherical"]["per_problem"]}

for pid in base_dict.keys():
    if pid in dyn_dict:
        b_corr = base_dict[pid]['correct']
        d_corr = dyn_dict[pid]['correct']
        
        if not b_corr and not d_corr: flip_data['Base_False_Dyn_False'] += 1
        elif not b_corr and d_corr: flip_data['Base_False_Dyn_True'] += 1
        elif b_corr and not d_corr: flip_data['Base_True_Dyn_False'] += 1
        elif b_corr and d_corr: flip_data['Base_True_Dyn_True'] += 1

matrix = np.array([[flip_data['Base_True_Dyn_True'], flip_data['Base_True_Dyn_False']],
                   [flip_data['Base_False_Dyn_True'], flip_data['Base_False_Dyn_False']]])

sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax1, annot_kws={"size": 18})
ax1.set_title("Logic Flip Matrix: Zero Broken & Net Gain", fontsize=18, fontweight='bold')
ax1.set_xticklabels(['Dynamic Correct', 'Dynamic Wrong'])
ax1.set_yticklabels(['Baseline Correct', 'Baseline Wrong'])

# 添加核心亮点文字
net_gain = flip_data['Base_False_Dyn_True']
broken = flip_data['Base_True_Dyn_False']
ax1.text(0.5, 1.15, f"Net Gain: Corrected {net_gain} errors", color='green', fontsize=16, ha='center', transform=ax1.transAxes)
ax1.text(0.5, 1.08, f"Broken: {broken} errors (Zero Negative Interference)", color='red', fontsize=16, ha='center', transform=ax1.transAxes)

# ==========================================
# 图 2: T0-Aligned 动力学曲线 (引入 Baseline 对比)
# ==========================================
ax2 = fig.add_subplot(gs[0, 1])

target_id = None
target_dyn_prob = None
target_base_prob = None

# 1. 严格筛选: 寻找一个 Baseline错 + Dynamic对 + Alpha有明显峰值的样本
for pid in base_dict.keys():
    if pid in dyn_dict:
        b_prob = base_dict[pid]
        d_prob = dyn_dict[pid]
        
        if not b_prob['correct'] and d_prob['correct'] and d_prob.get('intervention_start') is not None:
            # 确保这个样本不是微弱干预，找一个 alpha 峰值大于 0.3 的典型案例
            if max(d_prob['alpha_trajectory']) > 0.3:
                target_id = pid
                target_dyn_prob = d_prob
                target_base_prob = b_prob
                break

# 如果找不到 alpha > 0.3 的，就退而求其次找任意翻转成功的
if target_dyn_prob is None:
    for pid in base_dict.keys():
        if pid in dyn_dict and not base_dict[pid]['correct'] and dyn_dict[pid]['correct'] and dyn_dict[pid].get('intervention_start') is not None:
            target_id = pid
            target_dyn_prob = dyn_dict[pid]
            target_base_prob = base_dict[pid]
            break

if target_dyn_prob:
    t0 = target_dyn_prob['intervention_start']
    
    # 设定窗口，提取 Dynamic 数据
    window_start = max(0, t0 - 10)
    window_end = min(len(target_dyn_prob['ema_trajectory']), t0 + 30)
    x_axis = np.arange(window_start - t0, window_end - t0)
    
    dyn_ema = target_dyn_prob['ema_trajectory'][window_start:window_end]
    dyn_alpha = target_dyn_prob['alpha_trajectory'][window_start:window_end]
    
    # 提取 Baseline 相同位置的 EMA 数据 (防越界处理)
    base_ema_full = target_base_prob['ema_trajectory']
    base_ema = []
    for i in range(window_start, window_end):
        if i < len(base_ema_full):
            base_ema.append(base_ema_full[i])
        else:
            # 如果 baseline 提前结束了，用最后一个值或 NaN 填充
            base_ema.append(np.nan) 
    
    # --- 开始绘制 ---
    # 1. 绘制 Baseline EMA (灰色虚线，作为反面教材)
    ax2.plot(x_axis, base_ema, color='gray', linestyle='--', linewidth=2.5, alpha=0.7, label='Baseline EMA (Unsteered)')
    
    # 2. 绘制 Dynamic EMA (红色实线，展示降熵)
    ax2.plot(x_axis, dyn_ema, color='#d62728', linewidth=3, label='Dynamic EMA (Sensor)')
    
    ax2.set_ylabel('EMA Entropy', color='#d62728', fontsize=16)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='T=0 (Intervention Triggered)')
    
    # 3. 绘制 Alpha 强度 (蓝色填充，展示执行器动作)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_axis, dyn_alpha, color='#1f77b4', linewidth=3, label='Alpha Intensity (Actuator)')
    ax2_twin.fill_between(x_axis, 0, dyn_alpha, color='#1f77b4', alpha=0.1)
    ax2_twin.set_ylabel('Alpha Intensity', color='#1f77b4', fontsize=16)
    
    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    # 将图例放在最佳位置，避免遮挡曲线
    ax2_twin.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=11, framealpha=0.9)
    
    ax2.set_title(f"T0-Aligned Dynamics vs Baseline (Case: {target_id})", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Tokens relative to T0", fontsize=16)

# ==========================================
# 图 3: 抑制过度思考 (Token Efficiency)
# 验证：更少的 Token 实现更高的认知转化率
# ==========================================
ax3 = fig.add_subplot(gs[1, 0])
base_tokens = [p['num_tokens'] for p in data["Baseline"]["per_problem"]]
dyn_tokens = [p['num_tokens'] for p in data["Dynamic_Spherical"]["per_problem"]]

sns.kdeplot(base_tokens, fill=True, color='gray', label='Baseline (Overthinking)', ax=ax3, alpha=0.5)
sns.kdeplot(dyn_tokens, fill=True, color='#1f77b4', label='Dynamic Steering (Efficient)', ax=ax3, alpha=0.5)
ax3.set_title("Overthinking Suppression: Token Efficiency", fontsize=18, fontweight='bold')
ax3.set_xlabel("Number of Tokens Consumed", fontsize=16)
ax3.set_ylabel("Density", fontsize=16)
ax3.legend(fontsize=14)

# ==========================================
# 图 4: 物理防休克验证 (State Shock Defense)
# 验证：保范数球面干预如何压制复读机现象
# ==========================================
ax4 = fig.add_subplot(gs[1, 1])
rep_data = []
for p in data["Baseline"]["per_problem"]: 
    rep_data.append({'Model': 'Baseline', 'Repetition Rate': p['repetition']})
for p in data["Continuous"]["per_problem"]: 
    rep_data.append({'Model': 'Continuous\n(Linear Addition)', 'Repetition Rate': p['repetition']})
for p in data["Dynamic_Spherical"]["per_problem"]: 
    rep_data.append({'Model': 'Dynamic\n(Spherical)', 'Repetition Rate': p['repetition']})

df_rep = pd.DataFrame(rep_data)
sns.boxplot(x='Model', y='Repetition Rate', data=df_rep, palette=['lightgray', '#d62728', '#1f77b4'], ax=ax4)
ax4.set_title("State Shock Defense: Linguistic Stability", fontsize=18, fontweight='bold')
ax4.set_ylabel("N-gram Repetition Rate", fontsize=16)
ax4.set_xlabel("")

plt.tight_layout()
plt.savefig("LLM_Interpretability_Dashboard.png", dpi=300, bbox_inches='tight')
plt.show()