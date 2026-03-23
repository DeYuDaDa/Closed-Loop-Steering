import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ==========================================
# 1. 学术级样式设置
# ==========================================
plt.style.use('seaborn-v0_8-paper') 
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5}) 

COLOR_BASE = '#95a5a6' 
COLOR_CONT = '#e74c3c' 
COLOR_DYN  = '#2980b9' 
DTR_COLOR  = '#27ae60' 

L_max = 36
DTR_THRESHOLD = 31

# ==========================================
# 2. 数据加载
# ==========================================
def load_real_data():
    file_path = r"F:\academic\dynamic-spherical-result\MATH500_4k-03\experiment_results_fixed.json"
    print(f"[1/4] Loading real data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
        return {
            "Baseline": data_json["Baseline"], 
            "Continuous": data_json["Continuous"], 
            "Dynamic_Spherical": data_json["Dynamic_Spherical"]
        }

try:
    data = load_real_data()
except FileNotFoundError:
    print(f"Error: File not found.")
    exit()

# ==========================================
# 3. 数据重组 (修复内存问题的核心)
# ==========================================
print(f"[2/4] Aggregating data...")

violin_dtr_records = []
token_level_records = [] 

for model_name, group_key, color in [("Baseline", "Baseline", COLOR_BASE),
                                    ("Continuous", "Continuous", COLOR_CONT),
                                    ("Dynamic", "Dynamic_Spherical", COLOR_DYN)]:
    
    probs_dict = data[group_key]["per_problem"]
    for p in probs_dict:
        if p.get('local_dtr') is not None:
            violin_dtr_records.append({'Model': model_name, 'Local DTR': p['local_dtr']})
            
    for p in probs_dict:
        ema_traj = p.get('ema_trajectory')
        if ema_traj:
            for val in ema_traj:
                if not np.isnan(val):
                    token_level_records.append({'Model': model_name, 'EMA Entropy': val})

df_dtr = pd.DataFrame(violin_dtr_records)
df_ema_kde = pd.DataFrame(token_level_records)

# --- 修复内存崩溃：不再使用 pd.merge，改为横向拼接或分开保存 ---
# 方案：使用 concat 避免笛卡尔积，这样生成的行数等于最长的那组数据，而不是相乘
# df_macro_export = pd.concat([
#     df_ema_kde.reset_index(drop=True), 
#     df_dtr.reset_index(drop=True).rename(columns={'Model': 'Model_Prob'})
# ], axis=1)

# df_macro_export.to_csv("Macro_Data_Fixed.csv", index=False)
# print(f"[!] Data saved to: Macro_Data_Fixed.csv")

# ==========================================
# 4. 画板初始化
# ==========================================
fig = plt.figure(figsize=(24, 7)) 
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1, 1, 1.3])

# --- 图 1: 标注分位点数值 ---
ax1 = fig.add_subplot(gs[0, 0])
sns.violinplot(x='Model', y='Local DTR', data=df_dtr, 
               order=["Baseline", "Continuous", "Dynamic"], # 显式固定顺序
               hue='Model', 
               palette=[COLOR_BASE, COLOR_CONT, COLOR_DYN], 
               ax=ax1, inner="quartile", cut=0, legend=False)

# 定义模型顺序和颜色，确保一致性
models = ["Baseline", "Continuous", "Dynamic"]
palette = [COLOR_BASE, COLOR_CONT, COLOR_DYN]

# ==========================================
# 4. 独立保存三幅图
# ==========================================

# --- 图 1: Violin Plot with Quartile Values ---
plt.figure(figsize=(8, 6))
ax1 = sns.violinplot(x='Model', y='Local DTR', data=df_dtr, 
                    order=models, hue='Model', palette=palette,
                    inner="quartile", cut=0, legend=False)

# 标注分位点
for i, model in enumerate(models):
    stats = df_dtr[df_dtr['Model'] == model]['Local DTR'].quantile([0.25, 0.5, 0.75]).values
    for val in stats:
        plt.text(i + 0.12, val, f'{val:.2f}', 
                 fontsize=10, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.title("A. Retaining Deep Thinking Strength", fontsize=14, fontweight='bold', loc='left')
plt.ylim(-0.05, 1.1)
plt.savefig("Fig1_DTR_Violin.png", dpi=300, bbox_inches='tight')
print("[!] Saved: Fig1_DTR_Violin.png")
plt.close()

# --- 图 2: KDE Plot ---
plt.figure(figsize=(8, 6))
for m, c, ls, a, lab in [('Baseline', COLOR_BASE, '--', 0.1, 'Baseline (Chaotic)'),
                         ('Continuous', COLOR_CONT, '-', 0.3, 'Continuous (State Shock)'),
                         ('Dynamic', COLOR_DYN, '-', 0.6, 'Dynamic (Spherical)')]:
    sns.kdeplot(df_ema_kde[df_ema_kde['Model'] == m]['EMA Entropy'], 
                fill=True, color=c, alpha=a, linestyle=ls, label=lab)

plt.title("B. Elimination of Steering Uncertainty", fontsize=14, fontweight='bold', loc='left')
plt.xlim(0, 5)
plt.legend(fontsize=10, loc='upper right')
plt.savefig("Fig2_Entropy_KDE.png", dpi=300, bbox_inches='tight')
print("[!] Saved: Fig2_Entropy_KDE.png")
plt.close()

# # --- 图 3: Case Study (Causal Loop) ---
# if target_sample_dyn:
#     fig, ax3 = plt.subplots(figsize=(10, 6))
#     t0 = target_sample_dyn['intervention_start']
#     win_s, win_e = max(0, t0 - 15), min(len(target_sample_dyn['ema_trajectory']), t0 + 25)
#     x_axis = np.arange(win_s - t0, win_e - t0)
    
#     # 左轴: EMA
#     ax3.plot(x_axis, target_sample_base['ema_trajectory'][win_s:win_e], 
#              color=COLOR_BASE, linestyle='--', label='Baseline EMA')
#     ax3.plot(x_axis, target_sample_dyn['ema_trajectory'][win_s:win_e], 
#              color=COLOR_CONT, linewidth=3, label='Dynamic EMA')
#     ax3.set_xlabel("Tokens relative to T0")
#     ax3.set_ylabel("EMA Entropy / Trajectory")
    
#     # 右轴: Alpha
#     ax3_twin = ax3.twinx()
#     ax3_twin.plot(x_axis, target_sample_dyn['alpha_trajectory'][win_s:win_e], 
#                   color=COLOR_DYN, linewidth=3, label='Alpha Intensity')
#     ax3_twin.set_ylabel("Alpha Intensity")
    
#     plt.title(f"C. Causal Loop Contrast (Case {target_sample_dyn['id']})", 
#               fontsize=14, fontweight='bold', loc='left')
    
#     # 合并图例
#     lines1, labels1 = ax3.get_legend_handles_labels()
#     lines2, labels2 = ax3_twin.get_legend_handles_labels()
#     ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
#     plt.savefig("Fig3_Causal_Loop.png", dpi=300, bbox_inches='tight')
#     print("[!] Saved: Fig3_Causal_Loop.png")
#     plt.close()
# # ==========================================
# 5. 保存 (修复 tight_layout 警告)
# ==========================================
print(f"[4/4] Finalizing...")
# 使用 subplots_adjust 代替 tight_layout 解决 twinx 兼容性问题
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)
plt.savefig("Steering_Dynamics_Fixed.png", dpi=300, bbox_inches='tight')
print(f"[!] Plot saved as Steering_Dynamics_Fixed.png")
plt.show()