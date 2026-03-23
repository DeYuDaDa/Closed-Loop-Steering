import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 基础样式设置
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# ================================
# 1. 加载数据
# ================================
file_path = r"F:\academic\dynamic-spherical-result\MATH500_4k-03\experiment_results_fixed.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

baseline_probs = {p['id']: p for p in data['Baseline']['per_problem']}
dynamic_probs = {p['id']: p for p in data['Dynamic_Spherical']['per_problem']}

# ================================
# 图 1: Taming Test-Time Compute (已移除红线和截断标签)
# ================================
plt.figure(figsize=(10, 6))
base_tokens = [p['num_tokens'] for p in data['Baseline']['per_problem']]
dyn_tokens = [p['num_tokens'] for p in data['Dynamic_Spherical']['per_problem']]

sns.kdeplot(base_tokens, fill=True, color='gray', label='Baseline (Unsteered)', alpha=0.5, linewidth=2)
sns.kdeplot(dyn_tokens, fill=True, color='#1f77b4', label='Dynamic Steering', alpha=0.5, linewidth=2)

# 移除 axvline (红线)
# 移除 label 中的 "Rescuing Truncation"

plt.title("Taming Test-Time Compute:\nToken Efficiency Analysis", fontsize=16, fontweight='bold')
plt.xlabel("Number of Tokens Consumed", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig("plot_1_token_efficiency.png", dpi=300)
plt.close()
print("Saved: plot_1_token_efficiency.png (Red line removed)")

# ================================
# 图 2: Net Gain 构成分析 (保持不变)
# ================================
plt.figure(figsize=(10, 8))
rescued_count = 0
corrected_count = 0

for pid, b_prob in baseline_probs.items():
    if pid in dynamic_probs:
        d_prob = dynamic_probs[pid]
        if not b_prob.get('correct') and d_prob.get('correct'):
            b_pred = b_prob.get('predicted')
            if b_pred is None or str(b_pred).strip() == "" or str(b_pred).strip().lower() == "none":
                rescued_count += 1
            else:
                corrected_count += 1

labels = ['Rescued from Truncation\n(Overthinking Trap)', 'True Logic Correction\n(Wrong Answer -> Right)']
sizes = [rescued_count, corrected_count]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 14})
plt.title("Net Gain Composition:\nBreaking the Overthinking Trap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_2_net_gain.png", dpi=300)
plt.close()
print("Saved: plot_2_net_gain.png")

# ================================
# 图 3: 唯一真·纠错样本 (保持不变)
# ================================
target_id = "test/number_theory/357.json"
sample = dynamic_probs.get(target_id)
base_sample = baseline_probs.get(target_id)

if sample and sample.get('intervention_start') is not None and base_sample:
    fig, ax3 = plt.subplots(figsize=(12, 6))
    t0 = sample['intervention_start']
    window_start = max(0, t0 - 15)
    window_end = min(len(sample['ema_trajectory']), t0 + 40)
    x_axis = np.arange(window_start - t0, window_end - t0)

    ema_window = sample['ema_trajectory'][window_start:window_end]
    alpha_window = sample['alpha_trajectory'][window_start:window_end]

    base_ema_window = [base_sample['ema_trajectory'][i] if i < len(base_sample['ema_trajectory']) else np.nan 
                       for i in range(window_start, window_end)]

    line1, = ax3.plot(x_axis, base_ema_window, color='gray', linestyle='--', linewidth=2.5, label='Baseline EMA (Wrong: 63)')
    line2, = ax3.plot(x_axis, ema_window, color='#d62728', linewidth=3, label='Dynamic EMA (Sensor)')
    ax3.set_ylabel('EMA Entropy', color='#d62728', fontsize=14)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, label='T=0 (Intervention)')
    ax3.set_xlabel("Tokens relative to T0", fontsize=14)

    ax3_twin = ax3.twinx()
    line3, = ax3_twin.plot(x_axis, alpha_window, color='#1f77b4', linewidth=3, label='Alpha Intensity (Actuator)')
    ax3_twin.set_ylabel('Alpha Intensity', color='#1f77b4', fontsize=14)
    
    ax3.legend(handles=[line1, line2, line3], loc='upper left', fontsize=11, framealpha=0.9)
    plt.title(f"Micro-Dynamics of True Logic Correction\n(Case: 357.json)", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plot_3_case_study.png", dpi=300)
    plt.close()
    print("Saved: plot_3_case_study.png")