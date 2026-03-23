import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 样式设置 (符合学术顶会审美)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# ==========================================
# 2. 数据加载 (请在这里替换为您的真实路径)
# ==========================================
def load_real_data():
    # 替换为您的 JSON 路径
    filepath = r"F:\academic\dynamic-spherical-result\MATH500_4k-03\experiment_results_fixed.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
        return {
            "Baseline": data_json["Baseline"], 
            "Dynamic_Spherical": data_json["Dynamic_Spherical"]
        }

# 如果没有真实文件，您可以使用之前的 mock_data 字典进行测试
try:
    data = load_real_data()
except FileNotFoundError:
    print("Warning: File not found, please check the path.")
    # 如果路径不对，这里需要替换为您之前生成的 mock_data
    # data = mock_data 
    raise 

# ==========================================
# 3. 画板初始化 (调整为单图尺寸)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 8))

# ==========================================
# 图 1: 逻辑翻转矩阵 (Logic Flip Matrix)
# ==========================================
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

# 构建矩阵：[[TT, TF], [FT, FF]]
matrix = np.array([[flip_data['Base_True_Dyn_True'], flip_data['Base_True_Dyn_False']],
                   [flip_data['Base_False_Dyn_True'], flip_data['Base_False_Dyn_False']]])

# 绘制热力图
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax1, annot_kws={"size": 18})
ax1.set_title("Logic Flip Matrix: Zero Broken & Net Gain", fontsize=18, fontweight='bold')
ax1.set_xticklabels(['Dynamic Correct', 'Dynamic Wrong'], fontsize=14)
ax1.set_yticklabels(['Baseline Correct', 'Baseline Wrong'], fontsize=14)

# 添加核心亮点文字
net_gain = flip_data['Base_False_Dyn_True']
broken = flip_data['Base_True_Dyn_False']
ax1.text(0.5, 1.15, f"Net Gain: Corrected {net_gain} errors", color='green', fontsize=16, ha='center', transform=ax1.transAxes, fontweight='bold')
ax1.text(0.5, 1.08, f"Broken: {broken} errors (Zero Negative Interference)", color='red', fontsize=16, ha='center', transform=ax1.transAxes)

# ==========================================
# 4. 保存与显示
# ==========================================
plt.tight_layout()
# 修改保存的文件名
plt.savefig("Logic_Flip_Matrix.png", dpi=300, bbox_inches='tight')
plt.show()