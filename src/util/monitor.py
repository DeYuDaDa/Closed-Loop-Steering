import json
import numpy as np

# 1. 加载您的真实数据
file_path = r"F:\academic\dynamic-spherical-result\MATH500_4k-045\experiment_results_fixed.json"
print(f"Loading data from {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

baseline_probs = {p['id']: p for p in data['Baseline']['per_problem']}
dynamic_probs = {p['id']: p for p in data['Dynamic_Spherical']['per_problem']}
continuous_probs = {p['id']: p for p in data['Continuous']['per_problem']}

# ==========================================
# 任务一：精准分类的答案对照提取 (Logic Flips & Broken)
# ==========================================
print("\n" + "="*50)
print("Task 1: Exact Outcome Comparison (Net Gain vs Broken)")
print("="*50)

net_gain_cases = []  # Baseline 错，Dynamic 对
broken_cases = []    # Baseline 对，Dynamic 错
both_wrong_diff_ans = [] # 俩都错，但给出了不同的错误答案

for pid, b_prob in baseline_probs.items():
    if pid in dynamic_probs:
        d_prob = dynamic_probs[pid]
        
        b_corr = b_prob.get('correct', False)
        d_corr = d_prob.get('correct', False)
        b_pred = b_prob.get('predicted')
        d_pred = d_prob.get('predicted')
        expected = b_prob.get('expected')

        # 构建统一的样本卡片
        sample_card = {
            "id": pid,
            "expected_answer": expected,
            "baseline_pred": b_pred,
            "dynamic_pred": d_pred
        }

        # 1. 抓取正向净收益 (Net Gain)
        if not b_corr and d_corr:
            net_gain_cases.append(sample_card)
            
        # 2. 抓取负面干扰 (Broken)
        elif b_corr and not d_corr:
            broken_cases.append(sample_card)
            
        # 3. 抓取“殊途同归的错” (用于观察干预是如何改变错误轨迹的)
        elif not b_corr and not d_corr and str(b_pred) != str(d_pred):
            both_wrong_diff_ans.append(sample_card)

print(f"Total Net Gain (Baseline Wrong -> Dynamic Right): {len(net_gain_cases)}")
print(f"Total Broken (Baseline Right -> Dynamic Wrong): {len(broken_cases)}")
print(f"Total Both Wrong but Divergent Paths: {len(both_wrong_diff_ans)}")

# 分别打包输出，方便你做 Qualitative Analysis (定性分析)
with open("cases_net_gain.json", 'w', encoding='utf-8') as f:
    json.dump(net_gain_cases, f, indent=4, ensure_ascii=False)
    
with open("cases_broken.json", 'w', encoding='utf-8') as f:
    json.dump(broken_cases, f, indent=4, ensure_ascii=False)

print("\n[!] Data successfully exported to 'cases_net_gain.json' and 'cases_broken.json'")


# ==========================================
# 任务二：EMA 与 DTR 全量统计分布分析
# ==========================================
print("\n" + "="*50)
print("Task 2: EMA and DTR Statistical Distributions")
print("="*50)

def calculate_stats(arr):
    if not arr or len(arr) == 0: 
        return None
    return {
        "mean": np.mean(arr), 
        "std": np.std(arr),
        "min": np.min(arr), 
        "25%": np.percentile(arr, 25),
        "median": np.median(arr), 
        "75%": np.percentile(arr, 75),
        "max": np.max(arr)
    }

# 设定 DTR 深度阈值 (36 * 0.85 = 31)
DTR_THRESHOLD = 31 
stats_results = {}

for model_name, probs_dict in [("Baseline", baseline_probs),
                               ("Continuous", continuous_probs),
                               ("Dynamic_Spherical", dynamic_probs)]:
    all_ema_tokens = []
    all_local_dtr = []
    deep_thinking_ratios = [] # 每道题中，深度>=31的Token占比

    for pid, prob in probs_dict.items():
        # 收集所有的 EMA
        if 'ema_trajectory' in prob and prob['ema_trajectory']:
            # 过滤掉可能的 NaN 值
            clean_ema = [e for e in prob['ema_trajectory'] if not np.isnan(e)]
            all_ema_tokens.extend(clean_ema)
            
        # 收集宏观的 local_dtr
        if 'local_dtr' in prob and prob['local_dtr'] is not None and not np.isnan(prob['local_dtr']):
            all_local_dtr.append(prob['local_dtr'])
            
        # 根据 1-36 的离散深度，计算微观的“深思比例”
        if 'dtr_trajectory' in prob and prob['dtr_trajectory']:
            dtr_traj = np.array(prob['dtr_trajectory'])
            # 计算该题目中有多少比例的 Token 收敛层数 >= 31
            dt_ratio = np.mean(dtr_traj >= DTR_THRESHOLD)
            deep_thinking_ratios.append(dt_ratio)

    stats_results[model_name] = {
        "EMA (Token-level)": calculate_stats(all_ema_tokens),
        "Local DTR (Problem-level)": calculate_stats(all_local_dtr),
        "Deep Thinking Token Ratio (>= 31 Layers)": calculate_stats(deep_thinking_ratios)
    }

# 格式化输出统计结果
for model, metrics in stats_results.items():
    print(f"\n>>> [{model}]")
    for metric_name, stats in metrics.items():
        if stats is None:
            print(f"  - {metric_name}: No valid data")
        else:
            print(f"  - {metric_name}:")
            print(f"      Mean: {stats['mean']:.4f}  |  Std: {stats['std']:.4f}  |  Median: {stats['median']:.4f}")
            print(f"      Min:  {stats['min']:.4f}  |  25%: {stats['25%']:.4f}  |  75%: {stats['75%']:.4f}  |  Max: {stats['max']:.4f}")

print("\n" + "="*50)
print("Statistical Analysis Complete.")