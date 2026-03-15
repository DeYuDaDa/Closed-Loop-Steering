import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ==========================================
# 替换为你的实际 JSON 路径
JSON_PATH = "/root/Closed-Loop-Steering-System/src/results/MATH500_40_20260315_221725/experiment_results.json"
# ==========================================

def load_data(path):
    print(f"Loading JSON from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_deep_analysis(data):
    if "Dynamic_Spherical" not in data or "Baseline" not in data:
        print("❌ 缺少必要的实验组数据 (Baseline 或 Dynamic_Spherical)")
        return

    ds_data = data["Dynamic_Spherical"]
    base_data = data["Baseline"]

    # ✅ 修复核心：从 per_problem 数组中提取每一道题的具体轨迹，而不是用外层被拍扁的全局列表
    ds_problems = ds_data.get("per_problem", [])
    base_problems = base_data.get("per_problem", [])

    print("\n" + "="*50)
    print("🚀 DEEP ANALYSIS REPORT (FIXED) 🚀")
    print("="*50)

    # ---------------------------------------------------------
    # 1. 触发烈度分析 (Intervention Energy)
    # ---------------------------------------------------------
    triggered_count = 0
    max_alphas = []
    total_energies = []

    # 用于绘图的有效轨迹容器
    valid_ema_trajs = []
    valid_alpha_trajs = []

    for p in ds_problems:
        alpha = p.get("alpha_trajectory", [])
        ema = p.get("ema_trajectory", [])
        
        # 确保这道题记录了合法的轨迹数组
        if isinstance(alpha, list) and len(alpha) > 0:
            valid_ema_trajs.append(ema)
            valid_alpha_trajs.append(alpha)
            
            max_a = max(alpha)
            if max_a > 0.05:
                triggered_count += 1
                max_alphas.append(max_a)
                total_energies.append(sum(alpha))
                
    print(f"\n📊 1. 触发烈度统计")
    print(f"  - 总样本数 (per_problem): {len(ds_problems)}")
    print(f"  - 真实触发干预的样本数 (Max α > 0.05): {triggered_count}")
    if max_alphas:
        print(f"  - 触发样本的平均最大 α: {np.mean(max_alphas):.3f}")
        print(f"  - 触发样本的平均干预总能量 (Σα): {np.mean(total_energies):.1f}")

    # ---------------------------------------------------------
    # 2. T0 事件对齐动力学 (Event-Aligned Dynamics)
    # ---------------------------------------------------------
    WINDOW_PRE = 20
    WINDOW_POST = 30
    aligned_ema = []
    aligned_alpha = []

    for ema, alpha in zip(valid_ema_trajs, valid_alpha_trajs):
        # 过滤过短或没有明显干预的无效数据
        if len(ema) < 10 or not alpha or max(alpha) <= 0.05: 
            continue
        
        # 寻找首次突破 0.05 的索引 T0
        t0 = next((i for i, a in enumerate(alpha) if a > 0.05), -1)
        # 抛弃两头不靠的数据（保证前后有足够长的窗口画图）
        if t0 == -1 or t0 < WINDOW_PRE or t0 + WINDOW_POST > len(ema):
            continue 
            
        aligned_ema.append(ema[t0 - WINDOW_PRE : t0 + WINDOW_POST])
        aligned_alpha.append(alpha[t0 - WINDOW_PRE : t0 + WINDOW_POST])

    if aligned_ema:
        mean_ema = np.mean(aligned_ema, axis=0)
        mean_alpha = np.mean(aligned_alpha, axis=0)
        
        print(f"\n📊 2. T0 事件对齐分析 (对齐了 {len(aligned_ema)} 个有效波峰)")
        print(f"  - 注入前 (T-5) 熵均值: {np.mean(mean_ema[WINDOW_PRE-5:WINDOW_PRE]):.3f}")
        print(f"  - 注入最高点熵均值: {np.max(mean_ema):.3f}")
        print(f"  - 注入后 (T+10) 熵均值: {np.mean(mean_ema[WINDOW_PRE+10:WINDOW_PRE+15]):.3f}")
        
        # 绘制 T0 对齐图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x_axis = np.arange(-WINDOW_PRE, WINDOW_POST)
        
        ax1.plot(x_axis, mean_ema, 'b-', linewidth=3, label="Mean EMA Entropy")
        ax1.set_xlabel("Tokens relative to Intervention Trigger (T=0)", fontsize=12)
        ax1.set_ylabel("EMA Entropy", color='b', fontsize=12)
        ax1.axvline(0, color='k', linestyle='--', alpha=0.5, label="Intervention Triggered")
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(x_axis, mean_alpha, 'r-', linewidth=3, label="Mean Alpha Strength")
        ax2.set_ylabel("Alpha", color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title("Event-Aligned Intervention Dynamics", fontsize=14)
        fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_plot = os.path.join(os.path.dirname(JSON_PATH), "T0_aligned_dynamics.png")
        plt.savefig(out_plot, dpi=300)
        print(f"  👉 T0 对齐图已生成: {out_plot}")
    else:
        print("\n📊 2. T0 事件对齐分析: 触发样本的序列长度不够提取完整的 T0 窗口。")

    # ---------------------------------------------------------
    # 3. 翻转分析 (Flip Analysis) - 精确 ID 匹配
    # ---------------------------------------------------------
    if ds_problems and base_problems:
        # 构建 Baseline 字典: {题号id : 是否做对}
        base_correct_map = {p["id"]: p.get("correct", False) for p in base_problems}
        
        fixed_count = 0
        broken_count = 0
        both_correct = 0
        both_wrong = 0
        
        for dp in ds_problems:
            pid = dp["id"]
            d_correct = dp.get("correct", False)
            b_correct = base_correct_map.get(pid, False)
            
            if not b_correct and d_correct:
                fixed_count += 1      # 救活了
            elif b_correct and not d_correct:
                broken_count += 1     # 搞坏了
            elif b_correct and d_correct:
                both_correct += 1
            else:
                both_wrong += 1

        print(f"\n📊 3. 逻辑翻转矩阵 (Flip Analysis)")
        print(f"  - 完美解决 (Baseline错 -> Dynamic对): {fixed_count} 题 🚀")
        print(f"  - 负面干扰 (Baseline对 -> Dynamic错): {broken_count} 题 ⚠️")
        print(f"  - 共同正确: {both_correct} 题")
        print(f"  - 共同错误: {both_wrong} 题")
        
        if fixed_count > broken_count:
            print(f"  💡 结论: 干预系统具有【正向净收益】 (+{fixed_count - broken_count}题)！")
        elif fixed_count < broken_count:
            print(f"  💡 结论: 干预系统具有【负向净收益】，说明注入引发了副作用（过度纯化或强度过高）。")
        else:
            print(f"  💡 结论: 收益与干扰抵消（准确率宏观不变的根源）。")

    print("="*50)

if __name__ == "__main__":
    data = load_data(JSON_PATH)
    run_deep_analysis(data)