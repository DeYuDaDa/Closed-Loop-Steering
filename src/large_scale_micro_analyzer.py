import json
import argparse
import os
import re  # 新增：用于Task2的正则过滤
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.ticker import MaxNLocator

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

class LargeScaleMicroAnalyzer:
    def __init__(self, json_path, model_path="Qwen/Qwen2.5-Math-7B"):
        self.json_path = json_path
        self.output_dir = os.path.dirname(json_path)
        self.model_path = model_path
        
        print(f"Loading dataset from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        self.baseline = self.data.get("Baseline", {}).get("per_problem", [])
        self.continuous = self.data.get("Continuous", {}).get("per_problem", [])
        self.dynamic = self.data.get("Dynamic_Spherical", {}).get("per_problem", [])
        
        # Create ID to problem mappings for easy access
        self.baseline_map = {p["id"]: p for p in self.baseline}
        self.continuous_map = {p["id"]: p for p in self.continuous}
        self.dynamic_map = {p["id"]: p for p in self.dynamic}
        
        print(f"Loaded {len(self.baseline)} Baseline, {len(self.continuous)} Continuous, and {len(self.dynamic)} Dynamic Spherical trajectories.")
        
        # Set plotting style
        plt.style.use("default") # Reset first
        sns.set_theme(style="whitegrid")
        
        # 新增：初始化tokenizer（简化后续调用）
        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                print(f"Failed to load tokenizer from {self.model_path}: {e}")
        
    def plot_stratified_dynamics(self):
        print("\n--- Running Task 1: Stratified Event-Aligned Dynamics ---")
        fixed_trajectories = []
        stubborn_trajectories = []
        # 新增：存储Baseline和Continuous的对比轨迹
        fixed_baseline_trajectories = []
        stubborn_baseline_trajectories = []
        fixed_continuous_trajectories = []
        stubborn_continuous_trajectories = []
        
        for ds_prob in self.dynamic:
            prob_id = ds_prob["id"]
            if prob_id not in self.baseline_map:
                continue
            
            # 获取三个组的对应数据
            base_prob = self.baseline_map[prob_id]
            cont_prob = self.continuous_map.get(prob_id)
            alpha_traj = ds_prob.get("alpha_trajectory", [])
            entropy_traj = ds_prob.get("entropy_trajectory", []) or ds_prob.get("ema_trajectory", [])
            
            # 获取Baseline和Continuous的熵轨迹
            base_entropy_traj = base_prob.get("ema_trajectory", []) or base_prob.get("entropy_trajectory", [])
            cont_entropy_traj = cont_prob.get("ema_trajectory", []) or cont_prob.get("entropy_trajectory", []) if cont_prob else []
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            is_fixed = (not base_prob["correct"]) and ds_prob["correct"]
            is_stubborn = (not base_prob["correct"]) and (not ds_prob["correct"])
            
            if not is_fixed and not is_stubborn:
                continue
                
            # 对齐到Alpha峰值
            t_peak = np.argmax(alpha_traj)
            start_idx = t_peak - 20
            end_idx = t_peak + 30
            
            # 提取Dynamic窗口
            window = np.full(50, np.nan)
            traj_start = max(0, start_idx)
            traj_end = min(len(entropy_traj), end_idx)
            win_start = max(0, -start_idx)
            win_end = win_start + (traj_end - traj_start)
            if traj_end > traj_start:
                window[win_start:win_end] = entropy_traj[traj_start:traj_end]
            
            # 提取Baseline窗口
            base_window = np.full(50, np.nan)
            if base_entropy_traj:
                base_traj_start = max(0, start_idx)
                base_traj_end = min(len(base_entropy_traj), end_idx)
                base_win_start = max(0, -start_idx)
                base_win_end = base_win_start + (base_traj_end - base_traj_start)
                if base_traj_end > base_traj_start:
                    base_window[base_win_start:base_win_end] = base_entropy_traj[base_traj_start:base_traj_end]
            
            # 提取Continuous窗口
            cont_window = np.full(50, np.nan)
            if cont_entropy_traj and cont_prob:
                cont_traj_start = max(0, start_idx)
                cont_traj_end = min(len(cont_entropy_traj), end_idx)
                cont_win_start = max(0, -start_idx)
                cont_win_end = cont_win_start + (cont_traj_end - cont_traj_start)
                if cont_traj_end > cont_traj_start:
                    cont_window[cont_win_start:cont_win_end] = cont_entropy_traj[cont_traj_start:cont_traj_end]
            
            # 分类存储
            if is_fixed:
                fixed_trajectories.append(window)
                fixed_baseline_trajectories.append(base_window)
                if cont_prob:
                    fixed_continuous_trajectories.append(cont_window)
            elif is_stubborn:
                stubborn_trajectories.append(window)
                stubborn_baseline_trajectories.append(base_window)
                if cont_prob:
                    stubborn_continuous_trajectories.append(cont_window)
                
        print(f"Found {len(fixed_trajectories)} Fixed and {len(stubborn_trajectories)} Stubborn trajectories.")
        
        if not fixed_trajectories and not stubborn_trajectories:
            print("Not enough data for plot_stratified_dynamics. Skipping.")
            return
            
        plt.figure(figsize=(12, 7))
        x_axis = np.arange(-20, 30)  # X=0对应Alpha峰值
        
        # 绘制Fixed类轨迹（Success）
        if fixed_trajectories:
            fixed_arr = np.nanmean(np.array(fixed_trajectories), axis=0)
            plt.plot(x_axis, fixed_arr, label="Fixed - Dynamic", color="green", linewidth=2.5)
            
            # Baseline对比
            if fixed_baseline_trajectories:
                fixed_base_arr = np.nanmean(np.array(fixed_baseline_trajectories), axis=0)
                plt.plot(x_axis, fixed_base_arr, label="Fixed - Baseline", color="green", linewidth=2, linestyle="--", alpha=0.8)
            
            # Continuous对比
            if fixed_continuous_trajectories:
                fixed_cont_arr = np.nanmean(np.array(fixed_continuous_trajectories), axis=0)
                plt.plot(x_axis, fixed_cont_arr, label="Fixed - Continuous", color="green", linewidth=2, linestyle=":", alpha=0.9)
        
        # 绘制Stubborn类轨迹（Failed）
        if stubborn_trajectories:
            stubborn_arr = np.nanmean(np.array(stubborn_trajectories), axis=0)
            plt.plot(x_axis, stubborn_arr, label="Stubborn - Dynamic", color="red", linewidth=2.5)
            
            # Baseline对比
            if stubborn_baseline_trajectories:
                stubborn_base_arr = np.nanmean(np.array(stubborn_baseline_trajectories), axis=0)
                plt.plot(x_axis, stubborn_base_arr, label="Stubborn - Baseline", color="red", linewidth=2, linestyle="--", alpha=0.8)
            
            # Continuous对比
            if stubborn_continuous_trajectories:
                stubborn_cont_arr = np.nanmean(np.array(stubborn_continuous_trajectories), axis=0)
                plt.plot(x_axis, stubborn_cont_arr, label="Stubborn - Continuous", color="red", linewidth=2, linestyle=":", alpha=0.9)
            
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Intervention Peak ($T=0$)")
        
        plt.title("Stratified Event-Aligned Dynamics (Aligned to Peak Alpha)", fontsize=16, pad=15)
        plt.xlabel("Tokens relative to Peak Intervention ($T-T_{peak}$)", fontsize=14)
        plt.ylabel("Mean EMA Entropy", fontsize=14)
        plt.legend(fontsize=10, loc="best")
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task1_stratified_dynamics.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def plot_pivot_tokens(self):
        print("\n--- Running Task 2: Automated Pivot Token Mining ---")
        if self.tokenizer is None:
            print("transformers library not found or tokenizer load failed. Skipping plot_pivot_tokens.")
            return
            
        pivot_words = []
        # 定义思维链的特殊token ID（固定值）
        THOUGHT_OPEN_ID = 151667
        THOUGHT_CLOSE_ID = 151668

        for ds_prob in self.dynamic:
            alpha_traj = ds_prob.get("alpha_trajectory", [])
            output_ids = ds_prob.get("output_ids", [])
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            # 步骤1：定位思维链<thought>和</thought>的位置
            thought_start_idx = -1
            thought_end_idx = -1
            # 找<thought>的起始位置
            for idx, token_id in enumerate(output_ids):
                if token_id == THOUGHT_OPEN_ID:
                    thought_start_idx = idx + 1  # 跳过<thought>本身，取其后的内容
                    break
            # 若找到<thought>，再找</thought>的结束位置
            if thought_start_idx != -1:
                for idx in range(thought_start_idx, len(output_ids)):
                    if output_ids[idx] == THOUGHT_CLOSE_ID:
                        thought_end_idx = idx  # 不包含</thought>本身
                        break
                # 兼容截断场景：无</thought>则取到output_ids末尾
                if thought_end_idx == -1:
                    thought_end_idx = len(output_ids)
            
            # 步骤2：仅处理有思维链的情况
            if thought_start_idx == -1 or thought_start_idx >= thought_end_idx:
                continue  # 无<thought>标签，跳过该样本
                
            # 步骤3：计算Alpha峰值位置，并确保其在思维链范围内
            t_peak_relative = np.argmax(alpha_traj)
            input_len = ds_prob.get("input_len", 0)
            t_peak_absolute = t_peak_relative + input_len
            
            # 若峰值超出思维链范围，取思维链中间位置（兜底）
            if t_peak_absolute < thought_start_idx or t_peak_absolute >= thought_end_idx:
                t_peak_absolute = (thought_start_idx + thought_end_idx) // 2
            
            # 提取范围：t_peak前后各2个token（共5个），且不超出思维链边界
            slice_start = max(thought_start_idx, t_peak_absolute - 2)
            slice_end = min(thought_end_idx, t_peak_absolute + 3)  # 左闭右开，所以+3
            if slice_end - slice_start < 1:
                continue  # 无有效token，跳过
                
            slice_ids = output_ids[slice_start:slice_end]
            try:
                # 解码思维链内的token片段，避免系统Prompt干扰
                decoded = self.tokenizer.decode(slice_ids, clean_up_tokenization_spaces=True)
                cleaned = decoded.strip().replace("\n", " ").replace("\r", "")
                # 过滤≥3个字母的英文单词（保留核心语义）
                words = re.findall(r'\b[A-Za-z]{3,}\b', cleaned)
                pivot_words.extend([w.lower() for w in words])
            except Exception as e:
                print(f"Token decode error for problem {ds_prob.get('id', 'unknown')}: {e}")
                pass
                
        counter = Counter(pivot_words)
        top_20 = counter.most_common(20)
        
        print(f"Mined {len(pivot_words)} pivot words (only in <thought>). Top 5: {top_20[:5]}")
        
        if not top_20:
            print("No pivot tokens found in <thought> tags. Skipping plot.")
            return
            
        labels = [item[0] for item in top_20]
        counts = [item[1] for item in top_20]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=counts, y=labels, palette="viridis")
        
        plt.title("Automated Pivot Token Mining (Only <thought> Content)", fontsize=16, pad=15)
        plt.xlabel("Frequency", fontsize=14)
        plt.ylabel("Filtered English Words (≥3 Letters) in <thought>", fontsize=14)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task2_pivot_tokens.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_divergence_anatomy(self):
        print("\n--- Running Task 3: Divergence Point Anatomy ---")
        # 存储三个组的分歧点熵
        baseline_emas = []
        dynamic_emas = []
        continuous_emas = []
        
        for pid, p_ds in self.dynamic_map.items():
            p_base = self.baseline_map.get(pid)
            p_cont = self.continuous_map.get(pid)
            
            # 过滤条件：仅保留“Baseline错、DS对”的样本
            if not p_base or p_base["correct"] or not p_ds["correct"]:
                continue
                
            ids_base = p_base.get("output_ids", [])
            ids_ds = p_ds.get("output_ids", [])
            ids_cont = p_cont.get("output_ids", []) if p_cont else []
            
            # 获取三个组的EMA轨迹
            base_ema = p_base.get("ema_trajectory", [])
            ds_ema = p_ds.get("ema_trajectory", [])
            cont_ema = p_cont.get("ema_trajectory", []) if p_cont else []
            
            if not ids_base or not ids_ds:
                continue
                
            # 计算分歧点
            t_div = -1
            min_len = min(len(ids_base), len(ids_ds))
            for i in range(min_len):
                if ids_base[i] != ids_ds[i]:
                    t_div = i
                    break
            if t_div == -1 and len(ids_base) != len(ids_ds):
                t_div = min_len
                
            # 收集三个组在分歧点的熵
            if t_div != -1:
                gen_t_div = t_div - p_base.get("input_len", 0)
                if gen_t_div >= 0:
                    # Baseline熵
                    if gen_t_div < len(base_ema):
                        baseline_emas.append(base_ema[gen_t_div])
                    # Dynamic熵
                    if gen_t_div < len(ds_ema):
                        dynamic_emas.append(ds_ema[gen_t_div])
                    # Continuous熵（需确保有对应数据且分歧点有效）
                    if p_cont and gen_t_div < len(cont_ema) and gen_t_div < (len(ids_cont) - p_cont.get("input_len", 0)):
                        continuous_emas.append(cont_ema[gen_t_div])
                    
        print(f"Found divergence point entropy - Baseline: {len(baseline_emas)}, Dynamic: {len(dynamic_emas)}, Continuous: {len(continuous_emas)}")
        
        if not baseline_emas and not dynamic_emas and not continuous_emas:
            print("No divergence point entropy data found. Skipping plot.")
            return
            
        plt.figure(figsize=(10, 6))
        # 绘制三个组的熵分布
        if baseline_emas:
            sns.histplot(baseline_emas, kde=True, bins=20, color="green", alpha=0.5, label="Baseline", stat="density")
        if dynamic_emas:
            sns.histplot(dynamic_emas, kde=True, bins=20, color="blue", alpha=0.5, label="Dynamic Spherical", stat="density")
        if continuous_emas:
            sns.histplot(continuous_emas, kde=True, bins=20, color="orange", alpha=0.5, label="Continuous", stat="density")
        
        # 干预阈值线
        threshold = 0.15
        plt.axvline(x=threshold, color="black", linestyle="--", label=f"Intervention Threshold ({threshold})")
        
        plt.title("Entropy at the Exact Moment of Divergence (Three Groups Comparison)", fontsize=14)
        plt.xlabel("EMA Entropy at Divergence Point", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task3_divergence_anatomy.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def plot_thinkbrake_boundary(self):
        print("\n--- Running Task 4: ThinkBrake Boundary Mapping ---")
        tokens_remaining = []
        
        for ds_prob in self.dynamic:
            alpha_traj = ds_prob.get("alpha_trajectory", [])
            output_ids = ds_prob.get("output_ids", [])
            convergence = ds_prob.get("convergence", False)
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            # Identify if it dropped to 0 and stayed 0
            if convergence:
                t_peak = np.argmax(alpha_traj)
                t_brake = -1
                for i in range(t_peak, len(alpha_traj)):
                    if alpha_traj[i] <= 1e-5:
                        if all(a <= 1e-5 for a in alpha_traj[i:]):
                            t_brake = i
                            break
                            
                if t_brake != -1:
                    rem = len(alpha_traj) - t_brake
                    tokens_remaining.append(rem)
                    
        print(f"Found {len(tokens_remaining)} ThinkBrake activations.")
        
        if not tokens_remaining:
            print("No ThinkBrake activations found. Skipping plot.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(tokens_remaining, bins=50, color="teal", kde=True)
        
        plt.title("ThinkBrake Physical Boundary Mapping", fontsize=16, pad=15)
        plt.xlabel("Tokens Remaining Output Since ThinkBrake Triggered", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlim(left=0)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task4_thinkbrake_boundary.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def plot_energy_shock(self, continuous_alpha_val=0.45):
        print("\n--- Running Task 5: 'State Shock' Energy Scatter Plot ---")
        # 存储三个组的数据
        base_E, base_R = [], []    # Baseline（无干预，能量=0）
        cont_E, cont_R = [], []    # Continuous
        dyn_E, dyn_R = [], []      # Dynamic Spherical
        
        # 处理Baseline
        for p in self.baseline:
            rep = p.get("repetition", 0.0) * 100
            base_E.append(0.0)  # Baseline无干预，能量恒为0
            base_R.append(max(0, rep))
        
        # 处理Continuous
        for p in self.continuous:
            ema = p.get("ema_trajectory", [])
            out_ids = p.get("output_ids", [])
            valid_len = len(ema) if ema else len(out_ids)
            
            energy = continuous_alpha_val * valid_len  # 固定alpha × 步数
            rep = p.get("repetition", 0.0) * 100
            
            cont_E.append(energy)
            cont_R.append(max(0, rep))
            
        # 处理Dynamic
        for p in self.dynamic:
            alpha_traj = p.get("alpha_trajectory", [])
            rep = p.get("repetition", 0.0) * 100
            if alpha_traj:
                dyn_E.append(sum(alpha_traj))
                dyn_R.append(max(0, rep))
                
        print(f"Mapping - Baseline: {len(base_E)}, Continuous: {len(cont_E)}, Dynamic: {len(dyn_E)}")
        
        if not base_E and not cont_E and not dyn_E:
            print("No valid energy-repetition data found. Skipping plot.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 绘制三个组的散点和密度图
        if base_E:
            sns.kdeplot(x=base_E, y=base_R, fill=True, color="green", alpha=0.2, levels=5, label="Baseline (KDE)")
            plt.scatter(base_E, base_R, color="green", alpha=0.5, s=20, label="Baseline")
            
        if cont_E:
            sns.kdeplot(x=cont_E, y=cont_R, fill=True, color="red", alpha=0.2, levels=5, label="Continuous (KDE)")
            plt.scatter(cont_E, cont_R, color="red", alpha=0.5, s=20, label="Continuous")
            
        if dyn_E:
            sns.kdeplot(x=dyn_E, y=dyn_R, fill=True, color="blue", alpha=0.2, levels=5, label="Dynamic Spherical (KDE)")
            plt.scatter(dyn_E, dyn_R, color="blue", alpha=0.5, s=20, label="Dynamic Spherical")
            
        plt.title("Energy-Shock Dimensionality Scatter (Three Groups Comparison)", fontsize=16, pad=15)
        plt.xlabel("Total Intervention Energy ($\sum \\alpha_t$, 0 for Baseline)", fontsize=14)
        plt.ylabel("N-Gram Repetition Rate (%)", fontsize=14)
        plt.legend(fontsize=11, loc="best")
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task5_energy_shock.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def run_all(self, continuous_alpha_val=0.45):
        self.plot_stratified_dynamics()
        self.plot_pivot_tokens()
        self.plot_divergence_anatomy()
        self.plot_thinkbrake_boundary()
        self.plot_energy_shock(continuous_alpha_val)
        print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability Micro-Analyzer for Steering")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the experiment outputs json file")
    parser.add_argument("--model_path", type=str, default=r"F:\academic\Closed-Loop-Steering-System\src\tokenizer", help="HuggingFace model path for tokenizer")
    parser.add_argument("--continuous_alpha", type=float, default=0.45, help="Alpha constant used in Continuous mode (for energy calculation)")
    args = parser.parse_args()
    
    analyzer = LargeScaleMicroAnalyzer(args.json_path, args.model_path)
    analyzer.run_all(args.continuous_alpha)