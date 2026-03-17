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
        
        for ds_prob in self.dynamic:
            prob_id = ds_prob["id"]
            if prob_id not in self.baseline_map:
                continue
            
            base_prob = self.baseline_map[prob_id]
            alpha_traj = ds_prob.get("alpha_trajectory", [])
            entropy_traj = ds_prob.get("entropy_trajectory", [])  # 或ema_trajectory，根据JSON字段调整
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            is_fixed = (not base_prob["correct"]) and ds_prob["correct"]
            is_stubborn = (not base_prob["correct"]) and (not ds_prob["correct"])
            
            if not is_fixed and not is_stubborn:
                continue
                
            # 核心修改：对齐到Alpha峰值（而非首次>0.05）
            t_peak = np.argmax(alpha_traj)
            start_idx = t_peak - 20
            end_idx = t_peak + 30
            
            # 窗口提取逻辑（保持不变，仅对齐点改变）
            window = np.full(50, np.nan)
            traj_start = max(0, start_idx)
            traj_end = min(len(entropy_traj), end_idx)
            win_start = max(0, -start_idx)
            win_end = win_start + (traj_end - traj_start)
            
            if traj_end > traj_start:
                window[win_start:win_end] = entropy_traj[traj_start:traj_end]
                
            if is_fixed:
                fixed_trajectories.append(window)
            elif is_stubborn:
                stubborn_trajectories.append(window)
                
        print(f"Found {len(fixed_trajectories)} Fixed and {len(stubborn_trajectories)} Stubborn trajectories.")
        
        if not fixed_trajectories and not stubborn_trajectories:
            print("Not enough data for plot_stratified_dynamics. Skipping.")
            return
            
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(-20, 30)  # X=0对应Alpha峰值
        
        if fixed_trajectories:
            fixed_arr = np.nanmean(np.array(fixed_trajectories), axis=0)
            plt.plot(x_axis, fixed_arr, label="Fixed (Success)", color="green", linewidth=2.5)
            
        if stubborn_trajectories:
            stubborn_arr = np.nanmean(np.array(stubborn_trajectories), axis=0)
            plt.plot(x_axis, stubborn_arr, label="Stubborn (Failed)", color="red", linewidth=2.5)
            
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Intervention Peak ($T=0$)")  # 注释更新
        
        plt.title("Stratified Event-Aligned Dynamics (Aligned to Peak Alpha)", fontsize=16, pad=15)  # 标题更新
        plt.xlabel("Tokens relative to Peak Intervention ($T-T_{peak}$)", fontsize=14)  # X轴注释更新
        plt.ylabel("Mean EMA Entropy", fontsize=14)
        plt.legend(fontsize=12)
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
            t_peak = np.argmax(alpha_traj)
            # 若峰值超出思维链范围，取思维链中间位置（兜底）
            if t_peak < thought_start_idx or t_peak >= thought_end_idx:
                t_peak = (thought_start_idx + thought_end_idx) // 2
            
            # 步骤4：提取峰值附近的token片段（限定在思维链内）
            # 提取范围：t_peak前后各2个token（共5个），且不超出思维链边界
            slice_start = max(thought_start_idx, t_peak - 2)
            slice_end = min(thought_end_idx, t_peak + 3)  # 左闭右开，所以+3
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
        divergence_emas = []  # 重命名：更贴合DS的ema_trajectory
        
        for pid, p_ds in self.dynamic_map.items():
            p_base = self.baseline_map.get(pid)
            # 核心修改1：过滤条件调整（仅保留“Baseline错、DS对”的样本）
            if not p_base or p_base["correct"] or not p_ds["correct"]:
                continue
                
            ids_base = p_base.get("output_ids", [])
            ids_ds = p_ds.get("output_ids", [])
            # 核心修改2：读取DS的ema_trajectory（而非Baseline的entropy）
            ds_ema = p_ds.get("ema_trajectory", [])  # 或entropy_trajectory，根据JSON字段调整
            
            if not ids_base or not ids_ds:
                continue
                
            # 分歧点计算逻辑（保持不变）
            t_div = -1
            min_len = min(len(ids_base), len(ids_ds))
            for i in range(min_len):
                if ids_base[i] != ids_ds[i]:
                    t_div = i
                    break
            if t_div == -1 and len(ids_base) != len(ids_ds):
                t_div = min_len
                
            # 核心修改3：取DS在分歧点的熵
            if t_div != -1 and t_div < len(ds_ema):
                divergence_emas.append(ds_ema[t_div])
                    
        print(f"Found {len(divergence_emas)} divergence points with recorded DS entropy.")
        
        if not divergence_emas:
            print("Not enough DS entropy data at divergence points. Skipping plot.")
            return
            
        plt.figure(figsize=(8, 5))  # 调整画布尺寸更紧凑
        sns.histplot(divergence_emas, kde=True, bins=20, color="purple", alpha=0.6)
        
        threshold = 0.15
        plt.axvline(x=threshold, color="black", linestyle="--", label=f"Intervention Threshold ({threshold})")
        
        plt.title("Dynamic Entropy at the Exact Moment of Divergence", fontsize=14)  # 标题更新
        plt.xlabel("EMA Entropy (Dynamic Spherical)", fontsize=12)  # X轴注释更新
        plt.ylabel("Density / Count", fontsize=12)
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
                            
                if t_brake != -1 and len(output_ids) > t_brake:
                    rem = len(output_ids) - t_brake
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
        
    def plot_energy_shock(self):
        print("\n--- Running Task 5: 'State Shock' Energy Scatter Plot ---")
        cont_E, cont_R = [], []
        dyn_E, dyn_R = [], []
        
        for p in self.continuous:
            alpha_traj = p.get("alpha_trajectory", [])
            rep = p.get("repetition", 0.0)
            if alpha_traj:
                cont_E.append(sum(alpha_traj))
                cont_R.append(max(0, rep * 100))
                
        for p in self.dynamic:
            alpha_traj = p.get("alpha_trajectory", [])
            rep = p.get("repetition", 0.0)
            if alpha_traj:
                dyn_E.append(sum(alpha_traj))
                dyn_R.append(max(0, rep * 100))
                
        print(f"Mapping {len(cont_E)} Continuous and {len(dyn_E)} Dynamic pairs.")
        
        if not cont_E and not dyn_E:
            print("No valid energy-repetition data found. Skipping plot.")
            return
            
        plt.figure(figsize=(10, 8))
        
        if cont_E:
            sns.kdeplot(x=cont_E, y=cont_R, fill=True, color="red", alpha=0.3, levels=5)
            plt.scatter(cont_E, cont_R, color="red", alpha=0.5, s=20, label="Continuous")
            
        if dyn_E:
            sns.kdeplot(x=dyn_E, y=dyn_R, fill=True, color="blue", alpha=0.3, levels=5)
            plt.scatter(dyn_E, dyn_R, color="blue", alpha=0.5, s=20, label="Dynamic Spherical")
            
        plt.title("Energy-Shock Dimensionality Scatter", fontsize=16, pad=15)
        plt.xlabel("Total Intervention Energy ($\sum \\alpha_t$)", fontsize=14)
        plt.ylabel("N-Gram Repetition Rate (%)", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task5_energy_shock.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def run_all(self):
        self.plot_stratified_dynamics()
        self.plot_pivot_tokens()
        self.plot_divergence_anatomy()
        self.plot_thinkbrake_boundary()
        self.plot_energy_shock()
        print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability Micro-Analyzer for Steering")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the experiment outputs json file")
    parser.add_argument("--model_path", type=str, default=r"F:\academic\Closed-Loop-Steering-System\src\tokenizer", help="HuggingFace model path for tokenizer")
    args = parser.parse_args()
    
    analyzer = LargeScaleMicroAnalyzer(args.json_path, args.model_path)
    analyzer.run_all()