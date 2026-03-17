import json
import argparse
import os
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
            entropy_traj = ds_prob.get("entropy_trajectory", [])
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            is_fixed = (not base_prob["correct"]) and ds_prob["correct"]
            is_stubborn = (not base_prob["correct"]) and (not ds_prob["correct"])
            
            if not is_fixed and not is_stubborn:
                continue
                
            # Find T_0 (first index >= 0.05)
            # Alpha threshold is 0.05 for intervention
            try:
                t_0 = next(i for i, a in enumerate(alpha_traj) if a > 0.05)
            except StopIteration:
                continue
                
            # We want EMA entropy window [T_0 - 20, T_0 + 30]
            start_idx = t_0 - 20
            end_idx = t_0 + 30
            
            # Since trajectories might be shorter, we PAD with NaNs or just skip if we don't have enough right-side data
            # To be robust, we'll extract what we can and pad with NaN
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
        
        x_axis = np.arange(-20, 30)
        
        if fixed_trajectories:
            fixed_arr = np.nanmean(np.array(fixed_trajectories), axis=0)
            plt.plot(x_axis, fixed_arr, label="Fixed (Success)", color="green", linewidth=2.5)
            
        if stubborn_trajectories:
            stubborn_arr = np.nanmean(np.array(stubborn_trajectories), axis=0)
            plt.plot(x_axis, stubborn_arr, label="Stubborn (Failed)", color="red", linewidth=2.5)
            
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Intervention Trigger ($T=0$)")
        
        plt.title("Stratified Event-Aligned Dynamics", fontsize=16, pad=15)
        plt.xlabel("Tokens relative to Intervention ($T-T_0$)", fontsize=14)
        plt.ylabel("Mean EMA Entropy", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task1_stratified_dynamics.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def plot_pivot_tokens(self):
        print("\n--- Running Task 2: Automated Pivot Token Mining ---")
        if AutoTokenizer is None:
            print("transformers library not found. Skipping plot_pivot_tokens.")
            return
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Failed to load tokenizer from {self.model_path}: {e}")
            return
            
        pivot_phrases = []
        
        for ds_prob in self.dynamic:
            alpha_traj = ds_prob.get("alpha_trajectory", [])
            output_ids = ds_prob.get("output_ids", [])
            
            if not alpha_traj or max(alpha_traj) <= 0.05:
                continue
                
            t_peak = np.argmax(alpha_traj)
            
            # Ensure we have at least up to T_peak + 3
            if t_peak < len(output_ids):
                # Extract 1 token at peak, or up to 3 tokens to see the "phrase"
                slice_ids = output_ids[t_peak : min(t_peak + 3, len(output_ids))]
                try:
                    decoded = tokenizer.decode(slice_ids, clean_up_tokenization_spaces=True)
                    cleaned = decoded.strip().replace("\n", " ").replace("\r", "")
                    if cleaned:
                        pivot_phrases.append(cleaned)
                except Exception:
                    pass
                    
        counter = Counter(pivot_phrases)
        top_20 = counter.most_common(20)
        
        print(f"Mined {len(pivot_phrases)} pivot phrases. Top 5: {top_20[:5]}")
        
        if not top_20:
            print("No pivot tokens found. Skipping plot.")
            return
            
        labels = [item[0] for item in top_20]
        counts = [item[1] for item in top_20]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=counts, y=labels, palette="viridis")
        
        plt.title("Automated Pivot Token Mining (Top 20 @ Peak Alpha)", fontsize=16, pad=15)
        plt.xlabel("Frequency", fontsize=14)
        plt.ylabel("Decoded Tokens (Length 3)", fontsize=14)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "task2_pivot_tokens.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
    def plot_divergence_anatomy(self):
        print("\n--- Running Task 3: Divergence Point Anatomy ---")
        divergence_entropies = []
        
        for ds_prob in self.dynamic:
            prob_id = ds_prob["id"]
            if prob_id not in self.baseline_map:
                continue
                
            base_prob = self.baseline_map[prob_id]
            
            is_fixed = (not base_prob["correct"]) and ds_prob["correct"]
            if not is_fixed:
                continue
                
            base_ids = base_prob.get("output_ids", [])
            ds_ids = ds_prob.get("output_ids", [])
            base_entropy = base_prob.get("entropy_trajectory", [])
            
            if not base_ids or not ds_ids:
                continue
                
            # Find divergence point
            t_div = -1
            min_len = min(len(base_ids), len(ds_ids))
            for i in range(min_len):
                if base_ids[i] != ds_ids[i]:
                    t_div = i
                    break
            
            if t_div == -1 and len(base_ids) != len(ds_ids):
                t_div = min_len
                
            if t_div != -1:
                # Get baseline entropy at T_div
                # Note: Baseline might not have entropy_trajectory if it wasn't recorded.
                # If so, we can't do this analysis exactly, or we fallback if it exists.
                if t_div < len(base_entropy):
                    divergence_entropies.append(base_entropy[t_div])
                else:
                    # Look for other proxies if entropy is missing?
                    pass
                    
        print(f"Found {len(divergence_entropies)} divergence points with recorded baseline entropy.")
        
        if not divergence_entropies:
            print("Not enough baseline entropy data at divergence points. This may occur if baseline didn't record entropy. Skipping plot.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(divergence_entropies, kde=True, color="purple", bins=30)
        
        threshold = 0.15 # Typical empirical threshold
        plt.axvline(x=threshold, color="red", linestyle="--", label=f"Typical Threshold ({threshold})")
        
        plt.title("Divergence Point Anatomy", fontsize=16, pad=15)
        plt.xlabel("Baseline EMA Entropy at Divergence Point ($T_{div}$)", fontsize=14)
        plt.ylabel("Density / Count", fontsize=14)
        plt.legend(fontsize=12)
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
            # A simple way based on logic: convergence is True usually means it triggered the brake
            if convergence:
                # Find where it drops to 0 after peaking
                t_peak = np.argmax(alpha_traj)
                t_brake = -1
                for i in range(t_peak, len(alpha_traj)):
                    if alpha_traj[i] <= 1e-5:
                        # Check if it stays 0
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
        plt.xlim(left=0) # ensure x starts at 0
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
                cont_R.append(max(0, rep * 100)) # Ensure %
                
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
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B", help="HuggingFace model path for tokenizer")
    args = parser.parse_args()
    
    analyzer = LargeScaleMicroAnalyzer(args.json_path, args.model_path)
    analyzer.run_all()
