"""
Standalone Evaluation & Visualization Module
=============================================
手动运行此脚本以生成可视化报告，避免污染 Git 仓库。

用法:
  python evaluation_visualizer.py --result ./results/MATH500_20260315_xxxxx
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 使用非交互式后端，适合服务器环境
matplotlib.use("Agg")

try:
    from config import ENTROPY_THRESHOLD, ALPHA_MAX
except ImportError:
    ENTROPY_THRESHOLD = 0.15
    ALPHA_MAX = 0.5


class PlotVisualizer:
    COLORS = {
        "Baseline": "#8C8C8C",           # 灰色
        "Continuous": "#E07B54",         # 珊瑚红
        "Dynamic_Spherical": "#4A90D9",  # 科技蓝
    }

    def __init__(self, result: str):
        self.result = result
        self.json_path = os.path.join(result, "experiment_results.json")
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"找不到结果文件: {self.json_path}")
        
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        # 设置论文级别的图表样式
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({
            "font.family": "serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

    def run(self):
        """执行所有可视化流水线"""
        print(f"📊 正在处理结果目录: {self.result}")
        self._plot_global_metrics()
        self._plot_intervention_dynamics()
        print(f"✅ 所有可视化图片已保存至: {self.result}")

    def _plot_global_metrics(self):
        """绘制全局核心指标（Accuracy, Repetition, Tokens, DTR）柱状图"""
        modes = list(self.results.keys())
        metrics = ["accuracy", "repetition", "tokens", "local_dtr"]
        titles = ["1. Accuracy (Higher is better)", 
                  "2. Repetition Rate (Lower is better)", 
                  "3. Avg Tokens (Efficiency)", 
                  "4. Local DTR (Depth)"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # 提取数据，处理可能的 NaN
            vals = []
            for m in modes:
                val = self.results[m].get(metric, 0)
                if val is None or np.isnan(val): val = 0
                # Repetition 转换为百分比
                if metric == "repetition": val *= 100
                vals.append(val)
            
            colors = [self.COLORS.get(m, "#333333") for m in modes]
            bars = ax.bar(modes, vals, color=colors, width=0.6)
            ax.set_title(titles[i], fontweight='bold', pad=15)
            
            # Repetition 的安全线
            if metric == "repetition":
                ax.axhline(5.0, color='green', linestyle=':', label="Safe (<5%)")
                ax.set_ylabel("Percentage (%)")
                ax.legend()
            
            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.2f}" if isinstance(height, float) else f"{int(height)}"
                ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')
            
            # 优化 X 轴
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=15 if len(modes)>3 else 0)

        plt.tight_layout()
        save_path = os.path.join(self.result, "global_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_intervention_dynamics(self):
        """提取触发干预的典型样本，绘制高解释性的 EMA 熵与 Alpha 双轴动力学曲线"""
        if "Dynamic_Spherical" not in self.results:
            return
            
        ds_data = self.results["Dynamic_Spherical"]
        if "ema_trajectory" not in ds_data or "alpha_trajectory" not in ds_data:
            print("⚠️ 未找到 ema_trajectory 或 alpha_trajectory，跳过动力学曲线绘制。")
            return

        ema_trajs = ds_data["ema_trajectory"]
        alpha_trajs = ds_data["alpha_trajectory"]

        # 寻找真正触发了干预的样本 (Alpha > 0)
        triggered_indices = []
        for i, alpha_t in enumerate(alpha_trajs):
            if max(alpha_t) > 0.05:  # 只有最大干预强度超过 0.05 才值得画
                triggered_indices.append(i)

        if not triggered_indices:
            print("ℹ️ 本次实验中没有样本触发显著干预 (Alpha均接近0)，跳过动力学绘图。")
            return

        # 最多只画前 2 个典型样本，避免图表过长
        samples_to_plot = triggered_indices[:2]
        fig, axes = plt.subplots(len(samples_to_plot), 1, figsize=(12, 5 * len(samples_to_plot)))
        if len(samples_to_plot) == 1:
            axes = [axes]

        for idx, ax in zip(samples_to_plot, axes):
            ema = np.array(ema_trajs[idx])
            alpha = np.array(alpha_trajs[idx])
            steps = np.arange(len(ema))

            # 主 Y 轴：绘制 EMA 熵
            ax.plot(steps, ema, color='#1f77b4', linewidth=2.5, label='EMA Entropy')
            ax.axhline(ENTROPY_THRESHOLD, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({ENTROPY_THRESHOLD})')
            ax.set_ylabel('EMA Entropy', color='#1f77b4', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#1f77b4')
            ax.set_xlabel('Generation Step (Tokens)')
            
            # 寻找 ThinkBrake 收敛瞬间（EMA 未降，但 Alpha 强行归零）
            # 简单启发式：如果 Alpha 突然变 0 但 EMA 还在高位，画一条垂线
            if len(alpha) > 5:
                for t in range(1, len(alpha)):
                    if alpha[t-1] > 0.05 and alpha[t] == 0 and ema[t] > ENTROPY_THRESHOLD:
                        ax.axvline(t, color='green', linestyle=':', linewidth=2, label='ThinkBrake Cutoff')
                        break

            # 次 Y 轴：绘制干预强度 Alpha
            ax2 = ax.twinx()
            ax2.plot(steps, alpha, color='#d62728', linewidth=2, linestyle='-', label='Alpha (Steering Strength)')
            ax2.fill_between(steps, 0, alpha, color='#d62728', alpha=0.15)
            ax2.set_ylabel('Alpha', color='#d62728', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#d62728')
            ax2.set_ylim(0, max(ALPHA_MAX * 1.2, max(alpha) * 1.2))

            # 合并图例
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', framealpha=0.9)
            
            ax.set_title(f"Intervention Dynamics (Prompt Index: {idx})", fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.result, "intervention_dynamics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成实验可视化图表")
    parser.add_argument("--result", type=str, required=True, help="指向包含 experiment_results.json 的结果目录")
    args = parser.parse_args()

    visualizer = PlotVisualizer(args.result)
    visualizer.run()