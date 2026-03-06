"""
Evaluation & Visualization Module
====================================
Generates publication-quality 2×2 figure panels for the paper:
  1. Accuracy Comparison (Grouped Bar Chart)
  2. TECA + Intervention Trajectory (Dual-axis Line Plot)
  3. Language Stability — Repetition Rate (Bar Chart)
  4. Reasoning Efficiency — Tokens vs. Accuracy (Scatter Plot)

Also generates a formatted statistical summary table.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os

from config import RESULTS_DIR

# Use non-interactive backend for server environments
matplotlib.use("Agg")


class PlotVisualizer:
    """
    Generates publication-quality evaluation plots (ICLR/NeurIPS style).
    """

    # Color palette
    COLORS = {
        "Baseline": "#8C8C8C",           # Grey
        "Continuous": "#E07B54",          # Coral/Red
        "Dynamic_Spherical": "#4A90D9",  # Blue
    }
    COLOR_LIST = ["#8C8C8C", "#E07B54", "#4A90D9"]

    def __init__(self, save_dir: str = RESULTS_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Paper style
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        })
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    def generate_comprehensive_report(self, experiment_results: dict):
        """
        Generate the full 2×2 evaluation figure.

        Args:
            experiment_results: Dict of dicts, keyed by mode name.
                Each mode dict should contain:
                    - accuracy: float (0-1)
                    - repetition: float (0-1)
                    - ppl: float
                    - tokens: int (total generated tokens)
                    - local_dtr: float (0-1)
                    - teca_trajectory: list[float]
                    - alpha_trajectory: list[float]
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Accuracy (top-left)
        self._plot_accuracy(axes[0, 0], experiment_results)

        # 2. TECA + Alpha trajectory (top-right)
        self._plot_teca_trajectory(axes[0, 1], experiment_results)

        # 3. Language stability (bottom-left)
        self._plot_stability(axes[1, 0], experiment_results)

        # 4. Reasoning efficiency (bottom-right)
        self._plot_efficiency(axes[1, 1], experiment_results)

        plt.tight_layout(pad=2.0)

        save_path = os.path.join(self.save_dir, "intervention_evaluation_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(
            save_path.replace(".png", ".pdf"), bbox_inches="tight"
        )  # Also save PDF for LaTeX
        plt.close()

        print(f"✅ Evaluation figure saved to {save_path}")
        print(f"✅ PDF version saved to {save_path.replace('.png', '.pdf')}")

        # Print statistical summary
        self._print_summary_table(experiment_results)

    def _plot_accuracy(self, ax, results: dict):
        """Plot 1: Logical Accuracy comparison bar chart."""
        modes = list(results.keys())
        accs = [results[m]["accuracy"] for m in modes]

        bars = ax.bar(
            modes, accs,
            color=[self.COLORS.get(m, "#999999") for m in modes],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_ylim(0, 1.15)
        ax.set_title("(a) Logical Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy")

        # Data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Beautify x-tick labels
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in modes], fontsize=9
        )

    def _plot_teca_trajectory(self, ax, results: dict):
        """
        Plot 2: TECA and intervention strength α trajectory.
        Shows data from the Dynamic_Spherical experiment.
        """
        # Use Dynamic_Spherical data for the trajectory
        ds_key = "Dynamic_Spherical"
        if ds_key not in results:
            ax.text(0.5, 0.5, "No Dynamic_Spherical data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        ds = results[ds_key]
        teca = ds.get("teca_trajectory", [])
        alpha = ds.get("alpha_trajectory", [])

        if not teca:
            ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        steps = np.arange(len(teca))

        # TECA line (left Y-axis)
        color_teca = "#2166AC"
        ax.plot(steps, teca, label="TECA", color=color_teca, linewidth=2, alpha=0.9)
        ax.set_ylabel("TECA (Entropy)", color=color_teca)
        ax.tick_params(axis="y", labelcolor=color_teca)
        ax.set_xlabel("Generation Step (Tokens)")

        # Threshold line
        from config import TECA_THRESHOLD
        ax.axhline(
            y=TECA_THRESHOLD, color="green", linestyle=":", linewidth=1,
            label=f"Threshold ({TECA_THRESHOLD})", alpha=0.7,
        )

        # Alpha line (right Y-axis)
        if alpha:
            ax2 = ax.twinx()
            color_alpha = "#B2182B"
            ax2.plot(steps[:len(alpha)], alpha, label="α (Intervention)",
                     color=color_alpha, linestyle="--", linewidth=2, alpha=0.8)
            ax2.fill_between(
                steps[:len(alpha)], 0, alpha, color=color_alpha, alpha=0.08
            )
            ax2.set_ylabel("Rotation Angle α", color=color_alpha)
            ax2.tick_params(axis="y", labelcolor=color_alpha)

        ax.set_title("(b) Entropy Drop & Intervention Trajectory", fontweight="bold")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if alpha:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax.legend(loc="upper right", fontsize=8)

    def _plot_stability(self, ax, results: dict):
        """Plot 3: Language stability — Repetition Rate comparison."""
        modes = list(results.keys())
        reps = [results[m].get("repetition", 0) * 100 for m in modes]

        bars = ax.bar(
            modes, reps,
            color=[self.COLORS.get(m, "#999999") for m in modes],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title("(c) Language Stability (Repetition Rate)", fontweight="bold")
        ax.set_ylabel("N-gram Repetition Rate (%)")

        # Safety threshold line
        ax.axhline(
            y=5.0, color="green", linestyle=":", linewidth=1.5,
            label="Safe Threshold (<5%)", alpha=0.8,
        )
        ax.legend(loc="upper right", fontsize=8)

        # Data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticklabels(
            [m.replace("_", "\n") for m in modes], fontsize=9
        )

    def _plot_efficiency(self, ax, results: dict):
        """Plot 4: Reasoning efficiency — Tokens consumed vs. Accuracy."""
        modes = list(results.keys())
        tokens = [results[m].get("tokens", 0) for m in modes]
        accs = [results[m].get("accuracy", 0) for m in modes]

        for i, mode in enumerate(modes):
            local_dtr = results[mode].get("local_dtr", 0.5)
            marker_size = max(80, local_dtr * 500)  # Scale marker by DTR
            color = self.COLORS.get(mode, "#999999")

            ax.scatter(
                tokens[i], accs[i],
                s=marker_size,
                color=color,
                label=f"{mode} (DTR={local_dtr:.2f})",
                alpha=0.8,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            # Annotate
            ax.annotate(
                mode.replace("_", "\n"),
                (tokens[i], accs[i]),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=8,
                alpha=0.8,
            )

        ax.set_title("(d) Reasoning Efficiency", fontweight="bold")
        ax.set_xlabel("Total Consumed Tokens")
        ax.set_ylabel("Accuracy")

        # Mean token line
        if tokens:
            ax.axvline(
                x=np.mean(tokens), color="grey", linestyle="--",
                alpha=0.3, label="Avg Token Count",
            )

        ax.legend(loc="best", fontsize=8)

        # Annotate ideal region
        ax.annotate(
            "Ideal: ↑ Accuracy\n↓ Tokens",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="green",
            alpha=0.6,
            style="italic",
        )

    def _print_summary_table(self, results: dict):
        """Print a formatted statistical summary table to stdout."""
        print("\n" + "=" * 80)
        print(f"{'EVALUATION SUMMARY':^80}")
        print("=" * 80)
        print(
            f"{'Mode':<22} {'Acc':>7} {'Rep%':>7} {'PPL':>8} "
            f"{'Tokens':>7} {'DTR':>7}"
        )
        print("-" * 80)

        for mode, data in results.items():
            acc = data.get("accuracy", 0)
            rep = data.get("repetition", 0) * 100
            ppl = data.get("ppl", float("nan"))
            tokens = data.get("tokens", 0)
            dtr = data.get("local_dtr", 0)

            print(
                f"{mode:<22} {acc:>7.2f} {rep:>6.1f}% {ppl:>8.2f} "
                f"{tokens:>7d} {dtr:>7.2f}"
            )

        print("=" * 80)


def generate_single_plot(
    experiment_results: dict,
    plot_type: str,
    save_path: str,
):
    """
    Generate a single standalone plot (useful for paper figures).

    Args:
        experiment_results: Same format as generate_comprehensive_report.
        plot_type: One of "accuracy", "trajectory", "stability", "efficiency".
        save_path: Output file path.
    """
    viz = PlotVisualizer(save_dir=os.path.dirname(save_path))
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_map = {
        "accuracy": viz._plot_accuracy,
        "trajectory": viz._plot_teca_trajectory,
        "stability": viz._plot_stability,
        "efficiency": viz._plot_efficiency,
    }

    if plot_type not in plot_map:
        raise ValueError(f"Unknown plot_type: {plot_type}. Choose from {list(plot_map.keys())}")

    plot_map[plot_type](ax, experiment_results)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Single plot '{plot_type}' saved to {save_path}")
