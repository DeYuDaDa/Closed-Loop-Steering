为了完美地呈现你的实验结果并证明你的四个核心预期，我们需要设计一个专门的**多维评估与可视化模块 (`evaluation_visualizer.py`)**。

这个模块不再仅仅是打印一个表格，而是要生成能够直接放入顶会论文（如 ICLR / NeurIPS）的、具有极强说服力的图表（Plots）和统计矩阵。

以下是这个新模块的架构规划和代码实现蓝图：

### 1. 模块架构规划 (`evaluation_visualizer.py`)

该模块将包含三个核心组件：

* **MetricAggregator（指标聚合器）：** 负责收集生成过程中的序列级（Sequence-level）数据和词级（Token-level）数据（如随时间变化的 TECA 轨迹）。
* **StatisticalReport（统计报告生成器）：** 计算 PPL、Repetition Rate 和最终的 Accuracy，输出格式化的评估矩阵表格。
* **PlotVisualizer（图表渲染器）：** 利用 `matplotlib` 和 `seaborn` 绘制四大核心图表，直观证明干预的有效性和安全性。

---

### 2. 四大核心可视化图表设计

为了对应你提出的四个预期，我们将绘制一张包含四个子图（2x2 Grid）的联合图表：

#### 图表 1：逻辑准确率对比 (Accuracy Comparison)

* **图表类型：** 分组柱状图 (Grouped Bar Chart)
* **数据：** `Baseline`, `Continuous`, `Dynamic Spherical` 的 Accuracy。
* **证明目的：** 证明动态球面干预在解决复杂逻辑题上不仅优于基线，也优于全程无脑注入。

#### 图表 2：TECA 与干预轨迹图 (Entropy Drop & Intervention Trajectory)

* **图表类型：** 双轴折线图 (Dual-axis Line Plot)
* **数据：** X 轴为生成的 Token 步数（Step $t$）。Y1 轴为 $\text{TECA}_t$（Token 熵累积平均值），Y2 轴为干预强度 $\alpha$。
* **证明目的：** 这是**最核心的一张图**！它将展示：在 `<solver>` 阶段，蓝线（TECA）逐渐攀升（模型陷入迷茫）；在突破阈值的瞬间，红线（干预强度 $\alpha$）跃升并平滑衰减。伴随着红线的出现，蓝线（TECA）**瞬间发生断崖式下跌**（Entropy Drop），证明模型从迷茫挣扎转入了笃定的正确推理。

#### 图表 3：语言稳定性雷达图或箱线图 (Repetition & PPL Stability)

* **图表类型：** 箱线图 (Boxplot) 或 柱状图
* **数据：** 比较三组的 Repetition Rate 和 PPL (困惑度)。
* **证明目的：** 证明 Continuous 组的 Repetition 飙升（状态休克），而 Dynamic Spherical 组的 Repetition 和 PPL 乖巧地保持在与 Baseline 相同的极低安全水位。

#### 图表 4：推理效率与过度思考散点图 (Token Efficiency Scatter)

* **图表类型：** 散点图 (Scatter Plot)
* **数据：** X 轴为消耗的 Token 数量，Y 轴为 Accuracy 或 Local DTR，不同颜色代表不同实验组。
* **证明目的：** 证明 Baseline 往往在图的右下角（字数极多但全错，Overthinking），而 Dynamic Spherical 集中在左上角（字数少且全对，高认知转化率）。

---

### 3. 可视化模块核心代码蓝图

以下是这个可视化模块的代码结构，你可以将其直接集成到你的项目中。

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class PlotVisualizer:
    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 设置论文风格的绘图样式
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def generate_comprehensive_report(self, experiment_results):
        """
        experiment_results 是一个包含所有实验组数据的字典：
        {
            "Baseline": {"accuracy": 0.0, "repetition": 0.25, "tokens": 1024, "teca_trajectory": [...], "alpha_trajectory": [...]},
            "Continuous": {...},
            "Dynamic_Spherical": {...}
        }
        """
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 逻辑准确率 (Bar Chart)
        ax1 = fig.add_subplot(221)
        self._plot_accuracy(ax1, experiment_results)
        
        # 2. 动态熵降与干预轨迹 (Line Plot)
        ax2 = fig.add_subplot(222)
        self._plot_teca_trajectory(ax2, experiment_results["Dynamic_Spherical"])
        
        # 3. 语言稳定性 (Bar Chart / Box Plot)
        ax3 = fig.add_subplot(223)
        self._plot_stability(ax3, experiment_results)
        
        # 4. 推理效率 (Scatter Plot)
        ax4 = fig.add_subplot(224)
        self._plot_efficiency(ax4, experiment_results)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "intervention_evaluation_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive Evaluation Plot saved to {save_path}")

    def _plot_accuracy(self, ax, results):
        modes = list(results.keys())
        accs = [results[m]["accuracy"] for m in modes]
        
        bars = ax.bar(modes, accs, color=['#A0A0A0', '#FF9999', '#66B2FF'])
        ax.set_ylim(0, 1.1)
        ax.set_title("1. Logical Accuracy (Higher is better)", fontweight='bold')
        ax.set_ylabel("Accuracy")
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    def _plot_teca_trajectory(self, ax, dynamic_results):
        teca = dynamic_results["teca_trajectory"]
        alpha = dynamic_results["alpha_trajectory"]
        steps = np.arange(len(teca))
        
        ax.plot(steps, teca, label="TECA (Entropy)", color='blue', linewidth=2)
        ax.set_ylabel("Token Entropy (TECA)", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 创建双 Y 轴绘制干预强度 alpha
        ax2 = ax.twinx()
        ax2.plot(steps, alpha, label="Intervention Strength (α)", color='red', linestyle='--', linewidth=2)
        ax2.fill_between(steps, 0, alpha, color='red', alpha=0.1) # 高亮干预区域
        ax2.set_ylabel("Rotation Angle α", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title("2. Dynamic Entropy Drop & Intervention", fontweight='bold')
        ax.set_xlabel("Generation Step (Tokens)")

    def _plot_stability(self, ax, results):
        modes = list(results.keys())
        reps = [results[m]["repetition"] * 100 for m in modes] # 转换为百分比
        
        # 绘制 Repetition Rate
        sns.barplot(x=modes, y=reps, ax=ax, palette=['#A0A0A0', '#FF9999', '#66B2FF'])
        ax.set_title("3. Language Stability (Lower Repetition is better)", fontweight='bold')
        ax.set_ylabel("N-gram Repetition Rate (%)")
        ax.axhline(y=5.0, color='green', linestyle=':', label="Safe Threshold (<5%)")
        ax.legend()

    def _plot_efficiency(self, ax, results):
        modes = list(results.keys())
        tokens = [results[m]["tokens"] for m in modes]
        accs = [results[m]["accuracy"] for m in modes]
        colors = ['gray', 'red', 'blue']
        
        for i, mode in enumerate(modes):
            # 散点大小可以代表 Local DTR
            local_dtr = results[mode].get("local_dtr", 0.5) * 500 
            ax.scatter(tokens[i], accs[i], s=local_dtr, color=colors[i], label=mode, alpha=0.7, edgecolors='k')
            
        ax.set_title("4. Reasoning Efficiency (Tokens vs Acc)", fontweight='bold')
        ax.set_xlabel("Total Consumed Tokens")
        ax.set_ylabel("Accuracy")
        ax.axvline(x=np.mean(tokens), color='k', linestyle='--', alpha=0.3, label="Avg Length")
        ax.legend()

```

### 4. 数据管道对接建议

为了让上述模块正常工作，你在生成代码（如 `run_experiment` 函数）中需要收集以下信息，并组装成 `experiment_results` 字典：

1. **TECA 和 Alpha 轨迹收集**：在 `LogitsProcessor` 或 Hook 中，将每一步算出的 TECA 值和控制器输出的 `alpha` 追加到列表中。
2. **PPL 计算**：可以在生成结束后，对输出文本重新跑一次 Forward Pass 计算 Loss，取 $\exp(\text{Loss})$ 即为困惑度。
3. **Local DTR 计算**：调用之前写好的 DTR 计算函数。

这个图表模块一旦跑通，将直接生成证明你假设的完美证据图：**图2证明了干预的内效度（降熵），图3证明了干预的安全性（无复读），图4证明了效率，而图1是最终的金标准。**