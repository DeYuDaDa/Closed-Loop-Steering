当样本量从 32 题扩展到 500 题（总计 1500 条完整的生成轨迹）时，**微观分析的范式必须发生改变**。

面对总计几百万个 Token 的海量张量数据，你不能再依赖人工去逐条阅读文本和对齐图表了。你需要进行**“聚合微观分析”（Aggregated Micro-Analysis）**，也就是用统计学的手段去挖掘和证明模型底层的机制规律。

对于 500 个样本的量级，我为你设计了以下 **5 个自动化的高阶微观分析方案**，这些方案生成的图表和数据，可以直接作为顶会论文的“Mechanistic Interpretability（机制可解释性）”核心章节：

------

### 方案 1：分层事件对齐动力学 (Stratified Event-Aligned Dynamics)

**🔍 目的：** 证明“成功的干预”和“失败的干预”在底层物理特征上有着本质的区别。

- **操作方法：** 不再将所有触发干预的轨迹混在一起平均。根据我们在翻转矩阵中的定义，将 Dynamic 组的轨迹分为两拨：

  1. **救活组 (Fixed)**：Baseline 错，Dynamic 对。

  2. **顽固组 (Both Wrong)**：Baseline 错，Dynamic 依然错。

     分别对这两组进行 $T=0$（首次干预触发点）的对齐，画出两根不同的 EMA 熵均值曲线。

- **学术爆点 (Expected Outcome)：** 你大概率会看到：**救活组**的蓝线在 $T=0$ 之后会出现显著的“断崖式降熵”；而**顽固组**的蓝线在 $T=0$ 后依然在高位剧烈震荡。这在统计学上完美证明了：**系统生效的因果机制，正是成功诱发了模型内部的降熵（认知收敛）。**

### 方案 2：自动化语义轴心词挖掘 (Automated Pivot Token Mining)

**🔍 目的：** 将连续的物理干预（$\alpha$ 值）与离散的自然语言语义建立因果映射。

- **操作方法：** 写脚本遍历 Dynamic 组所有 500 个样本的 `alpha_trajectory`。

  锁定每一个 $\alpha$ 达到峰值（比如 $\alpha > 0.25$）的索引 $i$。

  提取该点及随后 3 个 Token 的 `output_ids` 并解码。统计在这些极值点，频率最高的 Top-20 个词是什么。

- **学术爆点 (Expected Outcome)：** 你将能用数据证明：在最强烈的物理干预下，模型最常吐出的词是 **"Wait", "Let", "However", "Alternatively", "Error"** 等反思词汇。这直接证明了 Critic 向量（高维流形）被成功投影为了“自我纠错”的语义实体。

### 方案 3：分歧点解剖学 (Divergence Point Anatomy)

**🔍 目的：** 证明 Dynamic 系统是在模型“即将犯错的悬崖边”拉住了它。

- **操作方法：** 对比同一道题的 Baseline 输出和 Dynamic 输出。写一段脚本，逐字对比两者的 `output_ids`，找到两者**首次产生分歧**的那个 Token 索引 $T_{div}$。

  然后，去提取 Baseline 在 $T_{div}$ 附近（如前 5 个 Token）的 `ema_trajectory` 熵值。

- **学术爆点 (Expected Outcome)：** 统计发现，大多数分歧点 $T_{div}$ 都恰好发生在 Baseline 熵值飙升（超过 0.15）的地方。这证明了我们的 EMA 探针极其精准——**Baseline 刚开始犹豫（高熵），Dynamic 就果断介入强行掰平了马尔可夫链，改变了生成轨迹，从而剪除了后续冗长的“过度思考”。**

### 方案 4：ThinkBrake 物理边界的统计测绘 (ThinkBrake Boundary Mapping)

**🔍 目的：** 解释为什么 Dynamic 没有像 Continuous 那样引发高达 30% 的重复率。

- **操作方法：** 在 Dynamic 组中，统计所有 `convergence = True`（ThinkBrake 被触发）那一刻，距离最终输出结束 `<|im_end|>` 还有多少个 Token。

  分析紧跟在 `convergence` 触发点之后的 Token 分布（比如是不是大量集中在 `\boxed` 或者特定的数字、结尾标记上）。

- **学术爆点 (Expected Outcome)：** 画出一个直方图，展示“刹车点在序列中的相对位置”。你可以向审稿人证明：ThinkBrake 不是随机截断的，它极其聪明地在模型得出“确定性计算结果”（如 \boxed 区域）的瞬间启动了保护机制，精准隔绝了 $\alpha$ 的后续污染，从而保全了语法流形。

### 方案 5：“状态休克”的二维能量散点图 (Energy-Shock Scatter Plot)

**🔍 目的：** 直观展示 Continuous 和 Dynamic 在干预能量与副作用上的本质区别。

- **操作方法：** 定义一个变量叫 **“干预总能量” (Total Intervention Energy)**，即一整道题里 $\sum \alpha_t$ 的积分。

  对 Continuous 组和 Dynamic 组，画一张二维散点图：

  - **X轴**：干预总能量

  - **Y轴**：N-gram 重复率 (Repetition Rate)

    每个点代表 500 个样本中的一道题。

- **学术爆点 (Expected Outcome)：** 你会看到两团截然不同的点簇：

  - **Continuous 簇**：位于图的右上方，能量极大（比如 $\sum \alpha > 500$），且重复率极高（20%~80%），呈现正相关。这证明了持续注入必然导致休克。
  - **Dynamic 簇**：集中在图的左下方，能量适中（因为介入后很快撤出），重复率死死压在 10% 左右。这证明了我们通过闭环控制，实现了**“低能量、高收益、零休克”**的最佳费效比



---


# Role & Context
You are an expert Data Scientist and Machine Learning Engineer specializing in Mechanistic Interpretability of Large Language Models (LLMs).
Your task is to write a standalone Python script named `large_scale_micro_analyzer.py` to analyze a large-scale experiment dataset (1500+ generated trajectories) and generate 5 publication-quality mechanistic interpretability charts.

# Task Breakdown: 5 Micro-Analysis Modules
Please structure the script using a main class `LargeScaleMicroAnalyzer`. Implement the following 5 analysis modules as class methods:

## Task 1: Stratified Event-Aligned Dynamics (`plot_stratified_dynamics`)
**Goal:** Compare the entropy trajectories of successful interventions vs. failed interventions.
**Logic:**
1. Match problems by `"id"` between `Baseline` and `Dynamic_Spherical` (DS).
2. Create two cohorts from DS problems where `max(alpha_trajectory) > 0.05`:
   - `Fixed` cohort: Baseline `correct` is False, DS `correct` is True.
   - `Stubborn` cohort: Baseline `correct` is False, DS `correct` is False.
3. For each trajectory in both cohorts, find $T_0$ (the first index where `alpha > 0.05`).
4. Extract the `ema_trajectory` window `[T_0 - 20, T_0 + 30]`. 
5. Compute the mean trajectory for both cohorts.
**Plot:** A dual-line chart plotting the mean EMA entropy of the `Fixed` (Green) and `Stubborn` (Red) cohorts over the aligned window [-20, 30]. Add a vertical dashed line at X=0 (Intervention Trigger).

## Task 2: Automated Pivot Token Mining (`plot_pivot_tokens`)
**Goal:** Identify what semantic tokens the model generates at the peak of intervention.
**Logic:**
1. Load the tokenizer using `transformers.AutoTokenizer.from_pretrained(MODEL_PATH)`. (Accept MODEL_PATH as a CLI argument, default to "Qwen/Qwen2.5-Math-7B").
2. Iterate through all DS problems where `max(alpha_trajectory) > 0.05`.
3. Find the index $T_{peak}$ where `alpha` reaches its maximum value.
4. Extract the `output_ids` slice `[T_{peak}, T_{peak} + 3]`.
5. Decode these 3 tokens into strings. Clean up whitespaces and special characters.
6. Count the frequencies of these "Pivot Tokens".
**Plot:** A horizontal Bar Chart of the Top-20 most frequent Pivot Tokens.

## Task 3: Divergence Point Anatomy (`plot_divergence_anatomy`)
**Goal:** Prove that interventions happen exactly when Baseline starts to wander off into high-entropy rationalization.
**Logic:**
1. Focus only on the `Fixed` cohort (Baseline False, DS True).
2. For each problem, compare `Baseline["output_ids"]` and `DS["output_ids"]` token-by-token.
3. Find the first index $T_{div}$ where the two token sequences differ.
4. Fetch the value of `Baseline["ema_trajectory"]` at $T_{div}$. 
5. Collect these entropy values into a list.
**Plot:** A Histogram (with KDE) showing the distribution of Baseline EMA entropy values at the exact moment of semantic divergence. Add a vertical line for `ENTROPY_THRESHOLD` (e.g., 0.15) to show alignment.

## Task 4: ThinkBrake Boundary Mapping (`plot_thinkbrake_boundary`)
**Goal:** Show where the ThinkBrake (alpha drops to 0 after intervention) triggers relative to the end of the sequence.
**Logic:**
1. In the DS mode, find trajectories that were intervened (`max(alpha) > 0.05`) but eventually dropped to exactly `0.0` before the sequence ended.
2. Find the index $T_{brake}$ where `alpha` drops back to 0.0 and stays 0.0.
3. Calculate `tokens_remaining = len(output_ids) - T_{brake}`.
**Plot:** A Histogram showing the distribution of `tokens_remaining`. This proves the brake doesn't just trigger at the EOS token, but actively protects the final deductive steps.

## Task 5: Energy-Shock Scatter Plot (`plot_energy_shock`)
**Goal:** Visualize the trade-off between intervention energy and the state shock (repetition rate).
**Logic:**
1. For every problem in `Continuous` and `Dynamic_Spherical`:
2. Calculate Total Energy: $E = \sum \text{alpha\_trajectory}$
3. Fetch Repetition Rate: $R = \text{repetition} * 100$
4. Collect $(E, R)$ pairs for both groups.
**Plot:** A Scatter Plot with Total Energy on the X-axis and Repetition Rate (%) on the Y-axis. 
- Use semi-transparent red dots for Continuous and blue dots for Dynamic_Spherical.
- Add marginal KDE plots (like seaborn `jointplot` or `kdeplot` contours) to highlight the clustering differences (Continuous will be top-right, DS will be bottom-left).

# Global Output & Coding Guidelines
1. **Robustness:** Use `try-except` blocks around token decoding and index slicing. Some trajectories might be too short, simply `continue` and ignore them.
2. **Performance:** Do not load the JSON into pandas dataframe, iterate through native Python dicts directly for memory efficiency.
3. **Style:** Use `matplotlib.use("Agg")`. Apply `seaborn.set_theme(style="whitegrid")`. Make titles, labels, and legends publication-ready (font size 12-14, clean layouts).
4. **Execution:** Save all 5 plots as high-resolution PNGs (DPI=300) in the same directory as the JSON file. Print a beautiful text summary of the findings to the console.
5. Provide a CLI interface using `argparse` to accept `--json_path` and `--model_path`.

