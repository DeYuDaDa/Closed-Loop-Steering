为了复现《Mitigating Overthinking in Large Reasoning Models via Manifold Steering》这篇论文，我们可以将其核心算法（Manifold Steering，流形引导）拆解为明确的流水线，并为 coding agent 设计出模块化的代码架构。

以下是该论文的算法梳理与代码构建方案：

### 一、 核心算法梳理

该算法的核心思想是通过主成分分析（PCA）找到模型激活空间中的低维流形子空间，将高维的“过度思考（overthinking）”引导方向投影到该子空间上，以消除高维空间带来的干扰噪声，从而在不降低准确率的前提下大幅减少推理生成的 token 数量。

完整的算法流程可以分为以下六个步骤：

**1. 数据集构建 (Data Selection)**
*   **原始数据：** 从 OpenMathInstruct-2 训练集中随机抽取 2万个问题，为每个问题生成 5 个独立回答（参数：temperature=0.6, top-p=0.95, max_length=16k）。
*   **冗余集 ($D_{redundant}$)：** 筛选出 5 个回答均超过 16k tokens 且包含超过 20 次犹豫关键词（如 "wait", "alternatively"）的问题。将提示词与包含关键词截断的部分回答拼接在一起构成输入。
*   **简明集 ($D_{concise}$)：** 筛选出 5 个回答均少于 1k tokens 且不含犹豫关键词的问题。仅保留问题的提示词作为输入。
*   **清洗与采样：** 通过 IsolationForest 算法过滤异常值后，各保留 500 个样本用于流形估计，其中各取 100 个样本用于计算引导方向。

**2. 激活值提取 (Activation Extraction)**
*   分别将 $D_{redundant}$ 和 $D_{concise}$ 的输入喂给模型，提取指定层（Layer $l$）最后一个 token 的残差流激活值 $h^{(l)}(x)$。
*   **关键超参数（提取层 $l$）：** DeepSeek-R1-Distill-Qwen-1.5B (层 27)，7B (层 27)，14B (层 47)；DeepSeek-R1-Distill-Llama-8B (层 31)。

**3. 计算初始引导方向 (Initial Steering Direction)**
*   使用均值差技术（Difference-in-means）计算初始向量：
    $r^{(l)} = \frac{1}{|D_{redundant}|} \sum h^{(l)}(x) - \frac{1}{|D_{concise}|} \sum h^{(l)}(x)$。
*   对方向进行 L2 归一化：$r^{(l)} = r^{(l)} / \|r^{(l)}\|_2$。

**4. 估计低维流形子空间 (Manifold Subspace Estimation)**
*   使用全量推理数据集 $D_{reasoning} = D_{redundant} \cup D_{concise}$（共 1000 个样本）提取激活值，计算协方差矩阵 $C^{(l)}$。
*   对 $C^{(l)}$ 进行特征分解（PCA），提取前 $k$ 个主成分 $U^{(l)}_{eff}$。论文指出 $k=10$ 即可捕获超过 70% 的方差。

**5. 流形引导投影 (Manifold Steering)**
*   计算投影矩阵：$P_M = U^{(l)}_{eff} (U^{(l)}_{eff})^T$。
*   将初始方向 $r^{(l)}$ 投影到流形子空间，消除干扰噪声：$r^{(l)}_{overthinking} = P_M r^{(l)}$。
*   对投影后的方向进行归一化：$r^{(l)}_{overthinking} = r^{(l)}_{overthinking} / \|r^{(l)}_{overthinking}\|_2$。

**6. 推理干预 (Inference Intervention)**
*   在模型生成阶段，**对所有层**的每一个 token 激活值 $h^{(l)}(x_i)$ 进行消融干预：
    $h'^{(l)}(x_i) = h^{(l)}(x_i) - \alpha \times r^{(l)}_{overthinking} (r^{(l)}_{overthinking})^T h^{(l)}(x_i)$。
*   **关键超参数（干预强度 $\alpha$）：** 1.5B ($\alpha=0.7$)，7B ($\alpha=0.3$)，8B ($\alpha=0.5$)，14B ($\alpha=0.3$)。

---

### 二、 代码构建方案 (For Coding Agent)

为了让 coding agent 能够高效复现，建议将代码架构分为四个核心模块（Modules）。推荐使用 PyTorch 和 HuggingFace `transformers` 库。

#### 模块 1: DataProcessor (数据构建与过滤)
**功能：** 负责数据生成、截断与过滤。
*   `generate_responses()`: 调用模型生成回答，设置 `max_new_tokens=16384`。
*   `filter_redundant()`: 匹配犹豫关键词（如 "wait"），若次数 > 20 且达到 token 上限，则保留并在第一个 "wait" 处截断拼接至 Prompt。
*   `filter_concise()`: 提取小于 1k tokens 且无关键词的样本，仅返回 Prompt。
*   `remove_outliers()`: 使用 `sklearn.ensemble.IsolationForest` 剔除异常样本。

#### 模块 2: ActivationExtractor (激活值提取器)
**功能：** 注入 PyTorch Forward Hook，精准提取指定层的残差流状态。
*   `register_hooks(layer_idx)`: 在模型的特定 `decoder_layer` 的末尾（如 MLP 与 Attention 残差相加后）注册 hook。
*   `get_last_token_activation(batch_inputs)`: 仅保存输入序列最后一个 token（序列长度维度上的最后一个元素）的 hidden state。

#### 模块 3: ManifoldSteeringCalculator (流形引导计算器)
**功能：** 核心数学运算模块，计算投影方向。
*   `compute_mean_difference(act_redundant, act_concise)`: 实现步骤3的均值差公式并做 L2 归一化。
*   `compute_pca_projection(act_all, k=10)`:
    *   输入融合了冗余和简明的 1000 个样本激活值。
    *   计算协方差矩阵，使用 `torch.linalg.eigh` 或 `sklearn.decomposition.PCA` 提取 Top-10 特征向量 $U_{eff}$。
    *   计算投影矩阵 $P_M = U_{eff} @ U_{eff}^T$。
*   `get_steering_vector()`: 将初始向量乘以投影矩阵，并进行 L2 归一化，返回最终的 $r_{overthinking}$。

#### 模块 4: InferenceIntervention (推理干预器)
**功能：** 在推理阶段实时修改模型所有层的内部激活值。
*   由于需要在前向传播中**作用于所有层**，建议重写模型的 forward 函数，或者在所有 transformer block 后挂载修改 hook。
*   `intervention_hook(module, inputs, outputs)`:
    *   拦截输出的 hidden state $h$。
    *   应用公式：$h' = h - \alpha \cdot r_{overthinking} \cdot (r_{overthinking}^T \cdot h)$。
    *   **注意维度广播**：$h$ 的 shape 通常为 `(batch_size, seq_len, hidden_dim)`，而 $r$ 为 `(hidden_dim,)`，需要用 `torch.matmul` 等价实现点乘缩放。

#### 提供给 Coding Agent 的启动配置示例：
```python
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "target_layer_for_extraction": 27,  # 根据论文设置
    "intervention_layers": "all",       # 论文指出应用于所有层
    "pca_components_k": 10,             #
    "alpha": 0.3,                       # 7B 模型的干预强度
    "hesitation_keyword": "wait",       #
}
```