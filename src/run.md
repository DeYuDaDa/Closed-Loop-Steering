# Closed-Loop Steering System — 运行手册

## 一、环境要求

- Python 3.10+，PyTorch 2.x，transformers
- GPU：推荐 80G A100（跑 full dataset）；消融实验建议只跑 50 个最难样本
- 需要先完成向量提取（`extract_critic_vector.py`）：在 `./vectors/qwen3-8b/` 下存在 `critic.pt`（PCA 纯化向量）和 `critic_raw.pt`（原始 CAA 向量）

---

## 二、核心参数说明（`config.py`）

| 参数 | 含义 | 关键消融值 |
|------|------|---|
| `ALPHA_MAX` | PID 最大干预强度 | 0.45 (full) |
| `CONTINUOUS_ALPHA` | Continuous 模式固定 SLERP α | 0.30 / 0.45 |
| `CONTINUOUS_LINEAR_ALPHA` | Continuous_Linear 模式固定线性系数 | 0.45 / 0.65 |
| `LAYER_ID` | Hook 注入的 Transformer 层 | 24 |
| `AIME_MAX_TOKENS` | 最大生成 token 数 | 4096 |
| `BATCH_SIZE` | 批量推理大小 | 16 |

> `α_linear = sin(α_slerp × π/2)` 换算公式保证等效物理扰动

---

## 三、实验模式一览

| 模式字符串 | 描述 | 关闭的模块 |
|------------|------|-----------|
| `Baseline` | 正常推理，无干预 | - |
| `Continuous` | 每步固定强度 SLERP 旋转 | - |
| `Continuous_Linear` | 每步固定强度线性叠加（无 SLERP） | SLERP |
| `Dynamic_Spherical` | **完整方法**：PD + EMA + ThinkBrake + PCA + SLERP | - |
| `Dynamic_Spherical_No_Manifold` | 完整方法但使用 **原始向量**（无 PCA） | Manifold/PCA |
| `Dynamic_Linear` | 完整方法但使用 **线性叠加**（无 SLERP） | SLERP |
| `Dynamic_Spherical_No_ThinkBrake` | 完整方法但禁用 **ThinkBrake 收敛锁** | ThinkBrake |
| `Dynamic_Spherical_No_EMA` | 完整方法但使用 **瞬时熵**（无 EMA 平滑） | EMA |

---

## 四、运行命令

### 4.1 主对照实验（完整数据集）

```bash
# 完整方法 vs Continuous_Linear（核心对照）
python run_experiment.py --dataset ./dataset/aime2024.jsonl \
    --modes Continuous_Linear Dynamic_Spherical

# 包含 Baseline（获取原始准确率基线）
python run_experiment.py --dataset ./dataset/aime2024.jsonl \
    --modes Baseline Continuous Dynamic_Spherical
```

### 4.2 消融实验（建议仅跑 50 个最难样本）

```bash
# 1. w/o Manifold（验证 PCA 纯化噪声的作用）
#    需要确保 ./vectors/qwen3-8b/critic_raw.pt 存在
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes Dynamic_Spherical Dynamic_Spherical_No_Manifold

# 2. w/o SLERP（验证保范数旋转的关键性，线性叠加会导致状态休克）
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes Dynamic_Spherical Dynamic_Linear

# 3. w/o ThinkBrake（验证 ThinkBrake 作为安全刹车的作用）
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes Dynamic_Spherical Dynamic_Spherical_No_ThinkBrake

# 4. w/o EMA（验证 EMA 平滑稳定性的作用，瞬时熵会引起触发抖动）
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes Dynamic_Spherical Dynamic_Spherical_No_EMA
```

### 4.3 TAE 竞品对照实验（EMNLP 2025）

TAE (Token-level Adaptive Entropy) 是完全开环的对照方法。

```bash
# Version A: True TAE — 完全复现 (开环 H_t → 线性注入 → 原始向量)
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes Baseline True_TAE Dynamic_Spherical

# Version B: TAE + Spherical — 控制变量 (开环 H_t → SLERP → PCA向量)
# 用于证明即使有最好的"方向盘"，开环瞬时震荡导致的问题依然存在
python run_experiment.py --dataset ./dataset/aime2024_hard50.jsonl \
    --modes True_TAE TAE_Spherical Dynamic_Spherical
```

> **注意**：
> - `True_TAE` 使用 `critic_raw.pt` 原始向量（无PCA），α_t = clamp(H_t × k, 0, α_max) 直接线性注入
> - `TAE_Spherical` 使用 PCA 纯化向量 + SLERP，控制变量已隔离至纯控制器差异
> - 增益系数 `k = ALPHA_MAX / 3.0`（3.0 对应高困惑 token 的典型熵值）

### 4.4 超参敏感性分析

```bash
# α=0.30 vs α=0.45（在 config.py 中修改 ALPHA_MAX 后运行）
# 弱干预档：ALPHA_MAX=0.30, CONTINUOUS_ALPHA=0.30, CONTINUOUS_LINEAR_ALPHA=0.45
# 强干预档：ALPHA_MAX=0.45, CONTINUOUS_ALPHA=0.45, CONTINUOUS_LINEAR_ALPHA=0.65
python run_experiment.py --dataset ./dataset/aime2024.jsonl \
    --modes Continuous_Linear Dynamic_Spherical
```

### 4.4 多数据集（Math500 / ZebraLogic）

```bash
# Math500
python run_experiment.py --dataset ./dataset/math500.jsonl \
    --modes Baseline Dynamic_Spherical

# ZebraLogic
python run_experiment.py --dataset ./dataset/zebralogic.jsonl \
    --modes Baseline Dynamic_Spherical
```

---

## 五、结果文件

运行后结果保存在：
```
./results/{dataset_name}_{timestamp}/experiment_results.json
```

JSON 结构：
```json
{
  "Dynamic_Spherical": {
    "accuracy": 0.45,
    "correct_count": 9,
    "total_count": 20,
    "repetition": 0.02,
    "tokens": 2134,
    "per_problem": [
      {
        "id": "2024_I_1",
        "correct": true,
        "ema_trajectory": [...],
        "alpha_trajectory": [...],
        "entropy_trajectory": [...],
        "alpha_active_steps": 120
      }
    ]
  }
}
```

---

## 六、向量文件准备

消融实验 `w/o Manifold` 需要原始 CAA 向量。运行提取脚本时确保保存了 raw 版本：

```bash
# extract_critic_vector.py 会同时生成:
#   ./vectors/qwen3-8b/critic.pt      (PCA 纯化后)
#   ./vectors/qwen3-8b/critic_raw.pt  (原始 CAA, 无 PCA)
python extract_critic_vector.py
```

---

## 八、后处理脚本

### 8.1 补全缺失的 EMA/熵轨迹（backfill_ema.py）

适用场景：Baseline / Continuous_Linear / 所有消融组的结果文件中缺少 `ema_trajectory` 字段。
脚本会自动遍历 JSON 中**所有**实验组，按各组的注入方式正确回放 hook。

```bash
# 默认输出到 *_fixed.json（不覆盖原文件）
python backfill_ema.py --json_path ./results/.../experiment_results.json

# 覆盖写回原文件
python backfill_ema.py \
    --json_path ./results/.../experiment_results.json \
    --output_path ./results/.../experiment_results.json

# 只处理指定模式（用于调试）
python backfill_ema.py \
    --json_path ./results/.../experiment_results.json \
    --modes Continuous_Linear Dynamic_Spherical_No_EMA \
    --limit 3
```

> **注意**：`Dynamic_Spherical_No_Manifold` 需要 `critic_raw.pt` 存在；
> `Dynamic_Spherical_No_EMA` 会用 `ema_beta=1.0` 重建瞬时熵作为 EMA。

### 8.2 计算 DTR 和 PPL（evaluate_dtr_offline.py）

覆盖全部实验组（包括消融变体），自动选择正确的 replay 向量。

```bash
python evaluate_dtr_offline.py \
    --results ./results/.../experiment_results.json
```

结果直接写回同一文件，追加 `local_dtr` 和 `ppl` 字段。

---

## 七、论文中的消融声明模板

> *"To ensure a mathematically fair comparison in the 'w/o SLERP' ablation, the linear intervention coefficient α_linear was calibrated via Equal Orthogonal Projection: α_linear = sin(α_slerp × π/2). Thus, α_slerp ∈ {0.3, 0.45} strictly corresponds to α_linear ∈ {0.45, 0.65} scaled by the hidden state norm."*
