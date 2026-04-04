# Continuous Batching 状态管理与调用关系审计报告

本报告旨在梳理推理引擎中各组件的状态生命周期，识别在“槽位复用（Slot Reuse）”场景下必须初始化的变量，并评估潜在的干扰风险。

## 1. 核心状态对象清单

| 组件 | 对象及属性 | 存储位置 | 生命周期/重置机制 | 审计结论 |
| :--- | :--- | :--- | :--- | :--- |
| **KV Cache** | `DynamicCache` (key/value tensors) | `_Slot.past_key_values` | 槽位释放时设为 `None`；`_prefill_slot` 时重新创建。 | **Safe** |
| **状态监控器** | `state.ema_entropy` | `InjectionState` | `_reset_slot_state` 时设置为 `0.0`。 | **Safe** |
| **ThinkBrake** | `state.margin` | `InjectionState` | **[BUG]** 未在 `_reset_slot_state` 中重置。 | **⚠️ Potential Leak** |
| **ThinkBrake** | `state.is_converged` | `InjectionState` | `_reset_slot_state` 时设为 `False`。 | **Safe** |
| **PID 内部状态** | `pid.prev_error` | `PIDController` | `_reset_slot_state` 时设为 `0.0`。 | **⚠️ Initialization Artifact** |
| **PID 内部状态** | `pid.integral` | `PIDController` | `_reset_slot_state` 时设为 `0.0`。 | **Safe** |
| **干扰计数器** | `state.step_count` | `InjectionState` | `_reset_slot_state` 时设为 `0`。 | **Safe** |
| **注入标志位** | `state.trigger_perturbation` | `InjectionState` | `_reset_slot_state` 时设为 `False`。 | **Safe** |
| **轨迹记录** | `ema_trajectory` 等 | `InjectionState` | `_reset_slot_state` 时设为 `[]`。 | **Safe** |
| **活性掩码** | `state.active_mask` | `InjectionState` | 槽位完成时置 `False`，Replenish 后置 `True`。 | **Safe** |

---

## 2. 推理生命周期与调用关系

整个系统围绕 `max_concurrent_seqs` 个物理槽位（Slots）运行。

### A. 初始填充与更换阶段 (`_prefill_slot`)
1. **Reset State**: 调用 `_reset_slot_state(slot_idx)`，清理物理索引对应的 Tensor 切片。
2. **First Forward**: 执行 Prefill（输入全量 Prompt）。
   - **Hook**: 由于 `seq_len > 1`，Hook 自动跳过（防止 Prefill 干扰）。
3. **State Prime**: 手动调用一次 `monitor()`。
   - 计算 Prefill Logits 的熵，初始化 EMA。
   - **Bug Note**: PID 在此步计算时，由于 `prev_error=0`，会产生一个假的 D 项增益。
4. **Sample**: 获取第一个生成的 Token。

### B. 核心解码循环 (`while` loop)
每一轮 Step 同步处理 $K$ 个活跃槽位：
1. **Model Forward**: 传入 $K$ 个槽位的最后一个 Token。
   - **Hook**: 读取 `state.alpha[active_indices]`，执行 SLERP 旋转。
2. **StateMonitor**: 传入 $K$ 个槽位的 Logits。
   - 更新 EMA、Margin。
   - 检查 `is_converged` 锁存。
   - **Controller Step**: 计算新的 `alpha` 并写入 `state.alpha`。
3. **Sample**: 生成 $K$ 个新 Token。

### C. 槽位释放与重组 (`has_finished`)
1. **KB Split**: 将 `batched_pkv` (DynamicCache) 拆回各 Slot 的独立对象。
2. **Harvest**: 识别 `done` 的槽位，`yield` 结果，并在 `slots` 中置空。
3. **Refill**: 立即从 `pending` 队列中取新样本，调用 `_prefill_slot` 填入物理索引。
4. **Resync**: 重建 `active_indices` 和 `active_mask`。

---

## 3. 潜在 BUG 与风险分析

### 3.1 `state.margin` 残留学
- **现象**：`_reset_slot_state` 遗漏了 `margin` 的重置。
- **风险**：虽然 `is_converged` 已重置，但如果新样本在 `monitor()` 更新 margin 之前（或由于某些逻辑分支未进入 margin 计算时）读取了该值，可能导致基于数值的二次判断逻辑出错。它是目前唯一的“不洁”槽位变量。

### 3.2 PID 导数项“开幕雷击”
- **现象**：`prev_error` 初始为 0，第一步计算 `D = kd * (error - 0)`。
- **风险**：每个样本开启干预时的瞬间冲量会偏大。这虽不属于批次间干扰，但属于单样本内的状态初始化不当。

### 3.3 `dummy_logits_buf` 静态风险
- **代码位置**：`run_experiment.py:711`
- **现象**：该 Buffer 在外层分配，内层循环按槽位索引填充。
- **分析**：目前依赖 `state.active_mask` 进行过滤是安全的。但如果未来有组件不经过 `active_mask` 直接遍历 Buffer，则会读到上一个批次残留的 Logits 数据。
- **建议**：在 `monitor()` 调用前，应对 Buffer 进行某种清理，或者确保所有组件都通过物理掩码访问。

### 3.4 隐状态历史记录器 (`history_hidden`)
- **风险**：`steering_hook` 内部的 `history_hidden` 列表是闭包共享的，且没有按槽位索引区分。
- **分析**：目前生产环境下 `capture_hidden_states=False`，所以没有问题。一旦开启该功能进行大规模分析，不同槽位的张量会交织在同一个 List 中，导致数据混乱。

---

## 4. 结论

**系统隔离性目前处于“准优良”状态**。唯一的物理级潜在干扰点是 `state.margin` 的残留。PID 的初始化问题属于控制算法的实现细节调整。建议在后续修复工作中，不仅补充 `margin` 的重置，同时对 PID 的首步逻辑进行“软着陆”处理。
