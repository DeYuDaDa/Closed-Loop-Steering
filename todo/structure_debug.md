# 连续批处理推理管线 (Continuous Batching) 结构与调试指南

本文档旨在梳理当前代码库的执行逻辑，特别是如何运行 `python src/run_experiment.py --dataset ... --modes Dynamic_Spherical` 时的代码路径。这份指南将深至张量和插槽（Slot）级别，可直接用于排查各种诡异的状态越界或泄漏 Bug。

## 1. 宏观启动流程 (run_experiment.py -> main)

当你执行程序时，入口点在 `run_experiment.py` 的 `main()` 或 `run_full_experiment()` 函数。

1. **环境与模型加载：** 加载 dataset，实例化 `AutoModelForCausalLM` 和 `AutoTokenizer`。加载用于控制的 PCA 降维方向向量 (`load_control_vectors`)。
2. **初始化多批次推理：** 为每个选择的 `--modes` 创建后台结果保存线程 (`results_queue`) 和相关的统计字典。
3. **调用生成器生成推理结果：** 进入 `run_continuous_batching_generation()`，它是整个动态连续批处理的核心！

## 2. 连续批处理引擎内部逻辑 (run_continuous_batching_generation)

这是整个代码中最复杂，也是极其容易出 Bug 的地方。
参数 `max_concurrent_seqs` (例如 16) 定义了物理 Slot (插槽) 的大小。这是底层的真实 GPU 吞吐载体。

**A. 全局共享状态分配：**
调用 `_build_global_components()`，创建 **唯一的、长度为 16 的全局共享状态 `InjectionState`**。里面所有张量都是 `[16,]`。
- `state_monitor.StateMonitor` 也会被创建，并且绑定到这同一个 `InjectionState` 上。
- `pid_controller.PIDController` 也被创建。
- `spherical_injector.create_steering_hook` 被创建并注册到模型的特定 Transformer 层（通过 `register_forward_hook`）。

**B. Slot 管理：**
有一个包含 16 个元素的 List `slots`，初始全为 `None`。有一个任务队列 `pending` 包含所有等待推理的题目的 Index。

**C. `_prefill_slot(slot_idx, prompt_idx)` 函数 (危险区!)：**
当有任务进入某个空的 `slot_idx` 时：
1. `_reset_slot_state(slot_idx)`: 将共享的 `state` 里的此 Slot 对应的变量清零重置 (EMA 熵设为 0，active_mask 设为 True 等等)。
2. 进行第一次的 Model Forward (无 cache)。这被称为 Prefill 阶段。
3. 得到 `first_logits_2d`。之后手动调用一次 `monitor(dummy_ids, dummy_logits)` 为这个 newly prefilled token 计算第一步的控制变量 (如 Alpha)。
   **👉 关键隔离点：** 此时只希望影响当前的 `slot_idx`，因此在调用前有一句 `saved_mask = state.active_mask.clone(); state.active_mask.fill_(False); state.active_mask[slot_idx] = True`。调用完再恢复。这是我们之前修过的越界泄漏的核心点！如果忘加 `active_mask` 的判定，EMA 会污染整个 [16,] 的向量！

**D. 主解码循环 (Hot-path):**
循环直到 `slots` 全是 `None`。
每次经过 1 步 (1 Token generation)：
1. 找出当前所有活着的 Slot: `active_indices`。
2. `state.active_batch_indices = active_indices` (告知 Hook 现在真正活着的插槽物理索引，这样它可以通过 `alpha[self.state.active_batch_indices]` 取出对应的控制力)。
3. 获取这 `K` 个存活序列的当前最后一个 token IDs `batched_last`，并组合 KV Cache（`batched_pkv`）。
4. **模型 Forward！** 模型仅对这 `K` 个 tokens 进行推理（不会浪费没有用的空槽计算）。
5. 获取输出的 Logits `logits_K`。
6. **调用监控器 (Watchdog + PD Controller)！**
   准备 `ids_buf` 包裹这 K 个序列的真实 Input IDs，准备 `dummy_logits_buf` 把这 K 个 logits 放回它们真正的 0-15 的物理位置！然后调用 `monitor(ids_buf, dummy_logits_buf)`。
   这里的逻辑极大简化了 `state_monitor.py` 内的对齐负担，`StateMonitor` 只需无脑按照 `[16]` 物理长度处理，用 `self.state.active_mask` 来屏蔽无需处理的位置。
7. 根据 Logits，用 `_sample_batch_tokens` 抽样出下一个 Token。如果触发了 EOS，对应的 slot 标记结束，进行 KV Cache 的切割剥离 `_unpad_and_split_kv_caches`。
8. 循环补上空缺的 pending 题目 (又调用 `_prefill_slot`)。

## 3. 防坍塌流形扰动系统 (StateMonitor + Injector) 逻辑环路

现在我们来梳理，随着 Continuous Batching 喷出每一个 token，我们的干预模块是如何相互协同的。

**第一步：监控站 (`StateMonitor.__call__`)，每生成 1 个词执行一次**
输入的是 `[16, V]` 大小的 logits 张量，此时 `active_mask` 为 True 的位置包含真实的 Logits。
1. 计算当前 token 的瞬时熵 $H_t$。计算出当前 EMA 熵，并存在 `self.state.ema_entropy`。
2. 判断 `ThinkBrake` 是否触发，如果边际太低判定收敛（`self.state.is_converged = True`）。
3. 调用 `PIDController` 给出一个试图把当前流形拉向目标空间的意愿度 `alpha`。
4. **防坍塌检测中心 (Anti-Collapse Watchdog)：**
   - 如果 $H_t$ < 0.02 并且该 slot 活跃，`low_entropy_count` 计时器累加。
   - 对这个批次的每条语句截取尾部，找 N-Gram。比如对于长度为 $3$ 的段，看后面 6 个词构没构成重复！
   - 如果两败俱伤 (熵极低还复读超过了忍耐度)，那么判定进入无限死亡复读泥潭。给这个槽的 `trigger_perturbation` 盖章！同时上报该槽进入冷却期 `cooldown_counter`。
   - **一旦进入冷却期或本回合要执行扰动踢（kick），强制把 PID 的这回合 `alpha` 削成 0！** 不能让大车的惯性干扰我们的精准修正。
5. 保存轨迹，然后继续生成管线。

**第二步：拦截钩 (steering_hook)，伴随内部 Forward() 被触发**
这是一个 PyTorch Forward Hook。在每一层计算结束时触发。
由于注册在我们规定的层 (LAYER_ID = 24)，所以在经过了 24 层注意力和前馈层后，这个钩子拿到了长为 `[K, 1, d]` 的隐状态 `hidden`。
1. 通过 `state.active_batch_indices` 获取活跃状态的索引，把 `[16]` 长的 `alpha` 和 `trigger_perturbation` 缩减为当前的纯净 `[K]` 大小的 `alpha` 和 `trigger_mask`。
2. 先通过 SLERP 获取常规拉扯后的目标 `h_new`（前提是有干预强度 `alpha > 0` 且没被判定坍塌）。
3. 如果 `has_perturb==True`（刚才有任意一个 slot 被判定坍塌），强制生成一团 `torch.randn_like` 的白噪音。
4. **流形正交踢击 (Gram-Schmidt):** 将噪音投影到与 `h_new` 当前隐空间流形上截然无关的方向上（作减法），归一化得到 `z_unit`。
5. 朝这个无关痛痒但也绝不重复的方向猛踹一脚 `gamma` 力量，并在保证 L2-Norm 原来的范数的前提下获得 `h_pushed`。
6. 把踢过的状态覆盖进 `trigger` 为 True 的那一部分，然后销毁触发标志（重置刚才这个 slot 的 `trigger_perturbation` 为 False，它已经踢完了！）。

**第三步：下一回合**
在未来几个 Token `cooldown_counter` 会一直拦截 PID。让微扰能够自然散开，形成完全迥异的高纬流形！

## 调试排查清单 (Cross-Slot State Infection Checklists)

如果在增加或修改代码时导致模型答不出题，或者性能突然崩溃。可以通过这几点打断点（`import pdb; pdb.set_trace()`）：

1. **_prefill_slot 不可污染环境**
   在 `run_experiment.py` 的 `_prefill_slot()` 底下，`monitor(dummy_ids, dummy_logits)` 的调用前后，必须只有 `state.active_mask[slot_idx] == True`。如果其他的槽也是 True，会导致那些明明还在生成的题目突然被强行切成了 EMA_entropy=计算出来的值。这会引发大雪崩！

2. **monitor() 的张量操作不能有 Indexing 缺陷**
   诸如 `is_low_entropy = (H_t < COLLAPSE_ENTROPY_MIN) & self.state.active_mask` 这些操作，一定不要忘记加上 `& self.state.active_mask`！
   因为一旦 `StateMonitor` 中少了一个 mask 过滤，比如：
   `self.state.low_entropy_count += 1`
   那些已经结束了、还在等下一题补充进来的僵尸 Slot，它们传进来的 Dummy Logits 全是 0，计算出的假熵永远低于 0.02。这会让空闲槽的计数器暴涨。虽然空闲没什么大不了的，但是一旦它被拿去 prefill 变活，那个值就没有被清零，一开始就爆炸！

3. **run_continuous 的 `_reset_slot_state` 必须大扫除**
   每一个写在 `InjectionState` 内部，有着 `[self.batch_size]` 长度的张量，在下一道题上机进入这个物理座位 (slot) 时，必须在 `_reset_slot_state` 里被**彻底清洗重置** (`=0` 或 `=False`)！
   比如新增了 `low_entropy_count` 就一定要记得在这里重置，否则上一题因为死结被踢，这题前三步如果刚好熵低了点马上也被踢飞了，属于被波及的无妄之灾。

4. **steering_hook 中 Tensor 维度的统一选取：**
   `state.trigger_perturbation` 是 `[16]`, 而此时正在执行 forward 的 `h` 只有存活的 `[K, 1, d]`！它们不是一个长度！
   所以必须通过 `state.active_batch_indices` 进行桥接：
   `trigger_mask = state.trigger_perturbation[state.active_batch_indices]`
   这就是为什么写 Hook 时千万要注意形状。一旦这里错配（直接拿去相乘或者寻址），就会报错退出或者踢错人。
