"""
Extract Grounding Vector — Gradient-Based Optimization
=======================================================
寻找一个最小扰动向量 V，注入到模型第 L 层后，
最大化"严谨推导轨迹"的预测概率，同时保持 V 的范数尽量小。

目标函数（单目标，稳定版）：
    L = CE(H_L + V, Y_grounded) + λ · ||V||²

    ·  只优化正样本方向，不计算负样本对比 Loss。
    ·  使用 L2 平方正则（||V||²）而非线性 L2 （||V||），
       平方正则的梯度随范数线性增长，能有效抑制范数爆炸。
    ·  当 ||V|| 超过 norm_ceiling 时触发早停，防止踹出流形空间。

模型全参数冻结，只优化 V（shape = [hidden_dim]）。

Usage:
    cd /path/to/src
    python extract_grounding_vector.py [options]

    # 示例
    python extract_grounding_vector.py \\
        --target_layer 24 \\
        --learning_rate 0.01 \\
        --l2_penalty 0.1 \\
        --max_epochs 80

Outputs (saved to VECTOR_DIR):
    math_grounding.pt          — L2 归一化向量（与 critic.pt 格式一致）
    math_grounding_raw.pt      — 未归一化版本（保留 norm 信息）
    math_grounding_meta.json   — 训练日志（loss 曲线、最终 norm、超参）
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── local imports (不修改这些模块) ────────────────────────────────────────────
from config import MODEL_PATH, LAYER_ID, VECTOR_DIR, ATTN_IMPLEMENTATION
from loaders.aime_loader import build_aime_prompt


# ════════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Gradient-based Contextual Grounding Steering Vector Extraction"
    )
    p.add_argument("--target_layer",  type=int,   default=LAYER_ID,
                   help=f"注入层编号，默认 {LAYER_ID}（config.LAYER_ID）")
    p.add_argument("--learning_rate", type=float, default=0.01,
                   help="Adam 学习率，默认 0.01")
    p.add_argument("--l2_penalty",    type=float, default=0.1,
                   help="L2² 正则权重 λ（作用于 ||V||²），默认 0.1")
    p.add_argument("--max_epochs",    type=int,   default=80,
                   help="最大训练轮数，默认 80")
    p.add_argument("--norm_floor",    type=float, default=0.01,
                   help="下限早停：||V|| < 此值时停止（防止 V 收缩到零），默认 0.01")
    p.add_argument("--norm_ceiling",  type=float, default=30.0,
                   help="上限早停：||V|| > 此值时停止（防止范数爆炸），默认 30.0")
    p.add_argument("--data_dir",      type=str,   default="./dataset/train",
                   help="训练数据根目录，默认 ./dataset/train")
    p.add_argument("--vector_dir",    type=str,   default=VECTOR_DIR,
                   help=f"向量保存目录，默认 config.VECTOR_DIR = {VECTOR_DIR}")
    p.add_argument("--seed",          type=int,   default=42,
                   help="随机种子，默认 42")
    p.add_argument("--max_seq_len",   type=int,   default=2048,
                   help="分词最大序列长度，默认 2048")
    p.add_argument("--log_interval",  type=int,   default=10,
                   help="每隔多少轮打印一次日志，默认 10")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════════
#  Dataset Loading
# ════════════════════════════════════════════════════════════════════════════════

def _read_txt(path: str) -> str:
    """读取 TXT 文件，多行合并为单个字符串，并 strip 首尾空白。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_samples(data_dir: str) -> list[dict]:
    """
    遍历 data_dir/{1,2,...}/，读取 problem.txt / positive.txt / negative.txt。
    或如果 data_dir 指向一个 .jsonl 文件，将读取并解析 jsonl 数据。

    Returns:
        List of dicts:
            {
                "idx":      int,
                "problem":  str,   # 题干（context）
                "positive": str,   # 严谨推导轨迹
                "negative": str | None,  # 瞎蒙轨迹（文件为空则为 None）
            }
    """
    samples = []
    data_path = Path(data_dir)
    
    if data_path.is_file() and data_path.suffix == ".jsonl":
        print(f"  📂 检测到 JSONL 数据集：{data_path.name}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip(): continue
                item = json.loads(line)
                if item.get("category") != "math":
                    continue
                
                problem = item.get("problem", "")
                thinking = item.get("thinking", "")
                solution = item.get("solution", "")
                
                if not problem or not thinking:
                    continue
                
                positive = f"<think>\n{thinking}\n</think>\n\n{solution}".strip()
                samples.append({
                    "idx": item.get("id", idx),
                    "problem": problem,
                    "positive": positive,
                    "negative": None
                })
                # 提取过程日志太多会刷屏，这里不逐个打印，交由后续阶段打印统计即可
                
    elif data_path.is_dir():
        # 按编号排序遍历子目录
        subdirs = sorted(
            [d for d in data_path.iterdir() if d.is_dir()],
            key=lambda d: int(d.name) if d.name.isdigit() else 0
        )

        for subdir in subdirs:
            prob_path = subdir / "problem.txt"
            pos_path  = subdir / "positive.txt"
            neg_path  = subdir / "negative.txt"

            if not prob_path.exists() or not pos_path.exists():
                print(f"  ⚠️  {subdir.name}: 缺少 problem.txt 或 positive.txt，跳过")
                continue

            problem  = _read_txt(str(prob_path))
            positive = _read_txt(str(pos_path))

            # 负样本：文件不存在或内容为空则设为 None
            negative = None
            if neg_path.exists():
                neg_text = _read_txt(str(neg_path))
                if neg_text:
                    negative = neg_text

            samples.append({
                "idx":      int(subdir.name) if subdir.name.isdigit() else str(subdir.name),
                "problem":  problem,
                "positive": positive,
                "negative": negative,
            })
            neg_status = "✅ 有负样本" if negative else "⚪ 无负样本"
            print(f"  [{subdir.name}] problem={len(problem)}chars  "
                  f"pos={len(positive)}chars  {neg_status}")
    else:
        raise FileNotFoundError(f"数据源不存在或不合法：{data_dir}")

    if not samples:
        raise ValueError(f"数据源中没有有效样本：{data_dir}")

    print(f"\n  共加载 {len(samples)} 个样本，"
          f"其中 {sum(1 for s in samples if s['negative'])} 个含负样本。\n")
    return samples


# ════════════════════════════════════════════════════════════════════════════════
#  Tokenization & Label Masking
# ════════════════════════════════════════════════════════════════════════════════

def build_token_pair(
    problem: str,
    trajectory: str,
    tokenizer,
    max_seq_len: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构建 (input_ids, labels) 对，其中 context 部分的 label 被设为 -100。

    流程：
        1. 用 build_aime_prompt(problem) 获取 chat messages
        2. apply_chat_template(..., add_generation_prompt=True) 获取 context token 串
        3. 将 trajectory 追加到 context 后，整体 tokenize
        4. context 部分的 labels 设为 -100（不计算 Loss）
        5. trajectory 部分的 labels 与 input_ids 相同

    Returns:
        input_ids: [seq_len]
        labels:    [seq_len]  (context 部分为 -100)
    """
    # Step 1 & 2：构建 context（题干 prompt，不含 trajectory）
    messages = build_aime_prompt(problem)
    context_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # 末尾加 <|im_start|>assistant\n
    )

    # Step 3：tokenize context 和 full sequence（context + trajectory）
    #   使用 add_special_tokens=False 避免重复 BOS
    ctx_ids  = tokenizer.encode(context_str, add_special_tokens=False)
    full_str = context_str + trajectory
    full_ids = tokenizer.encode(full_str,    add_special_tokens=False,
                                truncation=True, max_length=max_seq_len)

    input_ids = torch.tensor(full_ids, dtype=torch.long)

    # Step 4 & 5：Label Masking
    labels = input_ids.clone()
    ctx_len = len(ctx_ids)
    if ctx_len >= len(labels):
        # 极端情况：context 已占满，没有可学习的 trajectory token
        print("  ⚠️  Context 超过 max_seq_len，此样本将跳过 loss 计算。")
        labels[:] = -100
    else:
        labels[:ctx_len] = -100   # mask context

    return input_ids, labels


# ════════════════════════════════════════════════════════════════════════════════
#  Vector Optimizer — Hook 注入 + 梯度优化
# ════════════════════════════════════════════════════════════════════════════════

class VectorOptimizer:
    """
    持有可训练向量 V，通过 Forward Hook 将其注入模型第 target_layer 层。
    使用 Adam 优化 V，模型全参数冻结。
    """

    def __init__(
        self,
        model,
        hidden_dim: int,
        target_layer: int,
        lr: float,
        l2_penalty: float,
    ):
        self.model        = model
        self.target_layer = target_layer
        self.l2_penalty   = l2_penalty
        self._hook_handle: Optional[object] = None

        # ── 冻结模型 ──────────────────────────────────────────────────
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        # ── 可训练向量 V（shape [hidden_dim]，float32，初始全零）──────
        self.V = torch.zeros(hidden_dim, dtype=torch.float32,
                             requires_grad=True,
                             device=next(model.parameters()).device)

        # ── 优化器 ───────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam([self.V], lr=lr)

    # ── Hook 管理 ────────────────────────────────────────────────────

    def _make_hook(self):
        """创建在 target_layer 注入 V 的 forward hook。"""
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # V 广播到 [batch, seq, hidden_dim]
            v = self.V.to(dtype=hidden.dtype, device=hidden.device)
            hidden = hidden + v
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    def register_hook(self):
        layer = self.model.model.layers[self.target_layer]
        self._hook_handle = layer.register_forward_hook(self._make_hook())

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ── Loss 计算 ────────────────────────────────────────────────────

    def compute_loss(
        self,
        input_ids_pos: torch.Tensor,   # [seq_len]
        labels_pos:    torch.Tensor,   # [seq_len]
    ) -> tuple[torch.Tensor, dict]:
        """
        单目标稳定版目标函数：
            L = CE(H_L + V, Y_pos) + λ · ||V||²

        · 只优化正样本，使模型在注入 V 后更倾向于严谨推导。
        · L2 平方正则（非线性 L2），梯度 = 2λV，随范数增大而同步增大，
          能自然地平衡 CE_pos 梯度，防止范数爆炸。
        · 负样本 Loss 已完全移除（对比惩罚会导致优化器"傻瓜化"模型）。

        Returns:
            total_loss: scalar tensor (梯度已附在 V 上)
            info:       dict，记录各分项 loss 数值
        """
        device = next(self.model.parameters()).device

        # ── 正样本前向 ────────────────────────────────────────────────
        ids_pos = input_ids_pos.unsqueeze(0).to(device)   # [1, seq]
        lbl_pos = labels_pos.unsqueeze(0).to(device)      # [1, seq]

        out_pos = self.model(input_ids=ids_pos, labels=lbl_pos)
        loss_pos = out_pos.loss   # HuggingFace 计算 CE（已 mask -100）

        # ── L2² 正则：λ · ||V||² ─────────────────────────────────────
        # 平方正则梯度 = 2λV，相比线性 L2（梯度 = λ·V/||V||）
        # 其抑制力随范数线性增大，可有效对抗范数爆炸。
        v_norm_sq = (self.V.float() ** 2).sum()   # ||V||²（标量，保留梯度图）
        l2_sq_term = v_norm_sq                     # 用于 loss（乘以 λ 在 total 中）
        l2_norm    = v_norm_sq.sqrt().item()       # 仅用于日志记录（无梯度）

        # ── 总 Loss ───────────────────────────────────────────────────
        total = loss_pos + self.l2_penalty * l2_sq_term

        info = {
            "loss_pos": loss_pos.item(),
            "l2_norm":  l2_norm,
            "total":    total.item(),
        }
        return total, info

    # ── 优化步 ───────────────────────────────────────────────────────

    def step(
        self,
        input_ids_pos: torch.Tensor,
        labels_pos:    torch.Tensor,
        **_kwargs,   # 忽略 input_ids_neg / labels_neg（保持调用兼容性）
    ) -> dict:
        """执行一步梯度更新，返回 loss 信息字典。"""
        self.optimizer.zero_grad()
        loss, info = self.compute_loss(input_ids_pos, labels_pos)
        loss.backward()
        self.optimizer.step()
        info["V_norm"] = self.V.detach().float().norm().item()
        return info


# ════════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════════════════════

def train_loop(
    optimizer: VectorOptimizer,
    token_pairs: list[dict],
    max_epochs: int,
    norm_floor: float,
    norm_ceiling: float,
    log_interval: int,
) -> list[dict]:
    """
    主训练循环：每个 epoch 遍历所有样本，mini-batch=1 独立 step。

    早停条件（双向保护）：
        · ||V|| < norm_floor   → V 趋于零，l2_penalty 太大，请降低
        · ||V|| > norm_ceiling → 范数爆炸，l2_penalty 太小或 lr 太大

    Returns:
        history: List of per-step info dicts.
    """
    history = []
    global_step = 0
    n_samples = len(token_pairs)

    print(f"\n{'='*60}")
    print(f"  开始优化 V  —  max_epochs={max_epochs}  n_samples={n_samples}")
    print(f"  norm_floor={norm_floor}  norm_ceiling={norm_ceiling}")
    print(f"  目标函数：L = CE_pos + λ·||V||²  （无对比 Loss）")
    print(f"{'='*60}\n")

    for epoch in range(1, max_epochs + 1):
        epoch_info_list = []

        for pair in token_pairs:
            info = optimizer.step(
                input_ids_pos = pair["input_ids_pos"],
                labels_pos    = pair["labels_pos"],
            )
            info["epoch"] = epoch
            info["step"]  = global_step
            history.append(info)
            epoch_info_list.append(info)
            global_step += 1

        # ── 每个 epoch 打印汇总 ───────────────────────────────────────
        if epoch % log_interval == 0 or epoch == 1:
            avg_total = sum(i["total"]    for i in epoch_info_list) / n_samples
            avg_pos   = sum(i["loss_pos"] for i in epoch_info_list) / n_samples
            v_norm    = epoch_info_list[-1]["V_norm"]
            print(f"  Epoch {epoch:>4d}/{max_epochs}  |  "
                  f"Loss={avg_total:.4f}  CE_pos={avg_pos:.4f}  "
                  f"||V||={v_norm:.4f}")

        # ── 双向早停 ──────────────────────────────────────────────────
        current_norm = optimizer.V.detach().float().norm().item()
        if current_norm < norm_floor:
            print(f"\n  ⚠️  [下限早停] ||V||={current_norm:.4f} < norm_floor={norm_floor}")
            print(f"       V 趋于零，请降低 --l2_penalty 后重试。")
            break
        if current_norm > norm_ceiling:
            print(f"\n  ⚠️  [上限早停] ||V||={current_norm:.4f} > norm_ceiling={norm_ceiling}")
            print(f"       范数爆炸，已中止。请升高 --l2_penalty 或降低 --learning_rate。")
            break

    print(f"\n  ✅ 优化完成  共 {global_step} 步  最终 ||V|| = "
          f"{optimizer.V.detach().float().norm().item():.4f}")
    return history


# ════════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("  Grounding Vector Extraction — Gradient Optimization (单目标稳定版)")
    print("=" * 70)
    print(f"  target_layer   = {args.target_layer}")
    print(f"  learning_rate  = {args.learning_rate}")
    print(f"  l2_penalty (λ) = {args.l2_penalty}  [作用于 ||V||²]")
    print(f"  max_epochs     = {args.max_epochs}")
    print(f"  norm_floor     = {args.norm_floor}")
    print(f"  norm_ceiling   = {args.norm_ceiling}")
    print(f"  data_dir       = {args.data_dir}")
    print(f"  vector_dir     = {args.vector_dir}")
    print()

    # ── Step 1: 加载数据 ───────────────────────────────────────────────────────
    print("[Step 1] 加载训练数据...")
    samples = load_samples(args.data_dir)

    # ── Step 2: 加载模型 ───────────────────────────────────────────────────────
    print(f"[Step 2] 加载模型：{MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.eval()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    print(f"  模型设备：{device}  hidden_dim：{hidden_dim}")

    # ── Step 3: 分词 + Label Masking ──────────────────────────────────────────
    print("\n[Step 3] Tokenize + Label Masking...")
    token_pairs = []
    skipped = 0

    for s in samples:
        ids_pos, lbl_pos = build_token_pair(
            s["problem"], s["positive"], tokenizer, args.max_seq_len
        )

        # 检查是否有可学习的正样本 token（防止 context 占满的极端情况）
        if (lbl_pos != -100).sum().item() == 0:
            print(f"  ⚠️  样本 {s['idx']} 正样本 trajectory 全被截断，跳过。")
            skipped += 1
            continue

        pair = {
            "input_ids_pos": ids_pos,
            "labels_pos":    lbl_pos,
        }

        if s["negative"] is not None:
            ids_neg, lbl_neg = build_token_pair(
                s["problem"], s["negative"], tokenizer, args.max_seq_len
            )
            if (lbl_neg != -100).sum().item() > 0:
                pair["input_ids_neg"] = ids_neg
                pair["labels_neg"]    = lbl_neg
            else:
                print(f"  ⚠️  样本 {s['idx']} 负样本 trajectory 全被截断，跳过对比 Loss。")

        token_pairs.append(pair)
        n_pos  = (lbl_pos != -100).sum().item()
        n_neg  = (pair.get("labels_neg", torch.tensor([])) != -100).sum().item()
        print(f"  [{s['idx']:>2}] pos_traj_tokens={n_pos}  neg_traj_tokens={n_neg}")

    if not token_pairs:
        print("❌ 没有有效的训练样本，退出。")
        sys.exit(1)

    print(f"\n  有效样本：{len(token_pairs)}（跳过 {skipped}）")
    print(f"  ℹ️  负样本仅用于记录（对比 Loss 已禁用），训练只用正样本。")

    # ── Step 4: 构建优化器 + 注册 Hook ────────────────────────────────────────
    print(f"\n[Step 4] 构建 VectorOptimizer（target_layer={args.target_layer}）...")
    vec_opt = VectorOptimizer(
        model        = model,
        hidden_dim   = hidden_dim,
        target_layer = args.target_layer,
        lr           = args.learning_rate,
        l2_penalty   = args.l2_penalty,
    )
    vec_opt.register_hook()
    print("  ✅ Hook 已注册")

    # ── Step 5: 训练 ──────────────────────────────────────────────────────────
    history = train_loop(
        optimizer    = vec_opt,
        token_pairs  = token_pairs,
        max_epochs   = args.max_epochs,
        norm_floor   = args.norm_floor,
        norm_ceiling = args.norm_ceiling,
        log_interval = args.log_interval,
    )

    # ── Step 6: 移除 Hook，保存结果 ──────────────────────────────────────────
    vec_opt.remove_hook()

    V_final = vec_opt.V.detach().cpu().float()
    v_norm  = V_final.norm().item()

    os.makedirs(args.vector_dir, exist_ok=True)

    # Raw（未归一化）
    raw_path = os.path.join(args.vector_dir, "math_grounding_raw.pt")
    torch.save(V_final, raw_path)
    print(f"\n  💾 Raw vector saved → {raw_path}  (norm={v_norm:.4f})")

    # Normalized（与 critic.pt 格式一致：L2 单位向量）
    V_unit = V_final / V_final.norm().clamp_min(1e-8)
    norm_path = os.path.join(args.vector_dir, "math_grounding.pt")
    torch.save(V_unit, norm_path)
    print(f"  💾 Unit  vector saved → {norm_path}  (norm={V_unit.norm().item():.4f})")

    # 元数据
    meta = {
        "target_layer":  args.target_layer,
        "learning_rate": args.learning_rate,
        "l2_penalty":    args.l2_penalty,
        "loss_formula":  "CE_pos + lambda * ||V||^2",
        "max_epochs":    args.max_epochs,
        "norm_floor":    args.norm_floor,
        "norm_ceiling":  args.norm_ceiling,
        "n_samples":     len(token_pairs),
        "final_V_norm":  v_norm,
        "hidden_dim":    hidden_dim,
        "model_path":    MODEL_PATH,
        "loss_history": [
            {k: round(val, 6) if isinstance(val, float) else val
             for k, val in h.items()}
            for h in history[::max(1, len(history) // 200)]  # 最多记录 200 个点
        ],
    }
    meta_path = os.path.join(args.vector_dir, "math_grounding_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  💾 Meta  saved       → {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  提取完成 — Summary")
    print(f"{'='*70}")
    print(f"  Target layer:      {args.target_layer}")
    print(f"  Hidden dim:        {hidden_dim}")
    print(f"  Training steps:    {len(history)}")
    print(f"  Final ||V||:       {v_norm:.4f}")
    first_loss = history[0]["total"]   if history else float("nan")
    last_loss  = history[-1]["total"]  if history else float("nan")
    print(f"  Loss: {first_loss:.4f} → {last_loss:.4f}")
    print(f"\n  Output files:")
    print(f"    {raw_path}")
    print(f"    {norm_path}")
    print(f"    {meta_path}")
    print(f"\n  🎉 可使用 probe_vectors.py 或 spherical_injector.py 验收向量。")
    print(f"     修改 probe_vectors.py 中 VECTOR_DIR 路径后，")
    print(f"     将加载名称从 'critic_raw.pt'/'critic.pt' 换为")
    print(f"     'math_grounding_raw.pt'/'math_grounding.pt' 即可。")


if __name__ == "__main__":
    main()
