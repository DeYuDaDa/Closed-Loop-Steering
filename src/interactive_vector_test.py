"""
interactive_vector_test.py
==========================
交互式向量干预测试工具。

加载模型和控制向量，以固定强度持续使用 Spherical Steering 进行干预。
用户可以实时输入问题，程序实时流式输出生成内容，同时将以下信息
写入日志文件（可随时 Ctrl+C 中断）：
  - 每步生成的 token
  - Top-K=20 的 token 及概率（格式对齐的三元组，每行一条）

日志行格式（每 step 一行）：
  step | gen_token | [top20 tokens] | [top20 probs]
"""

import os
import sys
import datetime
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODEL_PATH,
    VECTOR_DIR,
    DEFAULT_DTYPE,
    DEVICE_MAP,
    LAYER_ID,
    DO_SAMPLE,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    MIN_P,
    ENABLE_THINKING,
    SAFE_SCORE_RANGE,
    ENDOFTEXT_ID,
    ATTN_IMPLEMENTATION,
)
from run_experiment import (
    load_control_vectors,
    _sample_batch_tokens,
    _safe_score_range_clean,
)
from state_monitor import InjectionState
from spherical_injector import create_steering_hook

# ═══════════════════════════════════════════════
# 配置项
# ═══════════════════════════════════════════════
TOP_K_LOG = 5          # 日志中记录的 top-k 个数
MAX_NEW_TOKENS = 4096*8   # 单次回复最大 token 数 (超过自动截断)
LOG_PATH = "interactive_vector_log.txt"  # 日志路径

# ════════════════════════════════════════════════════════
# Part 2: 模型加载 & Hook 装配
# ════════════════════════════════════════════════════════

def load_model():
    """加载模型与 tokenizer，并配置 pad_token。"""
    print("📦  Loading model from:", MODEL_PATH)
    model_dtype = getattr(torch, DEFAULT_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=DEVICE_MAP,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 确保 pad_token 与 eos_token 不同（Qwen3 兼容）
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == eos_id:
        tokenizer.pad_token_id = ENDOFTEXT_ID
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(ENDOFTEXT_ID)

    print("✅  Model loaded.  Device:", model.device)
    return model, tokenizer


def setup_hook(model, control_vector, alpha: float):
    """
    装配固定强度的 Spherical Steering Hook。

    Returns:
        hook_handle : 用于 .remove() 的句柄
        state       : InjectionState 对象
    """
    device = model.device
    state = InjectionState(batch_size=1, device=device)
    state.intervention_active.fill_(True)
    state.active_mask.fill_(True)
    state.alpha.fill_(alpha)
    state.active_batch_indices = None   # hook 对整批生效（batch=1）

    hook_fn, _ = create_steering_hook(
        state=state,
        control_vector=control_vector,
        mode="Continuous",
        continuous_alpha=alpha,
        continuous_linear_alpha=0.0,
        capture_hidden_states=False,
    )
    layer = model.model.layers[LAYER_ID]
    hook_handle = layer.register_forward_hook(hook_fn)
    print(f"🔗  Hook mounted → layer {LAYER_ID}, alpha={alpha}")
    return hook_handle, state


# ════════════════════════════════════════════════════════
# Part 3: 日志工具
# ════════════════════════════════════════════════════════

def _tok_repr(tokenizer, tok_id: int, max_len: int = 12) -> str:
    """将 token id 解码为可打印的 repr 字符串（截断到 max_len）。"""
    try:
        s = tokenizer.decode([tok_id])
        r = repr(s)[1:-1]          # 去掉 repr 外层引号
        if len(r) > max_len:
            r = r[:max_len - 1] + "…"
        return r
    except Exception:
        return f"<{tok_id}>"


def write_log_header(f, question: str, alpha: float) -> None:
    """写入每次问题的日志头。"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write("\n")
    f.write("=" * 100 + "\n")
    f.write(f"[{ts}]  alpha={alpha}\n")
    f.write(f"Q: {question}\n")
    f.write("-" * 100 + "\n")
    # 列头
    f.write(
        f"{'step':>5}  {'gen_token':<16}  "
        f"{'top-20 tokens':<150}  top-20 probs\n"
    )
    f.write("-" * 100 + "\n")
    f.flush()


def write_log_step(
    f,
    step: int,
    gen_tok_id: int,
    topk_ids,
    topk_probs,
    tokenizer,
) -> None:
    """
    写入单步日志，格式严格对齐：
      step | gen_token(repr,左对齐16) | [tok0, tok1, ...](每格14) | [p0, p1, ...](每格6)
    """
    gen_repr = _tok_repr(tokenizer, gen_tok_id, max_len=14)

    tok_parts  = []
    prob_parts = []
    for tid, p in zip(topk_ids, topk_probs):
        tok_r = _tok_repr(tokenizer, tid.item() if hasattr(tid, "item") else int(tid), max_len=12)
        tok_parts.append(f"'{tok_r}'".ljust(14))
        prob_parts.append(f"{p:.3f}".rjust(6))

    tok_str  = "[" + " ".join(tok_parts)  + "]"
    prob_str = "[" + " ".join(prob_parts) + "]"

    line = f"{step:>5}  {gen_repr:<16}  {tok_str:<150}  {prob_str}\n"
    f.write(line)
    f.flush()


# ════════════════════════════════════════════════════════
# Part 4: 单次生成（流式输出 + 实时写日志）
# ════════════════════════════════════════════════════════

def generate_one(
    model,
    tokenizer,
    question: str,
    alpha: float,
    log_file,
    eos_id: int,
) -> None:
    """
    对单个问题进行一次完整的自回归生成。
    - 实时 print 每个 token（无换行刷新）
    - 每 step 实时写入 log_file
    - 可被 KeyboardInterrupt 中断，中断后安全返回
    """
    device = model.device

    # ── 构建 chat prompt ──────────────────────────────────
    messages = [{"role": "user", "content": question}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids     = enc.input_ids        # [1, prompt_len]
    attention_mask = enc.attention_mask  # [1, prompt_len]

    # ── 写日志头 ──────────────────────────────────────────
    write_log_header(log_file, question, alpha)

    print("Assistant: ", end="", flush=True)

    # ── Prefill ───────────────────────────────────────────
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    past_key_values = out.past_key_values

    step = 0
    try:
        while step < MAX_NEW_TOKENS:
            step += 1

            # ── 取 logits 并做安全清洗 ──────────────────
            logits = out.logits[:, -1, :]          # [1, V]
            logits = _safe_score_range_clean(logits, eos_id)

            # ── 计算 Top-K 概率（用于日志）──────────────
            probs_all = torch.softmax(logits[0].float(), dim=-1)
            topk_probs, topk_ids = torch.topk(probs_all, TOP_K_LOG)

            # ── 采样下一个 token ─────────────────────────
            next_tok = _sample_batch_tokens(
                logits, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P
            )  # [1, 1]
            gen_id = next_tok[0, 0].item()

            # ── 实时写日志 ───────────────────────────────
            write_log_step(
                log_file, step, gen_id,
                topk_ids.tolist(), topk_probs.tolist(),
                tokenizer,
            )

            # ── 实时 CLI 输出 ────────────────────────────
            print(tokenizer.decode([gen_id]), end="", flush=True)

            # ── EOS 检测 ─────────────────────────────────
            if gen_id == eos_id:
                break

            # ── 追加 token，准备下一步 ───────────────────
            input_ids     = torch.cat([input_ids,     next_tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)],
                dim=1,
            )

            with torch.no_grad():
                out = model(
                    input_ids=next_tok,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = out.past_key_values

    except KeyboardInterrupt:
        log_file.write(f"\n[Interrupted at step {step}]\n")
        log_file.flush()
        print("\n[⚡ Interrupted]", flush=True)
        return

    print()  # 换行
    log_file.write(f"[Done at step {step}]\n")
    log_file.flush()


# ════════════════════════════════════════════════════════
# Part 5: 主入口 —— 交互式 REPL
# ════════════════════════════════════════════════════════

def main():
    # ── 加载模型 ────────────────────────────────────────
    model, tokenizer = load_model()
    device = model.device

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    # ── 加载控制向量 ─────────────────────────────────────
    print("\n🔬  Loading control vectors from:", VECTOR_DIR)
    control_vectors = load_control_vectors(
        VECTOR_DIR,
        device=device,
        dtype=getattr(torch, DEFAULT_DTYPE),
    )
    # 优先使用 purified，其次 raw
    vector = control_vectors.get("purified")
    if vector is None:
        vector = control_vectors.get("raw")
        
    if vector is None:
        print("❌  No control vector found — cannot apply steering. Exiting.")
        return
    print(f"✅  Using vector: {'purified' if 'purified' in control_vectors else 'raw'}")

    # ── 输入干预强度 ──────────────────────────────────────
    while True:
        raw = input("\nEnter steering alpha (-1.0 ~ 1.0, e.g. 0.3): ").strip()
        try:
            alpha = float(raw)
            if -1.0 <= alpha <= 1.0:
                break
            print("  ⚠️  Alpha must be in [-1.0, 1.0], try again.")
        except ValueError:
            print("  ⚠️  Invalid number, try again.")

    # ── 装配 Hook ─────────────────────────────────────────
    hook_handle, _ = setup_hook(model, vector, alpha)

    # ── 打开日志文件（追加模式）───────────────────────────
    log_abs = os.path.join(os.path.dirname(__file__), LOG_PATH)
    print(f"\n📝  Log file: {log_abs}")
    print("    (Ctrl+C mid-generation to skip to next question;")
    print("     type 'quit' or 'exit' to stop the program)\n")

    try:
        with open(log_abs, "a", encoding="utf-8") as log_file:
            while True:
                # ── 接收用户输入 ─────────────────────────
                try:
                    question = input("User> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋  Exiting...")
                    break

                if not question:
                    continue
                if question.lower() in {"quit", "exit", "q"}:
                    print("👋  Goodbye.")
                    break

                # ── 执行生成 ────────────────────────────
                generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    alpha=alpha,
                    log_file=log_file,
                    eos_id=eos_id,
                )
    finally:
        # 无论如何都移除 hook，恢复模型原始行为
        hook_handle.remove()
        print("\n🔌  Hook removed. Model restored to baseline.")


if __name__ == "__main__":
    main()
