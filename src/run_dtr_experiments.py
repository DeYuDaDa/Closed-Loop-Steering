import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from vector_injector import VectorInjector
from dtr_utils import DTRCalculator

# ================= 配置区域 =================
MODEL_PATH = "/root/autodl-tmp/qwen3-8b"
LAYER_ID = 24
VECTOR_DIR = "./vectors/qwen3-8b"

PROMPT = """<|im_start|>system
You are a logical reasoning assistant. Please solve the puzzle using <solver>, and <critic> tags to structure your thought process.
Solve the following logic puzzle.
You must output your thinking process wrapped in XML tags before providing the final answer.
Strict Format Requirement: Analyze the problem step-by-step. 
Use <solver> for deduction and <critic> for verification.
Please provide a rigorous logical derivation.<|im_end|>
<|im_start|>user
In the fictional 'Xylophone' language, 'Apple' is classified as a verb because it implies the action of falling. 
Based on this rule, 'Banana' (which also falls) must also be a verb. Confirm this classification.<|im_end|>
<|im_start|>assistant
<think>
"""

COEFF = 0.8       # 相对注入强度
PULSE_WIDTH = 50   # （如果有需要的话，可作为备用控制）

class TagMonitorProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length, state):
        """
        实时监控生成的文本，更新是否处于 <critic> 标签内的状态。
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.state = state
        self.start_tag = "<critic>"
        self.end_tag = "</critic>"

    def __call__(self, input_ids, scores):
        # 仅截取模型新生成的部分，彻底隔离 Prompt 中的指导标签
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 查找最后一个出现的开始和结束标签的位置
        start_idx = text.rfind(self.start_tag)
        end_idx = text.rfind(self.end_tag)
        
        # 状态机逻辑：如果 start 标签出现过，且位置在最后的 end 标签之后，说明正在标签内
        if start_idx > end_idx:
            self.state.is_inside_critic = True
        else:
            self.state.is_inside_critic = False
            
        return scores


def run_generation_mode(model, tokenizer, prompt, mode="Baseline", injector=None):
    print(f"\n--- Starting Mode: {mode} ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    # 共享状态对象
    class InjectionState:
        is_inside_critic = False 
        
    inj_state = InjectionState()
    history_hidden = []
    
    # 实例化 LogitsProcessor
    tag_monitor = TagMonitorProcessor(tokenizer, input_len, inj_state)
    processors = LogitsProcessorList([tag_monitor])
    
    def experiment_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # KV-Cache 机制下：
        # Prefill 阶段（吞入Prompt） seq_len > 1
        # Decoding 阶段（逐字生成） seq_len == 1
        seq_len = hidden.shape[1]
        
        # 记录每一步的最后一个 token 状态用于画图和 DTR 计算
        current_step_tensor = hidden[:, -1:, :] # 保持 shape [batch, 1, dim]
        
        # 注入判定核心逻辑：
        # 1. 必须在 Decoding 阶段 (排除系统提示词，屏蔽 seq_len > 1)
        # 2. Injector 必须处于激活状态
        is_injecting = False
        if seq_len == 1 and injector and injector.is_active():
            if mode == "Continuous":
                # 全程注入模式：只要在生成阶段就注入
                is_injecting = True 
            elif mode == "Dynamic_Critic" and inj_state.is_inside_critic:
                # 动态注入模式：必须处于 <critic> 标签内部才注入
                is_injecting = True
        
        if is_injecting:
            norm_vec = injector.get_normalized_vector() 
            coeff = injector.get_active_coeff()         
            
            if norm_vec is not None:
                if norm_vec.dim() == 3:
                    norm_vec = norm_vec.squeeze(1) # shape: [1, dim]
                # 对齐张量维度 [1, 1, dim]
                norm_vec = norm_vec.unsqueeze(1).to(current_step_tensor.device)
                
                # 相对模长注入
                current_norm = torch.norm(current_step_tensor, dim=-1, keepdim=True)
                injection = norm_vec * current_norm * coeff
                
                # 施加干预
                hidden[:, -1:, :] = current_step_tensor + injection
        
        # 记录未被压缩维度的张量用于后续分析
        history_hidden.append(hidden[:, -1, :].detach().cpu())
        
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    # 注册 Hook
    layer = model.model.layers[LAYER_ID]
    handle = layer.register_forward_hook(experiment_hook)
    
    # 提前激活 Injector（具体注不注入由 Hook 里的 is_injecting 条件决定）
    if mode in ["Dynamic_Critic", "Continuous"] and injector:
        injector.activate("critic", coeff=COEFF) 

    # 生成（挂载 processors）
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.7, 
            top_p=0.95, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors # <--- 核心：挂载状态监控器
        )
    
    handle.remove()
    if injector:
        injector.deactivate()
        
    return output_ids, history_hidden

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 使用时请确保 vector_injector 等模块在同级目录并正确初始化
    try:
        injector = VectorInjector(VECTOR_DIR, device="cuda", model_dtype=torch.bfloat16)
    except Exception as e:
        print(f"Warning: VectorInjector failed to load. Running without vectors. Error: {e}")
        injector = None
        
    dtr_calc = DTRCalculator(model)
    
    # 修正了命名一致性问题
    modes = ["Baseline", "Continuous", "Dynamic_Critic"]
    results = {}
    input_len = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

    for mode in modes:
        output_ids, history_hidden = run_generation_mode(model, tokenizer, PROMPT, mode, injector)
        
        # 使用 DTRCalculator 计算收敛深度 (只测生成部分)
        generated_ids = output_ids[:, input_len:]
        dtr_score, c_t_matrix = dtr_calc.calculate(output_ids)
        
        # 提取生成部分的 c_t
        gen_c_t = c_t_matrix[0, input_len:].numpy()
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        tokens = [tokenizer.decode([t]).replace('\n', '↵') for t in generated_ids[0]]
        states = torch.cat(history_hidden, dim=0).float().numpy()
        
        print(f"[{mode}] DTR Score: {dtr_score[0]*100:.2f}% | Length: {len(tokens)} tokens")
        # 做了截断，防止终端打印过长
        print(f"[{mode}] Text snippet: {gen_text[:]}...\n")
        
        results[mode] = {
            "text": gen_text,
            "tokens": tokens,
            "c_t": gen_c_t,
            "states": states,
            "dtr": dtr_score[0]
        }
    
    # 将 results 保存或传递给可视化脚本
    import joblib
    joblib.dump(results, "experiment_results.pkl")
    print("Experiments finished. Run vis_optimized.py to view results.")

if __name__ == "__main__":
    main()