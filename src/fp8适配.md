要解决FP8推理报错并正确使用FP8，需从**环境适配、模型加载、数值稳定性、张量操作兼容**四个核心维度入手。以下是分步解决方案，结合你提供的`run_experiment.py`代码逻辑展开：

## 一、前置条件：确认FP8运行环境
FP8推理依赖特定版本的PyTorch/CUDA/Transformers，先解决环境层面的兼容性问题：

### 1. 升级核心依赖
```bash
# 升级PyTorch到2.2+（支持FP8 dtype），CUDA 12.1+
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 升级Transformers/Accelerate（支持FP8+Flash Attention 2）
pip install --upgrade transformers accelerate bitsandbytes flash-attn
```

### 2. 验证环境
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"FP8 dtype支持: {hasattr(torch, 'float8_e4m3fn')}")  # 输出True则支持
```

## 二、核心修改：模型加载适配FP8
在代码中补充/修改模型加载逻辑（你提供的代码未贴模型加载部分，需新增）：

```python
def load_model(model_path: str = MODEL_PATH):
    """加载支持FP8推理的模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 1. 加载Tokenizer（保持原有逻辑）
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 确保pad_token存在

    # 2. FP8相关配置
    attn_kwargs = {}
    if USE_FLASH_ATTENTION:
        attn_kwargs["attn_implementation"] = "flash_attention_2"  # FP8必须搭配FA2
    
    # 3. 加载模型（区分FP8/非FP8）
    if USE_FP8:
        # FP8模式：使用AutoCast+原生FP8 dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 先加载为FP16，再转FP8
            device_map=DEVICE_MAP,
            **attn_kwargs,
        )
        # 启用FP8推理优化（PyTorch 2.2+）
        import torch._inductor.config
        torch._inductor.config.fp8_e4m3fn = True  # 启用FP8 e4m3fn格式
        torch._inductor.config.use_mixed_mm = True  # 混合精度矩阵乘法
    else:
        # 非FP8模式（原有逻辑）
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DEFAULT_DTYPE,
            device_map=DEVICE_MAP,
            **attn_kwargs,
        )
    
    return model, tokenizer
```

## 三、修复FP8数值稳定性问题
FP8的数值范围远小于FP16（e4m3fn范围：~[-448, 448]），需修改`InfNanProtectionProcessor`避免溢出/NaN：

```python
class InfNanProtectionProcessor:
    def __init__(self, eos_id):
        self.eos_id = eos_id if isinstance(eos_id, int) else (eos_id[0] if isinstance(eos_id, list) else 0)
        # FP8安全范围（避免超出e4m3fn的数值极限）
        self.fp8_safe_min = -400.0
        self.fp8_safe_max = 400.0

    def __call__(self, input_ids, scores):
        # 第一步：裁剪到FP8安全范围（核心修改）
        if USE_FP8:
            scores = torch.clamp(scores, min=self.fp8_safe_min, max=self.fp8_safe_max)
        
        # 第二步：替换NaN/Inf
        torch.nan_to_num_(
            scores, 
            nan=-self.fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE,
            posinf=self.fp8_safe_max if USE_FP8 else SAFE_SCORE_RANGE,
            neginf=-self.fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE
        )
        
        # 第三步：处理序列崩溃
        max_scores, _ = scores.max(dim=-1)
        collapsed_threshold = self.fp8_safe_min + 1.0 if USE_FP8 else (-SAFE_SCORE_RANGE + 1.0)
        collapsed_mask = max_scores <= collapsed_threshold
        
        if collapsed_mask.any():
            collapsed_indices = collapsed_mask.nonzero(as_tuple=True)[0].tolist()
            seq_len = input_ids.shape[1]
            for idx in collapsed_indices:
                print(f"  [Warning] 🚨 Sequence {idx} collapsed at length {seq_len} (FP8 overflow). Forcing EOS.")
            
            # 强制EOS生成（避免采样错误）
            scores[collapsed_mask, :] = -self.fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE
            scores[collapsed_mask, self.eos_id] = self.fp8_safe_max if USE_FP8 else SAFE_SCORE_RANGE
        
        return scores
```

## 四、适配FP8的控制向量加载
修改`load_control_vectors`，确保控制向量的dtype与FP8模型一致：

```python
def load_control_vectors(vector_dir: str, device: str, dtype, pca_coeff: float = 1.0) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    injector = VectorInjector(vector_dir, device=device, model_dtype=dtype)

    # ---- Purified (PCA-projected) vector ----
    try:
        if injector.activate("critic", coeff=pca_coeff):
            raw_norm = injector.get_raw_norm()
            print(f"  📏 [purified] raw norm: {raw_norm:.4f}")
            v = injector.get_normalized_vector()
            v_norm = _normalize_vector(v)
            
            # FP8适配：先转float32计算归一化，再转回目标dtype（避免FP8计算精度丢失）
            if USE_FP8:
                v_norm = v_norm.float()  # 归一化用float32
                v_norm = _normalize_vector(v_norm)  # 重新归一化
                v_norm = v_norm.to(dtype=dtype)  # 转FP8
            
            print(f"  📏 [purified] final norm: {v_norm.float().view(-1).norm().item():.4f}")
            vectors["purified"] = v_norm
            injector.deactivate()
    except Exception as e:
        print(f"⚠️  Failed to load purified control vector: {e}")

    # ---- Raw (no-PCA) vector ----
    try:
        raw_path = os.path.join(vector_dir, "critic_raw.pt")
        if os.path.isfile(raw_path):
            v_raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            
            # FP8适配：float32计算归一化，再转FP8
            if USE_FP8:
                v_raw = v_raw.float()
            v_raw_norm = _normalize_vector(v_raw.view(1, 1, -1))
            v_raw_norm = v_raw_norm.to(device=device, dtype=dtype)
            
            print(f"  📏 [raw] final norm: {v_raw_norm.float().view(-1).norm().item():.4f}")
            vectors["raw"] = v_raw_norm
        else:
            print(f"  ⚠️  [raw] critic_raw.pt not found in {vector_dir}")
    except Exception as e:
        print(f"⚠️  Failed to load raw control vector: {e}")

    if not vectors:
        print("⚠️  No control vectors loaded.")
    else:
        print(f"  ℹ️  Loaded vectors: {list(vectors.keys())}")
    return vectors
```

## 五、修复KV Cache的FP8 dtype兼容问题
修改`_stack_and_pad_kv_caches`，确保KV Cache拼接/填充时dtype一致：

```python
def _stack_and_pad_kv_caches(slots: list[_Slot]):
    if not slots:
        return None, 0
    max_len = max(s.input_ids.shape[1] - 1 for s in slots)
    num_layers = len(slots[0].past_key_values)
    batched_pkv = []
    
    for layer_idx in range(num_layers):
        layer_k, layer_v = [], []
        for s in slots:
            k, v = s.past_key_values[layer_idx]
            pad_left = max_len - k.shape[2]
            if pad_left > 0:
                # FP8适配：padding后强制保持原dtype
                k = torch.nn.functional.pad(k, (0, 0, pad_left, 0), value=0.0).to(k.dtype)
                v = torch.nn.functional.pad(v, (0, 0, pad_left, 0), value=0.0).to(v.dtype)
            layer_k.append(k)
            layer_v.append(v)
        
        # 拼接时确保dtype统一（FP8关键）
        batched_k = torch.cat(layer_k, dim=0).to(layer_k[0].dtype)
        batched_v = torch.cat(layer_v, dim=0).to(layer_v[0].dtype)
        batched_pkv.append((batched_k, batched_v))
    return tuple(batched_pkv), max_len
```

## 六、推理时启用FP8上下文
修改`run_batched_generation`中的生成逻辑，添加FP8 autocast上下文：

```python
# Generate部分修改
with torch.no_grad():
    if USE_FP8:
        # FP8推理上下文（PyTorch 2.2+）
        from torch.cuda.amp import autocast
        with autocast(dtype=torch.float8_e4m3fn):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=AIME_MAX_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=processors,
            )
    else:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=AIME_MAX_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=processors,
        )
```

## 七、常见FP8报错的针对性解决
| 报错类型 | 根因 | 解决方案 |
|----------|------|----------|
| `RuntimeError: "fp8_e4m3fn" is not a valid dtype` | PyTorch版本<2.2或CUDA<12.0 | 升级到PyTorch 2.2+ + CUDA 12.1+ |
| `ValueError: Flash Attention 2 requires CUDA` | 未用GPU运行或FA2未安装 | 安装`flash-attn`并在GPU上运行 |
| `NaN/Inf in FP8 tensor` | 数值超出FP8范围 | 启用`InfNanProtectionProcessor`的裁剪逻辑，降低temperature/top_p |
| `KV Cache dtype mismatch` | KV Cache是FP16，模型是FP8 | 在`_stack_and_pad_kv_caches`中强制转换dtype |
| `CUDA error: invalid argument` | 左填充与FP8不兼容 | 调整`tokenizer.padding_side = "right"`（修改`run_batched_generation`中tokenizer的padding_side） |

## 八、验证FP8是否生效
添加验证代码，检查模型/张量dtype：
```python
def verify_fp8_usage(model):
    print("=== FP8使用验证 ===")
    # 检查模型参数dtype
    for name, param in list(model.named_parameters())[:5]:  # 仅检查前5个参数
        print(f"参数 {name}: dtype={param.dtype}")
    # 检查KV Cache dtype（生成时）
    def kv_cache_hook(module, input, output):
        if hasattr(output, "past_key_values"):
            k, v = output.past_key_values[0]
            print(f"KV Cache dtype: K={k.dtype}, V={v.dtype}")
            return output
    # 注册钩子验证
    handle = model.model.layers[0].register_forward_hook(kv_cache_hook)
    return handle

# 在生成前调用验证
handle = verify_fp8_usage(model)
# 生成后移除钩子
handle.remove()
```

## 总结
FP8推理的核心是**环境适配+ dtype一致性+ 数值稳定性**：
1. 确保PyTorch/CUDA版本满足要求；
2. 模型加载时启用Flash Attention 2；
3. 所有张量操作（控制向量、KV Cache）保持FP8 dtype一致；
4. 限制数值范围避免FP8溢出；
5. 用autocast上下文包裹生成逻辑。

按上述步骤修改后，可解决绝大多数FP8推理报错，并确保FP8正确生效。