"""
Closed-Loop Steering System — Centralized Configuration
=========================================================
All hyperparameters for the 5-module pipeline are defined here.
"""

# ======================== Model Selection ========================
# Supported options: "qwen3-8b", "deepseek-1.5b", "gemma-4-e2b"
ACTIVE_MODEL = "gemma-4-e2b"  # Change this to switch models

# Model-specific parameters
MODEL_CONFIGS = {
    "qwen3-8b": {
        "path": "/root/autodl-tmp/qwen3-8b",
        "layer_id": 24,           # Layer index for 36-layer model (2/3 position)
        "vector_dir": "./vectors-copy/qwen3-8b",
        "endoftext_id": 151643,
        "enable_thinking": True,
        "attn_implementation": "flash_attention_2",
    },
    "deepseek-1.5b": {
        "paths": [
            "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B",
            "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        ],
        "layer_id": 19,           # Layer index for 28-layer model (2/3 position: 19; 3/4 position: 21)
        "vector_dir": "./vectors-copy/DeepSeek-R1-Distill-Qwen-1.5B",
        "endoftext_id": 151643,   # DeepSeek R1 Distill Qwen uses the same vocab/tokens
        "enable_thinking": True,
        "attn_implementation": "flash_attention_2",
    },
    "gemma-4-e2b": {
        "paths": [
            "/root/autodl-tmp/gemma-4-E2B-it",
            "/root/autodl-tmp/gemma-4-E2B",
            "/root/autodl-tmp/google/gemma-4-E2B-it",
            "/root/autodl-tmp/google/gemma-4-E2B"
        ],
        "layer_id": 23,           # Layer index for 35-layer model (2/3 position: 35 * 2/3 ≈ 23)
        "vector_dir": "./vectors-copy/gemma-4-E2B-it",
        "endoftext_id": 0,        # Pad token ID for Gemma models
        "enable_thinking": True,
        "attn_implementation": "sdpa", # Avoid FlashAttention head_dim limit (> 256)
    }
}

# Resolve active config
if ACTIVE_MODEL not in MODEL_CONFIGS:
    raise ValueError(f"Unknown ACTIVE_MODEL: {ACTIVE_MODEL}. Options: {list(MODEL_CONFIGS.keys())}")

_cfg = MODEL_CONFIGS[ACTIVE_MODEL]

# Resolve model path
if "paths" in _cfg:
    import os
    MODEL_PATH = _cfg["paths"][0]
    for p in _cfg["paths"]:
        if os.path.exists(p):
            MODEL_PATH = p
            break
else:
    MODEL_PATH = _cfg["path"]

LAYER_ID = _cfg["layer_id"]
VECTOR_DIR = _cfg["vector_dir"]
ENDOFTEXT_ID = _cfg["endoftext_id"]
ENABLE_THINKING = _cfg["enable_thinking"]
ATTN_IMPLEMENTATION = _cfg.get("attn_implementation", "flash_attention_2")

DEFAULT_DTYPE = "bfloat16"    # Options: "float16", "bfloat16", "float32"
DEVICE_MAP = "auto"           # Options: "auto", "cuda:0", "cpu"


# CUDA Memory Management
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.7"

# ======================== EMA & State Monitor ========================
EMA_BETA = 0.1                 # EMA smoothing factor (0.1 = 10% current, 90% history)
ENTROPY_THRESHOLD = 0.15       # SetPoint: EMA entropy above this triggers intervention
TECA_TEMPERATURE = 1.0        # Softmax temperature for entropy calculation
TECA_EPSILON = 1e-9           # Numerical stability for log
MATH_EPSILON = 1e-6           # Generic epsilon for geometry/clamping

# ======================== ThinkBrake (State Monitor) ========================
CONVERGENCE_MARGIN_TAU = 0.25  # Margin threshold for convergence detection
# The token id of </think> — will be resolved at runtime from the tokenizer

# ======================== Anti-Collapse Mechanism ========================
COLLAPSE_ENTROPY_MIN = 0.02           # Threshold below which entropy is considered "collapsed"
COLLAPSE_COUNT_THRESHOLD = 10          # Number of continuous steps below min entropy to trigger watchdog
PERTURBATION_GAMMA = 0.1              # Strength of the orthogonal noise perturbation (sideways kick)
PERTURBATION_COOLDOWN_STEPS = 3       # Number of steps to freeze PD intervention after a kick
N_GRAM_K = 3                          # Sequence match length for repetition detection


# ======================== PID Controller ========================
PID_KP = 2.0                  # Proportional gain 0.6
PID_KI = 0                  # Integral gain
PID_KD = 0.5                 # Derivative gain
ALPHA_MAX = 0.5              # Maximum rotation angle (radians), ~17 degrees

# ======================== EAST: Entropy-Scaled Steering (方案一) ========================
# Reference: Entropic Activation Steering (arXiv:2406.00244)
# α'_t = α_t * sigmoid_decay(EMA_t; θ_high) * (1 - normalized_entropy(H_t))
# When EMA_t >> θ_high (model confused), α'_t → 0 (avoid "surface hijacking").
# When EMA_t < θ_high (model on track), α'_t ≈ α_t (full steering power).
EAST_ENABLED = True           # Master switch for entropy-scaled steering
EAST_LAMBDA_SCALE = 10.0      # Sigmoid steepness around θ_high (higher = sharper cutoff)
EAST_HIGH_ENTROPY_THETA = 0.5  # Entropy threshold above which steering decays (mid-high confusion zone)
EAST_H_MIN = 0.0              # Min entropy for normalization (typically 0)
EAST_H_MAX = 1.0              # Max entropy for normalization (log-normalized to 0-1 range)

# ======================== Manifold Projection ========================
PCA_N_COMPONENTS = 10         # Number of principal components to retain

# ======================== DTR Evaluator ========================
DTR_G = 0.5                   # JSD convergence threshold
DTR_RHO = 0.85                # Deep-thinking layer fraction
DTR_CHUNK_SIZE = 256          # Processing chunk size to save VRAM
DEFAULT_DTR_LAYER = 16        # Default layer for DTR replay analysis
CONTEXT_WINDOW_LIMIT = 4096   # Safety limit for internal model forward passes (PPL etc)

# ======================== Spherical Steering ========================
CONTINUOUS_ALPHA = 0.45       # Default SLERP rotation angle for Continuous mode
# Continuous_Linear α is calibrated via Equal Orthogonal Projection from SLERP α:
#   α_linear = sin(α_slerp * π/2) * ||h||  (applied inside the hook)
#   α_slerp=0.30 → α_linear≈0.45 | α_slerp=0.45 → α_linear≈0.65
CONTINUOUS_LINEAR_ALPHA = 0.3  # Calibrated linear coefficient (matches SLERP=0.30 sweet spot)
CAPTURE_HIDDEN_STATES = False  # Whether to log all hidden states during hook (memory intensive)

# VECTOR_DIR resolved above dynamically from ACTIVE_MODEL
# VECTOR_DIR = "./vectors-copy/qwen3-8b"

# ======================== Generation ========================
DO_SAMPLE = True
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20                    # Hard cap on sampling pool (0 = disabled). Prevents long-tail noise tokens.
MIN_P = 0.05                  # Dynamic floor: discard tokens with P < min_p * P_max. Adapts to model confidence.
MAX_NEW_TOKENS = 4096*8
# ENDOFTEXT_ID resolved above dynamically from ACTIVE_MODEL
# ENDOFTEXT_ID = 151643         # Fallback pad/eos token for Qwen-style models
SAFE_SCORE_RANGE = 1e4        # Clamp range for Inf/NaN logits protection

# ======================== Ablation Dataset ========================
# Path to the new ablation dataset. Users should update this to point to their ablation dataset.
ABLATION_DATASET_PATH = "./dataset/ablation_dataset.jsonl"

# ======================== Experiment ========================
# Available modes: 
#   "Baseline", "Continuous", "Continuous_Linear", "Dynamic_Spherical"
# Ablation modes:
#   "Dynamic_Spherical_No_Manifold" (w/o PCA)
#   "Dynamic_Linear"                (w/o SLERP)
#   "Dynamic_Spherical_No_ThinkBrake" (no latch)
#   "Dynamic_Spherical_No_EMA"      (instantaneous entropy)
#   "Dynamic_Spherical_No_AntiCollapse" (w/o anti-collapse manifold perturbation)
# TAE competitor baselines (EMNLP 2025):
#   "True_TAE"       — open-loop H_t * k → linear inject with raw vector
#   "TAE_Spherical"  — open-loop H_t * k → SLERP with PCA vector (control variable)
EXPERIMENT_MODES = ["Continuous_Linear", "Dynamic_Spherical"]
GLOBAL_SEED = 42

RESULTS_DIR = "./results"
RESULTS_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
JSON_INDENT = 2

# ======================== Evaluation ========================
REPETITION_NGRAM = 4          # N-gram size for repetition rate calculation

# ======================== AIME Benchmark ========================
DATASET_DIR = "./dataset"     # Path to AIME JSONL dataset files
AIME_MAX_TOKENS = 4096*8        # Extended token budget for math reasoning
BATCH_SIZE = 4                # Legacy static batch size (kept for reference)
MAX_CONCURRENT_SEQS = 4      # Continuous batching: max slots active simultaneously
                               # Physical GPU throughput is ~= this number of parallel seqs