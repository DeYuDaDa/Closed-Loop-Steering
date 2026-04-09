"""
Closed-Loop Steering System — Centralized Configuration
=========================================================
All hyperparameters for the 5-module pipeline are defined here.
"""

# ======================== Model & System ========================
MODEL_PATH = "/root/autodl-tmp/qwen3-8b"
LAYER_ID = 24                 # Target transformer layer for hook injection
DEFAULT_DTYPE = "bfloat16"    # Options: "float16", "bfloat16", "float32"
DEVICE_MAP = "auto"           # Options: "auto", "cuda:0", "cpu"
ENABLE_THINKING = True

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
ALPHA_MAX = 0.45              # Maximum rotation angle (radians), ~17 degrees

# ======================== EAST: Entropy-Scaled Steering (方案一) ========================
# Reference: Entropic Activation Steering (arXiv:2406.00244)
# α'_t = α_t * sigmoid_decay(EMA_t; θ_high) * (1 - normalized_entropy(H_t))
# When EMA_t >> θ_high (model confused), α'_t → 0 (avoid "surface hijacking").
# When EMA_t < θ_high (model on track), α'_t ≈ α_t (full steering power).
EAST_ENABLED = True           # Master switch for entropy-scaled steering
EAST_LAMBDA_SCALE = 10.0      # Sigmoid steepness around θ_high (higher = sharper cutoff)
EAST_HIGH_ENTROPY_THETA = 0.45  # Entropy threshold above which steering decays (mid-high confusion zone)
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

# ======================== Vector Paths ========================
VECTOR_DIR = "./vectors/qwen3-8b"

# ======================== Generation ========================
DO_SAMPLE = True
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20                    # Hard cap on sampling pool (0 = disabled). Prevents long-tail noise tokens.
MIN_P = 0.05                  # Dynamic floor: discard tokens with P < min_p * P_max. Adapts to model confidence.
MAX_NEW_TOKENS = 4096*8
ENDOFTEXT_ID = 151643         # Fallback pad/eos token for Qwen-style models
SAFE_SCORE_RANGE = 1e4        # Clamp range for Inf/NaN logits protection

# ======================== Experiment ========================
# Available modes: 
#   "Baseline", "Continuous", "Continuous_Linear", "Dynamic_Spherical"
# Ablation modes:
#   "Dynamic_Spherical_No_Manifold" (w/o PCA)
#   "Dynamic_Linear"                (w/o SLERP)
#   "Dynamic_Spherical_No_ThinkBrake" (no latch)
#   "Dynamic_Spherical_No_EMA"      (instantaneous entropy)
# TAE competitor baselines (EMNLP 2025):
#   "True_TAE"       — open-loop H_t * k → linear inject with raw vector
#   "TAE_Spherical"  — open-loop H_t * k → SLERP with PCA vector (control variable)
EXPERIMENT_MODES = ["Continuous_Linear", "Dynamic_Spherical"]

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