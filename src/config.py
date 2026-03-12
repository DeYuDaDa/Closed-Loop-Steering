"""
Closed-Loop Steering System — Centralized Configuration
=========================================================
All hyperparameters for the 5-module pipeline are defined here.
"""

# ======================== Model ========================
MODEL_PATH = "/root/autodl-tmp/qwen3-8b"
LAYER_ID = 24  # Target transformer layer for hook injection

# ======================== TECA (State Monitor) ========================
TECA_THRESHOLD = 0.15          # SetPoint: TECA above this triggers intervention
TECA_TEMPERATURE = 1.0        # Softmax temperature for entropy calculation
TECA_EPSILON = 1e-9           # Numerical stability for log

# ======================== ThinkBrake (State Monitor) ========================
CONVERGENCE_MARGIN_TAU = 0.25  # Margin threshold for convergence detection
# The token id of </think> — will be resolved at runtime from the tokenizer

# ======================== PID Controller ========================
PID_KP = 0.6                  # Proportional gain
PID_KI = 0.1                  # Integral gain
PID_KD = 0.05                 # Derivative gain
ALPHA_MAX = 0.30              # Maximum rotation angle (radians), ~17 degrees

# ======================== Manifold Projection ========================
PCA_N_COMPONENTS = 10         # Number of principal components to retain

# ======================== DTR Evaluator ========================
DTR_G = 0.5                   # JSD convergence threshold
DTR_RHO = 0.85                # Deep-thinking layer fraction

# ======================== Spherical Steering ========================
# No additional params beyond alpha (from PID) and the control vector

# ======================== Vector Paths ========================
VECTOR_DIR = "./vectors/qwen3-8b"

# ======================== Generation ========================


# ======================== Experiment ========================
EXPERIMENT_MODES = ["Baseline", "Continuous", "Dynamic_Spherical"]
RESULTS_DIR = "./results"

# ======================== Evaluation ========================
REPETITION_NGRAM = 4          # N-gram size for repetition rate calculation

# ======================== AIME Benchmark ========================
DATASET_DIR = "./dataset"     # Path to AIME JSONL dataset files
AIME_MAX_TOKENS = 4096        # Extended token budget for math reasoning
BATCH_SIZE = 8                # Batch size for Baseline / Continuous modes
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.7
TOP_P = 0.95