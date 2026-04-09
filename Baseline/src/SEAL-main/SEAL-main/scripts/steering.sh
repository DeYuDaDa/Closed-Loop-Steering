gpu=${1:-0}
CUDA_VISIBLE_DEVICES=$gpu python eval_MATH_steering.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir results/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/steer-from-MATH-10000-use-original \
    --max_tokens 10000 \
    --use_chat_format \
    --batch_size  25 \
    --dataset MATH500 \
    --remove_bos \
    --steering \
    --steering_vector results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
    --steering_layer 20 \
    --steering_coef -1.0 \
    --start 0 \
    --max_examples 100


CUDA_VISIBLE_DEVICES=$gpu python eval_code_steering.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir results/LiveCode/release_v1/DeepSeek-R1-Distill-Qwen-1.5B/steer-from-MATH-10000-use-original \
    --max_tokens 10000 \
    --use_chat_format \
    --batch_size  25 \
    --release release_v1 \
    --remove_bos \
    --steering \
    --steering_vector results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
    --steering_layer 20 \
    --steering_coef -1.0 \
    --start 0 \
    --max_examples 100