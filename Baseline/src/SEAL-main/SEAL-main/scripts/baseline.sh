gpu=${1:-0}
echo CUDA_VISIBLE_DEVICES=$gpu python eval_MATH_vllm.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir results/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
    --max_tokens 10000 \
    --use_chat_format \
    --dataset "MATH500" \
    --remove_bos