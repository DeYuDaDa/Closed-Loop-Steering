gpu=${1:-0}
echo CUDA_VISIBLE_DEVICES=$gpu python eval_MATH_vllm.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
    --max_tokens 10000 \
    --use_chat_format \
    --dataset "MATH_train" \
    --remove_bos

echo CUDA_VISIBLE_DEVICES=${gpu} python hidden_analysis.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path data/MATH/train.jsonl \
    --data_dir results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
    --type incorrect \
    --start 0 \
    --sample 500 \

echo CUDA_VISIBLE_DEVICES=${gpu} python hidden_analysis.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path data/MATH/train.jsonl \
    --data_dir results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
    --type correct \
    --start 0 \
    --sample 500 \

echo python vector_generation.py \
    --data_dir results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
    --prefixs correct_0_500 incorrect_0_500 \
    --layers 20 \
    --save_prefix 500_500 