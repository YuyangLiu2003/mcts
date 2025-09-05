#!/bin/bash

# 默认使用AiHubMix模式
# CUDA_VISIBLE_DEVICES=0 python MCTS_reasoning.py \
#     --dataset_name amc23 \
#     --model qwen-max \
#     --case_start 1 \
#     --case_end 1 \
#     --num_iterations 4 \
#     --branch_factor 3 \
#     --rollout_num 1 \
#     --run_mode aihub

# 使用Transformers模式
# CUDA_VISIBLE_DEVICES=0 python MCTS_reasoning.py \
#     --dataset_name amc23 \
#     --model /root/data/wuyang/models/Meta-Llama-3.1-8B-Instruct \
#     --case_start 1 \
#     --case_end 1 \
#     --num_iterations 4 \
#     --branch_factor 3 \
#     --rollout_num 1 \
#     --run_mode transformer

# 使用vLLM模式
# CUDA_VISIBLE_DEVICES=0 python MCTS_reasoning.py \
#     --dataset_name amc23 \
#     --model /root/data/wuyang/models/Meta-Llama-3.1-8B-Instruct \
#     --case_start 1 \
#     --case_end 1 \
#     --num_iterations 4 \
#     --branch_factor 3 \
#     --rollout_num 1 \
#     --run_mode vllm

# 示例：使用Transformers模式
CUDA_VISIBLE_DEVICES=2 python MCTS_reasoning.py \
    --dataset_name GSM8K \
    --case_start 896 \
    --case_end 1035\
    --num_iterations 10 \
    --branch_factor 1 \
    --branch_factor_init 3 \
    --rollout_num 3 \
    --run_mode vllm \
    --model /root/autodl-tmp/Meta-Llama-3.1-8B-Instruct \
    --max_depth 5 \
    --verbose