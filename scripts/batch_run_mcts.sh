#!/bin/bash

# ================= 用户配置区域 =================
# 【关键】在这里填写你想运行的显卡编号，用空格隔开
# 例如：(0 5) 表示只运行 GPU 0 和 GPU 5
# 例如：(0 1 2 3 4 5 6 7) 表示运行所有
ENABLED_GPUS=(6)
# ==============================================

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 创建日志存放目录
LOG_DIR="./console_logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "准备批量任务..."
echo "计划运行的 GPU: ${ENABLED_GPUS[*]}"
echo "日志目录: $LOG_DIR"
echo "时间戳: $TIMESTAMP"
echo "=================================================="

# 定义一个函数用来检查某个GPU是否在启用列表中
is_enabled() {
    local target=$1
    for gpu in "${ENABLED_GPUS[@]}"; do
        if [[ "$gpu" == "$target" ]]; then
            return 0 # 0 表示真 (True)
        fi
    done
    return 1 # 1 表示假 (False)
}

# amc23, GPQA-Diamond, math500, Olympiadphysics
# Meta-Llama-3.1-8B-Instruct   Qwen2.5-7B-Instruct    Qwen3-8B

# ==================== GPU 0: 基准配置 ====================
if is_enabled 0; then
    echo "正在启动 GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_mcts.sh \
      --dataset "math500" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 500 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_0.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 0 (未在列表中)"
fi

# ==================== GPU 1: 增加迭代次数 ====================
if is_enabled 1; then
    echo "正在启动 GPU 1 ..."
    CUDA_VISIBLE_DEVICES=1 nohup ./scripts/run_mcts.sh \
      --dataset "math500" \
      --model "../models/Qwen2.5-7B-Instruct" \
      --case_start 1 \
      --case_end 500 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_1.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 1 (未在列表中)"
fi

# ==================== GPU 2: 增加探索性 ====================
if is_enabled 2; then
    echo "正在启动 GPU 2 ..."
    CUDA_VISIBLE_DEVICES=2 nohup ./scripts/run_mcts.sh \
      --dataset "math500" \
      --model "../models/Qwen3-8B" \
      --case_start 1 \
      --case_end 500 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_2.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 2 (未在列表中)"
fi

# ==================== GPU 3: 增加搜索广度 ====================
if is_enabled 3; then
    echo "正在启动 GPU 3 ..."
    CUDA_VISIBLE_DEVICES=3 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_3.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 3 (未在列表中)"
fi

# ==================== GPU 4: 依赖 Rollout 信号 ====================
if is_enabled 4; then
    echo "正在启动 GPU 4 ..."
    CUDA_VISIBLE_DEVICES=4 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Qwen2.5-7B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_4.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 4 (未在列表中)"
fi

# ==================== GPU 5: 依赖 Process 信号 ====================
if is_enabled 5; then
    echo "正在启动 GPU 5 ..."
    CUDA_VISIBLE_DEVICES=5 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Qwen3-8B" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 6 \
      --max_depth 15 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0.2 \
      --process_weight 0.4 \
      --rollout_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_5.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 5 (未在列表中)"
fi

# ==================== GPU 6: 消融实验 - 无Process ====================
if is_enabled 6; then
    echo "正在启动 GPU 6 ..."
    CUDA_VISIBLE_DEVICES=6 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 100 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 5 \
      --max_depth 10 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0.3 \
      --process_weight 0.4 \
      --rollout_weight 0.3 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_6.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 6 (未在列表中)"
fi

# ==================== GPU 7: 消融实验 - 有Pairwise ====================
if is_enabled 7; then
    echo "正在启动 GPU 7 ..."
    CUDA_VISIBLE_DEVICES=7 nohup ./scripts/run_mcts.sh \
      --dataset "amc23" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 40 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 3 \
      --rollout_num 6 \
      --max_depth 15 \
      --balance_beta 0.65 \
      --run_mode "async_vllm" \
      --pairwise_weight 0.2 \
      --process_weight 0.4 \
      --rollout_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_7.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 7 (未在列表中)"
fi

echo "=================================================="
echo "脚本执行完毕。"