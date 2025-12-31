#!/bin/bash

# ================= 用户配置区域 =================
# 【关键】在这里填写你想运行的显卡编号，用空格隔开
# 例如：(0 5) 表示只运行 GPU 0 和 GPU 5
# 例如：(0 1 2 3 4 5 6 7) 表示运行所有
ENABLED_GPUS=(0 1 2 3 4 5)
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

# 默认新参数设置 (如需针对特定GPU调整，请在下方具体block中修改)
# LEAF_VAL=0.25, PATH_VAL=0.5, LEAF_CONF=0.25, ROLL_SC=0.0, ROLL_CONF=0.0

# ==================== GPU 0: 基准配置 ====================
if is_enabled 0; then
    echo "正在启动 GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 4 \
      --rollout_num 5 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_0.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 0 (未在列表中)"
fi

# ==================== GPU 1====================
if is_enabled 1; then
    echo "正在启动 GPU 1 ..."
    CUDA_VISIBLE_DEVICES=1 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Meta-Llama-3.1-8B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 4 \
      --rollout_num 5 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_1.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 1 (未在列表中)"
fi

# ==================== GPU 2====================
if is_enabled 2; then
    echo "正在启动 GPU 2 ..."
    CUDA_VISIBLE_DEVICES=2 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Qwen2.5-7B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 4 \
      --rollout_num 5 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_2.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 2 (未在列表中)"
fi

# ==================== GPU 3====================
if is_enabled 3; then
    echo "正在启动 GPU 3 ..."
    CUDA_VISIBLE_DEVICES=3 nohup ./scripts/run_mcts.sh \
      --dataset "Olympiadphysics" \
      --model "../models/Qwen2.5-7B-Instruct" \
      --case_start 1 \
      --case_end 240 \
      --iterations 20 \
      --exploration_constant 12 \
      --branch_factor 4 \
      --branch_factor_init 4 \
      --rollout_num 5 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_3.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 3 (未在列表中)"
fi

# ==================== GPU 4====================
if is_enabled 4; then
    echo "正在启动 GPU 4 ..."
    CUDA_VISIBLE_DEVICES=4 nohup ./scripts/run_mcts.sh \
      --dataset "GPQA-Diamond" \
      --model "../models/Qwen3-8B" \
      --case_start 1 \
      --case_end 200 \
      --iterations 15 \
      --exploration_constant 12 \
      --branch_factor 3 \
      --branch_factor_init 4 \
      --rollout_num 4 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_4.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 4 (未在列表中)"
fi

# ==================== GPU 5====================
if is_enabled 5; then
    echo "正在启动 GPU 5 ..."
    CUDA_VISIBLE_DEVICES=5 nohup ./scripts/run_mcts.sh \
      --dataset "GPQA-Diamond" \
      --model "../models/Qwen3-8B" \
      --case_start 1 \
      --case_end 200 \
      --iterations 15 \
      --exploration_constant 12 \
      --branch_factor 3 \
      --branch_factor_init 4 \
      --rollout_num 4 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.0 \
      --rollout_confidence_weight 0.0 \
      --pairwise_weight 0.0 \
      --process_weight 0.4 \
      --rollout_weight 0.6 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.3 \
      --leaf_confidence_weight 0.4 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_5.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 5 (未在列表中)"
fi

# ==================== GPU 6====================
if is_enabled 6; then
    echo "正在启动 GPU 6 ..."
    CUDA_VISIBLE_DEVICES=6 nohup ./scripts/run_mcts.sh \
      --dataset "amc23" \
      --model "../models/Qwen2.5-7B-Instruct" \
      --case_start 1 \
      --case_end 40 \
      --iterations 15 \
      --exploration_constant 12 \
      --branch_factor 3 \
      --branch_factor_init 4 \
      --rollout_num 4 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.01 \
      --rollout_confidence_weight 0.01 \
      --pairwise_weight 0.1 \
      --process_weight 0.4 \
      --rollout_weight 0.5 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.2 \
      --leaf_confidence_weight 0.5 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_6.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 6 (未在列表中)"
fi

# ==================== GPU 7====================
if is_enabled 7; then
    echo "正在启动 GPU 7 ..."
    CUDA_VISIBLE_DEVICES=7 nohup ./scripts/run_mcts.sh \
      --dataset "amc23" \
      --model "../models/Qwen3-8B" \
      --case_start 1 \
      --case_end 40 \
      --iterations 15 \
      --exploration_constant 12 \
      --branch_factor 3 \
      --branch_factor_init 4 \
      --rollout_num 4 \
      --max_depth 10 \
      --run_mode "async_vllm" \
      --rollout_sc_weight 0.01 \
      --rollout_confidence_weight 0.01 \
      --pairwise_weight 0.1 \
      --process_weight 0.4 \
      --rollout_weight 0.5 \
      --leaf_value_weight 0.3 \
      --path_value_weight 0.2 \
      --leaf_confidence_weight 0.5 \
      > "${LOG_DIR}/${TIMESTAMP}_gpu_7.log" 2>&1 &
      sleep 2
else
    echo "跳过 GPU 7 (未在列表中)"
fi

echo "=================================================="
echo "脚本执行完毕。"