#!/bin/bash

# ================= 用户配置区域 =================
# 【关键】在这里填写你想运行的显卡编号，用空格隔开
ENABLED_GPUS=(0 1 3 4 6 7)

# 设置运行的总轮数 (这里设为 4)
TOTAL_ROUNDS=4
# ==============================================

# 获取脚本启动时的统一时间戳 (用于将两轮日志归档在同一批次下)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 创建日志存放目录
LOG_DIR="./console_logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "准备批量任务..."
echo "计划运行的 GPU: ${ENABLED_GPUS[*]}"
echo "计划总轮数: $TOTAL_ROUNDS"
echo "日志目录: $LOG_DIR"
echo "任务批次ID: $TIMESTAMP"
echo "=================================================="

# 定义检查GPU是否启用的函数
is_enabled() {
    local target=$1
    for gpu in "${ENABLED_GPUS[@]}"; do
        if [[ "$gpu" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

# ==================== 开始轮次循环 ====================
for (( ROUND=1; ROUND<=TOTAL_ROUNDS; ROUND++ ))
do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "正在启动第 [ $ROUND / $TOTAL_ROUNDS ] 轮任务..."
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # ==================== GPU 0 ====================
    if is_enabled 0; then
        echo "[Round $ROUND] 正在启动 GPU 0 ..."
        CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
          --model "../models/Meta-Llama-3.1-8B-Instruct" \
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
          --pairwise_weight 0.01 \
          --process_weight 0.4 \
          --rollout_weight 0.5 \
          --leaf_value_weight 0.3 \
          --path_value_weight 0.2 \
          --leaf_confidence_weight 0.5 \
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_0.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 1 ====================
    if is_enabled 1; then
        echo "[Round $ROUND] 正在启动 GPU 1 ..."
        CUDA_VISIBLE_DEVICES=1 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
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
          --pairwise_weight 0.01 \
          --process_weight 0.4 \
          --rollout_weight 0.5 \
          --leaf_value_weight 0.3 \
          --path_value_weight 0.2 \
          --leaf_confidence_weight 0.5 \
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_1.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 2 ====================
    if is_enabled 2; then
        echo "[Round $ROUND] 正在启动 GPU 2 ..."
        CUDA_VISIBLE_DEVICES=2 nohup ./scripts/run_mcts.sh \
          --dataset "GPQA-Diamond" \
          --model "../models/Meta-Llama-3.1-8B-Instruct" \
          --case_start 1 \
          --case_end 200 \
          --iterations 15 \
          --exploration_constant 12 \
          --branch_factor 3 \
          --branch_factor_init 4 \
          --rollout_num 3 \
          --max_depth 10 \
          --run_mode "async_vllm" \
          --rollout_sc_weight 0.05 \
          --rollout_confidence_weight 0.05 \
          --pairwise_weight 0.1 \
          --process_weight 0.4 \
          --rollout_weight 0.5 \
          --leaf_value_weight 0.3 \
          --path_value_weight 0.3 \
          --leaf_confidence_weight 0.4 \
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_2.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 3 ====================
    if is_enabled 3; then
        echo "[Round $ROUND] 正在启动 GPU 3 ..."
        CUDA_VISIBLE_DEVICES=3 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
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
          --pairwise_weight 0.01 \
          --process_weight 0.4 \
          --rollout_weight 0.5 \
          --leaf_value_weight 0.3 \
          --path_value_weight 0.2 \
          --leaf_confidence_weight 0.5 \
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_3.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 4 ====================
    if is_enabled 4; then
        echo "[Round $ROUND] 正在启动 GPU 4 ..."
        CUDA_VISIBLE_DEVICES=4 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
          --model "../models/Meta-Llama-3.1-8B-Instruct" \
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
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_4.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 5 ====================
    if is_enabled 5; then
        echo "[Round $ROUND] 正在启动 GPU 5 ..."
        CUDA_VISIBLE_DEVICES=5 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
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
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_5.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 6 ====================
    if is_enabled 6; then
        echo "[Round $ROUND] 正在启动 GPU 6 ..."
        CUDA_VISIBLE_DEVICES=6 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
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
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_6.log" 2>&1 &
          sleep 2
    fi

    # ==================== GPU 7 ====================
    if is_enabled 7; then
        echo "[Round $ROUND] 正在启动 GPU 7 ..."
        CUDA_VISIBLE_DEVICES=7 nohup ./scripts/run_mcts.sh \
          --dataset "aime2024" \
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
          > "${LOG_DIR}/${TIMESTAMP}_round${ROUND}_gpu_7.log" 2>&1 &
          sleep 2
    fi

    echo "--------------------------------------------------"
    echo "[Round $ROUND] 所有任务已后台启动。"
    echo "[Round $ROUND] 正在等待所有任务完成..."
    echo "请不要关闭此终端，否则后续轮次将不会执行。"
    
    # 核心指令：阻塞直到当前轮次的所有后台任务结束
    wait 

    echo "[Round $ROUND] 本轮任务全部完成！"
    echo "--------------------------------------------------"

done
# ==================== 循环结束 ====================

echo "=================================================="
echo "所有 $TOTAL_ROUNDS 轮任务已全部执行完毕。"
echo "=================================================="