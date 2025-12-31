#!/bin/bash

# 这个脚本用于运行MCTS_reasoning.py，实现基于蒙特卡洛树搜索的推理
# 指定显卡
#export CUDA_VISIBLE_DEVICES=7

# ================= 配置区域 =================

# 基础设置
DATASET="blocksworld_local_2"         # 数据集名称
MODEL="../models/Qwen2.5-7B-Instruct" # 模型路径或名称
#MODEL="../models/Qwen3-8B"

# 案例范围
CASE_START=1           # 起始案例索引
CASE_END=1             # 结束案例索引

# MCTS 核心参数
ITERATIONS=20          # MCTS迭代次数
EXPLORATION_CONSTANT=12 # 探索常量 (UCT)
BRANCH_FACTOR_INIT=3    # 初始步骤分支因子
BRANCH_FACTOR=3         # 后续步骤分支因子
MAX_DEPTH=10            # MCTS树的最大深度
RUN_MODE="async_vllm"   # 运行模式

# Rollout 参数
ROLLOUT_NUM=3           # 展开次数

# --- 奖励权重参数 (根据 Python 代码更新) ---

# 1. 节点与路径价值权重 (和为 1.0 左右)
LEAF_VALUE_WEIGHT=0.25      # 叶子节点自身价值权重
PATH_VALUE_WEIGHT=0.5       # 推理路径平均价值权重
LEAF_CONFIDENCE_WEIGHT=0.25 # 基于答案频率的置信度权重

# 2. Rollout 内部权重
ROLLOUT_SC_WEIGHT=0.0          # Rollout 中的自洽性(Self-consistency)权重
ROLLOUT_CONFIDENCE_WEIGHT=0.0  # Rollout 中的置信度权重

# 3. 总奖励合成权重
PAIRWISE_WEIGHT=0.2     # 对比奖励信号的权重
PROCESS_WEIGHT=0.4      # 过程评估信号的权重
ROLLOUT_WEIGHT=0.4      # rollout模拟评估信号的权重

# ===========================================

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --case_start)
      CASE_START="$2"
      shift 2
      ;;
    --case_end)
      CASE_END="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --exploration_constant)
      EXPLORATION_CONSTANT="$2"
      shift 2
      ;;
    --branch_factor)
      BRANCH_FACTOR="$2"
      shift 2
      ;;
    --branch_factor_init)
      BRANCH_FACTOR_INIT="$2"
      shift 2
      ;;
    --rollout_num)
      ROLLOUT_NUM="$2"
      shift 2
      ;;
    --max_depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --run_mode)
      RUN_MODE="$2"
      shift 2
      ;;
    # --- 新增权重参数解析 ---
    --leaf_value_weight)
      LEAF_VALUE_WEIGHT="$2"
      shift 2
      ;;
    --path_value_weight)
      PATH_VALUE_WEIGHT="$2"
      shift 2
      ;;
    --leaf_confidence_weight)
      LEAF_CONFIDENCE_WEIGHT="$2"
      shift 2
      ;;
    --rollout_sc_weight)
      ROLLOUT_SC_WEIGHT="$2"
      shift 2
      ;;
    --rollout_confidence_weight)
      ROLLOUT_CONFIDENCE_WEIGHT="$2"
      shift 2
      ;;
    # --- 原有权重参数 ---
    --pairwise_weight)
      PAIRWISE_WEIGHT="$2"
      shift 2
      ;;
    --process_weight)
      PROCESS_WEIGHT="$2"
      shift 2
      ;;
    --rollout_weight)
      ROLLOUT_WEIGHT="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "部分可用选项:"
      echo "  --dataset DATASET             数据集名称"
      echo "  --leaf_value_weight NUM       叶子节点价值权重 (默认: 0.25)"
      echo "  --path_value_weight NUM       路径价值权重 (默认: 0.5)"
      echo "  --leaf_confidence_weight NUM  叶子置信度权重 (默认: 0.25)"
      echo "  --rollout_sc_weight NUM       Rollout自洽性权重 (默认: 0.0)"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

# 打印运行信息
echo "================ 运行配置 ================"
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "案例范围: $CASE_START-$CASE_END"
echo "模式: $RUN_MODE"
echo "---------------- MCTS 参数 ----------------"
echo "迭代次数: $ITERATIONS"
echo "最大深度: $MAX_DEPTH"
echo "探索常量: $EXPLORATION_CONSTANT"
echo "分支因子 (Init/Exp): $BRANCH_FACTOR_INIT / $BRANCH_FACTOR"
echo "Rollout 次数: $ROLLOUT_NUM"
echo "---------------- 权重参数 ----------------"
echo "[节点选择] Leaf Value: $LEAF_VALUE_WEIGHT"
echo "[节点选择] Path Value: $PATH_VALUE_WEIGHT"
echo "[节点选择] Leaf Conf : $LEAF_CONFIDENCE_WEIGHT"
echo "[一致性奖励 ] SC Weight : $ROLLOUT_SC_WEIGHT"
echo "[置信度奖励 ] Conf Weight: $ROLLOUT_CONFIDENCE_WEIGHT"
echo "[对比奖励 ] Pairwise   : $PAIRWISE_WEIGHT"
echo "[过程奖励 ] Process    : $PROCESS_WEIGHT"
echo "[Rollout奖励 ] Rollout    : $ROLLOUT_WEIGHT"
echo "=========================================="
echo ""

# 设置数据集目录的绝对路径
DATASET_DIR="/data/wuyang/MCTS_Reasoning/datasets"

export VLLM_LOGGING_LEVEL=error

# 运行Python脚本
cd /data/wuyang/MCTS_Reasoning && python MCTS_reasoning.py \
  --dataset_name "$DATASET" \
  --model "$MODEL" \
  --case_start "$CASE_START" \
  --case_end "$CASE_END" \
  --num_iterations "$ITERATIONS" \
  --exploration_constant "$EXPLORATION_CONSTANT" \
  --branch_factor "$BRANCH_FACTOR" \
  --branch_factor_init "$BRANCH_FACTOR_INIT" \
  --rollout_num "$ROLLOUT_NUM" \
  --max_depth "$MAX_DEPTH" \
  --run_mode "$RUN_MODE" \
  --leaf_value_weight "$LEAF_VALUE_WEIGHT" \
  --path_value_weight "$PATH_VALUE_WEIGHT" \
  --leaf_confidence_weight "$LEAF_CONFIDENCE_WEIGHT" \
  --rollout_sc_weight "$ROLLOUT_SC_WEIGHT" \
  --rollout_confidence_weight "$ROLLOUT_CONFIDENCE_WEIGHT" \
  --pairwise_weight "$PAIRWISE_WEIGHT" \
  --process_weight "$PROCESS_WEIGHT" \
  --rollout_weight "$ROLLOUT_WEIGHT" \
  --show_runtime \
  --show_runtime_case \
  --verbose