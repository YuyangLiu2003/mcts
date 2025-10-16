#!/bin/bash

# 这个脚本用于运行MCTS_reasoning.py，实现基于蒙特卡洛树搜索的推理
# 指定显卡
export CUDA_VISIBLE_DEVICES=1
# 设置默认参数
DATASET="math500"          # 数据集名称，可选：amc23, aime2024等
# MODEL="../llama3/models/Qwen2.5-7B-Instruct"  # 模型路径或名称
MODEL="../models/Meta-Llama-3.1-8B-Instruct"  # 模型路径或名称
CASE_START=1           # 起始案例索引
CASE_END=500            # 结束案例索引
ITERATIONS=15           # MCTS迭代次数
BRANCH_FACTOR_INIT=3    # 初始步骤生成的分支因子
BRANCH_FACTOR=3         # MCTS扩展的分支因子
ROLLOUT_NUM=3           # 展开次数
MAX_DEPTH=10            # MCTS树的最大深度
BALANCE_BETA=0.65       # 过程奖励和rollout奖励的加权系数 (0-1)
RUN_MODE="async_vllm"   # 运行模式：aihub, transformer, vllm, async_vllm, debug
PAIRWISE_WEIGHT=0.5      # 对比奖励信号的权重
PROCESS_WEIGHT=0.2       # 过程评估信号的权重
ROLLOUT_WEIGHT=0.3       # rollout模拟评估信号的权重



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
    --balance_beta)
      BALANCE_BETA="$2"
      shift 2
      ;;
    --run_mode)
      RUN_MODE="$2"
      shift 2
      ;;
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
      echo "选项:"
      echo "  --dataset DATASET            数据集名称 (默认: amc23)"
      echo "  --model MODEL                模型路径或名称 (默认: /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct)"
      echo "  --case_start START           起始案例索引 (默认: 1)"
      echo "  --case_end END               结束案例索引 (默认: 10)"
      echo "  --iterations NUM             MCTS迭代次数 (默认: 10)"
      echo "  --branch_factor NUM          MCTS扩展的分支因子 (默认: 3)"
      echo "  --branch_factor_init NUM     初始步骤生成的分支因子 (默认: 4)"
      echo "  --rollout_num NUM            展开次数 (默认: 2)"
      echo "  --max_depth NUM              MCTS树的最大深度 (默认: 5)"
      echo "  --balance_beta NUM           过程奖励和rollout奖励的加权系数 (0-1) (默认: 0.5)"
      echo "  --run_mode MODE              运行模式: aihub, transformer, vllm, async_vllm, debug (默认: async_vllm)"
      echo "  --pairwise_weight NUM        对比奖励信号的权重 (默认: 0.5)"
      echo "  --process_weight NUM         过程评估信号的权重 (默认: 0.2)"
      echo "  --rollout_weight NUM         rollout模拟评估信号的权重 (默认: 0.3)"
      echo "  --help                       显示此帮助信息"
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
echo "运行MCTS推理..."
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "案例范围: $CASE_START-$CASE_END"
echo "迭代次数: $ITERATIONS"
echo "分支因子: $BRANCH_FACTOR"
echo "初始分支因子: $BRANCH_FACTOR_INIT"
echo "展开次数: $ROLLOUT_NUM"
echo "最大深度: $MAX_DEPTH"
echo "平衡系数: $BALANCE_BETA"
echo "运行模式: $RUN_MODE"
echo "对比奖励信号的权重: $PAIRWISE_WEIGHT"
echo "过程评估信号的权重: $PROCESS_WEIGHT"
echo "rollout模拟评估信号的权重: $ROLLOUT_WEIGHT"
echo ""

# 设置数据集目录的绝对路径
DATASET_DIR="/data/wuyang/MCTS_Reasoning/datasets"

# 运行Python脚本

export VLLM_LOGGING_LEVEL=error

cd /data/wuyang/MCTS_Reasoning && python MCTS_reasoning.py \
  --dataset_name "$DATASET" \
  --model "$MODEL" \
  --case_start "$CASE_START" \
  --case_end "$CASE_END" \
  --num_iterations "$ITERATIONS" \
  --branch_factor "$BRANCH_FACTOR" \
  --branch_factor_init "$BRANCH_FACTOR_INIT" \
  --rollout_num "$ROLLOUT_NUM" \
  --max_depth "$MAX_DEPTH" \
  --balance_beta "$BALANCE_BETA" \
  --run_mode "$RUN_MODE" \
  --pairwise_weight "$PAIRWISE_WEIGHT" \
  --process_weight "$PROCESS_WEIGHT" \
  --rollout_weight "$ROLLOUT_WEIGHT" \
