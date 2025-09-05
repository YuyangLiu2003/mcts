#!/bin/bash

# 这个脚本用于运行MCTS_reasoning.py，实现基于蒙特卡洛树搜索的推理

# 设置默认参数
DATASET="math500"          # 数据集名称，可选：amc23, GSM8K, aime2024等
MODEL="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"  # 模型路径或名称
CASE_START=1           # 起始案例索引
CASE_END=1            # 结束案例索引
ITERATIONS=15           # MCTS迭代次数
BRANCH_FACTOR=3         # MCTS扩展的分支因子
BRANCH_FACTOR_INIT=4    # 初始步骤生成的分支因子
ROLLOUT_NUM=5           # 展开次数
MAX_DEPTH=6             # MCTS树的最大深度
BALANCE_BETA=0.5        # 过程奖励和rollout奖励的加权系数 (0-1)
EXPAND_GUIDANCE=""      # 扩展指导（如果为空则从search_guide.json读取）
PROCESS_CRITERIONS=""   # 过程评价标准（如果为空则从search_guide.json读取）
REWARD_OBJECTIVES=""    # 奖励目标（如果为空则从search_guide.json读取）
RUN_MODE="async_vllm"  # 运行模式：aihub, transformer, vllm, debug, async_vllm

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
    --expand_guidance)
      EXPAND_GUIDANCE="$2"
      shift 2
      ;;
    --process_criterions)
      PROCESS_CRITERIONS="$2"
      shift 2
      ;;
    --reward_objectives)
      REWARD_OBJECTIVES="$2"
      shift 2
      ;;
    --run_mode)
      RUN_MODE="$2"
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
      echo "  --expand_guidance STR        扩展指导 (默认: 从search_guide.json读取)"
      echo "  --process_criterions STR     过程评价标准 (默认: 从search_guide.json读取)"
      echo "  --reward_objectives STR      奖励目标 (默认: 从search_guide.json读取)"
      echo "  --run_mode MODE              运行模式: aihub, transformer, vllm, debug (默认: transformer)"
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
[[ -n "$EXPAND_GUIDANCE" ]] && echo "扩展指导: $EXPAND_GUIDANCE" || echo "扩展指导: 从search_guide.json读取"
[[ -n "$PROCESS_CRITERIONS" ]] && echo "过程评价标准: $PROCESS_CRITERIONS" || echo "过程评价标准: 从search_guide.json读取"
[[ -n "$REWARD_OBJECTIVES" ]] && echo "奖励目标: $REWARD_OBJECTIVES" || echo "奖励目标: 从search_guide.json读取"
echo "运行模式: $RUN_MODE"
echo ""

# 设置数据集目录的绝对路径
DATASET_DIR="/root/MCTS_Async/datasets"

# 运行Python脚本

export VLLM_LOGGING_LEVEL=error

cd /root/MCTS_Async && python MCTS_reasoning.py \
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
  ${EXPAND_GUIDANCE:+--expand_guidance "$EXPAND_GUIDANCE"} \
  ${PROCESS_CRITERIONS:+--process_criterions "$PROCESS_CRITERIONS"} \
  ${REWARD_OBJECTIVES:+--reward_objectives "$REWARD_OBJECTIVES"} \
  --run_mode "$RUN_MODE" \