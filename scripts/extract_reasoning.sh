#!/bin/bash

# 使用示例1：处理单个日志文件
# python extract_reasoning.py \
#     --input logs/GSM8K_20250510_084213/log_GSM8K_case_99 \
#     --reasoning_output output/reasoning_paths \
#     --answer_output output/answers

# 使用示例2：处理整个目录中的日志文件
# python extract_reasoning.py \
#     --input logs/GSM8K_20250510_084213 \
#     --reasoning_output output/reasoning_paths \
#     --answer_output output/answers

# 使用示例3：指定输出到特定目录
# 确保输出目录存在
# mkdir -p output/analysis
# python extract_reasoning.py \
#     --input logs/GSM8K_20250510_084213 \
#     --reasoning_output output/analysis/reasoning_paths \
#     --answer_output output/analysis/answers

# 实际执行的命令
# 默认处理最新的日志目录
LOG_DIR="/root/MCTS_Reasoning/logs/amc23_20250824_165011_1_10"
OUTPUT_DIR="/root/MCTS_Reasoning/extract/amc23_20250824_165011_1_10"

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}

# 执行提取脚本
python extract_reasoning.py \
    --input ${LOG_DIR} \
    --reasoning_output ${OUTPUT_DIR}/reasoning_paths \
    --answer_output ${OUTPUT_DIR}/answers

echo "提取完成。"
echo "推理路径已保存到: ${OUTPUT_DIR}/reasoning_paths"
echo "答案已保存到: ${OUTPUT_DIR}/answers" 