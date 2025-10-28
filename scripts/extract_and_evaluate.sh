#!/bin/bash

PROJECT_DIR="/data/wuyang/MCTS_Reasoning"
# 设置变量
# 这里只需要修改LOG_DIR中的文件地址和DATASET_PATH中的文件地址即可，其余不需要修改
LOG_DIR="${PROJECT_DIR}/logs/GPQA-Diamond_20251020_173006_1_198"
DATASET_PATH="${PROJECT_DIR}/datasets/GPQA-Diamond.json"
# LOG_DIR="/root/autodl-tmp/MCTS_Async/logs/combined_dataset_20251010_082349_0_69"
# DATASET_PATH="datasets/combined_dataset.json"
RESULTS_DIR="${PROJECT_DIR}/results"
API_KEY="sk-1HeJFqeZxQEgTBfoD402D4694eE04580874298579142A414"

# 从LOG_DIR提取文件夹名称
LOG_DIR_BASENAME=$(basename "$LOG_DIR")

# 设置OUTPUT_DIR，使用与LOG_DIR相同的文件夹名称
OUTPUT_DIR="${PROJECT_DIR}/extract/${LOG_DIR_BASENAME}"

# 从DATASET_PATH提取dataset_name（JSON文件的前缀名称）
DATASET_NAME=$(basename "$DATASET_PATH" .json)

# 设置结果文件路径 - 在results下创建子文件夹
RESULTS_SUBDIR="${RESULTS_DIR}/${LOG_DIR_BASENAME}"
RESULTS_FILE="${RESULTS_SUBDIR}/results.json"

echo "开始执行答案提取和评估流程..."
echo "输入日志目录: $LOG_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "数据集名称: $DATASET_NAME"
echo "结果子目录: $RESULTS_SUBDIR"
echo "结果文件: $RESULTS_FILE"

# 第一步：答案提取
echo "步骤1: 提取答案..."
mkdir -p ${OUTPUT_DIR}

python extract_reasoning.py \
    --input ${LOG_DIR} \
    --reasoning_output ${OUTPUT_DIR}/reasoning_paths \
    --answer_output ${OUTPUT_DIR}/answers \
    --dataset_name ${DATASET_NAME}

echo "提取完成。"
echo "推理路径已保存到: ${OUTPUT_DIR}/reasoning_paths"
echo "答案已保存到: ${OUTPUT_DIR}/answers"

# 第二步：答案评估
echo "步骤2: 评估答案..."
# 确保results主目录和子目录都存在
mkdir -p ${RESULTS_SUBDIR}

python evaluate_answers.py \
    --dataset ${DATASET_PATH} \
    --answers_dir ${OUTPUT_DIR}/answers \
    --dataset_name ${DATASET_NAME} \
    --detailed \
    --output ${RESULTS_FILE} \
    --api_key ${API_KEY} \
    --log_file ${LOG_DIR}/case_0001.txt

echo "评估完成。"
echo "评估结果已保存到: ${RESULTS_FILE}"
echo "详细结果保存在子目录: ${RESULTS_SUBDIR}"

echo "整个流程执行完毕！"