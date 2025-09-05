#!/bin/bash

# 评估GSM8K数据集的结果
python evaluate_answers.py \
    --dataset datasets/amc23.json \
    --answers_dir extract/amc23_20250822_224056_1_40/answers \
    --dataset_name amc23 \
    --detailed \
    --output results/amc23_20250822_224056_1_40.json

# 你可以取消注释下面的命令来评估其他数据集
# python evaluate_answers.py \
#     --dataset datasets/AQUA.json \
#     --answers_dir extract/AQUA_results/answers \
#     --dataset_name AQUA \
#     --detailed