#!/bin/bash

# 评估GSM8K数据集的结果
python evaluate_answers.py \
    --dataset datasets/math500.json \
    --answers_dir /root/data1/hesaikenew/data/ted_LLM/MCTS_Async/extract/math500_20250908_233346_1_500 \
    --dataset_name math500 \
    --detailed \
    --output results/math500_20250908_233346_1_500.json

# 你可以取消注释下面的命令来评估其他数据集
# python evaluate_answers.py \
#     --dataset datasets/AQUA.json \
#     --answers_dir extract/AQUA_results/answers \
#     --dataset_name AQUA \
#     --detailed