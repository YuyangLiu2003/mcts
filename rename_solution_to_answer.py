#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：将GPQA.json文件中所有顶层的"solution"键重命名为"answer"
要求：
1. 仅修改键名而不影响键值
2. 确保只修改顶层键名，不处理嵌套在值中的"solution"字符串
3. 保持JSON文件原有结构和格式不变
4. 处理完成后将结果保存回原文件
"""

import json
import os

def rename_solution_to_answer(file_path):
    """
    将JSON文件中所有顶层的"solution"键重命名为"answer"
    
    Args:
        file_path (str): JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False
    
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计修改的数量
        modified_count = 0
        
        # 处理数据
        if isinstance(data, list):
            # 如果是列表，遍历每个元素
            for item in data:
                if isinstance(item, dict) and 'solution' in item:
                    # 重命名键
                    item['answer'] = item.pop('solution')
                    modified_count += 1
        elif isinstance(data, dict):
            # 如果是字典，直接处理顶层键
            if 'solution' in data:
                data['answer'] = data.pop('solution')
                modified_count += 1
        
        # 将修改后的数据写回文件，保持原有格式
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！共修改了 {modified_count} 个 'solution' 键为 'answer' 键")
        return True
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        return False
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        return False

def main():
    """
    主函数
    """
    file_path = "/root/MCTS_Reasoning/datasets/GPQA.json"
    
    print(f"开始处理文件：{file_path}")
    
    # 执行重命名操作
    success = rename_solution_to_answer(file_path)
    
    if success:
        print("文件处理成功！")
    else:
        print("文件处理失败！")

if __name__ == "__main__":
    main()