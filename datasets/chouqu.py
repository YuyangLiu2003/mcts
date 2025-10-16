import json
import random

def merge_json_files(file1, file2, output_file):
    """
    合并两个JSON文件，去除重复的ID
    """
    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 使用字典来去重，以id为键
    merged_dict = {}
    
    # 添加第一个文件的数据
    for item in data1:
        item_id = item.get('id', '')
        if item_id:  # 确保有id字段
            merged_dict[item_id] = item
    
    # 添加第二个文件的数据，跳过重复的id
    for item in data2:
        item_id = item.get('id', '')
        if item_id and item_id not in merged_dict:
            merged_dict[item_id] = item
    
    # 转换为列表
    merged_data = list(merged_dict.values())
    
    # 保存合并后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功合并两个文件，共{len(merged_data)}个问题，保存到{output_file}")
    return merged_data

def sample_questions_from_merged(data, output_file, sample_size=100):
    """
    从合并的数据中随机抽取指定数量的问题
    """
    # 确保数据量足够
    if len(data) < sample_size:
        raise ValueError(f"数据量不足，需要{sample_size}个问题，但只有{len(data)}个")
    
    # 随机抽样
    sampled_questions = random.sample(data, sample_size)
    
    # 保存抽样结果到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_questions, f, ensure_ascii=False, indent=2)
    
    print(f"成功从{len(data)}个问题中抽取{sample_size}个问题，保存到{output_file}")

# 使用示例
if __name__ == "__main__":
    file1 = "Math100_1.json"
    file2 = "Math100_2.json"
    merged_file = "Math_merged.json"
    output_file = "Math100.json"
    
    # 合并两个文件
    merged_data = merge_json_files(file1, file2, merged_file)
    
    # 从合并的数据中抽取100个问题
    sample_questions_from_merged(merged_data, output_file, 100)