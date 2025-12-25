import json
import re
import os

def reformat_problem_text(original_text):
    """
    使用正则表达式提取初始状态和目标状态，并填充到新模板中。
    """
    # 1. 定义正则表达式提取关键信息
    # 提取初始条件：查找 "As initial conditions I have that," 和 "\nMy goal" 之间的内容
    # re.DOTALL 允许 . 匹配换行符，虽然这里主要处理单行文本，但加上更稳健
    init_pattern = r"As initial conditions I have that,\s*(.*?)\s*\nMy goal"
    
    # 提取目标状态：查找 "My goal is to have that" 和 ".\n\nPlease" 之间的内容
    goal_pattern = r"My goal is to have that\s*(.*?)\.\n\nPlease"

    # 执行匹配
    init_match = re.search(init_pattern, original_text, re.DOTALL)
    goal_match = re.search(goal_pattern, original_text, re.DOTALL)

    # 如果匹配失败，返回原始文本或错误提示（防止程序崩溃）
    if not init_match or not goal_match:
        return original_text 

    # 获取原始内容并去除首尾空白
    raw_init = init_match.group(1).strip()
    raw_goal = goal_match.group(1).strip()

    # 2. 文本清洗函数
    def clean_sentence(text):
        if not text:
            return ""
        # 将 ", the" 替换为 ". The" 以分句，增加可读性
        text = text.replace(", the", ". The")
        # 确保第一个字母大写
        text = text[0].upper() + text[1:]
        return text

    # 处理提取出的文本
    formatted_init = clean_sentence(raw_init)
    formatted_goal = clean_sentence(raw_goal)

    # 3. 定义新的模板
    new_template = (
        "##Question\n"
        "I am playing a block stacking game. The game begins with an initial configuration of blocks. "
        "I need to move the blocks step by step using four specific actions (pick up, put down, unstack, stack) "
        "to ultimately reach a preset goal state.\n"
        "**Initial Conditions**\n"
        "As initial conditions, I have the following setup:\n"
        f"{formatted_init}.\n"
        "**Goal State**\n"
        "My goal is to rearrange the blocks to achieve the following state:\n"
        f"{formatted_goal}."
    )

    return new_template

def process_dataset():
    input_path = './datasets/blocksworld_step_6.json'
    output_path = './datasets/blocksworld_local_6.json'

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        # 为了演示，如果文件不存在，我可以创建一个假的数据文件供测试（可选）
        return

    print(f"Reading from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")
        return

    processed_data = []

    for item in data:
        # 构建新的数据对象
        new_item = {
            "index": item.get("id"),
            "question": reformat_problem_text(item.get("problem", "")),
            "pddl_content": item.get("pddl_content"),
            "answer": item.get("answer")
        }
        processed_data.append(new_item)

    print(f"Processing complete. Writing {len(processed_data)} items to {output_path}...")

    # 写入新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文字符（如果有）能正常显示，indent=4 美化输出
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    process_dataset()