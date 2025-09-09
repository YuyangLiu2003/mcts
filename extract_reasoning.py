import re
import os
import argparse

def extract_reasoning_and_answer(log_file_path):
    """
    从日志文件中提取推理路径和最终答案
    
    参数:
        log_file_path: 日志文件路径
        
    返回:
        tuple: (推理路径, 最终答案)
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 查找文件末尾的推理路径部分，从"Reasoning Path:"到文件结尾
            reasoning_match = re.search(r'Reasoning Path:(.*)', content, re.DOTALL)
            
            if not reasoning_match:
                print(f"警告: 在{log_file_path}中未找到推理路径")
                return None, None
                
            reasoning_path = reasoning_match.group(1).strip()
            
            # 提取推理路径中被\boxed{}包围的所有答案
            answer_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', reasoning_path, re.DOTALL)
            
            final_answer = None
            # 选择第一个非空的boxed内容作为答案
            if answer_matches:
                for match in answer_matches:
                    possible_answer = match.strip()
                    if possible_answer:  # 确保答案不是空字符串
                        final_answer = possible_answer
                        break
                
                if final_answer is None:  # 如果所有boxed内容都是空的
                    print(f"在{log_file_path}中未找到非空的\\boxed{{}}格式答案")
            
            # 如果未找到有效的boxed答案，使用最后一行作为答案
            if final_answer is None:
                lines = reasoning_path.strip().split('\n')
                if lines:
                    final_answer = lines[-1].strip()
                    print(f"在{log_file_path}中未找到有效的\\boxed{{}}格式答案，使用最后一行作为答案")
                else:
                    print(f"警告: 在{log_file_path}中未找到最终答案")
                    final_answer = None
                
            return reasoning_path, final_answer
    except Exception as e:
        print(f"处理{log_file_path}时出错: {str(e)}")
        return None, None

def extract_case_info(filename):
    """
    从日志文件名中提取案例ID
    
    参数:
        filename: 格式为case_{case_id}.txt的日志文件名
    
    返回:
        str: 案例ID
    """
    match = re.search(r'case_(\d+)\.txt', filename)
    if match:
        case_id = match.group(1)
        print(f"匹配到案例ID: {case_id}")
        return case_id
    return "unknown"

def process_log_files(input_path, reasoning_output_path, answer_output_path):
    """
    处理日志文件并提取推理路径和最终答案
    
    参数:
        input_path: 日志文件或包含日志文件的目录路径
        reasoning_output_path: 保存推理路径的目录路径
        answer_output_path: 保存最终答案的目录路径
    """
    # 如果输出目录不存在则创建
    os.makedirs(reasoning_output_path, exist_ok=True)
    os.makedirs(answer_output_path, exist_ok=True)
    
    if os.path.isdir(input_path):
        # 处理目录中的所有日志文件
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith('.txt') and ('log' in file or 'case_' in file):
                    file_path = os.path.join(root, file)
                    process_single_file(file_path, reasoning_output_path, answer_output_path)
    else:
        # 处理单个日志文件
        process_single_file(input_path, reasoning_output_path, answer_output_path)

def process_single_file(file_path, reasoning_output_path, answer_output_path):
    """
    处理单个日志文件并将结果写入单独的输出文件
    
    参数:
        file_path: 日志文件路径
        reasoning_output_path: 保存推理路径的目录
        answer_output_path: 保存最终答案的目录
    """
    reasoning_path, final_answer = extract_reasoning_and_answer(file_path)
    
    filename = os.path.basename(file_path)
    case_id = extract_case_info(filename)
    
    # 创建输出文件名
    reasoning_filename = f"case_{case_id}_path.txt"
    answer_filename = f"case_{case_id}_answer.txt"
    
    # 将推理路径保存到单独的文件
    if reasoning_path:
        reasoning_file_path = os.path.join(reasoning_output_path, reasoning_filename)
        with open(reasoning_file_path, 'w', encoding='utf-8') as f:
            f.write(reasoning_path)
        #print(f"已保存推理路径到: {reasoning_file_path}")
    
    # 将答案保存到单独的文件
    if final_answer:
        answer_file_path = os.path.join(answer_output_path, answer_filename)
        with open(answer_file_path, 'w', encoding='utf-8') as f:
            f.write(final_answer)
        #print(f"已保存答案到: {answer_file_path}")

def main():
    """主函数，解析命令行参数并开始处理"""
    parser = argparse.ArgumentParser(description='从日志文件中提取推理路径和最终答案')
    parser.add_argument('--input', required=True, help='日志文件或包含日志文件的目录路径')
    parser.add_argument('--reasoning_output', required=True, help='保存推理路径的目录')
    parser.add_argument('--answer_output', required=True, help='保存最终答案的目录')
    
    args = parser.parse_args()
    
    process_log_files(args.input, args.reasoning_output, args.answer_output)
    
if __name__ == '__main__':
    main()