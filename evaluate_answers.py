import json
import os
import re
import argparse
from collections import defaultdict

def load_dataset(dataset_path):
    """
    加载数据集并返回问题ID到答案的映射
    
    参数:
        dataset_path: 数据集文件路径
        
    返回:
        dict: 问题ID到正确答案的映射
    """
    correct_answers = {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        for item in dataset:
            case_id = item.get('index') + 1  # 索引从0开始，但case_id从1开始
            answer = item.get('answer', '').strip()
            correct_answers[case_id] = answer
            
        return correct_answers
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return {}

def load_extracted_answers(answers_dir, dataset_name):
    """
    加载提取的答案
    
    参数:
        answers_dir: 提取答案的目录
        dataset_name: 数据集名称
        
    返回:
        dict: 问题ID到提取答案的映射
    """
    extracted_answers = {}
    
    try:
        # 遍历答案目录
        for filename in os.listdir(answers_dir):
            if not filename.endswith('_answer.txt') or not filename.startswith('case_'):
                continue
                
            # 从文件名中提取case_id
            match = re.search(r'case_(\d+)_answer\.txt', filename)
            if not match:
                continue
                
            case_id = int(match.group(1))
            file_path = os.path.join(answers_dir, filename)
            
            # 读取答案内容
            with open(file_path, 'r', encoding='utf-8') as f:
                answer = f.read().strip()
                
            extracted_answers[case_id] = answer
            
        return extracted_answers
    except Exception as e:
        print(f"加载提取答案时出错: {str(e)}")
        return {}

def normalize_answer(answer):
    """
    标准化答案格式，处理常见的格式差异
    
    参数:
        answer: 原始答案字符串
        
    返回:
        str: 标准化后的答案
    """
    if answer is None:
        return ""
        
    # 移除所有非数字和非字母字符，转为小写
    answer = re.sub(r'[^\w\s]', '', answer.lower())
    answer = answer.strip()
    
    # 尝试提取数字答案
    num_match = re.search(r'\d+', answer)
    if num_match:
        return num_match.group(0)
        
    return answer

def evaluate_answers(correct_answers, extracted_answers):
    """
    评估答案的准确性
    
    参数:
        correct_answers: 正确答案的字典 {case_id: answer}
        extracted_answers: 提取的答案的字典 {case_id: answer}
        
    返回:
        tuple: (准确率, 正确数, 总数, 未找到的case_ids)
    """
    if not correct_answers or not extracted_answers:
        return 0.0, 0, 0, []
        
    correct_count = 0
    total_evaluated = 0
    not_found = []
    case_results = {}
    
    # 遍历所有正确答案
    for case_id, correct_answer in correct_answers.items():
        if case_id in extracted_answers:
            extracted_answer = extracted_answers[case_id]
            
            # 标准化答案进行比较
            norm_correct = normalize_answer(correct_answer)
            norm_extracted = normalize_answer(extracted_answer)
            
            is_correct = norm_correct == norm_extracted
            case_results[case_id] = {
                "correct_answer": correct_answer,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            }
            
            if is_correct:
                correct_count += 1
                
            total_evaluated += 1
        else:
            not_found.append(case_id)
            
    # 计算准确率
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    
    return accuracy, correct_count, total_evaluated, not_found, case_results

def print_detailed_results(case_results, dataset_name):
    """
    打印详细的评估结果
    
    参数:
        case_results: 每个案例的评估结果
        dataset_name: 数据集名称
    """
    print(f"\n详细结果 - {dataset_name}:")
    print("-" * 80)
    print(f"{'Case ID':<10} {'Correct':<10} {'Extracted':<20} {'Status':<10}")
    print("-" * 80)
    
    # 按case_id排序
    for case_id in sorted(case_results.keys()):
        result = case_results[case_id]
        status = "✓" if result["is_correct"] else "✗"
        print(f"{case_id:<10} {result['correct_answer']:<10} {result['extracted_answer']:<20} {status:<10}")

def save_results(accuracy, correct_count, total_evaluated, case_results, output_path, dataset_name):
    """
    保存评估结果到文件
    
    参数:
        accuracy: 准确率
        correct_count: 正确答案数
        total_evaluated: 评估的总答案数
        case_results: 每个案例的结果
        output_path: 输出文件路径
        dataset_name: 数据集名称
    """
    if not output_path:
        return
        
    result_data = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_evaluated": total_evaluated,
        "case_results": case_results
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='评估提取的答案与数据集标准答案的准确率')
    parser.add_argument('--dataset', required=True, help='数据集文件路径，如datasets/GSM8K.json')
    parser.add_argument('--answers_dir', required=True, help='提取答案的目录路径')
    parser.add_argument('--dataset_name', required=True, help='数据集名称，如GSM8K')
    parser.add_argument('--output', help='输出结果的JSON文件路径（可选）')
    parser.add_argument('--detailed', action='store_true', help='是否打印详细结果')
    
    args = parser.parse_args()
    
    # 加载正确答案
    correct_answers = load_dataset(args.dataset)
    if not correct_answers:
        print(f"错误: 无法加载数据集 {args.dataset}")
        return
        
    # 加载提取的答案
    extracted_answers = load_extracted_answers(args.answers_dir, args.dataset_name)
    if not extracted_answers:
        print(f"错误: 在 {args.answers_dir} 中未找到提取的答案")
        return
        
    # 评估答案
    accuracy, correct_count, total_evaluated, not_found, case_results = evaluate_answers(
        correct_answers, extracted_answers)
        
    # 打印结果
    print(f"\n评估结果 - {args.dataset_name}:")
    print(f"准确率: {accuracy:.2%} ({correct_count}/{total_evaluated})")
    
    if not_found:
        print(f"未找到的案例: {len(not_found)} 个")
        if len(not_found) <= 10:
            print(f"缺失的 Case IDs: {not_found}")
        else:
            print(f"前10个缺失的 Case IDs: {not_found[:10]}...")
    
    # 如果需要，打印详细结果
    if args.detailed:
        print_detailed_results(case_results, args.dataset_name)
        
    # 保存结果
    if args.output:
        save_results(accuracy, correct_count, total_evaluated, case_results, args.output, args.dataset_name)

if __name__ == '__main__':
    main()