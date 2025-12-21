import json
import os
import re
import argparse
from collections import defaultdict
import openai
import datetime

# 配置需要使用大模型辅助判断的文件名关键词列表
# USE_LLM_EVALUATOR_FILES = ["amc", "combined_dataset"]
USE_LLM_EVALUATOR_FILES = ["combined_dataset"]

# 答案等价性评估提示词模板
ANSWER_EQUIVALENCE_PROMPT_TEMPLATE = """You are an Answer Equivalence Evaluator. Your task is to judge if the ***EXTRACTED_ANSWER*** matches the standard ***GROUND_TRUTH*** .

Note: Ignore all irrelevant formatting differences (e.g., presence/absence of units, decimal vs. fraction representation, LaTeX formatting vs. plain text), determine if they are semantically identical in core meaning.

Input Information:
***PROMPT_TEXT*** : "{prompt_text}"
***GROUND_TRUTH*** : "{ground_truth}"
***EXTRACTED_ANSWER*** : "{extracted_answer}"

Please first provide a simple analysis process, then provide your final judgement (True indicates correct, False indicates incorrect). Your output format must strictly follow the following form:

[Analysis]: Simple analysis process (Do not re-solve the problem, only determine whether the provided ***EXTRACTED_ANSWER*** is consistent with the ***GROUND_TRUTH***)
[Judgement]: True or False
"""

class AIHubMix:
    def __init__(self, model_name="DeepSeek-V3", api_key=None):
        """
        初始化AI Hub Mix问答客户端
        
        参数:
            model_name: 模型名称(默认:DeepSeek-V3)
            api_key: 必须提供的API密钥
        """
        if not api_key:
            raise ValueError("API key is required for aihubmix.com service")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://aihubmix.com/v1"
        )
        self.model = model_name

    def ask(self, question, temperature=0.7, max_tokens=2048):
        """
        向大模型提出单个问题并获取回答
        
        参数:
            question: 要提问的问题文本
            temperature: 生成温度(0-1, 默认0.7)
            max_tokens: 最大返回token数(默认1024)
            
        返回:
            模型生成的回答文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        
        except openai.APIError as e:
            print(f"API请求失败: {e}")
            return None
        except Exception as e:
            print(f"发生意外错误: {e}")
            return None

class AnswerEquivalenceEvaluator:
    def __init__(self, llm_model):
        """
        初始化答案等价性评估器
        
        Args:
            llm_model: AIHubMix模型实例
        """
        self.llm_model = llm_model

    def evaluate_equivalence(self, prompt_text, ground_truth, extracted_answer):
        """评估答案等价性并返回判断结果和完整评估文本"""
        # 构建评估提示
        prompt = ANSWER_EQUIVALENCE_PROMPT_TEMPLATE.format(
            prompt_text=prompt_text,
            ground_truth=ground_truth,
            extracted_answer=extracted_answer
        )
        
        # 使用AIHubMix模型的ask方法生成评估
        evaluation_text = self.llm_model.ask(
            question=prompt,
            temperature=0.7,
            max_tokens=1024,
        )
        
        # 从响应中提取判断结果
        try:
            # 获取所有非空行
            lines = [line.strip() for line in evaluation_text.split('\n') if line.strip()]
            
            if lines:
                # 获取最后一行文本
                last_line = lines[-1]
                
                # 在最后一行中寻找True或False（忽略大小写）
                if re.search(r'\btrue\b', last_line, re.IGNORECASE):
                    return True, evaluation_text
                elif re.search(r'\bfalse\b', last_line, re.IGNORECASE):
                    return False, evaluation_text
                else:
                    # 如果在最后一行没找到，尝试在整个文本中寻找
                    if re.search(r'\btrue\b', evaluation_text, re.IGNORECASE):
                        return True, evaluation_text
                    elif re.search(r'\bfalse\b', evaluation_text, re.IGNORECASE):
                        return False, evaluation_text
                    else:
                        # 如果没有找到明确的判断，默认返回False
                        return False, evaluation_text
            else:
                # 如果没有有效行，默认返回False
                return False, evaluation_text
                
        except:
            # 出现异常时默认返回False
            return False, evaluation_text

def remove_newlines_only(text):
    """仅移除字符串中的换行符，保留空格"""
    if isinstance(text, str):
        return text.replace('\n', '').replace('\r', '').strip()
    return str(text)

def remove_all_whitespace(text):
    """移除字符串中的所有空白字符（包括空格、换行等）"""
    if isinstance(text, str):
        return re.sub(r'\s+', '', text).strip()
    return str(text)

def compare_answers(extracted, truth, llm_evaluator=None, prompt_text="", filename=""):
    """
    比较提取答案和真实答案是否一致
    使用多层次判断策略
    
    返回:
        (initial_result, llm_result, final_result, llm_evaluation_text): 初始结果，大模型结果，最终结果，大模型评估文本
    """
    # 首先尝试直接比较（不进行任何预处理）
    if str(extracted) == str(truth):
        return True, None, True, None
    
    # 对于非验证器的情况，移除所有空白字符进行比较
    extracted_clean = remove_all_whitespace(str(extracted))
    truth_clean = remove_all_whitespace(str(truth))
    
    # 检查字符串中是否包含True或False（忽略大小写）
    extracted_has_bool = "true" in extracted_clean.lower() or "false" in extracted_clean.lower()
    truth_has_bool = "true" in truth_clean.lower() or "false" in truth_clean.lower()
    
    # 如果任一字符串包含布尔值，则进行布尔值比较
    if extracted_has_bool or truth_has_bool or isinstance(truth, bool):
        # 提取布尔值（忽略大小写）
        extracted_bool = None
        if "true" in extracted_clean.lower():
            extracted_bool = True
        elif "false" in extracted_clean.lower():
            extracted_bool = False
            
        # 确定真实答案的布尔值
        if isinstance(truth, bool):
            truth_bool = truth
        elif "true" in truth_clean.lower():
            truth_bool = True
        elif "false" in truth_clean.lower():
            truth_bool = False
        else:
            truth_bool = None
            
        # 如果都能确定布尔值，则比较布尔值
        if extracted_bool is not None and truth_bool is not None:
            initial_result = extracted_bool == truth_bool
        else:
            # 如果不能确定布尔值，则回退到字符串比较
            initial_result = extracted_clean == truth_clean
    else:
        # 尝试提取数字（包括整数、小数、负数）
        num_match = re.search(r'-?\d+\.?\d*', extracted_clean)
        if num_match:
            # 提取到数字，尝试转换为浮点数
            try:
                extracted_num = float(num_match.group())
                # 与真实答案比较（允许浮点数精度误差）
                initial_result = abs(extracted_num - float(truth)) < 1e-5
            except (ValueError, TypeError):
                # 转换失败则回退到字符串比较
                initial_result = extracted_clean == truth_clean
        else:
            # 没有数字则处理纯字母情况
            initial_result = extracted_clean == truth_clean
    
    llm_result = None
    llm_evaluation_text = None
    
    # 检查是否应该使用大模型辅助判断
    use_llm_evaluator = any(keyword in filename for keyword in USE_LLM_EVALUATOR_FILES)
    
    # 如果初始判断为错误，且需要大模型辅助判断，则使用大模型进行辅助判断
    if not initial_result and use_llm_evaluator and llm_evaluator is not None:
        try:
            # 直接从当前数据项中获取prompt_text
            if not prompt_text:
                prompt_text = "No prompt text available"
                
            llm_result, llm_evaluation_text = llm_evaluator.evaluate_equivalence(
                prompt_text, str(truth), str(extracted)  # 确保所有参数都是字符串
            )
            print(f"大模型辅助判断: 初始结果={initial_result}, 大模型结果={llm_result}")
            final_result = llm_result
        except Exception as e:
            print(f"大模型评估错误: {e}")
            final_result = initial_result
    else:
        final_result = initial_result
    
    return initial_result, llm_result, final_result, llm_evaluation_text

def load_dataset(dataset_path):
    """
    加载数据集并返回问题ID到答案和问题的映射
    
    参数:
        dataset_path: 数据集文件路径
        
    返回:
        dict: 问题ID到正确答案和问题的映射
    """
    correct_data = {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        for item in dataset:
            case_id = item.get('index') + 1  # 索引从0开始，但case_id从1开始
            answer = str(item.get('answer', '')).strip()
            question = item.get('question', '').strip()
            correct_data[case_id] = {
                'answer': answer,
                'question': question
            }
            
        return correct_data
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
            if not filename.startswith(dataset_name) or not filename.endswith('_answer.txt'):
                continue
                
            # 从文件名中提取case_id
            match = re.search(rf'{dataset_name}_case_(\d+)_answer\.txt', filename)
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

def evaluate_answers(correct_data, extracted_answers, llm_evaluator, dataset_name):
    """
    评估答案的准确性，使用多层次判断策略
    
    参数:
        correct_data: 正确答案和问题的字典 {case_id: {'answer': answer, 'question': question}}
        extracted_answers: 提取的答案的字典 {case_id: answer}
        llm_evaluator: 大模型评估器实例
        dataset_name: 数据集名称
        
    返回:
        tuple: (准确率, 正确数, 总数, 未找到的case_ids, 案例结果详情, 大模型评估详情)
    """
    if not correct_data or not extracted_answers:
        return 0.0, 0, 0, [], {}, {}
        
    correct_count = 0
    total_evaluated = 0
    not_found = []
    case_results = {}
    llm_evaluations = {}  # 单独存储大模型评估文本
    
    # 遍历所有正确答案
    for case_id, data in correct_data.items():
        if case_id in extracted_answers:
            correct_answer = data['answer']
            question_text = data['question']
            extracted_answer = extracted_answers[case_id]
            
            # 使用多层次判断策略比较答案
            initial_result, llm_result, final_result, llm_evaluation_text = compare_answers(
                extracted_answer, 
                correct_answer,
                llm_evaluator=llm_evaluator,
                prompt_text=question_text,
                filename=dataset_name
            )
            
            # 主结果文件不包含大模型评估文本
            case_results[case_id] = {
                "correct_answer": correct_answer,
                "extracted_answer": extracted_answer,
                "question": question_text,
                "initial_result": initial_result,
                "llm_result": llm_result,
                "final_result": final_result
            }
            
            # 单独存储大模型评估文本
            if llm_evaluation_text is not None:
                llm_evaluations[case_id] = {
                    "question": question_text,
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "initial_result": initial_result,
                    "llm_result": llm_result,
                    "final_result": final_result,
                    "llm_evaluation_text": llm_evaluation_text
                }
            
            if final_result:
                correct_count += 1
                
            total_evaluated += 1
        else:
            not_found.append(case_id)
            
    # 计算准确率
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    
    return accuracy, correct_count, total_evaluated, not_found, case_results, llm_evaluations

def print_detailed_results(case_results, dataset_name):
    """
    打印详细的评估结果
    
    参数:
        case_results: 每个案例的评估结果
        dataset_name: 数据集名称
    """
    print(f"\n详细结果 - {dataset_name}:")
    print("-" * 150)  # 增加分隔线宽度
    # 增加各列的宽度
    print(f"{'Case ID':<12} {'Correct':<25} {'Extracted':<30} {'Initial':<12} {'LLM':<12} {'Final':<12}")
    print("-" * 150)  # 增加分隔线宽度
    
    # 按case_id排序
    for case_id in sorted(case_results.keys()):
        result = case_results[case_id]
        initial_status = "✓" if result["initial_result"] else "✗"
        llm_status = "✓" if result["llm_result"] else "✗" if result["llm_result"] is False else "-"
        final_status = "✓" if result["final_result"] else "✗"
        
        # 处理过长的文本显示，增加截断长度
        correct_display = result['correct_answer']
        extracted_display = result['extracted_answer']
        
        if len(correct_display) > 22:
            correct_display = correct_display[:22] + "..."
        if len(extracted_display) > 28:
            extracted_display = extracted_display[:28] + "..."
            
        # 使用增加后的列宽
        print(f"{case_id:<12} {correct_display:<25} {extracted_display:<30} {initial_status:^12} {llm_status:^12} {final_status:^12}")

def print_llm_evaluations(llm_evaluations, dataset_name):
    """
    打印大模型评估详情
    
    参数:
        llm_evaluations: 大模型评估详情字典
        dataset_name: 数据集名称
    """
    if not llm_evaluations:
        return
        
    print(f"\n大模型评估详情 - {dataset_name}:")
    print("=" * 120)
    
    for case_id, result in llm_evaluations.items():
        print(f"\nCase ID: {case_id}")
        print(f"Question: {result['question']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print(f"Extracted Answer: {result['extracted_answer']}")
        print(f"Initial Result: {'✓' if result['initial_result'] else '✗'}")
        print(f"LLM Result: {'✓' if result['llm_result'] else '✗'}")
        print(f"Final Result: {'✓' if result['final_result'] else '✗'}")
        print("LLM Evaluation:")
        print("-" * 40)
        print(result["llm_evaluation_text"])
        print("-" * 40)
        print()

def save_results(accuracy, correct_count, total_evaluated, case_results, llm_evaluations, output_path, dataset_name, log_file=None):
    """
    保存评估结果到文件
    
    参数:
        accuracy: 准确率
        correct_count: 正确答案数
        total_evaluated: 评估的总答案数
        case_results: 每个案例的结果（不包含大模型评估文本）
        llm_evaluations: 大模型评估详情
        output_path: 输出文件路径
        dataset_name: 数据集名称
        log_file: 日志文件路径，用于提取超参数信息（可选）
    """
    if not output_path:
        return
        
    # 生成基础文件名（不带扩展名）
    base_name = os.path.splitext(output_path)[0]
    
    # 保存主结果文件（不包含大模型评估文本）
    result_data = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_evaluated": total_evaluated,
        "evaluation_time": datetime.datetime.now().isoformat(),
        "case_results": case_results
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"主结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存主结果时出错: {str(e)}")
    
    # 保存大模型评估文本到单独文件
    if llm_evaluations:
        llm_output_path = f"{base_name}_llm_evaluations.json"
        llm_data = {
            "dataset": dataset_name,
            "llm_evaluated_count": len(llm_evaluations),
            "llm_corrected_count": sum(1 for eval_data in llm_evaluations.values() if eval_data["llm_result"]),
            "evaluation_time": datetime.datetime.now().isoformat(),
            "llm_evaluations": llm_evaluations
        }
        
        try:
            with open(llm_output_path, 'w', encoding='utf-8') as f:
                json.dump(llm_data, f, indent=2, ensure_ascii=False)
            print(f"大模型评估详情已保存到: {llm_output_path}")
        except Exception as e:
            print(f"保存大模型评估详情时出错: {str(e)}")
    
    # 保存文本格式的评估结果摘要
    text_output_path = f"{base_name}_summary.txt"
    try:
        with open(text_output_path, 'w', encoding='utf-8') as f:
            # 读取并写入log文件中的超参数（前23行）
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as log_f:
                        hyperparams = log_f.readlines()[:23]  # 读取前23行
                        f.write("超参数信息\n")
                        f.write("-" * 50 + "\n")
                        for line in hyperparams:
                            f.write(line)
                        f.write("\n")
                    print(f"已从log文件读取超参数并写入摘要文件")
                except Exception as e:
                    print(f"读取log文件时出错: {str(e)}")
            
            # 写入评估结果
            f.write(f"评估结果摘要 - {dataset_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"准确率: {accuracy:.2%} ({correct_count}/{total_evaluated})\n")
            f.write(f"大模型评估案例数: {len(llm_evaluations)}\n")
            f.write(f"大模型纠正案例数: {sum(1 for eval_data in llm_evaluations.values() if eval_data['llm_result'])}\n")
            f.write("\n详细结果:\n")
            f.write("-" * 150 + "\n")  # 增加分隔线宽度
            # 增加各列的宽度
            f.write(f"{'Case ID':<12} {'Correct':<25} {'Extracted':<30} {'Initial':<12} {'LLM':<12} {'Final':<12}\n")
            f.write("-" * 150 + "\n")  # 增加分隔线宽度
            
            for case_id in sorted(case_results.keys()):
                result = case_results[case_id]
                initial_status = "✓" if result["initial_result"] else "✗"
                llm_status = "✓" if result["llm_result"] else "✗" if result["llm_result"] is False else "-"
                final_status = "✓" if result["final_result"] else "✗"
                
                correct_display = result['correct_answer']
                extracted_display = result['extracted_answer']
                
                if len(correct_display) > 22:
                    correct_display = correct_display[:22] + "..."
                if len(extracted_display) > 28:
                    extracted_display = extracted_display[:28] + "..."
                    
                # 使用增加后的列宽
                f.write(f"{case_id:<12} {correct_display:<25} {extracted_display:<30} {initial_status:^12} {llm_status:^12} {final_status:^12}\n")
        
        print(f"评估结果摘要已保存到: {text_output_path}")
    except Exception as e:
        print(f"保存评估结果摘要时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='评估提取的答案与数据集标准答案的准确率（使用多层次判断策略）')
    parser.add_argument('--dataset', required=True, help='数据集文件路径，如datasets/GSM8K.json')
    parser.add_argument('--answers_dir', required=True, help='提取答案的目录路径')
    parser.add_argument('--dataset_name', required=True, help='数据集名称，如GSM8K')
    parser.add_argument('--output', help='输出结果的JSON文件路径（可选）')
    parser.add_argument('--detailed', action='store_true', help='是否打印详细结果')
    parser.add_argument('--show_llm', action='store_true', help='是否显示大模型评估详情')
    parser.add_argument('--api_key', help='AIHubMix API密钥，如不提供则使用环境变量AIHUBMIX_API_KEY')
    parser.add_argument('--log_file', help='日志文件路径，用于提取超参数信息（可选）')
    
    args = parser.parse_args()
    
    # 初始化大模型评估器
    llm_evaluator = None
    try:
        api_key = args.api_key or os.environ.get("AIHUBMIX_API_KEY")
        if api_key:
            aihubmix_model = AIHubMix(api_key=api_key)
            llm_evaluator = AnswerEquivalenceEvaluator(aihubmix_model)
            print("AIHubMix评估器初始化成功")
        else:
            print("警告: 未提供API密钥，大模型评估功能将不可用")
    except Exception as e:
        print(f"AIHubMix评估器初始化失败: {e}")
        llm_evaluator = None
    
    # 加载正确答案和问题
    correct_data = load_dataset(args.dataset)
    if not correct_data:
        print(f"错误: 无法加载数据集 {args.dataset}")
        return
        
    # 加载提取的答案
    extracted_answers = load_extracted_answers(args.answers_dir, args.dataset_name)
    if not extracted_answers:
        print(f"错误: 在 {args.answers_dir} 中未找到提取的答案")
        return
        
    # 评估答案
    accuracy, correct_count, total_evaluated, not_found, case_results, llm_evaluations = evaluate_answers(
        correct_data, extracted_answers, llm_evaluator, args.dataset_name)
        
    # 打印结果
    print(f"\n评估结果 - {args.dataset_name}:")
    print(f"准确率: {accuracy:.2%} ({correct_count}/{total_evaluated})")
    
    # 统计大模型评估情况
    llm_evaluated_count = len(llm_evaluations)
    llm_corrected_count = sum(1 for eval_data in llm_evaluations.values() if eval_data["llm_result"])
    
    if llm_evaluated_count > 0:
        print(f"大模型评估案例: {llm_evaluated_count} 个")
        print(f"大模型纠正案例: {llm_corrected_count} 个")
    
    if not_found:
        print(f"未找到的案例: {len(not_found)} 个")
        if len(not_found) <= 10:
            print(f"缺失的 Case IDs: {not_found}")
        else:
            print(f"前10个缺失的 Case IDs: {not_found[:10]}...")
    
    # 如果需要，打印详细结果
    if args.detailed:
        print_detailed_results(case_results, args.dataset_name)
        
    # 如果需要，打印大模型评估详情
    if args.show_llm:
        print_llm_evaluations(llm_evaluations, args.dataset_name)
        
    # 保存结果
    if args.output:
        save_results(accuracy, correct_count, total_evaluated, case_results, llm_evaluations, args.output, args.dataset_name, args.log_file)

if __name__ == '__main__':
    main()