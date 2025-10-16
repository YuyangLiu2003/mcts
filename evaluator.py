import re
import asyncio
import numpy as np
from typing import Any, List, Dict
from stop_sequences import get_stopping_criteria
from text_handler import PromptHandler, ResponseHandler
from log_func import test_out

class RewardModel:
    def __init__(self, llm_client: Any, prompt_handler: PromptHandler, response_handler: ResponseHandler, dataset_name: str = None):
        """
        初始化奖励模型
        
        Args:
            llm_client: LLM客户端实例
            prompt_handler: 提示词处理器实例
            dataset_name: 数据集名称
        """
        self.llm_client = llm_client
        self.prompt_handler = prompt_handler
        self.dataset_name = dataset_name
        self.response_handler = response_handler

    async def evaluate(self, full_solution: str, objectives: str = "", return_response: bool = False, case_idx: int = None):
        """
        使用LLM评估答案质量
        """
        prompt = self.prompt_handler.get_reward_prompt(full_solution, objectives)
        response = await self.llm_client.generate(prompt,
                                                stop_stage='reward',
                                                skip_special_tokens=False)
        score = self.response_handler.parse_reward_response(response)
        if return_response:
            return score, prompt, response
        return score

class ProcessRewardModel:
    def __init__(self, llm_client: Any, prompt_handler: PromptHandler, response_handler: ResponseHandler, dataset_name: str = None):
        """
        初始化过程奖励模型
        
        Args:
            llm_client: LLM客户端实例  
            prompt_handler: 提示词处理器实例
            response_handler: 响应处理器实例
            dataset_name: 数据集名称
        """
        self.llm_client = llm_client
        self.prompt_handler = prompt_handler
        self.response_handler = response_handler
        self.dataset_name = dataset_name

    async def evaluate(self, previous_steps: str, current_step: str, process_criterions: str, return_response: bool = False, case_idx: int = None):
        """
        使用LLM评估中间节点质量
        """
        prompt = self.prompt_handler.get_process_evaluation_prompt(previous_steps, current_step, process_criterions)
        response = await self.llm_client.generate(prompt,
                                                stop_stage='process_evaluation',
                                                skip_special_tokens=False)
        score = self.response_handler.parse_process_evaluation_response(response)
        if return_response:
            return score, prompt, response
        return score

class PairwiseRewardModel:
    def __init__(self, llm_client: Any, prompt_handler: PromptHandler, response_handler: ResponseHandler, dataset_name: str = None):
        """
        初始化成对比较奖励模型
        
        Args:
            llm_client: LLM客户端实例
            prompt_handler: 提示词处理器实例
            response_handler: 响应处理器实例
            dataset_name: 数据集名称
        """
        self.llm_client = llm_client
        self.prompt_handler = prompt_handler
        self.response_handler = response_handler
        self.dataset_name = dataset_name

    def _get_pairs(self, new_nodes: List[Any]) -> List[Dict[str, Any]]:
        """
        对于所有输入的new_nodes，构成两两比较对
        
        Args:
            new_nodes: 新生成的节点列表
        Returns:
            比较对列表，每个比较对包含stepA、stepB和score
        """
        pairs = []
        n = len(new_nodes)
        
        for i in range(n):
            for j in range(i + 1, n):
                pair = {
                    "stepA": new_nodes[i].state,
                    "stepB": new_nodes[j].state,
                    "score": 0,
                    "nodeA_idx": i,
                    "nodeB_idx": j
                }
                pairs.append(pair)
        
        return pairs

    async def _pair_comparison(self, previous_steps: str, pair: Dict[str, Any], process_criterions: str, case_idx: int) -> Dict[str, Any]:
        """
        针对一个比较对，调用LLM进行比较并获取分数
        """
        prompt = self.prompt_handler.get_pair_evaluation_prompt(
            previous_steps, 
            pair["stepA"], 
            pair["stepB"], 
            process_criterions
        )
        response = await self.llm_client.generate(prompt,
                                                stop_stage='pair_evaluation',
                                                skip_special_tokens=False,)
        score = self.response_handler.parse_pair_evaluation_response(response)
        pair["score"] = score
        pair["prompt"] = prompt
        pair["response"] = response
        return pair

    def _get_bradley_terry_score(self, pairs: List[Dict[str, Any]], num_nodes: int) -> List[float]:
        """
        使用考虑优势程度的Bradley-Terry模型计算每个节点的优势分数
        最终结果映射到0-10范围
        
        Args:
            pairs: 比较对列表，包含优势优势程度分数(-5到+5)
            num_nodes: 节点总数
        Returns:
            每个节点的优势分数列表(0-10)
        """
        # 初始化能力参数 r_i > 0
        r = np.full(num_nodes, 1e-5, dtype=np.float64)  # 初始值设为小正数
        epsilon = 1e-8  # 收敛阈值
        max_iterations = 1000  # 最大迭代次数
        min_r = 1e-10  # 防止r变为0的最小值
        
        # 整理比赛数据：记录每个节点的对手和优势程度
        sum_p = [0.0 for _ in range(num_nodes)]  # 每个节点的获胜概率总和
        opponents = [[] for _ in range(num_nodes)]  # 每个节点的所有对手
        pair_weights = [[] for _ in range(num_nodes)]  # 每次比较的权重
        
        for pair in pairs:
            i = pair["nodeA_idx"]
            j = pair["nodeB_idx"]
            score = pair["score"]
            
            # 将分数转换为A胜B的概率 (0-1之间)
            # 分数越高，A胜B的概率越大，但避免极端值0和1
            p_ij = 0.98 * (score + 5) / 10 + 0.01  # 映射到[0.01, 0.99]区间
            p_ji = 1.0 - p_ij  # B胜A的概率
            
            # 记录对手和对应的权重
            opponents[i].append(j)
            opponents[j].append(i)
            pair_weights[i].append(p_ij)
            pair_weights[j].append(p_ji)
            
            # 累加获胜概率
            sum_p[i] += p_ij
            sum_p[j] += p_ji
        
        # 牛顿-拉夫森迭代法最大化似然函数
        for _ in range(max_iterations):
            r_old = r.copy()
            
            # 对每个节点更新参数
            for i in range(num_nodes):
                # 计算分母：sum(1/(r_i + r_j)) for all j in opponents of i
                denominator = 0.0
                for j in opponents[i]:
                    denominator += 1.0 / (r[i] + r[j])
                
                # 更新参数，确保不会出现零或负值
                if denominator > 0 and sum_p[i] > 0:
                    new_r = sum_p[i] / denominator
                    r[i] = max(new_r, min_r)  # 确保不小于最小值
            
            # 检查收敛性（参数变化小于阈值）
            if np.linalg.norm(r - r_old) < epsilon:
                break
        
        # 将能力参数转换为0-10范围的分数
        try:
            # 1. 先取对数将乘性模型转为加性模型
            log_r = np.log(np.maximum(r, min_r))  # 确保安全取对数
            
            # 2. 标准化到[0, 1]区间
            min_log = np.min(log_r)
            max_log = np.max(log_r)
            log_range = max_log - min_log
            
            # 处理所有节点能力相同的特殊情况
            if log_range < 1e-10:
                return [5.0] * num_nodes  # 中间值5作为默认
            
            # 3. 映射到0-10范围（修改部分）
            normalized_scores = (log_r - min_log) / log_range
            node_adv_scores = normalized_scores * 10  # 缩放至0-10
            
            return node_adv_scores.tolist()
            
        except Exception as e:
            # 任何计算错误时返回默认分数
            print(f"计算过程中出现错误: {e}")
            return [5.0] * num_nodes  # 中间值5作为默认

    async def pairwise_evaluate(self, previous_steps: str, new_nodes: List[Any], process_criterions: str, case_idx: int = None) -> List[float]:
        """
        外部调用的方法，执行成对比较评估
        """
        pairs = self._get_pairs(new_nodes)
        evaluated_pairs = []
        if pairs:
            comparison_tasks = [
                self._pair_comparison(previous_steps, pair, process_criterions, case_idx)
                for pair in pairs
            ]
            evaluated_pairs = await asyncio.gather(*comparison_tasks)
        node_adv_scores = self._get_bradley_terry_score(evaluated_pairs, len(new_nodes))
        return node_adv_scores, evaluated_pairs
