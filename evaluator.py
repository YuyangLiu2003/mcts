import re
import asyncio
import numpy as np
from typing import Any, List, Dict, Tuple
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
        response = (await self.llm_client.generate(prompt,
                                                stop_stage='reward',
                                                skip_special_tokens=False,
                                                max_tokens=2048))[0]
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
        response = (await self.llm_client.generate(prompt,
                                                stop_stage='process_evaluation',
                                                skip_special_tokens=False))[0]
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
            for j in range(n):
                if i != j:  # 排除自身与自身的配对
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
        response = (await self.llm_client.generate(prompt, stop_stage='pair_evaluation', skip_special_tokens=False,))[0]
        score = self.response_handler.parse_pair_evaluation_response(response)
        pair["score"] = score
        pair["prompt"] = prompt
        pair["response"] = response
        return pair

    def _get_bradley_terry_score(self, pairs: List[Dict[str, Any]], num_nodes: int, 
                                k: float = 0.8, iter_mean: float = 6, 
                                final_scale: float = 3.8, 
                                score_range: Tuple[float, float] = (0, 10)) -> List[float]:
        """Compute ln(r) values with given comparison data using numpy for efficiency
        
        Args:
            pairs: List of dictionaries, each containing nodeA_idx, nodeB_idx, and score
            num_nodes: Number of nodes/players
            k: Sensitivity parameter for pij to score changes
            iter_mean: Mean value for iteration scaling
            final_scale: Scaling factor for final score
            score_range: Tuple of (min_score, max_score) for clipping final scores
            
        Returns:
            List of node advantage scores
        """
        r = np.ones(num_nodes, dtype=np.float64)
        min_r = 1e-8
        
        # Initialize opponents list (keeping list form as each node may have different number of opponents)
        opponents = [[] for _ in range(num_nodes)]
        sum_p = np.zeros(num_nodes, dtype=np.float64)
        
        # Process pair data, build opponents list and sum_p
        for pair in pairs:
            i = pair.get("nodeA_idx")
            j = pair.get("nodeB_idx")
            score = pair.get("score", 0)  # Default to 0 if score not provided
            
            if i is None or j is None:
                continue  # Skip invalid pairs
                
            # Calculate win probabilities
            # For situations with win/loss scores, use probability of i beating j
            # as a proxy for "number of wins" in standard BT model
            p_ij = k * score / 10 + 0.5
            p_ji = 1.0 - p_ij
            
            # Ensure indices are within valid range
            if i < num_nodes and j < num_nodes:
                opponents[i].append(j)
                opponents[j].append(i)
                sum_p[i] += p_ij
                sum_p[j] += p_ji
        
        max_iterations = 1000
        epsilon = 1e-8
        
        for iteration in range(max_iterations):
            r_old = r.copy()  # Save current r values for comparison
            
            # Update r value for each node
            for i in range(num_nodes):
                js = opponents[i]
                if not js:  # No opponents, denominator is 0
                    denominator = 0.0
                else:
                    # r[i] is current node's r value, r[js] are all opponent j's r values
                    denominator = np.sum(1.0 / (r[i] + r[js]))
                
                # Update r value (when conditions are met)
                if denominator > 0 and sum_p[i] > 0:
                    new_r = sum_p[i] / denominator
                    r[i] = max(new_r, min_r)
            
            # Scale r to have mean of iter_mean during iteration to avoid divergence
            # This doesn't change the ratio of r values
            r_sum = r.sum()
            if r_sum > 0:
                scale_factor = iter_mean * num_nodes / r_sum
                r *= scale_factor
                # Ensure not less than minimum threshold
                r = np.maximum(r, min_r)
            
            # Calculate convergence difference (Euclidean distance)
            diff = np.linalg.norm(r - r_old)
            if diff < epsilon:
                break
        
        # Convert r values to natural log, as BT model cares about ratio of r values
        bt_score = np.log(r)

        # Scale to final range
        final_score = bt_score * final_scale
        
        # Clip scores to specified range
        min_val, max_val = score_range
        final_score = np.clip(final_score, min_val, max_val)
        
        # Convert numpy array to list of floats
        return final_score.tolist()

    def get_borda_score(self, pairs: List[Dict[str, Any]], num_nodes: int, 
                        mode: str = "weighted", 
                        temperature: float = 2.0) -> List[float]:
        """
        Args:
            temperature: 控制分数分布的集中程度 (仅用于 weighted 模式)
                         - T=1.0: 标准 Sigmoid，+3分对应 9.5分 (较陡)
                         - T=2.0: 柔和，+3分对应 8.1分 (推荐，符合你 2-8 的需求)
                         - T=3.0: 非常柔和，+5分才对应 8.4分
        """
        import numpy as np
        
        points = np.zeros(num_nodes, dtype=np.float64)
        match_counts = np.zeros(num_nodes, dtype=np.float64)
        epsilon = 1e-6
        
        for pair in pairs:
            idx_a = pair.get("nodeA_idx")
            score = pair.get("score", 0) # 假设原始 score 范围是 -5 到 +5
            
            if idx_a is not None and idx_a < num_nodes:
                match_counts[idx_a] += 1
                
                if mode == "discrete":
                    # 离散模式逻辑不变
                    if score > epsilon: points[idx_a] += 1.0
                    elif score < -epsilon: points[idx_a] += 0.0
                    else: points[idx_a] += 0.5
                
                elif mode == "weighted":
                    # 加权模式：累加原始分差
                    # 我们可以稍微 clamp 一下防止极端脏数据，比如 -10 到 10
                    points[idx_a] += np.clip(score, -10, 10)

        final_scores = []
        for i in range(num_nodes):
            if match_counts[i] > 0:
                avg_val = points[i] / match_counts[i]
                
                if mode == "discrete":
                    # 离散模式：线性压缩 (Linear Compression)
                    # 原始是 0-1 (即0-10分)。
                    # 为了让它柔和，我们可以把 0.0-1.0 映射到 2.0-8.0
                    # 公式: 2 + (win_rate * 6)
                    # 如果你非常想保留0和10的可能性，这里比较难，建议用 weighted 模式
                    win_rate = avg_val
                    # 这种映射让 0%胜率=2分，100%胜率=8分
                    normalized_score = 2.0 + (win_rate * 6.0) 
                    
                else: # mode == "weighted" (推荐)
                    # Sigmoid 映射
                    # avg_val 是平均优势分 (例如 +3.5)
                    # 公式: 10 / (1 + exp(-x/T))，这个公式满足均值为5
                    normalized_score = 10 / (1 + np.exp(-avg_val / temperature))
            else:
                normalized_score = 5.0 # 无数据默认为中位数
            
            # 截断并保留小数
            final_scores.append(float(np.clip(round(normalized_score, 4), 0, 10)))
            
        return final_scores

    async def pairwise_evaluate(self, previous_steps: str, new_nodes: List[Any], 
                              process_criterions: str, case_idx: int = None,
                              score_method: str = 'borda_weighted') -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Args:
            score_method: 
                - 'bradley_terry': 使用 BT 模型
                - 'borda_discrete': 使用胜负平积分 (推荐，更稳健)
                - 'borda_weighted': 使用分差积分 (更能体现差距)
        """
        pairs = self._get_pairs(new_nodes)
        
        # ... (并发执行 pair_comparison 逻辑保持不变) ...
        if pairs:
             comparison_tasks = [
                self._pair_comparison(previous_steps, pair, process_criterions, case_idx)
                for pair in pairs
            ]
             evaluated_pairs = await asyncio.gather(*comparison_tasks)
        else:
             evaluated_pairs = []

        # 根据方法选择计算逻辑
        if score_method == 'borda_weighted':
            # 假设你的 Prompt 定义分数范围是 -5 到 5
            node_adv_scores = self.get_borda_score(evaluated_pairs, len(new_nodes), 
                                                 mode="weighted")
        elif score_method == 'borda_discrete':
            node_adv_scores = self.get_borda_score(evaluated_pairs, len(new_nodes), 
                                                 mode="discrete")
        else:
            # 默认 Bradley-Terry
            node_adv_scores = self._get_bradley_terry_score(evaluated_pairs, len(new_nodes))
            
        return node_adv_scores, evaluated_pairs
