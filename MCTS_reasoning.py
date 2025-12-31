import json
import math
import os
import time
import argparse
from typing import Any, List, Dict, Optional, Tuple
from tqdm import tqdm

from stop_sequences import get_stop_sequences
from LLMs import *
from dataloader import DataLoader
from text_handler import PromptHandler, ResponseHandler, SearchGuideHandler  # 添加 SearchGuideHandler 导入
from evaluator import RewardModel, ProcessRewardModel, PairwiseRewardModel
from log_func import test_out, timestamp, set_global_args
import asyncio

# 全局变量
log_file = None
tree_log_file = None
timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加全局时间戳

class ReasoningNode:
    def __init__(self, state, parent=None, action=None, coordinates=None):
        """
        初始化推理节点
        
        Args:
            state: 当前状态（当前的推理步骤）
            parent: 父节点
            action: 产生该节点的动作（推理步骤）
            coordinates: 节点在树中的坐标
        """
        # 从 Node 类继承的属性
        self.state = state         # 当前状态
        self.parent = parent       # 父节点
        self.children = []         # 孩子节点列表
        self.visits = 0            # 节点被访问的次数
        self.value = 0.0           # 累计奖励（价值）
        
        # ReasoningNode 特有的属性
        self.reasoning_path = []  # 记录推理路径
        self.coordinates = coordinates  # 节点在树中的坐标
        self.response = None  # 存储LLM的响应
        self.is_final = False  # 标记是否为终止节点
        self.full_context = ""  # 新增：完整的上下文（包含所有父节点内容）

        # 新增属性：节点深度
        self.depth = self._calculate_depth()
    
    def _calculate_depth(self) -> int:
        """计算并返回节点深度"""
        depth = 1
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def update_coordinates(self):
        """更新节点的坐标表示"""
        if self.parent is None:  # 根节点
            self.coordinates = "root"
        else:
            # 安全检查：确保当前节点在父节点的children列表中
            try:
                idx = self.parent.children.index(self) + 1
                if self.parent.coordinates == "root":
                    self.coordinates = f"({idx})"
                else:
                    # 去掉父节点坐标的最后一个括号，添加当前节点的索引
                    self.coordinates = f"{self.parent.coordinates[:-1]},{idx})"
            except ValueError:
                # 如果节点不在父节点的children列表中，使用默认坐标
                # 这种情况通常发生在节点刚创建但还未添加到父节点时
                if self.parent.coordinates == "root":
                    self.coordinates = f"({len(self.parent.children) + 1})"
                else:
                    self.coordinates = f"{self.parent.coordinates[:-1]},{len(self.parent.children) + 1})"
    
    def update_full_context(self):
        """
        更新节点的完整上下文
        在节点创建或状态更新后调用
        """
        if self.parent is None:
            # 根节点
            self.full_context = f"Question: {self.state}\n\n"
        else:
            # 非根节点：父节点的上下文 + 当前节点的状态
            self.full_context = f"{self.parent.full_context}Step {len(self.reasoning_path)}: {self.state}\n"
        
        # 更新坐标
        self.update_coordinates()

    def get_full_reasoning_path(self, new_response: str = "") -> str:
        """
        获取完整的推理路径，包括当前响应
        
        Args:
            response: 当前节点的响应（可选）
        Returns:
            完整的推理路径
        """
        if new_response:
            return f"{self.full_context}{new_response}"  # 如果输入了当前节点response，就拼接起来
        else:
            return self.full_context

    def get_previous_steps(self) -> str:
        """
        获取除了当前步骤以外之前的步骤
        """
        if self.parent is None:
            return self.full_context
        return self.parent.full_context


class MCTS_Reasoner:
    def __init__(self, question: str, reward_model: Any, process_reward_model: Any, pairwise_reward_model: Any,
                 prompt_handler: PromptHandler, response_handler: ResponseHandler, search_guide_handler: SearchGuideHandler,
                 llm_client: Any, case_idx: int, dataset_name: str, ground_truth: Any = None,
                 rollout_num: int = 1, exploration_constant: float = 3, 
                 show_runtime: bool = True, verbose: bool = True,
                 max_depth: int = 5, branch_factor: int = 3, branch_factor_init: Any = None,
                 leaf_value_weight: float = 0.2,  # <-- 新增
                 path_value_weight: float = 0.2,   # <-- 新增
                 leaf_confidence_weight: float = 0.6, # <-- 新增
                 rollout_weight: float = 0.3, process_weight: float = 0.4, pairwise_weight: float = 0.3,
                 rollout_sc_weight: float = 0.0, 
                 rollout_confidence_weight: float = 0.0):
        """
        初始化MCTS推理器
        """
        # 自动配置参数
        self.reward_model = reward_model
        self.process_reward_model = process_reward_model
        self.pairwise_reward_model = pairwise_reward_model
        self.prompt_handler = prompt_handler
        self.response_handler = response_handler
        self.llm_client = llm_client
        self.case_idx = case_idx
        self.dataset_name = dataset_name
        self.show_runtime = show_runtime
        self.verbose = verbose
        self.tree_log = {"case_idx": case_idx, "hyperparamters": {}, "nodes": {}, "rollouts": {}}
        self.question = question
        self.ground_truth = ground_truth
        
        # 用户手动配置参数
        self.exploration_constant = exploration_constant
        self.rollout_num = rollout_num
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.branch_factor_init = branch_factor_init if branch_factor_init is not None else branch_factor
        
        # 新增：各种奖励权重参数
        self.rollout_weight = rollout_weight  # 路径奖励权重
        self.process_weight = process_weight  # 过程奖励权重
        self.pairwise_weight = pairwise_weight  # 对比奖励权重
        self.rollout_sc_weight = rollout_sc_weight
        self.rollout_confidence_weight = rollout_confidence_weight

        # 选择答案时候的权重
        self.leaf_value_weight = leaf_value_weight
        self.path_value_weight = path_value_weight
        self.leaf_confidence_weight = leaf_confidence_weight
        
        # 从Search_Guide中加载各种引导信息
        self.search_guide_handler = search_guide_handler
        self.init_expand_guidance = self.search_guide_handler.get_init_expand_guidance()
        self.idea_guidance = self.search_guide_handler.get_idea_guidance()
        self.expand_guidance = self.search_guide_handler.get_expand_guidance()
        self.process_criterions = self.search_guide_handler.get_process_criterions()
        self.pairwise_criterions = self.search_guide_handler.get_pairwise_criterions()
        self.reward_objectives = self.search_guide_handler.get_reward_objectives()
        self.answer_restrictions = self.search_guide_handler.get_answer_restrictions()

        # 添加叶子节点列表维护
        self.leaf_nodes = []

    def check_final_answer(self, response: str) -> bool:
        """
        检查响应是否包含终止符号
        
        Args:
            response: LLM的响应文本
        Returns:
            是否为终止节点
        """
        # 如果包含多个</step>，说明不是终止节点
        if response.count('</step>') > 1:
            return False
        
        # 检查是否包含 \boxed{xxx} 或 </think>
        return r'\boxed{' in response

    async def _init_expand(self, root: ReasoningNode) -> List[ReasoningNode]:
        """
        封装初始扩展逻辑：生成初始prompt，调用LLM，并创建第一层子节点
        """
        # 使用initial_step模板进行第一次展开
        prompt = self.prompt_handler.get_init_prompt(self.question, self.init_expand_guidance)
        
        # 高温度采样使得模型有一些变数
        responses = await self.llm_client.generate(
            prompt, 
            n=self.branch_factor_init, 
            stop_stage="initial_step", 
            seed=None, 
            temperature=0.9
        )
        
        initial_nodes = []
        
        for i, response in enumerate(responses):
            initial_step = self.response_handler.get_expand_step_init(response)
            test_out("initial_step", initial_step, self.case_idx, self.dataset_name)

            # 创建初始子节点
            new_node = ReasoningNode(
                initial_step, 
                parent=root, 
                action=initial_step,
                coordinates=f"({i+1})"
            )
            new_node.reasoning_path = [initial_step]
            new_node.response = response
            new_node.is_final = self.check_final_answer(response)
            
            # 更新完整上下文
            new_node.update_full_context()
            
            # 添加到根节点
            root.children.append(new_node)
            initial_nodes.append(new_node)
            
            # 如果是叶子节点，添加到全局叶子节点列表
            if new_node.is_final:
                self.leaf_nodes.append(new_node)
            
            # 记录节点信息到日志
            self.tree_log["nodes"][new_node.coordinates] = {
                "state": initial_step,
                "response": response,
                "visits": 0,
                "value": 0.0,
                "is_final": new_node.is_final,
            }
            
        return initial_nodes

    async def search(self, num_iterations: int = 20) -> str:
        """
        执行MCTS搜索
        
        Args:
            num_iterations: 迭代次数
        """
        start_time = time.time()
        
        # 1. 创建并初始化根节点
        root = ReasoningNode(self.question, coordinates="root")
        self.root = root
        root.update_full_context()
        
        self.tree_log["nodes"]["root"] = {
            "state": self.question,
            "response": None,
            "visits": 0,
            "value": 0.0,
            "is_final": False,
            "full_context": root.full_context
        }

        # 2. 调用封装好的初始扩展方法，获取初始子节点列表
        initial_nodes = await self._init_expand(root)
        
        # -----------------------------------------------------------
        # 阶段 A: 并行处理初始节点 (Parallel Initial Exploration)
        # -----------------------------------------------------------
        # 初始生成的节点由于visits都为0，在标准MCTS中会被依次选中。
        # 这里直接通过 gather 并行处理，提高效率。
        
        async def process_initial_node(node):
            """内部帮助函数：处理单个初始节点的模拟和评估"""
            if not node.is_final:
                # 扩展并模拟
                new_nodes, new_values = await self._async_expand_and_simulate(node)
                return {
                    "type": "expand",
                    "parent": node, # 注意：这里的parent是expand出来的节点的父节点，即当前的initial_node
                    "new_nodes": new_nodes,
                    "new_values": new_values
                }
            else:
                # 已经是终止节点，直接评估
                reward = await self._final_simulate(node)
                return {
                    "type": "final",
                    "node": node,
                    "reward": reward
                }

        # 创建并行任务
        test_out("search_phase", "Starting parallel execution for initial nodes", self.case_idx, self.dataset_name)
        parallel_tasks = [process_initial_node(node) for node in initial_nodes]
        results = await asyncio.gather(*parallel_tasks)
        
        # 处理并行结果并进行反向传播 (Backpropagation)
        # 注意：反向传播涉及树状态更新，最好在主线程顺序执行以避免竞争（尽管Python有GIL，但逻辑上保持清晰）
        for res in results:
            if res["type"] == "expand":
                # 处理扩展产生的新节点
                for new_node, new_value in zip(res["new_nodes"], res["new_values"]):
                    self._backpropagate(new_node, new_value)
                    if new_node.is_final:
                        self.leaf_nodes.append(new_node)
            elif res["type"] == "final":
                # 处理直接终止的节点
                self._backpropagate(res["node"], res["reward"])

        # 计算剩余需要执行的迭代次数
        # 已经在并行阶段消耗了 len(initial_nodes) 次“探索机会”
        executed_iterations = len(initial_nodes)
        remaining_iterations = max(0, num_iterations - executed_iterations)

        # -----------------------------------------------------------
        # 阶段 B: 剩余迭代串行搜索 (Sequential MCTS Loop)
        # -----------------------------------------------------------
        test_out("search_phase", f"Starting sequential search for remaining {remaining_iterations} iterations", self.case_idx, self.dataset_name)
        
        for i in range(remaining_iterations):
            real_iter_idx = executed_iterations + i
            test_out("num_iterations:"+str(real_iter_idx), "", self.case_idx, self.dataset_name)
            
            # Select
            node = self._select(root)
            
            # Expand & Simulate & Backpropagate
            if not node.is_final:
                new_nodes, new_values = await self._async_expand_and_simulate(node)
                for new_node, new_value in zip(new_nodes, new_values):
                    self._backpropagate(new_node, new_value)
                    if new_node.is_final:
                        self.leaf_nodes.append(new_node)
            else:
                reward = await self._final_simulate(node)
                self._backpropagate(node, reward)

        # -----------------------------------------------------------
        # 结果汇总
        # -----------------------------------------------------------
        # 选择最优路径
        best_path, best_solution, answer_frequency = self._pick_final_answer(root)
        
        # [新增] 在调用output_best_answer前，先从best_solution中直接抽取答案
        extracted_answer = self._extract_answer_from_response(best_solution)

        # 生成最终答案
        final_answer = await self._output_best_answer(best_solution)

        self.tree_log['final_results'] = {
            "ground_truth": self.ground_truth,
            "extracted_answer": extracted_answer,  # [新增] 写入抽取出的答案
            "final_answer": final_answer,
            "answer_frequency": answer_frequency,
            "best_path": best_path,
        }
        self.save_tree_log()

        end_time = time.time()
        runtime = end_time - start_time
        
        if self.show_runtime:
            output = f"\n{'*'*60}\nRuntime: {runtime:.2f} seconds\n{'*'*60}"
            test_out("Runtime", output, self.case_idx, self.dataset_name)
            
        return best_path, final_answer

    def _select(self, node: ReasoningNode) -> ReasoningNode:
        """使用UCT公式选择节点，但不选择终止节点进行扩展"""
        selection_trace = []
        while node.children:
            parent_coords = node.coordinates
            test_out("selection_at_node", f"parent={parent_coords}, visits={node.visits}, children={len(node.children)}", self.case_idx, self.dataset_name)

            best_score = float('-inf')
            best_child = None

            for child in node.children:
                if child.visits == 0:
                    exploit_score = None
                    explore_score = None
                    score = float('inf')
                else:
                    # exploit_score = ((child.value / child.visits) - node.child_mean) / (node.child_diff if node.child_diff != 0 else 1.0)
                    exploit_score = child.value / child.visits  # 直接使用Q值作为 exploit_score
                    explore_score = math.sqrt(math.log(max(node.visits, 1)) / child.visits)
                    score = exploit_score + self.exploration_constant * explore_score

                child_coords = child.coordinates
                exp_str = f"{exploit_score:.6f}" if exploit_score is not None else "NA"
                explore_str = f"{explore_score:.6f}" if explore_score is not None else "NA"
                test_out(
                    "UCT",
                    f"{parent_coords} -> {child_coords} | visits={child.visits}, value={child.value:.6f}, exploit={exp_str}, explore={explore_str}, score={score}",
                    self.case_idx,
                    self.dataset_name
                )

                if score > best_score:
                    best_score = score
                    best_child = child

            test_out("selected_child", f"{parent_coords} -> {best_child.coordinates} | best_score={best_score}", self.case_idx, self.dataset_name)
            node = best_child
            selection_trace.append(node.coordinates)

            # 如果遇到终止节点，停止选择
            if node.is_final:
                break

        if selection_trace:
            test_out("selection_path", " -> ".join(selection_trace), self.case_idx, self.dataset_name)
        return node

    async def _async_expand_and_simulate(self, chosen_node: ReasoningNode) -> Tuple[List[ReasoningNode], List[float]]:
        """
        扩展并模拟节点（带详细交互日志版）。
        [Mod] 集成了 SC (Self-Consistency) 和 Confidence (置信度) 分数计算。
        """
        # 0. 边界条件：最大深度检查
        if chosen_node.depth >= self.max_depth:
            final_nodes = await self._max_depth_expand(chosen_node)
            final_node = final_nodes[0]
            reward = await self._final_simulate(final_node)
            return final_nodes, [reward]

        # 1. 生成核心想法
        context = chosen_node.full_context
        core_ideas = await self._generate_diverse_core_ideas(context, self.branch_factor)
        test_out("core_ideas", core_ideas, self.case_idx, self.dataset_name)

        # ------------------------------------------------------------------
        # 定义局部任务流 (Rollout Flow)
        # ------------------------------------------------------------------
        async def _run_rollout_flow(target_node: ReasoningNode) -> Tuple[float, float, float]:
            """
            执行 Rollout 流程。
            Returns:
                Tuple[float, float, float]: (avg_rollout_score, consistency_score, avg_confidence_score)
            """
            # [Mod] 检查 Rollout、SC、Confidence 三个权重，如果全为0则跳过
            if (self.rollout_weight == 0 and 
                self.rollout_sc_weight == 0 and 
                self.rollout_confidence_weight == 0):
                return 0.0, 0.0, 0.0
            
            paths = []
            avg_path_con = 0.0

            # Step A: 生成路径 (根据是否需要置信度分支处理)
            if target_node.is_final:
                final_path_str = target_node.get_full_reasoning_path()
                # 这样后续的遍历循环就会生成 rollout_num 个评估任务
                paths = [final_path_str] * 3
                # 这里暂时给 3.0 作为 "无信息时的默认值"
                avg_path_con = 3.0 
            else:
                if self.rollout_confidence_weight != 0:
                    paths, path_con_scores = await self._rollout_paths(target_node, confidence_tag=True)
                    
                    # [Mod] path_con_scores 列表里通常是 log probabilities (负数)
                    if path_con_scores:
                        # 1. 对每个 log_prob 取 exp 变成概率 (0-1)
                        # 2. 乘以 10 映射到 (0-10)
                        # 3. 取平均
                        probs_0_10 = [math.exp(score) * 10.0 for score in path_con_scores]
                        avg_path_con = sum(probs_0_10) / len(probs_0_10)
                    else:
                        avg_path_con = 3.0 # 或者 5.0
                else:
                    paths = await self._rollout_paths(target_node)
            
            if not paths: 
                return 0.0, 0.0, 0.0

            # [Mod] 计算 Self-Consistency Score (SC)
            sc_score = self._get_avg_sc_score(paths)
            #test_out("rollout_sc_score", f"{sc_score:.6f}", self.case_idx, self.dataset_name)

            # 如果 Rollout 评估权重为 0，就不需要跑昂贵的评估模型了，返回 (0, sc, con)
            if self.rollout_weight == 0:
                return 0.0, sc_score, avg_path_con

            # Step B: 并行评估路径 (原逻辑)
            eval_tasks = [
                self._rollout_evaluation(target_node, path, i) 
                for i, path in enumerate(paths)
            ]
            results = await asyncio.gather(*eval_tasks)
            
            # Step C: 记录日志并计算平均分
            scores = []
            for i, (score, r_prompt, r_response) in enumerate(results):
                scores.append(score)
                #test_out(f"rollout_{i}_prompt", r_prompt, self.case_idx, self.dataset_name)
                test_out(f"Node {target_node.coordinates}: rollout_{i}_reward_res", r_response, self.case_idx, self.dataset_name)
                test_out("rollout_reward_score", f"rollout_{i}: {score:.6f}", self.case_idx, self.dataset_name)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # [Mod] 返回三元组：平均分、一致性分数、置信度分数
            return avg_score, sc_score, avg_path_con

        # ------------------------------------------------------------------
        # 定义局部任务流 (Process Eval)
        # ------------------------------------------------------------------
        async def _run_process_eval(target_node: ReasoningNode) -> float:
            if self.process_weight == 0:
                return 0.0
            
            p_reward, p_prompt, p_response = await self._process_evaluation(
                target_node, target_node.full_context, target_node.state
            )
            
            #test_out("process_prompt", p_prompt, self.case_idx, self.dataset_name)
            test_out(f"Node {target_node.coordinates}: process_eval", p_response, self.case_idx, self.dataset_name)
            test_out("process_score", f"{p_reward:.6f}", self.case_idx, self.dataset_name)
            return p_reward

        # ------------------------------------------------------------------
        # 阶段 1: 扩展并启动局部任务
        # ------------------------------------------------------------------
        async def _expand_and_schedule(core_idea: str):
            new_node = await self._idea_expand(chosen_node, core_idea)
            p_task = asyncio.create_task(_run_process_eval(new_node))
            r_task = asyncio.create_task(_run_rollout_flow(new_node))
            return new_node, p_task, r_task

        expand_results = await asyncio.gather(*[_expand_and_schedule(idea) for idea in core_ideas])
        
        if not expand_results:
            return [], []

        new_nodes = [r[0] for r in expand_results]
        process_tasks = [r[1] for r in expand_results]
        rollout_tasks = [r[2] for r in expand_results]

        # ------------------------------------------------------------------
        # 阶段 2: 启动全局任务 (Pairwise)
        # ------------------------------------------------------------------
        pairwise_task = None
        if self.pairwise_weight != 0 and len(new_nodes) > 1:
            pairwise_task = asyncio.create_task(self._pairwise_evaluation(new_nodes))

        # ------------------------------------------------------------------
        # 阶段 3: 最终汇聚
        # ------------------------------------------------------------------
        all_wait_tasks = process_tasks + rollout_tasks
        if pairwise_task:
            all_wait_tasks.append(pairwise_task)
        
        if all_wait_tasks:
            await asyncio.gather(*all_wait_tasks)

        # ------------------------------------------------------------------
        # 阶段 4: 提取结果并计算最终奖励
        # ------------------------------------------------------------------
        final_values = []
        
        pairwise_scores = [0.0] * len(new_nodes)
        if pairwise_task:
            # 获取 pairwise_evaluate 返回的元组 (scores, details_list)
            pw_result = pairwise_task.result()
            pairwise_scores = pw_result[0]  # List[float]: 每个节点的最终得分
            evaluated_pairs = pw_result[1]  # List[Dict]: 每一对比较的详细信息
            
            # === [新增日志逻辑] 记录 Pairwise 详细过程 ===
            
            # 1. 记录每一对比较的详情 (Coordinates, Score, Response)
            for i, pair in enumerate(evaluated_pairs):
                # 获取比较对中两个节点的索引 (请确保 _get_pairs 方法中存入了 idx_a, idx_b)
                # 如果你的 _get_pairs 使用的是 index_1/index_2，请相应修改这里的 key
                idx_a = pair.get('idx_a') 
                idx_b = pair.get('idx_b')
                
                # 安全检查：确保索引存在且未越界
                if idx_a is not None and idx_b is not None and idx_a < len(new_nodes) and idx_b < len(new_nodes):
                    coord_a = new_nodes[idx_a].coordinates
                    coord_b = new_nodes[idx_b].coordinates
                    
                    log_content = (
                        f"Comparison Pair {i+1}:\n"
                        f"  Node A ({coord_a}) vs Node B ({coord_b})\n"
                        f"  Extracted Score: {pair.get('score')}\n"
                        f"  LLM Response: {pair.get('response')}"
                    )
                    
                    # 使用 test_out 写入日志
                    test_out(f"pairwise_detail_{coord_a}_vs_{coord_b}", log_content, self.case_idx, self.dataset_name)
            
            # 2. 记录最终聚合分数 (Node Adv Scores)
            # 构建一个 {坐标: 分数} 的映射字符串
            scores_map = {node.coordinates: round(score, 4) for node, score in zip(new_nodes, pairwise_scores)}
            test_out("pairwise_final_scores", f"Aggregated Scores: {scores_map}", self.case_idx, self.dataset_name)
            
            # === [日志逻辑结束] ===

        for i, node in enumerate(new_nodes):
            p_score = process_tasks[i].result()
            
            # [Mod] 解包 rollout 任务的结果 (r_score, sc_score, con_score)
            r_score, sc_score, con_score = rollout_tasks[i].result()
            
            pw_score = pairwise_scores[i]
            
            # [Mod] 计算 Final Reward，加入 Confidence 权重
            final_reward = (
                self.process_weight * p_score +
                self.rollout_weight * r_score +
                self.pairwise_weight * pw_score +
                self.rollout_sc_weight * sc_score +
                self.rollout_confidence_weight * con_score 
            )
            
            self.tree_log["nodes"][node.coordinates]["value"] = final_reward

            # [新增] 记录详细的信号分量和权重
            self.tree_log["nodes"][node.coordinates]["signals"] = {
                "process": {"score": p_score, "weight": self.process_weight},
                "rollout": {"score": r_score, "weight": self.rollout_weight},
                "pairwise": {"score": pw_score, "weight": self.pairwise_weight},
                "self_consistency": {"score": sc_score, "weight": self.rollout_sc_weight},
                "confidence": {"score": con_score, "weight": self.rollout_confidence_weight}
            }
            
            # [Mod] 更新日志显示 SC 和 Confidence 分数
            node_rewards_log = (f"final={final_reward:.6f} "
                                f"(p={p_score:.6f}, r={r_score:.6f}, pw={pw_score:.6f}, "
                                f"sc={sc_score:.6f}, con={con_score:.6f})")
            test_out("final_reward", f"Node {node.coordinates}: {node_rewards_log}", self.case_idx, self.dataset_name)
            
            final_values.append(final_reward)

        return new_nodes, final_values
    
    async def _generate_diverse_core_ideas(self, context: str, num_ideas: int) -> list:
        """
        生成多个不同的核心想法，用于扩展节点
        
        Args:
            context: 当前的推理上下文
            num_ideas: 需要生成的核心想法数量
        Returns:
            核心想法列表
        """
        prompt = self.prompt_handler.get_diverse_ideas_prompt(
            previous_steps=context, 
            num_ideas=num_ideas,
            idea_guidance=self.idea_guidance
        )
        #test_out("diverse_ideas_prompt", prompt, self.case_idx, self.dataset_name)

        # 添加多样化想法生成阶段的停止序列
        response = (await self.llm_client.generate(prompt, stop_stage='diverse_ideas'))[0]
        #test_out("diverse_ideas_response", response, self.case_idx, self.dataset_name)

        # 解析核心想法，调用ResponseHandler的方法
        core_ideas = self.response_handler.parse_diverse_ideas(response, num_ideas)
        return core_ideas

    async def _idea_expand(self, parent_node: ReasoningNode, core_idea: str):
        prompt = self.prompt_handler.get_expand_prompt(
            previous_steps=parent_node.full_context, 
            core_instruction=core_idea,
            expand_guidance=self.expand_guidance
        )
        #test_out("expand_prompt", prompt, self.case_idx, self.dataset_name)

        # 调用LLM生成下一步推理
        expand_response = (await self.llm_client.generate(prompt, 
                                            stop_stage='expand', 
                                            skip_special_tokens=False))[0]
        #test_out("expand_response", expand_response, self.case_idx, self.dataset_name)
        
        # 处理LLM的响应，提取下一步推理内容
        new_step = self.response_handler.get_expand_step(core_idea, expand_response)
        #test_out("new_step", new_step, self.case_idx, self.dataset_name)

        # 创建新的推理节点
        new_node = ReasoningNode(
            new_step,
            parent=parent_node,
            action=new_step,
            coordinates="temp"  # 先设置临时坐标
        )
        new_node.reasoning_path = parent_node.reasoning_path + [new_step]
        new_node.response = expand_response
        new_node.is_final = self.check_final_answer(expand_response)

        # 先将节点添加到父节点的children列表中
        parent_node.children.append(new_node)

        # 然后更新坐标（此时节点已经在父节点的children列表中）和节点路径
        new_node.update_coordinates()
        new_node.update_full_context()
        
        # 将节点信息记录到日志中
        self.tree_log["nodes"][new_node.coordinates] = {
            "state": new_step,
            "core_idea": core_idea,
            "full_step": expand_response,
            "visits": 0,
            "value": 0.0,
            "is_final": new_node.is_final,
        }
        return new_node
    
    async def _rollout_paths(self, new_node: ReasoningNode, confidence_tag=False):
        """
        用于对单个的new_node进行rollout后，返回多条轨迹full_reasoning_paths
        适配说明：基于 generate 方法始终返回 List 的逻辑进行简化
        """
        # 1. --- 计算 Rollout 数量 (保持不变) ---
        current_depth = new_node.depth
        
        if current_depth <= self.max_depth / 3:
            actual_rollout_num = self.rollout_num
            depth_ratio = 1.0
        elif current_depth <= 2 * self.max_depth / 3:
            actual_rollout_num = int(self.rollout_num * 2 / 3)
            depth_ratio = 2/3
        else:
            actual_rollout_num = int(self.rollout_num * 1 / 3)
            depth_ratio = 1/3
            
        actual_rollout_num = max(1, actual_rollout_num)
        
        #test_out("dynamic_rollout", f"Depth: {current_depth}/{self.max_depth}, Ratio: {depth_ratio}, Rollouts: {actual_rollout_num}/{self.rollout_num}", self.case_idx, self.dataset_name)
        #test_out("previous steps", new_node.full_context, self.case_idx, self.dataset_name)
        
        rollout_prompt = self.prompt_handler.get_rollout_prompt(new_node.full_context)
        #test_out("rollout_prompt:", rollout_prompt, self.case_idx, self.dataset_name)
        
        # 2. --- 调用 Generate (逻辑简化) ---
        # 无论 n=1 还是 n>1，返回的数据结构保持一致
        llm_output = await self.llm_client.generate(
            rollout_prompt, 
            stop_stage='rollout', 
            n=actual_rollout_num, 
            max_tokens=1024, 
            seed=None,
            confidence_tag=confidence_tag
        )

        # 3. --- 处理返回值 (解包逻辑) ---
        rollout_responses = []
        confidence_list = []

        if confidence_tag:
            # Generate 返回 ([text1, text2...], [conf1, conf2...])
            rollout_responses, confidence_list = llm_output
        else:
            # Generate 返回 [text1, text2...]
            rollout_responses = llm_output
            # 此时不需要处理 confidence_list

        # 4. --- 构建推理路径 (统一循环处理) ---
        full_reasoning_paths = []
        
        # 因为 rollout_responses 始终是列表，直接遍历即可，无需担心 string 类型报错
        for rollout_idx, rollout_response in enumerate(rollout_responses):
            # 获取完整路径
            full_reasoning_path = new_node.get_full_reasoning_path(rollout_response)
            full_reasoning_paths.append(full_reasoning_path)
            
            #test_out("rollout_response:", rollout_response, self.case_idx, self.dataset_name)
        
        # 5. --- 返回结果 ---
        if confidence_tag:
            return full_reasoning_paths, confidence_list
        else:
            return full_reasoning_paths

    async def _rollout_evaluation(self, node: ReasoningNode, full_reasoning_path: str, rollout_idx: int):
        rollout_key = f"{node.coordinates}_rollout_{rollout_idx+1}"
        rollout_reward, reward_prompt, reward_response = await self.reward_model.evaluate(
            full_reasoning_path, 
            self.reward_objectives, 
            True,
            self.case_idx
        )
        self.tree_log["rollouts"][rollout_key] = {
            "full_reasoning_path": full_reasoning_path,
            "rollout_reward": rollout_reward,
            "reward_response": reward_response,
        }
        return rollout_reward, reward_prompt, reward_response

    async def _process_evaluation(self, node: ReasoningNode, previous_steps: str, current_step: str):
        process_reward, process_prompt, process_response = await self.process_reward_model.evaluate(
            previous_steps, 
            current_step, 
            self.process_criterions, 
            True,
            self.case_idx
        )
        return process_reward, process_prompt, process_response

    async def _pairwise_evaluation(self, new_nodes: List[ReasoningNode]) -> List[float]:
        """
            异步执行pairwise评估
        """
        compared_rewards, pair_details = await self.pairwise_reward_model.pairwise_evaluate(
            new_nodes[0].get_previous_steps(),
            new_nodes,
            self.pairwise_criterions,
            self.case_idx
        )
        #print("compared_rewards:", compared_rewards)
        return compared_rewards, pair_details

    async def _max_depth_expand(self, node: ReasoningNode) -> List[ReasoningNode]:
        """
        当达到最大深度时，执行rollout并创建终止节点
        
        Args:
            node: 达到最大深度的节点
        Returns:
            包含一个终止节点的列表
        """
        test_out("max_depth_reached", f"Current depth: {node.depth}, max depth: {self.max_depth}", 
                self.case_idx, self.dataset_name)
        
        # 使用节点的full_context
        context = node.full_context
        
        # 执行rollout
        prompt = self.prompt_handler.get_rollout_prompt(context)
        test_out("rollout_prompt_at_max_depth", prompt, self.case_idx, self.dataset_name)
        
        response = (await self.llm_client.generate(prompt, stop_stage='rollout'))[0]
        test_out("rollout_response_at_max_depth", response, self.case_idx, self.dataset_name)
        
        # 创建终止节点
        new_node = ReasoningNode(
            response,
            parent=node,
            action=response,
            coordinates=f"{node.coordinates[:-1]},1)"
        )
        new_node.reasoning_path = node.reasoning_path + [response]
        new_node.response = response
        new_node.is_final = True  # 标记为终止节点
        
        # 更新完整上下文
        new_node.update_full_context()
        
        node.children.append(new_node)
        
        # 记录节点信息到日志
        self.tree_log["nodes"][new_node.coordinates] = {
            "state": response,
            "response": response,
            "visits": 0,
            "value": 0.0,
            "is_final": True,
            "max_depth_reached": True,
        }
        
        return [new_node]

    async def _final_simulate(self, node: ReasoningNode) -> float:
        """
        对终止节点进行最终评估，直接使用当前节点的推理路径进行评估
        
        Args:
            node: 终止节点
        Returns:
            最终奖励值
        """
        # 直接使用当前节点的推理路径进行评估
        full_reasoning_path = node.get_full_reasoning_path()

        # 直接评估终止节点，不需要rollout
        final_reward, reward_prompt, reward_response = await self.reward_model.evaluate(
            full_reasoning_path, 
            self.reward_objectives, 
            True,
            self.case_idx
        )
        # 终止节点：统一在此输出 prompt/response/score
        #test_out("terminal_reward_prompt", reward_prompt, self.case_idx, self.dataset_name)
        test_out("terminal_reward_response", reward_response, self.case_idx, self.dataset_name)
        test_out("terminal_reward", f"{final_reward:.6f}", self.case_idx, self.dataset_name)
        
        return final_reward

    def _backpropagate(self, node: ReasoningNode, reward: float):
        """反向传播更新节点统计信息"""
        while node is not None:
            node.visits += 1
            node.value += reward
            # 更新日志中的节点信息
            coordinates = node.coordinates if node.coordinates != "root" else "root"
            if coordinates in self.tree_log["nodes"]:
                self.tree_log["nodes"][coordinates]["visits"] = node.visits
                self.tree_log["nodes"][coordinates]["value"] = node.value
            node = node.parent

    def _pick_final_answer(self, root: ReasoningNode) -> List[str]:
        """
        从所有叶子节点中选择最佳答案路径
        """
        # [新增] 初始化 leaf_nodes 日志容器
        self.tree_log["leaf_nodes"] = {}

        # 0. 异常处理：如果没有叶子节点
        if not self.leaf_nodes:
            best_child = max(root.children, key=lambda c: c.visits)
            test_out("no_leaf_nodes", f"Using best child: {best_child.coordinates}", self.case_idx, self.dataset_name)
            return best_child.reasoning_path, best_child.full_context, {}
        
        # 1. 准备数据：提取所有叶子节点的 Response
        leaf_solution_list = [leaf.response for leaf in self.leaf_nodes]
        total_leaf_count = len(leaf_solution_list)
        
        # 2. 计算频率：基于 Response 列表提取答案并统计
        answer_frequency = self._calculate_answer_frequency(leaf_solution_list)
        
        # [日志] 仅保留汇总信息
        test_out("answer_stats", f"Leaves: {total_leaf_count}, Distribution: {answer_frequency}", self.case_idx, self.dataset_name)
        
        best_leaf = None
        best_weighted_value = float('-inf')
        
        # 遍历叶子节点寻找最佳路径
        for leaf in self.leaf_nodes:
            # --- A. 计算路径平均价值 (Path Mean Value) ---
            path_values = []
            current_node = leaf
            
            # 回溯路径
            path_nodes = []
            while current_node is not None:
                path_nodes.append(current_node)
                current_node = current_node.parent
            path_nodes.reverse() # 调整为 根 -> 叶 顺序
            
            for node in path_nodes:
                if node.visits > 0:
                    path_values.append(node.value / node.visits)
                else:
                    path_values.append(0.0)
            
            path_mean_value = sum(path_values) / len(path_values) if path_values else 0
            
            # --- B. 计算叶子节点价值 (Leaf Value) ---
            leaf_value = leaf.value / leaf.visits if leaf.visits > 0 else 0
            
            # --- C. 计算置信度 (Confidence Score) ---
            leaf_answer = self._extract_answer_from_response(leaf.response)
            confidence_prop = self._get_answer_likelihood(leaf_answer, answer_frequency, total_leaf_count)
            confidence_score = confidence_prop * 10
            
            # 新逻辑: 直接使用三个权重进行加权求和
            weighted_value = (
                self.path_value_weight * path_mean_value + 
                self.leaf_value_weight * leaf_value + 
                self.leaf_confidence_weight * confidence_score
            )
            
            # [新增] 将该叶子节点的详细信息写入 tree_log
            self.tree_log["leaf_nodes"][leaf.coordinates] = {
                "extracted_answer": leaf_answer,
                "path_mean_value": {"score": path_mean_value, "weight": self.path_value_weight},
                "leaf_value": {"score": leaf_value, "weight": self.leaf_value_weight},
                "confidence_score": {"score": confidence_score, "weight": self.leaf_confidence_weight},
                "weighted_value": weighted_value,
                "full_path": leaf.full_context  # 记录完整路径
            }

            # 更新最佳节点
            if weighted_value > best_weighted_value:
                best_weighted_value = weighted_value
                best_leaf = leaf
                # [日志] 仅在发现更好的节点时输出简要信息
                test_out("new_best_found", f"Leaf {leaf.coordinates} best so far (score: {weighted_value:.4f}, ans: '{leaf_answer}')", self.case_idx, self.dataset_name)
        
        # 构造返回值
        best_path = best_leaf.reasoning_path if best_leaf else []
        best_solution = best_leaf.full_context if best_leaf else ""
        best_coords = best_leaf.coordinates if best_leaf else "None"
        
        # [日志] 最终结果
        test_out("final_selection", f"Selected {best_coords}, Score: {best_weighted_value:.4f}", self.case_idx, self.dataset_name)
        
        return best_path, best_solution, answer_frequency

    def _extract_answer_from_response(self, response: str) -> str:
        """从响应中提取最终答案"""
        # 1. 尝试提取 \boxed{} (保持原样)
        boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)\}', response)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # 2. 文本模式匹配
        answer_patterns = [
            r'答案是[：:]?\s*([^\n]+)',
            r'答案为[：:]?\s*([^\n]+)',
            r'answer is[：:]?\s*([^\n]+)',
            r'the answer is[：:]?\s*([^\n]+)',
            r'final answer[：:]?\s*([^\n]+)',
            r'最终答案[：:]?\s*([^\n]+)'
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip()
                
                # --- 新增补丁开始 ---
                # 如果捕获的内容像 "to the question is \boxed{...}"
                # 我们再次检查其中是否包含 boxed，如果有，强行提取 boxed
                if "\\boxed" in candidate:
                    sub_boxed = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', candidate)
                    if sub_boxed:
                        return sub_boxed[-1].strip()
                
                # 去除常见的连接词前缀 (如果不含boxed，可能是纯文本答案)
                # 比如匹配到 "to the question is 5"，去除 "to the question is "
                candidate = re.sub(r'^(to the question is|is|:)\s*', '', candidate, flags=re.IGNORECASE).strip()
                # --- 新增补丁结束 ---
                
                return candidate
        
        return ""

    def _calculate_answer_frequency(self, leaf_solutions: List[str]) -> dict:
        """
        计算答案列表中的出现频率
        Args:
            leaf_solutions: 所有叶子节点的响应字符串列表
        Returns:
            answer_counts: 答案与其出现次数的字典
        """
        answer_counts = {}
        
        for response in leaf_solutions:
            answer = self._extract_answer_from_response(response)
            if answer:  # 只统计非空答案
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        return answer_counts

    def _get_answer_likelihood(self, answer: str, answer_frequency: dict, total_count: int) -> float:
        """
        根据答案频率字典计算特定答案的置信度值（占比）
        
        Args:
            answer: 提取出的答案字符串
            answer_frequency: 答案频率字典
            total_count: 总样本数（通常为叶子节点总数）
            
        Returns:
            float: 该答案的占比 (0.0 - 1.0)
        """
        if not answer or total_count == 0:
            return 0.0
            
        count = answer_frequency.get(answer, 0)
        return count / total_count

    def _get_avg_sc_score(self, solution_list: List[str]) -> float:
        """
        计算一组解的平均 Self-Consistency (SC) 分数。
        
        逻辑流程:
        1. 利用 _calculate_answer_frequency 得到所有答案的频率分布。
        2. 遍历 solution_list 中的每一条回复：
           - 提取其答案。
           - 查找该答案的置信度 (Confidence)。
        3. 计算所有回复置信度的平均值。
        
        Args:
            solution_list: 模型生成的回复列表
            
        Returns:
            float: 平均置信度分数 (0.0 - 1.0)
        """
        # print("calculating rollout self consistency")
        if not solution_list:
            return 0.0
        
        if len(solution_list)<2:
            return 10 / self.rollout_num

        # 1. 获取答案频率字典
        answer_frequency = self._calculate_answer_frequency(solution_list)
        total_count = len(solution_list)
        total_likelihood_score = 0.0
        
        # 2. 计算
        for response in solution_list:
            answer = self._extract_answer_from_response(response)
            likelihood = self._get_answer_likelihood(answer, answer_frequency, total_count)
            total_likelihood_score += likelihood
            
        # 3. 返回平均值 [Mod] 乘以 10 映射到 0-10 分
        avg_prob = total_likelihood_score / total_count
        return avg_prob * 10.0

    async def _output_best_answer(self, best_solution: str) -> str:
        """
        使用最佳推理路径生成最终答案
        
        Args:
            best_solution: 最佳推理路径字符串
        Returns:
            生成的最终答案字符串
        """
        question = self.question
        
        # 使用prompt_handler获取best_answer模板
        prompt = self.prompt_handler.get_best_answer_prompt(question, best_solution, self.answer_restrictions)
        #test_out("best_answer_prompt", prompt, self.case_idx, self.dataset_name)
        
        # 调用LLM生成最终答案
        final_answer_response = (await self.llm_client.generate(
            prompt, 
            stop_stage='best_answer', 
            skip_special_tokens=False
        ))[0]
        test_out("final_answer_response", "The final answer is: \\boxed{" + final_answer_response, self.case_idx, self.dataset_name)

        final_answer = self.response_handler.parse_best_answer(final_answer_response)
        
        test_out("final_answer", final_answer, self.case_idx, self.dataset_name)
        
        return final_answer

    def save_tree_log(self):
        """保存树的日志到文件"""
        log_dir = os.path.join("logs", f"{self.dataset_name}_{timestamp}_{args.case_start}_{args.case_end}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 构建文件名，每个case只使用一个文件
        filename = os.path.join(log_dir, f"case_{self.case_idx:04d}.json")
        
        # 保存当前树的状态
        tree_data = {
            "case_idx": self.case_idx,
            "dataset_name": self.dataset_name,
            "question": self.question,
            "final_results": self.tree_log["final_results"],
            "hyperparameters": {
                "exploration_constant": self.exploration_constant,
                "rollout_weight": self.rollout_weight,
                "process_weight": self.process_weight,
                "pairwise_weight": self.pairwise_weight,
                "rollout_sc_weight": self.rollout_sc_weight, # <-- 新增
                "rollout_confidence_weight": self.rollout_confidence_weight, # <-- 新增
                "rollout_num": self.rollout_num,
                "max_depth": self.max_depth,
                "branch_factor": self.branch_factor,
                "branch_factor_init": self.branch_factor_init,
                "leaf_value_weight": self.leaf_value_weight, # <-- 新增
                "path_value_weight": self.path_value_weight, # <-- 新增
                "leaf_confidence_weight": self.leaf_confidence_weight, # <-- 新增
            },
            "leaf_nodes": self.tree_log["leaf_nodes"], # <-- 新增
            "nodes": self.tree_log["nodes"],
            "rollouts": self.tree_log["rollouts"],
        }
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)

async def main():
    global args

    # 记录脚本开始时间
    script_start_time = time.time()
    script_start_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 初始化summary记录
    summary_data = {
        "start_time": script_start_datetime,
        "start_timestamp": script_start_time,
        "cases": [],
        "parameters": {}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Olympiadphysics", help="Name of the dataset") 
    parser.add_argument("--model", type=str, default="../models/Qwen3-8B", 
                       help="模型名称或路径。对于aihub模式，使用模型名称（如qwen-max）；对于transformer和vllm模式，使用模型路径")
    parser.add_argument("--case_start", type=int, default=2, help="Start index of cases")
    parser.add_argument("--case_end", type=int, default=2, help="End index of cases")
    parser.add_argument("--num_iterations", type=int, default=15, help="Number of MCTS iterations")
    parser.add_argument("--exploration_constant", type=float, default=1.41, help="Exploration constant for UCT formula")
    parser.add_argument("--branch_factor", type=int, default=3, 
                       help="Branch factor for MCTS expansion")
    parser.add_argument("--branch_factor_init", type=int, default=4,
                       help="Branch factor for initial step generation")
    parser.add_argument("--rollout_num", type=int, default=3, help="Number of rollouts")
    parser.add_argument("--show_runtime", action="store_true", default=True, help="是否显示运行时间")
    parser.add_argument("--show_runtime_case", action="store_true", default=True, help="是否显示每个case的用时")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Whether to print detailed information")
    parser.add_argument("--run_mode", type=str, default="async_vllm", 
                       choices=["aihub", "transformer", "vllm", "async_vllm", "debug"], 
                       help="LLM运行模式: aihub(AiHubMix), transformer(Transformers), vllm(vLLM), debug(调试模式)")
    parser.add_argument("--max_depth", type=int, default=10,
                       help="Maximum depth of the MCTS tree")
    parser.add_argument("--leaf_value_weight", type=float, default=0.25,
                        help="Weight for the value of the leaf node itself")
    parser.add_argument("--path_value_weight", type=float, default=0.5,
                        help="Weight for the mean value of the reasoning path")
    parser.add_argument("--leaf_confidence_weight", type=float, default=0.25,
                        help="Weight for the confidence score based on answer frequency")
    parser.add_argument("--pairwise_weight", type=float, default=0.5,
                       help="对比奖励信号的权重")
    parser.add_argument("--process_weight", type=float, default=0.2,
                       help="过程评估信号的权重")
    parser.add_argument("--rollout_weight", type=float, default=0.3,
                       help="rollout模拟评估信号的权重")
    parser.add_argument("--rollout_sc_weight", type=float, default=0.0,
                        help="Weight for self-consistency score in rollout")
    parser.add_argument("--rollout_confidence_weight", type=float, default=0.0,
                        help="Weight for confidence score in rollout")
    args = parser.parse_args()
    
    # 设置全局args引用
    set_global_args(args)

    
    # 记录运行参数
    summary_data["parameters"] = {
        "dataset_name": args.dataset_name,
        "model": args.model,
        "case_start": args.case_start,
        "case_end": args.case_end,
        "num_iterations": args.num_iterations,
        "exploration_constant": args.exploration_constant,
        "branch_factor": args.branch_factor,
        "branch_factor_init": args.branch_factor_init,
        "rollout_num": args.rollout_num,
        "run_mode": args.run_mode,
        "max_depth": args.max_depth,
        "leaf_value_weight": args.leaf_value_weight, # <-- 新增
        "path_value_weight": args.path_value_weight, # <-- 新增
        "leaf_confidence_weight": args.leaf_confidence_weight, # <-- 新增
        "pairwise_weight": args.pairwise_weight,
        "process_weight": args.process_weight,
        "rollout_weight": args.rollout_weight,
        "rollout_sc_weight": args.rollout_sc_weight, # <-- 新增
        "rollout_confidence_weight": args.rollout_confidence_weight, # <-- 新增
    }
    
    # 统计模型加载时间
    model_load_start = time.time()
    # 根据运行模式初始化LLM
    if args.run_mode == "aihub":
        llm = use_AiHubMix(args.model)
    elif args.run_mode == "transformer":
        llm = use_transformers(args.model)
    elif args.run_mode == "vllm":
        llm = use_vLLM(args.model)
    elif args.run_mode == "async_vllm":
        llm = use_async_vLLM(args.model)
        await llm.initialize_tokenizer()
    elif args.run_mode == "debug":
        llm = use_debug(args.model)
    else:
        raise ValueError(f"Unsupported run mode: {args.run_mode}")
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start
    if args.show_runtime_case:
        print(f"Model loaded in {model_load_time:.2f} seconds.")
    
    # 初始化数据加载器并加载数据集
    data_loader = DataLoader()
    dataset = data_loader.load_dataset(args.dataset_name)
    if not dataset:
        test_out("error", "Failed to load dataset", args.case_start, args.dataset_name)
        return
    
    # 修复：确保tokenizer是实际对象而不是协程
    tokenizer = llm.tokenizer
    if asyncio.iscoroutine(tokenizer):
        tokenizer = await tokenizer

    model_name = os.path.basename(args.model)
    prompt_handler = PromptHandler(tokenizer=tokenizer, model_name=model_name, thinking_mode=False)
    response_handler = ResponseHandler()
    search_guide_handler = SearchGuideHandler(dataset_name=args.dataset_name)

    reward_model = RewardModel(llm, prompt_handler, response_handler, args.dataset_name)
    process_reward_model = ProcessRewardModel(llm, prompt_handler, response_handler, args.dataset_name)
    pairwise_reward_model = PairwiseRewardModel(llm, prompt_handler, response_handler, args.dataset_name)
    
    # 初始化进度条
    pbar = tqdm(total=min(args.case_end, len(dataset)) - (args.case_start-1), 
                desc=f"Processing {args.dataset_name} cases")
    
    # 处理指定范围内的数据
    for case_idx in range(args.case_start-1, min(args.case_end, len(dataset))):
        case_start_time = time.time()  # 记录case开始时间
        
        case = dataset[case_idx]
        
        # 更新进度条显示当前处理的case
        pbar.set_description(f"Processing case {case_idx + 1}/{min(args.case_end, len(dataset))}")
        
        # 在每个case开始时输出超参数设置 (只写入日志文件)
        hyperparams = {
            "dataset_name": args.dataset_name,
            "model": args.model,
            "case_start": args.case_start,
            "case_end": args.case_end,
            "num_iterations": args.num_iterations,
            "exploration_constant": args.exploration_constant,
            "branch_factor": args.branch_factor,
            "branch_factor_init": args.branch_factor_init,
            "rollout_num": args.rollout_num,
            "show_runtime": args.show_runtime,
            "show_runtime_case": args.show_runtime_case,
            "verbose": args.verbose,
            "run_mode": args.run_mode,
            "max_depth": args.max_depth,
            "leaf_value_weight": args.leaf_value_weight, # <-- 新增
            "path_value_weight": args.path_value_weight, # <-- 新增
            "leaf_confidence_weight": args.leaf_confidence_weight, # <-- 新增
            "pairwise_weight": args.pairwise_weight,
            "process_weight": args.process_weight,
            "rollout_weight": args.rollout_weight,
            "rollout_sc_weight": args.rollout_sc_weight, # <-- 新增
            "rollout_confidence_weight": args.rollout_confidence_weight, # <-- 新增
        }
        
        # 构建超参数输出字符串
        hyperparams_str = "\nHyperparameters:\n" + "="*50 + "\n"
        for key, value in hyperparams.items():
            hyperparams_str += f"{key}: {value}\n"
        hyperparams_str += "="*50 + "\n"
        
        # 只输出到日志文件
        test_out("hyperparams", hyperparams_str, case_idx + 1, args.dataset_name)
        test_out("case_info", f"\nProcessing case {case_idx + 1}:", case_idx + 1, args.dataset_name)
        test_out("question", f"Question: {case['question']}", case_idx + 1, args.dataset_name)
        test_out("ground_truth", f"Ground Truth Answer: {case['answer']}", case_idx + 1, args.dataset_name)
        test_out("separator", '*'*100, case_idx + 1, args.dataset_name)

        
        reasoner = MCTS_Reasoner(
            question=case['question'],
            ground_truth=case['answer'],
            reward_model=reward_model,
            process_reward_model=process_reward_model,
            pairwise_reward_model=pairwise_reward_model,
            prompt_handler=prompt_handler,
            response_handler=response_handler,
            search_guide_handler=search_guide_handler,
            llm_client=llm,
            case_idx=case_idx + 1,
            dataset_name=args.dataset_name,
            exploration_constant=args.exploration_constant,
            rollout_num=args.rollout_num,
            show_runtime=args.show_runtime,
            verbose=args.verbose,
            max_depth=args.max_depth,
            branch_factor=args.branch_factor,   
            branch_factor_init=args.branch_factor_init,
            leaf_value_weight=args.leaf_value_weight, # <-- 新增
            path_value_weight=args.path_value_weight, # <-- 新增
            leaf_confidence_weight=args.leaf_confidence_weight, # <-- 新增
            pairwise_weight=args.pairwise_weight,
            process_weight=args.process_weight,
            rollout_weight=args.rollout_weight,
            rollout_sc_weight=args.rollout_sc_weight, # <-- 新增
            rollout_confidence_weight=args.rollout_confidence_weight, # <-- 新增
        )
        
        try:
            # 异步调用search方法
            reasoning_path, final_answer = await reasoner.search(
                num_iterations=args.num_iterations
            )
            
            test_out("reasoning_path", "\nReasoning Path:", case_idx + 1, args.dataset_name)
            for step in reasoning_path:
                test_out("reasoning_step", f"- {step}", case_idx + 1, args.dataset_name)
            
            #reasoner.save_tree_log()
                
            
            # 记录案例信息到summary
            case_end_time = time.time()
            case_duration = case_end_time - case_start_time
            
            case_info = {
                "case_id": case_idx + 1,
                "question": case['question'],
                "ground_truth": case['answer'],
                "final_answer": final_answer,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(case_start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(case_end_time)),
                "duration_seconds": case_duration,
                "reasoning_path": reasoning_path,
            }
            summary_data["cases"].append(case_info)


        except Exception as e:
            test_out("error", f"Error processing case {case_idx + 1}: {str(e)}", case_idx + 1, args.dataset_name)
            
            # 即使出错也记录案例信息
            case_end_time = time.time()
            case_duration = case_end_time - case_start_time
            
            case_info = {
                "case_id": case_idx + 1,
                "question": case['question'],
                "ground_truth": case['answer'],
                "final_answer": "Error",
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(case_start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(case_end_time)),
                "duration_seconds": case_duration,
                "reasoning_path": [f"Error: {str(e)}"]
            }
            summary_data["cases"].append(case_info)
            continue
        
        case_end_time = time.time()  # 记录case结束时间
        if args.show_runtime_case:
            print(f"Case {case_idx + 1} processed in {case_end_time - case_start_time:.2f} seconds.")
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()

    
    # 记录脚本结束时间并生成summary
    script_end_time = time.time()
    script_end_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    total_duration = script_end_time - script_start_time
    
    # 格式化总用时
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    if hours > 0:
        duration_str = f"{hours}时{minutes}分{seconds}秒"
    else:
        duration_str = f"{minutes}分{seconds}秒"
    
    # 生成summary内容
    summary_content = f"""MCTS推理脚本运行摘要
        {'='*50}

        开始时间: {script_start_datetime}
        结束时间: {script_end_datetime}
        总用时: {duration_str}

        运行参数:
        {'-'*30}
        """
    
    for key, value in summary_data["parameters"].items():
        summary_content += f"{key}: {value}\n"
    
    summary_content += f"\n案例处理详情:\n{'-'*30}\n"
    
    for case_info in summary_data["cases"]:
        case_duration = case_info["duration_seconds"]
        case_minutes = int(case_duration // 60)
        case_seconds = int(case_duration % 60)
        case_duration_str = f"{case_minutes}分{case_seconds}秒" if case_minutes > 0 else f"{case_seconds}秒"
        
        summary_content += f"\n案例 {case_info['case_id']}:\n"
        summary_content += f"  问题: {case_info['question']}\n"
        summary_content += f"  标准答案: {case_info['ground_truth']}\n"
        summary_content += f"  开始时间: {case_info['start_time']}\n"
        summary_content += f"  结束时间: {case_info['end_time']}\n"
        summary_content += f"  用时: {case_duration_str}\n"
        summary_content += f"  推理路径:\n"
        for i, step in enumerate(case_info['reasoning_path'], 1):
            summary_content += f"    {i}. {step}\n"
    
    # 保存summary.txt文件到logs目录
    log_dir = os.path.join("logs", f"{args.dataset_name}_{timestamp}_{args.case_start}_{args.case_end}")
    os.makedirs(log_dir, exist_ok=True)
    summary_file_path = os.path.join(log_dir, "summary.txt")
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"\n摘要已保存到: {summary_file_path}")
    except Exception as e:
        print(f"\n保存摘要文件时出错: {str(e)}")


def run_main():
    """同步包装器函数，用于运行异步主函数"""
    asyncio.run(main())

if __name__ == "__main__":
    # 限制只使用显卡x
    #os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    run_main()