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
        self.action = action       # 使该节点产生的动作
        self.children = []         # 孩子节点列表
        self.visits = 0            # 节点被访问的次数
        self.value = 0.0           # 累计奖励（价值）
        
        # ReasoningNode 特有的属性
        self.reasoning_path = []  # 记录推理路径
        self.coordinates = coordinates  # 节点在树中的坐标
        self.response = None  # 存储LLM的响应
        self.is_final = False  # 标记是否为终止节点
        self.full_context = ""  # 新增：完整的上下文（包含所有父节点内容）
        
        # 新增属性：子节点统计信息
        self.child_mean = 0.0      # 所有子节点的value均值
        self.child_diff = 1.0      # 所有子节点的value方差
        
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

    def is_leaf(self):
        # 当节点没有孩子时，认为是叶子节点
        return len(self.children) == 0

    def update_child_statistics(self):
        """
        更新子节点的统计信息（均值和方差）
        当子节点的value发生变化时调用此方法
        """
        if not self.children:
            self.child_mean = 0.0
            self.child_diff = 1.0
            return
        
        # 计算子节点value的均值，过滤掉visits为0的节点
        values = []
        for child in self.children:
            if child.visits > 0:
                values.append(child.value / child.visits)
        
        if not values:
            # 如果所有子节点的visits都为0，设置默认值
            self.child_mean = 0.0
            self.child_diff = 1.0
            return
        
        # 计算子节点value的均值
        self.child_mean = sum(values) / len(values)
        
        # 计算子节点value的方差
        if len(values) > 1:
            variance = sum((x - self.child_mean) ** 2 for x in values) / len(values)
            self.child_diff = variance
        else:
            self.child_diff = 1.0

class MCTS_Reasoner:
    def __init__(self, question: str, reward_model: Any, process_reward_model: Any, pairwise_reward_model: Any,
                 prompt_handler: PromptHandler, response_handler: ResponseHandler, search_guide_handler: SearchGuideHandler,
                 llm_client: Any, case_idx: int, dataset_name: str, ground_truth: Any = None,
                 rollout_num: int = 1, exploration_constant: float = 3, 
                 show_runtime: bool = True, verbose: bool = True,
                 max_depth: int = 5, branch_factor: int = 3, branch_factor_init: Any = None,
                 balance_beta: float = 0.5, rollout_weight: float = 0.3, process_weight: float = 0.4, pairwise_weight: float = 0.3):
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
        self.balance_beta = balance_beta  # 保留参数以保持向后兼容
        
        # 新增：三个奖励权重参数
        self.rollout_weight = rollout_weight  # 路径奖励权重
        self.process_weight = process_weight  # 过程奖励权重
        self.pairwise_weight = pairwise_weight  # 对比奖励权重
        
        # 从Search_Guide中加载各种引导信息
        self.search_guide_handler = search_guide_handler
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

    async def search(self, num_iterations: int = 20) -> str:
        """
        执行MCTS搜索
        
        Args:
            num_iterations: 迭代次数
        """
        start_time = time.time()
        
        # 创建根节点
        root = ReasoningNode(self.question, coordinates="root")
        self.root = root
        
        # 更新根节点的完整上下文
        root.update_full_context()
        
        # 为根节点创建日志记录
        self.tree_log["nodes"]["root"] = {
            "state": self.question,
            "response": None,
            "visits": 0,
            "value": 0.0,
            "is_final": False,
            "full_context": root.full_context
        }
        #self.save_tree_log()
        
        # 使用initial_step模板进行第一次展开，使用branch_factor_init
        prompt = self.prompt_handler.get_init_prompt(self.question)
        # 高温度采样使得模型有一些变数
        responses = await self.llm_client.generate(prompt, n=self.branch_factor_init, stop_stage="initial_step", seed=None, temperature=0.9)
        for i, response in enumerate(responses):  # 修复：添加enumerate获取索引
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
            
            root.children.append(new_node)
            
            # 如果是叶子节点，添加到叶子节点列表
            if new_node.is_final:
                self.leaf_nodes.append(new_node)
            
            # 记录节点信息到日志
            self.tree_log["nodes"][f"({i+1})"] = {
                "state": initial_step,
                "response": response,
                "visits": 0,
                "value": 0.0,
                "is_final": new_node.is_final,
                "full_context": new_node.full_context
            }

        # 执行MCTS迭代
        # print('*'*30+"searching now"+'*'*30)
        for _ in range(num_iterations):
            test_out("num_iterations:"+str(_), "", self.case_idx, self.dataset_name)
            node = self._select(root)
            # 如果选中的节点不是终止节点，则进行扩展
            if not node.is_final:
                new_nodes, new_values = await self._async_expand_and_simulate(node)
                # 遍历所有新节点，对每个节点进行 simulate 和 backpropagate
                for new_node, new_value in zip(new_nodes, new_values):
                    self._backpropagate(new_node, new_value)
                    
                    # 如果是叶子节点，添加到叶子节点列表
                    if new_node.is_final:
                        self.leaf_nodes.append(new_node)
            else:
                reward = await self._final_simulate(node)
                self._backpropagate(node, reward)

        # 选择最优路径 - 调用新的pick_final_answer方法
        best_path, best_solution, answer_frequency = self._pick_final_answer(root)
        #print("best_path:", ''.join(best_path))
        #print("best_solution:", best_solution)
        
        # 使用最佳推理路径生成最终答案
        final_answer = await self._output_best_answer(best_solution)

        self.tree_log['final_results'] = {
            "ground_truth": self.ground_truth,
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

            # 确保子节点统计量为最新
            node.update_child_statistics()

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

    # 新增核心方法：_async_expand_and_simulate
    # -------------------------------------------------------------------------
    async def _async_expand_and_simulate(self, chosen_node: ReasoningNode) -> Tuple[List[ReasoningNode], List[float]]:
        """
        合并了扩展和模拟步骤。
        基于DAG逻辑：每个Idea独立扩展 -> 并行执行(Process评估 + Rollout流水线) -> 聚合分数
        """
        # 0. 边界条件：最大深度检查
        if chosen_node.depth >= self.max_depth:
            # 复用原有的最大深度扩展逻辑
            final_nodes = await self._max_depth_expand(chosen_node)
            final_node = final_nodes[0]
            # 直接对终止节点进行打分
            reward = await self._final_simulate(final_node)
            return final_nodes, [reward]

        # 1. Root: 生成核心想法
        context = chosen_node.full_context
        core_ideas = await self._generate_diverse_core_ideas(context, self.branch_factor)
        test_out("core_ideas", core_ideas, self.case_idx, self.dataset_name)

        # 定义单分支流水线（对应DAG中的一个分支：Idea -> Expand -> (Process & Rollout)）
        async def _single_branch_pipeline(core_idea: str):
            # A. 节点扩展 (Idea_Expand)
            # [Dependency]: Generate_Core_Ideas -> Idea_Expand
            new_node = await self._idea_expand(chosen_node, core_idea)

            # B. 定义并行任务
            # 这里的 process_task 和 rollout_flow_task 互不阻塞
            
            # Task B1: Process Evaluation
            # [Dependency]: Idea_Expand -> Process_Eval
            async def _run_process_eval():
                if self.process_weight == 0:
                    return 0.0
                p_reward, p_prompt, p_response = await self._process_evaluation(
                    new_node, new_node.full_context, new_node.state
                )
                # 记录 Process 日志 (保持原有逻辑)
                # ...这里省略了详细的日志字典构建，实际使用时可直接复用原有的 log 逻辑...
                test_out("process_score", f"{p_reward:.6f}", self.case_idx, self.dataset_name)
                return p_reward

            # Task B2: Rollout Workflow (Paths -> Evals)
            # [Dependency]: Idea_Expand -> Rollout_Paths -> Rollout_Eval
            async def _run_rollout_flow():
                if self.rollout_weight == 0:
                    return 0.0
                
                # Step 1: Rollout Paths
                # 如果节点本身是Final，直接用Full Context，否则生成Paths
                if new_node.is_final:
                    paths = [new_node.get_full_reasoning_path()]
                else:
                    paths = await self._rollout_paths(new_node)
                
                if not paths: return 0.0

                # Step 2: Rollout Evals (Parallel Map)
                # [Dependency]: Rollout_Paths -> Rollout_Eval_X
                eval_tasks = [
                    self._rollout_evaluation(new_node, path, i) 
                    for i, path in enumerate(paths)
                ]
                results = await asyncio.gather(*eval_tasks)
                
                # Step 3: Local Aggregate (计算当前节点的平均Rollout分)
                scores = [res[0] for res in results] # res is (reward, prompt, response)
                avg_score = sum(scores) / len(scores)
                
                # 记录 Rollout 日志
                for i, (score, _, _) in enumerate(results):
                    test_out("rollout_reward_score", f"rollout_{i}: {score:.6f}", self.case_idx, self.dataset_name)
                
                return avg_score

            # C. 并行执行评估任务 (Process 和 Rollout 同时进行)
            # [Dependency]: Wait for Process_Eval AND Rollout_Flow
            process_reward, rollout_reward = await asyncio.gather(
                _run_process_eval(),
                _run_rollout_flow()
            )

            # D. 计算最终分数 (剔除 Pairwise)
            # [Dependency]: Calculate_Score
            final_reward = (self.process_weight * process_reward) + (self.rollout_weight * rollout_reward)
            
            # 记录最终分数日志
            self.tree_log["nodes"][new_node.coordinates]["value"] = final_reward
            test_out("final_reward", f"Node {new_node.coordinates}: final={final_reward:.6f} (p={process_reward:.6f}, r={rollout_reward:.6f})", self.case_idx, self.dataset_name)

            return new_node, final_reward

        # 2. 启动所有分支任务 (Fan-out)
        branch_tasks = [_single_branch_pipeline(idea) for idea in core_ideas]
        
        # 3. 等待所有分支完成 (Final Gather)
        results = await asyncio.gather(*branch_tasks)
        
        # 4. 解包结果
        if not results:
            return [], []
            
        new_nodes, new_values = zip(*results)
        return list(new_nodes), list(new_values)

    async def _async_expand(self, chosen_node: ReasoningNode):
        # 获取当前节点深度
        current_depth = chosen_node.depth

        # 如果达到最大深度，执行rollout而不是扩展
        if current_depth >= self.max_depth:
            return await self._max_depth_expand(chosen_node)

        # 使用节点的full_context
        context = chosen_node.full_context
        # test_out("chosen_node_context", context, self.case_idx, self.dataset_name)

        # 第一步：生成多个不同的核心想法
        core_ideas = await self._generate_diverse_core_ideas(context, self.branch_factor)
        test_out("core_ideas", core_ideas, self.case_idx, self.dataset_name)

        # 第二步：将每个idea都展开为完整的node
        expand_tasks = [self._idea_expand(chosen_node, core_idea) for core_idea in core_ideas]
        new_nodes = await asyncio.gather(*expand_tasks)

        return new_nodes

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
        test_out("diverse_ideas_prompt", prompt, self.case_idx, self.dataset_name)

        # 添加多样化想法生成阶段的停止序列
        response = await self.llm_client.generate(prompt, stop_stage='diverse_ideas')
        test_out("diverse_ideas_response", response, self.case_idx, self.dataset_name)

        # 解析核心想法，调用ResponseHandler的方法
        core_ideas = self.response_handler.parse_diverse_ideas(response, num_ideas)
        return core_ideas

    async def _idea_expand(self, parent_node: ReasoningNode, core_idea: str):
        prompt = self.prompt_handler.get_expand_prompt(
            previous_steps=parent_node.full_context, 
            core_instruction=core_idea,
            expand_guidance=self.expand_guidance
        )
        test_out("expand_prompt", prompt, self.case_idx, self.dataset_name)

        # 调用LLM生成下一步推理
        expand_response = await self.llm_client.generate(prompt, 
                                            stop_stage='expand', 
                                            skip_special_tokens=False)
        test_out("expand_response", expand_response, self.case_idx, self.dataset_name)
        
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
            "full_path": new_node.full_context  # 记录完整上下文
        }
        return new_node
        
    async def _async_simulate(self, new_nodes: List[ReasoningNode]) -> List[float]:
        """
        异步模拟方法，根据信号开关决定执行哪些计算
        
        Args:
            new_nodes: 新生成的节点列表
        Returns:
            每个节点的最终奖励列表
        """
        final_rewards = []
        node_rewards = {}  # 存储每个节点的各种奖励值
        
        # 初始化所有节点的奖励字典
        for node in new_nodes:
            test_out("simulating", f"Processing node: {node.coordinates}", self.case_idx, self.dataset_name)
            node_rewards[node.coordinates] = {
                'rollout_reward': 0.0,
                'process_reward': 0.0,
                'compared_reward': 0.0
            }
        
        # 第一阶段：如果rollout_weight != 0，先执行所有节点的rollout路径生成
        rollout_paths_results = {}
        if self.rollout_weight != 0:
            test_out("rollout_phase", "Starting rollout paths generation phase", self.case_idx, self.dataset_name)
            
            # 为所有节点创建rollout路径生成任务（对终止节点直接使用full_context）
            rollout_paths_tasks = []
            node_task_pairs = []
            for node in new_nodes:
                if node.is_final:
                    full_reasoning_path = node.get_full_reasoning_path()
                    rollout_paths_results[node.coordinates] = [full_reasoning_path]
                    test_out("rollout_paths_skipped_final", f"Node {node.coordinates} is final; using full_context as rollout path", 
                            self.case_idx, self.dataset_name)
                else:
                    task = self._rollout_paths(node)
                    rollout_paths_tasks.append(task)
                    node_task_pairs.append(node)
            
            # 执行非终止节点的rollout路径生成任务
            if rollout_paths_tasks:
                rollout_paths_results_list = await asyncio.gather(*rollout_paths_tasks)
                
                # 将结果映射到对应的节点
                for node, full_reasoning_paths in zip(node_task_pairs, rollout_paths_results_list):
                    rollout_paths_results[node.coordinates] = full_reasoning_paths
                    test_out("rollout_paths_generated", f"Node {node.coordinates}: generated {len(full_reasoning_paths)} rollout paths", 
                            self.case_idx, self.dataset_name)
        
        # 第二阶段：创建所有评估任务（包括rollout奖励评估、process评估、pairwise评估）
        all_tasks = []
        task_mapping = {}  # 用于映射任务到节点和奖励类型

        
        # 1. 创建process评估任务
        if self.process_weight != 0:
            for node in new_nodes:
                process_task = self._process_evaluation(node, node.full_context, node.state)
                all_tasks.append(process_task)
                task_mapping[process_task] = {
                    'node': node,
                    'reward_type': 'process_reward',
                    'signal': 'process'
                }
        
        # 2. 创建rollout奖励评估任务（基于第一阶段生成的路径）
        if self.rollout_weight != 0:
            for node in new_nodes:
                full_reasoning_paths = rollout_paths_results[node.coordinates]
                # 为每个路径创建奖励评估任务
                for rollout_idx, full_reasoning_path in enumerate(full_reasoning_paths):
                    rollout_reward_task = self._rollout_evaluation(node, full_reasoning_path, rollout_idx)
                    all_tasks.append(rollout_reward_task)
                    task_mapping[rollout_reward_task] = {
                        'node': node,
                        'reward_type': 'rollout_reward',
                        'signal': 'rollout',
                        'rollout_idx': rollout_idx,
                        'full_reasoning_path': full_reasoning_path
                    }
        
        # 3. 创建pairwise评估任务
        if self.pairwise_weight != 0 and len(new_nodes) > 1:
            pairwise_task = self._pairwise_evaluation(new_nodes)
            all_tasks.append(pairwise_task)
            task_mapping[pairwise_task] = {
                'nodes': new_nodes,
                'reward_type': 'compared_reward',
                'signal': 'pairwise',
                'is_batch_task': True
            }
        
        # 执行所有评估任务
        if all_tasks:
            test_out("evaluation_phase", f"Starting evaluation phase with {len(all_tasks)} tasks", self.case_idx, self.dataset_name)
            results = await asyncio.gather(*all_tasks)
            
            # 收集日志并更新奖励（按类型归类）
            process_logs = []
            rollout_logs = []
            pairwise_logs = []
            
            # 处理结果
            for task, result in zip(all_tasks, results):
                task_info = task_mapping[task]
                
                if task_info['signal'] == 'process':
                    node = task_info['node']
                    process_reward, process_prompt, process_response = result
                    node_rewards[node.coordinates]['process_reward'] = process_reward
                    process_logs.append({
                        'node_coords': node.coordinates,
                        'prompt': process_prompt,
                        'response': process_response,
                        'score': process_reward
                    })
                    
                elif task_info['signal'] == 'rollout':
                    node = task_info['node']
                    rollout_reward, rollout_prompt, rollout_response = result
                    if 'rollout_rewards' not in node_rewards[node.coordinates]:
                        node_rewards[node.coordinates]['rollout_rewards'] = []
                    node_rewards[node.coordinates]['rollout_rewards'].append(rollout_reward)
                    rollout_logs.append({
                        'node_coords': node.coordinates,
                        'rollout_idx': task_info['rollout_idx'] + 1,
                        'prompt': rollout_prompt,
                        'response': rollout_response,
                        'score': rollout_reward
                    })
                    
                elif task_info['signal'] == 'pairwise' and task_info.get('is_batch_task'):
                    compared_rewards, pair_details = result
                    for i, node in enumerate(new_nodes):
                        node_rewards[node.coordinates]['compared_reward'] = compared_rewards[i]
                    # 记录每个pair的prompt/response/score
                    for pd in pair_details:
                        pairwise_logs.append({
                            'prompt': pd.get('prompt', ''),
                            'response': pd.get('response', ''),
                            'score': pd.get('score', 0.0),
                            'nodeA_idx': pd.get('nodeA_idx'),
                            'nodeB_idx': pd.get('nodeB_idx')
                        })
            
            # 计算每个节点的平均rollout奖励
            if self.rollout_weight != 0:
                for node in new_nodes:
                    coords = node.coordinates
                    if 'rollout_rewards' in node_rewards[coords]:
                        rollout_rewards = node_rewards[coords]['rollout_rewards']
                        avg_rollout_reward = sum(rollout_rewards) / len(rollout_rewards)
                        node_rewards[coords]['rollout_reward'] = avg_rollout_reward
            
            # 按顺序输出日志：process -> rollout -> pairwise
            for log in process_logs:
                test_out("process_prompt", log['prompt'], self.case_idx, self.dataset_name)
                test_out("process_response", log['response'], self.case_idx, self.dataset_name)
                test_out("process_score", f"{log['score']:.6f}", self.case_idx, self.dataset_name)
            
            for log in rollout_logs:
                test_out("rollout_reward_prompt", log['prompt'], self.case_idx, self.dataset_name)
                test_out("rollout_reward_response", log['response'], self.case_idx, self.dataset_name)
                test_out("rollout_reward_score", f"rollout_{log['rollout_idx']}: {log['score']:.6f}", self.case_idx, self.dataset_name)
            
            for log in pairwise_logs:
                test_out("pairwise_prompt", log['prompt'], self.case_idx, self.dataset_name)
                test_out("pairwise_response", log['response'], self.case_idx, self.dataset_name)
                test_out("pairwise_score", f"{log['score']}", self.case_idx, self.dataset_name)


        # 计算最终奖励
        for node in new_nodes:
            coords = node.coordinates
            rollout_reward = node_rewards[coords]['rollout_reward']
            process_reward = node_rewards[coords]['process_reward']
            compared_reward = node_rewards[coords]['compared_reward']
            
            # 加权计算最终奖励
            final_reward = self.rollout_weight * rollout_reward + self.process_weight * process_reward + self.pairwise_weight * compared_reward
            
            test_out("final_reward", f"Node {coords}: final_reward={final_reward:.6f} (process={process_reward:.6f}, rollout={rollout_reward:.6f}, compared={compared_reward:.6f})", 
                    self.case_idx, self.dataset_name)
            
            final_rewards.append(final_reward)
        
        return final_rewards

    async def _rollout_paths(self, new_node: ReasoningNode):
        """
            用于对单个的new_node进行rollout后，返回多条轨迹full_reasoning_paths
        """
        # 获取当前节点深度
        current_depth = new_node.depth
        
        # 根据深度动态调整rollout次数
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
        
        test_out("dynamic_rollout", f"Depth: {current_depth}/{self.max_depth}, Ratio: {depth_ratio}, Rollouts: {actual_rollout_num}/{self.rollout_num}", self.case_idx, self.dataset_name)
        test_out("previous steps", new_node.full_context, self.case_idx, self.dataset_name)
        # 执行rollout
        rollout_prompt = self.prompt_handler.get_rollout_prompt(new_node.full_context)
        test_out("rollout_prompt:", rollout_prompt, self.case_idx, self.dataset_name)
        
        # 添加rollout阶段的停止序列
        rollout_responses = await self.llm_client.generate(rollout_prompt, stop_stage='rollout', n=actual_rollout_num, max_tokens=1000)

        full_reasoning_paths = []
        for rollout_idx, rollout_response in enumerate(rollout_responses):
            full_reasoning_path = new_node.get_full_reasoning_path(rollout_response)
            full_reasoning_paths.append(full_reasoning_path)
            test_out("rollout_response:", rollout_response, self.case_idx, self.dataset_name)
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
        print("compared_rewards:", compared_rewards)
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
        
        response = await self.llm_client.generate(prompt, stop_stage='rollout')
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
            "full_context": new_node.full_context
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
        test_out("terminal_reward_prompt", reward_prompt, self.case_idx, self.dataset_name)
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
        
        Args:
            root: MCTS树的根节点
        Returns:
            最佳推理路径
        """
        if not self.leaf_nodes:
            # 如果没有找到任何叶子节点，返回访问次数最多的子节点路径
            best_child = max(root.children, key=lambda c: c.visits)
            best_path = best_child.reasoning_path
            best_solution = best_child.full_context
            test_out("no_leaf_nodes_found", f"Using best child with most visits: {best_child.coordinates}", self.case_idx, self.dataset_name)
            return best_path, best_solution, {}
        
        # 计算答案频率作为置信度得分
        answer_frequency = self._calculate_answer_frequency()
        test_out("answer_frequency", f"Answer frequency distribution: {answer_frequency}", self.case_idx, self.dataset_name)
        
        # 输出所有答案的置信度排序（从大到小）
        self._output_answer_confidence_ranking(answer_frequency)
        
        # 计算每个叶子节点的加权价值
        best_leaf = None
        best_weighted_value = float('-inf')
        
        test_out("evaluating_leaf_nodes", f"Found {len(self.leaf_nodes)} leaf nodes to evaluate", self.case_idx, self.dataset_name)
        
        for leaf in self.leaf_nodes:
            leaf_coords = leaf.coordinates
            test_out("evaluating_leaf", f"Evaluating leaf node: {leaf_coords}", self.case_idx, self.dataset_name)
            
            # 计算路径平均价值：从根节点到叶子节点路径上所有节点的value/visits的平均值
            path_values = []
            test_out("path_analysis", f"Analyzing reasoning path for leaf {leaf_coords}: {leaf.reasoning_path}", self.case_idx, self.dataset_name)
            
            # 从叶子节点向上遍历到根节点，收集路径上所有节点的value/visits
            current_node = leaf
            path_nodes = []
            while current_node is not None:
                path_nodes.append(current_node)
                current_node = current_node.parent
            
            # 从根节点到叶子节点的顺序（reverse path_nodes）
            path_nodes.reverse()
            
            for node in path_nodes:
                if node.visits > 0:
                    step_value = node.value / node.visits
                    path_values.append(step_value)
                    test_out("step_value", f"Node {node.coordinates}: visits={node.visits}, value={node.value:.6f}, step_value={step_value:.6f}", self.case_idx, self.dataset_name)
                else:
                    test_out("step_value", f"Node {node.coordinates}: visits=0, step_value=0", self.case_idx, self.dataset_name)
                    path_values.append(0.0)
            
            path_mean_value = sum(path_values) / len(path_values) if path_values else 0
            test_out("path_mean_calculation", f"Leaf {leaf_coords}: path_values={path_values}, path_mean_value={path_mean_value:.6f}", self.case_idx, self.dataset_name)
            
            # 计算叶子节点价值：value/visits
            leaf_value = leaf.value / leaf.visits if leaf.visits > 0 else 0
            test_out("leaf_value_calculation", f"Leaf {leaf_coords}: visits={leaf.visits}, total_value={leaf.value:.6f}, leaf_value={leaf_value:.6f}", self.case_idx, self.dataset_name)
            
            # 计算置信度得分
            leaf_answer = self._extract_answer_from_response(leaf.response)
            confidence_score = answer_frequency.get(leaf_answer, 0) / len(self.leaf_nodes) * 10 if leaf_answer else 0 #保持量纲一致，都是0-10分
            test_out("confidence_calculation", f"Leaf {leaf_coords}: answer='{leaf_answer}', confidence_score={confidence_score:.6f}", self.case_idx, self.dataset_name)
            
            # 加权计算最终价值 - 新公式：weighted_value = balance_beta * path_mean_value + (1 - balance_beta) * (leaf_value + confidence_score) / 2
            enhanced_leaf_value = (leaf_value + confidence_score) / 2
            weighted_value = self.balance_beta * path_mean_value + (1 - self.balance_beta) * enhanced_leaf_value
            test_out("weighted_value_calculation", f"Leaf {leaf_coords}: balance_beta={self.balance_beta}, enhanced_leaf_value={enhanced_leaf_value:.6f} (leaf_value={leaf_value:.6f} + confidence_score={confidence_score:.6f}), weighted_value={weighted_value:.6f}", self.case_idx, self.dataset_name)
            
            if weighted_value > best_weighted_value:
                best_weighted_value = weighted_value
                best_leaf = leaf
                test_out("new_best_leaf", f"Leaf {leaf_coords} becomes new best with weighted_value={weighted_value:.6f}", self.case_idx, self.dataset_name)
            else:
                test_out("leaf_comparison", f"Leaf {leaf_coords} weighted_value={weighted_value:.6f} <= current_best={best_weighted_value:.6f}", self.case_idx, self.dataset_name)
        
        # 从根节点到最佳叶子节点的完整路径
        best_path = best_leaf.reasoning_path if best_leaf else []
        best_solution = best_leaf.full_context if best_leaf else ""
        best_leaf_coords = best_leaf.coordinates if best_leaf else "None"
        test_out("final_path_selection", f"Selected best leaf: {best_leaf_coords} with weighted_value={best_weighted_value:.6f}", self.case_idx, self.dataset_name)
        test_out("final_path_details", f"Best path coordinates: {best_leaf_coords}, path length: {len(best_path)}", self.case_idx, self.dataset_name)
        
        return best_path, best_solution, answer_frequency

    def _extract_answer_from_response(self, response: str) -> str:
        """从响应中提取最终答案"""
        # 首先尝试提取\boxed{}格式的答案
        boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
        if boxed_matches:
            # 返回最后一个boxed答案
            return boxed_matches[-1].strip()
        
        # 如果没有找到boxed格式，尝试其他常见的答案格式
        # 查找"答案是"、"答案为"、"answer is"等模式
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
                return matches[-1].strip()
        
        # 如果都没找到，返回空字符串
        return ""
    
    def _calculate_answer_frequency(self) -> dict:
        """计算所有叶子节点中答案的出现频率"""
        answer_counts = {}
        
        for leaf in self.leaf_nodes:
            answer = self._extract_answer_from_response(leaf.response)
            if answer:  # 只统计非空答案
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        return answer_counts
    
    def _output_answer_confidence_ranking(self, answer_frequency: dict):
        """输出所有答案的置信度排序（从大到小）"""
        if not answer_frequency:
            test_out("confidence_ranking", "No valid answers found in leaf nodes", self.case_idx, self.dataset_name)
            return
        
        total_leaf_nodes = len(self.leaf_nodes)
        
        # 计算每个答案的置信度得分并排序
        confidence_scores = []
        for answer, count in answer_frequency.items():
            confidence_score = count / total_leaf_nodes
            confidence_scores.append((answer, count, confidence_score))
        
        # 按置信度得分从大到小排序
        confidence_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 输出排序结果
        test_out("confidence_ranking", f"Answer confidence ranking (total leaf nodes: {total_leaf_nodes}):", self.case_idx, self.dataset_name)
        for i, (answer, count, confidence) in enumerate(confidence_scores, 1):
            test_out("confidence_ranking", f"  {i}. Answer: '{answer}' | Count: {count} | Confidence: {confidence:.4f}", self.case_idx, self.dataset_name)

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
        test_out("best_answer_prompt", prompt, self.case_idx, self.dataset_name)
        
        # 调用LLM生成最终答案
        final_answer_response = await self.llm_client.generate(
            prompt, 
            stop_stage='best_answer', 
            skip_special_tokens=False
        )
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
                "rollout_num": self.rollout_num,
                "max_depth": self.max_depth,
                "branch_factor": self.branch_factor,
                "branch_factor_init": self.branch_factor_init,
                "balance_beta": self.balance_beta,
            },
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
    parser.add_argument("--balance_beta", type=float, default=0.65,
                       help="过程奖励和rollout奖励的加权系数 (0-1)")
    parser.add_argument("--pairwise_weight", type=float, default=0.5,
                       help="对比奖励信号的权重")
    parser.add_argument("--process_weight", type=float, default=0.2,
                       help="过程评估信号的权重")
    parser.add_argument("--rollout_weight", type=float, default=0.3,
                       help="rollout模拟评估信号的权重")
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
        "balance_beta": args.balance_beta,
        "pairwise_weight": args.pairwise_weight,
        "process_weight": args.process_weight,
        "rollout_weight": args.rollout_weight,
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
            "balance_beta": args.balance_beta,
            "pairwise_weight": args.pairwise_weight,
            "process_weight": args.process_weight,
            "rollout_weight": args.rollout_weight
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
            balance_beta=args.balance_beta,
            pairwise_weight=args.pairwise_weight,
            process_weight=args.process_weight,
            rollout_weight=args.rollout_weight
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    run_main()