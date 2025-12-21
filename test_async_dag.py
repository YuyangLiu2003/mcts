import asyncio
import time  # 新增：用于统计耗时
from typing import List
from aiodag import task

class IdeaProcessor:
    def __init__(self, num_ideas: int = 3, rollout_num: int = 3):
        self.num_ideas = num_ideas
        self.rollout_num = rollout_num
    
    @task
    async def _generate_diverse_core_ideas(self, context: str, num_ideas: int) -> List[str]:
        print(f"Generating {num_ideas} core ideas based on context: {context}")  # 英文日志（可选）
        await asyncio.sleep(1)
        ideas = [f"idea_{i}" for i in range(num_ideas)]
        print(f"Generated core ideas: {ideas}")
        return ideas
    
    # B层单个任务
    @task
    async def _idea_expand(self, new_idea: str) -> str:
        print(f"Expanding idea: {new_idea}")
        await asyncio.sleep(0.5)
        expanded = f"expanded_{new_idea}"
        print(f"Expanded idea: {expanded}")
        return expanded
    
    # 中间task：A层列表 → B层多个任务（一对多依赖）
    @task
    async def _create_expand_tasks(self, core_ideas: List[str]) -> List[str]:
        expand_tasks = [self._idea_expand(idea) for idea in core_ideas]
        return await asyncio.gather(*expand_tasks) # 返回多个b层任务，是一个列表
    
    # 单个c层任务
    @task
    async def _process_evaluation(self, full_step: str) -> float:
        print(f"Processing evaluation for: {full_step}")
        await asyncio.sleep(0.3)
        process_score = 1
        print(f"Evaluation result: {process_score}")
        return process_score
    
    # 中间task：B层列表 → C层多个任务（一对多依赖）
    @task
    async def _create_process_eval_tasks(self, full_steps: List[str]) -> List[str]:
        # 将b层的多个任务作为一个列表传入后，每个都创建一个c层单任务
        eval_tasks = [self._process_evaluation(step) for step in full_steps]
        return await asyncio.gather(*eval_tasks) # 返回多个c层任务，是一个列表
    
    @task
    async def _pairwise_evaluation(self, full_steps: List[str]) -> List[float]:
        print(f"Performing pairwise evaluation on: {full_steps}")
        await asyncio.sleep(1)
        scores = [i * 0.1 for i in range(len(full_steps))]
        print(f"Pairwise evaluation scores: {scores}")
        return scores
    
    # 单个e层任务：对一个expand结果进行rollout
    @task
    async def _rollout_paths(self, full_step: str, rollout_num: int) -> List[str]:
        print(f"Generating {rollout_num} rollout paths for: {full_step}")
        await asyncio.sleep(0.8)
        paths = [f"rollout_{i}" for i in range(rollout_num)]
        print(f"Generated rollout paths: {paths}")
        return paths
    
    # 中间task1：B层列表 → E层多个任务（每个B层结果生成N个rollout路径，一对多依赖）
    @task
    async def _create_rollout_path_tasks(self, full_steps: List[str]) -> List[List[str]]:
        rollout_tasks = [self._rollout_paths(step, self.rollout_num) for step in full_steps]
        return await asyncio.gather(*rollout_tasks)  # 返回：[[path1-1, path1-2], [path2-1, path2-2], ...]
    
    @task
    async def _rollout_evaluation(self, rollout_result: str, rollout_idx: int) -> float:
        print(f"Evaluating rollout result {rollout_idx}: {rollout_result}")
        await asyncio.sleep(0.2)
        rollout_score=1
        print(f"Rollout evaluation result: {rollout_score}")
        return rollout_score
    
    # 中间task2：E层嵌套列表 → F层多个任务（一对多依赖，需两层拆解）
    @task
    async def _create_rollout_eval_tasks(self, rollout_paths_list: List[List[str]]) -> List[str]:
        eval_tasks = []
        for step_idx, paths in enumerate(rollout_paths_list):  # 遍历每个B层结果对应的rollout列表
            for path_idx, path in enumerate(paths):  # 遍历每个rollout路径
                eval_tasks.append(self._rollout_evaluation(path, path_idx))
        return await asyncio.gather(*eval_tasks)

# 将所有的依赖关系都建模为DAG，理论上最快
async def main1():
    # 新增：记录任务开始时间
    start_time = time.time()
    
    processor = IdeaProcessor(num_ideas=3, rollout_num=3)
    context = "User requirement analysis"  # 英文上下文（可选）
    
    # 1. A层：核心想法任务
    core_ideas_task = processor._generate_diverse_core_ideas(context, processor.num_ideas)
    
    # 2. B层：依赖A层（通过中间task拆解）
    full_steps_task = processor._create_expand_tasks(core_ideas_task)
    
    # 3. C层：依赖B层（通过中间task拆解）
    process_evaluations_task = processor._create_process_eval_tasks(full_steps_task)
    
    # 4. D层：依赖B层完整列表（直接传任务对象，aiodag自动等B层完成）
    pairwise_scores_task = processor._pairwise_evaluation(full_steps_task)
    
    # 5. E层：依赖B层（通过中间task拆解）
    rollout_paths_list_task = processor._create_rollout_path_tasks(full_steps_task)
    
    # 6. F层：依赖E层（通过中间task拆解嵌套列表）
    rollout_evaluations_task = processor._create_rollout_eval_tasks(rollout_paths_list_task)
    
    # 等待所有顶层任务完成（aiodag自动处理所有子依赖）
    final_results = await asyncio.gather(
        core_ideas_task,
        full_steps_task,
        process_evaluations_task,
        pairwise_scores_task,
        rollout_paths_list_task,
        rollout_evaluations_task,
        return_exceptions=True
    )
    
    # 新增：记录任务结束时间并计算总耗时
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 要求1：依次输出所有任务结果（英文）
    print("\n=== All Task Results ===")
    # 1. Core ideas (A层)
    core_ideas = final_results[0]
    if isinstance(core_ideas, Exception):
        print(f"1. Core ideas: Error - {str(core_ideas)}")
    else:
        print(f"1. Core ideas: {core_ideas}")
    
    # 2. Expanded ideas (B层)
    expanded_ideas = final_results[1]
    if isinstance(expanded_ideas, Exception):
        print(f"2. Expanded ideas: Error - {str(expanded_ideas)}")
    else:
        print(f"2. Expanded ideas: {expanded_ideas}")
    
    # 3. Process evaluations (C层)
    process_evals = final_results[2]
    if isinstance(process_evals, Exception):
        print(f"3. Process evaluations: Error - {str(process_evals)}")
    else:
        print(f"3. Process evaluations: {process_evals}")
    
    # 4. Pairwise evaluation scores (D层)
    pairwise_scores = final_results[3]
    if isinstance(pairwise_scores, Exception):
        print(f"4. Pairwise evaluation scores: Error - {str(pairwise_scores)}")
    else:
        print(f"4. Pairwise evaluation scores: {pairwise_scores}")
    
    # 5. Rollout paths list (E层)
    rollout_paths = final_results[4]
    if isinstance(rollout_paths, Exception):
        print(f"5. Rollout paths list: Error - {str(rollout_paths)}")
    else:
        print(f"5. Rollout paths list: {rollout_paths}")
    
    # 6. Rollout evaluations (F层)
    rollout_evals = final_results[5]
    if isinstance(rollout_evals, Exception):
        print(f"6. Rollout evaluations: Error - {str(rollout_evals)}")
    else:
        print(f"6. Rollout evaluations: {rollout_evals}")
    
    # 要求2：输出总耗时
    print(f"\n=== Execution Summary ===")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    return final_results

# 这个版本是先等expand完成后，再rollout，最后做所有的evaluation
async def main2():
    start_time = time.time()
    processor = IdeaProcessor(num_ideas=3, rollout_num=3)
    context = "User requirement analysis"
    
    # ===================== 阶段1：执行A层+B层（await完成后进入下一阶段） =====================
    print("\n=== Phase 1: Generate core ideas and expand ideas ===")
    # A层：核心想法任务
    core_ideas_task = processor._generate_diverse_core_ideas(context, processor.num_ideas)
    # B层：依赖A层的扩展任务
    full_steps_task = processor._create_expand_tasks(core_ideas_task)
    # 等待阶段1所有任务完成
    core_ideas, full_steps = await asyncio.gather(
        core_ideas_task,
        full_steps_task,
        return_exceptions=True
    )
    
    # ===================== 阶段2：执行E层（rollout_paths）（await完成后进入下一阶段） =====================
    print("\n=== Phase 2: Generate rollout paths for each expanded idea ===")
    # E层：基于阶段1的B层结果生成rollout路径
    rollout_paths_list_task = processor._create_rollout_path_tasks(full_steps_task)
    # 等待阶段2任务完成
    rollout_paths_list = await rollout_paths_list_task
    
    # ===================== 阶段3：执行C层+D层+F层（await完成） =====================
    print("\n=== Phase 3: Execute evaluations (process/pairwise/rollout) ===")
    # C层：处理评估
    process_evaluations_task = processor._create_process_eval_tasks(full_steps_task)
    # D层：两两评估
    pairwise_scores_task = processor._pairwise_evaluation(full_steps_task)
    # F层：rollout评估
    rollout_evaluations_task = processor._create_rollout_eval_tasks(rollout_paths_list_task)
    # 等待阶段3所有任务完成
    process_evals, pairwise_scores, rollout_evals = await asyncio.gather(
        process_evaluations_task,
        pairwise_scores_task,
        rollout_evaluations_task,
        return_exceptions=True
    )
    
    # 汇总所有结果
    final_results = [
        core_ideas, full_steps, process_evals,
        pairwise_scores, rollout_paths_list, rollout_evals
    ]
    
    # ===================== 结果输出 =====================
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n=== All Task Results ===")
    # 1. Core ideas (A层)
    if isinstance(final_results[0], Exception):
        print(f"1. Core ideas: Error - {str(final_results[0])}")
    else:
        print(f"1. Core ideas: {final_results[0]}")
    # 2. Expanded ideas (B层)
    if isinstance(final_results[1], Exception):
        print(f"2. Expanded ideas: Error - {str(final_results[1])}")
    else:
        print(f"2. Expanded ideas: {final_results[1]}")
    # 3. Process evaluations (C层)
    if isinstance(final_results[2], Exception):
        print(f"3. Process evaluations: {final_results[2]}")
    else:
        print(f"3. Process evaluations: {final_results[2]}")
    # 4. Pairwise evaluation scores (D层)
    if isinstance(final_results[3], Exception):
        print(f"4. Pairwise evaluation scores: Error - {str(final_results[3])}")
    else:
        print(f"4. Pairwise evaluation scores: {final_results[3]}")
    # 5. Rollout paths list (E层)
    if isinstance(final_results[4], Exception):
        print(f"5. Rollout paths list: Error - {str(final_results[4])}")
    else:
        print(f"5. Rollout paths list: {final_results[4]}")
    # 6. Rollout evaluations (F层)
    if isinstance(final_results[5], Exception):
        print(f"6. Rollout evaluations: Error - {str(final_results[5])}")
    else:
        print(f"6. Rollout evaluations: {final_results[5]}")
    
    print(f"\n=== Execution Summary ===")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    return final_results

if __name__ == "__main__":
    asyncio.run(main1())