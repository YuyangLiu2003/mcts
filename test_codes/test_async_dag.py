import asyncio
import time
from typing import List, Any, Callable
from aiodag import task

class IdeaProcessor:
    def __init__(self, num_ideas: int = 3, rollout_num: int = 3):
        self.num_ideas = num_ideas
        self.rollout_num = rollout_num
    
    # --- 通用工具 Task (核心修改) ---

    @task
    async def _map_tasks(self, func: Callable, inputs: List[Any], *args, **kwargs) -> List[Any]:
        """
        通用的一对多映射任务 (Map)。
        将 func 应用于 inputs 列表中的每一个元素，并并行执行。
        支持传递额外参数 (*args, **kwargs) 给 func。
        """
        # 注意：aiodag 会自动等待 inputs 依赖完成，这里 inputs 是实际列表
        tasks = [func(item, *args, **kwargs) for item in inputs]
        return await asyncio.gather(*tasks)

    @task
    async def _flat_map_tasks(self, func: Callable, nested_inputs: List[List[Any]]) -> List[Any]:
        """
        通用的嵌套列表展平映射任务 (FlatMap)。
        处理 List[List] 结构，将其展平后并行执行 func。
        专用于处理类似 rollout 这种嵌套结构。
        """
        tasks = []
        for sublist in nested_inputs:
            for idx, item in enumerate(sublist):
                # 适配原逻辑：将 item 和 idx (在子列表中的索引) 传给 func
                tasks.append(func(item, idx))
        return await asyncio.gather(*tasks)

    # --- 业务逻辑 Task (保持不变) ---

    @task
    async def _generate_diverse_core_ideas(self, context: str, num_ideas: int) -> List[str]:
        print(f"Generating {num_ideas} core ideas based on context: {context}")
        await asyncio.sleep(1)
        ideas = [f"idea_{i}" for i in range(num_ideas)]
        print(f"Generated core ideas: {ideas}")
        return ideas
    
    @task
    async def _idea_expand(self, new_idea: str) -> str:
        print(f"Expanding idea: {new_idea}")
        await asyncio.sleep(0.5)
        expanded = f"expanded_{new_idea}"
        print(f"Expanded idea: {expanded}")
        return expanded
    
    @task
    async def _process_evaluation(self, full_step: str) -> float:
        print(f"Processing evaluation for: {full_step}")
        await asyncio.sleep(0.3)
        process_score = 1.0
        print(f"Evaluation result: {process_score}")
        return process_score
    
    @task
    async def _pairwise_evaluation(self, full_steps: List[str]) -> List[float]:
        # D层：依赖完整列表进行对比，不走 map
        print(f"Performing pairwise evaluation on: {full_steps}")
        await asyncio.sleep(1)
        scores = [i * 0.1 for i in range(len(full_steps))]
        print(f"Pairwise evaluation scores: {scores}")
        return scores
    
    @task
    async def _rollout_paths(self, full_step: str, rollout_num: int) -> List[str]:
        print(f"Generating {rollout_num} rollout paths for: {full_step}")
        await asyncio.sleep(0.8)
        paths = [f"rollout_{i}" for i in range(rollout_num)]
        print(f"Generated rollout paths: {paths}")
        return paths

    @task
    async def _rollout_evaluation(self, rollout_result: str, rollout_idx: int) -> float:
        print(f"Evaluating rollout result {rollout_idx}: {rollout_result}")
        await asyncio.sleep(0.2)
        rollout_score = 1.0
        print(f"Rollout evaluation result: {rollout_score}")
        return rollout_score


async def main1():
    start_time = time.time()
    
    processor = IdeaProcessor(num_ideas=3, rollout_num=3)
    context = "User requirement analysis"
    
    # 1. A层：核心想法
    core_ideas_task = processor._generate_diverse_core_ideas(context, processor.num_ideas)
    
    # 2. B层：使用通用 _map_tasks 替代 _create_expand_tasks
    # 逻辑：对 core_ideas_task 结果中的每一项，执行 _idea_expand
    full_steps_task = processor._map_tasks(
        processor._idea_expand, 
        core_ideas_task
    )
    
    # 3. C层：使用通用 _map_tasks 替代 _create_process_eval_tasks
    process_evaluations_task = processor._map_tasks(
        processor._process_evaluation, 
        full_steps_task
    )
    
    # 4. D层：直接传入列表任务 (保持原样，因为它是 List-to-List 的整体处理，不是 map)
    pairwise_scores_task = processor._pairwise_evaluation(full_steps_task)
    
    # 5. E层：使用通用 _map_tasks 替代 _create_rollout_path_tasks
    # 注意：这里需要传递 rollout_num 参数，通过 kwargs 传入
    rollout_paths_list_task = processor._map_tasks(
        processor._rollout_paths, 
        full_steps_task, 
        rollout_num=processor.rollout_num
    )
    
    # 6. F层：使用通用 _flat_map_tasks 替代 _create_rollout_eval_tasks
    # 逻辑：处理嵌套列表 [[p1, p2], [p3, p4]] -> 展平 -> 执行 _rollout_evaluation
    rollout_evaluations_task = processor._flat_map_tasks(
        processor._rollout_evaluation,
        rollout_paths_list_task
    )
    
    # 等待所有结果
    final_results = await asyncio.gather(
        core_ideas_task,
        full_steps_task,
        process_evaluations_task,
        pairwise_scores_task,
        rollout_paths_list_task,
        rollout_evaluations_task,
        return_exceptions=True
    )
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n=== All Task Results ===")
    labels = [
        "Core ideas", "Expanded ideas", "Process evaluations", 
        "Pairwise evaluation scores", "Rollout paths list", "Rollout evaluations"
    ]
    
    for i, res in enumerate(final_results):
        status = f"Error - {str(res)}" if isinstance(res, Exception) else res
        print(f"{i+1}. {labels[i]}: {status}")
    
    print(f"\n=== Execution Summary ===")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    return final_results

if __name__ == "__main__":
    asyncio.run(main1())