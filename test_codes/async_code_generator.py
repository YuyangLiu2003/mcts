import re
from dataclasses import dataclass
from typing import List, Dict, Optional

# 定义任务元数据，用于辅助代码生成和类型推导
@dataclass
class TaskMeta:
    name: str
    input_type: str  # 'Single', 'List', 'List[List]'
    output_type: str # 'Single', 'List', 'List[List]'
    # 模拟任务逻辑的代码（可选，用于生成能运行的Demo）
    mock_impl: str = "return f'{name}_result'" 

class DAG2Async:
    def __init__(self, dag_str: str, tasks: List[TaskMeta]):
        self.raw_dag = dag_str
        self.tasks_meta = {t.name: t for t in tasks}
        self.adj_list = {}
        self.reverse_adj = {} # 子指向父，用于查找依赖
        self.roots = []
        self._parse_dag()
        
    def _parse_dag(self):
        """解析简单的文本DAG格式"""
        lines = self.raw_dag.strip().split('\n')
        nodes = set()
        
        for line in lines:
            line = line.strip()
            if line.startswith("root:"):
                # 处理 root: a
                root_name = line.split(":")[1].strip()
                self.roots.append(root_name)
                nodes.add(root_name)
            elif " to " in line:
                # 处理 a to b
                # 兼容 "edges: a to b" 或直接 "a to b"
                clean_line = line.replace("edges:", "").strip()
                parts = clean_line.split(" to ")
                if len(parts) == 2:
                    u, v = parts[0].strip(), parts[1].strip()
                    if u not in self.adj_list: self.adj_list[u] = []
                    self.adj_list[u].append(v)
                    self.reverse_adj[v] = u # 简化版：假设是树状结构或单亲节点
                    nodes.add(u)
                    nodes.add(v)
        
        self.all_nodes = nodes

    def _validate(self):
        """检查DAG节点是否都在任务列表中定义"""
        defined_tasks = set(self.tasks_meta.keys())
        diff = self.all_nodes - defined_tasks
        if diff:
            raise ValueError(f"DAG中存在未定义的任务: {diff}")
        print(">> Validation Passed: DAG structure matches Task Definitions.")

    def _topological_sort(self) -> List[str]:
        """简单的拓扑排序，决定代码生成顺序"""
        in_degree = {node: 0 for node in self.all_nodes}
        for u in self.adj_list:
            for v in self.adj_list[u]:
                in_degree[v] += 1
        
        queue = [node for node in self.all_nodes if in_degree[node] == 0]
        sorted_nodes = []
        
        while queue:
            u = queue.pop(0)
            sorted_nodes.append(u)
            if u in self.adj_list:
                for v in self.adj_list[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        return sorted_nodes

    def generate_code(self) -> str:
        self._validate()
        sorted_nodes = self._topological_sort()
        
        # 跟踪运行时数据的形状 (Runtime Shape)
        # key: task_name, value: 'Single' | 'List' | 'List[List]'
        # 初始状态：Root 节点的输出形状由其定义决定
        data_shapes = {}
        
        # 代码构建器
        code_lines = []
        
        # 1. 头部引用
        code_lines.append("import asyncio")
        code_lines.append("from typing import List, Any, Callable")
        code_lines.append("from aiodag import task")
        code_lines.append("")
        
        # 2. 类定义
        code_lines.append("class AutoGenProcessor:")
        code_lines.append("    def __init__(self):")
        code_lines.append("        pass")
        code_lines.append("")
        
        # 3. 插入通用的 Map/FlatMap 工具函数 (直接硬编码)
        code_lines.append(self._get_utility_methods())
        
        # 4. 生成具体的 Task 方法
        for t_name in sorted_nodes:
            meta = self.tasks_meta[t_name]
            # 构造函数签名
            sig_in = "inputs: List[Any]" if meta.input_type != 'Single' else "input_data: Any"
            sig_out = "List[Any]" if meta.output_type != 'Single' else "Any"
            
            code_lines.append(f"    @task")
            code_lines.append(f"    async def {t_name}(self, {sig_in}) -> {sig_out}:")
            code_lines.append(f"        print(f'Running task: {t_name}')")
            code_lines.append(f"        await asyncio.sleep(0.1)")
            code_lines.append(f"        # Logic from meta: {meta.mock_impl}")
            # 生成简单的模拟返回值
            if meta.output_type == 'List':
                code_lines.append(f"        return [f'{t_name}_res_{{i}}' for i in range(2)]")
            elif meta.output_type == 'List[List]':
                code_lines.append(f"        return [[f'{t_name}_p{{i}}_{{j}}' for j in range(2)] for i in range(2)]")
            else:
                code_lines.append(f"        return f'{t_name}_result'")
            code_lines.append("")

        # 5. 生成 Main 逻辑 (最核心的拼接部分)
        code_lines.append("async def main():")
        code_lines.append("    p = AutoGenProcessor()")
        code_lines.append("    print('=== Starting Pipeline ===')")
        
        # 生成调用链
        for node in sorted_nodes:
            meta = self.tasks_meta[node]
            
            # Case A: Root Node
            if node in self.roots:
                code_lines.append(f"    # Task {node} (Root)")
                code_lines.append(f"    task_{node} = p.{node}('initial_context')")
                data_shapes[node] = meta.output_type
                continue
            
            # Case B: Non-Root Node
            parent = self.reverse_adj.get(node)
            if not parent:
                continue # Should not happen in connected graph
            
            parent_shape = data_shapes[parent]
            child_input_req = meta.input_type
            
            code_lines.append(f"    # Task {node} (Parent: {parent}, Shape: {parent_shape} -> Req: {child_input_req})")
            
            # --- 核心模式匹配逻辑 ---
            
            # 1. Map模式: 父级给List，子级要Single -> 用 _map_tasks
            if parent_shape == 'List' and child_input_req == 'Single':
                code_lines.append(f"    task_{node} = p._map_tasks(p.{node}, task_{parent})")
                # 这里的输出形状会被 map 改变：如果子任务输出 Single，结果就是 List[Single]
                # 如果子任务输出 List，结果就是 List[List]
                new_shape = 'List' if meta.output_type == 'Single' else 'List[List]'
                data_shapes[node] = new_shape
                
            # 2. FlatMap模式: 父级给List[List]，子级要Single -> 用 _flat_map_tasks
            elif parent_shape == 'List[List]' and child_input_req == 'Single':
                code_lines.append(f"    task_{node} = p._flat_map_tasks(p.{node}, task_{parent})")
                # flat map 通常将嵌套展平并执行，返回结果集
                # 假设 flat_map 后返回的是 List (所有子任务结果的集合)
                new_shape = 'List' 
                data_shapes[node] = new_shape
            
            # 3. Reduce/Aggregate模式: 父级给List，子级要List -> 直接传
            elif parent_shape == 'List' and child_input_req == 'List':
                 code_lines.append(f"    task_{node} = p.{node}(task_{parent})")
                 data_shapes[node] = meta.output_type
                 
            # 4. Simple模式: 父级给Single，子级要Single -> 直接传
            elif parent_shape == 'Single' and child_input_req == 'Single':
                code_lines.append(f"    task_{node} = p.{node}(task_{parent})")
                data_shapes[node] = meta.output_type
            
            else:
                code_lines.append(f"    # WARNING: Type mismatch or unimplemented pattern: {parent_shape} -> {child_input_req}")
                code_lines.append(f"    task_{node} = p.{node}(task_{parent})")
                data_shapes[node] = meta.output_type

        # 6. Gather results
        leaf_nodes = [n for n in self.all_nodes if n not in self.adj_list]
        code_lines.append("")
        code_lines.append("    # Wait for execution")
        # 收集所有变量名
        all_task_vars = [f"task_{n}" for n in sorted_nodes]
        code_lines.append(f"    results = await asyncio.gather({', '.join(all_task_vars)})")
        
        code_lines.append("    for i, res in enumerate(results):")
        code_lines.append("        print(f'Result {i}: {res}')")

        code_lines.append("")
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    asyncio.run(main())")
        
        return "\n".join(code_lines)

    def _get_utility_methods(self):
        """返回预制的通用工具代码"""
        return """
    @task
    async def _map_tasks(self, func: Callable, inputs: List[Any], *args, **kwargs) -> List[Any]:
        # 通用 Map 逻辑
        tasks = [func(item, *args, **kwargs) for item in inputs]
        return await asyncio.gather(*tasks)

    @task
    async def _flat_map_tasks(self, func: Callable, nested_inputs: List[List[Any]]) -> List[Any]:
        # 通用 FlatMap 逻辑
        tasks = []
        for sublist in nested_inputs:
            for item in sublist:
                tasks.append(func(item))
        return await asyncio.gather(*tasks)
"""

# ================= 使用示例 =================

# 1. 定义任务列表 (模拟用户输入)
# 我们定义一个类似 IdeaProcessor 的流程
# A(Root) -> generates List
# B(Expand) -> takes Single, returns Single (Needs Map)
# C(Eval) -> takes Single, returns Single (Needs Map)
# D(Pairwise) -> takes List, returns List (Aggregate)
# E(Rollout) -> takes Single, returns List (Needs Map, creates List[List])
# F(RollEval) -> takes Single (Needs FlatMap from E)

task_definitions = [
    TaskMeta(name="a", input_type="Single", output_type="List"),       # Generate Ideas
    TaskMeta(name="b1", input_type="Single", output_type="Single"),    # Expand Idea 1
    TaskMeta(name="b2", input_type="Single", output_type="Single"),    # Expand Idea 2 (并行分支演示)
    TaskMeta(name="c1", input_type="Single", output_type="Single"),    # Eval b1
    TaskMeta(name="d", input_type="List",   output_type="List"),       # Pairwise (Aggregate)
    TaskMeta(name="e", input_type="Single", output_type="List"),       # Rollout (Generates nested)
    TaskMeta(name="f", input_type="Single", output_type="Single"),     # Rollout Eval (Process nested)
]

# 2. 定义 DAG 结构
dag_input = """
root: a
edges: a to b1
edges: a to b2
edges: b1 to c1
edges: b1 to d
edges: b1 to e
edges: e to f
"""

# 3. 执行生成
generator = DAG2Async(dag_input, task_definitions)
generated_code = generator.generate_code()

print("---------------- 生成的代码如下 ----------------")
print(generated_code)