import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F
import time
from stop_sequences import get_stopping_criteria
import random
import asyncio
import re
import os

class use_transformers:
    def __init__(self, model_name='models/Meta-Llama-3.1-8B-Instruct', device="cuda", trust_remote_code=True, local_files_only=True, use_stop=False):
        self.model_name = model_name
        self.device = device
        # Suppress tokenizer loading warnings
        logging.set_verbosity_error()
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        self.model.eval()  # Set the model to evaluation mode
        # 新增：控制是否使用停止序列/正则
        self.use_stop = use_stop

    def generate(self, prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, stop_stage=None, skip_special_tokens=True):
        """
        直接使用给定的 prompt 生成文本，添加终止条件
        
        Args:
            stop_stage: MCTS阶段名称，用于自动获取停止条件
        """
        print("###start generating: ", stop_stage)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = model_inputs.input_ids.shape[-1]
        
        # 创建停止条件
        stopping_criteria = None
        if stop_stage and self.use_stop:
            # 使用stop_stage自动获取停止条件
            stopping_criteria = get_stopping_criteria(self.tokenizer, stop_stage, prompt_length = prompt_length)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                stopping_criteria=stopping_criteria
            )
            generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        
        response = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=skip_special_tokens, 
            truncate_before_pattern=[r"\n\n^#", "^'''$", "\n\n\n"]
        )
        
        return response

class use_AiHubMix:
    def __init__(self, model_name, api_key="sk-wcNZCZttk93aGyeB21740aF1F4574f56A3621c1a4c5d4b4e", use_stop=True):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://aihubmix.com/v1"
        )
        self.model = model_name
        # 保持接口一致
        self.use_stop = use_stop

    def generate(self, prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, stop_stage=None, skip_special_tokens=True):
        """添加generate方法以保持接口一致性"""
        # 由于API调用，我们仍然使用chat方法，但添加stop_stage支持
        return self.chat(user_prompt=prompt, temperature=temperature)

    def chat(self, user_prompt="", system_prompt="", temperature=0.7, top_p=0.9):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            if not messages:
                messages.append({"role": "user", "content": ""})

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                top_p=top_p,
            )
            
            # 使用 getattr 安全地获取 reasoning_content
            reasoning = getattr(chat_completion.choices[0].message, 'reasoning_content', None)
            content = chat_completion.choices[0].message.content

            if reasoning:
                response = f'<think>\n{reasoning}</think>\n{content}'
            else:
                response = content
                
            return response
        except openai.error.APIError as e:
            print(f"API Error occurred: {e}")
            return None

class use_vLLM:
    def __init__(self, model_name='models/Meta-Llama-3.1-8B-Instruct', device="cuda", max_model_len=8192, trust_remote_code=True,
                 tensor_parallel_size=1, use_stop=False):
        """
        初始化vLLM模型
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型，默认为cuda
            max_model_len: 模型最大长度
            trust_remote_code: 是否信任远程代码
            tensor_parallel_size: 张量并行大小，用于多GPU部署
        """
        try:
            from vllm import LLM, SamplingParams
            from vllm.outputs import RequestOutput
            # 导入PyTorch分布式相关模块（关键）
            import torch.distributed as dist
        except ImportError as e:
            if "torch.distributed" in str(e):
                raise ImportError("PyTorch distributed module not found. Please install PyTorch with distributed support.")
            else:
                raise ImportError(f"Required package not found: {e}. Install with 'pip install vllm torch'.")
            
        self.model_name = model_name
        self.device = device
        self.max_model_len = max_model_len
        # 保存分布式模块引用，用于后续销毁
        self.dist = dist
        
        # 保存vLLM特有的类引用
        self.SamplingParams = SamplingParams
        
        # 初始化vLLM模型
        print(f"Loading vLLM model: {model_name}")
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            dtype="float16",  # 已适配Tesla V100的计算能力7.0
        )
        
        # 获取tokenizer以便与stop_sequences.py兼容
        self.tokenizer = self.model.get_tokenizer()
        
        # 存储常用的停止序列
        self.stop_sequences_cache = {}
        # 新增：控制是否使用停止序列/正则
        self.use_stop = use_stop

    def __del__(self):
        """
        析构函数：程序结束时自动调用，销毁分布式进程组
        """
        # 检查分布式进程组是否已初始化（避免重复销毁或未初始化的错误）
        if self.dist.is_initialized():
            try:
                # 销毁分布式进程组
                self.dist.destroy_process_group()
                print("Successfully destroyed PyTorch distributed process group.")
            except Exception as e:
                print(f"Warning: Failed to destroy process group: {e}")

    def generate(self, prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, stop_stage=None, skip_special_tokens=True):
        """
        直接使用给定的 prompt 生成文本，添加终止条件
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            do_sample: 是否使用采样
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            stop_stage: MCTS阶段名称，用于自动获取停止条件。只能通过token来stop，正则表达式是后裁剪
            skip_special_tokens: 是否跳过特殊token
        """
        print("###start generating with vLLM: ", stop_stage)
        
        # 设置采样参数
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        
        # 创建采样参数
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,  # 匹配use_transformers的设置
        )
        
        # 生成文本
        outputs = self.model.generate(prompt, sampling_params)
        
        # 提取生成的文本
        response = outputs[0].outputs[0].text
        
        return response

    def batch_generate(self, prompt, max_new_tokens=500, n=1, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, stop_stage=None, skip_special_tokens=True):
        """
        直接使用给定的 prompt 生成文本，添加终止条件
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            do_sample: 是否使用采样
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            stop_stage: MCTS阶段名称，用于自动获取停止条件。只能通过token来stop，正则表达式是后裁剪
            skip_special_tokens: 是否跳过特殊token
        """
        print("###start generating with vLLM: ", stop_stage)
        
        # 设置采样参数
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        
        # 创建采样参数
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,  # 匹配use_transformers的设置
            n=n,
        )
        
        # 生成文本
        outputs = self.model.generate(prompt, sampling_params)
        
        # 修改：提取所有生成的文本并返回列表
        texts = [output.text for output in outputs[0].outputs]
        
        return texts

class use_async_vLLM:
    def __init__(self, model_name='models/Meta-Llama-3.1-8B-Instruct', device="cuda", max_model_len=8192, 
                 trust_remote_code=True, tensor_parallel_size=None, gpu_memory_utilization=0.9, max_num_seqs=256, use_stop=False):
        """
        初始化异步vLLM模型引擎
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型，默认为cuda
            max_model_len: 模型最大长度
            trust_remote_code: 是否信任远程代码
            tensor_parallel_size: 张量并行大小，用于多GPU部署，默认使用所有可用GPU
            gpu_memory_utilization: GPU内存利用率
            max_num_seqs: 最大并发序列数
        """
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from vllm.sampling_params import SamplingParams
            from vllm.outputs import RequestOutput
            import torch.distributed as dist
        except ImportError as e:
            if "torch.distributed" in str(e):
                raise ImportError("PyTorch distributed module not found. Please install PyTorch with distributed support.")
            else:
                raise ImportError(f"Required package not found: {e}. Install with 'pip install vllm torch'.")
            
        self.model_name = model_name
        self.device = device
        self.max_model_len = max_model_len
        self.dist = dist
        
        # 保存vLLM特有的类引用
        self.SamplingParams = SamplingParams
        
        # 确定张量并行大小，默认为可用GPU数量
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        
        # 初始化异步vLLM引擎
        print(f"Loading asynchronous vLLM model: {model_name}")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 获取tokenizer以便与stop_sequences.py兼容
        # self.tokenizer = asyncio.run(self.engine.get_tokenizer())
        
        # 存储常用的停止序列
        self.stop_sequences_cache = {}
        # 新增：控制是否使用停止序列/正则
        self.use_stop = use_stop

    # 2. 新增异步方法，专门用于获取 tokenizer（在主线程异步上下文调用）
    async def initialize_tokenizer(self):
        """必须在主线程的异步函数中调用，直接用 await 获取 tokenizer"""
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = await self.engine.get_tokenizer()  # 直接 await，无循环问题
            print("Tokenizer initialized successfully")

    def __del__(self):
        """
        析构函数：程序结束时自动调用，销毁分布式进程组
        """
        # 检查分布式进程组是否已初始化（避免重复销毁或未初始化的错误）
        if hasattr(self, 'dist') and self.dist.is_initialized():
            try:
                # 销毁分布式进程组
                self.dist.destroy_process_group()
                print("Successfully destroyed PyTorch distributed process group.")
            except Exception as e:
                print(f"Warning: Failed to destroy process group: {e}")

    async def generate(self, prompt, confidence_tag=False, stop_stage=None, **kwargs):
        # 设置默认参数
        defaults = {
            "max_tokens": 1024,
            "n": 1,
            "temperature": 0.7,
            "skip_special_tokens": True,
            "seed": None,
            "repetition_penalty": 1.1,
            "logprobs": 1,
        }
        
        # 合并默认参数和传入的kwargs（传入的参数会覆盖默认值）
        params = {** defaults, **kwargs}
        #print(f"###start asynchronous generating with vLLM (batch): {stop_stage}")
        
        sampling_kwargs = params
        
        sampling_params = self.SamplingParams(** sampling_kwargs)
        
        request_id = f"req-{asyncio.get_running_loop().time()}"
        results = []
        completed_outputs = {}  # 存储已完成的序列，key为index
        
        async for request_output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(request_output)
            
            # 检查每个输出是否完成
            for output in request_output.outputs:
                if output.finish_reason is not None:  # 序列已完成
                    completed_outputs[output.index] = output
        
        # 处理最终结果
        if len(completed_outputs) == params["n"]:
            # 所有序列都已完成
            final_outputs = [completed_outputs[i] for i in range(params["n"])]
        else:
            # 使用最后一个request_output中的结果
            final_output = results[-1] if results else None
            if final_output:
                final_outputs = final_output.outputs
            else:
                final_outputs = []
        
        # 确保我们有n个输出 (防御性编程)
        if len(final_outputs) < params["n"]:
            print(f"Warning: Expected {params['n']} outputs, but got {len(final_outputs)}")
        
        # --- 优化核心：统一数据处理 ---
        
        # 1. 提取文本 (永远是列表)
        texts = [o.text for o in final_outputs]

        # 2. 根据 confidence_tag 决定返回逻辑
        if confidence_tag:
            # 只有需要置信度时才进行计算，节省资源
            cumulative_logprob = [o.cumulative_logprob for o in final_outputs]
            token_counts = [len(o.token_ids) for o in final_outputs]
            confidences = [sum_p / count if count > 0 else 0.0 for sum_p, count in zip(cumulative_logprob, token_counts)]
            
            # 返回两个列表：([str...], [float...])
            return texts, confidences
        
        else:
            # 只返回文本列表：[str...]
            return texts

class use_debug:
    def __init__(self, model_name=None):
        self.model_name = model_name


    def generate(self, prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, stop_stage=None, skip_special_tokens=True):
        """返回预设的调试响应"""
        pass

    def message_chat(self, messages, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, top_k=20):
        """返回预设的调试响应"""
        pass

async def async_test():
    async_vllm = use_async_vLLM(model_name="../models/Meta-Llama-3.1-8B-Instruct")
    await async_vllm.initialize_tokenizer()
    start_time = time.time()
    responses, confidences = await async_vllm.generate(prompt=r"""PhD is hard to get.""", max_tokens=100, n=4, stop_stage="rollout", confidence_tag=True, seed=None)
    print(confidences)
    for res in responses:
        print(res+"\n**************")
        #print(res)
    # 计算并显示运行时间
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")

def sync_test():
    LLM = use_vLLM('../models/Meta-Llama-3.1-8B-Instruct')
    responses = LLM.batch_generate(
        r"PhD is hard to get.",
        temperature=0.7, stop_stage='rollout',max_new_tokens=100, n=3
    )
    #print(responses)
    print("haha: ",len(responses))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    asyncio.run(async_test())
    #sync_test()

"""
if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()
    
    LLM = use_vLLM('/root/data1/hesaikenew/data/ted_LLM/llama3/models/Meta-Llama-3.1-8B-Instruct')
    #LLM.api_mode(chat_mode='multi')
    #LLM=HubLLM("aihubmix-Llama-3-3-70B-Instruct")
    #response=LLM.chat(r"PhD is hard to get.",temperature=0.7,max_new_tokens=50)
    #response=LLM.generate(r"PhD is hard to get.",temperature=0.7)
    response = LLM.generate(
        r"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        temperature=0.7, stop_stage='expand'
    )
    
    print(response)
    
    # 计算并显示运行时间
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")
"""