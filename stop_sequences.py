"""
定义MCTS各个阶段的停止字符串序列和停止条件类
"""
import re
from transformers import StoppingCriteria, StoppingCriteriaList

# 初始化阶段的停止序列
INITIAL_STEP_STOPS = [
    "</step>",  # 基本的步骤结束标记
    "\n\n",     # 两个换行表示内容结束
    "[Next Step]", # 下一步标记
]

# 扩展阶段的停止序列
EXPAND_STEP_STOPS = [
    # "</step>",
    # "\n\n",
    # "[Next Step]",
    # "Let's approach", # 常见的转折句开头
    # "Now let's",
    "</think>",
]

# Rollout阶段的停止序列
ROLLOUT_STOPS = [
    # "$ \\boxed{",  # LaTeX答案框的开始
    # "\\boxed{",   # 可能的变体
    # "The answer is",  # 明确的答案标记
    # "Therefore,",    # 结论开始
    "</step> \n</step>",  
    "}$</step>",   
    "</step></step>",
    "</step>\n</step>",
    "</step> </step>",
    "</think>",
]

# 奖励评估阶段的停止序列
REWARD_STOPS = [
    # "[Score]:",     # 分数标记
    # "Score:",       # 分数标记的变体
    # "Rating:",      # 评分标记
    # "\n\n",        # 两个换行表示内容结束
    "</evaluation>"
]

PROCESS_EVALUATION_STOPS = [
    "</evaluation>"
]

# 多样化想法生成阶段的停止序列
DIVERSE_IDEAS_STOPS = [
    # "[Core idea]:",  # 核心想法标记
    # "Core idea:",    # 核心想法标记的变体
    # "\n\n",         # 两个换行表示内容结束
    # "[Next idea]",   # 下一个想法的标记
    "</ideas>",
]

# 直接推理阶段的停止序列
DIRECT_REASONING_STOPS = [
    # "\\boxed{",      # 最终答案开始标记
    # "The answer is", # 明确的答案标记
    "</think>",
]

# 成对比较评估阶段的停止序列
PAIR_EVALUATION_STOPS = [
    "</comparison>",  # 比较结果结束标记
    #"\n\n",          # 两个换行表示内容结束
    "[Analysis]:",   # 分析部分开始标记（作为备选停止点）
]

# 通用的停止序列，适用于所有阶段
COMMON_STOPS = [
    "Human:",       # 对话中的角色转换
    "Assistant:",   # 对话中的角色转换
    "<|im_end|>",  # 一些模型特定的结束标记
    "<|endoftext|>",
]

class StopOnTokens(StoppingCriteria):
    """
    用于控制文本生成停止的条件类
    """
    def __init__(self, tokenizer, stop_sequences, prompt_length=0):
        """
        初始化停止条件
        
        Args:
            tokenizer: 分词器实例
            stop_sequences: 停止序列列表
            prompt_length: prompt的token数量
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        """
        检查是否应该停止生成
        
        Args:
            input_ids: 当前生成的token IDs
            scores: token的分数
            **kwargs: 其他参数
            
        Returns:
            bool: 是否应该停止生成
        """
        # 只解码新生成的token部分
        if input_ids.shape[-1] > self.prompt_length:
            new_tokens = input_ids[0, self.prompt_length:]
            current_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            return False
        
        # 检查停止序列是否存在于新生成的文本中
        for stop_sequence in self.stop_sequences:
            if stop_sequence in current_text:
                #print(f"####found stop sequence: '{stop_sequence}'")
                #print(f"####new generated text: '{current_text}'")
                return True
        return False

class StopOnRegex(StoppingCriteria):
    """
    使用正则表达式匹配的停止条件类
    """
    def __init__(self, tokenizer, regex_patterns, prompt_length=0):
        """
        初始化正则表达式停止条件
        
        Args:
            tokenizer: 分词器实例
            regex_patterns: 正则表达式模式列表
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.regex_patterns = [re.compile(pattern) for pattern in regex_patterns]
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        """
        检查是否应该停止生成
        
        Args:
            input_ids: 当前生成的token IDs
            scores: token的分数
            **kwargs: 其他参数
            
        Returns:
            bool: 是否应该停止生成
        """
        # 解码当前生成的文本
        # 只解码新生成的token部分
        if input_ids.shape[-1] > self.prompt_length:
            new_tokens = input_ids[0, self.prompt_length:]
            current_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            return False
        
        # 检查是否匹配任何正则表达式模式
        for pattern in self.regex_patterns:
            if pattern.search(current_text):
                #print(f"####found pattern: '{pattern.pattern}'")
                #print(f"####current text: '{current_text}'")
                return True
        return False

def get_regex_patterns(stage: str) -> list:
    """
    根据MCTS的不同阶段返回对应的正则表达式模式
    
    Args:
        stage: MCTS的阶段名称
        
    Returns:
        list: 该阶段使用的正则表达式模式列表
    """
    # 新增：支持空/none阶段，直接返回空列表
    if not stage or stage == 'none' or stage == '':
        return []
    stage_regex = {
        'initial': [],
        'expand': [
            r'<\/step>.*?<.*?>',
        ],
        'rollout': [
            r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}',
        ],
        'reward': [],
        'process_evaluation': [],
        'diverse_ideas': [],
        'direct_reasoning': [
            r'\\\\boxed\{[^}]*\}',  # 匹配 \boxed{xxx} 格式
            r'\$\\\\boxed\{[^}]*\}\$',  # 匹配 $\boxed{xxx}$ 格式
        ],
        'pair_evaluation': [
        ],
    }
    
    return stage_regex.get(stage, [])

def get_stop_sequences(stage: str) -> list:
    """
    根据MCTS的不同阶段返回对应的停止序列
    
    Args:
        stage: MCTS的阶段名称，可选值：
               'initial', 'expand', 'rollout', 'reward', 'process_evaluation', 
               'diverse_ideas', 'direct_reasoning', 'pair_evaluation'
               或空/none代表不使用停止序列
                
    Returns:
        list: 该阶段使用的停止序列列表
    """
    # 新增：支持空/none阶段，直接返回空列表
    if not stage or stage == 'none' or stage == '':
        return []
    stage_stops = {
        'initial': INITIAL_STEP_STOPS,
        'expand': EXPAND_STEP_STOPS,
        'rollout': ROLLOUT_STOPS,
        'reward': REWARD_STOPS,
        'process_evaluation': PROCESS_EVALUATION_STOPS,
        'diverse_ideas': DIVERSE_IDEAS_STOPS,
        'direct_reasoning': DIRECT_REASONING_STOPS,
        'pair_evaluation': PAIR_EVALUATION_STOPS,
    }
    
    if stage not in stage_stops:
        raise ValueError(f"Unknown stage: {stage}. Available stages: {list(stage_stops.keys())}")
    
    # 合并特定阶段的停止序列和通用停止序列
    return stage_stops[stage] + COMMON_STOPS

# 最终调用的是这个函数
def get_stopping_criteria(tokenizer, stage: str, prompt_length: int = 0) -> StoppingCriteriaList:
    """
    根据MCTS的不同阶段返回对应的停止条件
    
    Args:
        tokenizer: 分词器实例
        stage: MCTS的阶段名称
        prompt_length: prompt的token数量
    """
    stop_sequences = get_stop_sequences(stage)
    regex_patterns = get_regex_patterns(stage)
    
    stopping_criteria = [StopOnTokens(tokenizer, stop_sequences, prompt_length)]
    
    if regex_patterns:
        stopping_criteria.append(StopOnRegex(tokenizer, regex_patterns, prompt_length))
    
    return StoppingCriteriaList(stopping_criteria)