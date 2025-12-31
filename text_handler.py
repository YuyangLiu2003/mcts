import json
import re
from typing import Dict, List, Any, Optional, Tuple

class SearchGuideHandler:
    def __init__(self, guide_file: str = "search_guide.json", dataset_name: str = None):
        """
        初始化SearchGuideHandler，负责管理和加载搜索指导配置
        
        Args:
            guide_file: 搜索指导配置文件的路径
            dataset_name: 数据集名称
        """
        self.guide_file = guide_file
        self.dataset_name = dataset_name
        self.guide_config = self._load_guide_config()
    
    def _load_guide_config(self) -> Dict[str, str]:
        """从JSON文件加载搜索指导配置"""
        try:
            with open(self.guide_file, 'r', encoding='utf-8') as f:
                guide_config = json.load(f)
                if self.dataset_name in guide_config:
                    return guide_config[self.dataset_name]
                else:
                    #print(f"Warning: {self.dataset_name} not found in {self.guide_file}, using default values")
                    return guide_config['default']
        except FileNotFoundError:
            print(f"Warning: {self.guide_file} not found, using default values")
            return None
        except Exception as e:
            print(f"Error loading {self.guide_file}: {e}")
            return None

    def get_init_expand_guidance(self) -> str:
        """获取初始扩展指导"""
        return self.guide_config.get("init_expand_guidance", "")

    def get_idea_guidance(self) -> str:
        """获取核心想法指导"""
        return self.guide_config.get("idea_guidance", "")
    
    def get_expand_guidance(self) -> str:
        """获取扩展指导"""
        return self.guide_config.get("expand_guidance", "")
    
    def get_process_criterions(self) -> str:
        """获取过程评价标准"""
        return self.guide_config.get("process_criterions", "")
    
    def get_reward_objectives(self) -> str:
        """获取奖励目标"""
        return self.guide_config.get("reward_objectives", "")
    
    def get_answer_restrictions(self) -> str:
        """获取答案限制"""
        return self.guide_config.get("answer_restrictions", "")
    
    def get_pairwise_criterions(self) -> str:
        """获取成对评价标准"""
        pairwise_criterions = self.guide_config.get("pairwise_criterions", "")
        if not pairwise_criterions:
            pairwise_criterions = self.get_process_criterions()
        return pairwise_criterions
    
    def get_all_config(self) -> Dict[str, str]:
        """获取所有配置"""
        return self.guide_config.copy()
    
    def reload_config(self):
        """重新加载配置文件"""
        self.guide_config = self._load_guide_config()

class PromptHandler:
    def __init__(self, template_file: str = "prompt_template_chat.json", tokenizer: Any = None, 
                 model_name: str = "", thinking_mode: bool = False):
        """
        初始化PromptHandler，负责管理和加载prompt模板
        
        Args:
            template_file: prompt模板文件的路径
            tokenizer: tokenizer对象
            model_name: 模型名称，用于检查thinking属性
            thinking_mode: thinking模式开关
        """
        self.template_file = template_file
        self.templates = self._load_templates()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.thinking_mode = thinking_mode
        self.thinking_attr_dict = {'Qwen3-8B': "enable_thinking"}  # 可以根据需要扩展
        self.thinking_attr_name = None
        
        # 检查是否支持thinking模式
        print("arg.model", self.model_name)
        if model_name and (model_name in self.thinking_attr_dict.keys()):
            attr_name = self.thinking_attr_dict[model_name]
            self.thinking_attr_name = attr_name
            print(f"模型 {model_name} 支持thinking模式，思考模式: {thinking_mode}")
        elif model_name:
            pass
            #print(f"模型 {model_name} 不在thinking_attr_dict中，将不会添加thinking参数")
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """从JSON文件加载模板"""
        with open(self.template_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        统一的apply_chat_template方法，处理thinking模式
        
        Args:
            messages: 消息列表
        Returns:
            处理后的prompt字符串
        """
        # 如果模型支持thinking属性，总是添加该参数，值为thinking_mode
        if self.thinking_attr_name:
            # 使用正确的参数传递方式
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                **{self.thinking_attr_name: self.thinking_mode}
            )
        else:
            # 正常调用，不添加thinking参数
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    
    def get_init_prompt(self, question: str, init_expand_guidance: str = "") -> str:
        """
        生成初始化步骤的prompt
        
        Args:
            question: 用户输入的问题
            init_expand_guidance: 初始扩展指导
        Returns:
            完整的prompt字符串
        """
        template = self.templates.get("initial_expand")
        if not template:
            raise ValueError("Initial expand template not found in template file")
            
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        # 替换用户问题占位符
        prompt = prompt.replace("[user question]", question)
        # 替换初始扩展指导占位符
        prompt = prompt.replace("[init_expand_guidance]", init_expand_guidance)
        
        return prompt

    def get_diverse_ideas_prompt(self, previous_steps: str, num_ideas: int, idea_guidance: str = "") -> str:
        """
        生成用于获取多样化核心想法的prompt
        
        Args:
            previous_steps: 之前的推理步骤
            idea_guidance: 核心想法指导
            num_ideas: 需要生成的核心想法数量
        Returns:
            完整的prompt字符串
        """
        template = self.templates.get("diverse_ideas")
        if not template:
            raise ValueError("Diverse ideas template not found in template file")
        
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        # 替换占位符
        prompt = prompt.replace("[previous steps]", previous_steps)
        prompt = prompt.replace("[idea guidance]", idea_guidance)
        prompt = prompt.replace("[num_ideas]", str(num_ideas))
        
        return prompt
    
    def get_expand_prompt(self, previous_steps: str, core_instruction: str = "", expand_guidance: str = "") -> str:
        """
        生成扩展步骤的prompt
        
        Args:
            previous_steps: 之前的推理步骤
            core_instruction: 该步骤的核心思路
            expand_guidance: 扩展指导
        Returns:
            完整的prompt字符串
        """
        template = self.templates.get("idea_expand")
        if not template:
            raise ValueError("Expand step template not found in template file")
            
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        # 替换占位符
        prompt = prompt.replace("[previous steps]", previous_steps)
        prompt = prompt.replace("[core instruction]", core_instruction)
        prompt = prompt.replace("[expand guidance]", expand_guidance)
        
        return prompt
    
    def get_rollout_prompt(self, previous_steps: str) -> str:
        template = self.templates.get("fast_rollout")
        if not template:
            raise ValueError("Fast Rollout template not found in template file")
            
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        # 替换占位符
        prompt = prompt.replace("[previous steps]", previous_steps)
        
        return prompt

    def get_reward_prompt(self, full_solution: str, objectives: str) -> str:
        """
        构建reward评估的prompt
        """
        template = self.templates.get("reward_evaluation")
        if not template:
            raise ValueError("Reward prompt template not found")
        
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        prompt = prompt.replace("[full solution]", full_solution)
        prompt = prompt.replace("[objectives]", objectives)
        return prompt

    def get_process_evaluation_prompt(self, previous_steps: str, current_step: str, process_criterions: str) -> str:
        """
        构建process evaluation的prompt
        """
        template = self.templates.get("process_evaluation")
        if not template:
            raise ValueError("Process evaluation template not found")
        
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])

        prompt = prompt.replace("[previous steps]", previous_steps)
        prompt = prompt.replace("[current step]", current_step)
        prompt = prompt.replace("[process criterions]", process_criterions)
        return prompt

    def get_pair_evaluation_prompt(self, previous_steps: str, stepA: str, stepB: str, pairwise_criterions: str) -> str:
        """
        构建成对比较评估的prompt
        
        Args:
            stepA: 步骤A的内容
            stepB: 步骤B的内容
            pairwise_criterions: 成对比较标准
        Returns:
            完整的prompt字符串
        """
        template = self.templates.get("pair_evaluation")
        if not template:
            raise ValueError("Pair evaluation template not found")
        
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])

        # 替换占位符
        prompt = prompt.replace("[previous steps]", previous_steps)  
        prompt = prompt.replace("[pairwise criterions]", pairwise_criterions)
        prompt = prompt.replace("[Step A]", stepA)
        prompt = prompt.replace("[Step B]", stepB)
        
        return prompt
        
    def get_best_answer_prompt(self, question: str, best_solution: str, answer_restrictions: str) -> str:
        """
        构建生成最终答案的prompt
        
        Args:
            question: 用户输入的问题
            best_solution: 最佳推理路径
            answer_restrictions: 答案限制
        Returns:
            完整的prompt字符串
        """
        template = self.templates.get("best_answer")
        if not template:
            raise ValueError("Best answer template not found")
        
        prefix = template.get("prefix", "")
        system_prompt = template.get("system_prompt", "")
        user_prompt = template.get("user_prompt", "")
        suffix = template.get("suffix", "")

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._apply_chat_template(messages)
        prompt = "".join([prefix, prompt, suffix])
        
        # 替换占位符
        prompt = prompt.replace("[best solution]", best_solution)
        prompt = prompt.replace("[answer restrictions]", answer_restrictions)
        
        return prompt

class ResponseHandler:
    def get_expand_step_init(self, response: str) -> List[str]:
        """
        从LLM的扩展步骤响应中提取下一步可能的推理步骤
        
        Args:
            response: LLM的响应文本
        Returns:
            可能的下一步推理步骤列表
        """
        response = "<step>\n[Core idea]:" + response
        # 删除</step>后面的内容
        if "</step>" in response:
            response = response.split("</step>")[0] + "</step>"
        # print("response:\n",response)
        return response
    
    def get_expand_step_backup(self, core_idea: str, response: str) -> str:
        """
        从LLM的扩展步骤响应中基于核心想法提取下一步推理步骤
        
        Args:
            core_idea: 用于引导扩展的核心想法
            response: LLM的响应文本
        Returns:
            格式化后的下一步推理步骤
        """
        # 处理[Check terminal]部分（包括各种大小写和格式变体）
        check_terminal_content = ""
        import re
        # 匹配各种Check terminal格式：[Check terminal], [Check Terminal], [check terminal], Check Terminal等
        check_terminal_pattern = r'(\[?[Cc]heck [Tt]erminal\]?|Check terminal|check terminal)'
        match = re.search(check_terminal_pattern, response)
        if match:
            # 分割响应，移除Check terminal及其后续内容
            split_pos = match.start()
            check_terminal_content = response[match.end():].strip()
            response = response[:split_pos].strip()
        
        # 格式化响应为标准步骤格式
        formatted_step = f"<step>\n[Core idea]: {core_idea}\n[Full step]: {response}"
        
        # 如果包含多个</step>，截取到第一个</step>
        """
            防止response中包含多个</step>和final answer的情况被误判为终止节点
        """
        if response.count('</step>') > 1:
            formatted_step = formatted_step.split('</step>')[0] + '</step>'
            return formatted_step
        
        # 检查是否是终止节点
        if r'\boxed{' in response or '</think>' in response or r'\boxed{' in check_terminal_content:
            # 如果boxed{answer}出现在check terminal中，将其添加到Full step末尾
            if r'\boxed{' in check_terminal_content and r'\boxed{' not in response:
                # 提取boxed内容
                import re
                boxed_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', check_terminal_content)
                if boxed_match:
                    boxed_content = f"\\boxed{{{boxed_match.group(1)}}}"
                    # 在</step>之前添加boxed内容
                    if formatted_step.endswith('</step>'):
                        formatted_step = formatted_step[:-7] + f"\n\nThe final answer to the question is {boxed_content}\n</step>"
                    else:
                        formatted_step += f"\n\nThe final answer to the question is {boxed_content}"
            
            # 确保步骤以</think>结尾
            if not formatted_step.endswith("</think>"):
                formatted_step += "\n</think>"
            
            if '</think>' in formatted_step:
                formatted_step = formatted_step.split('</think>')[0] + '</think>'
        else:
            # 确保步骤以</step>结尾
            if not formatted_step.endswith("</step>"):
                formatted_step += "\n</step>"
            # 如果不是终止节点，截取到 </step>
            if '</step>' in formatted_step:
                formatted_step = formatted_step.split('</step>')[0] + '</step>'
            
        return formatted_step

    def get_expand_step(self, core_idea: str, response: str) -> str:
        """
        主方法：净化 -> 解析 -> 提取 -> 组装
        """
        # --- 步骤 1: 主动净化 (核心修复点) ---
        # 在解析之前，先将原文中包含“Not terminal”的 \boxed{} 拆除
        # 变成普通文本，防止下游工具误判
        sanitized_response = self._sanitize_bad_boxes(response)

        # --- 步骤 2: 解析 ---
        full_step_content, check_terminal_content = self._parse_response(sanitized_response)
        
        # --- 步骤 3: 提取有效答案 ---
        final_answer = self._extract_final_answer(full_step_content, check_terminal_content)
        
        # --- 步骤 4: 组装 ---
        formatted_step = f"<step>\n[Core idea]: {core_idea}\n[Full step]: {full_step_content}"
        
        if final_answer:
            # 只有提取到了清洗后的有效答案，才封装进 \boxed
            formatted_step += f"\n\nThe final answer to the question is \\boxed{{{final_answer}}}"
        
        # --- 步骤 5: 清理标签 ---
        formatted_step = formatted_step.replace('</step>', '').strip() + "\n</step>"
        
        return formatted_step

    def _sanitize_bad_boxes(self, text: str) -> str:
        """
        核心防御机制：如果发现 \boxed{...} 里面是否定词，直接把 \boxed{}壳子 剥掉。
        例如：\boxed{Not terminal} -> Not terminal
        """
        # 匹配 boxed 的正则
        pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
        
        def replace_func(match):
            content = match.group(1)
            # 检查内容是否包含否定词
            if not self._is_valid_content(content):
                # 如果是无效内容（比如 Not terminal），只保留内容，去掉 \boxed
                return content
            # 如果是有效内容（比如数字或公式），保持原样
            return match.group(0)
            
        return re.sub(pattern, replace_func, text)

    def _extract_final_answer(self, full_text: str, check_text: str) -> Optional[str]:
        """提取逻辑保持不变，但现在处理的是干净的文本"""
        # 策略 1: 找 Boxed (此时剩下的 Boxed 应该都是有效的了，但为了保险依然检查)
        combined_text = f"{check_text} {full_text}" 
        boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', combined_text)
        
        for match in boxed_matches:
            if self._is_valid_content(match):
                return match.strip()

        # 策略 2: 救援模式 (找 "Answer is ...")
        if check_text and self._is_valid_content(check_text):
            rescue_pattern = r'(?i)(?:answer|result)\s*(?:is|=|:)\s*([^\.\n]+)'
            match = re.search(rescue_pattern, check_text)
            if match:
                candidate = match.group(1).strip()
                if self._is_valid_content(candidate):
                    return candidate
        return None

    def _is_valid_content(self, content: str) -> bool:
        """公共检查方法：判断内容是否有效（非空且不包含否定词）"""
        if not content: return False
        
        # 扩大了否定词库，涵盖更多模型可能用的词
        negative_keywords = [
            "not a terminal", "not terminal", "not the final", "no terminal",
            "continue", "process", "next step", "analyzing", "calculating",
            "fail", "error", "wait"
        ]
        
        c_lower = content.lower()
        # 如果包含任何否定词，则无效
        return not any(k in c_lower for k in negative_keywords)

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """切分响应文本"""
        split_pattern = r'(?i)(\[?\s*check\s*terminal\s*\]?)'
        parts = re.split(split_pattern, response, maxsplit=1)
        full_step = parts[0].strip()
        check_content = parts[2].strip() if len(parts) > 2 else ""
        return full_step, check_content

    def parse_diverse_ideas(self, response: str, num_ideas: int) -> list:
        """
        解析LLM生成的多样化核心想法
        Args:
            response: LLM输出的字符串
            num_ideas: 需要的核心想法数量
        Returns:
            核心想法列表
        """
        core_ideas = []
        pattern = r'\[(\d+)\]:\s*(.*?)(?=\[\d+\]:|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        for _, idea in matches:
            core_ideas.append(idea.strip())
        # 如果没有提取到足够的核心想法，则填充
        while len(core_ideas) < num_ideas:
            core_ideas.append(f"Alternative approach {len(core_ideas) + 1}")
        # 如果提取到的核心想法超过需要的数量，则截取
        core_ideas = core_ideas[:num_ideas]
        return core_ideas

    def parse_reward_response(self, response: str) -> int:
        """
        解析reward评估的分数
        """
        try:
            score_match = re.search(r'\[Score\]\s*:?\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(10, score))
            else:
                #print(f"Warning: Could not extract rollout score from response: {response}")
                score = 5.0
        except Exception as e:
            print(f"Error evaluating response: {e}")
            score = 5.0
        return score

    def parse_process_evaluation_response(self, response: str) -> int:
        """
        解析process evaluation的分数
        """
        try:
            score_match = re.search(r'\[Score\]\s*:?\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(10, score))
            else:
                #print(f"Warning: Could not extract process score from response: {response}")
                score = 5.0
        except Exception as e:
            print(f"Error evaluating process response: {e}")
            score = 5.0
        return score
    
    def parse_pair_evaluation_response(self, response: str) -> int:
        """
        解析成对比较评估的分数
        
        Args:
            response: LLM的比较响应文本
        Returns:
            比较分数（-5到5之间的整数）
        """
        try:
            # 简化匹配逻辑，直接查找关键模式
            which_match = re.search(r'\[\s*The\s+better\s+step\s*\][:：]?\s*(\w+)', response, re.IGNORECASE)
            score_match = re.search(r'\[\s*Score\s*\][:：]?\s*([+\-]?\d+(?:\.\d+)?)', response, re.IGNORECASE)
            
            # 确定胜者
            winner = None
            if which_match:
                which_text = which_match.group(1).lower()
                if which_text in ['a', 'step a', 'a is better']:
                    winner = 'A'
                elif which_text in ['b', 'step b', 'b is better']:
                    winner = 'B'
                elif which_text in ['tie', 'tied', 'equal']:
                    winner = 'Tie'
            
            # 如果没有明确的Which部分，回退到全文搜索
            if not winner:
                if re.search(r'(?:step\s+)?a\s+is\s+better', response, re.IGNORECASE):
                    winner = 'A'
                elif re.search(r'(?:step\s+)?b\s+is\s+better', response, re.IGNORECASE):
                    winner = 'B'
                else:
                    winner = 'Tie'  # 默认平局
            
            # 提取分数
            if score_match:
                raw_score = float(score_match.group(1))
                base_score = min(5, max(0, int(round(abs(raw_score)))))  # 限制在0-5范围内
                
                # 根据胜者确定分数正负
                if winner == 'A':
                    return base_score
                elif winner == 'B':
                    return -base_score
                else:
                    return 0
            else:
                # 没有分数时，根据胜者返回默认分数
                if winner == 'A':
                    return 3
                elif winner == 'B':
                    return -3
                else:
                    return 0
                    
        except Exception as e:
            print(f"Error parsing pair evaluation response: {e}")
            return 0  # 出错时默认平局

    def parse_best_answer(self, response: str) -> str:
        """
        解析最佳答案
        
        Args:
            response: LLM生成的最终答案响应文本
        Returns:
            提取出的答案字符串
        """
        # 完整响应文本，用于解析
        complete_response = "The final answer is: \\boxed{" + response
        
        try:
            # 尝试从响应中提取\\boxed{}格式的答案
            boxed_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', complete_response)
            if boxed_match:
                return boxed_match.group(1).strip()
            
            # 如果没有找到boxed格式，尝试其他常见的答案格式
            answer_patterns = [
                r'answer is[：:]?\s*([^\n]+)',
                r'the answer is[：:]?\s*([^\n]+)',
                r'final answer[：:]?\s*([^\n]+)',
            ]
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, complete_response, re.IGNORECASE)
                if matches:
                    return matches[-1].strip()
            
            # 如果都没找到，返回原始响应的前100个字符作为回退
            return response.strip()[:100]
            
        except Exception as e:
            print(f"Error parsing best answer: {e}")
            # 出错时返回原始响应的前100个字符
            return response.strip()[:100]
