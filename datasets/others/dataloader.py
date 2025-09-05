from typing import Any, List, Dict, Optional
import json

class DataLoader:
    """数据集加载器"""
    
    def __init__(self, dataset_path: str = "datasets"):
        self.dataset_path = dataset_path
    
    def load_dataset(self, dataset_name: str = "gsm8k") -> List[Dict[str, str]]:
        """
        根据数据集名称加载对应的数据集
        
        Args:
            dataset_name: 数据集名称
        Returns:
            包含question和answer的数据列表
        """
        dataset_name = dataset_name.lower()
        if dataset_name == "gsm8k":
            return self._get_gsm8k()
        elif dataset_name == "aime2024":
            return self._get_aime2024()
        elif dataset_name == "aime2025":
            return self._get_aime2025()
        else:
            return self._get_other()
    
    def _get_gsm8k(self) -> List[Dict[str, str]]:
        """加载GSM8K数据集"""
        dataset = []
        try:
            with open(f"{self.dataset_path}/GSM8K/test_all.json", 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for item in raw_data:
                solution = item["solution"]
                parts = solution.split("####")
                
                if len(parts) == 2:
                    process = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    process = solution
                    answer = solution
                    
                dataset.append({
                    "question": item["problem"],
                    "solution": process,
                    "answer": answer
                })
                
            return dataset
        except FileNotFoundError:
            print("Error: GSM8K dataset file not found")
            return []
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in GSM8K dataset file")
            return []

    def _get_aime2024(self) -> List[Dict[str, str]]:
        """
        加载AIME2024数据集
        从JSONL文件中读取，每行包含ID, Problem, Solution, Answer字段
        """
        dataset = []
        try:
            with open(f"{self.dataset_path}/AIME2024/aime_2024.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    dataset.append({
                        "question": item["Problem"],
                        "solution": item["Solution"],
                        "answer": item["Answer"]
                    })
                
            return dataset
        except FileNotFoundError:
            print("Error: AIME2024 dataset file not found")
            return []
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in AIME2024 dataset file")
            return []

    def _get_aime2025(self) -> List[Dict[str, str]]:
        """
        加载AIME2025数据集
        从JSONL文件中读取，每行包含question和answer字段
        """
        dataset = []
        try:
            with open(f"{self.dataset_path}/AIME2025/aime_2024.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    dataset.append({
                        "question": item["question"],
                        "solution": "",  # 空solution
                        "answer": item["answer"]
                    })
                
            return dataset
        except FileNotFoundError:
            print("Error: AIME2025 dataset file not found")
            return []
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in AIME2025 dataset file")
            return []
    
    def _get_other(self) -> List[Dict[str, str]]:
        """加载其他数据集"""
        raise NotImplementedError("Other dataset loading functions to be implemented")