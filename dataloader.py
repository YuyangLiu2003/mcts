import json
import os
from typing import Dict, List, Optional

class DataLoader:
    def __init__(self, dataset_dir: str = "datasets"):
        """
        初始化DataLoader
        
        Args:
            dataset_dir: 数据集所在的目录路径
        """
        self.dataset_dir = dataset_dir
        
    def get_available_datasets(self) -> List[str]:
        """
        获取可用的数据集列表
        
        Returns:
            数据集名称列表（不包括扩展名）
        """
        if not os.path.exists(self.dataset_dir):
            return []
            
        # 获取所有json文件，不包括others目录
        datasets = []
        for file in os.listdir(self.dataset_dir):
            if file.endswith('.json') and os.path.isfile(os.path.join(self.dataset_dir, file)):
                datasets.append(file[:-5])  # 移除.json后缀
        return datasets
        
    def load_dataset(self, dataset_name: str) -> Optional[List[Dict]]:
        """
        加载指定的数据集
        
        Args:
            dataset_name: 数据集名称（不包括.json后缀）
            
        Returns:
            包含问题和答案的字典列表，如果加载失败则返回None
            每个字典包含'question'和'answer'键
        """
        file_path = os.path.join(self.dataset_dir, f"{dataset_name}.json")
        
        if not os.path.exists(file_path):
            print(f"Dataset file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            # 验证数据集格式
            if not isinstance(dataset, list):
                print(f"Invalid dataset format: expected a list, got {type(dataset)}")
                return None
                
            for item in dataset:
                if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                    print("Invalid dataset format: each item should be a dict with 'question' and 'answer' keys")
                    return None
                    
            return dataset
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON file: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
