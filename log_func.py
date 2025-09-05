import os
import time
import argparse

# 全局变量
log_file = None
tree_log_file = None
timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加全局时间戳
args = None  # 全局参数引用

def set_global_args(global_args):
    """设置全局参数引用"""
    global args
    args = global_args
    
def get_global_args():
    """获取全局参数引用"""
    global args
    return args

def test_out(output_title, output_content="", case_idx=None, dataset_name=None):

    """
    保存调试信息到日志文件，不打印到控制台
    """
    global log_file, timestamp
    
    # 构建输出内容
    output = f"{'-'*30}{output_title}{'-'*30}\n{output_content}\n"
    
    # 保存到日志文件，使用新格式的目录名
    if case_idx is not None and dataset_name is not None:
        args = get_global_args()
        log_dir = os.path.join("logs", f"{dataset_name}_{timestamp}_{args.case_start}_{args.case_end}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 为每个case创建新的日志文件
        if log_file is None or not log_file.endswith(f"case_{case_idx:04d}.txt"):
            log_file = os.path.join(log_dir, f"case_{case_idx:04d}.txt")
        
        # 追加写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(output)
