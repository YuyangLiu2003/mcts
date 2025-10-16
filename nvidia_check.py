import os
import pynvml
import torch  # 以PyTorch为例，其他框架类似

# 初始化NVML
pynvml.nvmlInit()

# 查看当前设置的可见设备
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
print(f"当前CUDA_VISIBLE_DEVICES: {visible}")

# 打印程序中可见的设备及其对应系统编号和型号
if visible != "all":
    visible_ids = [int(id) for id in visible.split(",")]
else:
    visible_ids = list(range(pynvml.nvmlDeviceGetCount()))

for idx, sys_id in enumerate(visible_ids):
    handle = pynvml.nvmlDeviceGetHandleByIndex(sys_id)
    name = pynvml.nvmlDeviceGetName(handle).encode()
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"程序中编号 {idx} → 系统编号 {sys_id} → 型号 {name}")
    print(f"  总显存: {mem_info.total/1024**2:.2f} MiB")
    print(f"  已用显存: {mem_info.used/1024**2:.2f} MiB")
    print(f"  可用显存: {mem_info.free/1024**2:.2f} MiB\n")

# 验证PyTorch实际可用显存（如果用PyTorch）
if torch.cuda.is_available():
    print(f"PyTorch可见设备数: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"PyTorch设备 {idx}: {torch.cuda.get_device_name(idx)}")
        print(f"  总显存: {torch.cuda.get_device_properties(idx).total_memory/1024**2:.2f} MiB")

pynvml.nvmlShutdown()