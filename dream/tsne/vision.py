import torch

# 替换成你的 .pt 文件路径
path = "/home/zhangzhen/substring/substring/dream/erm_MSCN_0.5_1_5e-04_100.pkl.pt"

# 加载文件
data = torch.load(path, map_location="cpu")

# 打印前几个 batch 信息
for i, batch in enumerate(data[:1]):  # 只看第一个 batch，避免内存爆炸
    size, vec, label = batch
    print(f"Batch {i}")
    print(f"  Size shape: {size.shape}, sample: {size[:5]}")
    print(f"  Vector shape: {vec.shape}")
    print(f"  Label shape: {label.shape}, sample: {label[:5]}")
