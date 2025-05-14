import os,sys
os.chdir(sys.path[0]) #使用文件所在目录
import yaml
import argparse
import torch
import numpy as np
from types import SimpleNamespace
from src.preprocess import load_query_strings, get_cardinalities_train_test_valid
import src.util as ut
from src.model_torch import DREAMEstimator
from src.estimator import DBFactory
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ========== 1. 加载配置 ==========
conf_path = "/home/zhangzhen/substring/substring/dream/model/DBLP/DREAM_DBLP_substrNum_3_complexRate_0.5_flag_1_lossCoff_1.0_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_1.0_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.1_bs_32/conf.yaml"
with open(conf_path, "r") as f:
    conf_dict = yaml.safe_load(f)

# 保存原始 data 配置（DBFactory 需要 data 中的 "name" 字段）
data_config_orig = conf_dict.get("data", {})

# 提取 "alg" 与 "data" 部分，用于生成测试数据
alg_config = conf_dict.get("alg", {})
data_config = conf_dict.get("data", {}).copy()  # 复制一份用于生成测试数据

# 避免重复：如果 data_config 中存在 "name"，则将其重命名为 "dname"（用于生成测试数据）
if "name" in data_config:
    data_config["dname"] = data_config.pop("name")

# 设置 alg_config 的默认值，确保包含 prfx 与 Eprfx
alg_config.setdefault("prfx", False)
alg_config.setdefault("Eprfx", False)

# 合并配置（用于生成测试数据时）
merged_config = {**alg_config, **data_config}
args = argparse.Namespace(**merged_config)
for attr in ["analysis", "latency", "prfx", "Eprfx"]:
    if not hasattr(args, attr):
        setattr(args, attr, False)
# 补充缺失的参数，例如 p_val 与 p_test
if not hasattr(args, "p_val"):
    args.p_val = 0.1
if not hasattr(args, "p_test"):
    args.p_test = 0.05

# 将 alg_config 转换为支持属性访问的对象，供 DREAMEstimator 使用
alg_config_ns = SimpleNamespace(**alg_config)

# ========== 2. 生成训练/验证/测试数据 ==========
split_seed = 0
query_strings = load_query_strings(args.dname, seed=split_seed)
q_train, q_valid, q_test = ut.get_splited_train_valid_test_each_len(query_strings, split_seed, args)
train_data, valid_data, test_data = get_cardinalities_train_test_valid(q_train, q_valid, q_test, args)

# ========== 3. 构造模型实例并加载权重 ==========
estimator = DREAMEstimator(alg_config_ns)
# 注意：DBFactory 需要 data 中原始的配置（包含 "name" 字段），因此用 SimpleNamespace 包装 data_config_orig
estimator.set_db_factory(DBFactory(SimpleNamespace(**data_config_orig)))
# 指定模型权重保存路径（你已有的 saved_model.pth）
estimator.save_path = "/home/zhangzhen/substring/substring/dream/model/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.55_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.2_bs_32/saved_model.pth"
# 设置日志目录（build() 中会检查 logdir 不为空）
estimator.logdir = "./log_test"
# build() 方法需要 loss_coefficient 与 num_substrings，与训练时一致
loss_coefficient = alg_config_ns.loss_coefficient    # 例如 0.5
num_substrings = alg_config_ns.substrings_num          # 例如 4
# 调用 build()，如果 saved_model.pth 存在且 over_write=False，则直接加载权重
estimator.build(loss_coefficient, num_substrings, train_data, valid_data, test_data, over_write=False)

# ========== 4. 对测试数据进行预测 ==========
with torch.no_grad():
    y_pred, y_true = estimator.estimate(test_data)
print("Predicted cardinalities:", y_pred)
print("True cardinalities:", y_true)

# ========== 5. 提取测试数据隐藏向量并保存，然后用 t-SNE 可视化 ==========

import torch
from pathlib import Path

def extract_and_save_with_size(estimator, raw_test_data, output_path="dream_pre_dblp.pkl.pt"):
    """
    提取隐藏向量 + size(x_len)，并保存为符合 tsne 绘图要求的 pt 文件。
    每个 batch 结构为 (size, vec, label)，其中：
    - size: torch.Tensor (batch_size,)，表示每个输入序列长度
    - vec: torch.Tensor (batch_size, hidden_dim)，表示最终隐藏表示
    - label: torch.Tensor (batch_size, 1)，这里暂时使用预测输出或真实值
    """
    # 获取编码后的测试数据
    x_test, y_test = estimator._encode_data_test(raw_test_data)
    dl_test = estimator.dataloader(x_test, y_test, shuffle=False)

    model = estimator.model
    model.eval()
    device = estimator.device

    all_data = []

    for batch in dl_test:
        (x_seq, x_d, x_len), y_batch = batch
        x_seq = x_seq.to(device)
        x_d = x_d.to(device)
        x_len = x_len.to(device)
        y_batch = y_batch.to(device)

        # 获取 embedding
        embed = model.embedding(x_seq, x_d)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, x_len.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = model.rnn(packed)

        if not model.seq_out:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = torch.transpose(hidden, 0, 1)
            hidden = hidden[:, -1, :]
        else:
            padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            hidden = torch.stack([padded[i, l-1, :] for i, l in enumerate(x_len)], dim=0)

        all_data.append((x_len.cpu(), hidden.cpu().detach(), y_batch.cpu().detach()))

    torch.save(all_data, Path(output_path))
    print(f"✅ Saved tsne format data to: {output_path}")

# 提取隐藏向量
extract_and_save_with_size(estimator, test_data, output_path="dream_pre_dblp_0.5_1_5e-04_100.pkl.pt")

# 保存隐藏向量到文件
