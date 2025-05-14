import pandas as pd
import os

# 定义文件路径模板
file_template = "/home/zhangzhen/substring/substring/dream/exp_result/{dname}/DREAM_{dname}_substrNum_5_complexRate_0.5_flag_1_lossCoff_1.0_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_{p_train}_l2_1e-08_pat_5_clipGr_10.0_seed_{seed}_maxEpoch_{epoch}_maxD_3_pTest_0.1_bs_32/estimate_analysis.csv"

# 定义参数范围
dname_range=["DBLP","GENE"]
  # substrNum 从 1 到 3
seed_range = [0,9]  # complexRate 的取值
ptrain_range = [0.01,0.09,0.18,0.35,0.7,1.0]  # lossCoff 的取值
epoch_range = [100,80]


# 定义分位数
percentiles = [1, 5, 25, 50, 75, 95, 99]

# 初始化结果存储
results = []

# 遍历 substrNum、complexRate 和 lossCoff 的所有组合
for dname in dname_range:
    for p_train in ptrain_range:
        for seed in seed_range:
            for epoch in epoch_range:
                # 构造文件路径
                file_path = file_template.format(dname=dname,p_train=p_train,seed=seed,epoch=epoch)
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                # 读取数据
                data = pd.read_csv(file_path)

                # 计算 est_count/true_count
                data["est_to_true"] = data["est_count"] / data["true_count"]

                # 标记 simple 和 complex
                data["category"] = data["len"].apply(lambda x: "simple" if 1 <= x <= 13 else ("complex" if 14 <= x <= 20 else "unknown"))

                # 按类别计算分位数
                overall_percentiles = {f"overall_{p}%": data["est_to_true"].quantile(p / 100) for p in percentiles}
                simple_percentiles = {f"simple_{p}%": data[data["category"] == "simple"]["est_to_true"].quantile(p / 100) for p in percentiles}
                complex_percentiles = {f"complex_{p}%": data[data["category"] == "complex"]["est_to_true"].quantile(p / 100) for p in percentiles}

                # 合并结果
                percentiles_result = {
                    **overall_percentiles,
                    **simple_percentiles,
                    **complex_percentiles,
                    "dname":dname,
                    "epoch": epoch,
                    "ptrain": p_train,
                    "seed":seed
                }
                results.append(percentiles_result)

# 转换为 DataFrame 并保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("percentiles_analysis_results_ptrain_ord.csv", index=False)
results_df.to_excel("percentiles_analysis_results_ptrain_ord.xlsx", index=False)

print("Results saved to percentiles_analysis_results_with_ptrain.csv and percentiles_analysis_results_with_ptrain.xlsx")
