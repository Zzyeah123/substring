import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
d1 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.9999_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')
d2 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.91_l2_1e-08_pat_5_clipGr_10.0_seed_1_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')
d3 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.55_l2_1e-08_pat_5_clipGr_10.0_seed_0_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')
d4 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.28_l2_1e-08_pat_5_clipGr_10.0_seed_1_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')
d5 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.19_l2_1e-08_pat_5_clipGr_10.0_seed_1_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')
d7 = pd.read_csv('/home/zhangzhen/substring/substring/dream/exp_result/DBLP/DREAM_DBLP_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_pVal_0.1_ptrain_0.01_l2_1e-08_pat_5_clipGr_10.0_seed_1_maxEpoch_100_maxD_3_pTest_0.2_bs_32/estimate_analysis.csv')

# Add a column to differentiate datasets
d1['dataset'] = 'd1'
d2['dataset'] = 'd2'
d3['dataset'] = 'd3'
d4['dataset'] = 'd4'
d5['dataset'] = 'd5'
#d6['dataset'] = 'd6'
d7['dataset'] = 'd7'

# Concatenate the dataframes
df = pd.concat([d1, d2, d3,d4,d5,d7], ignore_index=True)
# Define the bins for 'len' column
bins = [0, 4, 10, 15, 20]
labels = ['0-4', '5-10', '11-15', '16-20']

# Create a new column to store the 'len' groups
df['len_group'] = pd.cut(df['len'], bins=bins, labels=labels, right=True)
# Set the plot style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='len_group', y='q_err', hue='dataset', data=df, palette={'d1': 'red', 'd2': 'green', 'd3': 'blue', 'd4': 'orange', 
          'd5': 'purple', 'd6': 'brown', 'd7': 'pink'},whis=1000,showfliers=False)
# Add title and labels
plt.title('Boxplot of q_err by len group', fontsize=16)
plt.xlabel('Length Group', fontsize=14)
plt.ylabel('q_err', fontsize=14)

# Show the plot
plt.legend(title="Dataset")
plt.show()
plt.savefig('boxplot_q_err_by_len_group.png', dpi=300, bbox_inches='tight')  # Save the plot with high resolution
