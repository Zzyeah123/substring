import pickle
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 显式切换到当前脚本所在目录，确保相对路径读取正常
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def norm(x):

def gen_and_save(para: str, simple_complex):
    with open(para + ".pkl.pt", "rb") as file:
        data = torch.load(para + ".pkl.pt")
        d = {}
        for batch in data:
            size = batch[0]
            vec = batch[1]
            label = batch[2]
            for s, v in zip(size, vec):
                num = s.item()
                if num not in d.keys():
                    d[num] = []
                # print(v.shape)
                d[num].append(v)
    
    for k in d.keys():
        print(k, len(d[k]))
    
    simple_list = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    complex_list = [ 14, 15, 16, 17, 18, 19, 20]
    
    simple_vector = []
    complex_vector = []
    
    for k in simple_list:
        simple_vector += d[k]
    for k in complex_list:
        complex_vector += d[k]
    
    print(len(simple_vector), len(complex_vector))
    
    simple_tensor = torch.stack(simple_vector).cpu().detach().numpy()
    complex_tensor = torch.stack(complex_vector).cpu().detach().numpy()
    
    tensor_5 = torch.stack(d[5]).cpu().detach().numpy()
    tensor_6 = torch.stack(d[6]).cpu().detach().numpy()
    tensor_7 = torch.stack(d[7]).cpu().detach().numpy()
    tensor_8 = torch.stack(d[8]).cpu().detach().numpy()
    tensor_9 = torch.stack(d[9]).cpu().detach().numpy()
    tensor_10 = torch.stack(d[10]).cpu().detach().numpy()
    tensor_11 = torch.stack(d[11]).cpu().detach().numpy()
    tensor_12 = torch.stack(d[12]).cpu().detach().numpy()
    tensor_13 = torch.stack(d[13]).cpu().detach().numpy()
    tensor_14 = torch.stack(d[14]).cpu().detach().numpy()
    tensor_15 = torch.stack(d[15]).cpu().detach().numpy()
    tensor_16 = torch.stack(d[16]).cpu().detach().numpy()
    tensor_17 = torch.stack(d[17]).cpu().detach().numpy()
    tensor_18 = torch.stack(d[18]).cpu().detach().numpy()
    tensor_19 = torch.stack(d[19]).cpu().detach().numpy()
    tensor_20 = torch.stack(d[20]).cpu().detach().numpy()
    
    
    
    data = np.concatenate([tensor_5, tensor_6, tensor_7, tensor_9, tensor_9,tensor_10,tensor_11,tensor_12,tensor_13,tensor_14,tensor_15,tensor_16,tensor_17,tensor_18,tensor_19,tensor_20], axis=0)
    
    if simple_complex is False:
        labels = np.concatenate([ 
            np.full(tensor_5.shape[0], 5)   
        ])
    else:
        labels = np.concatenate([
            np.full(tensor_1.shape[0], 's'),  
            np.full(tensor_2.shape[0], 's'),  
            np.full(tensor_3.shape[0], 's'),  
            np.full(tensor_4.shape[0], 'c'),  
            np.full(tensor_5.shape[0], 'c')   
        ])
    
    tsne = TSNE(n_components=2, perplexity=25, random_state=22)
    low_dim_data = tsne.fit_transform(data)
    
    def norm(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min)
    
    low_dim_data = norm(low_dim_data)

    torch.save(low_dim_data, "ttt_tsne.pt")
    torch.save(labels, "ttt_tsne.labels.pt")
    return low_dim_data, labels

def draw_ttt_fig(para: str):

    LOAD_EXISTING_TSNE=False
    simple_complex=True
    
    """ Prepare data and label for drawing scatter figure 
    - low_dim_data: 
        a torch tensor in the shape of (6750, 2), referring to 6750 2-dimension points ( x, y )
        the value of low_dim_data should be normalized to (0, 1), by calling function norm(), 
            e.g., low_dim_data = norm(low_dim_data)
    - label: in the shape of (6750), referring to the label for 6750 points 
    """

    if LOAD_EXISTING_TSNE == True:
        low_dim_data,labels = torch.load( "ttt_tsne.pt", weights_only = False), torch.load("ttt_tsne.labels.pt", weights_only = False)
    else:
        low_dim_data,labels = gen_and_save(para, simple_complex)
    print(low_dim_data.shape, labels.shape)
    
    if simple_complex is False:
        label_map = {
            1: 'Table 1',
            2: 'Table 2',
            3: 'Table 3',
            4: 'Table 4',
            5: 'Table 5'
        }
        color_map = {
            1: '#7D70C4', # blue
            2: '#E8926D', # orange 
            3: '#c22f2f', # red
            4: '#83639f', # purple
            5: '#1f70a9' # blue
        }
    else:
        label_map = {
            's': 'Simple',
            'c': 'Complex'
        }
        color_map = {
            's': '#7D70C4', # blue
            'c': '#E8926D' # orange 
        }
        marker_map = {
            's': '^',
            'c': 'o'
        }
   
    plt.figure(figsize=(8, 6))
    
    for label, color in color_map.items():
        mask = labels == label
        legend = f'{label_map[label]}'
        plt.scatter(
            low_dim_data[mask, 0],        
            low_dim_data[mask, 1],        
            c=color,                      
            label=legend,      
            s=10,                         
            alpha=1.0,                     
            marker=marker_map[label]
        )
    
    plt.xticks([ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fontsize=15)
    plt.yticks([ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fontsize=15)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.grid(True,alpha=0.5)

    plt.legend(markerscale=3,fontsize=15,handletextpad=0.2,bbox_to_anchor=(-0.01, -0.01), loc=3)
    
    plt.savefig(para + ".pdf")
    
# draw_ttt_fig("ttt_MSCN_3_0.5_1_1e-03_100")
# draw_ttt_fig("ttt_MSCN_3_0.5_1_1e-04_100")
# draw_ttt_fig("ttt_MSCN_3_0.5_1_5e-04_100") #

# draw_ttt_fig("ttt_MSCN_5_0.5_1_1e-03_100")
# draw_ttt_fig("ttt_MSCN_5_0.5_1_1e-04_100")
draw_ttt_fig("ttt_MSCN_5_0.5_1_5e-04_100") # 

# draw_ttt_fig("ttt_MSCN_8_0.5_1_1e-03_100")
# draw_ttt_fig("ttt_MSCN_8_0.5_1_1e-04_100") # 
# draw_ttt_fig("ttt_MSCN_8_0.5_1_5e-04_100")   

