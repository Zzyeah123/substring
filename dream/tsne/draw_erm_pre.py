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
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def gen_and_save(para, simple_complex):
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
    
    simple_list = [ 1, 2, 3 ]
    complex_list = [ 4 , 5 ]
    
    simple_vector = []
    complex_vector = []
    
    for k in simple_list:
        simple_vector += d[k]
    for k in complex_list:
        complex_vector += d[k]
    
    print(len(simple_vector), len(complex_vector))
    
    simple_tensor = torch.stack(simple_vector).detach().numpy()
    complex_tensor = torch.stack(complex_vector).detach().numpy()
    
    tensor_1 = torch.stack(d[1]).detach().numpy()
    tensor_2 = torch.stack(d[2]).detach().numpy()
    tensor_3 = torch.stack(d[3]).detach().numpy()
    tensor_4 = torch.stack(d[4]).detach().numpy()
    tensor_5 = torch.stack(d[5]).detach().numpy()
    
    data = np.concatenate([tensor_1, tensor_2, tensor_3, tensor_4, tensor_5], axis=0)
    
    if simple_complex is False:
        labels = np.concatenate([
            np.full(tensor_1.shape[0], 1),  
            np.full(tensor_2.shape[0], 2),  
            np.full(tensor_3.shape[0], 3),  
            np.full(tensor_4.shape[0], 4),  
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
    
    # normalized 
    
    low_dim_data = norm(low_dim_data)
    
    # torch.save(low_dim_data, para + ".pt")
    # torch.save(labels, para + ".labels.pt")
    torch.save(low_dim_data,  "erm_tsne.pt")
    torch.save(labels, "erm_tsne.labels.pt")
    return low_dim_data,labels

LOAD_EXISTING_TSNE=False
simple_complex=True



""" Prepare data and label for drawing scatter figure 
- low_dim_data: 
    a torch tensor in the shape of (6750, 2), referring to 6750 2-dimension points ( x, y )
    the value of low_dim_data are normalized to (0, 1)
- label: in the shape of (6750), referring to the label for 6750 points 
"""
if LOAD_EXISTING_TSNE == True:
    low_dim_data, labels = torch.load("erm_tsne.pt", map_location=torch.device('cpu'), weights_only = False), \
                                torch.load("erm_tsne.labels.pt", map_location=torch.device('cpu'), weights_only = False)
else:
    low_dim_data, labels = gen_and_save("erm_MSCN_0.5_1_5e-04_100", simple_complex)
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
        1: '#449945', # green
        2: '#ea7927', # orange 
        3: '#c22f2f', # red
        4: '#83639f', # purple
        5: '#1f70a9' # blue
    }
    marker_map = {
        1: '^',
        2: 'o',
        3: 'v',
        4: '-',
        5: 'x'
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

plt.legend(markerscale=3,fontsize=15,handletextpad=0.2,bbox_to_anchor=(-0.01, -0.01), loc=3)
plt.xticks([ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fontsize=15)
plt.yticks([ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fontsize=15)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.grid(True,alpha=0.5)
plt.savefig('erm_MSCN_0.5_1_5e-04_100.pdf')
