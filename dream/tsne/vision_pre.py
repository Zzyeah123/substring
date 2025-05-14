import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ========== 1. 从 hidden_vectors.npy 文件读取数据 ==========
hidden_vectors = np.load('/home/zhangzhen/substring/substring/dream/hidden_vectors.npy')
print("Loaded hidden vectors from hidden_vectors.npy")

# ========== 2. 用 t-SNE 进行降维 ==========
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(hidden_vectors)

# ========== 3. t-SNE 可视化并保存为图片 ==========
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', s=10)
plt.title("t-SNE Visualization of Test Data Hidden Vectors")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# 保存 t-SNE 图像
plt.savefig('tsne_visualization_from_saved_vectors.png')  # 保存图片为 .png 文件
plt.show()  # 显示图片
