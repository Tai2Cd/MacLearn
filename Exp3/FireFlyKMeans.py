import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# 数据集加载
def load_data():
    # 模拟数据或加载 Seeds 数据集
    data = np.random.rand(210, 7)  # 210 个样本，每个样本 7 个特征
    return data

# SSE 计算函数
def calculate_sse(data, centroids, labels):
    sse = 0
    for i, c in enumerate(centroids):
        cluster_points = data[labels == i]
        sse += np.sum((cluster_points - c) ** 2)
    return sse

# 萤火虫算法优化簇中心
def firefly_algorithm(data, k, num_fireflies=1, max_iter=1, alpha=0.2, beta=1.0, gamma=1.0):
    # 初始化萤火虫位置
    fireflies = [data[np.random.choice(data.shape[0], k, replace=False)] for _ in range(num_fireflies)]
    brightness = []

    for _ in range(max_iter):
        # 计算亮度 (SSE)
        brightness = [calculate_sse(data, f, np.argmin(pairwise_distances(data, f), axis=1)) for f in fireflies]
        sorted_indices = np.argsort(brightness)
        fireflies = [fireflies[i] for i in sorted_indices]

        # 更新萤火虫位置
        for i in range(num_fireflies):
            for j in range(i):
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                if brightness[j] < brightness[i]:  # 向亮度更高的萤火虫移动
                    fireflies[i] += beta * np.exp(-gamma * r**2) * (fireflies[j] - fireflies[i])
                    fireflies[i] += alpha * np.random.uniform(-1, 1, fireflies[i].shape)

    # 返回最佳簇中心
    best_centroids = fireflies[0]
    return best_centroids

# K-Means 聚类
def k_means(data, centroids, max_iter=100):
    for _ in range(max_iter):
        labels = np.argmin(pairwise_distances(data, centroids), axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(centroids.shape[0])])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# 主函数
data = load_data()
k = 3
best_centroids = firefly_algorithm(data, k)
final_centroids, final_labels = k_means(data, best_centroids)

# 可视化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.figure(figsize=(8, 6))

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=final_labels, cmap='viridis')
plt.scatter(pca.transform(final_centroids)[:, 0], pca.transform(final_centroids)[:, 1], c='red', marker='X')
plt.title('K-Means Clustering with Firefly Algorithm')
plt.show()

#非萤火虫
def k_means_new(data, k, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = data.shape

    # Randomly initialize centroids
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Assign samples to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return cluster_assignments, centroids


# Run K-Means with K=3
k = 3
clusters, centroids = k_means_new(data, k)

# Step 3: Visualization (Dimensionality Reduction)
from sklearn.decomposition import PCA  # Use PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(data)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(k):
    plt.scatter(
        reduced_features[clusters == cluster_id, 0],
        reduced_features[clusters == cluster_id, 1],
        label=f'Cluster {cluster_id + 1}'
    )

# Mark centroids
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=final_labels, cmap='viridis')
plt.scatter(pca.transform(final_centroids)[:, 0], pca.transform(final_centroids)[:, 1], c='red', marker='X')
plt.title('K-Means Clustering withOUT Firefly Algorithm')
plt.show()