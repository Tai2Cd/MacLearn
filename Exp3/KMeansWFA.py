import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
# Replace 'seeds_dataset.txt' with the actual path to your dataset
def load_data():
    # 模拟数据或加载 Seeds 数据集
    data = np.random.rand(210, 7)  # 210 个样本，每个样本 7 个特征
    return data
data = load_data()


# Step 2: K-Means implementation
def k_means(data, k, max_iters=100):
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
clusters, centroids = k_means(data, k)

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
centroids_2d = pca.transform(centroids)
#plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100, label='Centroids')

plt.title('K-Means Clustering of Seeds Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()



#purity = calculate_purity(true_labels, predicted_labels)
#print(f"Purity: {purity:.4f}")

