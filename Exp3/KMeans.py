import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
# Replace 'seeds_dataset.txt' with the actual path to your dataset
data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, header=None)
features = data.iloc[:, :-1].values  # First 7 columns as features
labels = data.iloc[:, -1].values  # Last column as labels


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
clusters, centroids = k_means(features, k)

# Step 3: Visualization (Dimensionality Reduction)
from sklearn.decomposition import PCA  # Use PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

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

#evaluate k-means
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ground truth labels (真实标签)
true_labels = labels  # Last column of the dataset

# Cluster labels from K-Means
predicted_labels = clusters  # Output from the K-Means algorithm

# 1. Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# 2. Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# 3. Purity Calculation
def calculate_purity(true_labels, predicted_labels):
    contingency_matrix = np.zeros((len(set(predicted_labels)), len(set(true_labels))))

    for pred_label, true_label in zip(predicted_labels, true_labels):
        contingency_matrix[pred_label - 1, true_label - 1] += 1  # Adjust index if needed

    max_in_clusters = contingency_matrix.max(axis=1)
    purity = np.sum(max_in_clusters) / len(true_labels)
    return purity

#purity = calculate_purity(true_labels, predicted_labels)
#print(f"Purity: {purity:.4f}")

