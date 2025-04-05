import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
# Replace 'seeds_dataset.txt' with the actual path to your dataset
data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, header=None)
features = data.iloc[:, :-1].values  # First 7 columns as features
labels = data.iloc[:, -1].values  # Last column as labels


# Step 2: Hierarchical clustering (Agglomerative)
def hierarchical_clustering(data, k):
    """
    Perform  hierarchical clustering on the dataset.
    Parameters:
        data (numpy.ndarray): Dataset features.
        k (int): Number of clusters to reduce to.
    Returns:
        cluster_assignments (list): Cluster assignments for each data point.
    """
    n_samples = data.shape[0]
    # Initial clusters: each sample is its own cluster
    cluster_assignments = {i: [i] for i in range(n_samples)}
    distances = calculate_distances(data)

    while len(cluster_assignments) > k:
        # Find the closest clusters
        c1, c2 = find_closest_clusters(distances,cluster_assignments)
        # Merge the two closest clusters
        cluster_assignments = merge_clusters(cluster_assignments, c1, c2)
        # Update distances
        distances = update_distances(data, distances, c1, c2, cluster_assignments)

    # Map points to final clusters
    final_clusters = {}
    for cluster_id, points in enumerate(cluster_assignments.values()):
        for point in points:
            final_clusters[point] = cluster_id

    return [final_clusters[i] for i in range(n_samples)]


def calculate_distances(data):
    """Calculate initial distance matrix."""
    n_samples = data.shape[0]
    distances = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = np.linalg.norm(data[i] - data[j])
    return distances


def find_closest_clusters(distances, cluster_assignments):
    """Find the indices of the two closest clusters."""
    min_dist = np.inf
    closest_clusters = (-1, -1)
    active_clusters = list(cluster_assignments.keys())
    for i in active_clusters:
        for j in active_clusters:
            if i < j and distances[i, j] < min_dist:
                min_dist = distances[i, j]
                closest_clusters = (i, j)
    return closest_clusters


def merge_clusters(cluster_assignments, c1, c2):
    if c1 in cluster_assignments and c2 in cluster_assignments:
        """Merge clusters c1 and c2."""
        cluster_assignments[c1].extend(cluster_assignments[c2])
        del cluster_assignments[c2]
    return cluster_assignments


def update_distances(data, distances, c1, c2, cluster_assignments):
    """Update the distance matrix after merging clusters."""
    for i in range(distances.shape[0]):
        if i != c1 and i in cluster_assignments:
            new_distance = np.mean([np.linalg.norm(data[p1] - data[p2])
                                    for p1 in cluster_assignments[c1]
                                    for p2 in cluster_assignments[i]])
            distances[c1, i] = distances[i, c1] = new_distance
    distances[c1, c2] = distances[c2, c1] = np.inf
    return distances


# Step 3: Perform clustering
k = 3  # Set desired number of clusters
cluster_assignments = hierarchical_clustering(features, k)

# Step 4: Visualize the results (dimensionality reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
for cluster_id in range(k):
    plt.scatter(
        reduced_features[np.array(cluster_assignments) == cluster_id, 0],
        reduced_features[np.array(cluster_assignments) == cluster_id, 1],
        label=f'Cluster {cluster_id + 1}'
    )

plt.title('Hierarchical Clustering of Seeds Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#evaluate
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ground truth labels (真实标签)
true_labels = labels  # Last column of the dataset

# Cluster labels from
predicted_labels = cluster_assignments# Output from the layer algorithm

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
