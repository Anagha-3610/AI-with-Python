from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=5,cluster_std=0.60, random_state=0)
# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=5)
y_agg = agg.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis', s=20)
plt.title("Agglomerative (Hierarchical) Clustering")
plt.show()

# Optional: Plot dendrogram
linked = linkage(X, 'ward')
dendrogram(linked, truncate_mode='lastp', p=12)
plt.title("Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()