# backend/pipeline/cluster.py
import numpy as np
from sklearn.cluster import KMeans

def run_kmeans(embeddings: np.ndarray, k: int = 25, random_state: int = 42):
    """
    Run KMeans clustering on embeddings.

    Args:
        embeddings (np.ndarray): Embedding vectors for all documents.
        k (int): Number of clusters.
        random_state (int): Random seed for reproducibility.

    Returns:
        kmeans (KMeans): Trained KMeans model.
        labels (np.ndarray): Cluster labels for each document.
        centroids (np.ndarray): Cluster centroids.
    """
    print("Running kmeans...")
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return kmeans, labels, centroids
