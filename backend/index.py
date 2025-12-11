# backend/pipeline/index.py
import bm25s

def build_cluster_indexes(news, labels, k):
    """
    Build BM25 indexes for each cluster.

    Args:
        news (DataFrame): Dataset with 'Title' column.
        labels (np.ndarray): Cluster labels for each document.
        k (int): Number of clusters.

    Returns:
        dict: Mapping cluster_id -> BM25 index
    """
    print("Building cluster indexes...")
    cluster_indexes = {}
    for cluster_id in range(k):
        # Get all titles in this cluster
        cluster_titles = news[labels == cluster_id]["Title"].tolist()

        # Build BM25 index
        index = bm25s.Index()
        index.fit(cluster_titles)
        cluster_indexes[cluster_id] = index

    return cluster_indexes
