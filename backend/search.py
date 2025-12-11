# backend/pipeline/search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_nearest_cluster(query_embedding, centroids):
    """
    Find the cluster whose centroid is closest to the query embedding.
    """
    print("Finding nearest cluster...")
    similarities = cosine_similarity([query_embedding], centroids)
    return int(np.argmax(similarities))

def search_in_cluster(query, query_embedding, cluster_indexes, centroids):
    """
    Search for relevant documents inside the nearest cluster using BM25.

    Args:
        query (str): User's search query (text).
        query_embedding (np.ndarray): Embedding vector for the query.
        cluster_indexes (dict): Mapping cluster_id -> BM25 index.
        centroids (np.ndarray): Cluster centroids.

    Returns:
        list: Top search results (documents).
    """
    print("Searching in cluster...")
    # Step 1: Find nearest cluster
    cluster_id = find_nearest_cluster(query_embedding, centroids)

    # Step 2: Run BM25 search inside that cluster
    index = cluster_indexes[cluster_id]
    results = index.search(query, k=5)  # top 5 results

    return results
