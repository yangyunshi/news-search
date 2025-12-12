# backend/pipeline/search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import bm25s

def find_nearest_cluster(query_embedding, centroids):
    """
    Find the cluster whose centroid is closest to the query embedding.
    """
    print("Finding nearest cluster...")
    similarities = cosine_similarity(query_embedding, centroids)
    return int(np.argmax(similarities))

def search_in_cluster(query, query_embedding, cluster_indexes, centroids,
                      stemmer=None, top_k_results=10, news=None):
    """
    Search for relevant documents inside the nearest cluster using BM25.

    Args:
        query (str): User's search query (text).
        query_embedding (np.ndarray): Embedding vector for the query.
        cluster_indexes (dict): Mapping cluster_id -> (BM25 retriever, docs, indices).
        centroids (np.ndarray): Cluster centroids.
        stemmer: Optional stemmer for tokenization.
        top_k_results (int): Number of results to return from the cluster.
        news (pd.DataFrame, optional): DataFrame with Title/Description for mapping.

    Returns:
        list[dict]: Top search results with title/description and score.
    """
    print("Searching in cluster...")
    # Step 1: Find nearest cluster
    cluster_id = find_nearest_cluster(query_embedding, centroids)

    # Step 2: Unpack retriever + docs + indices
    retriever, docs, doc_indices = cluster_indexes[cluster_id]

    # Step 3: Tokenize query (string, not list)
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # Step 4: Run BM25 retrieval for top_k_results
    results, scores = retriever.retrieve(query_tokens, k=top_k_results)

    # Step 5: Build results list with scores
    output = []
    for i in range(results.shape[1]):
        row_idx = doc_indices[int(results[0, i])]
        if news is not None:
            title = news["Title"].iloc[row_idx]
            desc = news["Description"].iloc[row_idx]
        else:
            title = docs[int(results[0, i])]
            desc = title
        output.append({
            "rank": i + 1,
            "title": title,
            "description": desc,
            "score": float(scores[0, i]),
            "cluster": int(cluster_id)
        })

    return output
