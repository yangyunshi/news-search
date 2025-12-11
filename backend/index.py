# backend/pipeline/index.py
import bm25s
from bm25s import BM25

def build_cluster_indexes(news, labels, k, stemmer=None):
    """
    Build BM25 retrievers for each cluster.

    Args:
        news (DataFrame): Dataset with 'Title' and 'Description' columns.
        labels (np.ndarray): Cluster labels for each document.
        k (int): Number of clusters.
        stemmer: Optional stemmer to use for tokenization.

    Returns:
        dict: Mapping cluster_id -> (BM25 retriever, docs, indices)
              where docs are the raw text strings indexed by BM25,
              and indices are the corresponding DataFrame row indices.
    """
    print("Building cluster indexes...")
    cluster_indexes = {}
    for cluster_id in range(k):
        # Get all rows in this cluster
        cluster_rows = news[labels == cluster_id]

        # Use description (or title+description) as BM25 text
        cluster_docs = cluster_rows["Description"].tolist()
        cluster_indices = cluster_rows.index.tolist()

        # Tokenize documents
        cluster_tokens = bm25s.tokenize(cluster_docs, stopwords="en", stemmer=stemmer)

        # Build BM25 retriever
        retriever = BM25()
        retriever.index(cluster_tokens)

        # Store retriever + docs + indices
        cluster_indexes[cluster_id] = (retriever, cluster_docs, cluster_indices)

    return cluster_indexes
