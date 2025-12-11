# backend/pipeline/index.py
import bm25s
from bm25s import BM25

def build_cluster_indexes(news, labels, k, stemmer=None):
    """
    Build BM25 retrievers for each cluster.

    Args:
        news (DataFrame): Dataset with 'Title' column.
        labels (np.ndarray): Cluster labels for each document.
        k (int): Number of clusters.
        stemmer: Optional stemmer to use for tokenization.

    Returns:
        dict: Mapping cluster_id -> (BM25 retriever, raw docs)
    """
    print("Building cluster indexes...")
    cluster_indexes = {}
    for cluster_id in range(k):
        # Get all titles in this cluster
        cluster_titles = news[labels == cluster_id]["Title"].tolist()

        # Tokenize documents
        cluster_tokens = bm25s.tokenize(cluster_titles, stopwords="en", stemmer=stemmer)

        # Build BM25 retriever
        retriever = BM25()
        retriever.index(cluster_tokens)

        # Store retriever + raw docs
        cluster_indexes[cluster_id] = (retriever, cluster_titles)

    return cluster_indexes
