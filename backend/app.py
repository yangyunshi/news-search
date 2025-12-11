# backend/app.py
import os
import joblib
import Stemmer
from flask import Flask, request, jsonify
from data import load_news
from embed import load_model, encode_titles
from cluster import run_kmeans
from index import build_cluster_indexes
from search import search_in_cluster

app = Flask(__name__)

CACHE_FILE = "pipeline_cache.pkl"

# ----------------------------
# Load or build pipeline
# ----------------------------
print("Initializing system...")

if os.path.exists(CACHE_FILE):
    print("Loading cached pipeline...")
    news, model, embeddings, labels, centroids, cluster_indexes = joblib.load(CACHE_FILE)
else:
    print("Building pipeline from scratch...")
    # Step 1: Load dataset
    news = load_news()

    # Step 2: Load model
    model = load_model()

    # Step 3: Encode titles
    embeddings = encode_titles(model, news["Title"])

    # Step 4: Clustering
    k = 10
    kmeans, labels, centroids = run_kmeans(embeddings, k)
    news["Cluster"] = labels

    # Step 5: Indexing
    stemmer = Stemmer.Stemmer("english")
    cluster_indexes = build_cluster_indexes(news, labels, k, stemmer=stemmer)

    # Save everything to cache
    joblib.dump((news, model, embeddings, labels, centroids, cluster_indexes), CACHE_FILE)

print("System ready ✅")


# ----------------------------
# API endpoints
# ----------------------------
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    # Embed query (fast, since model is already loaded)
    query_embedding = model.encode([query])[0]

    # Search inside nearest cluster
    results = search_in_cluster(query, query_embedding, cluster_indexes, centroids)

    return jsonify({"query": query, "results": results})


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """
    Rebuild the pipeline and overwrite the cache.
    Useful if the dataset has changed or you want to refresh everything.
    """
    global news, model, embeddings, labels, centroids, cluster_indexes

    print("Rebuilding pipeline...")
    # Step 1: Load dataset
    news = load_news()

    # Step 2: Load model
    model = load_model()

    # Step 3: Encode titles
    embeddings = encode_titles(model, news["Title"])

    # Step 4: Clustering
    k = 10
    kmeans, labels, centroids = run_kmeans(embeddings, k)
    news["Cluster"] = labels

    # Step 5: Indexing
    stemmer = Stemmer.Stemmer("english")
    cluster_indexes = build_cluster_indexes(news, labels, k, stemmer=stemmer)

    # Save everything to cache
    joblib.dump((news, model, embeddings, labels, centroids, cluster_indexes), CACHE_FILE)

    print("Pipeline rebuilt ✅")
    return jsonify({"status": "Pipeline rebuilt successfully"})


if __name__ == "__main__":
    app.run(debug=True)
