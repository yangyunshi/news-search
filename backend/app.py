# backend/app.py
import os
import joblib
import Stemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
from data import load_news
from embed import load_model, encode_titles
from cluster import run_kmeans
from index import build_cluster_indexes
from search import search_in_cluster

app = Flask(__name__)

# Enable CORS for your Next.js frontend (localhost:3000)
CORS(app, origins=["http://localhost:3000"])

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
    k = 25
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
    query_embedding = model.encode([query])[0]
    results = search_in_cluster(
        query,
        query_embedding,
        cluster_indexes,
        centroids,
        stemmer=Stemmer.Stemmer("english"),
        news=news
    )
    return jsonify({"query": query, "results": results})




@app.route("/rebuild", methods=["POST"])
def rebuild():
    global news, model, embeddings, labels, centroids, cluster_indexes

    print("Rebuilding pipeline...")
    news = load_news()
    model = load_model()
    embeddings = encode_titles(model, news["Title"])
    k = 25
    kmeans, labels, centroids = run_kmeans(embeddings, k)
    news["Cluster"] = labels
    stemmer = Stemmer.Stemmer("english")
    cluster_indexes = build_cluster_indexes(news, labels, k, stemmer=stemmer)
    joblib.dump((news, model, embeddings, labels, centroids, cluster_indexes), CACHE_FILE)

    print("Pipeline rebuilt ✅")
    return jsonify({"status": "Pipeline rebuilt successfully"})


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the News Search API 🚀",
        "endpoints": {
            "POST /search": "Search news by query",
            "POST /rebuild": "Rebuild pipeline cache"
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
