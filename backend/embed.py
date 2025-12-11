# backend/pipeline/embed.py
from sentence_transformers import SentenceTransformer

def load_model(model_name="multi-qa-mpnet-base-dot-v1"):
    print("Loading model...")
    return SentenceTransformer(model_name)

def encode_titles(model, titles):
    print("Encoding Titles...")
    return model.encode(titles)
