import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Setup
CHROMA_DIR = "rag_storage"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path="rag_storage")
collection = client.get_or_create_collection(name="tweets_rag")

def store_tweets_in_chroma(df):
    docs = df["text"].astype(str).tolist()
    metadata = df[["sentiment", "cyberbullying_type", "month"]].to_dict(orient="records")
    ids = [f"id_{i}" for i in range(len(docs))]
    
    # Clear existing data if needed
    # collection.delete(ids=collection.get()["ids"])

    # Split into batches
    batch_size = 256
    for i in range(0, len(docs), batch_size):
        batch_texts = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        collection.add(documents=batch_texts, metadatas=batch_metadata, ids=batch_ids)

