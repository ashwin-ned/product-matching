# src/vector_db.py
"""
Initialize Pinecone client, upsert CLIP embeddings, and query vectors using the Pinecone SDK.
"""
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.openapi_support.exceptions import PineconeApiException
from mongodb_logger import log_event
from dotenv import load_dotenv

load_dotenv()
# Configuration: set these env vars in your shell or .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  
PINECONE_ENV     = os.getenv("PINECONE_ENV")      # e.g., "us-west1-gcp"
INDEX_NAME       = os.getenv("PINECONE_INDEX", "product-clip-index")
EMB_PATH         = os.path.join(os.path.dirname(__file__), "..", "embeddings", "product_clip_embeddings.npy")
VECTOR_DIM       = 512  # CLIP output dim

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


def create_index():
    """
    Create a Pinecone index if it doesn't exist.
    """
    try:
        existing = pc.list_indexes()
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created Pinecone index '{INDEX_NAME}'")
        else:
            print(f"Index '{INDEX_NAME}' already exists")
    except PineconeApiException as e:
        if e.status == 409:
            # Index already exists - No need to log this
            print(f"Index '{INDEX_NAME}' already exists in Pinecone")
        else:
            log_event("error", f"Failed to create index: {e}")
            raise e


def upsert_embeddings(batch_size: int = 100):
    """
    Load embeddings from local file and upsert to Pinecone in batches.
    """
    data = np.load(EMB_PATH, allow_pickle=True).item()  # {id: vector}
    index = pc.Index(INDEX_NAME)

    items = [(pid, vec.tolist()) for pid, vec in data.items()]
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted embeddings {i} to {i + len(batch)}")
    print("All embeddings upserted into Pinecone.")


def query_vector(query_vec: np.ndarray, top_k: int = 5, namespace: str = None):
    """
    Query Pinecone for nearest neighbors to query_vec.
    Returns list of (id, score).
    """
    try:
        index = pc.Index(INDEX_NAME)
        response = index.query(
            vector=query_vec.tolist(),
            top_k=top_k,
            include_values=False,
            include_metadata=False,
            namespace=namespace
        )
        matches = [(m.id, m.score) for m in response.matches]
        return matches
    except Exception as e:
        log_event("error", f"Failed to query vector: {e}")
        raise e


if __name__ == "__main__":
    create_index()
    upsert_embeddings()

    # Query example
    results = query_vector(np.random.rand(VECTOR_DIM), top_k=2)
    print("Query results:", results)
