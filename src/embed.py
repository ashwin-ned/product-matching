# src/embed.py
"""
Compute CLIP embeddings for product and query images and save as NumPy files.
Inspired by Pinecone’s CLIP image-search tutorial.
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
PRODUCT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
QUERY_IMAGES_DIR   = os.path.join(os.path.dirname(__file__), "..", "query_images")
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "..", "embeddings")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model + processor
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

def embed_image(image_path: str) -> np.ndarray:
    """
    Load an image, preprocess it, and return a normalized CLIP embedding vector.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    # L2-normalize for cosine similarity
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype(np.float32)

def embed_directory(image_dir: str) -> dict:
    """
    Embed all images in a directory. Filenames without extension are keys.
    Returns: { image_id: embedding }
    """
    embeddings = {}
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_id = os.path.splitext(fname)[0]
        path = os.path.join(image_dir, fname)
        embeddings[image_id] = embed_image(path)
        print(f"Embedded {image_id}")
    return embeddings

def save_embeddings(emb_dict: dict, filename: str):
    """
    Save embedding dict to a .npy file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    np.save(out_path, emb_dict)
    print(f"Saved {len(emb_dict)} embeddings to {out_path}")

def read_embeddings(filename: str) -> dict:
    """
    Load embeddings from a .npy file.
    """
    emb_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"File {emb_path} does not exist.")
    return np.load(emb_path, allow_pickle=True).item()

if __name__ == "__main__":
    # 1. Embed product images
    prod_embs = embed_directory(PRODUCT_IMAGES_DIR)
    save_embeddings(prod_embs, "product_clip_embeddings.npy")

    # 2. Embed query images
    query_embs = embed_directory(QUERY_IMAGES_DIR)
    save_embeddings(query_embs, "query_clip_embeddings.npy")

    # 3. Load and verify embeddings
    loaded_prod_embs = read_embeddings("../embeddings/product_clip_embeddings.npy")
    loaded_query_embs = read_embeddings("../embeddings/query_clip_embeddings.npy")
    print(f"Loaded {len(loaded_prod_embs)} product embeddings.")
    print(f"Loaded {len(loaded_query_embs)} query embeddings.")
    # Example: print first 3 product IDs and their embeddings
    for pid, emb in list(loaded_prod_embs.items())[:3]:
        print(f"Product ID: {pid}, Embedding: {emb[:5]}...")  # Print first 5 values
    # Example: print first 3 query IDs and their embeddings
    for qid, emb in list(loaded_query_embs.items())[:3]:
        print(f"Query ID: {qid}, Embedding: {emb[:5]}...")  # Print first 5 values
    