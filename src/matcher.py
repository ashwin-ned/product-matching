# src/matcher.py
"""
Match a text or image query against product embeddings stored in Pinecone and show metadata.
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Local modules
from vector_db import query_vector, create_index, upsert_embeddings
from metadata_db import MetadataDB

# Configuration
MODEL_NAME         = "openai/clip-vit-base-patch32"
PRODUCT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

class Matcher:
    def __init__(self, model_name: str = MODEL_NAME):
        # Load CLIP model + processor
        self.device    = DEVICE
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        # Metadata DB
        self.meta_db = MetadataDB()

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb[0].cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb[0].cpu().numpy().astype(np.float32)

    def match(self, query_emb: np.ndarray, top_k: int = 5):
        # Ensure Pinecone index exists and has data
        create_index()
        # upsert_embeddings()  # run once, not every query
        # Query Pinecone
        matches = query_vector(query_emb, top_k=top_k)
        # Fetch metadata
        results = []
        for pid, score in matches:
            meta = self.meta_db.get_product(pid)
            results.append({"id": pid, "score": score, "metadata": meta})
        return results

    def display(self, results: list):
        print(f"Top {len(results)} matches:")
        for res in results:
            pid = res['id']
            score = res['score']
            meta = res['metadata'] or {}
            print(f"• {pid} (score: {score:.4f})")
            if meta:
                print(f"  Name: {meta.get('name')}")
                print(f"  Category: {meta.get('category')}")
                print(f"  Price: {meta.get('price')}")
            img_path = os.path.join(PRODUCT_IMAGES_DIR, f"{pid}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img.show()
            print()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to query image")
    group.add_argument("--text",  type=str, help="Text query")
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    matcher = Matcher()
    if args.image:
        q_emb = matcher.embed_image(args.image)
    else:
        q_emb = matcher.embed_text(args.text)

    results = matcher.match(q_emb, top_k=args.top_k)
    matcher.display(results)

if __name__ == "__main__":
    main()