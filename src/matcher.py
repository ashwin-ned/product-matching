import os
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# Local modules
from vector_db import query_vector
from mongo_db import MongoDB
from mongodb_logger import log_event

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
PRODUCT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Matcher:
    def __init__(self, model_name: str = MODEL_NAME):
        # Load CLIP model + processor
        self.device = DEVICE
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Initialize MongoDB metadata store
        self.meta_db = MongoDB()

    def embed_image(self, image_path: str) -> np.ndarray:

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb[0].cpu().numpy().astype(np.float32)
        except Exception as e:
            log_event(f"Error embedding image '{image_path}': {e}")
            return np.zeros((512,), dtype=np.float32)
        
    def embed_text(self, text: str) -> np.ndarray:
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb[0].cpu().numpy().astype(np.float32)
        except Exception as e:
            log_event(f"Error embedding text '{text}': {e}")
            return np.zeros((512,), dtype=np.float32)

    def match(self, query_emb: np.ndarray, top_k: int = 5):
        """
        Match the query embedding against the vector database and fetch metadata for the top results.
        Dynamically adjusts the query buffer size and parallelizes metadata fetching for speed.
        """
        buffer = top_k  # Start with a buffer equal to top_k
        unique_matches = []

        while True:
            # Query the vector database
            matches = query_vector(query_emb, top_k=buffer)

            # Deduplicate by product ID, keeping the highest-scoring instance
            seen = set()
            unique_matches = []
            for pid, score in matches:
                base_pid = pid.split("_")[0]  # Extract base product ID (e.g., "001")
                if base_pid not in seen:
                    seen.add(base_pid)
                    unique_matches.append((pid, score))
                if len(unique_matches) >= top_k:
                    break  # Early exit if we have enough unique products

            # If we have enough unique matches or the buffer is too large, stop querying
            if len(unique_matches) >= top_k or buffer > top_k * 10:
                break

            # Increase the buffer size for the next iteration
            buffer *= 2

        # Fetch metadata for the top K unique products in parallel
        def fetch_metadata(pid):
            return self.meta_db.get_product(pid.split("_")[0])  # Fetch base product metadata

        with ThreadPoolExecutor() as executor:
            metadata = list(executor.map(fetch_metadata, [pid for pid, _ in unique_matches[:top_k]]))

        # Combine results with metadata
        results = []
        for (pid, score), meta in zip(unique_matches[:top_k], metadata):
            results.append({
                "id": pid.split("_")[0],  # Return base product ID
                "score": score,
                "metadata": meta
            })

        return results

    def display(self, results: list):
        print(f"Top {len(results)} matches:")
        for res in results:
            pid = res['id']
            score = res['score']
            meta = res.get('metadata', {}) or {}

            print(f"• {pid} (score: {score:.4f})")
            # Always print raw metadata for clarity
            print(f"  Raw metadata: {meta}")

            if meta:
                print("  Parsed Metadata:")
                for k, v in meta.items():
                    print(f"    {k}: {v}")

            # Optionally show image if exists
            img_path = os.path.join(PRODUCT_IMAGES_DIR, f"{pid}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img.show()
                except Exception as e:
                    log_event(f"Error displaying image for {pid}: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Match a text or image query against product embeddings.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to query image")
    group.add_argument("--text", type=str, help="Text query string")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top matches to return")
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
