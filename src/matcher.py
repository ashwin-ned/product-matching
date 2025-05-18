import os
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor
import onnxruntime
from concurrent.futures import ThreadPoolExecutor

# Local modules
from vector_db import query_vector
from mongo_db import MongoDB
from mongodb_logger import log_event

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
PRODUCT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Matcher:
    def __init__(self, use_quantized: bool = True):
        self.device = DEVICE
        self.use_quantized = use_quantized
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model_type = "fp16" if use_quantized else "fp32"
        model_path = os.path.join(MODELS_DIR, f"clip_{model_type}.onnx")
        self.ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider' if DEVICE == 'cuda' else 'CPUExecutionProvider']
        )
        self.meta_db = MongoDB()

    def embed_image(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            dtype = np.float16 if self.use_quantized else np.float32
            pixel_values = inputs.pixel_values.numpy().astype(dtype)
            ort_inputs = {
                "pixel_values": pixel_values,
                "input_ids": np.zeros((1, 77), dtype=np.int64),  # Dummy text
                "attention_mask": np.zeros((1, 77), dtype=np.int64)
            }
            emb = self.ort_session.run(["image_embeds"], ort_inputs)[0][0]
            return emb.astype(np.float32)
        except Exception as e:
            log_event(f"Error embedding image '{image_path}': {e}")
            return np.zeros((512,), dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            dtype = np.float16 if self.use_quantized else np.float32
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
                "pixel_values": np.zeros((1, 3, 224, 224), dtype=dtype)
            }
            emb = self.ort_session.run(["text_embeds"], ort_inputs)[0][0]
            return emb.astype(np.float32)
        except Exception as e:
            log_event(f"Error embedding text '{text}': {e}")
            return np.zeros((512,), dtype=np.float32)

    def match(self, query_emb: np.ndarray, top_k: int = 5):
        buffer = top_k
        unique_matches = []

        while True:
            matches = query_vector(query_emb, top_k=buffer)
            seen = set()
            unique_matches = []
            for pid, score in matches:
                base_pid = pid.split("_")[0]
                if base_pid not in seen:
                    seen.add(base_pid)
                    unique_matches.append((pid, score))
                if len(unique_matches) >= top_k:
                    break
            if len(unique_matches) >= top_k or buffer > top_k * 10:
                break
            buffer *= 2

        def fetch_metadata(pid):
            return self.meta_db.get_product(pid.split("_")[0])

        with ThreadPoolExecutor() as executor:
            metadata = list(executor.map(fetch_metadata, [pid for pid, _ in unique_matches[:top_k]]))

        results = []
        for (pid, score), meta in zip(unique_matches[:top_k], metadata):
            results.append({
                "id": pid.split("_")[0],
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
            print(f"  Raw metadata: {meta}")

            if meta:
                print("  Parsed Metadata:")
                for k, v in meta.items():
                    print(f"    {k}: {v}")

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
    parser.add_argument("--use_quantized", action="store_true", 
                      help="Use quantized FP16 model instead of FP32 ONNX model")
    args = parser.parse_args()

    matcher = Matcher(use_quantized=args.use_quantized)
    
    if args.image:
        q_emb = matcher.embed_image(args.image)
    else:
        q_emb = matcher.embed_text(args.text)

    results = matcher.match(q_emb, top_k=args.top_k)
    matcher.display(results)

if __name__ == "__main__":
    main()