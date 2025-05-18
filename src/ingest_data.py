import os
import json
import argparse
import random
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import CLIPProcessor
from PIL import Image
from onnxruntime import InferenceSession

import vector_db
import mongo_db
from mongodb_logger import log_event

# Load .env file
load_dotenv()

# Configuration
DEFAULT_MODEL_PATH = os.path.join("..", "models", "clip_fp16.onnx")

def initialize_model(model_path=DEFAULT_MODEL_PATH):
    global processor, onnx_session
    
    print(f"Loading model from {model_path}...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    if model_path.endswith(".onnx"):
        onnx_session = InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    else:
        raise ValueError("Only ONNX models are supported.")

def embed_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="np")
        
        ort_inputs = {
            "pixel_values": inputs["pixel_values"].astype(np.float16),
            "input_ids": np.zeros((1, 77), dtype=np.int64),
            "attention_mask": np.zeros((1, 77), dtype=np.int64)
        }
        
        outputs = onnx_session.run(["image_embeds"], ort_inputs)
        embedding = outputs[0][0]
        
        embedding /= np.linalg.norm(embedding)
        return embedding.astype(np.float32)
        
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        return None

def ingest_data(images_dir: str, metadata_file_path: str, batch_size: int = 50, model_path: str = None):
    print("Starting data ingestion process...")

    initialize_model(model_path)

    print("Initializing MongoDB connection...")
    try:
        mongo_client = mongo_db.MongoDB(
            uri=os.getenv("MONGO_URI"),
            db_name=os.getenv("MONGO_DB_NAME"),
            collection_name=os.getenv("MONGO_COLLECTION")
        )
        print(f"Successfully connected to MongoDB: DB='{mongo_client.db.name}', Collection='{mongo_client.collection.name}'")
    except RuntimeError as e:
        print(f"Error initializing MongoDB: {e}")
        log_event("error", f"MongoDB initialization error: {e}")
        return

    print(f"Using Pinecone index: '{vector_db.INDEX_NAME}' with dimension: {vector_db.VECTOR_DIM}")
    
    print("Ensuring Pinecone index exists...")
    try:
        vector_db.create_index() 
        pinecone_index = vector_db.pc.Index(vector_db.INDEX_NAME)
        print(f"Pinecone index '{vector_db.INDEX_NAME}' is ready.")
    except Exception as e:
        print(f"Error ensuring Pinecone index: {e}")
        log_event("error", f"Pinecone index error: {e}")
        return

    print(f"Reading metadata from: {metadata_file_path}")
    if not os.path.exists(metadata_file_path):
        print(f"  Metadata file not found: {metadata_file_path}. Exiting.")
        return
    
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            all_products_metadata = json.load(f)
        if not isinstance(all_products_metadata, list):
            print("  Metadata file should contain a JSON list of products. Exiting.")
            return
        print(f"  Successfully read {len(all_products_metadata)} product entries from metadata file.")
    except json.JSONDecodeError:
        print(f"  Error decoding JSON from: {metadata_file_path}. Exiting.")
        log_event("error", f"JSON decode error: {metadata_file_path}")
        return
    except Exception as e:
        print(f"  Error reading metadata file {metadata_file_path}: {e}. Exiting.")
        return

    processed_image_count = 0
    all_vector_ids_ingested = []
    vectors_to_upsert_batch = []

    for product_meta in all_products_metadata:
        main_product_id = product_meta.get("id")
        product_images = product_meta.get("images")

        if not main_product_id:
            print(f"  Product metadata missing 'id'. Skipping entry: {product_meta.get('name', 'N/A')}")
            continue
        
        if not product_images or not isinstance(product_images, list):
            print(f"  Product ID {main_product_id} missing 'images' list or it's invalid. Skipping images for this product.")
            try:
                mongo_client.add_product(main_product_id, product_meta)
                print(f"  Metadata for {main_product_id} (no valid images) stored/updated in MongoDB.")
            except RuntimeError as e:
                print(f"  Error storing metadata for {main_product_id} in MongoDB: {e}.")
                log_event("error", f"MongoDB error storing metadata for {main_product_id}: {e}")
            continue

        print(f"\nProcessing Product ID (from metadata): {main_product_id} - Name: {product_meta.get('name', 'N/A')}")

        try:
            mongo_client.add_product(main_product_id, product_meta)
            print(f"  Metadata for {main_product_id} stored/updated in MongoDB.")
        except RuntimeError as e:
            print(f"  Error storing metadata for {main_product_id} in MongoDB: {e}. Skipping image processing for this product.")
            log_event("error", f"MongoDB error storing metadata for {main_product_id}: {e}")
            continue 

        for image_filename in product_images:
            vector_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_dir, image_filename)

            print(f"  Processing image: {image_filename} (Vector ID: {vector_id})")

            if not os.path.exists(image_path):
                print(f"    Image file not found: {image_path}. Skipping this image.")
                continue
            
            try:
                embedding_vector = embed_image(image_path)
                if embedding_vector is None or embedding_vector.shape[0] != vector_db.VECTOR_DIM:
                    print(f"    Embedding generation failed or dimension mismatch for {image_filename}. Skipping this image.")
                    continue
                print(f"    Embedding generated successfully (shape: {embedding_vector.shape}).")
            except Exception as e:
                print(f"    Error generating embedding for {image_path}: {e}. Skipping this image.")
                log_event("error", f"Embedding error for {image_path}: {e}")
                continue
            
            vectors_to_upsert_batch.append((vector_id, embedding_vector.tolist()))
            all_vector_ids_ingested.append(vector_id)
            processed_image_count += 1

            if len(vectors_to_upsert_batch) >= batch_size:
                try:
                    print(f"    Upserting batch of {len(vectors_to_upsert_batch)} image vectors to Pinecone...")
                    pinecone_index.upsert(vectors=vectors_to_upsert_batch)
                    print(f"    Successfully upserted batch to Pinecone.")
                    vectors_to_upsert_batch = [] 
                except Exception as e:
                    print(f"    Error upserting batch to Pinecone: {e}. Some vectors may not have been stored.")
                    log_event("error", f"Pinecone upsert error: {e}")
                    vectors_to_upsert_batch = []

    if vectors_to_upsert_batch:
        try:
            print(f"  Upserting final batch of {len(vectors_to_upsert_batch)} image vectors to Pinecone...")
            pinecone_index.upsert(vectors=vectors_to_upsert_batch)
            print(f"  Successfully upserted final batch to Pinecone.")
        except Exception as e:
            print(f"  Error upserting final batch to Pinecone: {e}.")
            log_event("error", f"Pinecone final batch upsert error: {e}")

    print(f"\n--- Ingestion Summary ---")
    print(f"Total product metadata entries processed: {len(all_products_metadata)}")
    print(f"Total images processed and attempted for vector ingestion: {processed_image_count}")
    print(f"Total unique vector IDs ingested: {len(all_vector_ids_ingested)}")
    
    if not all_vector_ids_ingested:
        print("No image vectors were ingested. Exiting verification.")
        return

    print("\n--- Verification Step ---")
    num_samples_to_verify = min(3, len(all_vector_ids_ingested))
    if num_samples_to_verify == 0:
        print("No image vectors were successfully processed to verify.")
        return
        
    sample_vector_ids_to_verify = random.sample(all_vector_ids_ingested, num_samples_to_verify)
    print(f"Attempting to verify {num_samples_to_verify} random image vector(s): {sample_vector_ids_to_verify}")

    for v_id in sample_vector_ids_to_verify:
        print(f"\nVerifying Vector ID: {v_id}")
        
        main_product_id_for_mongo = v_id.split('_')[0] if '_' in v_id else v_id 
        
        try:
            metadata = mongo_client.get_product(main_product_id_for_mongo)
            if metadata:
                print(f"  MongoDB: Metadata for main product ID '{main_product_id_for_mongo}' (related to vector '{v_id}') found: Name - '{metadata.get('name', 'N/A')}'")
            else:
                print(f"  MongoDB: Metadata for main product ID '{main_product_id_for_mongo}' NOT FOUND.")
        except RuntimeError as e:
            print(f"  MongoDB: Error retrieving metadata for main product ID '{main_product_id_for_mongo}': {e}")
            log_event("error", f"MongoDB error retrieving metadata for {main_product_id_for_mongo}: {e}")

        original_image_filename = None

        for p_meta in all_products_metadata:
            if p_meta.get("images"):
                for img_fname_with_ext in p_meta["images"]:
                    if os.path.splitext(img_fname_with_ext)[0] == v_id:
                        original_image_filename = img_fname_with_ext
                        break
            if original_image_filename:
                break
        
        if original_image_filename:
            image_path_for_verification = os.path.join(images_dir, original_image_filename)
            if os.path.exists(image_path_for_verification):
                try:
                    print(f"  Re-embedding image {image_path_for_verification} for Pinecone query verification...")
                    query_embedding = embed_image(image_path_for_verification)
                    
                    print(f"  Querying Pinecone with embedding of {v_id} (top_k=3)...")
                    matches = vector_db.query_vector(query_vec=query_embedding, top_k=3)
                    
                    if matches:
                        print(f"  Pinecone: Top matches for {v_id}'s embedding:")
                        match_found = False
                        for match_id, score in matches:
                            print(f"    - ID: {match_id}, Score: {score:.4f}")
                            if match_id == v_id:
                                match_found = True
                        if match_found:
                            print(f"  Pinecone: VERIFIED - Vector ID {v_id} found in top matches for its own embedding.")
                        else:
                            print(f"  Pinecone: WARNING - Vector ID {v_id} NOT found in top matches. (This can happen if other items are extremely similar or if IDs differ).")
                    else:
                        print(f"  Pinecone: No matches found for {v_id}'s embedding.")
                except Exception as e:
                    print(f"  Pinecone: Error during vector query verification for {v_id}: {e}")
                    log_event("error", f"Pinecone verification error for {v_id}: {e}")
            else:
                print(f"  Pinecone: Could not find image file {image_path_for_verification} to verify vector {v_id}.")
        else:
             print(f"  Pinecone: Could not determine original image filename for vector ID {v_id} from metadata to perform verification.")


    print("\nIngestion and verification process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest product images and metadata into Vector DB and MongoDB from a single products.json file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing product images referenced in the metadata file.")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the single JSON metadata file (e.g., products.json).")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for upserting vectors to Pinecone.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                      help=f"Path to the ONNX model file. Default: {DEFAULT_MODEL_PATH}")
    
    args = parser.parse_args()

    ingest_data(args.images_dir, args.metadata_file, args.batch_size, args.model_path)