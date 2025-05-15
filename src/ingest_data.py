# src/ingest_data.py
import os
import json
import argparse
import random
from dotenv import load_dotenv
import numpy as np

import embed
import vector_db
import mongo_db
from mongodb_logger import log_event

def ingest_data(images_dir: str, metadata_file_path: str, batch_size: int = 50):
    """
    Reads product metadata from a single JSON file, processes associated images,
    creates embeddings, and stores them in Pinecone vector DB and MongoDB.
    """
    print("Starting data ingestion process...")

    # Load environment variables from .env file
    load_dotenv()

    # Initialize MongoDB client
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

    # Initialize Pinecone
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

    # --- Ingestion Process ---
    # 1. Read all metadata from the single JSON file
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
    all_vector_ids_ingested = [] # Stores vector IDs (image_filename_sans_ext)
    vectors_to_upsert_batch = []

    # Iterate through each product's metadata
    for product_meta in all_products_metadata:
        main_product_id = product_meta.get("id")
        product_images = product_meta.get("images")

        if not main_product_id:
            print(f"  Product metadata missing 'id'. Skipping entry: {product_meta.get('name', 'N/A')}")
            continue
        
        if not product_images or not isinstance(product_images, list):
            print(f"  Product ID {main_product_id} missing 'images' list or it's invalid. Skipping images for this product.")
            # Still store metadata if you want, or skip entirely
            # For now, we will store metadata even if images are missing/invalid,
            # but no vectors will be generated for this product.
            try:
                mongo_client.add_product(main_product_id, product_meta)
                print(f"  Metadata for {main_product_id} (no valid images) stored/updated in MongoDB.")
            except RuntimeError as e:
                print(f"  Error storing metadata for {main_product_id} in MongoDB: {e}.")
                log_event("error", f"MongoDB error storing metadata for {main_product_id}: {e}")
            continue # Skip to next product in metadata

        print(f"\nProcessing Product ID (from metadata): {main_product_id} - Name: {product_meta.get('name', 'N/A')}")

        # 2. Store metadata in MongoDB (once per product ID)
        try:
            mongo_client.add_product(main_product_id, product_meta)
            print(f"  Metadata for {main_product_id} stored/updated in MongoDB.")
        except RuntimeError as e:
            print(f"  Error storing metadata for {main_product_id} in MongoDB: {e}. Skipping image processing for this product.")
            log_event("error", f"MongoDB error storing metadata for {main_product_id}: {e}")
            continue 

        # 3. Process each image associated with this product
        for image_filename in product_images:
            vector_id = os.path.splitext(image_filename)[0] # e.g., "001_1" from "001_1.jpg"
            image_path = os.path.join(images_dir, image_filename)

            print(f"  Processing image: {image_filename} (Vector ID: {vector_id})")

            if not os.path.exists(image_path):
                print(f"    Image file not found: {image_path}. Skipping this image.")
                continue
            
            # 3a. Create CLIP embedding for the image
            try:
                embedding_vector = embed.embed_image(image_path)
                if not isinstance(embedding_vector, np.ndarray) or embedding_vector.shape[0] != vector_db.VECTOR_DIM:
                    print(f"    Embedding generation failed or dimension mismatch for {image_filename}. Expected {vector_db.VECTOR_DIM}, got {embedding_vector.shape if isinstance(embedding_vector, np.ndarray) else type(embedding_vector)}. Skipping this image.")
                    continue
                print(f"    Embedding generated successfully (shape: {embedding_vector.shape}).")
            except Exception as e:
                print(f"    Error generating embedding for {image_path}: {e}. Skipping this image.")
                log_event("error", f"Embedding error for {image_path}: {e}")
                continue
            
            # 3b. Prepare vector for Pinecone batch upsert
            vectors_to_upsert_batch.append((vector_id, embedding_vector.tolist()))
            all_vector_ids_ingested.append(vector_id)
            processed_image_count += 1

            # Upsert in batches
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

    # Upsert any remaining vectors
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

    # --- Verification Process ---
    print("\n--- Verification Step ---")
    num_samples_to_verify = min(3, len(all_vector_ids_ingested))
    if num_samples_to_verify == 0:
        print("No image vectors were successfully processed to verify.")
        return
        
    sample_vector_ids_to_verify = random.sample(all_vector_ids_ingested, num_samples_to_verify)
    print(f"Attempting to verify {num_samples_to_verify} random image vector(s): {sample_vector_ids_to_verify}")

    for v_id in sample_vector_ids_to_verify: # v_id is like "001_1"
        print(f"\nVerifying Vector ID: {v_id}")
        
        # 1. Verify metadata from MongoDB
        # Derive main product ID from vector ID (e.g., "001" from "001_1")
        # This assumes a naming convention like productID_imageNum for vector_id
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

        # 2. Verify vector in Pinecone (by querying with its own embedding)
        # Find the original image filename for the current v_id
        original_image_filename = None
        # We need to find which product_meta and image_filename corresponds to v_id
        # This is a bit inefficient here, but for verification it's okay.
        # A better way would be to store the (v_id, image_filename_with_ext) mapping if needed frequently.
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
                    query_embedding = embed.embed_image(image_path_for_verification)
                    
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
    
    args = parser.parse_args()

    ingest_data(args.images_dir, args.metadata_file, args.batch_size)