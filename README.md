# Semantic Product Matching with CLIP Embeddings

An end-to-end pipeline for visual product matching using multimodal embeddings and vector search.

## 📺 Demo

![Demo](./media/full_demo.gif)

---

## 🚀 Overview

This system lets you match an input image (or text query) against a product catalog by combining:

1. **CLIP**  
   Generates joint image/text embeddings (512-dimensional).
2. **Pinecone**  
   A high-performance vector database for k-nearest-neighbors search.
3. **MongoDB**  
   Stores and retrieves structured product metadata.

### 🔄 Architecture Flow

1. **Input**  
   - Image (or text) is fed to CLIP.  
   - CLIP produces a 512-dim embedding.  
2. **Vector Search**  
   - Pinecone finds the top-5 nearest embeddings in the product index.  
3. **Metadata Lookup**  
   - MongoDB returns full product details (name, price, category, etc.) for each match.

---

## 📂 Repository Structure
├── data_ingestion/ # Scripts to process images & metadata
├── models/ # Original & quantized CLIP weights
├── quantization/ # Utilities for model quantization
├── media/ # Demo GIFs and other assets
├── images/ # Product images (for ingestion)
├── metadata/ # JSON metadata files
├── app.py # Gradio demo interface
├── quantize_model.py # Standalone quantization script
└── requirements.txt # Python dependencies



---

## ⚙️ Setup & Installation

1. **Clone & install**  
   ```bash
   git clone https://github.com/yourusername/product-matching-pipeline.git
   cd product-matching-pipeline
   pip install -r requirements.txt

2. Prepare images & metadata

Place all product images in ./images/.

Add a single JSON file ./metadata/products.json containing an array of product entries.

Example entry for a product in products.json
{
  "id": "002",
  "name": "Vanilla & Coconut Shower Gel",
  "category": "Personal Care",
  "price": 5.49,
  "description": "Refreshing shower gel with vanilla fragrance and coconut extracts.",
  "images": ["002_1.jpg", "002_2.jpg"]
}

3. Configure environment variables
Create a .env file in the project src folder with the following keys:
### MongoDB for product metadata
MONGO_URI="your_mongo_connection_string"
MONGO_DB_NAME="your_db_name"
MONGO_COLLECTION="products_collection_name"

### MongoDB for logging (optional)
LOGGER_MONGO_URI="your_logger_connection_string"
LOGGER_DB_NAME="your_logger_db_name"
LOGGER_MONGO_COLLECTION="your_logger_collection_name"

### Pinecone API
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_ENVIRONMENT="your_pinecone_env"

4. Ingest Data
python data_ingestion/ingest.py

-Reads ./images/ and ./metadata/products.json

-Generates CLIP embeddings and upserts them to Pinecone

-Stores metadata in MongoDB

5. Quantize the Model 
python quantization/quantize_model.py

6. Run the Gradio Demo
python app.py

Opens a local web interface where you can query by image or text.

📝 Usage
-Image query: upload a photo of a product; the app returns the top-5 catalog matches.

-Text query: type a product description; the app performs the same embedding + search pipeline.
