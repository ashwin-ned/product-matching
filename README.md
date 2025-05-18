# Semantic Product Matching with CLIP Embeddings

An end-to-end pipeline for visual product matching using multimodal embeddings and vector search.

## 📺 Demo

![Demo](./media/full_demo.gif)

---

## 🚀 Overview

This system enables matching an input image or text query against a product catalog by leveraging:

1.  **CLIP (Contrastive Language-Image Pre-training)**: Generates 512-dimensional joint embeddings for images and text.
2.  **Pinecone**: A high-performance vector database used for efficient nearest-neighbors search using embeddings.
3.  **MongoDB**: Stores and retrieves structured product metadata.

### 🔄 Architecture Flow

The matching process follows these steps:

1.  **Input Processing**:
    *   An input image or text query is provided to the CLIP model.
    *   CLIP generates a 512-dimensional embedding vector representing the input.
2.  **Vector Search**:
    *   The generated embedding is used to query the Pinecone vector database.
    *   Pinecone returns the nearest embeddings from the indexed product catalog using cosine similarity.
3.  **Metadata Lookup**:
    *   For each of the top matches, corresponding product details (name, price, category, etc.) are retrieved from MongoDB.

---

## 📂 Repository Structure

```
.
├── app.py                  # Gradio demo interface
├── data_ingestion.py       # Script to process images & metadata
├── images/                 # Product images (for ingestion)
├── media/                  # Demo GIFs and other assets
├── metadata/               # JSON metadata files
├── models/                 # Original or quantized CLIP weights
├── quantization/           # Utilities for model quantization
├── .env                    # Environment file with API keys for Pinecone & MongoDB
├── environment.yml         # Conda environment file
└── requirements.txt        # Python dependencies

```

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the project:

1.  **Clone the Repository and Install Dependencies**:
    ```bash
    git clone https://github.com/yourusername/product-matching-pipeline.git
    cd product-matching-pipeline
    conda env create -f environment.yml
    pip install -r requirements.txt
    ```

2.  **Prepare Images and Metadata**:
    *   Place all product images in the `images/` directory.
    *   Create a single JSON file named `products.json` in the `metadata/` directory. This file should contain an array of product entries.

    **Example `products.json` entry**:
    ```json
    {
      "id": "002",
      "name": "Vanilla & Coconut Shower Gel",
      "category": "Personal Care",
      "price": 5.49,
      "description": "Refreshing shower gel with vanilla fragrance and coconut extracts.",
      "images": ["002_1.jpg", "002_2.jpg"]
    }
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the project's `src/` folder (or the root, adjust paths in scripts if necessary) with the following keys:
    ```env
    MONGO_URI="your_mongo_connection_string"
    MONGO_DB_NAME="your_db_name"
    MONGO_COLLECTION="products_collection_name"

    LOGGER_MONGO_URI="your_logger_connection_string"
    LOGGER_DB_NAME="your_logger_db_name"
    LOGGER_MONGO_COLLECTION="your_logger_collection_name"

    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="your_pinecone_env"
    ```

4.  **Ingest Data**:
    Run the data ingestion script. This script reads images from `./images/` and metadata from `./metadata/products.json`, generates CLIP embeddings, upserts them to Pinecone, and stores metadata in MongoDB.
    ```bash
    python ingest_data.py
    ```

5.  **Quantize the Model (Optional but Recommended)**:
    To optimize the model for inference, run the quantization script:
    ```bash
    python quantization/quantize_model.py
    ```
6.  **Run the Gradio Demo**:
    Launch the web interface:
    ```bash
    python app.py
    ```
    This opens a local web interface where you can query by image or text.

---

## 📝 Usage

Once the Gradio demo is running:

*   **Image Query**: Upload a photo of a product. The application will display the top-K matching products from the catalog or the top match.
*   **Text Query**: Type a description of a product. The application will perform the same embedding and search process to find and display the top-K matches.
