# Semantic Product Matching using CLIP Embeddings  
End-to-end pipeline for visual product matching using multimodal embeddings and vector search.

## Demo
![Demo](./media/full_demo.gif)

## Overview  
This system matches input images against product catalogs using three core components:
1. **CLIP** - Generates unified visual/text embeddings for images and text queries
2. **Pinecone** - High-performance vector database for nearest neighbor search
3. **MongoDB** - Stores and retrieves structured product metadata

### Architecture Flow  
1. **Input Image** → CLIP extracts 512-dim embedding
2. **Pinecone** finds top-5 similar product embeddings
3. **MongoDB** resolves matches to product details (name, price, category)

## Contents  
- `data_ingestion/`: Scripts for processing images/metadata
- `models/`: Quantized models (optional)
- `app.py`: Gradio demo interface
- `quantize_model.py`: Model optimization utilities

## Setup & Installation  
1. Clone repo:
```bash
git clone https://github.com/yourusername/product-matching-pipeline.git
cd product-matching-pipeline
