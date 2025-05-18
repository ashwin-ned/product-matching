import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
import onnxruntime

def load_onnx_model(model_path):
    """Load the quantized ONNX model and create an inference session"""
    session = onnxruntime.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']  # Change to CUDA if available
    )
    return session

def process_image(image_path, processor):
    """Process an image using CLIP processor"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs.pixel_values.numpy().astype(np.float16)  # Convert to FP16
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_embeddings(session, pixel_values, text=None, processor=None):
    """Get embeddings for images and optionally text"""
    # Image embeddings
    image_inputs = {
        "pixel_values": pixel_values.astype(np.float16),
        "input_ids": np.zeros((1, 77), dtype=np.int64),  # Dummy text inputs
        "attention_mask": np.zeros((1, 77), dtype=np.int64)
    }
    
    image_embeds = session.run(["image_embeds"], image_inputs)[0]
    
    text_embeds = None
    if text and processor:
        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )
        text_inputs = {
            "input_ids": text_inputs["input_ids"].numpy(),
            "attention_mask": text_inputs["attention_mask"].numpy(),
            "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float16)  # Dummy image inputs
        }
        text_embeds = session.run(["text_embeds"], text_inputs)[0]
    
    return image_embeds, text_embeds

def main():
    # Configuration
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    onnx_model_path = os.path.join(models_dir, "clip_fp16.onnx")
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    
    # Load resources
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    session = load_onnx_model(onnx_model_path)
    
    # Example texts to compare with images
    text_queries = [
        "wafers",
        "shampoo",
        "shirt"
    ]
    
    # Process all images in directory
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing {image_file}...")
        
        # Process image
        pixel_values = process_image(image_path, processor)
        if pixel_values is None:
            continue
            
        # Get embeddings
        image_embeds, text_embeds = get_embeddings(
            session, 
            pixel_values,
            text=text_queries,
            processor=processor
        )
        
        # Print results
        print(f"\nImage embeddings shape: {image_embeds.shape}")
        print(f"First 5 values: {image_embeds[0, :5]}\n")
        
        if text_embeds is not None:
            # Calculate similarity scores
            similarities = (text_embeds @ image_embeds.T).squeeze()
            print("Text-Image Similarity Scores:")
            for text, score in zip(text_queries, similarities):
                print(f"- {text}: {score:.4f}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()