import os
import torch
import onnx
from transformers import CLIPProcessor, CLIPModel
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnxruntime
from PIL import Image
import requests 
import numpy as np 
import traceback 


MODEL_NAME = "openai/clip-vit-base-patch32"
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "../models")

class CLIPONNXWrapper(torch.nn.Module):
    """
    A wrapper for the CLIPModel to ensure consistent output for ONNX export,
    specifically returning text_embeds and image_embeds.
    It expects input_ids, attention_mask, and pixel_values.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        # Pass inputs to the underlying CLIP model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True # Ensure outputs is a dict-like object
        )
        # Return the desired embeddings
        return outputs.text_embeds, outputs.image_embeds

def main():
    """
    Main function to download, convert, quantize, and test the CLIP model.
    """
    print(f"Creating models directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load CLIP model and processor from Hugging Face
    print(f"Loading model and processor for {MODEL_NAME}...")
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model.eval() # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        traceback.print_exc()
        return

    pytorch_model_dir = os.path.join(MODELS_DIR, "clip_pytorch")
    print(f"Saving PyTorch model and processor to {pytorch_model_dir}...")
    try:
        model.save_pretrained(pytorch_model_dir)
        processor.save_pretrained(pytorch_model_dir)
    except Exception as e:
        print(f"Error saving PyTorch model: {e}")
        traceback.print_exc()


    print("Preparing dummy inputs...")
    try:

        texts = ["a photo of a cat"]
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length", 
            truncation=True,      
            max_length=77         
        )
        dummy_input_ids = text_inputs['input_ids']
        dummy_attention_mask = text_inputs['attention_mask']


        try:
            img_url = "http://images.cocodataset.org/val2017/000000039769.jpg" # sample image
            image = Image.open(requests.get(img_url, stream=True, timeout=10).raw).convert("RGB")
            print(f"Successfully downloaded sample image from {img_url}")
        except Exception as img_e:
            print(f"Failed to download sample image (Error: {img_e}). Using a random dummy PIL image.")
            random_image_data = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_image_data, 'RGB')

        # Process the image: resize, normalize, and convert to tensor
        image_inputs = processor(images=image, return_tensors="pt")
        dummy_pixel_values = image_inputs['pixel_values']

        # Ensure batch sizes match (should be 1 for both text and image based on above)
        assert dummy_input_ids.shape[0] == 1, "Batch size for input_ids should be 1"
        assert dummy_pixel_values.shape[0] == 1, "Batch size for pixel_values should be 1"

    except Exception as e:
        print(f"Error preparing dummy inputs: {e}")
        traceback.print_exc()
        return

    # Instantiate the ONNX wrapper
    onnx_model_wrapper = CLIPONNXWrapper(model)
    
    # Define input and output names for the ONNX model graph
    input_names = ["input_ids", "attention_mask", "pixel_values"]
    output_names = ["text_embeds", "image_embeds"]
    
    # Arguments for torch.onnx.export
    onnx_export_args = (dummy_input_ids, dummy_attention_mask, dummy_pixel_values)

    # Define dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"}, # batch_size, seq_len
        "attention_mask": {0: "batch_size", 1: "sequence_length"}, # batch_size, seq_len
        "pixel_values": {0: "batch_size"}, # batch_size, channels, height, width
        "text_embeds": {0: "batch_size"},  # batch_size, embedding_dim
        "image_embeds": {0: "batch_size"}  # batch_size, embedding_dim
    }

    # 4. Export to ONNX FP32
    onnx_fp32_path = os.path.join(MODELS_DIR, "clip_fp32.onnx")
    print(f"Exporting model to ONNX FP32: {onnx_fp32_path}...")
    try:
        torch.onnx.export(
            onnx_model_wrapper,
            onnx_export_args,
            onnx_fp32_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,        
            do_constant_folding=True, 
            export_params=True        
        )
        print("ONNX FP32 export complete.")
    except Exception as e:
        print(f"Error exporting model to ONNX FP32: {e}")
        traceback.print_exc()
        return

    # Convert ONNX FP32 to FP16
    onnx_fp16_path = os.path.join(MODELS_DIR, "clip_fp16.onnx")
    print(f"Converting ONNX model to FP16: {onnx_fp16_path}...")
    try:
        model_fp32_onnx = onnx.load(onnx_fp32_path)
        model_fp16_onnx = convert_float_to_float16(model_fp32_onnx, keep_io_types=False)
        onnx.save(model_fp16_onnx, onnx_fp16_path)
        print("ONNX FP16 conversion complete.")
    except Exception as e:
        print(f"Error converting ONNX model to FP16: {e}")
        traceback.print_exc()
        return

    # Test the quantized model using ONNX Runtime
    print(f"Testing the FP16 ONNX model ({onnx_fp16_path}) with ONNX Runtime...")
    try:
    
        ort_session = onnxruntime.InferenceSession(
            onnx_fp16_path,
            providers=['CPUExecutionProvider']
        )

        ort_inputs = {
            "input_ids": dummy_input_ids.cpu().numpy(),
            "attention_mask": dummy_attention_mask.cpu().numpy(),
            "pixel_values": dummy_pixel_values.cpu().numpy().astype(np.float16)
        }

        # Run inference
        ort_outputs = ort_session.run(output_names, ort_inputs)
        
        # ort_outputs is a list of numpy arrays, corresponding to output_names
        text_embeds_onnx = ort_outputs[0]
        image_embeds_onnx = ort_outputs[1]

        print("\n--- ONNX FP16 Model Test Results ---")
        print("ONNX FP16 model executed successfully!")
        print(f"  Output text_embeds shape: {text_embeds_onnx.shape}")
        print(f"  Output image_embeds shape: {image_embeds_onnx.shape}")

        # Print first 5 elements of the first batch's embeddings for a quick check
        print(f"  First 5 values of text_embeds[0]: {text_embeds_onnx[0, :5]}")
        print(f"  First 5 values of image_embeds[0]: {image_embeds_onnx[0, :5]}")
        print("--- End of Test ---")


    except Exception as e:
        print(f"Error during ONNX FP16 model testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
