# quantize_model.py
"""
Dynamic quantization of a CLIP ONNX model using ONNX Runtime.

Takes an FP32 ONNX file and produces an INT8-quantized ONNX file,
quantizing weights to QInt8 while preserving activation precision.
"""

import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamically quantize an ONNX model to INT8"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join("..", "models", "clip-vit-base-patch32.onnx"),
        help="Path to the FP32 ONNX model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join("..", "models", "clip-vit-base-patch32-int8.onnx"),
        help="Path to write the INT8-quantized ONNX model"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Quantizing model:\n  Input:  {args.input}\n  Output: {args.output}")

    # Perform dynamic quantization: only weights are quantized to QInt8.
    # Activations remain in FP32, which preserves accuracy.
    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QInt8,      # Quantize weights to signed int8
        per_channel=False,                # Set to True for per-channel quantization (slightly better accuracy)
        reduce_range=False                # Use full int8 range
    )

    print("Quantization complete.  INT8 model saved to:", args.output)

if __name__ == "__main__":
    main()
