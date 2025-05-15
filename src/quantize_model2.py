#!/usr/bin/env python3
import os
import argparse
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
from torch.quantization import quantize_dynamic

# Optional ONNX‐Runtime quantization
try:
    from onnxruntime.quantization import quantize_dynamic as ort_quantize, QuantType
    _has_ort_quant = True
except ImportError:
    _has_ort_quant = False

def quantize_pytorch(model_name: str, out_dir: str):
    """
    1) Load HF CLIPModel
    2) Apply dynamic quantization to Linear → qint8 (dtype is implicit)
    3) Save:
       - quantized state_dict (via torch.save)
       - model config + processor
    """
    print(f"[PyTorch] Loading `{model_name}`…")
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    print("[PyTorch] Applying dynamic quantization…")
    # NOTE: no `dtype=` keyword here!
    quantized = quantize_dynamic(model, {torch.nn.Linear})

    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "pytorch_model.bin")
    print(f"[PyTorch] Saving quantized state_dict to `{state_path}`…")
    torch.save(quantized.state_dict(), state_path)  # workaround for save_pretrained bug

    print(f"[PyTorch] Saving config & processor to `{out_dir}`…")
    CLIPModel.from_pretrained(model_name).config.save_pretrained(out_dir)
    CLIPProcessor.from_pretrained(model_name).save_pretrained(out_dir)

    print("[PyTorch] Quantized model saved successfully.\n")

def export_onnx(model_name: str, onnx_path: str, opset: int = 13):
    """
    Export the full CLIP model (text+vision) to an FP32 ONNX graph.
    """
    print(f"[ONNX] Loading `{model_name}` for export…")
    model = CLIPModel.from_pretrained(model_name).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    dummy = processor(
        text=["a photo of a cat"],
        images=[Image.new("RGB", (224, 224), color=(128, 128, 128))],
        return_tensors="pt",
        padding=True
    )

    input_names  = ["input_ids", "attention_mask", "pixel_values"]
    output_names = ["text_embeds", "image_embeds"]
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "pixel_values":   {0: "batch"},
        "text_embeds":    {0: "batch"},
        "image_embeds":   {0: "batch"},
    }

    print(f"[ONNX] Exporting to `{onnx_path}` (opset {opset})…")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["pixel_values"]),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print("[ONNX] Export complete.\n")

def quantize_onnx(input_onnx: str, output_onnx: str):
    """
    Quantize an existing ONNX model to INT8 (weights-only dynamic quant).
    """
    if not _has_ort_quant:
        raise RuntimeError(
            "ONNX Runtime quantization APIs not available. "
            "Install `onnxruntime-tools` or `onnxruntime` with quantization support."
        )

    print(f"[ONNX-Quant] Quantizing `{input_onnx}` → `{output_onnx}`…")
    ort_quantize(
        model_input=input_onnx,
        model_output=output_onnx,
        weight_type=QuantType.QInt8
    )
    print("[ONNX-Quant] Done.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Quantize CLIP: PyTorch dynamic or ONNX dynamic bits."
    )
    parser.add_argument("--model-name", type=str,
                        default="openai/clip-vit-base-patch32",
                        help="Hugging Face model identifier")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save the quantized model or ONNX exports")
    parser.add_argument("--do-pytorch-quant", action="store_true",
                        help="Run PyTorch dynamic quantization and save manually")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export an FP32 ONNX model")
    parser.add_argument("--onnx-input", type=str,
                        help="Path to an existing ONNX model; if set, we'll quantize it")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version for export")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Must choose at least one action
    if not (args.onnx_input or args.export_onnx or args.do_pytorch_quant):
        parser.error("Require one of --do-pytorch-quant, --export-onnx, or --onnx-input")

    if args.onnx_input:
        out_path = os.path.join(args.output_dir, "model.quant.onnx")
        quantize_onnx(args.onnx_input, out_path)

    if args.export_onnx:
        fp32_path = os.path.join(args.output_dir, "model.fp32.onnx")
        export_onnx(args.model_name, fp32_path, opset=args.opset)

    if args.do_pytorch_quant:
        quantize_pytorch(args.model_name, args.output_dir)

if __name__ == "__main__":
    main()
