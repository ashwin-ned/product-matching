# build_trt_engine.py (fixed)
"""
Convert a quantized CLIP ONNX model to a TensorRT engine for Triton.
Fixed to handle dynamic shapes for both image and text inputs.
"""

import os
import sys
import logging
import argparse
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    def __init__(self, verbose: bool = False, workspace_gb: int = 16):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.network = None
        self.parser = None
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
        )

    def load_onnx(self, onnx_path: str):
        """Parse ONNX model and configure optimization profiles for all inputs."""
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error(f"Failed to parse ONNX model: {onnx_path}")
                for i in range(self.parser.num_errors):
                    log.error(self.parser.get_error(i))
                sys.exit(1)

        # Log input/output details
        for i in range(self.network.num_inputs):
            inp = self.network.get_input(i)
            log.info(f"Input {i}: {inp.name}, shape: {inp.shape}, dtype: {inp.dtype}")

        # Configure optimization profiles for all dynamic inputs
        profile = self.builder.create_optimization_profile()
        
        for i in range(self.network.num_inputs):
            inp = self.network.get_input(i)
            shape = inp.shape
            name = inp.name.lower()
            
            # Handle different input types based on naming conventions
            if "pixel" in name or "image" in name:  # Vision input
                if len(shape) != 4:
                    log.error(f"Image input {inp.name} has invalid shape {shape}")
                    sys.exit(1)
                
                # Dynamic batch + fixed spatial dimensions
                min_shape = (1, shape[1], shape[2], shape[3])
                opt_shape = (4, shape[1], shape[2], shape[3])
                max_shape = (8, shape[1], shape[2], shape[3])
                
            elif "input_ids" in name or "text" in name:  # Text input
                if len(shape) != 2:
                    log.error(f"Text input {inp.name} has invalid shape {shape}")
                    sys.exit(1)
                
                # Dynamic batch + fixed sequence length
                min_shape = (1, shape[1])
                opt_shape = (4, shape[1])
                max_shape = (8, shape[1])
                
            else:
                log.error(f"Unsupported input: {inp.name} with shape {shape}")
                sys.exit(1)

            log.info(f"Setting profile for {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

        self.config.add_optimization_profile(profile)

    def build_engine(self, model_dir: str, precision: str = "int8"):
        """Build and serialize the TensorRT engine."""
        precision = precision.lower()
        if precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 not supported, falling back to FP16")
                precision = "fp16"
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
        if precision == "fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Engine build failed")
            sys.exit(1)

        # Save to Triton model directory
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)
        plan_path = os.path.join(version_dir, "model.plan")
        
        with open(plan_path, "wb") as f:
            f.write(engine_bytes)
            log.info(f"Engine saved to {plan_path}")

def main():
    parser = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--triton_model_dir", required=True, help="Output directory")
    parser.add_argument("--precision", choices=["int8", "fp16"], default="int8")
    parser.add_argument("--workspace", type=int, default=16, help="Workspace in GB")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    builder = EngineBuilder(verbose=args.verbose, workspace_gb=args.workspace)
    builder.load_onnx(args.onnx)
    builder.build_engine(args.triton_model_dir, args.precision)

if __name__ == "__main__":
    main()