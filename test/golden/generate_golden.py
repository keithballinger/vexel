#!/usr/bin/env python3
"""
Generate golden test data from TinyLlama using PyTorch/Transformers.

This script runs inference and captures intermediate outputs at each stage
to use as reference data for validating the Go implementation.

Usage:
    python generate_golden.py --model-dir ../../models --output-dir ./data

    Or use HuggingFace model directly:
    python generate_golden.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output-dir ./data
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class OutputCapture:
    """Captures intermediate outputs during forward pass."""

    def __init__(self):
        self.outputs: Dict[str, np.ndarray] = {}
        self.hooks = []

    def _make_hook(self, name: str):
        """Create a hook that captures the output tensor."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Some modules return tuples, take the first element
                tensor = output[0]
            else:
                tensor = output
            # Convert to float32 numpy array
            self.outputs[name] = tensor.detach().float().cpu().numpy()
        return hook

    def register(self, module: torch.nn.Module, name: str):
        """Register a forward hook on a module."""
        hook = module.register_forward_hook(self._make_hook(name))
        self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def capture_layer_internals(model, layer_idx: int, capture: OutputCapture):
    """Register hooks to capture internals of a transformer layer."""
    layer = model.model.layers[layer_idx]
    prefix = f"layer_{layer_idx}"

    # Attention components
    capture.register(layer.input_layernorm, f"{prefix}.input_layernorm")
    capture.register(layer.self_attn.q_proj, f"{prefix}.q_proj")
    capture.register(layer.self_attn.k_proj, f"{prefix}.k_proj")
    capture.register(layer.self_attn.v_proj, f"{prefix}.v_proj")
    capture.register(layer.self_attn.o_proj, f"{prefix}.o_proj")
    capture.register(layer.self_attn, f"{prefix}.self_attn")

    # MLP components
    capture.register(layer.post_attention_layernorm, f"{prefix}.post_attention_layernorm")
    capture.register(layer.mlp.gate_proj, f"{prefix}.gate_proj")
    capture.register(layer.mlp.up_proj, f"{prefix}.up_proj")
    capture.register(layer.mlp.down_proj, f"{prefix}.down_proj")
    capture.register(layer.mlp, f"{prefix}.mlp")

    # Full layer output
    capture.register(layer, f"{prefix}.output")


def setup_local_model(model_dir: str) -> str:
    """
    Set up local model files to be compatible with transformers.
    Returns the path to use for loading.
    """
    model_path = Path(model_dir)

    # Check if files need renaming (tiny_* prefix)
    tiny_config = model_path / "tiny_config.json"
    tiny_model = model_path / "tiny_model.safetensors"
    tiny_tokenizer = model_path / "tiny_tokenizer.json"

    config_json = model_path / "config.json"
    model_safetensors = model_path / "model.safetensors"
    tokenizer_json = model_path / "tokenizer.json"

    # Create symlinks if needed
    links_created = []

    if tiny_config.exists() and not config_json.exists():
        config_json.symlink_to(tiny_config.name)
        links_created.append(config_json)

    if tiny_model.exists() and not model_safetensors.exists():
        model_safetensors.symlink_to(tiny_model.name)
        links_created.append(model_safetensors)

    if tiny_tokenizer.exists() and not tokenizer_json.exists():
        tokenizer_json.symlink_to(tiny_tokenizer.name)
        links_created.append(tokenizer_json)

    if links_created:
        print(f"Created symlinks: {[str(l.name) for l in links_created]}")

    return str(model_path)


def generate_golden_data(
    model_source: str,
    output_dir: str,
    input_tokens: List[int] = None,
    layers_to_capture: List[int] = None,
    is_local: bool = True,
):
    """
    Generate golden test data from the model.

    Args:
        model_source: Directory containing model files, or HuggingFace model name
        output_dir: Directory to save golden data
        input_tokens: Token IDs to use as input (default: [1] - BOS token)
        layers_to_capture: Which layers to capture (default: [0, 1, 2, -1] for first 3 and last)
        is_local: Whether model_source is a local directory
    """
    if input_tokens is None:
        input_tokens = [1]  # BOS token

    if layers_to_capture is None:
        layers_to_capture = [0, 1, 2, 21]  # First 3 and last layer for TinyLlama-1.1B

    os.makedirs(output_dir, exist_ok=True)

    # Set up model path
    if is_local:
        model_path = setup_local_model(model_source)
    else:
        model_path = model_source

    print(f"Loading model from {model_path}...")

    # Load model in float32 for maximum precision
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",  # Use CPU for reproducibility
        trust_remote_code=True,
        local_files_only=is_local,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get model config for metadata
    config = model.config
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    print(f"Model config: hidden_size={config.hidden_size}, "
          f"num_layers={config.num_hidden_layers}, "
          f"num_heads={config.num_attention_heads}, "
          f"num_kv_heads={num_kv_heads}")

    # Set up output capture
    capture = OutputCapture()

    # Register hooks for embedding
    capture.register(model.model.embed_tokens, "embedding")

    # Register hooks for each layer we want to capture
    num_layers = config.num_hidden_layers
    actual_layers = []
    for layer_idx in layers_to_capture:
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if 0 <= layer_idx < num_layers:
            actual_layers.append(layer_idx)
            capture_layer_internals(model, layer_idx, capture)

    # Register hooks for final components
    capture.register(model.model.norm, "final_norm")
    capture.register(model.lm_head, "lm_head")

    # Prepare input
    input_ids = torch.tensor([input_tokens], dtype=torch.long)
    print(f"Input tokens: {input_tokens}")

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    # Capture final logits
    capture.outputs["logits"] = outputs.logits.detach().float().cpu().numpy()

    # Remove hooks
    capture.remove_hooks()

    # Save metadata
    metadata = {
        "model_name": "TinyLlama-1.1B",
        "model_source": model_path,
        "input_tokens": input_tokens,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rms_norm_eps": config.rms_norm_eps,
        "layers_captured": actual_layers,
        "outputs": list(capture.outputs.keys()),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Save each captured output
    for name, array in capture.outputs.items():
        # Save as numpy binary for precision
        npy_path = os.path.join(output_dir, f"{name}.npy")
        np.save(npy_path, array)

        # Also save summary stats as JSON for quick inspection
        stats = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "first_10": array.flatten()[:10].tolist(),
            "last_10": array.flatten()[-10:].tolist(),
        }
        json_path = os.path.join(output_dir, f"{name}.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Saved {name}: shape={array.shape}, min={array.min():.6f}, max={array.max():.6f}")

    print(f"\nGolden data saved to {output_dir}")
    print(f"Total outputs captured: {len(capture.outputs)}")

    return capture.outputs


def main():
    parser = argparse.ArgumentParser(description="Generate golden test data from TinyLlama")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local directory containing model files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model name (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save golden data",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="1",
        help="Comma-separated token IDs to use as input",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,1,2,-1",
        help="Comma-separated layer indices to capture (-1 for last)",
    )

    args = parser.parse_args()

    if args.model_dir is None and args.model_name is None:
        args.model_dir = "../../models"

    input_tokens = [int(t.strip()) for t in args.tokens.split(",")]
    layers = [int(l.strip()) for l in args.layers.split(",")]

    if args.model_dir:
        generate_golden_data(
            model_source=args.model_dir,
            output_dir=args.output_dir,
            input_tokens=input_tokens,
            layers_to_capture=layers,
            is_local=True,
        )
    else:
        generate_golden_data(
            model_source=args.model_name,
            output_dir=args.output_dir,
            input_tokens=input_tokens,
            layers_to_capture=layers,
            is_local=False,
        )


if __name__ == "__main__":
    main()
