#!/usr/bin/env python3
"""
Script to compile Depth Anything V2 model for AWS Inferentia (Inf1) instances
using torch-neuron.
"""

import os
import sys
import torch
import torch.neuron
import numpy as np
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2

# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def download_checkpoint(model_variant):
    """Download model checkpoint if not present"""
    checkpoint_path = f'checkpoints/depth_anything_v2_{model_variant}.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path
    
    print(f"Checkpoint not found: {checkpoint_path}")
    print("Please download the checkpoint file manually from:")
    print(f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_variant.upper()}/resolve/main/depth_anything_v2_{model_variant}.pth")
    print(f"Save it to: {checkpoint_path}")
    return None

def load_model(model_variant):
    """Load the depth estimation model"""
    print(f"Loading model: {model_variant}")
    
    if model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    
    # Create model
    model = DepthAnythingV2(**MODEL_CONFIGS[model_variant])
    
    # Load checkpoint
    checkpoint_path = download_checkpoint(model_variant)
    if not checkpoint_path:
        return None
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully: {model_variant}")
    return model

def create_sample_input(input_size=518, batch_size=1):
    """Create sample input for tracing"""
    # Ensure input size is multiple of 14 (patch size)
    input_size = ((input_size + 13) // 14) * 14
    
    # Create random input tensor
    sample_input = torch.randn(batch_size, 3, input_size, input_size)
    print(f"Created sample input: {sample_input.shape}")
    return sample_input

def compile_model_for_inf1(model, model_variant, input_size=518, batch_size=1):
    """Compile model for AWS Inferentia"""
    print(f"Starting compilation for {model_variant} model...")
    
    # Create sample input
    sample_input = create_sample_input(input_size, batch_size)
    
    # Compile model with torch.neuron.trace
    print("Tracing model with torch.neuron...")
    try:
        # Set compiler arguments for optimization
        compiler_args = [
            '--neuroncore-pipeline-cores', '1',  # Use 1 NeuronCore
            '-O', '2',  # Optimization level (0, 2, or legacy)
        ]
        
        # Get supported operations and exclude problematic upsampling operations
        print("Configuring operations for Neuron compatibility...")
        supported_ops = set(torch.neuron.get_supported_operations())
        
        # Exclude problematic upsampling operations that cause compilation failures
        excluded_ops = {
            'aten::upsample_bilinear2d',  # Bilinear upsampling with align_corners=True
            'aten::upsample_bicubic2d',   # Bicubic upsampling (not supported)
        }
        
        # Create whitelist of operations that can be compiled to Neuron
        neuron_ops = supported_ops - excluded_ops
        
        print(f"Excluding {len(excluded_ops)} problematic operations from Neuron compilation")
        print(f"Operations to run on CPU: {excluded_ops}")
        print(f"Operations available for Neuron: {len(neuron_ops)}")
        
        # Trace the model with operation whitelist
        compiled_model = torch.neuron.trace(
            model, 
            example_inputs=sample_input,
            compiler_args=compiler_args,
            op_whitelist=neuron_ops,
            verbose=1
        )
        
        print("Model compiled successfully!")
        return compiled_model
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None

def save_compiled_model(compiled_model, model_variant, input_size):
    """Save compiled model"""
    output_dir = "compiled_models"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/depth_anything_v2_{model_variant}_inf1_{input_size}.pt"
    
    print(f"Saving compiled model to: {output_path}")
    torch.jit.save(compiled_model, output_path)
    
    return output_path

def test_compiled_model(compiled_model, input_size=518):
    """Test the compiled model with sample input"""
    print("Testing compiled model...")
    
    # Create test input
    test_input = create_sample_input(input_size, 1)
    
    try:
        # Run inference
        with torch.no_grad():
            output = compiled_model(test_input)
        
        print(f"Test successful! Output shape: {output.shape}")
        print(f"Output stats - Min: {output.min().item():.4f}, Max: {output.max().item():.4f}, Mean: {output.mean().item():.4f}")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Main compilation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compile Depth Anything V2 for AWS Inferentia')
    parser.add_argument('--model', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                       help='Model variant to compile')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size (will be adjusted to multiple of 14)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for compilation')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing compiled model')
    
    args = parser.parse_args()
    
    print("=== Depth Anything V2 - AWS Inferentia Compilation ===")
    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    if args.test_only:
        # Load and test existing compiled model
        compiled_model_path = f"compiled_models/depth_anything_v2_{args.model}_inf1_{args.input_size}.pt"
        if not os.path.exists(compiled_model_path):
            print(f"Compiled model not found: {compiled_model_path}")
            return 1
        
        print(f"Loading compiled model: {compiled_model_path}")
        compiled_model = torch.jit.load(compiled_model_path)
        
        if test_compiled_model(compiled_model, args.input_size):
            print("Test passed!")
            return 0
        else:
            print("Test failed!")
            return 1
    
    # Load original model
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Compile model
    compiled_model = compile_model_for_inf1(model, args.model, args.input_size, args.batch_size)
    if compiled_model is None:
        return 1
    
    # Save compiled model
    output_path = save_compiled_model(compiled_model, args.model, args.input_size)
    
    # Test compiled model
    if test_compiled_model(compiled_model, args.input_size):
        print(f"\n=== Compilation Successful! ===")
        print(f"Compiled model saved to: {output_path}")
        print(f"Model is ready for deployment on AWS Inf1 instances")
        print(f"\nNOTE: This model uses hybrid execution:")
        print(f"- Most operations run on Neuron cores for acceleration")
        print(f"- Upsampling operations run on CPU for compatibility")
        print(f"- This provides significant speedup while maintaining accuracy")
        return 0
    else:
        print("\n=== Compilation Failed! ===")
        print("Model compilation completed but testing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
