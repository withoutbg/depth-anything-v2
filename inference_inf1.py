#!/usr/bin/env python3
"""
Inference script for compiled Depth Anything V2 models on AWS Inferentia (Inf1) instances.
"""

import os
import sys
import cv2
import glob
import numpy as np
import torch
import torch.neuron
from pathlib import Path

# Hardcoded directories
SOURCE_DIR = "input_images"
TARGET_DIR = "output_depth_inf1"

def load_compiled_model(model_variant, input_size=518):
    """Load compiled model for AWS Inferentia"""
    compiled_model_path = f"compiled_models/depth_anything_v2_{model_variant}_inf1_{input_size}.pt"
    
    if not os.path.exists(compiled_model_path):
        raise FileNotFoundError(f"Compiled model not found: {compiled_model_path}")
    
    print(f"Loading compiled model: {compiled_model_path}")
    compiled_model = torch.jit.load(compiled_model_path)
    
    return compiled_model

def preprocess_image(raw_image, input_size=518):
    """Preprocess image for model input"""
    # Convert BGR to RGB
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Ensure input size is multiple of 14
    input_size = ((input_size + 13) // 14) * 14
    
    # Resize while maintaining aspect ratio
    scale = min(input_size / h, input_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Pad to target size
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad
    
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    # Ensure float32 dtype for Neuron compatibility
    image = image.float()
    
    return image, (h, w), (new_h, new_w), (top_pad, left_pad)

def postprocess_depth(depth_output, original_size, resized_size, pad_info, input_size=518):
    """Postprocess depth output to original image size"""
    h, w = original_size
    new_h, new_w = resized_size
    top_pad, left_pad = pad_info
    
    # Remove batch dimension
    depth = depth_output.squeeze(0).cpu().numpy()
    
    # Remove padding
    depth = depth[top_pad:top_pad+new_h, left_pad:left_pad+new_w]
    
    # Resize to original dimensions
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth

def get_image_files(source_dir):
    """Get all image files from source directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext.upper()), recursive=True))
    
    return sorted(image_files)

def process_image_inf1(compiled_model, image_path, output_path, input_size=518):
    """Process a single image using compiled Inf1 model"""
    # Read image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    try:
        # Preprocess
        image_tensor, original_size, resized_size, pad_info = preprocess_image(raw_image, input_size)
        
        # Run inference on Inf1
        with torch.no_grad():
            depth_output = compiled_model(image_tensor)
        
        # Postprocess
        depth = postprocess_depth(depth_output, original_size, resized_size, pad_info, input_size)
        
        # Normalize depth to 0-255 range
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Save depth map as grayscale
        cv2.imwrite(output_path, depth_normalized)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def get_model_type():
    """Prompt user to select model type"""
    print("Select compiled model type:")
    print("1. vits (smallest, fastest)")
    print("2. vitb (medium)")
    print("3. vitl (largest, most accurate)")
    
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == "1":
            return "vits"
        elif choice == "2":
            return "vitb"
        elif choice == "3":
            return "vitl"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main inference function for Inf1"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with compiled Depth Anything V2 on AWS Inf1')
    parser.add_argument('--model', type=str, choices=['vits', 'vitb', 'vitl'],
                       help='Model variant to use (if not specified, will prompt)')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size used during compilation')
    parser.add_argument('--input-dir', type=str, default=SOURCE_DIR,
                       help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, default=TARGET_DIR,
                       help='Output directory for depth maps')
    
    args = parser.parse_args()
    
    print("=== Depth Anything V2 - AWS Inferentia Inference ===")
    
    # Check if source directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Source directory '{args.input_dir}' does not exist.")
        print(f"Please create the directory and add your images.")
        return 1
    
    # Get model type
    if args.model is None:
        model_type = get_model_type()
    else:
        model_type = args.model
    
    print(f"Using model: {model_type}")
    print(f"Input size: {args.input_size}")
    
    try:
        # Load compiled model
        compiled_model = load_compiled_model(model_type, args.input_size)
        print("Compiled model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading compiled model: {e}")
        print("Make sure you have compiled the model first using compile_for_inf1.py")
        return 1
    
    # Create output directory
    model_output_dir = os.path.join(args.output_dir, f"{model_type}_inf1")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(args.input_dir)
    
    if not image_files:
        print(f"No image files found in '{args.input_dir}'")
        return 1
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: '{model_output_dir}'")
    print("Starting Inf1 inference...")
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Create output filename
        rel_path = os.path.relpath(image_path, args.input_dir)
        output_name = os.path.splitext(rel_path)[0] + '_depth_inf1.png'
        output_path = os.path.join(model_output_dir, output_name)
        
        # Create output subdirectories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process image
        if process_image_inf1(compiled_model, image_path, output_path, args.input_size):
            successful += 1
        else:
            print(f"Failed to process: {image_path}")
    
    print(f"\nInf1 inference completed!")
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    print(f"Results saved to: '{model_output_dir}'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
