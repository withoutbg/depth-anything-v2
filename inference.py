#!/usr/bin/env python3
import os
import sys
import cv2
import glob
import numpy as np
import torch
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2

# Hardcoded directories
SOURCE_DIR = "input_images"
TARGET_DIR = "output_depth"

def get_model_type():
    """Prompt user to select model type"""
    print("Select model type:")
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

def load_model(encoder):
    """Load the depth estimation model"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder: {encoder}")
    
    print(f"Loading model: {encoder}")
    print(f"Using device: {DEVICE}")
    
    # Load model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    return depth_anything, DEVICE

def get_image_files(source_dir):
    """Get all image files from source directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext.upper()), recursive=True))
    
    return sorted(image_files)

def process_image(depth_anything, image_path, output_path, input_size=518):
    """Process a single image"""
    # Read image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    # Run inference
    depth = depth_anything.infer_image(raw_image, input_size)
    
    # Normalize depth to 0-255 range
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    # Save depth map as grayscale
    cv2.imwrite(output_path, depth)
    return True

def main():
    """Main inference function"""
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print(f"Please create the directory and add your images.")
        return 1
    
    # Get model type from user
    model_type = get_model_type()
    
    try:
        # Load model
        depth_anything, device = load_model(model_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Create output directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(SOURCE_DIR)
    
    if not image_files:
        print(f"No image files found in '{SOURCE_DIR}'")
        return 1
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: '{TARGET_DIR}'")
    print("Starting inference...")
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Create output filename
        rel_path = os.path.relpath(image_path, SOURCE_DIR)
        output_name = os.path.splitext(rel_path)[0] + '_depth.png'
        output_path = os.path.join(TARGET_DIR, output_name)
        
        # Create output subdirectories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process image
        if process_image(depth_anything, image_path, output_path):
            successful += 1
        else:
            print(f"Failed to process: {image_path}")
    
    print(f"\nInference completed!")
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    print(f"Results saved to: '{TARGET_DIR}'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())