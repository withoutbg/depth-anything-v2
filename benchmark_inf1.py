#!/usr/bin/env python3
"""
Benchmark script for Depth Anything V2 compiled models on AWS Inferentia (Inf1) instances.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.neuron
from pathlib import Path

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
    
    return image

def benchmark_model(compiled_model, input_tensor, num_runs=30, warmup_runs=5):
    """Benchmark the compiled model"""
    print(f"Starting benchmark with {num_runs} runs (after {warmup_runs} warmup runs)...")
    
    # Warmup runs
    print("Performing warmup runs...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = compiled_model(input_tensor)
        print(f"Warmup {i+1}/{warmup_runs} completed")
    
    # Actual benchmark runs
    print("\nStarting benchmark runs...")
    inference_times = []
    
    for i in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        with torch.no_grad():
            output = compiled_model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        print(f"Run {i+1:2d}/{num_runs}: {inference_time:.2f}ms")
    
    return inference_times, output

def calculate_statistics(inference_times):
    """Calculate benchmark statistics"""
    inference_times = np.array(inference_times)
    
    stats = {
        'mean': np.mean(inference_times),
        'median': np.median(inference_times),
        'std': np.std(inference_times),
        'min': np.min(inference_times),
        'max': np.max(inference_times),
        'p95': np.percentile(inference_times, 95),
        'p99': np.percentile(inference_times, 99)
    }
    
    return stats

def main():
    """Main benchmarking function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Depth Anything V2 on AWS Inf1')
    parser.add_argument('--model', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                       help='Model variant to benchmark')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size')
    parser.add_argument('--num-runs', type=int, default=30,
                       help='Number of benchmark runs')
    parser.add_argument('--warmup-runs', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--test-image', type=str, default='assets/examples/demo01.jpg',
                       help='Test image path')
    
    args = parser.parse_args()
    
    print("=== Depth Anything V2 - AWS Inferentia Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Test image: {args.test_image}")
    print()
    
    # Load compiled model
    try:
        compiled_model = load_compiled_model(args.model, args.input_size)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Load and preprocess test image
    if not os.path.exists(args.test_image):
        print(f"Test image not found: {args.test_image}")
        return 1
    
    print(f"Loading test image: {args.test_image}")
    raw_image = cv2.imread(args.test_image)
    if raw_image is None:
        print(f"Failed to load image: {args.test_image}")
        return 1
    
    input_tensor = preprocess_image(raw_image, args.input_size)
    print(f"Input tensor shape: {input_tensor.shape}")
    print()
    
    # Run benchmark
    try:
        inference_times, output = benchmark_model(
            compiled_model, 
            input_tensor, 
            args.num_runs, 
            args.warmup_runs
        )
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output stats - Min: {output.min().item():.4f}, Max: {output.max().item():.4f}, Mean: {output.mean().item():.4f}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    # Calculate and display statistics
    stats = calculate_statistics(inference_times)
    
    print(f"\n=== Benchmark Results ===")
    print(f"Number of runs: {args.num_runs}")
    print(f"Mean latency:   {stats['mean']:.2f} ms")
    print(f"Median latency: {stats['median']:.2f} ms")
    print(f"Std deviation:  {stats['std']:.2f} ms")
    print(f"Min latency:    {stats['min']:.2f} ms")
    print(f"Max latency:    {stats['max']:.2f} ms")
    print(f"95th percentile: {stats['p95']:.2f} ms")
    print(f"99th percentile: {stats['p99']:.2f} ms")
    print()
    print(f"Throughput: {1000/stats['mean']:.2f} FPS")
    
    # Display all individual times
    print(f"\n=== Individual Run Times ===")
    for i, time_ms in enumerate(inference_times):
        print(f"Run {i+1:2d}: {time_ms:.2f} ms")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
