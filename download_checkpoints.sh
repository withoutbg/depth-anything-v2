#!/bin/bash

# Download script for Depth Anything V2 model checkpoints

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "=== Downloading Depth Anything V2 Model Checkpoints ==="

# Base URLs
BASE_URL_VITS="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
BASE_URL_VITB="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
BASE_URL_VITL="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

# Function to download with progress
download_checkpoint() {
    local url=$1
    local filename=$2
    local filepath="$CHECKPOINT_DIR/$filename"
    
    if [ -f "$filepath" ]; then
        echo "✓ $filename already exists, skipping download"
        return 0
    fi
    
    echo "Downloading $filename..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$filepath" "$url" --progress=bar:force 2>&1
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$filepath" "$url" --progress-bar
    else
        echo "Error: Neither wget nor curl is available"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $filename"
    else
        echo "✗ Failed to download $filename"
        rm -f "$filepath"
        return 1
    fi
}

# Download models
echo
echo "1. Downloading ViT-S model (24.8M parameters) - Recommended for testing"
download_checkpoint "$BASE_URL_VITS" "depth_anything_v2_vits.pth"

echo
echo "2. Downloading ViT-B model (97.5M parameters)"
download_checkpoint "$BASE_URL_VITB" "depth_anything_v2_vitb.pth"

echo
echo "3. Downloading ViT-L model (335.3M parameters)"
download_checkpoint "$BASE_URL_VITL" "depth_anything_v2_vitl.pth"

echo
echo "=== Download Summary ==="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Available models:"
for model in vits vitb vitl; do
    filepath="$CHECKPOINT_DIR/depth_anything_v2_${model}.pth"
    if [ -f "$filepath" ]; then
        size=$(du -h "$filepath" | cut -f1)
        echo "  ✓ $model - $size"
    else
        echo "  ✗ $model - Not downloaded"
    fi
done

echo
echo "Next steps:"
echo "1. Activate the Neuron environment: source aws_neuron_venv_pytorch_inf1/bin/activate"
echo "2. Compile a model: python compile_for_inf1.py --model vits"
echo "3. Test inference: python inference_inf1.py --model vits"
