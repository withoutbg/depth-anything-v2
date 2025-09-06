#!/bin/bash

# Download script for Depth Anything V2 models
# This script downloads the pre-trained models and places them in the checkpoints directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create checkpoints directory if it doesn't exist
CHECKPOINT_DIR="checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${YELLOW}Creating checkpoints directory...${NC}"
    mkdir -p "$CHECKPOINT_DIR"
fi

echo -e "${GREEN}Depth Anything V2 Model Downloader${NC}"
echo "======================================"

# Model information
declare -A models=(
    ["vits"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    ["vitb"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
    ["vitl"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
)

declare -A model_sizes=(
    ["vits"]="24.8M"
    ["vitb"]="97.5M"
    ["vitl"]="335.3M"
)

declare -A model_names=(
    ["vits"]="Depth-Anything-V2-Small"
    ["vitb"]="Depth-Anything-V2-Base"
    ["vitl"]="Depth-Anything-V2-Large"
)

# Function to download a model
download_model() {
    local model_key=$1
    local url=${models[$model_key]}
    local filename="depth_anything_v2_${model_key}.pth"
    local filepath="${CHECKPOINT_DIR}/${filename}"
    local model_name=${model_names[$model_key]}
    local size=${model_sizes[$model_key]}
    
    echo -e "${YELLOW}Downloading ${model_name} (${size})...${NC}"
    
    # Check if file already exists
    if [ -f "$filepath" ]; then
        echo -e "${GREEN}✓ ${filename} already exists, skipping download${NC}"
        return 0
    fi
    
    # Download with wget or curl
    if command -v wget >/dev/null 2>&1; then
        wget -O "$filepath" "$url" --progress=bar:force 2>&1
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$filepath" "$url" --progress-bar
    else
        echo -e "${RED}Error: Neither wget nor curl is available${NC}"
        exit 1
    fi
    
    # Check if download was successful
    if [ $? -eq 0 ] && [ -f "$filepath" ]; then
        echo -e "${GREEN}✓ Successfully downloaded ${filename}${NC}"
    else
        echo -e "${RED}✗ Failed to download ${filename}${NC}"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Download all models (vits, vitb, vitl)"
    echo "  --vits          Download Small model (24.8M)"
    echo "  --vitb          Download Base model (97.5M)"  
    echo "  --vitl          Download Large model (335.3M)"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all        # Download all models"
    echo "  $0 --vits       # Download only the small model"
    echo "  $0 --vitb --vitl # Download base and large models"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No arguments provided. Use --help for usage information.${NC}"
    echo -e "${YELLOW}Would you like to download all models? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        DOWNLOAD_ALL=true
    else
        show_usage
        exit 0
    fi
fi

DOWNLOAD_ALL=false
DOWNLOAD_VITS=false
DOWNLOAD_VITB=false
DOWNLOAD_VITL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --vits)
            DOWNLOAD_VITS=true
            shift
            ;;
        --vitb)
            DOWNLOAD_VITB=true
            shift
            ;;
        --vitl)
            DOWNLOAD_VITL=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Download models based on arguments
if [ "$DOWNLOAD_ALL" = true ]; then
    for model in vits vitb vitl; do
        download_model "$model"
    done
else
    [ "$DOWNLOAD_VITS" = true ] && download_model "vits"
    [ "$DOWNLOAD_VITB" = true ] && download_model "vitb"
    [ "$DOWNLOAD_VITL" = true ] && download_model "vitl"
fi

echo ""
echo -e "${GREEN}Download completed!${NC}"
echo -e "Models are saved in: ${CHECKPOINT_DIR}/"
echo ""
echo "Available models:"
ls -lh "$CHECKPOINT_DIR"/*.pth 2>/dev/null || echo "No .pth files found in checkpoints directory"

echo ""
echo -e "${GREEN}You can now run inference with:${NC}"
echo "  python run.py --encoder vits --img-path your_image.jpg"
echo "  python inference.py"