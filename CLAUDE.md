# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` as the package manager and uses Make for common tasks:

- **Setup**: `make setup` - Creates input_images/ and output_depth/ directories
- **Run inference**: `make inference` - Runs depth estimation using `uv run python inference.py`
- **Clean**: `make clean` - Removes output_depth/ directory
- **Install dependencies**: `uv install` (based on pyproject.toml)

The main inference script is `inference.py` which processes images from `input_images/` and saves depth maps to `output_depth/`.

## Architecture Overview

This is a **Depth Anything V2** implementation for monocular depth estimation using Vision Transformers.

### Core Components

- **`depth_anything_v2/dpt.py`**: Main DepthAnythingV2 model implementation using DPT (Dense Prediction Transformer) architecture
- **`depth_anything_v2/dinov2.py`**: DINOv2 backbone encoder integration
- **`depth_anything_v2/util/`**: Utility modules for feature fusion blocks, image transformations, and preprocessing

### Model Architecture

The system uses a DINOv2-DPT architecture:
- **Encoder**: DINOv2 Vision Transformer (vits/vitb/vitl variants) for feature extraction
- **Decoder**: DPT head with feature fusion blocks for dense depth prediction
- **Available models**: vits (24.8M), vitb (97.5M), vitl (335.3M) parameters

### Key Entry Points

- **`inference.py`**: Interactive script for batch image processing with model selection
- **`run.py`**: Command-line script for processing images/directories
- **`app.py`**: Gradio web interface
- **`main.py`**: Minimal example script
- **`metric_depth/`**: Separate module for metric depth estimation (absolute depth values)

### Model Configuration

Models are configured via dictionaries specifying encoder type, feature dimensions, and output channels:
```python
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
```

### Dependencies

- PyTorch >=2.8.0 with torchvision >=0.23.0
- OpenCV >=4.12.0.88 for image processing
- Model checkpoints must be downloaded separately and placed in `checkpoints/` directory

The project follows the standard PyTorch model structure with separate encoder and decoder components, making it easy to swap different backbone models or modify the decoder architecture.