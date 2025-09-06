# Depth Anything V2 Inference Makefile

.PHONY: inference help clean setup

# Default target
help:
	@echo "Available commands:"
	@echo "  inference  - Run depth estimation inference on images"
	@echo "  setup      - Create input/output directories"
	@echo "  clean      - Remove output directory"
	@echo "  help       - Show this help message"

# Run inference
inference:
	@echo "Starting Depth Anything V2 inference..."
	@echo "Source directory: input_images/"
	@echo "Target directory: output_depth/"
	@echo ""
	uv run python inference.py

# Setup directories
setup:
	@echo "Creating directories..."
	mkdir -p input_images
	mkdir -p output_depth
	@echo "Directories created:"
	@echo "  input_images/ - Place your input images here"
	@echo "  output_depth/ - Depth maps will be saved here"

# Clean output directory
clean:
	@echo "Cleaning output directory..."
	rm -rf output_depth/
	@echo "Output directory cleaned."