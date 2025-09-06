# Depth Anything V2 Inference Makefile

.PHONY: inference help clean setup download-model upload-model upload-output

# Default target
help:
	@echo "Available commands:"
	@echo "  inference     - Run depth estimation inference on images"
	@echo "  setup         - Create input/output directories"
	@echo "  download-model - Download pre-trained model checkpoints"
	@echo "  upload-model  - Upload neuron model checkpoint to S3 bucket"
	@echo "  upload-output - Upload output_depth_inf1 directory to S3 bucket"
	@echo "  clean         - Remove output directory"
	@echo "  help          - Show this help message"

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

# Download model checkpoints
download-model:
	@echo "Downloading Depth Anything V2 model checkpoints..."
	@chmod +x download_models.sh
	./download_models.sh --all

# Upload neuron model checkpoint to S3
upload-model:
	@echo "Uploading neuron model checkpoint to S3..."
	@if [ ! -f "compiled_models/depth_anything_v2_vits_inf1_518.pt" ]; then \
		echo "Error: Model checkpoint not found at compiled_models/depth_anything_v2_vits_inf1_518.pt"; \
		exit 1; \
	fi
	aws s3 cp compiled_models/depth_anything_v2_vits_inf1_518.pt s3://wbg-model-checkpoints/
	@echo "Model uploaded successfully to s3://wbg-model-checkpoints/"

# Upload output_depth_inf1 directory to S3
upload-output:
	@echo "Uploading output_depth_inf1 directory to S3..."
	@if [ ! -d "output_depth_inf1" ]; then \
		echo "Error: output_depth_inf1 directory not found"; \
		exit 1; \
	fi
	aws s3 cp output_depth_inf1/ s3://wbg-model-checkpoints/output_depth_inf1/ --recursive
	@echo "Output directory uploaded successfully to s3://wbg-model-checkpoints/output_depth_inf1/"

# Clean output directory
clean:
	@echo "Cleaning output directory..."
	rm -rf output_depth/
	@echo "Output directory cleaned."