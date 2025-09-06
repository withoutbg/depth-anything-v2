# Depth Anything V2 - AWS Inferentia (Inf1) Compilation Guide

This guide explains how to compile and run Depth Anything V2 models on AWS Inferentia (Inf1) instances using PyTorch Neuron.

## Prerequisites

1. **AWS Inferentia Environment**: Set up using the provided script
   ```bash
   ./setup_neuron_env.sh
   ```

2. **Activate the Neuron Environment**:
   ```bash
   source aws_neuron_venv_pytorch_inf1/bin/activate
   ```

3. **Download Model Checkpoints**: Download the model weights from HuggingFace:
   - [ViT-S (Small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth)
   - [ViT-B (Base)](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth)
   - [ViT-L (Large)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)

   Save them to the `checkpoints/` directory.

## Model Compilation

### Step 1: Compile Model for Inf1

Compile a model for AWS Inferentia using the compilation script:

```bash
# Compile ViT-S model (recommended for first test)
python compile_for_inf1.py --model vits --input-size 518 --batch-size 1

# Compile ViT-B model
python compile_for_inf1.py --model vitb --input-size 518 --batch-size 1

# Compile ViT-L model
python compile_for_inf1.py --model vitl --input-size 518 --batch-size 1
```

**Compilation Options:**
- `--model`: Choose from `vits`, `vitb`, or `vitl`
- `--input-size`: Input image size (default: 518, must be multiple of 14)
- `--batch-size`: Batch size for inference (default: 1)

### Step 2: Test Compiled Model

Test the compiled model without running full inference:

```bash
python compile_for_inf1.py --model vits --test-only
```

## Running Inference

### Step 1: Prepare Images

Create input directory and add your images:
```bash
mkdir -p input_images
# Copy your images to input_images/
```

### Step 2: Run Inf1 Inference

Use the compiled model for inference:

```bash
# Interactive mode (will prompt for model selection)
python inference_inf1.py

# Command line mode
python inference_inf1.py --model vits --input-size 518 --input-dir input_images --output-dir output_depth_inf1
```

**Inference Options:**
- `--model`: Model variant (`vits`, `vitb`, `vitl`)
- `--input-size`: Must match the size used during compilation
- `--input-dir`: Directory containing input images
- `--output-dir`: Directory for output depth maps

## Model Comparison

| Model | Parameters | Speed | Accuracy | Recommended Use |
|-------|------------|-------|----------|----------------|
| ViT-S | 24.8M | Fastest | Good | Real-time applications |
| ViT-B | 97.5M | Medium | Better | Balanced performance |
| ViT-L | 335.3M | Slowest | Best | High-accuracy applications |

## Performance Optimization

### Compilation Tips

1. **Input Size**: Use input sizes that are multiples of 14 (patch size)
2. **Batch Size**: Start with batch size 1, increase if you have multiple images to process simultaneously
3. **NeuronCore Usage**: The script uses 1 NeuronCore by default, adjust in `compile_for_inf1.py` if needed

### Memory Considerations

- **ViT-S**: ~1GB memory usage
- **ViT-B**: ~3GB memory usage  
- **ViT-L**: ~8GB memory usage

## Troubleshooting

### Common Issues

1. **Compilation Fails**:
   - Check that the checkpoint file exists in `checkpoints/`
   - Ensure input size is multiple of 14
   - Verify AWS Neuron environment is activated

2. **Memory Errors**:
   - Try smaller input size (e.g., 364 instead of 518)
   - Use smaller model variant (vits instead of vitl)

3. **Inference Errors**:
   - Ensure input size matches compilation settings
   - Check that compiled model exists in `compiled_models/`

### Environment Issues

If you encounter package conflicts:

```bash
# Reinstall with correct versions
source aws_neuron_venv_pytorch_inf1/bin/activate
pip install opencv-python==4.8.0.74
pip install numpy==1.22.2 --force-reinstall
```

## File Structure

```
depth-anything-v2/
├── aws_neuron_venv_pytorch_inf1/     # Python virtual environment
├── checkpoints/                      # Model checkpoint files
├── compiled_models/                  # Compiled Inf1 models
├── input_images/                     # Input images for inference
├── output_depth_inf1/               # Output depth maps
├── compile_for_inf1.py              # Compilation script
├── inference_inf1.py                # Inf1 inference script
├── setup_neuron_env.sh              # Environment setup script
└── README_INF1.md                   # This file
```

## Deployment Notes

- Compiled models are specific to the input size and batch size used during compilation
- Models compiled on one Inf1 instance can be used on other Inf1 instances
- Keep the virtual environment and compiled models together for deployment
- Consider using smaller models (ViT-S) for production workloads requiring real-time performance

## Next Steps

1. Download model checkpoints
2. Compile your preferred model variant
3. Test with sample images
4. Optimize for your specific use case
5. Deploy to production Inf1 instances
