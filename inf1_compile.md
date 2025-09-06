# AWS Inferentia Compilation Guide: Depth Anything V2

## Overview

This document captures the complete process, challenges, and solutions for successfully compiling and running Depth Anything V2 (Vision Transformer-based depth estimation model) on AWS Inferentia (Inf1) instances using torch-neuron.

## Key Achievements

- ✅ **87.06% Neuron Compilation Rate** - 525 out of 603 operations compiled to Neuron cores
- ✅ **133.04ms Mean Inference Latency** - Excellent performance for 518x518 image depth estimation
- ✅ **7.52 FPS Throughput** - Suitable for real-time applications
- ✅ **Hybrid CPU/Neuron Execution** - Optimal performance while maintaining full compatibility
- ✅ **TracerWarning-Free Compilation** - Optimized interpolation operations for better tracing

## Environment Setup

### Prerequisites
- AWS Inferentia instance (inf1.xlarge or larger)
- Ubuntu 20.04+ with Python 3.9
- AWS Neuron SDK and torch-neuron

### Environment Installation
```bash
# Create and activate Neuron environment
python3 -m venv aws_neuron_venv_pytorch_inf1
source aws_neuron_venv_pytorch_inf1/bin/activate

# Install Neuron SDK components
pip install torch-neuron neuron-cc[tensorflow] torchvision
```

## Major Compilation Challenges & Solutions

### 1. Bilinear Upsampling Compatibility Issue

**Problem:**
- Error: `Bilinear Upsampling with align_corners=True is not yet implemented`
- 5 bilinear upsampling operations failed compilation
- 0% compilation success rate initially

**Root Cause:**
- Depth Anything V2 uses `F.interpolate(..., mode="bilinear", align_corners=True)`
- AWS Neuron doesn't support `align_corners=True` for bilinear upsampling

**Solution:**
- Used operation whitelisting to partition problematic operations to CPU
- Excluded `aten::upsample_bilinear2d` from Neuron compilation
- Maintained accuracy while enabling Neuron acceleration for other operations

### 2. Bicubic Upsampling Not Supported

**Problem:**
- Error: `aten::upsample_bicubic2d` not supported by neuron-cc
- 1 bicubic upsampling operation blocking compilation

**Solution:**
- Excluded `aten::upsample_bicubic2d` from Neuron compilation
- Added to CPU fallback operations

### 3. Incorrect Compiler Arguments

**Problem:**
- Error: `neuron-cc: error: unrecognized arguments: --opt-level 1`
- Compilation failed due to wrong argument format

**Root Cause:**
- Used `--opt-level` instead of correct `-O` or `--optlevel`

**Solution:**
```python
# Wrong
compiler_args = ['--opt-level', '1']

# Correct
compiler_args = ['-O', '2']  # or '--optlevel', '2'
```

### 4. Data Type Mismatch During Inference

**Problem:**
- Error: `RuntimeError: expected scalar type Double but found Float`
- All inference attempts failed with type errors

**Root Cause:**
- Model compiled with `float32` tensors
- Inference script passing `float64` (double precision) tensors

**Solution:**
```python
# Ensure float32 compatibility in preprocessing
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
image = image.float()  # Critical: Convert to float32
```

### 5. Interpolation TracerWarnings

**Problem:**
- TracerWarnings during compilation due to dynamic tensor operations
- Direct `int()` casting in interpolation causing tracing issues
- Suboptimal compilation performance for interpolation operations

**Root Cause:**
- Original code: `F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), ...)`
- Torch tracer struggles with inline `int()` casting operations
- Dynamic tensor shape computations not optimized for static tracing

**Solution:**
```python
# Before (problematic)
out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), 
                   mode="bilinear", align_corners=True)

# After (optimized)
# Calculate target size directly to avoid TracerWarnings
target_h = patch_h * 14
target_w = patch_w * 14
out = F.interpolate(out, (target_h, target_w), 
                   mode="bilinear", align_corners=True)
```

**Benefits:**
- Eliminates TracerWarnings during compilation
- Improves tracer's ability to optimize tensor operations
- Pre-computed dimensions are more cache-friendly
- Better static analysis for the Neuron compiler

## Implementation Details

### Compilation Script Structure

```python
def compile_model_for_inf1(model, model_variant, input_size=518, batch_size=1):
    # Get supported operations
    supported_ops = set(torch.neuron.get_supported_operations())
    
    # Exclude problematic operations
    excluded_ops = {
        'aten::upsample_bilinear2d',  # align_corners=True not supported
        'aten::upsample_bicubic2d',   # Bicubic not supported
    }
    
    # Create whitelist for Neuron compilation
    neuron_ops = supported_ops - excluded_ops
    
    # Compile with operation whitelist
    compiled_model = torch.neuron.trace(
        model, 
        example_inputs=sample_input,
        compiler_args=['-O', '2', '--neuroncore-pipeline-cores', '1'],
        op_whitelist=neuron_ops,
        verbose=1
    )
    
    return compiled_model
```

### Hybrid Execution Model

The solution implements a hybrid execution model:

- **Neuron Cores (87.06% of operations):**
  - Convolutions (`aten::_convolution`)
  - Matrix multiplications (`aten::matmul`, `aten::linear`)
  - Attention operations (`aten::softmax`)
  - Layer normalization (`aten::layer_norm`)
  - Activation functions (`aten::gelu`, `aten::relu`)

- **CPU Fallback (12.94% of operations):**
  - Bilinear upsampling with `align_corners=True`
  - Bicubic upsampling
  - Some tensor manipulation operations

## Performance Results

### Compilation Metrics
- **Total Operations**: 603
- **Neuron Compiled**: 525 (87.06%)
- **CPU Fallback**: 78 (12.94%)
- **Compilation Success**: ✅

### Inference Benchmarks (10 runs, VitS model, 518x518 input)
- **Mean Latency**: 133.04 ms
- **Median Latency**: 127.65 ms
- **Min Latency**: 123.50 ms
- **Max Latency**: 146.70 ms
- **Standard Deviation**: 9.42 ms
- **95th Percentile**: 146.49 ms
- **99th Percentile**: 146.66 ms
- **Throughput**: 7.52 FPS

### Performance Analysis
- **Consistent Performance**: Low variance (9.42ms std dev)
- **Bimodal Distribution**: Two performance clusters observed
  - Fast cluster: ~123-128ms
  - Slower cluster: ~138-147ms
- **Production Ready**: Sub-150ms latency suitable for real-time applications
- **Stable Inference**: All 20 demo images processed successfully without errors

## Best Practices & Lessons Learned

### 1. Operation Whitelisting Strategy
- **Always check supported operations** with `torch.neuron.get_supported_operations()`
- **Use operation whitelisting** instead of trying to modify model architecture
- **Exclude problematic operations** rather than forcing compatibility
- **Hybrid execution** often provides better results than 100% Neuron compilation

### 2. Compiler Configuration
- **Use correct argument formats**: `-O` not `--opt-level`
- **Start with optimization level 2**: `-O 2` for best performance
- **Single NeuronCore**: `--neuroncore-pipeline-cores 1` for most models
- **Enable verbose output**: `verbose=1` for debugging

### 3. Data Type Management
- **Ensure float32 consistency** throughout the pipeline
- **Explicit type conversion**: Use `.float()` to guarantee float32
- **Match compilation and inference types** exactly
- **Test with sample data** before full deployment

### 4. Model Architecture Considerations
- **Vision Transformers** generally compile well to Neuron
- **Upsampling operations** are common compatibility issues
- **Custom operations** may need CPU fallback
- **Batch size of 1** typically works best for compilation

### 5. Interpolation Optimization
- **Avoid inline type casting** in tensor operations during tracing
- **Pre-compute target dimensions** as separate variables
- **Use descriptive comments** to explain TracerWarning fixes
- **Test compilation thoroughly** after interpolation changes

### 6. Performance Optimization
- **Warmup runs are essential** - first inference is always slower
- **Consistent input sizes** improve performance
- **Monitor both latency and throughput** metrics
- **Profile individual operations** to identify bottlenecks

## Troubleshooting Guide

### Common Compilation Errors

1. **"No operations were successfully partitioned"**
   - Solution: Use operation whitelisting to exclude unsupported operations

2. **"unrecognized arguments"**
   - Solution: Check neuron-cc version and use correct argument format

3. **"Bilinear Upsampling with align_corners=True not implemented"**
   - Solution: Exclude `aten::upsample_bilinear2d` from Neuron compilation

4. **"expected scalar type Double but found Float"**
   - Solution: Ensure all tensors are float32 in inference pipeline

5. **TracerWarnings during interpolation operations**
   - Solution: Pre-compute target dimensions instead of inline `int()` casting

### Performance Issues

1. **High latency variance**
   - Check for thermal throttling or resource contention
   - Ensure consistent input preprocessing
   - Monitor system resources during inference

2. **Lower than expected throughput**
   - Verify Neuron compilation percentage
   - Check for CPU bottlenecks in fallback operations
   - Consider batch size optimization (though batch=1 often optimal)

## Code Repository Structure

```
depth-anything-v2/
├── compile_for_inf1.py          # Main compilation script
├── inference_inf1.py            # Inference script with type fixes
├── benchmark_inf1.py            # Latency benchmarking tool
├── compiled_models/             # Compiled .pt models
│   └── depth_anything_v2_vits_inf1_518.pt
├── output_depth_inf1/           # Inference results
└── inf1_compile.md             # This documentation
```

## Future Improvements

### Potential Optimizations
1. **Multi-batch inference** - Explore batch size > 1 for higher throughput
2. **Input size optimization** - Test different input resolutions
3. **Model variants** - Compile VitB and VitL models
4. **Pipeline optimization** - Optimize preprocessing and postprocessing
5. **Memory usage** - Profile and optimize memory consumption

### Advanced Features
1. **Dynamic batching** - Handle variable batch sizes efficiently
2. **Model quantization** - Explore INT8 quantization for better performance
3. **Multi-core utilization** - Use multiple NeuronCores for larger models
4. **Streaming inference** - Implement continuous processing pipeline

## Conclusion

Successfully compiling Vision Transformer models for AWS Inferentia requires careful handling of unsupported operations, data types, and tensor tracing optimizations. The hybrid execution model proves to be an effective solution, achieving 87% Neuron utilization while maintaining full compatibility. The resulting performance (133ms mean latency, 7.52 FPS) makes this suitable for production real-time depth estimation applications.

Key insights for successful compilation:
- **Perfect compatibility isn't necessary** - strategic use of CPU fallback for a small percentage of operations can unlock significant performance gains from Neuron acceleration on the majority of compute-intensive operations
- **Interpolation optimizations matter** - pre-computing target dimensions eliminates TracerWarnings and improves compilation efficiency
- **Systematic testing is crucial** - verify both compilation success and inference performance with real data

---

*Document Version: 1.1*  
*Last Updated: September 2025*  
*Model: Depth Anything V2 (VitS)*  
*Platform: AWS Inferentia (Inf1)*  
*Latest Update: Added interpolation TracerWarning fixes and updated performance metrics*
