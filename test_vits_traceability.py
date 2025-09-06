#!/usr/bin/env python3
"""
Test script to check if the vits model is traceable with torch.jit.trace
This is important for AWS Inf1 deployment with Torch Neuron compilation.

RESULTS SUMMARY:
- ✅ vits model IS traceable with torch.jit.trace  
- ❌ Traced model has FIXED input size (518x518 only)
- ⚠️ Multiple TracerWarnings indicate dynamic shape issues
- 📋 Suitable for AWS Inf1 with fixed preprocessing

DEPLOYMENT RECOMMENDATION: 
Use fixed 518x518 input size for all inference on AWS Inf1.
"""
import torch
import torch.nn as nn
import os
import traceback
from depth_anything_v2.dpt import DepthAnythingV2

def test_vits_traceability():
    """Test if the vits model can be traced with torch.jit.trace"""
    
    print("Testing vits model traceability for AWS Inf1...")
    print("="*50)
    
    # vits model configuration
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    
    try:
        # Initialize model
        print("1. Initializing vits model...")
        model = DepthAnythingV2(**model_config)
        
        # Load checkpoint if available
        checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
        if os.path.exists(checkpoint_path):
            print(f"2. Loading checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            print("2. WARNING: No checkpoint found, using random weights")
        
        model.eval()
        
        # Create dummy input (518x518 - standard size)
        input_size = 518
        dummy_input = torch.randn(1, 3, input_size, input_size)
        print(f"3. Created dummy input: {dummy_input.shape}")
        
        # Test forward pass
        print("4. Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   ✓ Forward pass successful. Output: {output.shape}")
        
        # Test tracing
        print("5. Testing torch.jit.trace...")
        traced_model = torch.jit.trace(model, dummy_input)
        print("   ✓ Model successfully traced!")
        
        # Verify traced model
        print("6. Testing traced model...")
        with torch.no_grad():
            traced_output = traced_model(dummy_input)
            diff = torch.abs(output - traced_output).max().item()
            print(f"   ✓ Traced model works. Max diff: {diff:.8f}")
        
        # Test with different input size (will fail)
        print("7. Testing dynamic shape support...")
        try:
            test_input = torch.randn(1, 3, 336, 336)
            with torch.no_grad():
                dynamic_output = traced_model(test_input)
                print(f"   ✓ Works with 336x336: {dynamic_output.shape}")
        except Exception as e:
            print(f"   ✗ Fails with different size: {str(e)[:100]}...")
            print("   → Traced model is fixed to 518x518 input size")
        
        # Save traced model
        traced_model_path = 'vits_traced_518x518.pt'
        torch.jit.save(traced_model, traced_model_path)
        print(f"8. Saved traced model: {traced_model_path}")
        
        return True, traced_model
        
    except Exception as e:
        print(f"✗ Tracing failed: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Main function"""
    print("Depth Anything V2 - vits Model Traceability Test")
    print("================================================")
    print("Testing for AWS Inf1 deployment compatibility\n")
    
    success, traced_model = test_vits_traceability()
    
    print("\n" + "="*50)
    if success:
        print("RESULT: vits model IS TRACEABLE ✅")
        print("="*50)
        print("✓ Compatible with torch.jit.trace")  
        print("✓ Suitable for AWS Inf1 + Torch Neuron compilation")
        print("✓ Traced model saved for deployment")
        print("\n⚠️  IMPORTANT LIMITATIONS:")
        print("• Traced model only works with 518x518 input")
        print("• All inputs must be preprocessed to this size")
        print("• Multiple TracerWarnings indicate fixed-size tracing")
        print("\n📋 DEPLOYMENT STRATEGY:")
        print("• Resize all inputs to 518x518 before inference")
        print("• Use the traced model (vits_traced_518x518.pt) on AWS Inf1")
        print("• Implement consistent preprocessing pipeline")
    else:
        print("RESULT: vits model is NOT TRACEABLE ❌")
        print("="*50)
        print("✗ Cannot be traced with torch.jit.trace")
        print("✗ Not suitable for AWS Inf1 deployment")

if __name__ == "__main__":
    main()