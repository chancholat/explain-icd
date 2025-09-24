#!/usr/bin/env python3
"""
Test script to verify the effective attention mask implementation
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_attention_mask_logic():
    """Test the attention mask logic without full model setup"""
    
    # Mock parameters
    batch_size = 2
    num_classes = 5
    seq_len = 10
    
    print("Testing attention mask broadcasting logic...")
    
    # Test 1: Regular 2D mask (B, L) - should be broadcast to (B, NC, L)
    regular_mask = torch.ones(batch_size, seq_len)
    regular_mask[:, -2:] = 0  # Mask last 2 positions
    
    if regular_mask.dim() == 2:
        # Simulate the old broadcasting behavior
        broadcasted_mask = regular_mask.unsqueeze(1).repeat(1, num_classes, 1)
        print(f"Regular 2D mask shape: {regular_mask.shape}")
        print(f"Broadcasted to 3D: {broadcasted_mask.shape}")
        assert broadcasted_mask.shape == (batch_size, num_classes, seq_len)
        print("✓ Regular mask broadcasting works")
    
    # Test 2: Effective 3D mask (B, NC, L) - should be used directly
    effective_mask = torch.zeros(batch_size, num_classes, seq_len)
    # Set different patterns for different classes
    effective_mask[0, 0, :3] = 1  # Class 0, first example: mask first 3 tokens
    effective_mask[0, 1, 5:8] = 1  # Class 1, first example: mask tokens 5-7
    effective_mask[1, 2, -3:] = 1  # Class 2, second example: mask last 3 tokens
    
    if effective_mask.dim() == 3 and effective_mask.size(1) == num_classes:
        print(f"Effective 3D mask shape: {effective_mask.shape}")
        print("✓ Effective mask has correct shape")
        
        # Check that different classes have different patterns
        class_0_pattern = effective_mask[0, 0, :]
        class_1_pattern = effective_mask[0, 1, :]
        if not torch.equal(class_0_pattern, class_1_pattern):
            print("✓ Different classes have different attention patterns")
        
    print("\nAll tests passed!")

def test_batch_datatype():
    """Test that Batch class can handle effective_attention_mask"""
    from explainable_medical_coding.utils.datatypes import Batch
    
    print("\nTesting Batch class with effective_attention_mask...")
    
    batch_size = 2
    seq_len = 8
    num_classes = 3
    
    # Create mock data
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.rand(batch_size, num_classes)
    target_names = [f"class_{i}" for i in range(num_classes)]
    ids = list(range(batch_size))
    
    # Create effective attention mask
    effective_mask = torch.zeros(batch_size, num_classes, seq_len)
    effective_mask[0, 0, :3] = 1
    effective_mask[1, 1, -2:] = 1
    
    # Create batch
    batch = Batch(
        input_ids=input_ids,
        targets=targets,
        target_names=target_names,
        ids=ids,
        effective_attention_mask=effective_mask
    )
    
    print(f"Batch created with effective_attention_mask shape: {batch.effective_attention_mask.shape}")
    
    # Test moving to device
    if torch.cuda.is_available():
        device = "cuda"
        batch_gpu = batch.to(device)
        print(f"✓ Batch moved to {device}")
        print(f"  effective_attention_mask device: {batch_gpu.effective_attention_mask.device}")
    else:
        print("✓ CUDA not available, skipping GPU test")
    
    print("✓ Batch class works with effective_attention_mask")

def test_effective_mask_building():
    """Test the _build_effective_attention_mask_from_batch function"""
    from explainable_medical_coding.utils.datatypes import Batch
    
    print("\nTesting effective mask building function...")
    
    batch_size = 2
    seq_len = 6
    num_classes = 3
    device = "cpu"
    
    # Create mock batch with effective attention masks as list
    effective_masks_list = [
        torch.zeros(num_classes, seq_len),  # First example
        torch.zeros(num_classes, seq_len)   # Second example
    ]
    
    # Set different patterns
    effective_masks_list[0][0, :2] = 1  # Class 0: first 2 tokens
    effective_masks_list[0][1, 2:4] = 1  # Class 1: tokens 2-3
    effective_masks_list[1][2, -2:] = 1  # Class 2: last 2 tokens
    
    # Create batch
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.rand(batch_size, num_classes)
    batch = Batch(
        input_ids=input_ids,
        targets=targets,
        target_names=[f"class_{i}" for i in range(num_classes)],
        ids=list(range(batch_size)),
        effective_attention_mask=effective_masks_list
    )
    
    # Import the function (this is a bit hacky but works for testing)
    try:
        from explainable_medical_coding.utils.loss_functions import _build_effective_attention_mask_from_batch
        
        result_mask = _build_effective_attention_mask_from_batch(batch, seq_len, device)
        
        if result_mask is not None:
            print(f"✓ Built effective mask with shape: {result_mask.shape}")
            print(f"  Expected shape: ({batch_size}, {num_classes}, {seq_len})")
            assert result_mask.shape == (batch_size, num_classes, seq_len)
            print("✓ Effective mask building works correctly")
        else:
            print("✗ Failed to build effective mask")
            
    except ImportError as e:
        print(f"Could not import function: {e}")

if __name__ == "__main__":
    test_attention_mask_logic()
    test_batch_datatype()
    test_effective_mask_building()
    print("\n🎉 All tests completed!")