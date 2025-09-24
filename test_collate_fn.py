#!/usr/bin/env python3
"""
Test script to verify the collate_fn correctly handles effective_attention_mask
"""

import torch
from datasets import Dataset
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainable_medical_coding.data.dataloader import BaseDataset
from explainable_medical_coding.utils.tokenizer import TargetTokenizer

def test_collate_fn():
    """Test that collate_fn correctly converts list of effective masks to tensor"""
    
    print("Testing collate_fn with effective_attention_mask...")
    
    # Mock data setup
    batch_size = 3
    seq_len = 8
    num_classes = 4
    
    # Create mock dataset
    mock_data = {
        "input_ids": [[1, 2, 3, 4, 0, 0, 0, 0] for _ in range(batch_size)],
        "attention_mask": [[1, 1, 1, 1, 0, 0, 0, 0] for _ in range(batch_size)],
        "target_ids": [[0, 1], [1, 2], [0, 2, 3]],
        "length": [4, 4, 4],
        "_id": [0, 1, 2],
        "text": ["sample text"] * batch_size,
        "targets": [["class_0", "class_1"], ["class_1", "class_2"], ["class_0", "class_2", "class_3"]],
        "effective_attention_mask": []
    }
    
    # Create effective attention masks (list of tensors)
    for i in range(batch_size):
        mask = torch.zeros(num_classes, seq_len)
        # Set different patterns for each example
        if i == 0:
            mask[0, :2] = 1  # Class 0: first 2 tokens
            mask[1, 2:4] = 1  # Class 1: tokens 2-3
        elif i == 1:
            mask[1, :3] = 1  # Class 1: first 3 tokens
            mask[2, 3:5] = 1  # Class 2: tokens 3-4
        else:  # i == 2
            mask[0, :1] = 1  # Class 0: first token
            mask[2, 1:3] = 1  # Class 2: tokens 1-2
            mask[3, 4:6] = 1  # Class 3: tokens 4-5
        
        mock_data["effective_attention_mask"].append(mask)
    
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_dict(mock_data)
    
    # Create target tokenizer (mock)
    target_tokenizer = TargetTokenizer(autoregressive=False)
    target_tokenizer.fit([["class_0", "class_1", "class_2", "class_3"]])
    
    # Create dataset
    dataset = BaseDataset(hf_dataset, target_tokenizer, pad_token_id=0)
    
    # Test collate function
    indices = [0, 1, 2]  # All three examples
    batch = dataset.collate_fn(indices)
    
    print(f"✓ Batch created successfully")
    print(f"  Input IDs shape: {batch.input_ids.shape}")
    print(f"  Targets shape: {batch.targets.shape}")
    print(f"  Attention masks shape: {batch.attention_masks.shape}")
    
    if batch.effective_attention_mask is not None:
        print(f"  Effective attention mask shape: {batch.effective_attention_mask.shape}")
        expected_shape = (batch_size, num_classes, seq_len)
        
        if batch.effective_attention_mask.shape == expected_shape:
            print(f"  ✓ Effective mask has correct shape: {expected_shape}")
            
            # Test device movement
            if torch.cuda.is_available():
                device = "cuda"
                batch_gpu = batch.to(device)
                if batch_gpu.effective_attention_mask.device.type == device:
                    print(f"  ✓ Effective mask successfully moved to {device}")
                else:
                    print(f"  ✗ Failed to move effective mask to {device}")
            else:
                print("  ✓ CUDA not available, skipping GPU test")
            
            # Test that different examples have different patterns
            mask_0 = batch.effective_attention_mask[0]  # First example
            mask_1 = batch.effective_attention_mask[1]  # Second example
            
            if not torch.equal(mask_0, mask_1):
                print("  ✓ Different examples have different attention patterns")
            else:
                print("  ✗ Examples have identical attention patterns")
                
        else:
            print(f"  ✗ Incorrect shape: expected {expected_shape}, got {batch.effective_attention_mask.shape}")
    else:
        print("  ✗ Effective attention mask is None")
    
    print("\nCollate function test completed!")

if __name__ == "__main__":
    test_collate_fn()