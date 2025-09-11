#!/usr/bin/env python3

import sys
import os
import torch
from ttdata import AudioDataset

def test_dataset():
    """Test the AudioDataset to see what it returns"""
    print("Testing AudioDataset...")
    
    # Dataset paths
    input_dirs = [
        '/home/sbplab/ruizi/data/alldata_40ms/box',
        '/home/sbplab/ruizi/data/alldata_40ms/mac', 
        '/home/sbplab/ruizi/data/alldata_40ms/papercup',
        '/home/sbplab/ruizi/data/alldata_40ms/plastic'
    ]
    
    target_dir = '/home/sbplab/ruizi/data/alldata_40ms/clean'
    
    # Create dataset
    dataset = AudioDataset(input_dirs, target_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test first few items
    for i in range(min(3, len(dataset))):
        print(f"\nTesting item {i}:")
        try:
            item = dataset[i]
            print(f"Item type: {type(item)}")
            print(f"Item length: {len(item)}")
            
            if len(item) == 3:
                input_wav, target_wav, content_id = item
                print(f"Input wav type: {type(input_wav)}, shape: {input_wav.shape if hasattr(input_wav, 'shape') else 'No shape'}")
                print(f"Target wav type: {type(target_wav)}, shape: {target_wav.shape if hasattr(target_wav, 'shape') else 'No shape'}")
                print(f"Content ID type: {type(content_id)}, value: {content_id}")
            else:
                print(f"Unexpected item structure: {item}")
                
        except Exception as e:
            print(f"Error loading item {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
