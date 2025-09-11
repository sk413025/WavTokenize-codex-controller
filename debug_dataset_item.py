#!/usr/bin/env python3

"""
Debug script to check what AudioDataset.__getitem__ actually returns
"""

import sys
sys.path.append('/home/sbplab/ruizi/c_code')

from ttdata import AudioDataset
import torch

# Test dataset initialization
input_dirs = [
    'data/raw/box',
    'data/raw/mac', 
    'data/raw/papercup',
    'data/raw/plastic'
]
target_dir = 'data/clean/box2'

print("Creating AudioDataset...")
try:
    dataset = AudioDataset(input_dirs, target_dir)
    print(f"Dataset created successfully with {len(dataset)} items")
    
    print("\nTesting first item...")
    item = dataset[0]
    print(f"Item type: {type(item)}")
    print(f"Item length: {len(item) if hasattr(item, '__len__') else 'No len'}")
    
    if isinstance(item, tuple):
        for i, element in enumerate(item):
            print(f"Element {i}: type={type(element)}, shape={element.shape if hasattr(element, 'shape') else 'No shape'}")
            if hasattr(element, 'dtype'):
                print(f"  dtype: {element.dtype}")
            if isinstance(element, str):
                print(f"  string value: {element}")
    else:
        print(f"Unexpected item type: {type(item)}")
        print(f"Item value: {item}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
