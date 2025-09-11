#!/usr/bin/env python3

import torch
import torchaudio
import os

def test_audio_loading():
    """Test audio loading directly"""
    print("Testing audio loading...")
    
    # Test a specific audio file
    test_file = "/home/sbplab/ruizi/data/alldata_40ms/box/nor_boy1_box_LDV_001.wav"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        # Try to find any wav file
        box_dir = "/home/sbplab/ruizi/data/alldata_40ms/box"
        if os.path.exists(box_dir):
            wav_files = [f for f in os.listdir(box_dir) if f.endswith('.wav')]
            if wav_files:
                test_file = os.path.join(box_dir, wav_files[0])
                print(f"Using: {test_file}")
            else:
                print("No wav files found")
                return
        else:
            print(f"Box directory not found: {box_dir}")
            return
    
    try:
        # Load audio file
        wav, sr = torchaudio.load(test_file)
        print(f"Audio loaded successfully:")
        print(f"  Shape: {wav.shape}")
        print(f"  Sample rate: {sr}")
        print(f"  Type: {type(wav)}")
        print(f"  Dtype: {wav.dtype}")
        
        # Test conversion
        from torchaudio.transforms import Resample
        
        if sr != 24000:
            resample = Resample(sr, 24000)
            wav_resampled = resample(wav)
            print(f"Resampled shape: {wav_resampled.shape}")
        else:
            wav_resampled = wav
            
        # Test normalization
        if wav_resampled.abs().max() > 0:
            wav_norm = wav_resampled / (wav_resampled.abs().max() + 1e-8)
            print(f"Normalized shape: {wav_norm.shape}")
        else:
            wav_norm = wav_resampled
            
        print("Audio processing successful!")
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_loading()
