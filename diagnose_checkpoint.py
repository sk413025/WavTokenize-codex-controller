#!/usr/bin/env python3
"""
診斷 Checkpoint 問題
檢查模型是否正確儲存和載入
"""

import torch
import sys

print("="*70)
print("診斷 Checkpoint - Epoch 400")
print("="*70)

checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359/checkpoint_epoch_400.pth"

print(f"\n[1/3] 載入 checkpoint...")
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint 載入成功")
    
    print(f"\n[2/3] 檢查 checkpoint 內容...")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n[3/3] 檢查模型參數...")
        print(f"   總參數數量: {len(state_dict)}")
        
        # 檢查關鍵層
        transformer_keys = [k for k in state_dict.keys() if 'transformer' in k]
        print(f"   Transformer 相關參數: {len(transformer_keys)}")
        
        if transformer_keys:
            print(f"\n   前 10 個 Transformer 參數:")
            for key in transformer_keys[:10]:
                param = state_dict[key]
                print(f"      {key}: shape={param.shape}, dtype={param.dtype}")
        
        # 檢查 encoder/decoder
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
        print(f"\n   Encoder 參數: {len(encoder_keys)}")
        print(f"   Decoder 參數: {len(decoder_keys)}")
        
        # 檢查 wavtokenizer
        wavtokenizer_keys = [k for k in state_dict.keys() if 'wavtokenizer' in k]
        print(f"   WavTokenizer 參數: {len(wavtokenizer_keys)}")
        
        print("\n" + "="*70)
        print("📊 診斷總結")
        print("="*70)
        
        if len(transformer_keys) == 0:
            print("❌ 錯誤：Checkpoint 中沒有 Transformer 參數！")
            print("   可能原因：")
            print("   1. 儲存時出錯")
            print("   2. 模型結構不匹配")
        else:
            print(f"✅ Checkpoint 看起來正常")
            print(f"   - 包含 {len(state_dict)} 個參數")
            print(f"   - Transformer: {len(transformer_keys)} 個參數")
            print(f"   - Encoder: {len(encoder_keys)} 個參數")
            print(f"   - Decoder: {len(decoder_keys)} 個參數")
        
    else:
        print("❌ 錯誤：Checkpoint 中沒有 'model_state_dict' 鍵")
        
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
