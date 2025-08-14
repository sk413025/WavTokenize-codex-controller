#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
檢查 WavTokenizer 模型結構和輸出形狀
"""

import os
import torch
import torchaudio
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

# 配置信息
CONFIG_PATH = "/home/sbplab/ruizi/WavTokenize/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
MODEL_PATH = "/home/sbplab/ruizi/WavTokenize/wavtokenizer_large_speech_320_24k.ckpt"

def load_wavtokenizer():
    """加載 WavTokenizer 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    try:
        wavtokenizer = WavTokenizer.from_pretrained0802(CONFIG_PATH, MODEL_PATH)
    except AttributeError:
        # 嘗試其他可能的加載方法
        wavtokenizer = WavTokenizer.from_pretrained(CONFIG_PATH, MODEL_PATH)
    
    wavtokenizer = wavtokenizer.to(device)
    return wavtokenizer, device

def load_audio(wav_path, device):
    """加載和預處理音頻"""
    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, 24000, 1)
    wav = wav.to(device)
    return wav

def get_discrete_code(wav, wavtokenizer):
    """獲取音頻的離散編碼"""
    with torch.no_grad():
        bandwidth_id = torch.zeros(1, dtype=torch.long, device=wav.device)
        _, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return discrete_code

def inspect_model():
    """檢查模型輸出形狀"""
    wavtokenizer, device = load_wavtokenizer()
    
    # 定義測試音頻路徑
    test_audio1 = "/home/sbplab/ruizi/WavTokenize/1c/nor_boy1_clean_001.wav"
    test_audio2 = "/home/sbplab/ruizi/WavTokenize/1c/nor_girl1_clean_001.wav"
    
    # 加載音頻
    wav1 = load_audio(test_audio1, device)
    wav2 = load_audio(test_audio2, device)
    
    # 獲取離散編碼
    try:
        discrete1 = get_discrete_code(wav1, wavtokenizer)
        discrete2 = get_discrete_code(wav2, wavtokenizer)
        
        print("\n====== 離散編碼形狀信息 ======")
        print(f"離散編碼1總形狀: {discrete1.shape}")
        print(f"離散編碼2總形狀: {discrete2.shape}")
        
        n_layers = discrete1.shape[0]
        print(f"\n每層的形狀對比:")
        
        for i in range(n_layers):
            print(f"層 {i}: 形狀1 {discrete1[i].shape}, 形狀2 {discrete2[i].shape}")
            if discrete1[i].shape != discrete2[i].shape:
                print(f"  >>> 警告: 第 {i} 層形狀不匹配!")
                
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n====== WavTokenizer 模型結構 ======")
    # 檢查 WavTokenizer 方法
    for method_name in dir(wavtokenizer):
        if not method_name.startswith('_') and method_name not in ['__class__', '__doc__', '__module__']:
            method = getattr(wavtokenizer, method_name)
            try:
                if callable(method):
                    print(f"{method_name}: {method.__doc__}")
                else:
                    print(f"{method_name}: {type(method)}")
            except:
                print(f"{method_name}: 無法獲取描述")

if __name__ == "__main__":
    inspect_model()
