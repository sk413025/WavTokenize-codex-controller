#!/usr/bin/env python3
"""
音頻維度調試腳本 - 實驗編號: EXP_DEBUG_AUDIO_DIM_20250924
目的: 診斷並修復音頻樣本保存時的維度不匹配問題

錯誤: Mask size should match input size
分析: 可能是在 WavTokenizer decode 過程中產生了不正確的維度
"""

import os
import sys
import torch
import logging
from datetime import datetime
import numpy as np

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/debug_audio_dim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_wavtokenizer():
    """載入 WavTokenizer 模型並測試維度"""
    try:
        from decoder.pretrained import WavTokenizer
        
        # 載入 WavTokenizer 模型
        config_path = "config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        ckpt_path = "models/wavtokenizer_large_speech_320_24k.ckpt"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用設備: {device}")
        
        wavtokenizer = WavTokenizer.from_pretrained0802(config_path, ckpt_path).to(device)
        logging.info("WavTokenizer 載入成功")
        
        return wavtokenizer, device
        
    except Exception as e:
        logging.error(f"載入 WavTokenizer 失敗: {e}")
        return None, None

def test_wavtokenizer_dimensions():
    """測試 WavTokenizer 的編碼解碼維度"""
    wavtokenizer, device = load_wavtokenizer()
    if wavtokenizer is None:
        return
    
    try:
        # 創建測試音頻 (模擬真實情況)
        batch_size = 2
        audio_length = 80000  # 大約3.3秒 @ 24kHz
        test_audio = torch.randn(batch_size, 1, audio_length, device=device)
        
        logging.info(f"測試音頻維度: {test_audio.shape}")
        
        # 步驟1: 音頻編碼為 tokens
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=device)
            
            # 編碼
            features, discrete_code = wavtokenizer.encode_infer(test_audio, bandwidth_id=bandwidth_id)
            logging.info(f"編碼特徵維度: {features.shape}")
            logging.info(f"離散代碼維度: {discrete_code.shape}")
            
            # 步驟2: 使用 codes_to_features 
            features_from_codes = wavtokenizer.codes_to_features(discrete_code)
            logging.info(f"codes_to_features 輸出維度: {features_from_codes.shape}")
            
            # 步驟3: 解碼回音頻
            decoded_audio = wavtokenizer.decode(features_from_codes, bandwidth_id=bandwidth_id)
            logging.info(f"解碼音頻維度: {decoded_audio.shape}")
            
            # 檢查不同情況下的維度處理
            if decoded_audio.dim() == 4:
                logging.info(f"4D 張量詳細形狀: {decoded_audio.shape}")
                if decoded_audio.size(1) == 1:
                    squeezed_1 = decoded_audio.squeeze(1)
                    logging.info(f"squeeze(1) 後: {squeezed_1.shape}")
                if decoded_audio.size(2) == 1:
                    squeezed_2 = decoded_audio.squeeze(2)  
                    logging.info(f"squeeze(2) 後: {squeezed_2.shape}")
            
            # 測試現有的維度處理邏輯
            audio = decoded_audio
            if audio.dim() == 4 and audio.size(1) == 1:
                audio = audio.squeeze(1)  # [batch, 1, 1, time] -> [batch, 1, time]
                logging.info(f"邏輯1處理後: {audio.shape}")
            elif audio.dim() == 4:
                audio = audio.squeeze(2)  # [batch, channels, 1, time] -> [batch, channels, time]
                logging.info(f"邏輯2處理後: {audio.shape}")
            
            # 模擬 save_sample_ttt2_style 中的處理
            for j in range(min(2, batch_size)):
                sample = audio[j:j+1]  # 保持 [1, C, T] 形狀
                logging.info(f"樣本 {j+1} 形狀: {sample.shape}")
                
                # 正規化
                sample = sample / (torch.max(torch.abs(sample)) + 1e-8)
                logging.info(f"正規化後形狀: {sample.shape}")
                
                # 維度調整
                if sample.dim() == 3:  # [1, 1, T] -> [1, T]
                    sample = sample.squeeze(1)
                    logging.info(f"squeeze(1) 後樣本形狀: {sample.shape}")
            
            logging.info("✅ 維度測試完成")
            
    except Exception as e:
        logging.error(f"維度測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.info("開始音頻維度調試...")
    test_wavtokenizer_dimensions()
    logging.info("調試完成")