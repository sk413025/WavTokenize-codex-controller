#!/usr/bin/env python3
"""
音頻維度修復驗證腳本 - 實驗編號: EXP_VERIFY_AUDIO_FIX_20250925
目的: 驗證修復後的音頻維度處理是否正確

修復內容: 在 decode_tokens_to_audio 中正確處理 WavTokenizer 的 2D 輸出
"""

import torch
import logging
from datetime import datetime
import sys
import os

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_fixed_dimensions():
    """測試修復後的維度處理"""
    try:
        # 導入修復後的模型
        from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用設備: {device}")
        
        # 創建模型實例 (使用最小參數以節省時間)
        model = WavTokenizerTransformerDenoiser(
            config_path="config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            model_path="models/wavtokenizer_large_speech_320_24k.ckpt",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            max_length=500
        ).to(device)
        
        logging.info("✅ 模型創建成功")
        
        # 創建測試 tokens (模擬實際訓練中的情況)
        batch_size = 2
        seq_len = 250
        test_tokens = torch.randint(0, 4096, (batch_size, seq_len), device=device)
        
        logging.info(f"測試 tokens 維度: {test_tokens.shape}")
        
        # 測試 decode_tokens_to_audio 方法
        with torch.no_grad():
            decoded_audio = model.decode_tokens_to_audio(test_tokens)
            logging.info(f"解碼音頻維度: {decoded_audio.shape}")
            
            # 檢查是否是正確的 3D 形狀 [batch, channels, time]
            if decoded_audio.dim() == 3:
                batch, channels, time_steps = decoded_audio.shape
                logging.info(f"✅ 正確的 3D 維度: batch={batch}, channels={channels}, time={time_steps}")
                
                # 模擬 save_sample_ttt2_style 中的處理
                for j in range(min(2, batch_size)):
                    sample = decoded_audio[j:j+1]  # [1, channels, time]
                    logging.info(f"樣本 {j+1} 維度: {sample.shape}")
                    
                    # 正規化
                    sample = sample / (torch.max(torch.abs(sample)) + 1e-8)
                    
                    # 如果需要調整到 2D (為了保存音頻)
                    if sample.dim() == 3 and sample.size(0) == 1:
                        sample_2d = sample.squeeze(0)  # [1, channels, time] -> [channels, time]
                        logging.info(f"樣本 {j+1} 2D 維度: {sample_2d.shape}")
                    
                logging.info("✅ 所有維度處理正常")
                
            else:
                logging.error(f"❌ 錯誤的維度: {decoded_audio.shape}, 期望 3D [batch, channels, time]")
                
    except Exception as e:
        logging.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.info("開始驗證音頻維度修復...")
    test_fixed_dimensions()
    logging.info("驗證完成")