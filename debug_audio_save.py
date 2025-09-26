#!/usr/bin/env python3
"""
WavTokenizer 音頻保存調試腳本
用於診斷音頻樣本保存過程中的具體錯誤
"""

import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加必要的路徑
sys.path.append('/home/sbplab/ruizi/WavTokenizer')
sys.path.append('/home/sbplab/ruizi/c_code')

from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser

def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug_audio_save.log')
        ]
    )

def test_audio_dimensions():
    """測試音頻維度處理"""
    logging.info("🔧 開始音頻維度測試...")
    
    try:
        # 初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用設備: {device}")
        
        # 創建模型實例
        model = WavTokenizerTransformerDenoiser(
            config_path='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            model_path='models/wavtokenizer_large_speech_320_24k.ckpt',
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256
        ).to(device)
        
        model.eval()
        
        # 創建測試音頻數據
        batch_size = 2
        audio_length = 24000  # 1 second at 24kHz
        test_audio = torch.randn(batch_size, 1, audio_length).to(device)
        
        logging.info(f"輸入音頻維度: {test_audio.shape}")
        
        with torch.no_grad():
            # 編碼到 tokens
            logging.info("🔄 編碼音頻到 tokens...")
            tokens = model.encode_audio_to_tokens(test_audio)
            logging.info(f"Tokens 維度: {tokens.shape}")
            
            # 解碼回音頻
            logging.info("🔄 解碼 tokens 到音頻...")
            reconstructed_audio = model.decode_tokens_to_audio(tokens)
            logging.info(f"重建音頻維度: {reconstructed_audio.shape}")
            
            # 測試模型前向傳播
            logging.info("🔄 測試模型推理...")
            model_output = model(test_audio)
            
            if isinstance(model_output, dict):
                if 'denoised_audio' in model_output:
                    denoised_audio = model_output['denoised_audio']
                    logging.info(f"降噪音頻維度: {denoised_audio.shape}")
                else:
                    logging.error("模型輸出不包含 'denoised_audio'")
                    return False
            else:
                denoised_audio = model_output
                logging.info(f"降噪音頻維度: {denoised_audio.shape}")
            
            # 測試音頻保存功能
            logging.info("🔄 測試音頻保存...")
            test_save_audio(test_audio, denoised_audio, test_audio)  # input, output, target
            
        return True
        
    except Exception as e:
        logging.error(f"❌ 維度測試失敗: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def test_save_audio(input_audio, output_audio, target_audio):
    """測試音頻保存功能"""
    logging.info("💾 開始音頻保存測試...")
    
    try:
        # 創建測試目錄
        save_dir = Path("debug_audio_output")
        save_dir.mkdir(exist_ok=True)
        audio_dir = save_dir / "audio_samples"
        audio_dir.mkdir(exist_ok=True)
        
        logging.info(f"保存目錄: {audio_dir}")
        
        # 處理每個樣本
        for j in range(min(2, input_audio.size(0))):  # 只處理前2個樣本
            base_name = f"debug_sample_{j+1}"
            logging.info(f"處理樣本: {base_name}")
            
            # 提取單個樣本
            input_sample = input_audio[j:j+1].cpu()
            target_sample = target_audio[j:j+1].cpu()  
            output_sample = output_audio[j:j+1].cpu()
            
            logging.info(f"樣本維度 - input: {input_sample.shape}, output: {output_sample.shape}, target: {target_sample.shape}")
            
            # 正規化
            input_sample = input_sample / (torch.max(torch.abs(input_sample)) + 1e-8)
            target_sample = target_sample / (torch.max(torch.abs(target_sample)) + 1e-8)
            output_sample = output_sample / (torch.max(torch.abs(output_sample)) + 1e-8)
            
            # 確保音頻是正確的2D形狀 [1, time] 以便保存
            if input_sample.dim() == 3:  # [1, 1, T] -> [1, T]
                input_sample = input_sample.squeeze(1)
            if target_sample.dim() == 3:  # [1, 1, T] -> [1, T]  
                target_sample = target_sample.squeeze(1)
            if output_sample.dim() == 3:  # [1, 1, T] -> [1, T]
                output_sample = output_sample.squeeze(1)
            
            logging.info(f"調整後樣本維度 - input: {input_sample.shape}, output: {output_sample.shape}, target: {target_sample.shape}")
            
            # 保存每個音頻樣本
            for audio, prefix in [
                (output_sample, "denoised"),
                (input_sample, "input"),
                (target_sample, "target")
            ]:
                # 保存音頻文件
                audio_path = audio_dir / f"{base_name}_{prefix}.wav"
                try:
                    import torchaudio
                    torchaudio.save(str(audio_path), audio, 24000)
                    logging.info(f"✅ 成功保存{prefix}音頻到: {audio_path}")
                    
                    # 驗證保存的文件
                    if audio_path.exists() and audio_path.stat().st_size > 0:
                        logging.info(f"✅ 文件驗證通過: {audio_path} ({audio_path.stat().st_size} bytes)")
                    else:
                        logging.error(f"❌ 文件保存失敗或為空: {audio_path}")
                        
                except Exception as save_err:
                    logging.error(f"❌ 保存音頻失敗: {str(save_err)}")
                    import traceback
                    logging.error(traceback.format_exc())
                
        return True
                        
    except Exception as e:
        logging.error(f"❌ 音頻保存測試失敗: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """主函數"""
    print("🔍 WavTokenizer 音頻保存調試開始...")
    setup_logging()
    
    success = test_audio_dimensions()
    
    if success:
        print("✅ 音頻保存調試完成，請查看日誌文件 debug_audio_save.log")
    else:
        print("❌ 音頻保存調試失敗，請查看日誌文件 debug_audio_save.log")
    
    return success

if __name__ == "__main__":
    main()