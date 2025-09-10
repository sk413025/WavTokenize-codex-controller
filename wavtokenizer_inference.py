#!/usr/bin/env python3
"""
WavTokenizer-Transformer 端到端音頻降噪推理腳本

使用訓練好的模型對音頻進行降噪：
Audio → WavTokenizer Encoder → Noisy Tokens → Transformer → Denoised Tokens → WavTokenizer Decoder → Audio
"""

import os
import sys
import torch
import torchaudio
import argparse
import logging
from pathlib import Path

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser

def load_model(checkpoint_path, device):
    """載入訓練好的模型"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # 從 config 中獲取模型參數
    model_args = {
        'config_path': config.get('config_path', 'config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'),
        'model_path': config.get('model_path', 'models/wavtokenizer_large_speech_320_24k.ckpt'),
        'd_model': config.get('d_model', 512),
        'nhead': config.get('nhead', 8),
        'num_encoder_layers': config.get('num_encoder_layers', 6),
        'num_decoder_layers': config.get('num_decoder_layers', 6),
        'dim_feedforward': config.get('dim_feedforward', 2048),
        'max_length': config.get('max_length', 512),
        'dropout': config.get('dropout', 0.1)
    }
    
    # 創建模型
    model = WavTokenizerTransformerDenoiser(**model_args)
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"模型已載入，epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    
    return model

def preprocess_audio(audio_path, target_sr=24000, max_length=None):
    """預處理音頻文件"""
    
    # 載入音頻
    audio, sr = torchaudio.load(audio_path)
    
    # 轉換採樣率
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # 轉換為單聲道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # 限制長度
    if max_length and audio.shape[1] > max_length:
        audio = audio[:, :max_length]
        logging.info(f"音頻已截斷到 {max_length} 樣本點 ({max_length/target_sr:.2f} 秒)")
    
    logging.info(f"音頻預處理完成：{audio.shape}, 採樣率: {target_sr}")
    
    return audio

def denoise_audio(model, noisy_audio, device):
    """對音頻進行降噪"""
    
    with torch.no_grad():
        noisy_audio = noisy_audio.to(device)
        
        # 添加 batch 維度
        if noisy_audio.dim() == 2:  # [channels, length]
            noisy_audio = noisy_audio.unsqueeze(0)  # [batch, channels, length]
        
        # 前向傳播進行降噪
        output = model(noisy_audio)
        
        denoised_audio = output['denoised_audio']
        
        # 移除 batch 維度
        if denoised_audio.dim() == 3:
            denoised_audio = denoised_audio.squeeze(0)
        
        return denoised_audio.cpu()

def save_audio(audio, output_path, sample_rate=24000):
    """保存音頻文件"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, audio, sample_rate, encoding='PCM_S', bits_per_sample=16)
    logging.info(f"降噪音頻已保存到: {output_path}")

def batch_denoise(model, input_dir, output_dir, device, audio_extensions=('.wav', '.flac', '.mp3')):
    """批量降噪處理"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 找到所有音頻文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.rglob(f'*{ext}'))
    
    logging.info(f"找到 {len(audio_files)} 個音頻文件")
    
    for audio_file in audio_files:
        try:
            # 預處理音頻
            audio = preprocess_audio(str(audio_file))
            
            # 降噪
            denoised_audio = denoise_audio(model, audio, device)
            
            # 構建輸出路徑（保持相對路徑結構）
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.wav')
            
            # 保存結果
            save_audio(denoised_audio, str(output_file))
            
            logging.info(f"處理完成: {audio_file.name}")
            
        except Exception as e:
            logging.error(f"處理 {audio_file} 時發生錯誤: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='WavTokenizer-Transformer 音頻降噪推理')
    
    # 模型參數
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='訓練好的模型檢查點路徑')
    
    # 輸入輸出
    parser.add_argument('--input', type=str, required=True,
                        help='輸入音頻文件或目錄')
    parser.add_argument('--output', type=str, required=True,
                        help='輸出音頻文件或目錄')
    
    # 音頻參數
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='目標採樣率')
    parser.add_argument('--max_length', type=int, default=None,
                        help='最大音頻長度（樣本點數）')
    
    # 其他參數
    parser.add_argument('--batch', action='store_true',
                        help='批量處理模式（輸入為目錄）')
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 載入模型
    logging.info("載入模型...")
    model = load_model(args.checkpoint, device)
    
    if args.batch:
        # 批量處理模式
        logging.info("開始批量降噪處理...")
        batch_denoise(model, args.input, args.output, device)
    else:
        # 單文件處理模式
        logging.info("開始單文件降噪處理...")
        
        # 預處理音頻
        audio = preprocess_audio(args.input, args.sample_rate, args.max_length)
        
        # 降噪
        denoised_audio = denoise_audio(model, audio, device)
        
        # 保存結果
        save_audio(denoised_audio, args.output, args.sample_rate)
    
    logging.info("降噪處理完成！")

if __name__ == "__main__":
    main()
