#!/usr/bin/env python3
"""
評估 Frozen Codebook 模型並保存音頻樣本

使用方式:
    python evaluate_frozen_codebook.py --checkpoint <path> --num_samples 10
"""

import os
import sys
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display
import numpy as np
from pathlib import Path

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset
from token_denoising_transformer import TokenDenoisingTransformer


def plot_spectrogram(audio, save_path, sr=24000, title=None):
    """繪製頻譜圖"""
    try:
        audio_numpy = audio.cpu().numpy().squeeze()
        D = librosa.stft(audio_numpy, n_fft=2048, hop_length=512, win_length=2048)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            D_db,
            y_axis='hz',
            x_axis='time',
            sr=sr,
            hop_length=512
        )
        plt.colorbar(format='%+2.0f dB')
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting spectrogram: {e}")


def evaluate_model(args):
    """評估模型並保存音頻樣本"""
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入 WavTokenizer
    print("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config, args.model_path)
    wavtokenizer.eval()
    wavtokenizer = wavtokenizer.to(device)
    
    # 獲取 codebook
    codebook_weights = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    print(f"Codebook 形狀: {codebook_weights.shape}")
    
    # 載入模型
    print(f"載入 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = TokenDenoisingTransformer(
        codebook=codebook_weights,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已載入 (Epoch {checkpoint['epoch']})")
    print(f"訓練準確率: {checkpoint.get('train_acc', 'N/A')}")
    print(f"驗證準確率: {checkpoint.get('val_acc', 'N/A')}")
    
    # 準備數據
    print("準備數據...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dirs = [os.path.join(project_root, 'data/raw/box')]
    target_dir = os.path.join(project_root, 'data/raw/clean')
    
    audio_dataset = AudioDataset(input_dirs, target_dir, 
                                max_sentences_per_speaker=None,
                                allowed_speakers=args.val_speakers if args.val_speakers else None)
    
    print(f"數據集大小: {len(audio_dataset)}")
    
    # 創建輸出目錄
    output_dir = os.path.join(args.output_dir, 'evaluation_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # 評估樣本
    print(f"\n開始評估 {args.num_samples} 個樣本...")
    
    with torch.no_grad():
        for idx in range(min(args.num_samples, len(audio_dataset))):
            print(f"\n處理樣本 {idx+1}/{args.num_samples}...")
            
            # 獲取音頻
            noisy_audio, clean_audio, content_id = audio_dataset[idx]
            noisy_audio = noisy_audio.to(device).unsqueeze(0)
            clean_audio = clean_audio.to(device).unsqueeze(0)
            
            # 編碼為 tokens
            _, noisy_tokens = wavtokenizer.encode_infer(
                noisy_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )
            _, clean_tokens = wavtokenizer.encode_infer(
                clean_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )
            
            # 只使用第一層量化器
            noisy_tokens = noisy_tokens[0]  # (1, seq_len)
            clean_tokens = clean_tokens[0]  # (1, seq_len)
            
            print(f"  Noisy tokens shape: {noisy_tokens.shape}")
            print(f"  Clean tokens shape: {clean_tokens.shape}")
            
            # 預測 clean tokens
            pred_tokens_logits = model(noisy_tokens, return_logits=True)  # (1, seq_len, 4096)
            pred_tokens = pred_tokens_logits.argmax(dim=-1)  # (1, seq_len)
            
            # 計算 token 準確率
            token_acc = (pred_tokens == clean_tokens).float().mean().item() * 100
            print(f"  Token Accuracy: {token_acc:.2f}%")
            
            # 解碼為音頻
            # 需要將 tokens 轉回 (num_quantizers, batch, seq_len) 格式
            pred_tokens_expanded = pred_tokens.unsqueeze(0)  # (1, 1, seq_len)
            
            # 使用 WavTokenizer 的 codes_to_features 和 decode
            pred_features = wavtokenizer.codes_to_features(pred_tokens_expanded)  # (1, C, T)
            pred_audio = wavtokenizer.decode(pred_features)  # (1, 1, audio_len)
            pred_audio = pred_audio.squeeze(1)  # (1, audio_len)
            
            # 保存音頻
            prefix = f"sample_{idx+1}_id{content_id}"
            
            # Noisy
            noisy_path = os.path.join(output_dir, f"{prefix}_noisy.wav")
            torchaudio.save(noisy_path, noisy_audio.cpu(), 24000)
            noisy_spec_path = os.path.join(output_dir, f"{prefix}_noisy_spec.png")
            plot_spectrogram(noisy_audio, noisy_spec_path, title="Noisy Audio")
            
            # Clean (Ground Truth)
            clean_path = os.path.join(output_dir, f"{prefix}_clean.wav")
            torchaudio.save(clean_path, clean_audio.cpu(), 24000)
            clean_spec_path = os.path.join(output_dir, f"{prefix}_clean_spec.png")
            plot_spectrogram(clean_audio, clean_spec_path, title="Clean Audio (GT)")
            
            # Predicted
            pred_path = os.path.join(output_dir, f"{prefix}_pred.wav")
            torchaudio.save(pred_path, pred_audio.cpu(), 24000)
            pred_spec_path = os.path.join(output_dir, f"{prefix}_pred_spec.png")
            plot_spectrogram(pred_audio, pred_spec_path, title=f"Predicted Audio (Acc: {token_acc:.1f}%)")
            
            print(f"  ✓ 已保存音頻和頻譜圖")
    
    print(f"\n✅ 評估完成！樣本已保存至: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='評估 Frozen Codebook 模型')
    
    # 模型路徑
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型 checkpoint 路徑')
    parser.add_argument('--config', type=str,
                      default='../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                      help='WavTokenizer 配置文件')
    parser.add_argument('--model_path', type=str,
                      default='../models/wavtokenizer_large_speech_320_24k.ckpt',
                      help='WavTokenizer 模型路徑')
    
    # 模型配置
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 評估設定
    parser.add_argument('--num_samples', type=int, default=10,
                      help='評估樣本數量')
    parser.add_argument('--val_speakers', nargs='+', 
                      default=['girl9', 'girl10', 'boy7', 'boy8'],
                      help='驗證語者')
    parser.add_argument('--output_dir', type=str, 
                      default='../results/frozen_codebook_evaluation',
                      help='輸出目錄')
    
    args = parser.parse_args()
    evaluate_model(args)
