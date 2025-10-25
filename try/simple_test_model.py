#!/usr/bin/env python3
"""
簡單測試腳本：使用 best_model.pth 生成音頻樣本

用法:
    python simple_test_model.py --checkpoint PATH_TO_CHECKPOINT --num_samples 5
"""

import torch
import torchaudio
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer
from token_denoising_transformer import TokenDenoisingTransformer


def test_model(checkpoint_path: str, num_samples: int = 5):
    """
    測試模型並生成音頻樣本
    
    參數:
        checkpoint_path: checkpoint 檔案路徑
        num_samples: 要生成的樣本數量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入 checkpoint
    print(f"\n載入 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化 WavTokenizer
    print("初始化 WavTokenizer...")
    config_path = "../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "../models/wavtokenizer_large_speech_320_24k.ckpt"
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
    
    # 取得 codebook（正確路徑）
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    print(f"Codebook 形狀: {codebook.shape}")
    
    # 初始化模型
    print("初始化 Transformer 模型...")
    model = TokenDenoisingTransformer(
        codebook=codebook,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 準備輸出目錄
    output_dir = Path('../results/test_audio_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n音頻將儲存至: {output_dir}")
    
    # 載入測試數據 - 直接從原始檔案
    print("\n載入測試數據...")
    noisy_dir = Path('../data/raw/box')
    clean_dir = Path('../data/clean/box2')
    
    noisy_files = sorted(list(noisy_dir.glob('*.wav')))
    if not noisy_files:
        print(f"錯誤: 找不到音頻檔案於 {noisy_dir}")
        return
    
    print(f"找到 {len(noisy_files)} 個音頻檔案")
    print(f"\n開始生成 {num_samples} 個測試樣本...\n")
    
    # 生成樣本
    with torch.no_grad():
        for idx, noisy_file in enumerate(noisy_files[:num_samples]):
            # 找到對應的 clean 檔案
            clean_file = clean_dir / noisy_file.name.replace('_box_LDV_', '_clean_')
            if not clean_file.exists():
                print(f"警告: 找不到 clean 檔案 {clean_file}")
                continue
            
            print(f"[{idx+1}/{num_samples}] 處理 {noisy_file.name}...")
            
            # 載入音頻
            noisy_audio, sr = torchaudio.load(noisy_file)
            clean_audio, _ = torchaudio.load(clean_file)
            
            # 重採樣至 24kHz（如需要）
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                noisy_audio = resampler(noisy_audio)
                clean_audio = resampler(clean_audio)
            
            # 轉換為單聲道並調整維度
            if noisy_audio.shape[0] > 1:
                noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
            if clean_audio.shape[0] > 1:
                clean_audio = clean_audio.mean(dim=0, keepdim=True)
            
            noisy_audio = noisy_audio.unsqueeze(0).to(device)  # (1, 1, T)
            clean_audio = clean_audio.unsqueeze(0).to(device)
            
            print(f"[{idx+1}/{num_samples}] 處理樣本...")
            
            # 編碼為 tokens
            _, noisy_tokens = wavtokenizer.encode_infer(
                noisy_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )  # 輸出: (1, T)
            
            # 模型預測
            pred_logits = model(noisy_tokens.squeeze(0), return_logits=True)  # (T,) -> (1, T, V)
            pred_tokens = pred_logits.argmax(dim=-1)  # (1, T, V) -> (1, T)
            
            # 準備 tokens 供解碼 (需要格式: 1, 1, T)
            noisy_tok = noisy_tokens.unsqueeze(0)  # (1, T) -> (1, 1, T)
            pred_tok = pred_tokens.unsqueeze(0)    # (1, T) -> (1, 1, T)
            
            # 使用 codes_to_features 轉換為 features
            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            
            # 解碼為音頻
            noisy_audio_out = wavtokenizer.decode(noisy_features).squeeze(0)  # (1, T) -> (1, T)
            pred_audio_out = wavtokenizer.decode(pred_features).squeeze(0)
            clean_audio_recon = clean_audio.squeeze(0)  # (1, 1, T) -> (1, T)
            
            # 保存音頻
            sample_dir = output_dir / f'sample_{idx}'
            sample_dir.mkdir(exist_ok=True)
            
            torchaudio.save(
                str(sample_dir / 'noisy.wav'),
                noisy_audio_out.cpu(),
                24000
            )
            torchaudio.save(
                str(sample_dir / 'predicted.wav'),
                pred_audio_out.cpu(),
                24000
            )
            torchaudio.save(
                str(sample_dir / 'clean.wav'),
                clean_audio_recon.cpu(),
                24000
            )
            
            print(f"    ✓ 已保存至 {sample_dir}/")
            
            # 計算 token 準確率
            _, clean_tokens = wavtokenizer.encode_infer(
                clean_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )
            clean_tok = clean_tokens.squeeze(0)  # (1, T)
            pred_tok_flat = pred_tokens.squeeze(0)  # (1, T)
            
            # 調整長度以匹配
            min_len = min(clean_tok.size(0), pred_tok_flat.size(0))
            accuracy = (pred_tok_flat[:min_len] == clean_tok[:min_len]).float().mean().item()
            print(f"    Token 準確率: {accuracy*100:.2f}%\n")
    
    print(f"\n✓ 完成! 所有音頻已儲存至 {output_dir}")
    print(f"\n請聆聽音頻以評估模型品質")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='測試 Token Denoising Transformer 模型')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint 檔案路徑'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='要生成的樣本數量 (預設: 5)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"錯誤: Checkpoint 檔案不存在: {args.checkpoint}")
        return
    
    test_model(args.checkpoint, args.num_samples)


if __name__ == '__main__':
    main()
