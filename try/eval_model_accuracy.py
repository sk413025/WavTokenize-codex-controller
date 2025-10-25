#!/usr/bin/env python3
"""
快速評估模型的 Token 準確率（不生成音頻）

用法:
    python eval_model_accuracy.py --checkpoint PATH_TO_CHECKPOINT
"""

import torch
import torchaudio
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer
from token_denoising_transformer import TokenDenoisingTransformer


def evaluate_model(checkpoint_path: str, num_samples: int = 50):
    """評估模型的 Token 準確率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}\n")
    
    # 載入 checkpoint
    print(f"載入 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化 WavTokenizer
    print("初始化 WavTokenizer...")
    config_path = "../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "../models/wavtokenizer_large_speech_320_24k.ckpt"
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
    
    # 取得 codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    
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
    
    # 載入測試數據
    print("\n載入測試數據...")
    noisy_dir = Path('../data/raw/box')
    clean_dir = Path('../data/clean/box2')
    
    noisy_files = sorted(list(noisy_dir.glob('*.wav')))[:num_samples]
    
    print(f"評估 {len(noisy_files)} 個樣本...\n")
    
    accuracies = []
    
    with torch.no_grad():
        for idx, noisy_file in enumerate(noisy_files):
            # 找到對應的 clean 檔案
            clean_file = clean_dir / noisy_file.name.replace('_box_LDV_', '_clean_')
            if not clean_file.exists():
                continue
            
            # 載入音頻
            noisy_audio, sr = torchaudio.load(noisy_file)
            clean_audio, _ = torchaudio.load(clean_file)
            
            # 重採樣至 24kHz
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                noisy_audio = resampler(noisy_audio)
                clean_audio = resampler(clean_audio)
            
            # 轉換為單聲道
            if noisy_audio.shape[0] > 1:
                noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
            if clean_audio.shape[0] > 1:
                clean_audio = clean_audio.mean(dim=0, keepdim=True)
            
            noisy_audio = noisy_audio.unsqueeze(0).to(device)  # (1, 1, T)
            clean_audio = clean_audio.unsqueeze(0).to(device)
            
            # 編碼為 tokens
            _, noisy_tokens = wavtokenizer.encode_infer(
                noisy_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )  # (1, T)
            
            _, clean_tokens = wavtokenizer.encode_infer(
                clean_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )  # (1, T)
            
            # 模型預測
            pred_logits = model(noisy_tokens.squeeze(0), return_logits=True)  # (T,) -> (1, T, V)
            pred_tokens = pred_logits.argmax(dim=-1)  # (1, T, V) -> (1, T)
            
            # 調整長度以匹配
            min_len = min(clean_tokens.size(1), pred_tokens.size(1))
            clean_tok_trim = clean_tokens[0, :min_len]
            pred_tok_trim = pred_tokens[0, :min_len]
            
            # 計算準確率
            accuracy = (pred_tok_trim == clean_tok_trim).float().mean().item()
            accuracies.append(accuracy)
            
            if (idx + 1) % 10 == 0:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"已評估 {idx+1}/{len(noisy_files)} 個樣本，平均準確率: {avg_acc*100:.2f}%")
    
    # 最終統計
    print(f"\n{'='*50}")
    print(f"評估完成!")
    print(f"{'='*50}")
    print(f"樣本數: {len(accuracies)}")
    print(f"平均 Token 準確率: {sum(accuracies)/len(accuracies)*100:.2f}%")
    print(f"最高準確率: {max(accuracies)*100:.2f}%")
    print(f"最低準確率: {min(accuracies)*100:.2f}%")
    print(f"{'='*50}\n")
    
    # 建議
    avg_acc = sum(accuracies) / len(accuracies)
    if avg_acc < 0.3:
        print("⚠️  模型表現較差 (< 30%)，建議重新訓練")
    elif avg_acc < 0.6:
        print("⚠️  模型表現一般 (30-60%)，可能需要更多訓練或調整超參數")
    elif avg_acc < 0.8:
        print("✓ 模型表現良好 (60-80%)，可以繼續訓練或調整")
    else:
        print("✓✓ 模型表現優秀 (> 80%)！")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='評估 Token Denoising Transformer 模型')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint 檔案路徑'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='要評估的樣本數量 (預設: 50)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"錯誤: Checkpoint 檔案不存在: {args.checkpoint}")
        return
    
    evaluate_model(args.checkpoint, args.num_samples)


if __name__ == '__main__':
    main()
