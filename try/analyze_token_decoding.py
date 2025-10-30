#!/usr/bin/env python3
"""
深入分析 token 解碼過程，找出為什麼音質差
"""
import sys
import torch
import torchaudio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer

# 配置
WAVTOKENIZER_CONFIG = '/home/sbplab/ruizi/c_code/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
WAVTOKENIZER_CHECKPOINT = '/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# 載入 WavTokenizer
print("載入 WavTokenizer...")
wavtokenizer = WavTokenizer.from_pretrained0802(WAVTOKENIZER_CONFIG, WAVTOKENIZER_CHECKPOINT).to(device).eval()

# 測試音檔
test_audio = "/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl9_clean_001.wav"
print(f"\n測試音檔: {test_audio}")

# 方法 1: debug_single_sample.py 的方式（直接載入）
print("\n" + "=" * 80)
print("方法 1: debug_single_sample.py 的方式")
print("=" * 80)
wav1, sr1 = torchaudio.load(test_audio)
wav1 = wav1.to(device)
print(f"原始音檔形狀: {wav1.shape}, sr={sr1}")
print(f"原始音量範圍: [{wav1.min():.4f}, {wav1.max():.4f}]")

# Resample 到 24kHz
if sr1 != 24000:
    wav1 = torchaudio.functional.resample(wav1, orig_freq=sr1, new_freq=24000)
    print(f"Resample 後: {wav1.shape}")

# Encode
with torch.no_grad():
    _, tokens1 = wavtokenizer.encode_infer(wav1.unsqueeze(0), bandwidth_id=torch.tensor([0], device=device))
print(f"Tokens 形狀: {tokens1.shape}")
print(f"Tokens 範圍: [{tokens1.min()}, {tokens1.max()}]")
print(f"Tokens 前 10 個: {tokens1[0, 0, :10].cpu().tolist()}")

# Decode
tokens1_3d = tokens1  # 已經是 [1, 1, T]
features1 = wavtokenizer.codes_to_features(tokens1_3d)
if features1.dim() == 4:
    features1 = features1.squeeze(2)
audio1 = wavtokenizer.decode(features1, bandwidth_id=torch.tensor([0], device=device))
print(f"重建音檔形狀: {audio1.shape}")
print(f"重建音量範圍: [{audio1.min():.4f}, {audio1.max():.4f}]")

# 計算重建誤差
mse1 = torch.mean((audio1.squeeze() - wav1.squeeze()[:audio1.shape[-1]]) ** 2).item()
print(f"重建 MSE: {mse1:.6f}")

# 方法 2: AudioDataset 的方式（使用 process_audio - normalize=False）
print("\n" + "=" * 80)
print("方法 2: AudioDataset 方式 (normalize=False)")
print("=" * 80)
from encoder.utils import convert_audio

wav2, sr2 = torchaudio.load(test_audio)
wav2 = convert_audio(wav2, sr2, 24000, 1).to(device)  # 不正規化
print(f"process_audio 後形狀: {wav2.shape}")
print(f"音量範圍: [{wav2.min():.4f}, {wav2.max():.4f}]")

# Encode (collate_fn 的方式)
wav2_input = wav2.unsqueeze(0)  # [1, T] -> [1, 1, T]
with torch.no_grad():
    _, tokens2 = wavtokenizer.encode_infer(wav2_input, bandwidth_id=torch.tensor([0], device=device))
print(f"Tokens 形狀: {tokens2.shape}")
print(f"Tokens 範圍: [{tokens2.min()}, {tokens2.max()}]")
print(f"Tokens 前 10 個: {tokens2[0, 0, :10].cpu().tolist()}")

# 比較 tokens
tokens_diff = (tokens1[0, 0] != tokens2[0, 0]).sum().item()
print(f"\n⭐ Tokens 差異: {tokens_diff} / {tokens1.shape[-1]} ({tokens_diff/tokens1.shape[-1]*100:.2f}%)")

# Decode
tokens2_3d = tokens2
features2 = wavtokenizer.codes_to_features(tokens2_3d)
if features2.dim() == 4:
    features2 = features2.squeeze(2)
audio2 = wavtokenizer.decode(features2, bandwidth_id=torch.tensor([0], device=device))
print(f"重建音檔形狀: {audio2.shape}")
print(f"重建音量範圍: [{audio2.min():.4f}, {audio2.max():.4f}]")

mse2 = torch.mean((audio2.squeeze() - wav2.squeeze()[:audio2.shape[-1]]) ** 2).item()
print(f"重建 MSE: {mse2:.6f}")

# 方法 3: 從 DataLoader 的 padded tokens 解碼
print("\n" + "=" * 80)
print("方法 3: 使用 padded tokens (模擬 DataLoader collate)")
print("=" * 80)

# 模擬 padding
tokens3 = tokens2[0].squeeze(0)  # [1, T] -> [T]
print(f"原始 tokens: {tokens3.shape}")

# Padding to a larger size (模擬 batch collation)
max_len = tokens3.shape[0] + 100
tokens3_padded = torch.nn.functional.pad(tokens3, (0, max_len - tokens3.shape[0]), value=0)
print(f"Padded tokens: {tokens3_padded.shape}")

# Decode padded tokens
tokens3_3d = tokens3_padded.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
features3 = wavtokenizer.codes_to_features(tokens3_3d)
if features3.dim() == 4:
    features3 = features3.squeeze(2)
audio3 = wavtokenizer.decode(features3, bandwidth_id=torch.tensor([0], device=device))
print(f"重建音檔形狀: {audio3.shape}")
print(f"重建音量範圍: [{audio3.min():.4f}, {audio3.max():.4f}]")

# 只取前面非 padding 部分比較
audio3_trimmed = audio3[:, :audio1.shape[-1]]
mse3 = torch.mean((audio3_trimmed.squeeze() - wav2.squeeze()[:audio3_trimmed.shape[-1]]) ** 2).item()
print(f"重建 MSE (trimmed): {mse3:.6f}")

print("\n" + "=" * 80)
print("總結")
print("=" * 80)
print(f"方法 1 (debug) MSE:      {mse1:.6f}")
print(f"方法 2 (AudioDataset) MSE: {mse2:.6f}")
print(f"方法 3 (padded) MSE:      {mse3:.6f}")

if tokens_diff == 0:
    print("\n✅ Tokens 完全相同")
else:
    print(f"\n⚠️  Tokens 有 {tokens_diff} 個不同")

# 檢查模型輸入形狀
print("\n" + "=" * 80)
print("檢查訓練時的 token 形狀")
print("=" * 80)
print("token_collate_fn 輸出:")
print(f"  - noisy_tokens: [B, T] 例如 [8, 420]")
print(f"  - clean_tokens: [B, T] 例如 [8, 420]")
print("\nmodel(noisy_tokens) 輸入:")
print(f"  - 期望: [B, T] 2D tensor")
print(f"  - 實際: {tokens3_padded.unsqueeze(0).shape}")

print("\n保存測試時的 token 形狀:")
print(f"  - 從 DataLoader: noisy_tokens_batch[i:i+1] = {tokens3_padded.unsqueeze(0).shape}")
print(f"  - 輸入 model: [1, T] 2D tensor")
