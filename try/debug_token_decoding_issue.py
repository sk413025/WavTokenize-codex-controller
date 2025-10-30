#!/usr/bin/env python3
"""
核心問題：token 100% 正確，為什麼音檔品質差？
假設：clean tokens 本身解碼後就有損失
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
print(f"使用設備: {device}\n")

# 載入 WavTokenizer
print("載入 WavTokenizer...")
wavtokenizer = WavTokenizer.from_pretrained0802(WAVTOKENIZER_CONFIG, WAVTOKENIZER_CHECKPOINT).to(device).eval()

# 測試音檔 - 使用驗證集的第一個音檔
test_clean = "/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy7_clean_001.wav"
print(f"測試音檔: {test_clean}\n")

# 載入原始 clean 音檔
print("=" * 80)
print("步驟 1: 載入原始 clean 音檔")
print("=" * 80)
clean_wav, sr = torchaudio.load(test_clean)
clean_wav = clean_wav.to(device)
print(f"原始形狀: {clean_wav.shape}, sr={sr}")
print(f"原始音量: [{clean_wav.min():.4f}, {clean_wav.max():.4f}]")
print(f"原始長度: {clean_wav.shape[-1]/sr:.3f}秒\n")

# Resample 到 24kHz (不正規化)
if sr != 24000:
    clean_wav = torchaudio.functional.resample(clean_wav, orig_freq=sr, new_freq=24000)
print(f"Resample 後: {clean_wav.shape}")
print(f"音量: [{clean_wav.min():.4f}, {clean_wav.max():.4f}]\n")

# Encode 成 tokens
print("=" * 80)
print("步驟 2: Encode 成 tokens")
print("=" * 80)
with torch.no_grad():
    _, clean_tokens = wavtokenizer.encode_infer(
        clean_wav.unsqueeze(0),  # [1, T] -> [1, 1, T]
        bandwidth_id=torch.tensor([0], device=device)
    )
print(f"Tokens 形狀: {clean_tokens.shape}")
print(f"Tokens 範圍: [{clean_tokens.min()}, {clean_tokens.max()}]")
print(f"Tokens 前 20 個: {clean_tokens[0, 0, :20].cpu().tolist()}\n")

# 測試不同的解碼方式
print("=" * 80)
print("步驟 3: 測試不同的解碼方式")
print("=" * 80)

# 方式 1: 直接解碼 (debug方式 - [1, T])
print("\n方式 1: debug方式 - [1, T]")
tokens_1 = clean_tokens[0, 0].unsqueeze(0)  # [1, 1, T] -> [T] -> [1, T]
print(f"  Input tokens shape: {tokens_1.shape}")
features_1 = wavtokenizer.codes_to_features(tokens_1)
if features_1.dim() == 4:
    features_1 = features_1.squeeze(2)
audio_1 = wavtokenizer.decode(features_1, bandwidth_id=torch.tensor([0], device=device)).cpu()
print(f"  Output audio shape: {audio_1.shape}")
print(f"  音量: [{audio_1.min():.4f}, {audio_1.max():.4f}]")
mse_1 = torch.mean((audio_1.squeeze()[:clean_wav.shape[-1]] - clean_wav.cpu().squeeze()) ** 2).item()
print(f"  MSE vs 原始: {mse_1:.6f}")

# 方式 2: 保持 3D (train舊方式 - [1, 1, T])
print("\n方式 2: train舊方式 - [1, 1, T]")
tokens_2 = clean_tokens  # [1, 1, T]
print(f"  Input tokens shape: {tokens_2.shape}")
features_2 = wavtokenizer.codes_to_features(tokens_2)
if features_2.dim() == 4:
    features_2 = features_2.squeeze(2)
audio_2 = wavtokenizer.decode(features_2, bandwidth_id=torch.tensor([0], device=device)).cpu()
print(f"  Output audio shape: {audio_2.shape}")
print(f"  音量: [{audio_2.min():.4f}, {audio_2.max():.4f}]")
mse_2 = torch.mean((audio_2.squeeze()[:clean_wav.shape[-1]] - clean_wav.cpu().squeeze()) ** 2).item()
print(f"  MSE vs 原始: {mse_2:.6f}")

# 方式 3: 新修正方式 ([1, T] -> [T] -> [1, T])
print("\n方式 3: 新修正方式 - squeeze+unsqueeze")
tokens_3_input = clean_tokens[0]  # [1, 1, T] -> [1, T]
tokens_3 = tokens_3_input.squeeze(0).unsqueeze(0)  # [1, T] -> [T] -> [1, T]
print(f"  Input tokens shape: {tokens_3.shape}")
features_3 = wavtokenizer.codes_to_features(tokens_3)
if features_3.dim() == 4:
    features_3 = features_3.squeeze(2)
audio_3 = wavtokenizer.decode(features_3, bandwidth_id=torch.tensor([0], device=device)).cpu()
print(f"  Output audio shape: {audio_3.shape}")
print(f"  音量: [{audio_3.min():.4f}, {audio_3.max():.4f}]")
mse_3 = torch.mean((audio_3.squeeze()[:clean_wav.shape[-1]] - clean_wav.cpu().squeeze()) ** 2).item()
print(f"  MSE vs 原始: {mse_3:.6f}")

# 比較結果
print("\n" + "=" * 80)
print("總結對比")
print("=" * 80)
print(f"方式 1 (debug [1,T]):           MSE = {mse_1:.6f}")
print(f"方式 2 (train舊 [1,1,T]):      MSE = {mse_2:.6f}")
print(f"方式 3 (train新 squeeze+unsq): MSE = {mse_3:.6f}")

if abs(mse_1 - mse_3) < 1e-6:
    print("\n✅ 方式 1 和方式 3 完全相同！維度修正有效")
else:
    print(f"\n⚠️  方式 1 和方式 3 不同，差異: {abs(mse_1 - mse_3):.6f}")

if mse_2 > mse_1 * 1.5:
    print(f"❌ 方式 2 (舊方式) 比方式 1 差 {mse_2/mse_1:.1f} 倍 - 這就是問題所在！")

# 測試 WavTokenizer 本身的重建誤差
print("\n" + "=" * 80)
print("WavTokenizer 自身的重建誤差")
print("=" * 80)
print(f"這是 WavTokenizer 的固有損失，無法避免")
print(f"即使 token 100% 正確，解碼後也會有這個誤差")
print(f"MSE = {mse_1:.6f}")

# 保存音檔供人工比較
output_dir = Path("debug_outputs")
output_dir.mkdir(exist_ok=True)
torchaudio.save(str(output_dir / "original_clean.wav"), clean_wav.cpu(), 24000)
torchaudio.save(str(output_dir / "decoded_method1.wav"), audio_1, 24000)
torchaudio.save(str(output_dir / "decoded_method2.wav"), audio_2, 24000)
torchaudio.save(str(output_dir / "decoded_method3.wav"), audio_3, 24000)
print(f"\n✅ 音檔已保存至 {output_dir}/")
