#!/usr/bin/env python3
"""
比較修正後的 exp3 與 debug_single_sample.py 的音檔品質
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path

# 最新實驗的音檔
exp3_dir = Path("/home/sbplab/ruizi/c_code/results/exp3_mini_dataset_20251030_052739/audio_samples/epoch_100")

# debug_single_sample.py 的結果（需要找最近的）
import glob
debug_dirs = sorted(glob.glob("/home/sbplab/ruizi/c_code/results/overfit_test_*/epoch_100"), reverse=True)
if debug_dirs:
    debug_dir = Path(debug_dirs[0])
    print(f"使用 debug 目錄: {debug_dir}")
else:
    print("找不到 debug 結果，使用預設路徑")
    debug_dir = Path("/home/sbplab/ruizi/c_code/results/overfit_test_20241029_071131/epoch_100")

def load_audio(path):
    wav, sr = torchaudio.load(str(path))
    return wav, sr

def calculate_mse(audio1, audio2):
    """計算兩個音檔的 MSE"""
    # 確保長度一致
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    audio1 = audio1[..., :min_len]
    audio2 = audio2[..., :min_len]
    return torch.mean((audio1 - audio2) ** 2).item()

print("=" * 80)
print("音檔品質比較分析")
print("=" * 80)

# 載入 exp3 音檔
exp3_noisy, _ = load_audio(exp3_dir / "sample_0_noisy.wav")
exp3_clean, _ = load_audio(exp3_dir / "sample_0_clean.wav")
exp3_pred, _ = load_audio(exp3_dir / "sample_0_predicted.wav")

print(f"\n修正後 exp3 (epoch 100):")
print(f"  noisy:     {exp3_noisy.shape[-1]:6d} samples ({exp3_noisy.shape[-1]/24000:.3f}s)")
print(f"  clean:     {exp3_clean.shape[-1]:6d} samples ({exp3_clean.shape[-1]/24000:.3f}s)")
print(f"  predicted: {exp3_pred.shape[-1]:6d} samples ({exp3_pred.shape[-1]/24000:.3f}s)")

# 計算 MSE
mse_exp3 = calculate_mse(exp3_pred, exp3_clean)
print(f"\n  MSE (pred vs clean): {mse_exp3:.6f}")

# 音量分析
print(f"\n  音量分析:")
print(f"    noisy max:     {exp3_noisy.abs().max():.4f}")
print(f"    clean max:     {exp3_clean.abs().max():.4f}")
print(f"    predicted max: {exp3_pred.abs().max():.4f}")

# 嘗試載入 debug 音檔比較
try:
    if debug_dir.exists():
        debug_clean, _ = load_audio(debug_dir / "sample_0_clean.wav")
        debug_pred, _ = load_audio(debug_dir / "sample_0_predicted.wav")

        print(f"\n對比 debug_single_sample.py:")
        print(f"  clean:     {debug_clean.shape[-1]:6d} samples ({debug_clean.shape[-1]/24000:.3f}s)")
        print(f"  predicted: {debug_pred.shape[-1]:6d} samples ({debug_pred.shape[-1]/24000:.3f}s)")

        mse_debug = calculate_mse(debug_pred, debug_clean)
        print(f"\n  MSE (pred vs clean): {mse_debug:.6f}")

        print(f"\n  音量分析:")
        print(f"    clean max:     {debug_clean.abs().max():.4f}")
        print(f"    predicted max: {debug_pred.abs().max():.4f}")

        print(f"\n比較結果:")
        print(f"  MSE 比值: {mse_exp3/mse_debug:.2f}x")
        if mse_exp3 < mse_debug * 1.5:
            print(f"  ✅ exp3 音質與 debug 接近 (差距 <1.5x)")
        else:
            print(f"  ❌ exp3 音質仍比 debug 差 {mse_exp3/mse_debug:.1f} 倍")

        # Token 比較（如果可能）
        print(f"\n音檔長度比較:")
        len_ratio = exp3_pred.shape[-1] / debug_pred.shape[-1]
        print(f"  exp3/debug 長度比: {len_ratio:.3f}")
        if abs(len_ratio - 1.0) < 0.01:
            print(f"  ✅ 長度匹配")
        else:
            print(f"  ⚠️  長度不同，可能使用不同音檔")

except Exception as e:
    print(f"\n⚠️  無法載入 debug 音檔進行比較: {e}")

# 載入舊版本 exp3 比較（如果有）
old_exp3_dirs = sorted(glob.glob("/home/sbplab/ruizi/c_code/results/exp3_mini_dataset_20251030_05*/audio_samples/epoch_100"), reverse=True)
if len(old_exp3_dirs) > 1:
    old_exp3_dir = Path(old_exp3_dirs[1])  # 第二新的
    print(f"\n與舊版 exp3 比較 ({old_exp3_dir.parent.parent.name}):")
    try:
        old_pred, _ = load_audio(old_exp3_dir / "sample_0_predicted.wav")
        old_clean, _ = load_audio(old_exp3_dir / "sample_0_clean.wav")
        mse_old = calculate_mse(old_pred, old_clean)
        print(f"  舊版 MSE: {mse_old:.6f}")
        print(f"  新版 MSE: {mse_exp3:.6f}")
        improvement = (mse_old - mse_exp3) / mse_old * 100
        if improvement > 0:
            print(f"  ✅ 改善了 {improvement:.1f}%")
        else:
            print(f"  ❌ 反而變差了 {abs(improvement):.1f}%")
    except Exception as e:
        print(f"  無法載入: {e}")

print("\n" + "=" * 80)
