#!/usr/bin/env python3
"""
直接分析 Epoch 400 生成的音頻檔案
不需要重新推理，直接檢查已經生成的音頻
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

print("="*70)
print("Epoch 400 音頻品質分析")
print("="*70)

# 檢查音頻目錄
audio_dir = Path("results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_400")

if not audio_dir.exists():
    print(f"❌ 找不到音頻目錄: {audio_dir}")
    exit(1)

print(f"\n✅ 找到音頻目錄: {audio_dir}")

# 分析第一個樣本
input_file = audio_dir / "batch_0_sample_1_input.wav"
enhanced_file = audio_dir / "batch_0_sample_1_enhanced.wav"
target_file = audio_dir / "batch_0_sample_1_target.wav"

print(f"\n[1/3] 分析音頻檔案...")

for label, filepath in [("Input (Noisy)", input_file), 
                         ("Enhanced (Model)", enhanced_file),
                         ("Target (Clean)", target_file)]:
    if filepath.exists():
        audio, sr = torchaudio.load(filepath)
        duration = audio.shape[1] / sr
        rms = torch.sqrt(torch.mean(audio**2)).item()
        max_amp = torch.max(torch.abs(audio)).item()
        
        print(f"\n   {label}:")
        print(f"      檔案: {filepath.name}")
        print(f"      長度: {duration:.2f} 秒")
        print(f"      採樣率: {sr} Hz")
        print(f"      Shape: {audio.shape}")
        print(f"      RMS: {rms:.4f}")
        print(f"      Max 振幅: {max_amp:.4f}")
        
        # 檢查是否有聲音
        if rms < 0.001:
            print(f"      ⚠️  警告：音頻振幅過低，可能無聲音")
        elif rms < 0.01:
            print(f"      ⚠️  警告：音頻振幅較低")
        else:
            print(f"      ✅ 音頻振幅正常")
    else:
        print(f"\n   {label}: ❌ 檔案不存在")

# 比較多個 samples
print(f"\n[2/3] 檢查多個樣本...")
enhanced_files = list(audio_dir.glob("batch_*_enhanced.wav"))
print(f"   找到 {len(enhanced_files)} 個 enhanced 音頻檔案")

if enhanced_files:
    rms_values = []
    for f in enhanced_files[:5]:  # 檢查前 5 個
        audio, _ = torchaudio.load(f)
        rms = torch.sqrt(torch.mean(audio**2)).item()
        rms_values.append(rms)
        print(f"      {f.name}: RMS={rms:.4f}")
    
    avg_rms = np.mean(rms_values)
    print(f"\n   平均 RMS: {avg_rms:.4f}")
    
    if avg_rms < 0.001:
        result = "❌ 失敗 - 所有音頻都幾乎無聲"
    elif avg_rms < 0.01:
        result = "⚠️  可能有問題 - 音頻振幅過低"
    else:
        result = "✅ 正常 - 音頻有明顯振幅"
    
    print(f"   {result}")

# 檢查訓練記錄
print(f"\n[3/3] 檢查訓練損失趨勢...")
log_file = Path("logs/large_tokenloss_fixed_ce10_20251020_035953.log")
if log_file.exists():
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 找出包含 Total Loss 的行
    loss_lines = [l for l in lines if 'Total=' in l and 'Epoch' in l]
    
    if len(loss_lines) > 10:
        # 提取最近 10 個 epoch 的損失
        recent_losses = []
        for line in loss_lines[-100:]:  # 檢查最後 100 行
            if 'Total=' in line:
                try:
                    loss_str = line.split('Total=')[1].split(']')[0]
                    loss = float(loss_str)
                    recent_losses.append(loss)
                except:
                    pass
        
        if recent_losses:
            print(f"   最近訓練損失統計 (最後 {len(recent_losses)} 個 batch):")
            print(f"      平均: {np.mean(recent_losses):.4f}")
            print(f"      最小: {np.min(recent_losses):.4f}")
            print(f"      最大: {np.max(recent_losses):.4f}")
            print(f"      標準差: {np.std(recent_losses):.4f}")
            
            # 檢查損失是否下降
            if len(recent_losses) >= 20:
                first_half = np.mean(recent_losses[:len(recent_losses)//2])
                second_half = np.mean(recent_losses[len(recent_losses)//2:])
                if second_half < first_half:
                    print(f"      ✅ 損失持續下降 ({first_half:.4f} → {second_half:.4f})")
                else:
                    print(f"      ⚠️  損失未明顯下降 ({first_half:.4f} → {second_half:.4f})")

print("\n" + "="*70)
print("📊 總結")
print("="*70)

print("\n下一步建議:")
print("  1. 聽聽這些音頻檔案:")
print(f"     - Input:    {input_file}")
print(f"     - Enhanced: {enhanced_file}")
print(f"     - Target:   {target_file}")
print("\n  2. 比較 Epoch 100, 200, 300, 400 的音頻品質")
print("\n  3. 如果 Epoch 400 音頻品質良好，可以考慮提前停止訓練")

print("\n" + "="*70)
