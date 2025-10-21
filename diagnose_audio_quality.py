#!/usr/bin/env python3
"""
音頻品質診斷腳本
檢查為什麼 enhanced 音頻無法重建人聲
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_audio_file(wav_path):
    """分析單個音頻檔案"""
    waveform, sr = torchaudio.load(wav_path)
    
    # 基本統計
    duration = waveform.shape[-1] / sr
    mean_amp = waveform.abs().mean().item()
    max_amp = waveform.abs().max().item()
    rms = torch.sqrt(torch.mean(waveform**2)).item()
    
    # 檢查是否為靜音
    is_silent = max_amp < 0.001
    
    # 檢查是否只有雜訊（振幅很小但不是完全靜音）
    is_noise_only = max_amp < 0.01 and not is_silent
    
    # 計算能量分佈
    energy_ratio = rms / (max_amp + 1e-8)
    
    return {
        'duration': duration,
        'mean_amp': mean_amp,
        'max_amp': max_amp,
        'rms': rms,
        'is_silent': is_silent,
        'is_noise_only': is_noise_only,
        'energy_ratio': energy_ratio,
        'shape': waveform.shape
    }

def compare_sample(epoch, batch, sample):
    """比較一組樣本（input, target, enhanced）"""
    base_path = Path(f"results/transformer_large_tokenloss_large_tokenloss_202510190523/audio_samples/epoch_{epoch}")
    
    files = {
        'input': base_path / f"batch_{batch}_sample_{sample}_input.wav",
        'target': base_path / f"batch_{batch}_sample_{sample}_target.wav",
        'enhanced': base_path / f"batch_{batch}_sample_{sample}_enhanced.wav"
    }
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch} - Batch {batch} - Sample {sample}")
    print('='*70)
    
    results = {}
    for name, path in files.items():
        if path.exists():
            results[name] = analyze_audio_file(path)
            print(f"\n{name.upper()}:")
            print(f"  Duration: {results[name]['duration']:.2f}s")
            print(f"  Shape: {results[name]['shape']}")
            print(f"  Max Amplitude: {results[name]['max_amp']:.4f}")
            print(f"  RMS: {results[name]['rms']:.4f}")
            print(f"  Mean Amplitude: {results[name]['mean_amp']:.4f}")
            print(f"  Energy Ratio: {results[name]['energy_ratio']:.4f}")
            
            if results[name]['is_silent']:
                print(f"  ⚠️  警告：音頻幾乎靜音！")
            elif results[name]['is_noise_only']:
                print(f"  ⚠️  警告：音頻振幅過小，可能只有雜訊！")
        else:
            print(f"\n{name.upper()}: 檔案不存在")
    
    # 比較分析
    if 'enhanced' in results and 'target' in results:
        print(f"\n比較分析：")
        amp_ratio = results['enhanced']['max_amp'] / (results['target']['max_amp'] + 1e-8)
        rms_ratio = results['enhanced']['rms'] / (results['target']['rms'] + 1e-8)
        
        print(f"  Enhanced/Target 振幅比: {amp_ratio:.4f}")
        print(f"  Enhanced/Target RMS比: {rms_ratio:.4f}")
        
        if amp_ratio < 0.1:
            print(f"  ❌ Enhanced 音頻振幅遠低於 Target（<10%）")
            print(f"     可能原因：模型輸出接近零，沒有真正重建音頻")
        elif amp_ratio < 0.5:
            print(f"  ⚠️  Enhanced 音頻振幅偏低（<50%）")
        else:
            print(f"  ✅ Enhanced 音頻振幅正常")
    
    return results

def visualize_waveforms(epoch, batch, sample):
    """視覺化波形對比"""
    base_path = Path(f"results/transformer_large_tokenloss_large_tokenloss_202510190523/audio_samples/epoch_{epoch}")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    for idx, name in enumerate(['input', 'target', 'enhanced']):
        path = base_path / f"batch_{batch}_sample_{sample}_{name}.wav"
        if path.exists():
            waveform, sr = torchaudio.load(path)
            waveform = waveform[0].numpy()  # 取第一個聲道
            time = np.arange(len(waveform)) / sr
            
            axes[idx].plot(time, waveform, linewidth=0.5)
            axes[idx].set_title(f"{name.upper()} - Max: {np.abs(waveform).max():.4f}, RMS: {np.sqrt(np.mean(waveform**2)):.4f}")
            axes[idx].set_xlabel("Time (s)")
            axes[idx].set_ylabel("Amplitude")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(-1, 1)
    
    plt.tight_layout()
    output_path = f"audio_diagnosis_epoch{epoch}_batch{batch}_sample{sample}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n波形對比圖已儲存: {output_path}")
    plt.close()

def main():
    print("="*70)
    print("音頻品質診斷")
    print("="*70)
    
    # 檢查多個 epoch 的樣本
    epochs = [100, 200, 300]
    
    for epoch in epochs:
        # 檢查第一個 batch 的第一個樣本
        results = compare_sample(epoch, 0, 1)
        
        # 生成波形對比圖
        visualize_waveforms(epoch, 0, 1)
    
    print("\n" + "="*70)
    print("診斷總結")
    print("="*70)
    
    print("\n如果 Enhanced 音頻振幅遠低於 Target：")
    print("  可能原因 1：模型輸出的 logits 沒有正確轉換為 tokens")
    print("  可能原因 2：Token-to-Audio 解碼過程有問題")
    print("  可能原因 3：模型沒有學到有意義的 token 預測")
    print("  可能原因 4：Token Loss 權重不平衡，導致模型傾向預測某些特定 tokens")
    
    print("\n建議檢查項目：")
    print("  1. 檢查模型輸出的 token 分佈（是否過度集中）")
    print("  2. 檢查 token-to-embedding 的轉換是否正確")
    print("  3. 檢查 WavTokenizer Decoder 是否正常工作")
    print("  4. 比較模型預測的 tokens 與 target tokens 的差異")

if __name__ == "__main__":
    main()
