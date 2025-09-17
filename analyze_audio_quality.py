#!/usr/bin/env python3
"""
測試保存的音檔是否包含有效的人聲還原
對比 ttt2.py 和 wavtokenizer_transformer_denoising.py 的音檔品質
"""

import torch
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_audio_file(audio_path):
    """分析音檔的詳細資訊"""
    print(f"\n分析音檔: {audio_path}")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print("❌ 檔案不存在")
        return None
    
    # 載入音檔
    try:
        audio, sr = torchaudio.load(audio_path)
        print(f"✅ 載入成功 - 形狀: {audio.shape}, 採樣率: {sr}")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return None
    
    # 基本統計
    print(f"音頻統計:")
    print(f"  長度: {audio.shape[-1] / sr:.2f} 秒")
    print(f"  最大值: {audio.max().item():.6f}")
    print(f"  最小值: {audio.min().item():.6f}")
    print(f"  均值: {audio.mean().item():.6f}")
    print(f"  標準差: {audio.std().item():.6f}")
    print(f"  RMS能量: {torch.sqrt(torch.mean(audio**2)).item():.6f}")
    
    # 檢查靜音
    silence_threshold = 0.001
    non_silent_samples = torch.sum(torch.abs(audio) > silence_threshold).item()
    silence_ratio = 1 - (non_silent_samples / audio.numel())
    print(f"  靜音比例: {silence_ratio:.2%}")
    
    # 頻域分析
    audio_np = audio.cpu().numpy().flatten()
    
    # 計算頻譜
    fft = np.fft.fft(audio_np)
    freqs = np.fft.fftfreq(len(audio_np), 1/sr)
    magnitude = np.abs(fft)
    
    # 人聲頻段能量 (85-255 Hz 基頻 + 300-3400 Hz 主要頻段)
    voice_fundamental = np.sum(magnitude[(freqs >= 85) & (freqs <= 255)])
    voice_main = np.sum(magnitude[(freqs >= 300) & (freqs <= 3400)])
    total_energy = np.sum(magnitude[freqs >= 0])
    
    voice_ratio = (voice_fundamental + voice_main) / total_energy if total_energy > 0 else 0
    print(f"  人聲頻段能量比例: {voice_ratio:.2%}")
    
    # 零交叉率 (Zero Crossing Rate) - 語音特徵
    zcr = np.sum(np.diff(np.sign(audio_np)) != 0) / len(audio_np)
    print(f"  零交叉率: {zcr:.4f}")
    
    # 檢查是否為常數或近似常數
    if torch.allclose(audio, audio[0, 0], atol=1e-6):
        print("⚠️  音檔為常數值")
    elif silence_ratio > 0.9:
        print("⚠️  音檔過於安靜")
    elif voice_ratio < 0.1:
        print("⚠️  人聲頻段能量較低")
    else:
        print("✅ 音檔包含正常音頻信號")
    
    return {
        'path': audio_path,
        'shape': audio.shape,
        'sr': sr,
        'duration': audio.shape[-1] / sr,
        'rms_energy': torch.sqrt(torch.mean(audio**2)).item(),
        'silence_ratio': silence_ratio,
        'voice_ratio': voice_ratio,
        'zcr': zcr
    }

def compare_audio_sets():
    """比較不同來源的音檔"""
    base_dir = "/home/sbplab/ruizi/c_code"
    
    # 我們的測試音檔
    our_audio_dir = os.path.join(base_dir, "test_audio_save/audio_samples/epoch_100")
    
    print("🔍 分析我們生成的音檔...")
    our_results = []
    
    if os.path.exists(our_audio_dir):
        for filename in sorted(os.listdir(our_audio_dir)):
            if filename.endswith('.wav'):
                audio_path = os.path.join(our_audio_dir, filename)
                result = analyze_audio_file(audio_path)
                if result:
                    our_results.append(result)
    
    # 檢查是否有 ttt2.py 生成的音檔作為對比
    ttt2_dirs = [
        "/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output2/audio_samples",
        "/home/sbplab/ruizi/WavTokenize/results/temp_test_save/audio_samples"
    ]
    
    print(f"\n🔍 搜索 ttt2.py 生成的音檔進行對比...")
    ttt2_results = []
    
    for ttt2_dir in ttt2_dirs:
        if os.path.exists(ttt2_dir):
            print(f"檢查目錄: {ttt2_dir}")
            for root, dirs, files in os.walk(ttt2_dir):
                for filename in files:
                    if filename.endswith('.wav'):
                        audio_path = os.path.join(root, filename)
                        result = analyze_audio_file(audio_path)
                        if result:
                            ttt2_results.append(result)
                        break  # 只分析每個目錄的第一個音檔
                if ttt2_results:
                    break
    
    # 比較結果
    print(f"\n📊 比較結果:")
    print("=" * 60)
    
    if our_results:
        print(f"我們的音檔 ({len(our_results)} 個):")
        avg_rms = np.mean([r['rms_energy'] for r in our_results])
        avg_voice = np.mean([r['voice_ratio'] for r in our_results])
        avg_silence = np.mean([r['silence_ratio'] for r in our_results])
        print(f"  平均RMS能量: {avg_rms:.6f}")
        print(f"  平均人聲比例: {avg_voice:.2%}")
        print(f"  平均靜音比例: {avg_silence:.2%}")
    
    if ttt2_results:
        print(f"\nttt2.py 音檔 ({len(ttt2_results)} 個):")
        avg_rms = np.mean([r['rms_energy'] for r in ttt2_results])
        avg_voice = np.mean([r['voice_ratio'] for r in ttt2_results])
        avg_silence = np.mean([r['silence_ratio'] for r in ttt2_results])
        print(f"  平均RMS能量: {avg_rms:.6f}")
        print(f"  平均人聲比例: {avg_voice:.2%}")
        print(f"  平均靜音比例: {avg_silence:.2%}")
    
    if not our_results and not ttt2_results:
        print("❌ 未找到音檔進行分析")
    
    return our_results, ttt2_results

if __name__ == "__main__":
    print("🎵 音檔品質分析工具")
    print("=" * 60)
    
    # 分析和比較音檔
    our_results, ttt2_results = compare_audio_sets()
    
    print(f"\n🎯 結論:")
    print("=" * 60)
    
    if our_results:
        # 檢查我們的音檔是否正常
        problems = []
        for result in our_results:
            if result['silence_ratio'] > 0.8:
                problems.append(f"{result['path']}: 過於安靜")
            elif result['voice_ratio'] < 0.05:
                problems.append(f"{result['path']}: 缺乏人聲特徵")
            elif result['rms_energy'] < 0.001:
                problems.append(f"{result['path']}: 能量過低")
        
        if problems:
            print("⚠️  發現問題:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("✅ 我們的音檔品質正常")
    
    if our_results and ttt2_results:
        print("\n💡 建議:")
        print("  - 比較兩組音檔的頻譜圖")
        print("  - 檢查音頻預處理步驟是否一致")
        print("  - 確認WavTokenizer編碼/解碼參數")
