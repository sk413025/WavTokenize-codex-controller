"""
檢查 VAL 樣本的前後靜音情況
確認 clean vs noisy 的差異來源
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

# 路徑配置
VAL_CACHE = '/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt'
DATA_BASE = Path('/home/sbplab/ruizi/c_code/data')

def resolve_path(p):
    """解析音頻路徑"""
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    fn = p.name
    if "_clean_" in fn:
        return DATA_BASE / "clean/box2" / fn
    if "_box_" in fn:
        return DATA_BASE / "raw/box" / fn
    if "_papercup_" in fn:
        return DATA_BASE / "raw/papercup" / fn
    if "_plastic_" in fn:
        return DATA_BASE / "raw/plastic" / fn
    return p

def analyze_silence(waveform, sr=24000, threshold_db=-40):
    """
    分析音頻的靜音部分

    Returns:
        leading_silence_ms: 開頭靜音時長
        trailing_silence_ms: 結尾靜音時長
        total_duration_ms: 總時長
    """
    if torch.is_tensor(waveform):
        waveform = waveform.numpy()

    waveform = waveform.squeeze()

    # 轉換成 dB
    eps = 1e-10
    energy = np.abs(waveform)

    # 找到超過閾值的位置
    threshold = 10 ** (threshold_db / 20) * np.max(energy)
    above_threshold = energy > threshold

    if not np.any(above_threshold):
        return len(waveform) / sr * 1000, 0, len(waveform) / sr * 1000

    # 第一個和最後一個超過閾值的位置
    first_active = np.argmax(above_threshold)
    last_active = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])

    leading_silence_ms = first_active / sr * 1000
    trailing_silence_ms = (len(waveform) - 1 - last_active) / sr * 1000
    total_duration_ms = len(waveform) / sr * 1000

    return leading_silence_ms, trailing_silence_ms, total_duration_ms

def main():
    print("=" * 70)
    print("VAL 樣本靜音分析")
    print("=" * 70)

    # 載入 VAL 資料
    val_samples = torch.load(VAL_CACHE, weights_only=False)
    print(f"總共 {len(val_samples)} 個 VAL 樣本")

    # 找出長度差異大的樣本
    mismatched_samples = []

    for i, s in enumerate(val_samples[:100]):  # 先檢查前 100 個
        noisy_path = resolve_path(s['noisy_path'])
        clean_path = resolve_path(s['clean_path'])

        if not noisy_path.exists() or not clean_path.exists():
            continue

        noisy_info = torchaudio.info(str(noisy_path))
        clean_info = torchaudio.info(str(clean_path))

        noisy_dur = noisy_info.num_frames / noisy_info.sample_rate * 1000
        clean_dur = clean_info.num_frames / clean_info.sample_rate * 1000

        diff = clean_dur - noisy_dur

        if diff > 100:  # 差異超過 100ms
            mismatched_samples.append({
                'idx': i,
                'noisy_path': noisy_path,
                'clean_path': clean_path,
                'noisy_dur': noisy_dur,
                'clean_dur': clean_dur,
                'diff': diff
            })

    print(f"\n找到 {len(mismatched_samples)} 個差異 > 100ms 的樣本")

    # 詳細分析前 5 個
    print("\n" + "=" * 70)
    print("詳細分析 (前 5 個差異大的樣本)")
    print("=" * 70)

    for sample in mismatched_samples[:5]:
        print(f"\n--- 樣本 {sample['idx']} ---")
        print(f"Noisy: {sample['noisy_path'].name}")
        print(f"Clean: {sample['clean_path'].name}")
        print(f"時長: Noisy={sample['noisy_dur']:.0f}ms, Clean={sample['clean_dur']:.0f}ms, 差異={sample['diff']:.0f}ms")

        # 載入音頻
        noisy_wav, noisy_sr = torchaudio.load(str(sample['noisy_path']))
        clean_wav, clean_sr = torchaudio.load(str(sample['clean_path']))

        # Resample to 24kHz
        if noisy_sr != 24000:
            noisy_wav = torchaudio.transforms.Resample(noisy_sr, 24000)(noisy_wav)
        if clean_sr != 24000:
            clean_wav = torchaudio.transforms.Resample(clean_sr, 24000)(clean_wav)

        # 分析靜音
        noisy_lead, noisy_trail, noisy_total = analyze_silence(noisy_wav)
        clean_lead, clean_trail, clean_total = analyze_silence(clean_wav)

        print(f"\n  Noisy 靜音分析:")
        print(f"    開頭靜音: {noisy_lead:.0f}ms")
        print(f"    結尾靜音: {noisy_trail:.0f}ms")
        print(f"    總時長: {noisy_total:.0f}ms")

        print(f"\n  Clean 靜音分析:")
        print(f"    開頭靜音: {clean_lead:.0f}ms")
        print(f"    結尾靜音: {clean_trail:.0f}ms")
        print(f"    總時長: {clean_total:.0f}ms")

        print(f"\n  差異:")
        print(f"    開頭靜音差: {clean_lead - noisy_lead:.0f}ms (Clean - Noisy)")
        print(f"    結尾靜音差: {clean_trail - noisy_trail:.0f}ms (Clean - Noisy)")

    # 統計分析
    print("\n" + "=" * 70)
    print("統計分析 (所有差異 > 100ms 的樣本)")
    print("=" * 70)

    lead_diffs = []
    trail_diffs = []

    for sample in mismatched_samples[:50]:  # 分析 50 個
        noisy_wav, noisy_sr = torchaudio.load(str(sample['noisy_path']))
        clean_wav, clean_sr = torchaudio.load(str(sample['clean_path']))

        if noisy_sr != 24000:
            noisy_wav = torchaudio.transforms.Resample(noisy_sr, 24000)(noisy_wav)
        if clean_sr != 24000:
            clean_wav = torchaudio.transforms.Resample(clean_sr, 24000)(clean_wav)

        noisy_lead, noisy_trail, _ = analyze_silence(noisy_wav)
        clean_lead, clean_trail, _ = analyze_silence(clean_wav)

        lead_diffs.append(clean_lead - noisy_lead)
        trail_diffs.append(clean_trail - noisy_trail)

    lead_diffs = np.array(lead_diffs)
    trail_diffs = np.array(trail_diffs)

    print(f"\n開頭靜音差異 (Clean - Noisy):")
    print(f"  Mean: {np.mean(lead_diffs):.0f}ms")
    print(f"  Std: {np.std(lead_diffs):.0f}ms")
    print(f"  Min: {np.min(lead_diffs):.0f}ms")
    print(f"  Max: {np.max(lead_diffs):.0f}ms")

    print(f"\n結尾靜音差異 (Clean - Noisy):")
    print(f"  Mean: {np.mean(trail_diffs):.0f}ms")
    print(f"  Std: {np.std(trail_diffs):.0f}ms")
    print(f"  Min: {np.min(trail_diffs):.0f}ms")
    print(f"  Max: {np.max(trail_diffs):.0f}ms")

    # 判斷主要原因
    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)

    if abs(np.mean(trail_diffs)) > abs(np.mean(lead_diffs)):
        print("→ 主要原因: Clean 的【結尾靜音】比 Noisy 長")
        print(f"   平均多出 {np.mean(trail_diffs):.0f}ms 的尾部靜音")
    else:
        print("→ 主要原因: Clean 的【開頭靜音】比 Noisy 長")
        print(f"   平均多出 {np.mean(lead_diffs):.0f}ms 的開頭靜音")

if __name__ == '__main__':
    main()
