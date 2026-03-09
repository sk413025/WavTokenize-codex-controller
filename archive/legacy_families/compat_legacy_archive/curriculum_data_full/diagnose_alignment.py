"""
診斷 Noisy-Clean 時間對齊問題

檢查項目：
1. 長度差異統計
2. Cross-correlation 找最佳偏移
3. 偏移對 token accuracy 的影響
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
from families.deps.wavtokenizer_core.config import VAL_CACHE, WAVTOK_CONFIG, WAVTOK_CKPT
from torch.utils.data import DataLoader


def compute_cross_correlation(noisy: torch.Tensor, clean: torch.Tensor,
                               max_lag: int = 2400) -> tuple:
    """
    計算 cross-correlation 找最佳偏移

    Args:
        noisy: (T,) noisy audio
        clean: (T,) clean audio
        max_lag: 最大搜索偏移 (samples)，2400 = 0.1秒 @ 24kHz

    Returns:
        best_lag: 最佳偏移 (正 = noisy 領先)
        max_corr: 最大相關係數
    """
    # Normalize
    noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)
    clean = (clean - clean.mean()) / (clean.std() + 1e-8)

    # 計算 cross-correlation
    # 使用 conv1d 來加速
    noisy_padded = F.pad(noisy.unsqueeze(0).unsqueeze(0), (max_lag, max_lag))
    clean_kernel = clean.flip(0).unsqueeze(0).unsqueeze(0)

    # 這會很慢，改用 numpy
    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()

    correlations = []
    for lag in range(-max_lag, max_lag + 1, 80):  # 每 80 samples (~3.3ms) 採樣
        if lag < 0:
            # clean 領先
            n_aligned = noisy_np[-lag:]
            c_aligned = clean_np[:len(n_aligned)]
        elif lag > 0:
            # noisy 領先
            c_aligned = clean_np[lag:]
            n_aligned = noisy_np[:len(c_aligned)]
        else:
            n_aligned = noisy_np
            c_aligned = clean_np[:len(n_aligned)]

        min_len = min(len(n_aligned), len(c_aligned))
        if min_len > 1000:  # 至少需要一些樣本
            corr = np.corrcoef(n_aligned[:min_len], c_aligned[:min_len])[0, 1]
            correlations.append((lag, corr if not np.isnan(corr) else 0))

    if not correlations:
        return 0, 0.0

    best_lag, max_corr = max(correlations, key=lambda x: x[1])
    return best_lag, max_corr


def main():
    print("=" * 60)
    print("Noisy-Clean Alignment Diagnosis")
    print("=" * 60)

    # 載入資料
    dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=50)

    length_diffs = []
    best_lags = []
    max_corrs = []

    print("\n分析樣本...")
    for i in range(min(30, len(dataset))):
        item = dataset[i]
        noisy = item['noisy_audio']
        clean = item['clean_audio']

        # 長度差異 (在截斷前)
        # 這裡已經截斷了，所以需要看原始數據

        # Cross-correlation
        lag, corr = compute_cross_correlation(noisy, clean, max_lag=4800)  # 0.2 秒
        best_lags.append(lag)
        max_corrs.append(corr)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} samples...")

    # 統計
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print("\n[1] Best Lag Statistics (samples @ 24kHz)")
    print(f"    Mean: {np.mean(best_lags):.1f} ({np.mean(best_lags)/24:.1f} ms)")
    print(f"    Std:  {np.std(best_lags):.1f} ({np.std(best_lags)/24:.1f} ms)")
    print(f"    Min:  {np.min(best_lags)} ({np.min(best_lags)/24:.1f} ms)")
    print(f"    Max:  {np.max(best_lags)} ({np.max(best_lags)/24:.1f} ms)")

    # 統計偏移分佈
    lag_counts = Counter([l // 240 for l in best_lags])  # 每 10ms 一個 bin
    print("\n    Lag distribution (10ms bins):")
    for lag_bin in sorted(lag_counts.keys()):
        count = lag_counts[lag_bin]
        print(f"      {lag_bin*10:+4d}ms: {'#' * count} ({count})")

    print("\n[2] Max Correlation Statistics")
    print(f"    Mean: {np.mean(max_corrs):.4f}")
    print(f"    Std:  {np.std(max_corrs):.4f}")
    print(f"    Min:  {np.min(max_corrs):.4f}")
    print(f"    Max:  {np.max(max_corrs):.4f}")

    # 判斷
    mean_lag_ms = abs(np.mean(best_lags)) / 24
    encoder_stride_ms = 320 / 24  # ~13.3 ms per frame

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    print(f"\n  Encoder stride: {encoder_stride_ms:.1f} ms/frame")
    print(f"  Mean lag: {mean_lag_ms:.1f} ms")

    if mean_lag_ms > encoder_stride_ms / 2:
        print(f"\n  ⚠️  WARNING: Mean lag ({mean_lag_ms:.1f}ms) > half frame ({encoder_stride_ms/2:.1f}ms)")
        print("     → Token-level misalignment is likely!")
        print("     → Consider implementing frame-level alignment correction")
    else:
        print(f"\n  ✓ Mean lag is within acceptable range")

    # 計算有多少樣本的偏移超過半個 frame
    frames_misaligned = sum(1 for l in best_lags if abs(l) > 160)  # 160 samples = ~6.7ms
    print(f"\n  Samples with lag > half frame: {frames_misaligned}/{len(best_lags)} ({frames_misaligned/len(best_lags)*100:.1f}%)")

    # 保存結果
    results = {
        'best_lags': best_lags,
        'max_corrs': max_corrs,
        'mean_lag_ms': mean_lag_ms,
        'frames_misaligned': frames_misaligned,
    }

    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist([l/24 for l in best_lags], bins=20, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero lag')
    ax.set_xlabel('Best Lag (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Best Lags')
    ax.legend()

    ax = axes[1]
    ax.scatter(range(len(max_corrs)), max_corrs)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Max Correlation')
    ax.set_title('Max Correlation per Sample')
    ax.axhline(np.mean(max_corrs), color='red', linestyle='--', label=f'Mean: {np.mean(max_corrs):.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'alignment_diagnosis.png', dpi=150)
    print(f"\n  Saved plot to families/compat_legacy/curriculum_data/alignment_diagnosis.png")

    return results


if __name__ == '__main__':
    main()
