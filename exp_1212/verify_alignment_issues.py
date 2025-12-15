"""
exp_1212: 驗證資料對齊問題

驗證兩個潛在問題源:
1. 來源1: per-pair mismatch (同一對 noisy/clean 長度不同)
2. 來源2: cross-sample mismatch (batch 內不同樣本長度差異大)

基於 exp_1210/DATASET_ALIGNMENT_REPORT.md 的分析
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# 路徑配置
TRAIN_CACHE = '/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt'
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

    # 嘗試其他路徑
    for subdir in ["clean/box2", "raw/box", "raw/papercup", "raw/plastic"]:
        candidate = DATA_BASE / subdir / fn
        if candidate.exists():
            return candidate

    return p

def get_wav_length(path, target_sr=24000):
    """獲取 wav 檔案長度 (samples at target_sr)"""
    try:
        info = torchaudio.info(str(path))
        samples = info.num_frames
        sr = info.sample_rate
        # 換算到目標採樣率
        return int(samples * target_sr / sr)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def analyze_per_pair_mismatch(samples, split_name, max_samples=None):
    """
    來源1: 分析 per-pair mismatch
    檢查每一對 noisy/clean 的長度差異
    """
    print(f"\n{'='*60}")
    print(f"來源1: Per-Pair Mismatch 分析 ({split_name})")
    print(f"{'='*60}")

    if max_samples:
        samples = samples[:max_samples]

    diffs = []
    noisy_longer = 0
    clean_longer = 0
    mismatch_details = []

    for i, s in enumerate(samples):
        noisy_path = resolve_path(s['noisy_path'])
        clean_path = resolve_path(s['clean_path'])

        noisy_len = get_wav_length(noisy_path)
        clean_len = get_wav_length(clean_path)

        if noisy_len is None or clean_len is None:
            continue

        diff = noisy_len - clean_len  # positive = noisy longer
        diffs.append(diff)

        if diff > 0:
            noisy_longer += 1
        elif diff < 0:
            clean_longer += 1

        if abs(diff) > 240:  # > 10ms at 24kHz
            mismatch_details.append({
                'idx': i,
                'noisy_len': noisy_len,
                'clean_len': clean_len,
                'diff_samples': diff,
                'diff_sec': diff / 24000
            })

    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)

    total = len(diffs)
    mismatch_count = np.sum(abs_diffs > 0)
    significant_mismatch = np.sum(abs_diffs > 240)  # > 10ms
    severe_mismatch = np.sum(abs_diffs > 2400)  # > 100ms

    result = {
        'split': split_name,
        'total_samples': total,
        'mismatch_count': int(mismatch_count),
        'mismatch_pct': float(mismatch_count / total * 100),
        'significant_mismatch_count': int(significant_mismatch),
        'significant_mismatch_pct': float(significant_mismatch / total * 100),
        'severe_mismatch_count': int(severe_mismatch),
        'severe_mismatch_pct': float(severe_mismatch / total * 100),
        'noisy_longer_count': noisy_longer,
        'clean_longer_count': clean_longer,
        'diff_stats': {
            'mean_samples': float(np.mean(abs_diffs)),
            'mean_sec': float(np.mean(abs_diffs) / 24000),
            'median_samples': float(np.median(abs_diffs)),
            'std_samples': float(np.std(abs_diffs)),
            'p90_samples': float(np.percentile(abs_diffs, 90)),
            'p95_samples': float(np.percentile(abs_diffs, 95)),
            'p99_samples': float(np.percentile(abs_diffs, 99)),
            'max_samples': float(np.max(abs_diffs)),
            'max_sec': float(np.max(abs_diffs) / 24000),
        }
    }

    # 打印結果
    print(f"\n總樣本數: {total}")
    print(f"\n--- 長度不一致統計 ---")
    print(f"有不一致的 pair: {mismatch_count} ({mismatch_count/total*100:.2f}%)")
    print(f"  - 顯著不一致 (>10ms): {significant_mismatch} ({significant_mismatch/total*100:.2f}%)")
    print(f"  - 嚴重不一致 (>100ms): {severe_mismatch} ({severe_mismatch/total*100:.2f}%)")
    print(f"\n方向性:")
    print(f"  - Noisy 更長: {noisy_longer} ({noisy_longer/total*100:.2f}%)")
    print(f"  - Clean 更長: {clean_longer} ({clean_longer/total*100:.2f}%)")
    print(f"\n--- 長度差統計 (絕對值) ---")
    print(f"  Mean: {np.mean(abs_diffs):.1f} samples ({np.mean(abs_diffs)/24000*1000:.1f} ms)")
    print(f"  Median: {np.median(abs_diffs):.1f} samples")
    print(f"  P90: {np.percentile(abs_diffs, 90):.1f} samples ({np.percentile(abs_diffs, 90)/24000*1000:.1f} ms)")
    print(f"  P95: {np.percentile(abs_diffs, 95):.1f} samples ({np.percentile(abs_diffs, 95)/24000*1000:.1f} ms)")
    print(f"  Max: {np.max(abs_diffs):.1f} samples ({np.max(abs_diffs)/24000*1000:.1f} ms)")

    return result, diffs

def analyze_cross_sample_mismatch(samples, split_name, batch_size=16, max_samples=None):
    """
    來源2: 分析 cross-sample mismatch
    模擬 batch 內的長度差異，評估 padding 影響
    """
    print(f"\n{'='*60}")
    print(f"來源2: Cross-Sample Mismatch 分析 ({split_name})")
    print(f"{'='*60}")

    if max_samples:
        samples = samples[:max_samples]

    # 收集所有樣本長度
    lengths = []
    for s in samples:
        noisy_path = resolve_path(s['noisy_path'])
        noisy_len = get_wav_length(noisy_path)
        if noisy_len:
            lengths.append(noisy_len)

    lengths = np.array(lengths)

    # 全局統計
    print(f"\n--- 樣本長度分佈 ---")
    print(f"  總樣本: {len(lengths)}")
    print(f"  Min: {np.min(lengths)} samples ({np.min(lengths)/24000:.2f}s)")
    print(f"  Max: {np.max(lengths)} samples ({np.max(lengths)/24000:.2f}s)")
    print(f"  Mean: {np.mean(lengths):.1f} samples ({np.mean(lengths)/24000:.2f}s)")
    print(f"  Std: {np.std(lengths):.1f} samples ({np.std(lengths)/24000:.2f}s)")
    print(f"  Range: {np.max(lengths) - np.min(lengths)} samples ({(np.max(lengths) - np.min(lengths))/24000:.2f}s)")

    # 模擬 batch padding
    print(f"\n--- 模擬 Batch Padding (batch_size={batch_size}) ---")

    # 隨機模擬 100 個 batch
    np.random.seed(42)
    batch_stats = []
    total_valid_frames = 0
    total_padded_frames = 0

    for _ in range(100):
        batch_indices = np.random.choice(len(lengths), batch_size, replace=False)
        batch_lengths = lengths[batch_indices]

        max_len = np.max(batch_lengths)
        min_len = np.min(batch_lengths)

        # 計算有效 vs padding 比例
        valid = np.sum(batch_lengths)
        padded = max_len * batch_size - valid

        total_valid_frames += valid
        total_padded_frames += padded

        batch_stats.append({
            'max_len': max_len,
            'min_len': min_len,
            'range': max_len - min_len,
            'padding_ratio': padded / (valid + padded)
        })

    ranges = [b['range'] for b in batch_stats]
    padding_ratios = [b['padding_ratio'] for b in batch_stats]

    overall_padding_ratio = total_padded_frames / (total_valid_frames + total_padded_frames)

    result = {
        'split': split_name,
        'total_samples': len(lengths),
        'length_stats': {
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'range': int(np.max(lengths) - np.min(lengths)),
        },
        'batch_simulation': {
            'batch_size': batch_size,
            'num_batches_simulated': 100,
            'avg_range_in_batch': float(np.mean(ranges)),
            'avg_padding_ratio': float(np.mean(padding_ratios)),
            'overall_padding_ratio': float(overall_padding_ratio),
        }
    }

    print(f"\n  模擬 100 個 batch:")
    print(f"  Batch 內長度差 (range):")
    print(f"    Mean: {np.mean(ranges):.1f} samples ({np.mean(ranges)/24000:.2f}s)")
    print(f"    Max: {np.max(ranges):.1f} samples ({np.max(ranges)/24000:.2f}s)")
    print(f"  Padding 比例:")
    print(f"    平均每 batch: {np.mean(padding_ratios)*100:.1f}%")
    print(f"    整體: {overall_padding_ratio*100:.1f}%")

    # Encoder 輸出 frame 估算 (stride ~320)
    encoder_stride = 320
    frame_lengths = lengths // encoder_stride

    print(f"\n--- Encoder Frame 統計 (stride={encoder_stride}) ---")
    print(f"  Frame 數範圍: {np.min(frame_lengths)} ~ {np.max(frame_lengths)}")
    print(f"  Mean: {np.mean(frame_lengths):.1f} frames")

    # 估算 padding frame 比例
    valid_frames_per_batch = []
    padded_frames_per_batch = []

    for _ in range(100):
        batch_indices = np.random.choice(len(frame_lengths), batch_size, replace=False)
        batch_frame_lens = frame_lengths[batch_indices]

        max_frames = np.max(batch_frame_lens)
        valid = np.sum(batch_frame_lens)
        padded = max_frames * batch_size - valid

        valid_frames_per_batch.append(valid)
        padded_frames_per_batch.append(padded)

    frame_padding_ratio = np.sum(padded_frames_per_batch) / (np.sum(valid_frames_per_batch) + np.sum(padded_frames_per_batch))

    result['frame_padding_ratio'] = float(frame_padding_ratio)

    print(f"  Frame-level padding 比例: {frame_padding_ratio*100:.1f}%")

    return result, lengths

def check_token_alignment(samples, split_name, max_samples=100):
    """
    檢查 cache 中的 token 長度是否對齊
    """
    print(f"\n{'='*60}")
    print(f"Token 長度對齊檢查 ({split_name})")
    print(f"{'='*60}")

    samples = samples[:max_samples]

    mismatches = 0
    for s in samples:
        if 'noisy_tokens' in s and 'clean_tokens' in s:
            noisy_tok_len = len(s['noisy_tokens']) if hasattr(s['noisy_tokens'], '__len__') else s['noisy_tokens'].shape[-1]
            clean_tok_len = len(s['clean_tokens']) if hasattr(s['clean_tokens'], '__len__') else s['clean_tokens'].shape[-1]
            if noisy_tok_len != clean_tok_len:
                mismatches += 1

    print(f"  檢查 {len(samples)} 個樣本")
    print(f"  Token 長度不一致: {mismatches} ({mismatches/len(samples)*100:.1f}%)")

    return mismatches

def main():
    print("=" * 70)
    print("exp_1212: 資料對齊問題驗證")
    print("=" * 70)

    results = {}

    # 載入資料
    print("\n載入資料...")
    train_samples = torch.load(TRAIN_CACHE, weights_only=False)
    val_samples = torch.load(VAL_CACHE, weights_only=False)
    print(f"  TRAIN: {len(train_samples)} samples")
    print(f"  VAL: {len(val_samples)} samples")

    # ========== 來源1: Per-Pair Mismatch ==========
    # TRAIN (抽樣)
    train_pair_result, train_diffs = analyze_per_pair_mismatch(
        train_samples, "TRAIN", max_samples=2000
    )
    results['train_per_pair'] = train_pair_result

    # VAL (全量)
    val_pair_result, val_diffs = analyze_per_pair_mismatch(
        val_samples, "VAL", max_samples=None
    )
    results['val_per_pair'] = val_pair_result

    # ========== 來源2: Cross-Sample Mismatch ==========
    train_cross_result, train_lengths = analyze_cross_sample_mismatch(
        train_samples, "TRAIN", batch_size=16, max_samples=2000
    )
    results['train_cross_sample'] = train_cross_result

    val_cross_result, val_lengths = analyze_cross_sample_mismatch(
        val_samples, "VAL", batch_size=16, max_samples=None
    )
    results['val_cross_sample'] = val_cross_result

    # ========== Token 對齊檢查 ==========
    train_tok_mismatch = check_token_alignment(train_samples, "TRAIN")
    val_tok_mismatch = check_token_alignment(val_samples, "VAL")

    results['token_alignment'] = {
        'train_mismatch': train_tok_mismatch,
        'val_mismatch': val_tok_mismatch
    }

    # ========== 總結 ==========
    print("\n" + "=" * 70)
    print("問題嚴重程度總結")
    print("=" * 70)

    print("\n【來源1: Per-Pair Mismatch】")
    print(f"  TRAIN: {train_pair_result['significant_mismatch_pct']:.1f}% 有顯著不一致 (>10ms)")
    print(f"  VAL: {val_pair_result['significant_mismatch_pct']:.1f}% 有顯著不一致 (>10ms)")
    if val_pair_result['significant_mismatch_pct'] > 10:
        print(f"  ⚠️  VAL 的 per-pair mismatch 嚴重！")

    print("\n【來源2: Cross-Sample Mismatch (Padding)】")
    print(f"  TRAIN: 約 {train_cross_result['frame_padding_ratio']*100:.1f}% frames 是 padding")
    print(f"  VAL: 約 {val_cross_result['frame_padding_ratio']*100:.1f}% frames 是 padding")
    if train_cross_result['frame_padding_ratio'] > 0.1:
        print(f"  ⚠️  Padding 比例較高，會稀釋 loss/acc！")

    print("\n【Token 長度對齊】")
    print(f"  TRAIN token mismatch: {results['token_alignment']['train_mismatch']}")
    print(f"  VAL token mismatch: {results['token_alignment']['val_mismatch']}")

    # 保存結果
    output_path = Path(__file__).parent / 'alignment_verification_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已保存到: {output_path}")

    return results

if __name__ == '__main__':
    main()
