"""
從 baseline checkpoint 收集真實的 token 數據
採用極度節省記憶體的策略來避免 CUDA OOM
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
import gc
import re
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from torch.utils.data import DataLoader

# 設定
BASELINE_CHECKPOINT = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613/checkpoints/checkpoint_epoch300.pt"
OUTPUT_DIR = Path(__file__).parent / "baseline_token_analysis"
DEVICE = "cuda:0"
CODEBOOK_SIZE = 4096

# LoRA config (from baseline exp_k_v6)
LORA_CONFIG = {
    'rank': 256,
    'alpha': 512,
    'dropout': 0.1,
    'intermediate_indices': [2, 5, 8, 11, 14, 17, 20, 23]
}


def _maybe_get_sample_rate_from_config(config_path: str) -> Optional[int]:
    try:
        text = Path(config_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    patterns = [
        r"^\s*sample_rate\s*:\s*(\d+)\s*$",
        r"^\s*sr\s*:\s*(\d+)\s*$",
        r"^\s*sampling_rate\s*:\s*(\d+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _describe_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _accumulate_audio_stats(
    *,
    clean_audio: torch.Tensor,
    noisy_audio: torch.Tensor,
    lengths: Optional[torch.Tensor],
    sample_rate: Optional[int],
    rms_silence_threshold: float = 1e-3,
) -> Dict[str, Any]:
    """
    Compute per-sample audio-content statistics on-the-fly (lightweight).

    Notes:
    - clean_audio/noisy_audio should be shaped [B, 1, T] or [B, T].
    - lengths (if provided) are in samples (before padding).
    """
    if clean_audio.dim() == 3:
        clean = clean_audio.squeeze(1)
    else:
        clean = clean_audio
    if noisy_audio.dim() == 3:
        noisy = noisy_audio.squeeze(1)
    else:
        noisy = noisy_audio

    bsz, t = clean.shape
    if lengths is None:
        lengths = torch.full((bsz,), t, device=clean.device, dtype=torch.long)
    else:
        lengths = lengths.to(device=clean.device, dtype=torch.long)
        lengths = torch.clamp(lengths, min=1, max=t)

    idx = torch.arange(t, device=clean.device)[None, :]
    mask = idx < lengths[:, None]
    mask_f = mask.to(dtype=clean.dtype)

    # Clean RMS / Abs mean
    clean_sq_sum = (clean * clean * mask_f).sum(dim=1)
    clean_mean_sq = clean_sq_sum / lengths.to(dtype=clean.dtype)
    clean_rms = torch.sqrt(torch.clamp(clean_mean_sq, min=0.0))
    clean_abs_mean = (clean.abs() * mask_f).sum(dim=1) / lengths.to(dtype=clean.dtype)

    # Noise residual stats (noisy-clean)
    noise = noisy - clean
    noise_sq_sum = (noise * noise * mask_f).sum(dim=1)
    noise_mean_sq = noise_sq_sum / lengths.to(dtype=clean.dtype)
    noise_rms = torch.sqrt(torch.clamp(noise_mean_sq, min=0.0))

    # SNR estimate per sample
    eps = 1e-12
    snr_db = 10.0 * torch.log10((clean_mean_sq + eps) / (noise_mean_sq + eps))

    # Length stats
    lengths_cpu = lengths.detach().cpu().numpy().astype(np.int64)
    if sample_rate and sample_rate > 0:
        duration_s = (lengths_cpu / float(sample_rate)).tolist()
    else:
        duration_s = []

    clean_rms_cpu = clean_rms.detach().cpu().numpy().astype(np.float64).tolist()
    clean_abs_mean_cpu = clean_abs_mean.detach().cpu().numpy().astype(np.float64).tolist()
    noise_rms_cpu = noise_rms.detach().cpu().numpy().astype(np.float64).tolist()
    snr_db_cpu = snr_db.detach().cpu().numpy().astype(np.float64).tolist()

    silence_flags = [1 if v < rms_silence_threshold else 0 for v in clean_rms_cpu]
    silence_frac = float(np.mean(silence_flags)) if silence_flags else None

    return {
        "length_samples": lengths_cpu.tolist(),
        "duration_seconds": duration_s,
        "clean_rms": clean_rms_cpu,
        "clean_abs_mean": clean_abs_mean_cpu,
        "noise_rms": noise_rms_cpu,
        "snr_db": snr_db_cpu,
        "rms_silence_threshold": float(rms_silence_threshold),
        "silence_frac": silence_frac,
    }


def _merge_audio_stats(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a:
        return b
    out = dict(a)
    for k in ["length_samples", "duration_seconds", "clean_rms", "clean_abs_mean", "noise_rms", "snr_db"]:
        out.setdefault(k, [])
        out[k].extend(b.get(k, []))
    # keep the threshold; recompute silence_frac at finalize time
    out["rms_silence_threshold"] = a.get("rms_silence_threshold", b.get("rms_silence_threshold", 1e-3))
    return out


def _finalize_audio_stats(raw: Dict[str, Any], sample_rate: Optional[int]) -> Dict[str, Any]:
    thr = float(raw.get("rms_silence_threshold", 1e-3))
    clean_rms = raw.get("clean_rms", []) or []
    silence_frac = float(np.mean([1 if v < thr else 0 for v in clean_rms])) if clean_rms else None
    return {
        "sample_rate_hz": sample_rate,
        "rms_silence_threshold": thr,
        "silence_frac": silence_frac,
        "length_samples": _describe_distribution(raw.get("length_samples", []) or []),
        "duration_seconds": _describe_distribution(raw.get("duration_seconds", []) or []),
        "clean_rms": _describe_distribution(clean_rms),
        "clean_abs_mean": _describe_distribution(raw.get("clean_abs_mean", []) or []),
        "noise_rms": _describe_distribution(raw.get("noise_rms", []) or []),
        "snr_db": _describe_distribution(raw.get("snr_db", []) or []),
    }


def collect_tokens_ultra_efficient(
    model,
    loader,
    device,
    max_batches: Optional[int] = 50,
    split_name: str = "val",
    sample_rate: Optional[int] = None,
):
    """
    Ultra efficient token collection - 只收集 token counts
    不儲存個別 tokens，直接累計統計
    """
    print(f"Collecting {split_name} tokens (ultra efficient mode)...")

    model.eval()

    # 只儲存 token 計數，不儲存所有 tokens
    student_token_counts = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
    teacher_token_counts = np.zeros(CODEBOOK_SIZE, dtype=np.int64)

    total_tokens = 0
    raw_audio_stats: Dict[str, Any] = {}

    with torch.no_grad():
        total_est = len(loader) if max_batches is None else min(max_batches, len(loader))
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{split_name} batches", total=total_est)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                clean_audio = batch['clean_audio'].to(device)
                noisy_audio = batch['noisy_audio'].to(device)
                lengths = batch.get("lengths", None)

                # Ensure correct shape
                if clean_audio.dim() == 1:
                    clean_audio = clean_audio.unsqueeze(0).unsqueeze(0)
                elif clean_audio.dim() == 2:
                    clean_audio = clean_audio.unsqueeze(1)

                if noisy_audio.dim() == 1:
                    noisy_audio = noisy_audio.unsqueeze(0).unsqueeze(0)
                elif noisy_audio.dim() == 2:
                    noisy_audio = noisy_audio.unsqueeze(1)

                # Forward pass
                output = model(clean_audio, noisy_audio)

                # Audio stats (content distribution)
                batch_audio_stats = _accumulate_audio_stats(
                    clean_audio=clean_audio,
                    noisy_audio=noisy_audio,
                    lengths=lengths,
                    sample_rate=sample_rate,
                )
                raw_audio_stats = _merge_audio_stats(raw_audio_stats, batch_audio_stats)

                # Extract tokens
                student_codes = output['student_codes'].cpu().numpy().flatten()
                teacher_codes = output['teacher_codes'].cpu().numpy().flatten()

                # 直接累計到 counts（不儲存個別 tokens）
                unique_s, counts_s = np.unique(student_codes, return_counts=True)
                unique_t, counts_t = np.unique(teacher_codes, return_counts=True)

                student_token_counts[unique_s] += counts_s
                teacher_token_counts[unique_t] += counts_t

                total_tokens += len(student_codes)

                # 清理記憶體
                del clean_audio, noisy_audio, output, student_codes, teacher_codes
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    print(f"Processed {total_tokens:,} tokens from {split_name}")

    audio_stats = _finalize_audio_stats(raw_audio_stats, sample_rate)
    return student_token_counts, teacher_token_counts, total_tokens, audio_stats


def create_token_ranking_from_counts(token_counts, total_tokens, split_name):
    """
    從 token counts 創建排名 DataFrame
    """
    # 找出非零的 tokens
    used_token_ids = np.nonzero(token_counts)[0]
    counts = token_counts[used_token_ids]

    # 排序（降序）
    sorted_indices = np.argsort(counts)[::-1]
    used_token_ids = used_token_ids[sorted_indices]
    counts = counts[sorted_indices]

    # 計算頻率
    frequencies = (counts / total_tokens) * 100
    cumulative_freqs = np.cumsum(frequencies)

    # 創建 DataFrame
    df = pd.DataFrame({
        'rank': np.arange(1, len(used_token_ids) + 1),
        'token_id': used_token_ids,
        'count': counts,
        'frequency': frequencies,
        'cumulative_freq': cumulative_freqs
    })

    # 計算統計指標
    entropy = -np.sum((frequencies / 100) * np.log2(frequencies / 100 + 1e-10))
    top10_mass = cumulative_freqs[9] if len(cumulative_freqs) >= 10 else cumulative_freqs[-1]
    top50_mass = cumulative_freqs[49] if len(cumulative_freqs) >= 50 else cumulative_freqs[-1]
    top100_mass = cumulative_freqs[99] if len(cumulative_freqs) >= 100 else cumulative_freqs[-1]
    used_codes = len(used_token_ids)

    stats = {
        'split': split_name,
        'total_tokens': int(total_tokens),
        'used_codes': int(used_codes),
        'usage_pct': float(used_codes / CODEBOOK_SIZE * 100),
        'entropy': float(entropy),
        # Backward-compatible keys (percent units)
        'top_10_mass': float(top10_mass),
        'top_50_mass': float(top50_mass),
        'top_100_mass': float(top100_mass),
        # Canonical explicit-unit keys
        'top_10_mass_pct': float(top10_mass),
        'top_50_mass_pct': float(top50_mass),
        'top_100_mass_pct': float(top100_mass),
        'top_10_mass_frac': float(top10_mass) / 100.0,
        'top_50_mass_frac': float(top50_mass) / 100.0,
        'top_100_mass_frac': float(top100_mass) / 100.0,
        'codebook_size': int(CODEBOOK_SIZE),
        'units': {
            'entropy': 'bits',
            'usage_pct': 'percent',
            'top_10_mass': 'percent',
            'top_50_mass': 'percent',
            'top_100_mass': 'percent',
        },
    }

    return df, stats


def main():
    print("=" * 80)
    print("Collecting REAL Token Data from Baseline Model")
    print("(Ultra Efficient Mode - Direct Count Aggregation)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_rate = _maybe_get_sample_rate_from_config(WAVTOK_CONFIG)

    # Load model
    print("\n[1/5] Loading baseline model...")
    device = torch.device(DEVICE)
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=LORA_CONFIG['rank'],
        lora_alpha=LORA_CONFIG['alpha'],
        lora_dropout=LORA_CONFIG['dropout'],
        intermediate_indices=LORA_CONFIG['intermediate_indices'],
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=device)
    lora_state = {}
    for k, v in checkpoint['lora_state_dict'].items():
        if k.startswith('student.'):
            lora_state[k[8:]] = v
        else:
            lora_state[k] = v
    model.student.load_state_dict(lora_state, strict=False)
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val acc: {checkpoint['val_acc']:.4f}, Train acc: {checkpoint['train_acc']:.4f}")

    # Create deterministic dataloaders (no curriculum sampler / no shuffle)
    print("\n[2/5] Creating dataloaders...")
    train_dataset = CurriculumDataset(
        TRAIN_CACHE,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    val_dataset = CurriculumDataset(
        VAL_CACHE,
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_curriculum,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_curriculum,
        pin_memory=True,
    )
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Collect train tokens
    print("\n[3/7] Collecting train tokens...")
    train_student_counts, train_teacher_counts, train_total, train_audio_stats = collect_tokens_ultra_efficient(
        model, train_loader, device, max_batches=None, split_name='train', sample_rate=sample_rate
    )

    # Collect validation tokens
    print("\n[4/7] Collecting validation tokens...")
    val_student_counts, val_teacher_counts, val_total, val_audio_stats = collect_tokens_ultra_efficient(
        model, val_loader, device, max_batches=None, split_name='val', sample_rate=sample_rate
    )

    # Create train ranking
    print("\n[5/7] Creating train token ranking...")
    train_student_df, train_student_stats = create_token_ranking_from_counts(
        train_student_counts, train_total, 'train_student'
    )
    train_teacher_df, train_teacher_stats = create_token_ranking_from_counts(
        train_teacher_counts, train_total, 'train_teacher'
    )

    # Create val ranking
    print("\n[6/7] Creating validation token ranking...")
    val_student_df, val_student_stats = create_token_ranking_from_counts(
        val_student_counts, val_total, 'val_student'
    )
    val_teacher_df, val_teacher_stats = create_token_ranking_from_counts(
        val_teacher_counts, val_total, 'val_teacher'
    )

    # Save results
    print("\n[7/7] Saving results...")
    train_student_df.to_csv(OUTPUT_DIR / 'real_train_student_token_ranking.csv', index=False)
    train_teacher_df.to_csv(OUTPUT_DIR / 'real_train_teacher_token_ranking.csv', index=False)
    val_student_df.to_csv(OUTPUT_DIR / 'real_val_student_token_ranking.csv', index=False)
    val_teacher_df.to_csv(OUTPUT_DIR / 'real_val_teacher_token_ranking.csv', index=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("TRAIN SET STATISTICS (REAL DATA)")
    print("=" * 80)

    print("\n【Student (Baseline Model) - TRAIN】")
    print(f"  Total tokens: {train_student_stats['total_tokens']:,}")
    print(f"  Used codes: {train_student_stats['used_codes']}/{CODEBOOK_SIZE} ({train_student_stats['usage_pct']:.2f}%)")
    print(f"  Entropy: {train_student_stats['entropy']:.2f} bits")
    print(f"  Top-10 mass: {train_student_stats['top_10_mass']:.2f}%")
    print(f"  Top-50 mass: {train_student_stats['top_50_mass']:.2f}%")
    print(f"  Top-100 mass: {train_student_stats['top_100_mass']:.2f}%")

    print("\n【Top-10 Most Frequent Tokens (Student Train)】")
    for i in range(min(10, len(train_student_df))):
        row = train_student_df.iloc[i]
        print(f"  #{int(row['rank']):2d}: Token {int(row['token_id']):4d} | {row['frequency']:6.2f}% | Count: {int(row['count']):8,}")

    print("\n" + "=" * 80)
    print("VALIDATION SET STATISTICS (REAL DATA)")
    print("=" * 80)

    print("\n【Student (Baseline Model) - VAL】")
    print(f"  Total tokens: {val_student_stats['total_tokens']:,}")
    print(f"  Used codes: {val_student_stats['used_codes']}/{CODEBOOK_SIZE} ({val_student_stats['usage_pct']:.2f}%)")
    print(f"  Entropy: {val_student_stats['entropy']:.2f} bits")
    print(f"  Top-10 mass: {val_student_stats['top_10_mass']:.2f}%")
    print(f"  Top-50 mass: {val_student_stats['top_50_mass']:.2f}%")
    print(f"  Top-100 mass: {val_student_stats['top_100_mass']:.2f}%")

    print("\n【Top-10 Most Frequent Tokens (Student Val)】")
    for i in range(min(10, len(val_student_df))):
        row = val_student_df.iloc[i]
        print(f"  #{int(row['rank']):2d}: Token {int(row['token_id']):4d} | {row['frequency']:6.2f}% | Count: {int(row['count']):8,}")

    print("\n【Teacher (Reference) - VAL】")
    print(f"  Total tokens: {val_teacher_stats['total_tokens']:,}")
    print(f"  Used codes: {val_teacher_stats['used_codes']}/{CODEBOOK_SIZE} ({val_teacher_stats['usage_pct']:.2f}%)")
    print(f"  Entropy: {val_teacher_stats['entropy']:.2f} bits")
    print(f"  Top-10 mass: {val_teacher_stats['top_10_mass']:.2f}%")

    print("\n" + "=" * 80)
    print("AUDIO CONTENT STATS (REAL DATA, Clean/Noisy)")
    print("=" * 80)
    print(f"Sample rate (from config): {sample_rate if sample_rate else 'N/A'}")
    for split_name, s in [("train", train_audio_stats), ("val", val_audio_stats)]:
        print(f"\n【{split_name.upper()}】")
        print(f"  Clean RMS mean/std: {s['clean_rms']['mean']:.6f} / {s['clean_rms']['std']:.6f}")
        print(f"  Clean RMS p10/p50/p90: {s['clean_rms']['p10']:.6f} / {s['clean_rms']['p50']:.6f} / {s['clean_rms']['p90']:.6f}")
        print(f"  Near-silence (RMS<{s['rms_silence_threshold']:.1e}) frac: {s['silence_frac'] if s['silence_frac'] is not None else 'N/A'}")
        print(f"  Noise RMS mean/std: {s['noise_rms']['mean']:.6f} / {s['noise_rms']['std']:.6f}")
        print(f"  SNR(dB) mean/std: {s['snr_db']['mean']:.3f} / {s['snr_db']['std']:.3f}")
        if s['duration_seconds']['n'] and s['duration_seconds']['mean'] is not None:
            print(f"  Duration(s) mean/p50: {s['duration_seconds']['mean']:.3f} / {s['duration_seconds']['p50']:.3f}")
        else:
            print(f"  Length(samples) mean/p50: {s['length_samples']['mean']:.1f} / {s['length_samples']['p50']:.1f}")

    # Save summary JSON
    summary = {
        'real_data': True,
        'checkpoint': BASELINE_CHECKPOINT,
        'epoch': int(checkpoint['epoch']),
        'wavtok_config': WAVTOK_CONFIG,
        'sample_rate_hz': sample_rate,
        'train_student': train_student_stats,
        'train_teacher': train_teacher_stats,
        'validation_student': val_student_stats,
        'validation_teacher': val_teacher_stats,
        'audio_stats': {
            'train': train_audio_stats,
            'val': val_audio_stats,
        },
        'top10_train_student_tokens': train_student_df.head(10).to_dict('records'),
        'top10_val_student_tokens': val_student_df.head(10).to_dict('records'),
        'top10_val_teacher_tokens': val_teacher_df.head(10).to_dict('records'),
    }

    with open(OUTPUT_DIR / 'real_token_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {OUTPUT_DIR}")
    print("  - real_train_student_token_ranking.csv")
    print("  - real_train_teacher_token_ranking.csv")
    print("  - real_val_student_token_ranking.csv")
    print("  - real_val_teacher_token_ranking.csv")
    print("  - real_token_statistics.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
