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

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import create_curriculum_dataloaders

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


def collect_tokens_ultra_efficient(model, loader, device, max_batches=50, split_name='val'):
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{split_name} batches", total=min(max_batches, len(loader)))):
            if batch_idx >= max_batches:
                break

            try:
                clean_audio = batch['clean_audio'].to(device)
                noisy_audio = batch['noisy_audio'].to(device)

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

    return student_token_counts, teacher_token_counts, total_tokens


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
        'top_10_mass': float(top10_mass),
        'top_50_mass': float(top50_mass),
        'top_100_mass': float(top100_mass),
    }

    return df, stats


def main():
    print("=" * 80)
    print("Collecting REAL Token Data from Baseline Model")
    print("(Ultra Efficient Mode - Direct Count Aggregation)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Create dataloaders (small batch size to avoid OOM)
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=16,  # Reduced from 32
        num_workers=0,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Collect train tokens
    print("\n[3/7] Collecting train tokens...")
    train_student_counts, train_teacher_counts, train_total = collect_tokens_ultra_efficient(
        model, train_loader, device, max_batches=100, split_name='train'
    )

    # Collect validation tokens
    print("\n[4/7] Collecting validation tokens...")
    val_student_counts, val_teacher_counts, val_total = collect_tokens_ultra_efficient(
        model, val_loader, device, max_batches=100, split_name='val'
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

    # Save summary JSON
    summary = {
        'real_data': True,
        'checkpoint': BASELINE_CHECKPOINT,
        'epoch': int(checkpoint['epoch']),
        'train_student': train_student_stats,
        'train_teacher': train_teacher_stats,
        'validation_student': val_student_stats,
        'validation_teacher': val_teacher_stats,
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
