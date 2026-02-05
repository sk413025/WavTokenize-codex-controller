"""
分析 Baseline (exp_k_v6 @ epoch 300) 的 Token 分布

目標：
1. 計算 train/val 的 student token 分布
2. 計算 train/val 的 teacher token 分布（作為 target 標準）
3. 統計每個 token 出現頻率並排序
4. 生成對比圖表：
   - Token frequency 排序圖（log scale）
   - Token 分布直方圖
   - Top-K token 集中度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import create_curriculum_dataloaders

# 設定
BASELINE_CHECKPOINT = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613/checkpoints/checkpoint_epoch300.pt"
OUTPUT_DIR = Path(__file__).parent / "baseline_token_analysis"
DEVICE = "cuda:0"
CODEBOOK_SIZE = 4096

# LoRA 配置（與 baseline 一致）
LORA_CONFIG = {
    'rank': 256,
    'alpha': 512,
    'dropout': 0.2,
    'intermediate_indices': [3, 6],
    'layer_weights': {3: 0.5, 6: 0.5},
}


def collect_tokens(model, data_loader, device, max_batches=None, split_name='train'):
    """
    收集 student 和 teacher 的 tokens

    Returns:
        student_codes: List of 1D tensors (all tokens, masked)
        teacher_codes: List of 1D tensors (all tokens, masked)
    """
    model.eval()

    student_codes = []
    teacher_codes = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Collecting {split_name} tokens")):
            if max_batches is not None and i >= max_batches:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            lengths = batch.get('lengths', None)

            # Ensure correct shape [B, 1, T]
            if clean_audio.dim() == 2:
                clean_audio = clean_audio.unsqueeze(1)
            if noisy_audio.dim() == 2:
                noisy_audio = noisy_audio.unsqueeze(1)

            # Forward pass
            output = model(clean_audio, noisy_audio)

            student_codes_batch = output['student_codes'].cpu()  # [B, T]
            teacher_codes_batch = output['teacher_codes'].cpu()  # [B, T]

            # Mask out padding
            if lengths is not None:
                lengths = lengths.cpu()
                hop = 320  # 24kHz / 75 fps
                frame_lens = (lengths + hop - 1) // hop
                frame_lens = torch.clamp(frame_lens, min=0, max=student_codes_batch.shape[1])

                for b in range(student_codes_batch.shape[0]):
                    L = int(frame_lens[b].item())
                    if L > 0:
                        student_codes.append(student_codes_batch[b, :L])
                        teacher_codes.append(teacher_codes_batch[b, :L])
            else:
                student_codes.append(student_codes_batch.reshape(-1))
                teacher_codes.append(teacher_codes_batch.reshape(-1))

    model.train()

    # Concatenate all tokens
    student_codes = torch.cat(student_codes, dim=0)  # [total_tokens]
    teacher_codes = torch.cat(teacher_codes, dim=0)  # [total_tokens]

    return student_codes, teacher_codes


def compute_token_statistics(codes, codebook_size, name=''):
    """
    計算 token 統計指標

    Returns:
        dict with statistics
    """
    # Count frequencies
    counts = torch.bincount(codes, minlength=codebook_size)
    probs = counts.float() / counts.sum()

    # Sort by frequency (descending)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    sorted_probs = sorted_counts.float() / counts.sum()

    # Entropy
    nonzero_probs = probs[probs > 0]
    entropy = -(nonzero_probs * torch.log2(nonzero_probs)).sum().item()

    # Top-K mass
    top_10_mass = sorted_probs[:10].sum().item()
    top_50_mass = sorted_probs[:50].sum().item()
    top_100_mass = sorted_probs[:100].sum().item()

    # Used codes
    used_codes = (counts > 0).sum().item()

    # Gini coefficient (measure of inequality)
    sorted_probs_np = sorted_probs.numpy()
    n = len(sorted_probs_np)
    index = np.arange(1, n + 1)
    gini = (2 * (index * sorted_probs_np).sum()) / (n * sorted_probs_np.sum()) - (n + 1) / n

    stats = {
        'name': name,
        'total_tokens': int(codes.shape[0]),
        'codebook_size': codebook_size,
        'used_codes': used_codes,
        'usage_pct': 100.0 * used_codes / codebook_size,
        'entropy': entropy,
        'max_entropy': np.log2(codebook_size),
        'entropy_ratio': entropy / np.log2(codebook_size),
        'top_10_mass': top_10_mass,
        'top_50_mass': top_50_mass,
        'top_100_mass': top_100_mass,
        'gini_coefficient': float(gini),
        'counts': counts.numpy(),
        'sorted_counts': sorted_counts.numpy(),
        'sorted_indices': sorted_indices.numpy(),
        'probs': probs.numpy(),
        'sorted_probs': sorted_probs.numpy(),
    }

    return stats


def plot_token_distributions(train_stats, val_stats, teacher_train_stats, teacher_val_stats, output_dir):
    """
    生成 4 組對比圖表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定風格
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # ==================== Figure 1: Token Frequency Ranking (Log Scale) ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Train Student
    ax = axes[0, 0]
    x = np.arange(1, len(train_stats['sorted_counts']) + 1)
    y = train_stats['sorted_counts']
    ax.loglog(x, y, linewidth=2, color='#e74c3c', alpha=0.8, label='Student (Baseline)')
    ax.set_xlabel('Token Rank', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Train Student Token Distribution\nEntropy: {train_stats["entropy"]:.2f}, Top-10: {train_stats["top_10_mass"]*100:.1f}%', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Train Teacher (Target)
    ax = axes[0, 1]
    x = np.arange(1, len(teacher_train_stats['sorted_counts']) + 1)
    y = teacher_train_stats['sorted_counts']
    ax.loglog(x, y, linewidth=2, color='#3498db', alpha=0.8, label='Teacher (Target)')
    ax.set_xlabel('Token Rank', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Train Teacher (Target) Token Distribution\nEntropy: {teacher_train_stats["entropy"]:.2f}, Top-10: {teacher_train_stats["top_10_mass"]*100:.1f}%', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Val Student
    ax = axes[1, 0]
    x = np.arange(1, len(val_stats['sorted_counts']) + 1)
    y = val_stats['sorted_counts']
    ax.loglog(x, y, linewidth=2, color='#e74c3c', alpha=0.8, label='Student (Baseline)')
    ax.set_xlabel('Token Rank', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Val Student Token Distribution\nEntropy: {val_stats["entropy"]:.2f}, Top-10: {val_stats["top_10_mass"]*100:.1f}%', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Val Teacher (Target)
    ax = axes[1, 1]
    x = np.arange(1, len(teacher_val_stats['sorted_counts']) + 1)
    y = teacher_val_stats['sorted_counts']
    ax.loglog(x, y, linewidth=2, color='#3498db', alpha=0.8, label='Teacher (Target)')
    ax.set_xlabel('Token Rank', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Val Teacher (Target) Token Distribution\nEntropy: {teacher_val_stats["entropy"]:.2f}, Top-10: {teacher_val_stats["top_10_mass"]*100:.1f}%', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'token_frequency_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'token_frequency_ranking.png'}")

    # ==================== Figure 2: Top-K Concentration ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Train
    ax = axes[0]
    k_values = [10, 50, 100, 200, 500, 1000]
    student_masses = [train_stats['sorted_probs'][:k].sum() for k in k_values]
    teacher_masses = [teacher_train_stats['sorted_probs'][:k].sum() for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35
    ax.bar(x - width/2, student_masses, width, label='Student', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, teacher_masses, width, label='Teacher (Target)', color='#3498db', alpha=0.7)
    ax.set_xlabel('Top-K Tokens', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Train: Top-K Token Concentration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in k_values], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Val
    ax = axes[1]
    student_masses = [val_stats['sorted_probs'][:k].sum() for k in k_values]
    teacher_masses = [teacher_val_stats['sorted_probs'][:k].sum() for k in k_values]

    ax.bar(x - width/2, student_masses, width, label='Student', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, teacher_masses, width, label='Teacher (Target)', color='#3498db', alpha=0.7)
    ax.set_xlabel('Top-K Tokens', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Val: Top-K Token Concentration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in k_values], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_k_concentration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'top_k_concentration.png'}")

    # ==================== Figure 3: Probability Distribution Histogram ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Train Student
    ax = axes[0, 0]
    used_probs = train_stats['probs'][train_stats['probs'] > 0]
    ax.hist(used_probs, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Token Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Train Student: Probability Distribution\n({train_stats["used_codes"]}/{train_stats["codebook_size"]} codes used)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Train Teacher
    ax = axes[0, 1]
    used_probs = teacher_train_stats['probs'][teacher_train_stats['probs'] > 0]
    ax.hist(used_probs, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Token Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Train Teacher: Probability Distribution\n({teacher_train_stats["used_codes"]}/{teacher_train_stats["codebook_size"]} codes used)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Val Student
    ax = axes[1, 0]
    used_probs = val_stats['probs'][val_stats['probs'] > 0]
    ax.hist(used_probs, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Token Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Val Student: Probability Distribution\n({val_stats["used_codes"]}/{val_stats["codebook_size"]} codes used)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Val Teacher
    ax = axes[1, 1]
    used_probs = teacher_val_stats['probs'][teacher_val_stats['probs'] > 0]
    ax.hist(used_probs, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Token Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Val Teacher: Probability Distribution\n({teacher_val_stats["used_codes"]}/{teacher_val_stats["codebook_size"]} codes used)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'probability_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'probability_histograms.png'}")

    # ==================== Figure 4: Comparison Summary ====================
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    metrics = ['Entropy', 'Top-10 Mass (%)', 'Top-100 Mass (%)', 'Used Codes (%)', 'Gini Coef']

    student_train_vals = [
        train_stats['entropy'],
        train_stats['top_10_mass'] * 100,
        train_stats['top_100_mass'] * 100,
        train_stats['usage_pct'],
        train_stats['gini_coefficient'] * 100
    ]

    teacher_train_vals = [
        teacher_train_stats['entropy'],
        teacher_train_stats['top_10_mass'] * 100,
        teacher_train_stats['top_100_mass'] * 100,
        teacher_train_stats['usage_pct'],
        teacher_train_stats['gini_coefficient'] * 100
    ]

    student_val_vals = [
        val_stats['entropy'],
        val_stats['top_10_mass'] * 100,
        val_stats['top_100_mass'] * 100,
        val_stats['usage_pct'],
        val_stats['gini_coefficient'] * 100
    ]

    teacher_val_vals = [
        teacher_val_stats['entropy'],
        teacher_val_stats['top_10_mass'] * 100,
        teacher_val_stats['top_100_mass'] * 100,
        teacher_val_stats['usage_pct'],
        teacher_val_stats['gini_coefficient'] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.2

    ax.bar(x - 1.5*width, student_train_vals, width, label='Student Train', color='#e74c3c', alpha=0.7)
    ax.bar(x - 0.5*width, teacher_train_vals, width, label='Teacher Train', color='#3498db', alpha=0.7)
    ax.bar(x + 0.5*width, student_val_vals, width, label='Student Val', color='#e67e22', alpha=0.7)
    ax.bar(x + 1.5*width, teacher_val_vals, width, label='Teacher Val', color='#1abc9c', alpha=0.7)

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Baseline Token Distribution Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")


def main():
    print("=" * 80)
    print("Baseline Token Distribution Analysis (exp_k_v6 @ epoch 300)")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/6] Loading baseline model...")
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
    # Remove 'student.' prefix from keys
    lora_state = {}
    for k, v in checkpoint['lora_state_dict'].items():
        if k.startswith('student.'):
            lora_state[k[8:]] = v  # Remove 'student.' prefix
        else:
            lora_state[k] = v
    model.student.load_state_dict(lora_state, strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val acc: {checkpoint['val_acc']:.4f}, Train acc: {checkpoint['train_acc']:.4f}")

    # Create dataloaders
    print("\n[2/6] Creating dataloaders...")
    train_loader, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=32,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("Note: Using num_workers=0 to avoid dataloader issues")

    # Collect train tokens
    print("\n[3/6] Collecting train tokens...")
    train_student_codes, train_teacher_codes = collect_tokens(
        model, train_loader, device, max_batches=None, split_name='train'
    )
    print(f"Collected {len(train_student_codes)} train tokens")

    # Collect val tokens
    print("\n[4/6] Collecting val tokens...")
    val_student_codes, val_teacher_codes = collect_tokens(
        model, val_loader, device, max_batches=None, split_name='val'
    )
    print(f"Collected {len(val_student_codes)} val tokens")

    # Compute statistics
    print("\n[5/6] Computing statistics...")
    train_stats = compute_token_statistics(train_student_codes, CODEBOOK_SIZE, 'train_student')
    val_stats = compute_token_statistics(val_student_codes, CODEBOOK_SIZE, 'val_student')
    teacher_train_stats = compute_token_statistics(train_teacher_codes, CODEBOOK_SIZE, 'train_teacher')
    teacher_val_stats = compute_token_statistics(val_teacher_codes, CODEBOOK_SIZE, 'val_teacher')

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)

    for stats, label in [
        (train_stats, "Train Student"),
        (teacher_train_stats, "Train Teacher (Target)"),
        (val_stats, "Val Student"),
        (teacher_val_stats, "Val Teacher (Target)"),
    ]:
        print(f"\n{label}:")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Used codes: {stats['used_codes']}/{stats['codebook_size']} ({stats['usage_pct']:.2f}%)")
        print(f"  Entropy: {stats['entropy']:.2f} / {stats['max_entropy']:.2f} ({stats['entropy_ratio']*100:.1f}%)")
        print(f"  Top-10 mass: {stats['top_10_mass']*100:.2f}%")
        print(f"  Top-50 mass: {stats['top_50_mass']*100:.2f}%")
        print(f"  Top-100 mass: {stats['top_100_mass']*100:.2f}%")
        print(f"  Gini coefficient: {stats['gini_coefficient']:.4f}")

    # Save statistics to JSON
    summary = {
        'baseline_checkpoint': BASELINE_CHECKPOINT,
        'codebook_size': CODEBOOK_SIZE,
        'train_student': {k: v for k, v in train_stats.items() if k not in ['counts', 'sorted_counts', 'sorted_indices', 'probs', 'sorted_probs']},
        'val_student': {k: v for k, v in val_stats.items() if k not in ['counts', 'sorted_counts', 'sorted_indices', 'probs', 'sorted_probs']},
        'train_teacher': {k: v for k, v in teacher_train_stats.items() if k not in ['counts', 'sorted_counts', 'sorted_indices', 'probs', 'sorted_probs']},
        'val_teacher': {k: v for k, v in teacher_val_stats.items() if k not in ['counts', 'sorted_counts', 'sorted_indices', 'probs', 'sorted_probs']},
    }

    with open(OUTPUT_DIR / 'statistics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved statistics to: {OUTPUT_DIR / 'statistics_summary.json'}")

    # Save full data
    np.savez(
        OUTPUT_DIR / 'token_distributions.npz',
        train_student_counts=train_stats['counts'],
        train_student_sorted=train_stats['sorted_counts'],
        val_student_counts=val_stats['counts'],
        val_student_sorted=val_stats['sorted_counts'],
        train_teacher_counts=teacher_train_stats['counts'],
        train_teacher_sorted=teacher_train_stats['sorted_counts'],
        val_teacher_counts=teacher_val_stats['counts'],
        val_teacher_sorted=teacher_val_stats['sorted_counts'],
    )
    print(f"Saved full distributions to: {OUTPUT_DIR / 'token_distributions.npz'}")

    # Plot
    print("\n[6/6] Generating plots...")
    plot_token_distributions(train_stats, val_stats, teacher_train_stats, teacher_val_stats, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
