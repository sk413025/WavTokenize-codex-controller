"""
高效版本：從 baseline checkpoint 收集真實的 token 分布數據
使用 batch 處理並限制樣本數量以加速
"""

import torch
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
OUTPUT_DIR = Path(__file__).parent / "baseline_token_analysis_real"
DEVICE = "cuda:0"
CODEBOOK_SIZE = 4096

# 限制處理的 batch 數量以加速（設為 None 處理全部）
MAX_TRAIN_BATCHES = 100  # ~3200 samples (100 * 32)
MAX_VAL_BATCHES = None   # Process all val data (~1440 samples)

LORA_CONFIG = {
    'rank': 256,
    'alpha': 512,
    'dropout': 0.2,
    'intermediate_indices': [3, 6],
}


def collect_tokens_efficient(model, data_loader, device, max_batches=None, split_name='train'):
    """高效收集 tokens"""
    model.eval()

    student_codes_list = []
    teacher_codes_list = []

    print(f"\nCollecting {split_name} tokens...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Processing {split_name}")):
            if max_batches is not None and i >= max_batches:
                break

            try:
                noisy_audio = batch['noisy_audio'].to(device)
                clean_audio = batch['clean_audio'].to(device)
                lengths = batch.get('lengths', None)

                # Ensure correct shape
                if clean_audio.dim() == 2:
                    clean_audio = clean_audio.unsqueeze(1)
                if noisy_audio.dim() == 2:
                    noisy_audio = noisy_audio.unsqueeze(1)

                # Forward pass
                output = model(clean_audio, noisy_audio)

                student_codes = output['student_codes'].cpu()  # [B, T]
                teacher_codes = output['teacher_codes'].cpu()  # [B, T]

                # Mask padding if lengths available
                if lengths is not None:
                    lengths_cpu = lengths.cpu()
                    hop = 320
                    frame_lens = (lengths_cpu + hop - 1) // hop
                    frame_lens = torch.clamp(frame_lens, min=0, max=student_codes.shape[1])

                    for b in range(student_codes.shape[0]):
                        L = int(frame_lens[b].item())
                        if L > 0:
                            student_codes_list.append(student_codes[b, :L])
                            teacher_codes_list.append(teacher_codes[b, :L])
                else:
                    student_codes_list.append(student_codes.reshape(-1))
                    teacher_codes_list.append(teacher_codes.reshape(-1))

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

    if len(student_codes_list) == 0:
        raise ValueError(f"No {split_name} tokens collected!")

    student_all = torch.cat(student_codes_list, dim=0)
    teacher_all = torch.cat(teacher_codes_list, dim=0)

    print(f"Collected {len(student_all):,} {split_name} tokens")

    model.train()
    return student_all, teacher_all


def compute_statistics(codes, codebook_size, name=''):
    """計算詳細統計"""
    codes_np = codes.numpy()

    # Frequency counts
    counts = np.bincount(codes_np, minlength=codebook_size)
    probs = counts / counts.sum()

    # Sort by frequency
    sorted_indices = np.argsort(-counts)
    sorted_counts = counts[sorted_indices]
    sorted_probs = probs[sorted_indices]

    # Entropy
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

    # Top-K mass
    top_10_mass = sorted_probs[:10].sum()
    top_50_mass = sorted_probs[:50].sum()
    top_100_mass = sorted_probs[:100].sum()

    # Used codes
    used_codes = (counts > 0).sum()

    # Gini coefficient
    n = len(sorted_probs)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_probs)) / (n * sorted_probs.sum()) - (n + 1) / n

    return {
        'name': name,
        'total_tokens': int(len(codes)),
        'codebook_size': codebook_size,
        'used_codes': int(used_codes),
        'usage_pct': 100.0 * used_codes / codebook_size,
        'entropy': float(entropy),
        'max_entropy': np.log2(codebook_size),
        'entropy_ratio': entropy / np.log2(codebook_size),
        'top_10_mass': float(top_10_mass),
        'top_50_mass': float(top_50_mass),
        'top_100_mass': float(top_100_mass),
        'gini_coefficient': float(gini),
        'counts': counts,
        'sorted_counts': sorted_counts,
        'sorted_indices': sorted_indices,
        'probs': probs,
        'sorted_probs': sorted_probs,
    }


def plot_comprehensive_analysis(stats_dict, output_dir):
    """生成綜合分析圖表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # Extract data
    target_train = stats_dict['target_train']
    target_val = stats_dict['target_val']
    student_train = stats_dict['student_train']
    student_val = stats_dict['student_val']

    # ==================== Figure 1: Token Frequency Ranking (6 subplots) ====================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Row 1: Target (Teacher)
    ax1 = fig.add_subplot(gs[0, 0])
    ranks = np.arange(1, len(target_train['sorted_counts']) + 1)
    ax1.loglog(ranks, target_train['sorted_counts'], 'o-', color='#3498db',
               markersize=1, linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Token Rank', fontsize=11)
    ax1.set_ylabel('Frequency (log)', fontsize=11)
    ax1.set_title(f'Target Train Distribution\nEntropy: {target_train["entropy"]:.2f}, Top-10: {target_train["top_10_mass"]*100:.1f}%',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')

    ax2 = fig.add_subplot(gs[0, 1])
    ranks = np.arange(1, len(target_val['sorted_counts']) + 1)
    ax2.loglog(ranks, target_val['sorted_counts'], 'o-', color='#2980b9',
               markersize=1, linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Token Rank', fontsize=11)
    ax2.set_ylabel('Frequency (log)', fontsize=11)
    ax2.set_title(f'Target Val Distribution\nEntropy: {target_val["entropy"]:.2f}, Top-10: {target_val["top_10_mass"]*100:.1f}%',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # Row 2: Student (Baseline)
    ax3 = fig.add_subplot(gs[1, 0])
    ranks = np.arange(1, len(student_train['sorted_counts']) + 1)
    ax3.loglog(ranks, student_train['sorted_counts'], 'o-', color='#e74c3c',
               markersize=1, linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Token Rank', fontsize=11)
    ax3.set_ylabel('Frequency (log)', fontsize=11)
    ax3.set_title(f'Student Train Distribution\nEntropy: {student_train["entropy"]:.2f}, Top-10: {student_train["top_10_mass"]*100:.1f}%',
                 fontsize=12, fontweight='bold', color='#c0392b')
    ax3.grid(True, alpha=0.3, which='both')

    ax4 = fig.add_subplot(gs[1, 1])
    ranks = np.arange(1, len(student_val['sorted_counts']) + 1)
    ax4.loglog(ranks, student_val['sorted_counts'], 'o-', color='#c0392b',
               markersize=1, linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Token Rank', fontsize=11)
    ax4.set_ylabel('Frequency (log)', fontsize=11)
    ax4.set_title(f'Student Val Distribution (COLLAPSED)\nEntropy: {student_val["entropy"]:.2f}, Top-10: {student_val["top_10_mass"]*100:.1f}%',
                 fontsize=12, fontweight='bold', color='#c0392b')
    ax4.grid(True, alpha=0.3, which='both')

    # Row 3: Direct Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.loglog(ranks, target_train['sorted_counts'], 'o-', color='#3498db',
               markersize=1, linewidth=2, alpha=0.8, label='Target (Ideal)')
    ax5.loglog(ranks, student_train['sorted_counts'], 'o-', color='#e74c3c',
               markersize=1, linewidth=2, alpha=0.8, label='Student (Collapsed)')
    ax5.set_xlabel('Token Rank', fontsize=11)
    ax5.set_ylabel('Frequency (log)', fontsize=11)
    ax5.set_title('Train: Target vs Student', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, which='both')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.loglog(ranks, target_val['sorted_counts'], 'o-', color='#2980b9',
               markersize=1, linewidth=2, alpha=0.8, label='Target (Ideal)')
    ax6.loglog(ranks, student_val['sorted_counts'], 'o-', color='#c0392b',
               markersize=1, linewidth=2, alpha=0.8, label='Student (Collapsed)')
    ax6.set_xlabel('Token Rank', fontsize=11)
    ax6.set_ylabel('Frequency (log)', fontsize=11)
    ax6.set_title('Val: Target vs Student', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, which='both')

    plt.savefig(output_dir / 'token_frequency_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'token_frequency_comprehensive.png'}")

    # ==================== Figure 2: Metrics Comparison ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    datasets = ['Target\nTrain', 'Target\nVal', 'Student\nTrain', 'Student\nVal']
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']

    # Entropy
    ax = axes[0, 0]
    values = [target_train['entropy'], target_val['entropy'],
             student_train['entropy'], student_val['entropy']]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=12.0, color='green', linestyle='--', linewidth=2, label='Ideal (12.0)')
    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Token Entropy', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Top-10 Mass
    ax = axes[0, 1]
    values = [target_train['top_10_mass']*100, target_val['top_10_mass']*100,
             student_train['top_10_mass']*100, student_val['top_10_mass']*100]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Ideal (<0.5%)')
    ax.set_ylabel('Top-10 Mass (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-10 Token Concentration', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Top-100 Mass
    ax = axes[0, 2]
    values = [target_train['top_100_mass']*100, target_val['top_100_mass']*100,
             student_train['top_100_mass']*100, student_val['top_100_mass']*100]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=2.44, color='green', linestyle='--', linewidth=2, label='Ideal (2.44%)')
    ax.set_ylabel('Top-100 Mass (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-100 Token Concentration', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Used Codes
    ax = axes[1, 0]
    values = [target_train['used_codes'], target_val['used_codes'],
             student_train['used_codes'], student_val['used_codes']]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=4096, color='green', linestyle='--', linewidth=2, label='Ideal (4096)')
    ax.set_ylabel('Used Codes', fontsize=12, fontweight='bold')
    ax.set_title('Codebook Usage (Count)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Usage Percentage
    ax = axes[1, 1]
    values = [target_train['usage_pct'], target_val['usage_pct'],
             student_train['usage_pct'], student_val['usage_pct']]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Ideal (100%)')
    ax.set_ylabel('Codebook Usage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Codebook Usage (Percentage)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Gini Coefficient
    ax = axes[1, 2]
    values = [target_train['gini_coefficient'], target_val['gini_coefficient'],
             student_train['gini_coefficient'], student_val['gini_coefficient']]
    bars = ax.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.0, color='green', linestyle='--', linewidth=2, label='Ideal (0.0)')
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Inequality', fontsize=13, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'metrics_comparison_detailed.png'}")

    # ==================== Figure 3: Top-20 Token Usage ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (stats, title, color) in enumerate([
        (target_train, 'Target Train', '#3498db'),
        (target_val, 'Target Val', '#2980b9'),
        (student_train, 'Student Train', '#e74c3c'),
        (student_val, 'Student Val (COLLAPSED)', '#c0392b'),
    ]):
        ax = axes[idx // 2, idx % 2]
        top_20_probs = stats['sorted_probs'][:20] * 100
        top_20_ids = stats['sorted_indices'][:20]

        bars = ax.bar(range(20), top_20_probs, color=color, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Token Rank', fontsize=11)
        ax.set_ylabel('Usage (%)', fontsize=11)
        ax.set_title(f'{title}\nTop-20 Tokens', fontsize=12, fontweight='bold')
        ax.set_xticks(range(20))
        ax.set_xticklabels([f'{i+1}' for i in range(20)], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add token IDs as text
        for i, (bar, token_id) in enumerate(zip(bars, top_20_ids)):
            height = bar.get_height()
            if height > 0.5:  # Only show for significant bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'#{token_id}\n{height:.1f}%',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'top20_tokens_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'top20_tokens_usage.png'}")


def main():
    print("=" * 80)
    print("Baseline Real Token Distribution Analysis")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/5] Loading model...")
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
    print(f"✓ Loaded epoch {checkpoint['epoch']}")

    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=32,
        num_workers=0,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    print(f"✓ Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Collect tokens
    print("\n[3/5] Collecting tokens...")
    train_student, train_teacher = collect_tokens_efficient(
        model, train_loader, device, MAX_TRAIN_BATCHES, 'train'
    )
    val_student, val_teacher = collect_tokens_efficient(
        model, val_loader, device, MAX_VAL_BATCHES, 'val'
    )

    # Compute statistics
    print("\n[4/5] Computing statistics...")
    stats = {
        'target_train': compute_statistics(train_teacher, CODEBOOK_SIZE, 'target_train'),
        'target_val': compute_statistics(val_teacher, CODEBOOK_SIZE, 'target_val'),
        'student_train': compute_statistics(train_student, CODEBOOK_SIZE, 'student_train'),
        'student_val': compute_statistics(val_student, CODEBOOK_SIZE, 'student_val'),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    for key, s in stats.items():
        print(f"\n{key.upper()}:")
        print(f"  Entropy: {s['entropy']:.2f} / {s['max_entropy']:.2f} ({s['entropy_ratio']*100:.1f}%)")
        print(f"  Top-10: {s['top_10_mass']*100:.2f}%, Top-100: {s['top_100_mass']*100:.2f}%")
        print(f"  Used: {s['used_codes']}/{s['codebook_size']} ({s['usage_pct']:.1f}%)")
        print(f"  Gini: {s['gini_coefficient']:.4f}")

    # Save stats
    stats_json = {k: {kk: vv for kk, vv in v.items()
                     if kk not in ['counts', 'sorted_counts', 'sorted_indices', 'probs', 'sorted_probs']}
                  for k, v in stats.items()}
    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'statistics.json'}")

    # Generate plots
    print("\n[5/5] Generating plots...")
    plot_comprehensive_analysis(stats, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"✓ Analysis complete! Results in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
