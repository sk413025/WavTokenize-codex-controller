"""
使用真實的 token 數據生成可視化圖表（Train & Val 在同一張圖）
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 設定
BASE_DIR = Path(__file__).parent / "baseline_token_analysis"
OUTPUT_DIR = BASE_DIR
CODEBOOK_SIZE = 4096

# 理想分布（Teacher target）
TEACHER_TARGET = {
    'entropy': 12.0,  # log2(4096) = 12 bits
    'top_10_mass': 0.244,  # 10/4096 = 0.244%
    'top_50_mass': 1.22,   # 50/4096
    'top_100_mass': 2.44,  # 100/4096
    'used_codes': 4096,
    'usage_pct': 100.0,
}


def load_real_data():
    """載入真實的 token 數據"""
    train_student_df = pd.read_csv(BASE_DIR / 'real_train_student_token_ranking.csv')
    val_student_df = pd.read_csv(BASE_DIR / 'real_val_student_token_ranking.csv')
    val_teacher_df = pd.read_csv(BASE_DIR / 'real_val_teacher_token_ranking.csv')

    with open(BASE_DIR / 'real_token_statistics.json', 'r') as f:
        stats = json.load(f)

    return train_student_df, val_student_df, val_teacher_df, stats


def generate_visualizations(train_student_df, val_student_df, val_teacher_df, stats):
    """生成所有可視化圖表（Train & Val 對比）"""

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    train_stats = stats['train_student']
    val_stats = stats['validation_student']
    teacher_stats = stats['validation_teacher']

    # ==================== Figure 1: Top-20 Token Comparison (Train vs Val) ====================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Student Train Top-20
    ax = axes[0]
    top20_train = train_student_df.head(20)
    x = np.arange(20)
    colors_train = plt.cm.Reds(np.linspace(0.5, 0.95, 20))

    bars = ax.bar(x, top20_train['frequency'], color=colors_train,
                  edgecolor='black', linewidth=1.5, alpha=0.9, label='Train')
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('TRAIN Set: Top-20 Most Frequent Tokens\n(REAL DATA - Severe Collapse)',
                 fontsize=13, fontweight='bold', color='darkred')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"#{i+1}" for i in x[::2]], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add token IDs
    for i in range(min(10, len(top20_train))):
        bar = bars[i]
        height = bar.get_height()
        token_id = int(top20_train.iloc[i]['token_id'])
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'T{token_id}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.text(0.98, 0.95, f'Top-10 Mass: {train_stats["top_10_mass"]:.1f}%\nEntropy: {train_stats["entropy"]:.2f} bits',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Student Val Top-20
    ax = axes[1]
    top20_val = val_student_df.head(20)
    colors_val = plt.cm.Blues(np.linspace(0.5, 0.95, 20))

    bars = ax.bar(x, top20_val['frequency'], color=colors_val,
                  edgecolor='black', linewidth=1.5, alpha=0.9, label='Val')
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('VAL Set: Top-20 Most Frequent Tokens\n(REAL DATA - Moderate Collapse)',
                 fontsize=13, fontweight='bold', color='darkblue')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"#{i+1}" for i in x[::2]], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add token IDs
    for i in range(min(10, len(top20_val))):
        bar = bars[i]
        height = bar.get_height()
        token_id = int(top20_val.iloc[i]['token_id'])
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'T{token_id}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.text(0.98, 0.95, f'Top-10 Mass: {val_stats["top_10_mass"]:.1f}%\nEntropy: {val_stats["entropy"]:.2f} bits',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_train_val_top20_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_train_val_top20_comparison.png")

    # ==================== Figure 2: Frequency Distribution (Train & Val on same plot) ====================
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Train distribution
    ax.loglog(train_student_df['rank'], train_student_df['frequency'], 'o-',
             color='#e74c3c', alpha=0.75, markersize=3, linewidth=2, label='Train Student')

    # Val distribution
    ax.loglog(val_student_df['rank'], val_student_df['frequency'], 's-',
             color='#3498db', alpha=0.75, markersize=3, linewidth=2, label='Val Student')

    # Teacher distribution
    ax.loglog(val_teacher_df['rank'], val_teacher_df['frequency'], '^-',
             color='#2ecc71', alpha=0.75, markersize=3, linewidth=2, label='Val Teacher (Reference)')

    # Ideal uniform
    ideal_freq = 100.0 / CODEBOOK_SIZE
    ax.axhline(y=ideal_freq, color='gold', linestyle='--', linewidth=2.5,
              alpha=0.7, label=f'Ideal Uniform ({ideal_freq:.4f}%)')

    ax.set_xlabel('Token Rank (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency % (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Token Frequency Distribution: Train vs Val vs Teacher (REAL DATA)\nLog-Log Plot',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_train_val_frequency_loglog.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_train_val_frequency_loglog.png")

    # ==================== Figure 3: Cumulative Distribution (Train & Val) ====================
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(train_student_df['rank'], train_student_df['cumulative_freq'],
           '-', color='#e74c3c', linewidth=3, alpha=0.8, label='Train Student')
    ax.plot(val_student_df['rank'], val_student_df['cumulative_freq'],
           '-', color='#3498db', linewidth=3, alpha=0.8, label='Val Student')
    ax.plot(val_teacher_df['rank'], val_teacher_df['cumulative_freq'],
           '-', color='#2ecc71', linewidth=3, alpha=0.8, label='Val Teacher (Reference)')

    # Ideal uniform
    ideal_cumulative = np.linspace(0, 100, CODEBOOK_SIZE)
    ax.plot(range(1, CODEBOOK_SIZE + 1), ideal_cumulative,
           '--', color='gold', linewidth=2.5, alpha=0.7, label='Ideal Uniform')

    # Mark key points
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax.axvline(x=50, color='orange', linestyle=':', alpha=0.5, linewidth=2)
    ax.axvline(x=100, color='gold', linestyle=':', alpha=0.5, linewidth=2)

    ax.text(10, 5, 'Top-10', rotation=90, verticalalignment='bottom', fontsize=10, fontweight='bold')
    ax.text(50, 5, 'Top-50', rotation=90, verticalalignment='bottom', fontsize=10, fontweight='bold')
    ax.text(100, 5, 'Top-100', rotation=90, verticalalignment='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Token Rank', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Frequency (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Token Distribution: Train vs Val vs Teacher (REAL DATA)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_train_val_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_train_val_cumulative.png")

    # ==================== Figure 4: Metrics Comparison (Train vs Val vs Target) ====================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    metrics_names = ['Entropy (bits)', 'Top-10 Mass (%)', 'Used Codes', 'Codebook Usage (%)']

    train_vals = [
        train_stats['entropy'],
        train_stats['top_10_mass'],
        train_stats['used_codes'],
        train_stats['usage_pct']
    ]

    val_vals = [
        val_stats['entropy'],
        val_stats['top_10_mass'],
        val_stats['used_codes'],
        val_stats['usage_pct']
    ]

    teacher_vals = [
        teacher_stats['entropy'],
        teacher_stats['top_10_mass'],
        teacher_stats['used_codes'],
        teacher_stats['usage_pct']
    ]

    target_vals = [
        TEACHER_TARGET['entropy'],
        TEACHER_TARGET['top_10_mass'],
        TEACHER_TARGET['used_codes'],
        TEACHER_TARGET['usage_pct']
    ]

    for idx, (ax, metric_name) in enumerate(zip(axes.flatten(), metrics_names)):
        x = np.arange(4)
        values = [train_vals[idx], val_vals[idx], teacher_vals[idx], target_vals[idx]]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        labels = ['Train\nStudent', 'Val\nStudent', 'Val\nTeacher', 'Ideal\nTarget']

        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison (REAL DATA)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if val < 100 or idx == 0 else f'{int(val)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_train_val_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_train_val_metrics_comparison.png")

    # ==================== Figure 5: Side-by-side Bar Chart (Key Metrics) ====================
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    metrics = ['Entropy\n(bits)', 'Top-10 Mass\n(%)', 'Top-50 Mass\n(%)', 'Top-100 Mass\n(%)', 'Used Codes', 'Usage\n(%)']
    train_data = [
        train_stats['entropy'],
        train_stats['top_10_mass'],
        train_stats['top_50_mass'],
        train_stats['top_100_mass'],
        train_stats['used_codes'] / 10,  # Scale down for visibility
        train_stats['usage_pct']
    ]
    val_data = [
        val_stats['entropy'],
        val_stats['top_10_mass'],
        val_stats['top_50_mass'],
        val_stats['top_100_mass'],
        val_stats['used_codes'] / 10,  # Scale down
        val_stats['usage_pct']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_data, width, label='Train Student',
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, val_data, width, label='Val Student',
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title('Train vs Val: Key Token Distribution Metrics (REAL DATA)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # Special handling for Used Codes (scaled)
            if i == 4:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height * 10)}',  # Unscale for display
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add note about Used Codes scaling
    ax.text(0.02, 0.98, 'Note: "Used Codes" values divided by 10 for visualization',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_train_val_sidebyside.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_train_val_sidebyside.png")


def main():
    print("=" * 80)
    print("Generating Visualizations from REAL Token Data (Train & Val Combined)")
    print("=" * 80)

    print("\n[1/2] Loading real token data...")
    train_student_df, val_student_df, val_teacher_df, stats = load_real_data()
    print(f"  ✓ Loaded {len(train_student_df)} train student tokens")
    print(f"  ✓ Loaded {len(val_student_df)} val student tokens")
    print(f"  ✓ Loaded {len(val_teacher_df)} val teacher tokens")

    print("\n[2/2] Generating combined visualizations...")
    generate_visualizations(train_student_df, val_student_df, val_teacher_df, stats)

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    print("\n📊 Generated files:")
    print("  1. REAL_train_val_top20_comparison.png - Top-20 tokens (Train vs Val)")
    print("  2. REAL_train_val_frequency_loglog.png - Frequency distribution (all on one plot)")
    print("  3. REAL_train_val_cumulative.png - Cumulative distribution (all on one plot)")
    print("  4. REAL_train_val_metrics_comparison.png - Metrics comparison (Train/Val/Teacher/Target)")
    print("  5. REAL_train_val_sidebyside.png - Side-by-side bar chart (Train vs Val)")


if __name__ == '__main__':
    main()
