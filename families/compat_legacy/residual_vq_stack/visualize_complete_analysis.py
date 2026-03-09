"""
完整的 Token 分布分析可視化
包含：Student Train/Val, Teacher Train/Val, Target (理想分布)
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

# 理想分布（Target）
IDEAL_FREQ = 100.0 / CODEBOOK_SIZE  # 0.0244%


def load_real_data():
    """載入真實的 token 數據"""
    train_student_df = pd.read_csv(BASE_DIR / 'real_train_student_token_ranking.csv')
    train_teacher_df = pd.read_csv(BASE_DIR / 'real_train_teacher_token_ranking.csv')
    val_student_df = pd.read_csv(BASE_DIR / 'real_val_student_token_ranking.csv')
    val_teacher_df = pd.read_csv(BASE_DIR / 'real_val_teacher_token_ranking.csv')

    with open(BASE_DIR / 'real_token_statistics.json', 'r') as f:
        stats = json.load(f)

    return train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats


def plot_top_tokens_ranking(train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats):
    """圖1: Top-20 Token 排名（所有 splits 在同一張圖）"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    datasets = [
        (train_student_df, 'Student Train', '#e74c3c', stats['train_student'], axes[0, 0]),
        (val_student_df, 'Student Val', '#3498db', stats['validation_student'], axes[0, 1]),
        (train_teacher_df, 'Teacher Train', '#2ecc71', stats['train_teacher'], axes[1, 0]),
        (val_teacher_df, 'Teacher Val', '#f39c12', stats['validation_teacher'], axes[1, 1]),
    ]

    for df, title, color, stat, ax in datasets:
        top20 = df.head(20)
        x = np.arange(20)

        bars = ax.bar(x, top20['frequency'], color=color, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Token Rank', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}: Top-20 Token Frequency\n(REAL DATA)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"#{i+1}" for i in x[::2]])
        ax.grid(True, alpha=0.3, axis='y')

        # Add token IDs on bars
        for i in range(min(10, len(top20))):
            bar = bars[i]
            height = bar.get_height()
            token_id = int(top20.iloc[i]['token_id'])
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'T{token_id}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Add statistics box
        textstr = f"Entropy: {stat['entropy']:.2f}\nTop-10: {stat['top_10_mass']:.1f}%\nUsed: {stat['used_codes']}"
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FINAL_top20_all_splits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: FINAL_top20_all_splits.png")


def plot_frequency_distribution(train_student_df, train_teacher_df, val_student_df, val_teacher_df):
    """圖2: 頻率分布 (Log-Log)（所有 splits 在同一張圖）"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Plot all distributions
    ax.loglog(train_student_df['rank'], train_student_df['frequency'], 'o-',
             color='#e74c3c', alpha=0.8, markersize=2.5, linewidth=2,
             label='Student Train', markevery=50)

    ax.loglog(val_student_df['rank'], val_student_df['frequency'], 's-',
             color='#3498db', alpha=0.8, markersize=2.5, linewidth=2,
             label='Student Val', markevery=50)

    ax.loglog(train_teacher_df['rank'], train_teacher_df['frequency'], '^-',
             color='#2ecc71', alpha=0.8, markersize=2.5, linewidth=2,
             label='Teacher Train', markevery=50)

    ax.loglog(val_teacher_df['rank'], val_teacher_df['frequency'], 'D-',
             color='#f39c12', alpha=0.8, markersize=2.5, linewidth=2,
             label='Teacher Val', markevery=50)

    # Ideal uniform distribution
    ax.axhline(y=IDEAL_FREQ, color='purple', linestyle='--', linewidth=3,
              alpha=0.7, label=f'Ideal Uniform ({IDEAL_FREQ:.4f}%)')

    ax.set_xlabel('Token Rank (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency % (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Token Frequency Distribution: All Splits (REAL DATA)\nLog-Log Plot',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FINAL_frequency_loglog_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: FINAL_frequency_loglog_all.png")


def plot_cumulative_distribution(train_student_df, train_teacher_df, val_student_df, val_teacher_df):
    """圖3: 累積分布（所有 splits）"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Plot cumulative distributions
    ax.plot(train_student_df['rank'], train_student_df['cumulative_freq'],
           '-', color='#e74c3c', linewidth=2.5, alpha=0.85, label='Student Train')

    ax.plot(val_student_df['rank'], val_student_df['cumulative_freq'],
           '-', color='#3498db', linewidth=2.5, alpha=0.85, label='Student Val')

    ax.plot(train_teacher_df['rank'], train_teacher_df['cumulative_freq'],
           '-', color='#2ecc71', linewidth=2.5, alpha=0.85, label='Teacher Train')

    ax.plot(val_teacher_df['rank'], val_teacher_df['cumulative_freq'],
           '-', color='#f39c12', linewidth=2.5, alpha=0.85, label='Teacher Val')

    # Ideal uniform
    ideal_cumulative = np.linspace(0, 100, CODEBOOK_SIZE)
    ax.plot(range(1, CODEBOOK_SIZE + 1), ideal_cumulative,
           '--', color='purple', linewidth=3, alpha=0.7, label='Ideal Uniform')

    # Mark key points
    for x_pos, label in [(10, 'Top-10'), (50, 'Top-50'), (100, 'Top-100')]:
        ax.axvline(x=x_pos, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(x_pos, 5, label, rotation=90, verticalalignment='bottom',
               fontsize=10, fontweight='bold')

    ax.set_xlabel('Token Rank', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Frequency (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Token Distribution: All Splits (REAL DATA)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FINAL_cumulative_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: FINAL_cumulative_all.png")


def plot_metrics_comparison(stats):
    """圖4: 指標對比（所有 splits）"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics_info = [
        ('entropy', 'Entropy (bits)', [12.0]),  # Ideal value
        ('top_10_mass', 'Top-10 Mass (%)', [0.244]),
        ('top_50_mass', 'Top-50 Mass (%)', [1.22]),
        ('top_100_mass', 'Top-100 Mass (%)', [2.44]),
        ('used_codes', 'Used Codes', [4096]),
        ('usage_pct', 'Codebook Usage (%)', [100.0]),
    ]

    for idx, (metric_key, metric_name, ideal_val) in enumerate(metrics_info):
        ax = axes[idx // 3, idx % 3]

        # Prepare data
        labels = ['Student\nTrain', 'Student\nVal', 'Teacher\nTrain', 'Teacher\nVal', 'Ideal\nTarget']
        values = [
            stats['train_student'].get(metric_key, 0),
            stats['validation_student'].get(metric_key, 0),
            stats['train_teacher'].get(metric_key, 0),
            stats['validation_teacher'].get(metric_key, 0),
            ideal_val[0]
        ]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)

        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if val < 100 or 'entropy' in metric_key or 'mass' in metric_key else f'{int(val)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FINAL_metrics_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: FINAL_metrics_comparison_all.png")


def create_top_tokens_table(train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats):
    """生成 Top-20 Token 排名表（Markdown + JSON）"""

    # Create summary dictionary
    summary = {
        'metadata': {
            'codebook_size': CODEBOOK_SIZE,
            'checkpoint': stats.get('checkpoint', ''),
            'epoch': stats.get('epoch', 0),
        },
        'student_train': {
            'statistics': stats['train_student'],
            'top20_tokens': train_student_df.head(20).to_dict('records'),
        },
        'student_val': {
            'statistics': stats['validation_student'],
            'top20_tokens': val_student_df.head(20).to_dict('records'),
        },
        'teacher_train': {
            'statistics': stats['train_teacher'],
            'top20_tokens': train_teacher_df.head(20).to_dict('records'),
        },
        'teacher_val': {
            'statistics': stats['validation_teacher'],
            'top20_tokens': val_teacher_df.head(20).to_dict('records'),
        },
        'ideal_target': {
            'entropy': 12.0,
            'top_10_mass': 0.244,
            'top_50_mass': 1.22,
            'top_100_mass': 2.44,
            'used_codes': 4096,
            'usage_pct': 100.0,
        }
    }

    # Save JSON
    with open(OUTPUT_DIR / 'FINAL_complete_token_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: FINAL_complete_token_analysis.json")

    # Create Markdown table
    md_content = "# Complete Token Distribution Analysis (REAL DATA)\n\n"
    md_content += f"**Checkpoint**: `{stats.get('checkpoint', 'N/A')}`\n"
    md_content += f"**Epoch**: {stats.get('epoch', 'N/A')}\n"
    md_content += f"**Codebook Size**: {CODEBOOK_SIZE}\n\n"

    md_content += "---\n\n"

    # Summary statistics table
    md_content += "## Summary Statistics\n\n"
    md_content += "| Split | Entropy (bits) | Top-10 Mass (%) | Top-50 Mass (%) | Used Codes | Usage (%) |\n"
    md_content += "|-------|----------------|-----------------|-----------------|------------|----------|\n"

    for split_name, split_key in [
        ('Student Train', 'train_student'),
        ('Student Val', 'validation_student'),
        ('Teacher Train', 'train_teacher'),
        ('Teacher Val', 'validation_teacher'),
    ]:
        s = stats[split_key]
        md_content += f"| {split_name} | {s['entropy']:.2f} | {s['top_10_mass']:.2f} | {s['top_50_mass']:.2f} | {s['used_codes']} | {s['usage_pct']:.2f} |\n"

    md_content += f"| **Ideal Target** | 12.00 | 0.24 | 1.22 | 4096 | 100.00 |\n\n"

    # Top-20 tokens for each split
    for split_name, df in [
        ('Student Train', train_student_df),
        ('Student Val', val_student_df),
        ('Teacher Train', train_teacher_df),
        ('Teacher Val', val_teacher_df),
    ]:
        md_content += f"## Top-20 Tokens: {split_name}\n\n"
        md_content += "| Rank | Token ID | Frequency (%) | Count | Cumulative (%) |\n"
        md_content += "|------|----------|---------------|-------|----------------|\n"

        for i in range(min(20, len(df))):
            row = df.iloc[i]
            md_content += f"| {int(row['rank'])} | **T{int(row['token_id'])}** | {row['frequency']:.2f} | {int(row['count']):,} | {row['cumulative_freq']:.2f} |\n"

        md_content += "\n"

    # Save Markdown
    with open(OUTPUT_DIR / 'FINAL_token_ranking_report.md', 'w') as f:
        f.write(md_content)
    print(f"✓ Saved: FINAL_token_ranking_report.md")


def main():
    print("=" * 80)
    print("Complete Token Distribution Analysis (REAL DATA)")
    print("Student Train/Val + Teacher Train/Val + Ideal Target")
    print("=" * 80)

    print("\n[1/6] Loading real token data...")
    train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats = load_real_data()
    print(f"  ✓ Loaded {len(train_student_df)} train student tokens")
    print(f"  ✓ Loaded {len(train_teacher_df)} train teacher tokens")
    print(f"  ✓ Loaded {len(val_student_df)} val student tokens")
    print(f"  ✓ Loaded {len(val_teacher_df)} val teacher tokens")

    print("\n[2/6] Generating Top-20 token rankings...")
    plot_top_tokens_ranking(train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats)

    print("\n[3/6] Generating frequency distribution (log-log)...")
    plot_frequency_distribution(train_student_df, train_teacher_df, val_student_df, val_teacher_df)

    print("\n[4/6] Generating cumulative distribution...")
    plot_cumulative_distribution(train_student_df, train_teacher_df, val_student_df, val_teacher_df)

    print("\n[5/6] Generating metrics comparison...")
    plot_metrics_comparison(stats)

    print("\n[6/6] Creating token ranking tables (JSON + Markdown)...")
    create_top_tokens_table(train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats)

    print("\n" + "=" * 80)
    print("All visualizations and reports generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    print("\n📊 Generated files:")
    print("  1. FINAL_top20_all_splits.png - Top-20 tokens (4 subplots)")
    print("  2. FINAL_frequency_loglog_all.png - Frequency distribution (all on one plot)")
    print("  3. FINAL_cumulative_all.png - Cumulative distribution (all on one plot)")
    print("  4. FINAL_metrics_comparison_all.png - Metrics comparison (6 metrics)")
    print("  5. FINAL_complete_token_analysis.json - Complete data in JSON format")
    print("  6. FINAL_token_ranking_report.md - Complete report in Markdown format")


if __name__ == '__main__':
    main()
