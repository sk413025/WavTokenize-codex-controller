"""
使用真實的 token 數據生成可視化圖表
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
    train_teacher_df = pd.read_csv(BASE_DIR / 'real_train_teacher_token_ranking.csv')
    val_student_df = pd.read_csv(BASE_DIR / 'real_val_student_token_ranking.csv')
    val_teacher_df = pd.read_csv(BASE_DIR / 'real_val_teacher_token_ranking.csv')

    with open(BASE_DIR / 'real_token_statistics.json', 'r') as f:
        stats = json.load(f)

    return train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats


def generate_visualizations(train_student_df, train_teacher_df, val_student_df, val_teacher_df, stats):
    """生成所有可視化圖表"""

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    train_student_stats = stats['train_student']
    val_student_stats = stats['validation_student']
    teacher_stats = stats['validation_teacher']

    # ==================== Figure 1: Train vs Val Student Comparison ====================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Train Top-20
    ax = axes[0]
    top20_train = train_student_df.head(20)
    x = np.arange(20)
    colors_student = plt.cm.Reds(np.linspace(0.5, 0.95, 20))

    bars = ax.bar(x, top20_student['frequency'], color=colors_student,
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Student (Baseline): Top-20 Most Frequent Tokens\n(REAL DATA - Severe Collapse)',
                 fontsize=13, fontweight='bold', color='darkred')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"#{i+1}" for i in x[::2]], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add token IDs on top 10 bars
    for i in range(min(10, len(top20_student))):
        bar = bars[i]
        height = bar.get_height()
        token_id = int(top20_student.iloc[i]['token_id'])
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'T{token_id}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add collapse annotation
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='1% threshold')
    ax.text(0.98, 0.95, f'Top-10 Mass: {student_stats["top_10_mass"]:.1f}%\nEntropy: {student_stats["entropy"]:.2f} bits',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Teacher Top-20
    ax = axes[1]
    top20_teacher = teacher_df.head(20)
    colors_teacher = plt.cm.Blues(np.linspace(0.5, 0.95, 20))

    bars = ax.bar(x, top20_teacher['frequency'], color=colors_teacher,
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Teacher (Reference): Top-20 Most Frequent Tokens\n(REAL DATA - Better Distribution)',
                 fontsize=13, fontweight='bold', color='darkblue')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"#{i+1}" for i in x[::2]], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add token IDs on top 10 bars
    for i in range(min(10, len(top20_teacher))):
        bar = bars[i]
        height = bar.get_height()
        token_id = int(top20_teacher.iloc[i]['token_id'])
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'T{token_id}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.text(0.98, 0.95, f'Top-10 Mass: {teacher_stats["top_10_mass"]:.1f}%\nEntropy: {teacher_stats["entropy"]:.2f} bits',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_top20_token_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_top20_token_comparison.png")

    # ==================== Figure 2: Frequency Distribution (Log-Log) ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Student distribution
    ax = axes[0]
    ax.loglog(student_df['rank'], student_df['frequency'], 'o-',
             color='#e74c3c', alpha=0.7, markersize=3, linewidth=1.5, label='Student (Baseline)')

    # Add ideal uniform distribution for reference
    ideal_freq = 100.0 / CODEBOOK_SIZE  # uniform: each token gets equal share
    ax.axhline(y=ideal_freq, color='green', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Ideal Uniform ({ideal_freq:.4f}%)')

    ax.set_xlabel('Token Rank (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency % (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Student Token Frequency Distribution\n(REAL DATA - Power Law Collapse)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Teacher distribution
    ax = axes[1]
    ax.loglog(teacher_df['rank'], teacher_df['frequency'], 'o-',
             color='#3498db', alpha=0.7, markersize=3, linewidth=1.5, label='Teacher (Reference)')
    ax.axhline(y=ideal_freq, color='green', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Ideal Uniform ({ideal_freq:.4f}%)')

    ax.set_xlabel('Token Rank (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency % (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Token Frequency Distribution\n(REAL DATA - More Uniform)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_frequency_distribution_loglog.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_frequency_distribution_loglog.png")

    # ==================== Figure 3: Cumulative Distribution ====================
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.plot(student_df['rank'], student_df['cumulative_freq'],
           '-', color='#e74c3c', linewidth=2.5, alpha=0.8, label='Student (Baseline)')
    ax.plot(teacher_df['rank'], teacher_df['cumulative_freq'],
           '-', color='#3498db', linewidth=2.5, alpha=0.8, label='Teacher (Reference)')

    # Add ideal uniform cumulative
    ideal_cumulative = np.linspace(0, 100, CODEBOOK_SIZE)
    ax.plot(range(1, CODEBOOK_SIZE + 1), ideal_cumulative,
           '--', color='green', linewidth=2, alpha=0.6, label='Ideal Uniform')

    # Mark key points
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=100, color='gold', linestyle=':', alpha=0.5)

    ax.text(10, 5, 'Top-10', rotation=90, verticalalignment='bottom', fontsize=9)
    ax.text(50, 5, 'Top-50', rotation=90, verticalalignment='bottom', fontsize=9)
    ax.text(100, 5, 'Top-100', rotation=90, verticalalignment='bottom', fontsize=9)

    ax.set_xlabel('Token Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Token Distribution (REAL DATA)\nStudent vs Teacher vs Ideal',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500)  # Focus on first 500 tokens for clarity

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_cumulative_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_cumulative_distribution.png")

    # ==================== Figure 4: Metrics Comparison ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_names = ['Entropy (bits)', 'Top-10 Mass (%)', 'Used Codes', 'Codebook Usage (%)']

    student_vals = [
        student_stats['entropy'],
        student_stats['top_10_mass'],
        student_stats['used_codes'],
        student_stats['usage_pct']
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
        x = np.arange(3)
        values = [student_vals[idx], teacher_vals[idx], target_vals[idx]]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        labels = ['Student\n(Baseline)', 'Teacher\n(Reference)', 'Ideal\n(Target)']

        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison (REAL DATA)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if val < 100 or idx == 0 else f'{int(val)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_metrics_comparison.png")

    # ==================== Figure 5: Gini Coefficient / Concentration ====================
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Calculate Lorenz curves (cumulative frequency vs cumulative token proportion)
    student_lorenz_x = np.arange(1, len(student_df) + 1) / len(student_df) * 100
    student_lorenz_y = student_df['cumulative_freq'].values

    teacher_lorenz_x = np.arange(1, len(teacher_df) + 1) / len(teacher_df) * 100
    teacher_lorenz_y = teacher_df['cumulative_freq'].values

    # Perfect equality line
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.7, label='Perfect Equality')

    # Student Lorenz curve
    ax.plot(student_lorenz_x, student_lorenz_y,
           color='#e74c3c', linewidth=2.5, alpha=0.8, label='Student (Baseline)')

    # Teacher Lorenz curve
    ax.plot(teacher_lorenz_x, teacher_lorenz_y,
           color='#3498db', linewidth=2.5, alpha=0.8, label='Teacher (Reference)')

    ax.fill_between(student_lorenz_x, student_lorenz_y, student_lorenz_x,
                    alpha=0.2, color='red', label='Student Inequality Area')

    ax.set_xlabel('Cumulative % of Tokens (ranked by frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Lorenz Curve: Token Usage Inequality (REAL DATA)\nFurther from diagonal = More collapse',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'REAL_lorenz_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: REAL_lorenz_curve.png")


def main():
    print("=" * 80)
    print("Generating Visualizations from REAL Token Data")
    print("=" * 80)

    print("\n[1/2] Loading real token data...")
    student_df, teacher_df, stats = load_real_data()
    print(f"  ✓ Loaded {len(student_df)} student tokens")
    print(f"  ✓ Loaded {len(teacher_df)} teacher tokens")

    print("\n[2/2] Generating visualizations...")
    generate_visualizations(student_df, teacher_df, stats)

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    print("\n📊 Generated files:")
    print("  1. REAL_top20_token_comparison.png - Top-20 token bar charts")
    print("  2. REAL_frequency_distribution_loglog.png - Log-log frequency distribution")
    print("  3. REAL_cumulative_distribution.png - Cumulative distribution curves")
    print("  4. REAL_metrics_comparison.png - Key metrics comparison")
    print("  5. REAL_lorenz_curve.png - Lorenz curve (inequality visualization)")


if __name__ == '__main__':
    main()
