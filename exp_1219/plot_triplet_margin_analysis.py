#!/usr/bin/env python3
"""
Triplet Margin 與 Cosine Weight 完整分析

比較:
- Exp48: triplet_margin=0.2, cosine_weight=0.0 (baseline)
- Exp50: triplet_margin=0.5, cosine_weight=0.0
- Exp51_v2: triplet_margin=0.5, cosine_weight=0.1
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_history(exp_path):
    with open(exp_path / 'history.json', 'r') as f:
        return json.load(f)

def main():
    # 載入數據
    exp48 = load_history(Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217/runs/exp48_best_config'))
    exp50 = load_history(Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1219/runs/exp50_margin'))
    exp51_v2 = load_history(Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1219/runs/exp51_combined_v2'))

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Triplet Margin & Cosine Weight Analysis\nComparing Exp48, Exp50, Exp51_v2',
                 fontsize=14, fontweight='bold')

    epochs_48 = range(1, len(exp48['val_masked_acc']) + 1)
    epochs_50 = range(1, len(exp50['val_masked_acc']) + 1)
    epochs_51 = range(1, len(exp51_v2['val_masked_acc']) + 1)

    # 1. Val Accuracy 比較
    ax1 = axes[0, 0]
    ax1.plot(epochs_48, [x * 100 for x in exp48['val_masked_acc']],
             'g-', label='Exp48 (margin=0.2, cos=0.0)', linewidth=2)
    ax1.plot(epochs_50, [x * 100 for x in exp50['val_masked_acc']],
             'b-', label='Exp50 (margin=0.5, cos=0.0)', linewidth=2)
    ax1.plot(epochs_51, [x * 100 for x in exp51_v2['val_masked_acc']],
             'r-', label='Exp51_v2 (margin=0.5, cos=0.1)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Masked Accuracy (%)')
    ax1.set_title('Validation Accuracy Comparison')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 標註最佳值
    best_48 = max(exp48['val_masked_acc']) * 100
    best_50 = max(exp50['val_masked_acc']) * 100
    best_51 = max(exp51_v2['val_masked_acc']) * 100

    # 2. Val Loss 比較
    ax2 = axes[0, 1]
    ax2.plot(epochs_48, exp48['val_loss'], 'g-', label='Exp48', linewidth=2)
    ax2.plot(epochs_50, exp50['val_loss'], 'b-', label='Exp50', linewidth=2)
    ax2.plot(epochs_51, exp51_v2['val_loss'], 'r-', label='Exp51_v2', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Triplet Loss 比較
    ax3 = axes[1, 0]
    ax3.plot(epochs_48, exp48['val_triplet_loss'], 'g-', label='Exp48 (margin=0.2)', linewidth=2)
    ax3.plot(epochs_50, exp50['val_triplet_loss'], 'b-', label='Exp50 (margin=0.5)', linewidth=2)
    ax3.plot(epochs_51, exp51_v2['val_triplet_loss'], 'r-', label='Exp51_v2 (margin=0.5)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Triplet Loss')
    ax3.set_title('Triplet Loss Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 結果摘要 (Bar Chart)
    ax4 = axes[1, 1]
    experiments = ['Exp48\n(m=0.2, c=0.0)', 'Exp50\n(m=0.5, c=0.0)', 'Exp51_v2\n(m=0.5, c=0.1)']
    best_accs = [best_48, best_50, best_51]
    colors = ['green', 'blue', 'red']

    bars = ax4.bar(experiments, best_accs, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Best Val Accuracy (%)')
    ax4.set_title('Best Validation Accuracy Summary')

    # 添加數值標籤
    for bar, acc in zip(bars, best_accs):
        ax4.annotate(f'{acc:.3f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontweight='bold')

    # 標記最佳
    best_idx = np.argmax(best_accs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()

    # 儲存圖表
    output_path = Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1219/triplet_cosine_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # 顯示統計摘要
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS SUMMARY")
    print("="*70)

    print("\n┌────────────────┬────────────┬──────────────┬────────────────┐")
    print("│ Experiment     │ Margin     │ Cosine Wt    │ Best Val Acc   │")
    print("├────────────────┼────────────┼──────────────┼────────────────┤")
    print(f"│ Exp48          │ 0.2        │ 0.0          │ {best_48:.4f}%       │")
    print(f"│ Exp50          │ 0.5        │ 0.0          │ {best_50:.4f}%       │")
    print(f"│ Exp51_v2       │ 0.5        │ 0.1          │ {best_51:.4f}%       │")
    print("└────────────────┴────────────┴──────────────┴────────────────┘")

    print("\n📊 ANALYSIS:")
    print("─" * 70)

    # Margin 分析
    print("\n1️⃣  Triplet Margin Analysis (Exp48 vs Exp50, both cosine=0.0):")
    margin_diff = best_48 - best_50
    if margin_diff > 0:
        print(f"   ✅ margin=0.2 is BETTER by {margin_diff:.4f}%")
        print(f"   📝 Smaller margin provides finer learning signal")
    else:
        print(f"   ❌ margin=0.5 is BETTER by {-margin_diff:.4f}%")

    # Cosine 分析
    print("\n2️⃣  Cosine Weight Analysis (Exp50 vs Exp51_v2, both margin=0.5):")
    cosine_diff = best_51 - best_50
    if cosine_diff > 0:
        print(f"   ⚠️  cosine=0.1 slightly BETTER by {cosine_diff:.4f}%")
        print(f"   📝 But difference is marginal (<0.01%)")
    else:
        print(f"   ❌ cosine=0.0 is BETTER by {-cosine_diff:.4f}%")

    # 最終建議
    print("\n3️⃣  FINAL RECOMMENDATION:")
    best_exp = ['Exp48', 'Exp50', 'Exp51_v2'][best_idx]
    print(f"   🏆 Best configuration: {best_exp}")
    print(f"   ├── triplet_margin: 0.2")
    print(f"   ├── cosine_weight: 0.0")
    print(f"   └── Best Val Acc: {max(best_accs):.4f}%")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
