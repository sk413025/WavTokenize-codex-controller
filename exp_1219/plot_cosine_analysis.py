#!/usr/bin/env python3
"""
Cosine Weight 分析視覺化

比較:
- Exp50: triplet_margin=0.5, cosine_weight=0.0
- Exp51_v2: triplet_margin=0.5, cosine_weight=0.1

目的: 驗證 cosine loss 是否有幫助
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_history(exp_path):
    with open(exp_path / 'history.json', 'r') as f:
        return json.load(f)

def main():
    base_dir = Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1219/runs')

    # 載入數據
    exp50 = load_history(base_dir / 'exp50_margin')
    exp51_v2 = load_history(base_dir / 'exp51_combined_v2')

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cosine Weight Analysis: 0.0 vs 0.1\n(Both with triplet_margin=0.5)',
                 fontsize=14, fontweight='bold')

    epochs_50 = range(1, len(exp50['val_masked_acc']) + 1)
    epochs_51 = range(1, len(exp51_v2['val_masked_acc']) + 1)

    # 1. Val Accuracy 比較
    ax1 = axes[0, 0]
    ax1.plot(epochs_50, [x * 100 for x in exp50['val_masked_acc']],
             'b-', label='Exp50 (cosine=0.0)', linewidth=2)
    ax1.plot(epochs_51, [x * 100 for x in exp51_v2['val_masked_acc']],
             'r-', label='Exp51_v2 (cosine=0.1)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Masked Accuracy (%)')
    ax1.set_title('Validation Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 標註最佳值
    best_50 = max(exp50['val_masked_acc']) * 100
    best_51 = max(exp51_v2['val_masked_acc']) * 100
    ax1.axhline(y=best_50, color='b', linestyle='--', alpha=0.5)
    ax1.axhline(y=best_51, color='r', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {best_50:.2f}%', xy=(len(epochs_50), best_50),
                 xytext=(len(epochs_50)-15, best_50+0.02), color='b')
    ax1.annotate(f'Best: {best_51:.2f}%', xy=(len(epochs_51), best_51),
                 xytext=(len(epochs_51)-15, best_51-0.05), color='r')

    # 2. Val Loss 比較
    ax2 = axes[0, 1]
    ax2.plot(epochs_50, exp50['val_loss'], 'b-', label='Exp50 (cosine=0.0)', linewidth=2)
    ax2.plot(epochs_51, exp51_v2['val_loss'], 'r-', label='Exp51_v2 (cosine=0.1)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cosine Similarity (只有 Exp51_v2 有這個指標)
    ax3 = axes[1, 0]
    if 'val_cos_sim' in exp51_v2 and any(x > 0 for x in exp51_v2['val_cos_sim']):
        ax3.plot(epochs_51, exp51_v2['val_cos_sim'], 'r-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Val Cosine Similarity')
        ax3.set_title('Exp51_v2: Cosine Similarity Over Training')
        ax3.grid(True, alpha=0.3)

        # 添加說明
        final_cos = exp51_v2['val_cos_sim'][-1]
        ax3.axhline(y=final_cos, color='r', linestyle='--', alpha=0.5)
        ax3.annotate(f'Final: {final_cos:.3f}', xy=(len(epochs_51), final_cos),
                     xytext=(len(epochs_51)-20, final_cos+0.02), color='r')
    else:
        ax3.text(0.5, 0.5, 'Cosine similarity not tracked\nin Exp50 (cosine_weight=0)',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Cosine Similarity (N/A for Exp50)')

    # 4. Train/Val Gap (過擬合程度)
    ax4 = axes[1, 1]
    train_val_gap_50 = [t - v for t, v in zip(exp50['train_masked_acc'], exp50['val_masked_acc'])]
    train_val_gap_51 = [t - v for t, v in zip(exp51_v2['train_masked_acc'], exp51_v2['val_masked_acc'])]

    ax4.plot(epochs_50, [x * 100 for x in train_val_gap_50],
             'b-', label='Exp50 (cosine=0.0)', linewidth=2)
    ax4.plot(epochs_51, [x * 100 for x in train_val_gap_51],
             'r-', label='Exp51_v2 (cosine=0.1)', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Accuracy (%)')
    ax4.set_title('Overfitting Gap (Train - Val)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()

    # 儲存圖表
    output_path = Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1219/cosine_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # 顯示統計摘要
    print("\n" + "="*60)
    print("COSINE WEIGHT ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nExp50 (cosine_weight=0.0):")
    print(f"  Best Val Acc: {best_50:.3f}%")
    print(f"  Final Val Loss: {exp50['val_loss'][-1]:.4f}")
    print(f"  Final Train-Val Gap: {train_val_gap_50[-1]*100:.3f}%")

    print(f"\nExp51_v2 (cosine_weight=0.1):")
    print(f"  Best Val Acc: {best_51:.3f}%")
    print(f"  Final Val Loss: {exp51_v2['val_loss'][-1]:.4f}")
    print(f"  Final Train-Val Gap: {train_val_gap_51[-1]*100:.3f}%")
    if 'val_cos_sim' in exp51_v2:
        print(f"  Final Cos Sim: {exp51_v2['val_cos_sim'][-1]:.3f}")

    print(f"\nConclusion:")
    if best_50 > best_51:
        print(f"  ❌ Cosine loss HURTS accuracy by {best_50 - best_51:.3f}%")
        print(f"  ✅ Recommendation: Do NOT use cosine_weight")
    else:
        print(f"  ✅ Cosine loss HELPS accuracy by {best_51 - best_50:.3f}%")
    print("="*60)

if __name__ == '__main__':
    main()
