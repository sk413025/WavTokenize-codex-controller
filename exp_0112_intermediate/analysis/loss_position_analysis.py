"""
分析訓練前後的變化，以及 Loss 位置配置策略
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def load_data():
    """載入分析數據"""
    analysis_dir = Path('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/analysis')

    # 噪音敏感度 (原始模型, Clean vs Noisy)
    with open(analysis_dir / 'noise_sensitivity.json') as f:
        noise_data = json.load(f)

    # 訓練後距離 (Student vs Teacher)
    with open(analysis_dir / 'layer_distances.json') as f:
        trained_data = json.load(f)

    return noise_data, trained_data


def analyze_before_after():
    """
    比較訓練前後的變化
    """
    noise_data, trained_data = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    layers = list(range(16))

    # =============================================
    # 1. 訓練前: 原始模型的噪音敏感度
    # =============================================
    ax1 = axes[0, 0]

    # Clean vs Noisy (原始模型)
    before_cos = [noise_data['cos_sim'][str(l)] for l in layers]
    before_sensitivity = [1 - c for c in before_cos]

    colors1 = ['#d62728' if s > 0.8 else '#ff7f0e' if s > 0.5 else '#2ca02c' for s in before_sensitivity]
    ax1.bar(layers, before_sensitivity, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Noise Sensitivity (1 - CosSim)')
    ax1.set_title('BEFORE Training: Original Model\nClean vs Noisy Feature Difference\n(Higher = More sensitive to noise)')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # 標記關鍵層
    for i, s in enumerate(before_sensitivity):
        if s > 0.9:
            ax1.annotate(f'L{i}', xy=(i, s), xytext=(i, s+0.05), ha='center', fontsize=8, color='red')

    # =============================================
    # 2. 訓練後: Student vs Teacher 距離
    # =============================================
    ax2 = axes[0, 1]

    # Student vs Teacher (訓練後)
    after_cos = [trained_data['cos_sim'][str(l)] for l in layers]
    after_loss = [1 - c for c in after_cos]

    colors2 = ['#d62728' if l > 0.8 else '#ff7f0e' if l > 0.5 else '#2ca02c' for l in after_loss]
    ax2.bar(layers, after_loss, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Loss (1 - CosSim)')
    ax2.set_title('AFTER Training: Student vs Teacher\nHow well did LoRA learn?\n(Lower = Better learned)')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    # 標記當前監督位置
    supervised = [3, 6]
    for sl in supervised:
        ax2.scatter([sl], [after_loss[sl]], s=200, c='blue', marker='*',
                   edgecolors='black', linewidth=1, zorder=5)
        ax2.annotate(f'Supervised\nL{sl}', xy=(sl, after_loss[sl]),
                    xytext=(sl+1, after_loss[sl]+0.1),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=8, color='blue')

    # =============================================
    # 3. 變化量 (訓練前 - 訓練後)
    # =============================================
    ax3 = axes[1, 0]

    # 這裡的「改善」定義需要仔細考慮:
    # - 訓練前: sensitivity = 1 - cos(clean, noisy) -- 噪音造成的差異
    # - 訓練後: loss = 1 - cos(student, teacher) -- 學習後的差距
    #
    # 理想情況: 訓練後 student 應該接近 teacher
    # 所以 after_loss 越小越好

    # 但這兩者不能直接比較，因為：
    # - before: 同一模型，不同輸入
    # - after: 不同模型，不同輸入

    # 我們改為顯示「學習難度」: 噪音敏感度高，但訓練後仍有差距的層
    learning_difficulty = [before_sensitivity[i] * after_loss[i] for i in range(16)]

    colors3 = plt.cm.RdYlGn_r(np.array(learning_difficulty) / max(learning_difficulty))
    bars3 = ax3.bar(layers, learning_difficulty, color=colors3, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Learning Difficulty\n(Sensitivity × Remaining Gap)')
    ax3.set_title('Learning Difficulty Score\n(Higher = Needs more supervision)')
    ax3.grid(axis='y', alpha=0.3)

    # 標記最需要關注的層
    sorted_indices = np.argsort(learning_difficulty)[::-1]
    for rank, idx in enumerate(sorted_indices[:4]):
        ax3.annotate(f'#{rank+1}', xy=(idx, learning_difficulty[idx]),
                    xytext=(idx, learning_difficulty[idx] + 0.02),
                    ha='center', fontsize=10, fontweight='bold', color='red')

    # =============================================
    # 4. Loss 配置建議
    # =============================================
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 計算建議
    top_sensitive = sorted(range(16), key=lambda i: before_sensitivity[i], reverse=True)[:5]
    top_difficult = sorted(range(16), key=lambda i: learning_difficulty[i], reverse=True)[:5]
    most_stable = [i for i in range(16) if before_sensitivity[i] < 0.2]

    analysis_text = f"""
    Loss 配置分析與建議
    {'='*50}

    1. 噪音最敏感的層 (sensitivity > 0.9):
       {[f'L{i}' for i in top_sensitive if before_sensitivity[i] > 0.9]}
       → 這些層受噪音影響最大，需要監督

    2. 學習最困難的層 (difficulty score top 4):
       {[f'L{i}' for i in top_difficult[:4]]}
       → 這些層「敏感且學習不好」，最需要加強

    3. 最穩定的層 (sensitivity < 0.2):
       {[f'L{i}' for i in most_stable]}
       → 這些層天然魯棒，可以作為錨點

    4. 當前監督位置:
       L3, L6
       → L3 在敏感層中，L6 是中層最敏感
       → 但 L0, L1 更敏感卻沒監督

    {'─'*50}
    建議的 Loss 配置:
    {'─'*50}

    方案 A: 全面覆蓋
    ┌─────────────────────────────────────────┐
    │ 淺層 (L1, L3):  Cosine Loss, weight=1.0 │
    │ 中層 (L6):      Cosine Loss, weight=0.8 │
    │ 錨點 (L10):     MSE Loss, weight=0.5    │
    │ 最後 (L17):     MSE + Triplet           │
    └─────────────────────────────────────────┘

    方案 B: 漸進式
    ┌─────────────────────────────────────────┐
    │ 根據 sensitivity 動態調整權重:          │
    │ weight[i] = sensitivity[i]              │
    │ 敏感層自動獲得更高權重                  │
    └─────────────────────────────────────────┘

    {'─'*50}
    為什麼深層也要訓練？
    {'─'*50}

    雖然深層對噪音穩定，但：
    1. 淺層 LoRA 改變了輸入給深層的 feature
    2. 深層需要「適應」新的淺層輸出
    3. 去噪是「協同任務」，需要全層配合

    但深層不需要「強監督」，因為：
    1. L10 已經很穩定 (cos_sim=0.946)
    2. 用輕量的 MSE 確保不偏離即可
    """

    ax4.text(0.02, 0.98, analysis_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/analysis/loss_position_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_position_analysis.png")

    # 打印詳細分析
    print("\n" + "="*60)
    print("Loss Position Analysis")
    print("="*60)

    print("\n各層分析:")
    print(f"{'Layer':>6} | {'Sensitivity':>11} | {'After Loss':>10} | {'Difficulty':>10} | Note")
    print("-" * 60)
    for i in range(16):
        note = ""
        if i in supervised:
            note = "★ Currently supervised"
        elif before_sensitivity[i] > 0.9:
            note = "⚠ Very sensitive, needs supervision"
        elif before_sensitivity[i] < 0.2:
            note = "✓ Stable (anchor candidate)"

        print(f"L{i:>5} | {before_sensitivity[i]:>11.4f} | {after_loss[i]:>10.4f} | {learning_difficulty[i]:>10.4f} | {note}")

    print("\n" + "="*60)


if __name__ == '__main__':
    analyze_before_after()
