"""
比較 Cosine, L1, L2 Loss 在中間層監督的適用性
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def compare_loss_types():
    """
    用實際數據比較不同 Loss 類型
    """
    # 載入噪音敏感度數據
    with open('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis/noise_sensitivity.json') as f:
        data = json.load(f)

    layers = list(range(16))

    # 各層的 MSE (可視為 L2 的代理)
    mse_values = [data['mse'][str(l)] for l in layers]
    cos_sim = [data['cos_sim'][str(l)] for l in layers]
    cos_loss = [1 - c for c in cos_sim]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # =============================================
    # 1. MSE (L2) 在不同層的分布
    # =============================================
    ax1 = axes[0, 0]
    ax1.bar(layers, mse_values, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('MSE Value')
    ax1.set_title('L2 (MSE) Loss Distribution\nAcross Layers')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)

    # 標註尺度差異
    ax1.annotate(f'Max: L6={mse_values[6]:.0f}', xy=(6, mse_values[6]),
                xytext=(8, mse_values[6]), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate(f'Min: L15={mse_values[15]:.2f}', xy=(15, mse_values[15]),
                xytext=(12, mse_values[15]*10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green'))

    # =============================================
    # 2. Cosine Loss 在不同層的分布
    # =============================================
    ax2 = axes[0, 1]
    colors = ['#d62728' if c > 0.8 else '#ff7f0e' if c > 0.5 else '#2ca02c' for c in cos_loss]
    ax2.bar(layers, cos_loss, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Loss (1 - cos_sim)')
    ax2.set_title('Cosine Loss Distribution\nAcross Layers')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    # =============================================
    # 3. 正規化後的比較
    # =============================================
    ax3 = axes[0, 2]

    # 正規化到 0-1 範圍
    mse_norm = np.array(mse_values) / max(mse_values)
    cos_norm = np.array(cos_loss)  # 已經在 0-1

    x = np.arange(16)
    width = 0.35

    ax3.bar(x - width/2, mse_norm, width, label='L2 (normalized)', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, cos_norm, width, label='Cosine', color='coral', alpha=0.7)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Normalized Loss')
    ax3.set_title('Normalized Comparison\n(Both scaled to 0-1)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # =============================================
    # 4. 各 Loss 的特性比較表
    # =============================================
    ax4 = axes[1, 0]
    ax4.axis('off')

    comparison_text = """
    Loss Type Comparison
    ════════════════════════════════════════════════════════

    ┌─────────────┬────────────────────────────────────────┐
    │ Loss Type   │ Characteristics                        │
    ├─────────────┼────────────────────────────────────────┤
    │ L2 (MSE)    │ • Measures Euclidean distance          │
    │             │ • Sensitive to magnitude               │
    │             │ • Penalizes large errors heavily       │
    │             │ • Range: [0, ∞)                        │
    │             │ • Problem: Scale varies 10000x         │
    ├─────────────┼────────────────────────────────────────┤
    │ L1 (MAE)    │ • Measures Manhattan distance          │
    │             │ • Less sensitive to outliers           │
    │             │ • Linear penalty                       │
    │             │ • Range: [0, ∞)                        │
    │             │ • Problem: Scale still varies          │
    ├─────────────┼────────────────────────────────────────┤
    │ Cosine      │ • Measures directional difference      │
    │             │ • Ignores magnitude completely         │
    │             │ • Range: [0, 2] (typically [0, 1])     │
    │             │ • Advantage: Scale-invariant           │
    │             │ • Best for cross-layer comparison      │
    └─────────────┴────────────────────────────────────────┘
    """
    ax4.text(0.02, 0.98, comparison_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # =============================================
    # 5. 推薦場景
    # =============================================
    ax5 = axes[1, 1]
    ax5.axis('off')

    recommendation_text = """
    Recommended Use Cases
    ════════════════════════════════════════════════════════

    Cosine Loss (RECOMMENDED for intermediate layers):
    ──────────────────────────────────────────────────
    ✓ When features have different scales across layers
    ✓ When you care about "semantic direction"
    ✓ When comparing multiple layers fairly
    ✓ Current experiment: L3, L6 supervision

    L2 (MSE) Loss:
    ──────────────────────────────────────────────────
    ✓ Final layer supervision (same scale)
    ✓ When exact value matching is needed
    ✓ L10 anchor (already similar, small MSE OK)
    ✓ Feature reconstruction loss

    L1 (MAE) Loss:
    ──────────────────────────────────────────────────
    △ Robust to outliers
    △ Could be used if there are extreme values
    △ Less common in feature matching
    △ Alternative to L2 but same scale problem

    Normalized MSE (Alternative):
    ──────────────────────────────────────────────────
    △ L2-normalize features first, then MSE
    △ Combines L2 with scale invariance
    △ Range: [0, 4] (for normalized vectors)
    △ Implemented in models_v2.py
    """
    ax5.text(0.02, 0.98, recommendation_text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # =============================================
    # 6. 結論
    # =============================================
    ax6 = axes[1, 2]
    ax6.axis('off')

    conclusion_text = """
    Conclusion
    ════════════════════════════════════════════════════════

    Q: Is Cosine the best?
    A: For intermediate layer supervision, YES.

    Reasons:
    ┌────────────────────────────────────────────────────┐
    │ 1. Scale Invariance                                │
    │    • L6 MSE = 3185, L15 MSE = 0.07                │
    │    • 45000x difference!                           │
    │    • Cosine: all in [0, 1]                        │
    ├────────────────────────────────────────────────────┤
    │ 2. Fair Cross-Layer Comparison                    │
    │    • Can weight layers equally                    │
    │    • L1, L2 would be dominated by high-scale      │
    ├────────────────────────────────────────────────────┤
    │ 3. Semantic Meaning                               │
    │    • Direction = "what the feature represents"    │
    │    • Magnitude = "how strong the activation"      │
    │    • We care more about direction                 │
    └────────────────────────────────────────────────────┘

    Final Recommendation:
    ┌────────────────────────────────────────────────────┐
    │ • Intermediate layers: Cosine Loss                │
    │ • Final layer: MSE (same scale)                   │
    │ • L10 anchor: MSE or Normalized MSE               │
    └────────────────────────────────────────────────────┘
    """
    ax6.text(0.02, 0.98, conclusion_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis/loss_type_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_type_comparison.png")

    # 打印數據比較
    print("\n" + "="*70)
    print("Loss Type Comparison - Actual Data")
    print("="*70)
    print(f"\n{'Layer':>6} | {'MSE (L2)':>12} | {'Cosine Loss':>12} | {'MSE Norm':>10} | Scale Factor")
    print("-" * 70)
    for i in range(16):
        scale = mse_values[i] / mse_values[15] if mse_values[15] > 0 else 0
        print(f"L{i:>5} | {mse_values[i]:>12.2f} | {cos_loss[i]:>12.4f} | {mse_norm[i]:>10.4f} | {scale:>10.1f}x")

    print("\n" + "="*70)
    print(f"MSE Range: {min(mse_values):.2f} ~ {max(mse_values):.2f} (ratio: {max(mse_values)/min(mse_values):.0f}x)")
    print(f"Cosine Range: {min(cos_loss):.4f} ~ {max(cos_loss):.4f} (ratio: {max(cos_loss)/min(cos_loss):.1f}x)")
    print("="*70)


if __name__ == '__main__':
    compare_loss_types()
