"""
比較 Cosine Loss 和 L2 Loss 的差異
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_loss_difference():
    """
    視覺化 Cosine vs L2 的差異
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # =============================================
    # 1. 2D 向量示意圖
    # =============================================
    ax1 = axes[0, 0]

    # Teacher 向量
    teacher = np.array([3, 4])  # magnitude = 5

    # 三種 Student 向量
    # Case A: 方向相同，幅度小
    student_a = np.array([1.5, 2])  # magnitude = 2.5, same direction
    # Case B: 方向相同，幅度大
    student_b = np.array([6, 8])  # magnitude = 10, same direction
    # Case C: 方向不同，幅度相同
    angle = np.pi / 6  # 30 degrees
    student_c = np.array([5 * np.cos(np.arctan(4/3) + angle),
                          5 * np.sin(np.arctan(4/3) + angle)])

    # 畫向量
    ax1.arrow(0, 0, teacher[0], teacher[1], head_width=0.3, head_length=0.2,
              fc='blue', ec='blue', linewidth=2, label='Teacher')
    ax1.arrow(0, 0, student_a[0], student_a[1], head_width=0.2, head_length=0.15,
              fc='green', ec='green', linewidth=1.5, label='Student A (同方向小幅度)')
    ax1.arrow(0, 0, student_b[0], student_b[1], head_width=0.2, head_length=0.15,
              fc='red', ec='red', linewidth=1.5, label='Student B (同方向大幅度)')
    ax1.arrow(0, 0, student_c[0], student_c[1], head_width=0.2, head_length=0.15,
              fc='orange', ec='orange', linewidth=1.5, label='Student C (不同方向)')

    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('向量示意圖')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # =============================================
    # 2. 計算兩種 Loss
    # =============================================
    ax2 = axes[0, 1]

    # 轉成 tensor
    t = torch.tensor(teacher, dtype=torch.float32)
    s_a = torch.tensor(student_a, dtype=torch.float32)
    s_b = torch.tensor(student_b, dtype=torch.float32)
    s_c = torch.tensor(student_c, dtype=torch.float32)

    # L2 Loss (MSE)
    l2_a = F.mse_loss(s_a, t).item()
    l2_b = F.mse_loss(s_b, t).item()
    l2_c = F.mse_loss(s_c, t).item()

    # Cosine Loss
    cos_a = 1 - F.cosine_similarity(s_a.unsqueeze(0), t.unsqueeze(0)).item()
    cos_b = 1 - F.cosine_similarity(s_b.unsqueeze(0), t.unsqueeze(0)).item()
    cos_c = 1 - F.cosine_similarity(s_c.unsqueeze(0), t.unsqueeze(0)).item()

    cases = ['A\n(同方向小)', 'B\n(同方向大)', 'C\n(不同方向)']
    x = np.arange(3)
    width = 0.35

    bars1 = ax2.bar(x - width/2, [l2_a, l2_b, l2_c], width, label='L2 (MSE)', color='steelblue')
    bars2 = ax2.bar(x + width/2, [cos_a, cos_b, cos_c], width, label='Cosine Loss', color='coral')

    ax2.set_ylabel('Loss Value')
    ax2.set_title('L2 vs Cosine Loss 比較')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 標註數值
    for bar, val in zip(bars1, [l2_a, l2_b, l2_c]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, [cos_a, cos_b, cos_c]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # =============================================
    # 3. 關鍵差異說明
    # =============================================
    ax3 = axes[0, 2]
    ax3.axis('off')

    explanation = """
    關鍵差異:

    L2 (MSE) Loss:
    ─────────────────────────────
    • 測量「歐幾里得距離」
    • 同時考慮方向和幅度
    • Case A: 幅度小 → loss 小
    • Case B: 幅度大 → loss 大
    • Case C: 方向偏 → loss 中等

    Cosine Loss:
    ─────────────────────────────
    • 測量「方向差異」
    • 忽略幅度，只看角度
    • Case A: 同方向 → loss ≈ 0
    • Case B: 同方向 → loss ≈ 0
    • Case C: 方向偏 → loss > 0

    為什麼用 Cosine?
    ─────────────────────────────
    1. 不同層的 feature 幅度差異大
       (L0: ~0.5, L6: ~3000)
    2. L2 會被大幅度層主導
    3. Cosine 讓所有層在相同尺度
    4. 我們更關心「特徵方向」
    """
    ax3.text(0.1, 0.95, explanation, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =============================================
    # 4. 實際數據：不同層的 MSE 差異
    # =============================================
    ax4 = axes[1, 0]

    # 從 noise_sensitivity.json 的 MSE 數據
    layer_mse = {
        0: 0.46, 1: 21.12, 2: 10.46, 3: 111.81, 4: 1277.38,
        5: 483.29, 6: 3185.23, 7: 2140.66, 8: 563.01,
        9: 440.17, 10: 73.18, 11: 2.01, 12: 0.17,
        13: 0.21, 14: 0.12, 15: 0.07
    }

    layers = list(range(16))
    mse_values = [layer_mse[l] for l in layers]

    ax4.bar(layers, mse_values, color='steelblue', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('MSE (Clean vs Noisy)')
    ax4.set_title('MSE 在不同層的尺度差異\n(尺度差異超過 10000 倍!)')
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3)

    # 標註最大和最小
    max_idx = np.argmax(mse_values)
    min_idx = np.argmin(mse_values)
    ax4.annotate(f'L{max_idx}: {mse_values[max_idx]:.0f}',
                xy=(max_idx, mse_values[max_idx]),
                xytext=(max_idx+1, mse_values[max_idx]*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')
    ax4.annotate(f'L{min_idx}: {mse_values[min_idx]:.2f}',
                xy=(min_idx, mse_values[min_idx]),
                xytext=(min_idx+2, mse_values[min_idx]*5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    # =============================================
    # 5. Cosine Loss 在不同層的表現
    # =============================================
    ax5 = axes[1, 1]

    # 從 noise_sensitivity.json 的 cos_sim 數據
    layer_cos = {
        0: 0.041, 1: 0.057, 2: 0.172, 3: 0.105, 4: 0.121,
        5: 0.351, 6: 0.081, 7: 0.165, 8: 0.303,
        9: 0.479, 10: 0.946, 11: 0.566, 12: 0.593,
        13: 0.589, 14: 0.556, 15: 0.379
    }

    cos_loss_values = [1 - layer_cos[l] for l in layers]

    colors = ['#d62728' if v > 0.8 else '#ff7f0e' if v > 0.5 else '#2ca02c' for v in cos_loss_values]
    ax5.bar(layers, cos_loss_values, color=colors, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Cosine Loss (1 - cos_sim)')
    ax5.set_title('Cosine Loss 在不同層\n(所有值都在 0-1 範圍)')
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # =============================================
    # 6. 總結
    # =============================================
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary = """
    總結: 何時用哪種 Loss?

    ╔═══════════════════════════════════════════╗
    ║  Cosine Loss 適合:                        ║
    ║  • 不同層/位置的 feature 尺度差異大       ║
    ║  • 關心「語義方向」而非精確數值           ║
    ║  • 中間層監督 (不同層需要可比較)          ║
    ╠═══════════════════════════════════════════╣
    ║  L2 (MSE) Loss 適合:                      ║
    ║  • 同一層的 feature (尺度相同)            ║
    ║  • 需要精確匹配數值                       ║
    ║  • 最後一層監督 (直接匹配 teacher)        ║
    ╠═══════════════════════════════════════════╣
    ║  建議組合:                                ║
    ║  • 中間層: Cosine Loss (跨層可比)         ║
    ║  • 最後層: MSE (精確匹配)                 ║
    ║  • L10 錨點: MSE (因為本來就很接近)       ║
    ╚═══════════════════════════════════════════╝
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/analysis/loss_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_comparison.png")


if __name__ == '__main__':
    visualize_loss_difference()
