"""
exp_0206 Plan Original: 結果分析腳本

分析 short-run 實驗結果，繪製曲線，判斷 P1/P2/P3 gate。

用法:
    python exp_0206/plan_ori/analyze_results.py <output_dir>
"""

import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def analyze_short_run(output_dir: str):
    """分析 short-run 實驗結果

    Args:
        output_dir: 實驗輸出目錄路徑
    """
    output_dir = Path(output_dir)

    # 載入 metrics
    metrics_path = output_dir / 'metrics_history.json'
    if not metrics_path.exists():
        print(f"❌ 找不到 {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics:
        print("❌ metrics_history.json 為空")
        return

    final = metrics[-1]

    # 基準比較
    baselines = {
        'Baseline (frozen)': {'entropy': 6.07, 'top10': 0.197, 'used': 740, 'usage_pct': 18.0},
        'RVQ (4×2048)': {'entropy': 9.03, 'top10': 0.158, 'used': 1089, 'usage_pct': 53.0},
    }

    print("\n" + "=" * 70)
    print("exp_0206 Plan Ori: Short-run Results")
    print("=" * 70)

    print(f"\nOutput directory: {output_dir}")
    print(f"Total steps: {final.get('step', 'N/A')}")

    print("\n--- Final Metrics ---")
    print(f"  Entropy:       {final['entropy']:.3f}")
    print(f"  Top-10 mass:   {final['top10_mass']:.4f} ({final['top10_mass']*100:.1f}%)")
    print(f"  Used codes:    {final['used_codes']}/4096 ({final['usage_pct']:.1f}%)")
    print(f"  Feature MSE:   {final['feature_mse']:.4f}")

    print("\n--- Gate Results ---")
    # P1 (step 200)
    p1_entry = next((m for m in metrics if m['step'] >= 200), None)
    if p1_entry:
        p1_pass = p1_entry.get('p1_pass', False)
        print(f"  P1 (step {p1_entry['step']}): {'✅ PASS' if p1_pass else '❌ FAIL'}")
        print(f"    top10={p1_entry['top10_mass']:.4f} (≤0.95), "
              f"used={p1_entry['used_codes']} (≥82), "
              f"mse={p1_entry['feature_mse']:.4f} (≤0.1)")

    # P2 (step 1000)
    p2_pass = final.get('p2_pass', False)
    print(f"  P2 (step {final.get('step', 'final')}): {'✅ PASS' if p2_pass else '❌ FAIL'}")
    print(f"    entropy={final['entropy']:.3f} (≥5.0), "
          f"top10={final['top10_mass']:.4f} (≤0.5), "
          f"used={final['used_codes']} (≥410), "
          f"mse={final['feature_mse']:.4f} (≤0.1)")

    # P3 (bonus)
    p3_pass = final.get('p3_pass', False)
    print(f"  P3 (bonus): {'🎯 ACHIEVED' if p3_pass else '⚠️ NOT MET'}")
    print(f"    entropy={final['entropy']:.3f} (>6.5), "
          f"top10={final['top10_mass']:.4f} (<0.15), "
          f"used={final['used_codes']} (≥2867)")

    # Baseline 比較
    print("\n--- Comparison with Baselines ---")
    print(f"  {'Method':<25} {'Entropy':>8} {'Top-10':>8} {'Used':>6} {'Usage%':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")
    for name, b in baselines.items():
        print(f"  {name:<25} {b['entropy']:>8.2f} {b['top10']:>8.3f} {b['used']:>6d} {b['usage_pct']:>6.1f}%")
    print(f"  {'Plan Ori (ours)':<25} {final['entropy']:>8.3f} {final['top10_mass']:>8.4f} "
          f"{final['used_codes']:>6d} {final['usage_pct']:>6.1f}%")

    # 繪圖
    plot_results(metrics, output_dir)

    # 決策建議
    print("\n--- Decision ---")
    if p2_pass:
        print("  ✅ P2 PASSED - 可考慮進行 long-run (300 epochs)")
        if p3_pass:
            print("  🎯 P3 ACHIEVED - 優秀表現！方案 A 有潛力")
        print("  建議: 與 RVQ 長期對比")
    else:
        print("  ❌ P2 FAILED - 方案 A 未通過 short-run 驗證")
        print("  建議: 分析失敗原因，考慮調整參數或終止方案 A")

    print("=" * 70)


def plot_results(metrics: list, output_dir: Path):
    """繪製實驗結果圖表

    Args:
        metrics: metrics 歷史列表
        output_dir: 輸出目錄
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('exp_0206 Plan Ori: Single VQ 4096 + EMA (Short-run)', fontsize=14)

    steps = [m['step'] for m in metrics]

    # 基準線
    baseline_entropy = 6.07
    rvq_entropy = 9.03
    baseline_top10 = 0.197
    rvq_top10 = 0.158
    baseline_used = 740

    # 1. Entropy
    ax = axes[0, 0]
    ax.plot(steps, [m['entropy'] for m in metrics], 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='P2: ≥5.0')
    ax.axhline(y=6.5, color='green', linestyle='--', alpha=0.7, label='P3: >6.5')
    ax.axhline(y=baseline_entropy, color='gray', linestyle=':', alpha=0.5, label=f'Baseline: {baseline_entropy}')
    ax.axhline(y=rvq_entropy, color='purple', linestyle=':', alpha=0.5, label=f'RVQ: {rvq_entropy}')
    ax.set_title('Entropy (bits)')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2. Top-10 mass
    ax = axes[0, 1]
    ax.plot(steps, [m['top10_mass'] for m in metrics], 'r-o', linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='P2: ≤0.5')
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='P3: <0.15')
    ax.axhline(y=baseline_top10, color='gray', linestyle=':', alpha=0.5, label=f'Baseline: {baseline_top10}')
    ax.axhline(y=rvq_top10, color='purple', linestyle=':', alpha=0.5, label=f'RVQ: {rvq_top10}')
    ax.set_title('Top-10 Mass')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 3. Used codes
    ax = axes[0, 2]
    ax.plot(steps, [m['used_codes'] for m in metrics], 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=410, color='orange', linestyle='--', alpha=0.7, label='P2: ≥410')
    ax.axhline(y=2867, color='green', linestyle='--', alpha=0.7, label='P3: ≥2867')
    ax.axhline(y=baseline_used, color='gray', linestyle=':', alpha=0.5, label=f'Baseline: {baseline_used}')
    ax.set_title('Used Codes / 4096')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 4. Feature MSE
    ax = axes[1, 0]
    ax.plot(steps, [m['feature_mse'] for m in metrics], 'brown', marker='o', linewidth=2, markersize=6)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold: ≤0.1')
    ax.set_title('Feature MSE (z_q vs t_e)')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    # 5. Training losses
    ax = axes[1, 1]
    if all('train_loss_quant' in m for m in metrics):
        ax.plot(steps, [m['train_loss_quant'] for m in metrics],
                'b-o', label='Quant', markersize=4)
    if all('train_loss_commit' in m for m in metrics):
        ax.plot(steps, [m['train_loss_commit'] for m in metrics],
                'r-o', label='Commit', markersize=4)
    if all('train_loss_inter' in m for m in metrics):
        ax.plot(steps, [m['train_loss_inter'] for m in metrics],
                'g-o', label='Inter', markersize=4)
    ax.set_title('Training Losses')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    # 6. Usage percentage
    ax = axes[1, 2]
    ax.plot(steps, [m['usage_pct'] for m in metrics], 'darkorange', marker='o', linewidth=2, markersize=6)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='P2: ≥10%')
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='P3: ≥70%')
    ax.set_title('Codebook Usage %')
    ax.set_xlabel('Step')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'analysis_curves_{timestamp}_analyze_results.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  ✅ Analysis plot saved: {plot_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python exp_0206/plan_ori/analyze_results.py <output_dir>")
        sys.exit(1)

    analyze_short_run(sys.argv[1])
