"""
整合 exp_1231_feature 與本次分析的噪音敏感度數據
找出 WavTokenizer 對噪音「最直覺處理」的層作為監督位置
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_data():
    """載入兩個實驗的數據"""

    # exp_1231_feature: 18 層完整分析 (L0-L17)
    with open('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_1231_feature/outputs/noise_sensitivity_results.json') as f:
        exp_1231 = json.load(f)

    # 本次分析: 16 層 (encoder.model 0-15)
    with open('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/analysis/noise_sensitivity.json') as f:
        current = json.load(f)

    return exp_1231, current


def analyze_integrated():
    """整合分析"""

    exp_1231, current = load_all_data()

    # exp_1231 的平均噪音敏感度
    exp_1231_avg = exp_1231['health_report']['avg_layer_similarities']

    # 層名對應 (exp_1231 的 18 層)
    layer_names_1231 = [
        "L0: model.0.conv (input)",
        "L1: model.1.block.1",
        "L2: model.1.block.3",
        "L3: model.1.shortcut",
        "L4: model.3.conv (downsample)",
        "L5: model.4.block.1",
        "L6: model.4.block.3",
        "L7: model.4.shortcut",
        "L8: model.6.conv (downsample)",
        "L9: model.7.block.1",
        "L10: model.7.block.3",
        "L11: model.7.shortcut",
        "L12: model.9.conv (downsample)",
        "L13: model.10.block.1",
        "L14: model.10.block.3",
        "L15: model.10.shortcut",
        "L16: model.12.conv (downsample)",
        "L17: model.15.conv (output)",
    ]

    # 層組定義 (exp_1231)
    layer_groups_1231 = {
        'input': [0],
        'low_level': [1, 2, 3, 4],
        'mid_level': [5, 6, 7, 8],
        'semantic': [9, 10, 11, 12],
        'abstract': [13, 14, 15, 16],
        'output': [17],
    }

    # 建立圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # =============================================
    # 1. exp_1231 的噪音敏感度 (18 層)
    # =============================================
    ax1 = axes[0, 0]

    sensitivity_1231 = [1 - s for s in exp_1231_avg]
    colors = []
    for i, s in enumerate(sensitivity_1231):
        if i in [5, 6, 7, 8]:  # mid_level
            colors.append('#d62728' if s > 0.5 else '#ff7f0e')
        elif i in [13, 14, 15]:  # abstract
            colors.append('#2ca02c')
        else:
            colors.append('#1f77b4')

    ax1.bar(range(18), sensitivity_1231, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Layer Index (exp_1231_feature)')
    ax1.set_ylabel('Noise Sensitivity (1 - CosSim)')
    ax1.set_title('exp_1231_feature: WavTokenizer Noise Sensitivity (18 Layers)\n'
                  'Red=Mid-level (most sensitive), Green=Abstract (robust)')
    ax1.set_xticks(range(18))
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    # 標記最敏感的層
    most_sensitive_1231 = sorted(enumerate(sensitivity_1231), key=lambda x: x[1], reverse=True)[:4]
    for rank, (idx, val) in enumerate(most_sensitive_1231):
        ax1.annotate(f'#{rank+1}', xy=(idx, val), xytext=(idx, val+0.05),
                    ha='center', fontsize=9, fontweight='bold', color='red')

    # =============================================
    # 2. 本次分析的噪音敏感度 (16 層)
    # =============================================
    ax2 = axes[0, 1]

    layers_current = list(range(16))
    cos_sim_current = [current['cos_sim'][str(l)] for l in layers_current]
    sensitivity_current = [1 - c for c in cos_sim_current]

    colors2 = ['#d62728' if s > 0.8 else '#ff7f0e' if s > 0.5 else '#2ca02c' for s in sensitivity_current]
    ax2.bar(layers_current, sensitivity_current, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index (encoder.model[i])')
    ax2.set_ylabel('Noise Sensitivity (1 - CosSim)')
    ax2.set_title('Current Analysis: encoder.model Noise Sensitivity (16 Layers)\n'
                  'Red=High sensitivity, Green=Low sensitivity')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    # 標記當前監督位置
    supervised = [3, 6]
    for sl in supervised:
        ax2.scatter([sl], [sensitivity_current[sl]], s=200, c='blue', marker='*',
                   edgecolors='black', linewidth=1, zorder=5)

    # =============================================
    # 3. 層組比較
    # =============================================
    ax3 = axes[1, 0]

    # exp_1231 層組平均
    group_names = ['input', 'low_level', 'mid_level', 'semantic', 'abstract', 'output']
    group_sens_1231 = []
    for name in group_names:
        indices = layer_groups_1231[name]
        vals = [sensitivity_1231[i] for i in indices]
        group_sens_1231.append(np.mean(vals))

    # 本次分析層組平均 (對應到 encoder.model)
    # encoder.model 的層組定義
    layer_groups_current = {
        'input': [0],
        'low_level': [1, 2, 3, 4],
        'mid_level': [5, 6, 7, 8],
        'semantic': [9, 10, 11, 12],
        'abstract': [13, 14, 15],
    }

    group_sens_current = []
    for name in ['input', 'low_level', 'mid_level', 'semantic', 'abstract']:
        indices = layer_groups_current[name]
        vals = [sensitivity_current[i] for i in indices]
        group_sens_current.append(np.mean(vals))
    group_sens_current.append(0)  # 沒有 output 層

    x = np.arange(len(group_names))
    width = 0.35

    bars1 = ax3.bar(x - width/2, group_sens_1231, width, label='exp_1231 (18 layers)', color='steelblue')
    bars2 = ax3.bar(x + width/2, group_sens_current[:5] + [0], width, label='Current (16 layers)', color='coral')

    ax3.set_xlabel('Layer Group')
    ax3.set_ylabel('Avg Noise Sensitivity')
    ax3.set_title('Layer Group Comparison\n(Higher = More Sensitive to Noise)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_names)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # =============================================
    # 4. 監督位置建議
    # =============================================
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 計算建議
    # exp_1231 最敏感: L5, L6 (mid_level)
    # 本次分析最敏感: L0, L1, L6

    recommendation_text = """
    Integrated Noise Sensitivity Analysis
    ══════════════════════════════════════════════════════════════════

    exp_1231_feature Findings (18 layers, WavTokenizer full model):
    ───────────────────────────────────────────────────────────────
    • Most sensitive: L6 (model.4.block.3), L5 (model.4.block.1)
      → Mid-level layers where noise-speech separation happens
    • Most robust: L13-L15 (model.10.block)
      → Abstract semantic layers, naturally noise-invariant
    • Output L17 is sensitive (noise propagates to codebook)

    Current Analysis (16 layers, encoder.model):
    ───────────────────────────────────────────────────────────────
    • Most sensitive: L0, L1 (input/low_level), L6 (mid_level)
    • Most robust: L10 (semantic), cos_sim=0.946

    Key Insight: WHERE DOES WAVTOKENIZER "NATURALLY" PROCESS NOISE?
    ───────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  L5-L6 (mid_level) is the NOISE PROCESSING LAYER           │
    │                                                             │
    │  Evidence:                                                  │
    │  1. exp_1231: mid_level avg sensitivity = 0.71 (highest!)  │
    │  2. Current: L6 sensitivity = 0.92 (mid_level's most)      │
    │  3. This is where noise-speech features separate           │
    │                                                             │
    │  → This should be the PRIMARY supervision target           │
    └─────────────────────────────────────────────────────────────┘

    RECOMMENDED SUPERVISION STRATEGY:
    ───────────────────────────────────────────────────────────────

    Priority 1: L5-L6 (Mid-level noise processing)
    ┌────────────────────────────────────────────────────┐
    │ • This is where WavTokenizer "decides" noise       │
    │ • Supervising here teaches correct separation      │
    │ • Weight: 1.0 (highest priority)                   │
    └────────────────────────────────────────────────────┘

    Priority 2: L1-L3 (Low-level, capture early noise)
    ┌────────────────────────────────────────────────────┐
    │ • Noise first enters here                          │
    │ • Current L3 supervision is correct                │
    │ • Consider adding L1                               │
    │ • Weight: 0.8                                      │
    └────────────────────────────────────────────────────┘

    Priority 3: L10 (Semantic anchor)
    ┌────────────────────────────────────────────────────┐
    │ • Most stable layer (cos_sim=0.946)                │
    │ • Use as "anchor" - light supervision              │
    │ • Ensure semantic content preserved                │
    │ • Weight: 0.3 (low, just regularization)           │
    └────────────────────────────────────────────────────┘

    NOT recommended: L0 (too early, mostly raw signal)
    NOT recommended: L13-L15 (already robust, no need)

    Final Configuration:
    ───────────────────────────────────────────────────────────────
    intermediate_indices = [3, 6]      # Current (OK)
    intermediate_indices = [1, 3, 6]   # Better (add L1)
    intermediate_indices = [3, 5, 6]   # Alternative (add L5)

    Weights: L6 > L5 > L3 > L1 (based on noise processing role)
    """

    ax4.text(0.02, 0.98, recommendation_text, transform=ax4.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/analysis/integrated_noise_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: integrated_noise_analysis.png")

    # 打印詳細結果
    print("\n" + "="*70)
    print("Integrated Noise Sensitivity Analysis")
    print("="*70)

    print("\n【exp_1231_feature 結果 (18 層)】")
    print(f"{'Layer':>5} | {'Name':>25} | {'CosSim':>8} | {'Sensitivity':>11}")
    print("-" * 60)
    for i, (sim, name) in enumerate(zip(exp_1231_avg, layer_names_1231)):
        sens = 1 - sim
        marker = "★" if sens > 0.7 else ""
        print(f"L{i:>4} | {name[:25]:>25} | {sim:>8.3f} | {sens:>11.3f} {marker}")

    print("\n【層組平均】")
    for name, sens in zip(group_names, group_sens_1231):
        print(f"  {name:15s}: {sens:.3f}")

    print("\n" + "-"*70)
    print("\n【本次分析結果 (encoder.model 16 層)】")
    print(f"{'Layer':>5} | {'CosSim':>8} | {'Sensitivity':>11} | Note")
    print("-" * 60)
    for i in range(16):
        sim = cos_sim_current[i]
        sens = sensitivity_current[i]
        note = ""
        if i in [3, 6]:
            note = "★ Currently supervised"
        elif sens > 0.9:
            note = "⚠ Very sensitive"
        elif sens < 0.1:
            note = "✓ Very stable"
        print(f"L{i:>4} | {sim:>8.3f} | {sens:>11.3f} | {note}")

    print("\n" + "="*70)
    print("【結論】")
    print("="*70)
    print("""
    1. 兩個實驗都確認: Mid-level (L5-L6) 是噪音處理的關鍵層
       - exp_1231: L6 (model.4.block.3) 最敏感
       - 本次: L6 在 mid_level 中最敏感

    2. 當前監督位置 (L3, L6) 是合理的選擇:
       - L3 覆蓋 low_level
       - L6 覆蓋 mid_level (噪音處理層)

    3. 可優化方向:
       - 加入 L5 (與 L6 協同處理噪音)
       - 或加入 L1 (更早捕捉噪音)
       - L10 作為錨點 (語義穩定層)

    4. 不需要監督的層:
       - L13-L15 (已經對噪音魯棒)
       - L0 (太早，主要是原始信號)
    """)


if __name__ == '__main__':
    analyze_integrated()
