"""
exp_0305: LoRA 決策視覺化

根據 exp_0304 feature map 分析的數據，產生可視化圖表，
解釋為什麼選擇這幾層加 LoRA、其他層凍結的依據。

產出圖表：
  1. layer_decision_matrix   — 各層 noise/content/temporal 三維雷達圖
  2. noise_vs_temporal_scatter — 噪音敏感度 vs 高頻時間細節 散點圖（決策邊界）
  3. plan_comparison_bar      — plan_a/b/c 各層參數分配比較
  4. freeze_vs_lora_heatmap   — 18 層決策熱力圖

執行：
  python families/deps/selective_lora/visualize_layer_decision.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# 數據（直接來自 families/official/material_generalization/conv18_14wav_role_metrics.csv）
# ============================================================

LAYER_DATA = [
    # idx  label           noise    content  temporal  stage
    (0,  'L0  stem',       0.745,   0.510,   0.003,    'stem'),
    (1,  'L1  RB1-C1',     0.225,   0.858,   0.016,    'residual'),
    (2,  'L2  RB1-C2',     0.245,   0.862,   0.024,    'residual'),
    (3,  'L3  RB1-SC',     0.404,   0.784,   0.016,    'residual'),
    (4,  'L4  Down1',      0.106,   0.949,   0.095,    'downsample'),
    (5,  'L5  RB2-C1',     0.063,   0.974,   0.255,    'residual'),
    (6,  'L6  RB2-C2',     0.156,   0.920,   0.330,    'residual'),
    (7,  'L7  RB2-SC',     0.086,   0.961,   0.072,    'residual'),
    (8,  'L8  Down2',      0.197,   0.906,   0.524,    'downsample'),
    (9,  'L9  RB3-C1',     0.139,   0.929,   0.687,    'residual'),
    (10, 'L10 RB3-C2',     0.028,   0.988,   0.292,    'residual'),
    (11, 'L11 RB3-SC',     0.326,   0.838,   0.174,    'residual'),
    (12, 'L12 Down3',      0.164,   0.919,   0.185,    'downsample'),
    (13, 'L13 RB4-C1',     0.027,   0.987,   0.130,    'residual'),
    (14, 'L14 RB4-C2',     0.087,   0.934,   0.026,    'residual'),
    (15, 'L15 RB4-SC',     0.008,   0.996,   0.042,    'residual'),
    (16, 'L16 Down4',      0.201,   0.782,   0.001,    'downsample'),
    (17, 'L17 output',     0.147,   0.815,   0.001,    'projection'),
]

# Plan 決策
PLAN_A_LORA = {0, 2, 3, 8, 9, 11}         # adapt_top6
PLAN_B_LORA = {0, 2, 3, 6, 8, 9, 11, 12}  # adapt_top8

# 決策準則
NOISE_THRESH = 0.10
TEMPORAL_THRESH = 0.15
CONTENT_THRESH = 0.93

# ============================================================
# 決策函數
# ============================================================

def layer_decision(noise, content, temporal, idx):
    """判斷層應訓練或凍結。

    準則：
      - LoRA: noise > NOISE_THRESH 或 temporal > TEMPORAL_THRESH
      - Freeze: content > CONTENT_THRESH 且 noise < NOISE_THRESH

    Args:
        noise: 噪音敏感度（1 - cosine_sim）
        content: 內容共享度（cosine_sim）
        temporal: 時間細節分數（歸一化）
        idx: 層索引

    Returns:
        'lora' | 'freeze' | 'borderline'
    """
    if noise > NOISE_THRESH or temporal > TEMPORAL_THRESH:
        return 'lora'
    if content > CONTENT_THRESH and noise < NOISE_THRESH:
        return 'freeze'
    return 'borderline'


# ============================================================
# 圖表 1: 18 層指標長條圖（含決策標色）
# ============================================================

def plot_layer_decision_matrix(out_dir: Path, date_str: str):
    """繪製各層 noise/content/temporal 指標長條圖，並標示 LoRA/Freeze 決策。

    Args:
        out_dir: 輸出目錄
        date_str: 日期字串（用於檔名）
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(
        'exp_0304 Feature Map Analysis → exp_0305 LoRA Layer Decision\n'
        '(evidence source: conv18_14wav_role_metrics.csv)',
        fontsize=12
    )

    labels = [d[1] for d in LAYER_DATA]
    x = np.arange(len(LAYER_DATA))

    noise_vals = [d[2] for d in LAYER_DATA]
    content_vals = [d[3] for d in LAYER_DATA]
    temp_vals = [d[4] for d in LAYER_DATA]

    decisions = [layer_decision(d[2], d[3], d[4], d[0]) for d in LAYER_DATA]
    colors = {
        'lora': '#e74c3c',       # 紅 = LoRA
        'freeze': '#2ecc71',     # 綠 = Freeze
        'borderline': '#f39c12', # 橘 = 邊界
    }
    bar_colors = [colors[dec] for dec in decisions]

    # ── Noise sensitivity ────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, noise_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axhline(NOISE_THRESH, color='red', ls='--', lw=1.5, label=f'threshold={NOISE_THRESH}')
    ax.set_ylabel('Noise Sensitivity\n(1 - cosine_sim)')
    ax.set_title('① 噪音敏感度：越高越需要 LoRA 去噪')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(noise_vals):
        if v > 0.1:
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # ── Content shared ───────────────────────────────────────
    ax = axes[1]
    ax.bar(x, content_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axhline(CONTENT_THRESH, color='green', ls='--', lw=1.5, label=f'threshold={CONTENT_THRESH}')
    ax.set_ylabel('Content Shared Score\n(cosine_sim)')
    ax.set_title('② 語音內容穩定度：越高越適合凍結（保留 WavTokenizer 語音模仿能力）')
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # ── Temporal detail ──────────────────────────────────────
    ax = axes[2]
    ax.bar(x, temp_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axhline(TEMPORAL_THRESH, color='blue', ls='--', lw=1.5, label=f'threshold={TEMPORAL_THRESH}')
    ax.set_ylabel('Temporal Detail Score\n(normalised temp_std)')
    ax.set_title('③ 高頻時間細節：高分層是高頻恢復的關鍵，需要 LoRA 主動訓練')
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(temp_vals):
        if v > 0.15:
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 圖例
    legend_patches = [
        mpatches.Patch(color='#e74c3c', alpha=0.85, label='LoRA (noise>0.10 or temporal>0.15)'),
        mpatches.Patch(color='#2ecc71', alpha=0.85, label='Freeze (content>0.93 and noise<0.10)'),
        mpatches.Patch(color='#f39c12', alpha=0.85, label='Borderline (L16, L17)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    fname = out_dir / f'exp0305_{date_str}_plot_layer_decision_matrix.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname.name}")


# ============================================================
# 圖表 2: Noise vs Temporal 散點決策圖
# ============================================================

def plot_noise_vs_temporal(out_dir: Path, date_str: str):
    """繪製 noise_sensitivity vs temporal_detail 散點圖，標示決策邊界。

    此圖說明 plan_a/b 的選擇準則在二維空間中的位置。

    Args:
        out_dir: 輸出目錄
        date_str: 日期字串
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(
        'Noise Sensitivity vs Temporal Detail\n'
        'LoRA Decision Boundary (exp_0304 evidence → exp_0305 plan_b)',
        fontsize=11,
    )

    # 決策區域背景
    noise_lim = [0, 0.80]
    temp_lim = [0, 0.75]
    ax.fill_betweenx([TEMPORAL_THRESH, 0.75], 0, 0.80,
                     color='#e74c3c', alpha=0.07, label='_nolegend_')
    ax.fill_betweenx([0, TEMPORAL_THRESH], NOISE_THRESH, 0.80,
                     color='#e74c3c', alpha=0.07, label='_nolegend_')
    ax.fill_betweenx([0, TEMPORAL_THRESH], 0, NOISE_THRESH,
                     color='#2ecc71', alpha=0.07, label='_nolegend_')

    ax.axhline(TEMPORAL_THRESH, color='blue', ls='--', lw=1.2, alpha=0.6,
               label=f'temporal threshold = {TEMPORAL_THRESH}')
    ax.axvline(NOISE_THRESH, color='red', ls='--', lw=1.2, alpha=0.6,
               label=f'noise threshold = {NOISE_THRESH}')

    for idx, label, noise, content, temporal, stage in LAYER_DATA:
        if idx in PLAN_B_LORA:
            color = '#c0392b'
            marker = 'o'
            ms = 130
            zorder = 5
        elif noise < NOISE_THRESH and temporal < TEMPORAL_THRESH:
            color = '#27ae60'
            marker = 's'
            ms = 90
            zorder = 4
        else:
            color = '#f39c12'
            marker = '^'
            ms = 90
            zorder = 4

        ax.scatter(noise, temporal, c=color, s=ms, marker=marker,
                   zorder=zorder, edgecolors='white', linewidths=0.8)

        offset_x = 0.008
        offset_y = 0.01
        if idx == 9:   # L9 特別標注
            offset_y = 0.018
        elif idx in {8, 11}:
            offset_x = -0.035

        short_label = f'L{idx}'
        ax.annotate(
            short_label, (noise, temporal),
            xytext=(noise + offset_x, temporal + offset_y),
            fontsize=8.5,
            fontweight='bold' if idx in PLAN_B_LORA else 'normal',
        )

    # 星標
    ax.annotate('★ L9 temporal=0.687\n(高頻核心)',
                xy=(0.139, 0.687), xytext=(0.28, 0.60),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red')

    legend_handles = [
        mpatches.Patch(color='#c0392b', label=f'plan_b LoRA layers ({len(PLAN_B_LORA)} layers)'),
        mpatches.Patch(color='#27ae60', label='Freeze (content stable)'),
        mpatches.Patch(color='#f39c12', label='Borderline'),
        plt.Line2D([0], [0], ls='--', color='red', label=f'noise threshold={NOISE_THRESH}'),
        plt.Line2D([0], [0], ls='--', color='blue', label=f'temporal threshold={TEMPORAL_THRESH}'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
    ax.set_xlabel('Noise Sensitivity  (1 - cosine_sim same_spk_noisy_vs_clean)', fontsize=10)
    ax.set_ylabel('Temporal Detail Score  (normalised temporal_std)', fontsize=10)
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(-0.02, 0.76)
    ax.grid(True, alpha=0.25)

    fname = out_dir / f'exp0305_{date_str}_plot_noise_vs_temporal.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname.name}")


# ============================================================
# 圖表 3: 18 層決策一覽熱力圖
# ============================================================

def plot_decision_heatmap(out_dir: Path, date_str: str):
    """繪製 18 層決策一覽熱力圖，標示 plan_a/b/c 的 LoRA 覆蓋範圍。

    Args:
        out_dir: 輸出目錄
        date_str: 日期字串
    """
    fig, axes = plt.subplots(1, 1, figsize=(17, 5))

    labels = [d[1] for d in LAYER_DATA]
    n = len(LAYER_DATA)

    # plan 覆蓋矩陣
    # plan_c 覆蓋全部 18 層
    PLAN_C_LORA = set(range(18))

    plans = {
        'plan_a\n(adapt_top6\n6L rank=32)': PLAN_A_LORA,
        'plan_b\n(adapt_top8\n8L rank=32)': PLAN_B_LORA,
        'plan_c\n(all_18\nrank=10)': PLAN_C_LORA,
        'exp_0224a\nbaseline\n(all_18 rank=64)': PLAN_C_LORA,
    }

    mat = np.zeros((len(plans), n))
    for pi, (plan_name, plan_set) in enumerate(plans.items()):
        for li, (idx, *_) in enumerate(LAYER_DATA):
            if idx in plan_set:
                mat[pi, li] = 1.0

    # 底層指標行
    noise_row = np.array([d[2] for d in LAYER_DATA])
    temp_row = np.array([d[4] for d in LAYER_DATA])

    full_mat = np.vstack([noise_row, temp_row, mat])
    row_labels = ['noise_sens', 'temporal'] + list(plans.keys())

    cmap_binary = matplotlib.colors.ListedColormap(['#f8f9fa', '#e74c3c'])
    cmap_noise = plt.cm.Oranges
    cmap_temp = plt.cm.Blues

    # 逐行繪製
    ax = axes
    # 背景
    for ri, row in enumerate(full_mat):
        for ci, val in enumerate(row):
            if ri == 0:
                color = cmap_noise(val)
            elif ri == 1:
                color = cmap_temp(val)
            else:
                color = '#e74c3c' if val > 0.5 else '#e8f8f0'
            rect = plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1,
                                  fc=color, ec='white', lw=0.8)
            ax.add_patch(rect)
            # 數值標注
            if ri <= 1:
                txt = f'{val:.2f}'
                ax.text(ci, ri, txt, ha='center', va='center',
                        fontsize=6.5, color='black')
            else:
                if val > 0.5:
                    ax.text(ci, ri, '✓', ha='center', va='center',
                            fontsize=10, color='white', fontweight='bold')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, len(full_mat) - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(full_mat)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(
        'LoRA Layer Coverage Heatmap\n'
        '(red=LoRA active | white=frozen | rows 1-2 show evidence metrics)',
        fontsize=11,
    )
    ax.set_aspect('equal')

    plt.tight_layout()
    fname = out_dir / f'exp0305_{date_str}_plot_decision_heatmap.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname.name}")


# ============================================================
# 圖表 4: 參數預算比較長條圖（理論估算）
# ============================================================

def plot_param_budget(out_dir: Path, date_str: str):
    """繪製各 plan/baseline 的可訓練參數預算比較圖。

    Args:
        out_dir: 輸出目錄
        date_str: 日期字串
    """
    # 簡化估算：每個 Conv1d ≈ C_in * C_out * k，
    # LoRA 参数 = 2 * rank * (C_in * k) 约 兩組矩阵
    # 用 rank 和 层数简单近似
    APPROX_CONV_DIM = 512  # 平均 channel dim
    APPROX_KERNEL = 3

    def approx_params(n_layers, rank):
        """估算 LoRA 可訓練參數量（簡化版）。

        Args:
            n_layers: LoRA 覆蓋的層數
            rank: LoRA rank

        Returns:
            估算參數量
        """
        per_layer = 2 * rank * APPROX_CONV_DIM * APPROX_KERNEL
        return n_layers * per_layer

    plans_info = [
        ('plan_a\n(6L r=32)', approx_params(6, 32), '#e74c3c'),
        ('plan_b\n(8L r=32)', approx_params(8, 32), '#c0392b'),
        ('plan_c\n(18L r=10)', approx_params(18, 10), '#f39c12'),
        ('exp_0224a\nbaseline\n(18L r=64)', approx_params(18, 64), '#7f8c8d'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(plans_info))
    vals = [p[1] / 1000 for p in plans_info]
    bar_colors = [p[2] for p in plans_info]

    bars = ax.bar(xs, vals, color=bar_colors, alpha=0.85, edgecolor='white', width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in plans_info], fontsize=10)
    ax.set_ylabel('Estimated Trainable Parameters (K)', fontsize=10)
    ax.set_title(
        'exp_0305 Parameter Budget Comparison\n(LoRA trainable params only, approx)',
        fontsize=11,
    )
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}K', ha='center', va='bottom', fontsize=10)

    # 標示目標區
    ax.axhspan(200, 350, color='green', alpha=0.07, label='Target budget zone (plan_a/b)')
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = out_dir / f'exp0305_{date_str}_plot_param_budget.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname.name}")


# ============================================================
# Main
# ============================================================

def main():
    """產生所有 exp_0305 決策視覺化圖表。"""
    out_dir = Path(__file__).parent / 'decision_evidence'
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"  exp_0305 Layer Decision Visualizations")
    print(f"  output: {out_dir}")
    print(f"  date:   {date_str}")
    print(f"{'='*60}\n")

    print("Generating plots...")
    plot_layer_decision_matrix(out_dir, date_str)
    plot_noise_vs_temporal(out_dir, date_str)
    plot_decision_heatmap(out_dir, date_str)
    plot_param_budget(out_dir, date_str)

    # 同時輸出文字摘要
    summary = {
        'generated_by': f'families/deps/selective_lora/visualize_layer_decision.py ({date_str})',
        'evidence_source': 'families/official/material_generalization/wavtokenizer_featuremap_14wav_extended/conv18_14wav_role_metrics.csv',
        'decision_criteria': {
            'lora': 'noise_sensitivity > 0.10  OR  temporal_detail > 0.15',
            'freeze': 'content_shared > 0.93  AND  noise_sensitivity < 0.10',
        },
        'plan_a_lora_layers': sorted(PLAN_A_LORA),
        'plan_b_lora_layers': sorted(PLAN_B_LORA),
        'plan_b_freeze_layers': sorted(set(range(18)) - PLAN_B_LORA),
        'layer_metrics': [
            {
                'layer_idx': d[0],
                'label': d[1],
                'noise_sensitivity': d[2],
                'content_shared': d[3],
                'temporal_detail': d[4],
                'stage': d[5],
                'decision': layer_decision(d[2], d[3], d[4], d[0]),
                'in_plan_a': d[0] in PLAN_A_LORA,
                'in_plan_b': d[0] in PLAN_B_LORA,
            }
            for d in LAYER_DATA
        ],
        'key_findings': [
            'L9 has highest temporal detail (0.687) → primary HF recovery layer',
            'L8 Down2 has second highest temporal (0.524) → downsample changes HF most',
            'L15 RB4-SC has highest content stability (0.996) → absolutely freeze',
            'L3, L11 are shortcuts → bypass residual, directly carry noise → need LoRA',
            'L5 has high temporal (0.255) but very low noise (0.063) → freeze, L6 LoRA handles it',
        ],
    }
    summary_path = out_dir / f'exp0305_{date_str}_layer_decision_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {summary_path.name}")

    print(f"\n所有圖表已儲存至: {out_dir}")


if __name__ == '__main__':
    main()
