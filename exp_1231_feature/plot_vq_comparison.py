"""
繪製 Train (Inside) vs Val (Outside) Audio Quality Comparison
使用 Noisy→VQ 替代原來的 Noisy
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 載入評估結果
results_path = Path(__file__).parent / 'outputs' / 'vq_comparison_results.json'
with open(results_path) as f:
    data = json.load(f)

results = data['results']

# 數據
metrics = ['PESQ', 'STOI', 'SI-SDR (dB)']
sources = ['Teacher\n(Upper Bound)', 'Noisy→VQ\n(Baseline)', 'Student\n(Ours)']
colors = ['#2ecc71', '#3498db', '#e74c3c']  # 綠、藍、紅

# Train 數據
train_data = {
    'PESQ': [results['train']['teacher_vq']['pesq'],
             results['train']['noisy_vq']['pesq'],
             results['train']['student']['pesq']],
    'STOI': [results['train']['teacher_vq']['stoi'],
             results['train']['noisy_vq']['stoi'],
             results['train']['student']['stoi']],
    'SI-SDR (dB)': [results['train']['teacher_vq']['si_sdr'],
                   results['train']['noisy_vq']['si_sdr'],
                   results['train']['student']['si_sdr']],
}

# Val 數據
val_data = {
    'PESQ': [results['val']['teacher_vq']['pesq'],
             results['val']['noisy_vq']['pesq'],
             results['val']['student']['pesq']],
    'STOI': [results['val']['teacher_vq']['stoi'],
             results['val']['noisy_vq']['stoi'],
             results['val']['student']['stoi']],
    'SI-SDR (dB)': [results['val']['teacher_vq']['si_sdr'],
                   results['val']['noisy_vq']['si_sdr'],
                   results['val']['student']['si_sdr']],
}

# 創建圖
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Train (Inside) vs Val (Outside) Audio Quality Comparison', fontsize=16, fontweight='bold')

# 設定 x 軸位置
x = np.arange(len(sources))
width = 0.6

for col, metric in enumerate(metrics):
    # Train (上排)
    ax_train = axes[0, col]
    train_values = train_data[metric]
    bars = ax_train.bar(x, train_values, width, color=colors, edgecolor='black', linewidth=1.2)

    # 添加數值標籤
    for bar, val in zip(bars, train_values):
        height = bar.get_height()
        label = f'{val:.2f}' if metric == 'SI-SDR (dB)' else f'{val:.3f}'
        ax_train.annotate(label,
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_train.set_title(metric, fontsize=14, fontweight='bold')
    ax_train.set_xticks(x)
    ax_train.set_xticklabels(sources, fontsize=9)
    ax_train.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加 baseline 虛線 (Noisy→VQ)
    ax_train.axhline(y=train_values[1], color='gray', linestyle='--', alpha=0.5, linewidth=1)

    if col == 0:
        ax_train.set_ylabel('TRAIN (Inside)', fontsize=12, fontweight='bold')

    # 設定 y 軸範圍
    if metric == 'PESQ':
        ax_train.set_ylim(0, max(train_values) * 1.2)
    elif metric == 'STOI':
        ax_train.set_ylim(0, 1.0)

    # Val (下排)
    ax_val = axes[1, col]
    val_values = val_data[metric]
    bars = ax_val.bar(x, val_values, width, color=colors, edgecolor='black', linewidth=1.2)

    # 添加數值標籤
    for bar, val in zip(bars, val_values):
        height = bar.get_height()
        label = f'{val:.2f}' if metric == 'SI-SDR (dB)' else f'{val:.3f}'
        # 處理負值
        if height < 0:
            va = 'top'
            offset = -3
        else:
            va = 'bottom'
            offset = 3
        ax_val.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=11, fontweight='bold')

    ax_val.set_xticks(x)
    ax_val.set_xticklabels(sources, fontsize=9)
    ax_val.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加 baseline 虛線 (Noisy→VQ)
    ax_val.axhline(y=val_values[1], color='gray', linestyle='--', alpha=0.5, linewidth=1)

    if col == 0:
        ax_val.set_ylabel('VAL (Outside)', fontsize=12, fontweight='bold')

    # 設定 y 軸範圍
    if metric == 'PESQ':
        ax_val.set_ylim(0, max(val_values) * 1.2)
    elif metric == 'STOI':
        ax_val.set_ylim(0, 1.0)

plt.tight_layout()

# 儲存
output_path = Path(__file__).parent / 'outputs' / 'train_vs_val_vq_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# 也顯示數據對比
print("\n" + "="*70)
print("數據對比 (Noisy→VQ 作為 baseline)")
print("="*70)

print("\n### TRAIN (Inside) ###")
print(f"{'Metric':<15} {'Teacher':<12} {'Noisy→VQ':<12} {'Student':<12} {'Δ Student':<12}")
print("-"*60)
for metric in metrics:
    t, n, s = train_data[metric]
    delta = s - n
    print(f"{metric:<15} {t:<12.3f} {n:<12.3f} {s:<12.3f} {delta:+.3f}")

print("\n### VAL (Outside) ###")
print(f"{'Metric':<15} {'Teacher':<12} {'Noisy→VQ':<12} {'Student':<12} {'Δ Student':<12}")
print("-"*60)
for metric in metrics:
    t, n, s = val_data[metric]
    delta = s - n
    print(f"{metric:<15} {t:<12.3f} {n:<12.3f} {s:<12.3f} {delta:+.3f}")
