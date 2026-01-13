"""
Redraw noise sensitivity analysis figure with English labels
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Layer groups
LAYER_GROUPS = {
    'input': [0],
    'low_level': [1, 2, 3, 4],
    'mid_level': [5, 6, 7, 8],
    'semantic': [9, 10, 11, 12],
    'abstract': [13, 14, 15, 16],
    'output': [17],
}

def visualize_noise_sensitivity_en(results_path: Path, output_path: Path):
    """Visualize noise sensitivity analysis results in English"""

    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Extract data - check for health_report structure
    health_report = data.get('health_report', {})
    avg_cos_sim = np.array(health_report.get('avg_layer_similarities', []))
    group_stats = health_report.get('group_stats', {})

    n_layers = len(avg_cos_sim)

    # Calculate group averages if not provided
    if not group_stats:
        for group_name, indices in LAYER_GROUPS.items():
            valid_indices = [i for i in indices if i < n_layers]
            group_stats[group_name] = float(np.mean([avg_cos_sim[i] for i in valid_indices]))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-layer cosine similarity
    ax = axes[0, 0]
    colors = []
    color_map = {
        'input': '#808080',
        'low_level': '#ADD8E6',
        'mid_level': '#87CEEB',
        'semantic': '#4682B4',
        'abstract': '#000080',
        'output': '#191970',
    }
    for i in range(n_layers):
        for gname, indices in LAYER_GROUPS.items():
            if i in indices:
                colors.append(color_map[gname])
                break

    bars = ax.bar(range(n_layers), avg_cos_sim, color=colors)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    ax.set_title('Noise Sensitivity by Layer\n(Clean vs Noisy Cosine Similarity)', fontsize=12)
    ax.set_ylabel('Cosine Similarity\n(High = Noise Robust)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Midpoint')

    # Add trend line
    z = np.polyfit(range(n_layers), avg_cos_sim, 1)
    p = np.poly1d(z)
    ax.plot(range(n_layers), p(range(n_layers)), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
    ax.legend()

    # 2. Group statistics
    ax = axes[0, 1]
    group_names = list(LAYER_GROUPS.keys())
    group_values = [group_stats.get(g, 0) for g in group_names]
    group_colors = ['#808080', '#ADD8E6', '#87CEEB', '#4682B4', '#000080', '#191970']

    bars = ax.bar(group_names, group_values, color=group_colors)
    ax.set_title('Noise Sensitivity by Layer Group', fontsize=12)
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_ylim(0, 1)
    for i, v in enumerate(group_values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # 3. Health indicators
    ax = axes[1, 0]

    # Calculate health metrics
    shallow_sim = group_stats.get('low_level', 0.5)
    deep_sim = group_stats.get('abstract', 0.5)

    health_metrics = {
        'Shallow Noise\nSensitivity\n(Expected: Low)': 1 - shallow_sim,
        'Deep Layer\nRobustness\n(Expected: High)': deep_sim,
        'Deep-Shallow\nDifference\n(Expected: Positive)': deep_sim - shallow_sim,
    }

    colors = ['#FF6B6B' if v < 0.3 else '#4ECDC4' if v > 0.5 else '#FFE66D'
              for v in health_metrics.values()]

    bars = ax.barh(list(health_metrics.keys()), list(health_metrics.values()), color=colors)
    ax.set_xlim(-0.5, 1)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title('Model Health Indicators (Noise Handling)', fontsize=12)

    for i, (k, v) in enumerate(health_metrics.items()):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)

    # 4. Diagnostic summary
    ax = axes[1, 1]
    ax.axis('off')

    diagnosis = []
    diagnosis.append("=" * 40)
    diagnosis.append("Model Health Report (Noise Handling)")
    diagnosis.append("=" * 40)

    # Check 1: Shallow layers noise sensitive
    if shallow_sim < 0.5:
        diagnosis.append("\n[!] Shallow layers noise-sensitive (Normal)")
        diagnosis.append(f"    Shallow similarity = {shallow_sim:.3f}")
        diagnosis.append("    -> Acoustic features affected by noise")
    else:
        diagnosis.append("\n[!] Shallow layers noise-robust (Unexpected)")
        diagnosis.append(f"    Shallow similarity = {shallow_sim:.3f}")
        diagnosis.append("    -> May not be processing acoustic features")

    # Check 2: Deep layers noise robust
    if deep_sim > 0.5:
        diagnosis.append("\n[v] Deep layers noise-robust (Normal)")
        diagnosis.append(f"    Deep similarity = {deep_sim:.3f}")
    else:
        diagnosis.append("\n[!] Deep layers noise-sensitive (Abnormal)")
        diagnosis.append(f"    Deep similarity = {deep_sim:.3f}")
        diagnosis.append("    -> Semantic representations may be corrupted")

    # Check 3: Deep-shallow difference
    diff = deep_sim - shallow_sim
    if diff > 0.1:
        diagnosis.append(f"\n[v] Layer hierarchy (diff = {diff:.3f})")
        diagnosis.append("    -> Proper hierarchical structure")
    else:
        diagnosis.append(f"\n[!] Layer hierarchy weak (diff = {diff:.3f})")
        diagnosis.append("    -> All layers may process similar features")

    ax.text(0.05, 0.95, '\n'.join(diagnosis), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "outputs" / "noise_sensitivity_results.json"
    output_path = Path(__file__).parent / "outputs" / "noise_sensitivity_analysis_en.png"

    visualize_noise_sensitivity_en(results_path, output_path)
    print("Done!")
