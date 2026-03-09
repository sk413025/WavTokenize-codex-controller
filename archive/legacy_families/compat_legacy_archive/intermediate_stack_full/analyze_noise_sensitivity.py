"""
分析噪音對各層特徵的影響 (原始模型，無 LoRA)

測量: 同一模型，Clean input vs Noisy input 的特徵差異
這才是真正的「噪音敏感度」分析
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from decoder.pretrained import WavTokenizer
from families.compat_legacy.curriculum_data.data_curriculum import create_curriculum_dataloaders


class OriginalModelWithIntermediate:
    """
    原始 WavTokenizer，可以提取中間層輸出
    """
    def __init__(self, config_path, ckpt_path, device='cuda'):
        self.device = device
        self.model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
        self.model.eval()
        self.model.to(device)

        # 凍結所有參數
        for param in self.model.parameters():
            param.requires_grad = False

        self.encoder = self.model.feature_extractor.encodec.encoder

    @torch.no_grad()
    def extract_all_layers(self, audio):
        """
        提取所有層的輸出

        Args:
            audio: (B, 1, T) or (B, T)

        Returns:
            intermediates: {layer_idx: (B, C, T')}
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        x = audio
        intermediates = {}

        for i, layer in enumerate(self.encoder.model):
            x = layer(x)
            intermediates[i] = x.clone()

        return intermediates


def compute_noise_sensitivity(model, dataloader, device, max_batches=50):
    """
    計算噪音對各層的影響

    測量: cos_sim(feature(clean), feature(noisy))
    """
    layer_cos_sims = {i: [] for i in range(16)}
    layer_mse = {i: [] for i in range(16)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing noise sensitivity")):
            if batch_idx >= max_batches:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # 提取特徵
            clean_features = model.extract_all_layers(clean_audio)
            noisy_features = model.extract_all_layers(noisy_audio)

            # 計算每層的差異
            for layer_idx in clean_features.keys():
                clean_feat = clean_features[layer_idx]
                noisy_feat = noisy_features[layer_idx]

                # 確保維度匹配
                min_t = min(clean_feat.shape[-1], noisy_feat.shape[-1])
                clean_feat = clean_feat[..., :min_t]
                noisy_feat = noisy_feat[..., :min_t]

                B, C, T = clean_feat.shape

                # Cosine Similarity
                c_flat = clean_feat.permute(0, 2, 1).reshape(-1, C)
                n_flat = noisy_feat.permute(0, 2, 1).reshape(-1, C)
                cos_sim = F.cosine_similarity(c_flat, n_flat, dim=1).mean().item()
                layer_cos_sims[layer_idx].append(cos_sim)

                # MSE
                mse = F.mse_loss(clean_feat, noisy_feat).item()
                layer_mse[layer_idx].append(mse)

    results = {
        'cos_sim': {k: np.mean(v) for k, v in layer_cos_sims.items() if v},
        'mse': {k: np.mean(v) for k, v in layer_mse.items() if v},
    }

    return results


def plot_comparison(orig_results, trained_results, save_path):
    """
    比較原始模型的噪音敏感度 vs 訓練後的 Student-Teacher 距離
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = sorted(orig_results['cos_sim'].keys())

    # 1. 原始模型噪音敏感度 (Clean vs Noisy)
    ax1 = axes[0, 0]
    orig_cos = [orig_results['cos_sim'][l] for l in layers]
    colors1 = ['#d62728' if c < 0.5 else '#2ca02c' if c > 0.7 else '#ff7f0e' for c in orig_cos]
    ax1.bar(layers, orig_cos, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Layer Index (encoder.model[i])')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Original Model: Clean vs Noisy Features\n(Lower = More Noise Sensitive)')
    ax1.axhline(y=np.mean(orig_cos), color='gray', linestyle='--', label=f'Mean: {np.mean(orig_cos):.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 標記最敏感的層
    min_idx = np.argmin(orig_cos)
    ax1.annotate(f'Most Sensitive\nL{layers[min_idx]}: {orig_cos[min_idx]:.3f}',
                xy=(layers[min_idx], orig_cos[min_idx]),
                xytext=(layers[min_idx]+2, orig_cos[min_idx]+0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # 2. 訓練後 Student-Teacher 距離
    ax2 = axes[0, 1]
    if trained_results:
        trained_cos = [trained_results['cos_sim'].get(str(l), 0) for l in layers]
        colors2 = ['#d62728' if c < 0.3 else '#2ca02c' if c > 0.5 else '#ff7f0e' for c in trained_cos]
        ax2.bar(layers, trained_cos, color=colors2, edgecolor='black', linewidth=0.5)
        ax2.set_title('After Training: Student vs Teacher Features\n(Lower = LoRA learned less)')
    else:
        ax2.text(0.5, 0.5, 'No trained model data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Noise Sensitivity Loss (1 - cos_sim) for supervision design
    ax3 = axes[1, 0]
    noise_loss = [1 - c for c in orig_cos]
    ax3.bar(layers, noise_loss, color=colors1, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('1 - Cosine Similarity')
    ax3.set_title('Noise Sensitivity Loss (Potential Supervision Weight)\n(Higher = Should supervise more)')
    ax3.grid(axis='y', alpha=0.3)

    # 標記建議的監督位置
    # 找到 noise_loss 最高的 4 個層
    sorted_layers = sorted(enumerate(noise_loss), key=lambda x: x[1], reverse=True)
    top_4 = sorted_layers[:4]
    for idx, (layer_idx, loss_val) in enumerate(top_4):
        ax3.annotate(f'#{idx+1}',
                    xy=(layer_idx, loss_val),
                    xytext=(layer_idx, loss_val + 0.03),
                    ha='center', fontsize=10, fontweight='bold', color='red')

    # 4. Layer Group Summary
    ax4 = axes[1, 1]
    layer_groups = {
        'input\n(L0)': [0],
        'low_level\n(L1-4)': [1, 2, 3, 4],
        'mid_level\n(L5-8)': [5, 6, 7, 8],
        'semantic\n(L9-12)': [9, 10, 11, 12],
        'abstract\n(L13-15)': [13, 14, 15],
    }

    group_names = list(layer_groups.keys())
    group_cos = []
    for name, indices in layer_groups.items():
        vals = [orig_results['cos_sim'][i] for i in indices if i in orig_results['cos_sim']]
        group_cos.append(np.mean(vals) if vals else 0)

    colors4 = ['#d62728' if c < 0.5 else '#2ca02c' if c > 0.7 else '#ff7f0e' for c in group_cos]
    bars = ax4.bar(group_names, group_cos, color=colors4, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Avg Cosine Similarity (Clean vs Noisy)')
    ax4.set_title('Noise Sensitivity by Layer Group\n(Lower = More Sensitive)')
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, group_cos):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_supervision_recommendation(orig_results, save_path):
    """
    基於噪音敏感度推薦監督位置
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = sorted(orig_results['cos_sim'].keys())
    noise_loss = [1 - orig_results['cos_sim'][l] for l in layers]

    # 繪製噪音敏感度
    ax.fill_between(layers, noise_loss, alpha=0.3, color='gray', label='Noise Sensitivity')
    ax.plot(layers, noise_loss, 'k--', alpha=0.7)

    # 當前 Exp K v2 的監督位置 (L3, L6)
    current_supervised = [3, 6]
    for sl in current_supervised:
        if sl in layers:
            ax.scatter([sl], [noise_loss[sl]], s=200, c='blue', marker='o',
                      edgecolors='black', linewidth=2, zorder=5,
                      label=f'Current: L{sl}' if sl == current_supervised[0] else '')

    # 找到真正最敏感的層 (排除 L0 因為太早)
    sorted_by_sensitivity = sorted(enumerate(noise_loss), key=lambda x: x[1], reverse=True)
    recommended = []
    for idx, (layer_idx, _) in enumerate(sorted_by_sensitivity):
        if layer_idx >= 1:  # 排除 L0
            recommended.append(layer_idx)
        if len(recommended) >= 4:
            break

    for i, sl in enumerate(recommended):
        marker = '*' if i == 0 else 'o'
        ax.scatter([sl], [noise_loss[sl]], s=150, c='red', marker=marker,
                  edgecolors='black', linewidth=1, zorder=5,
                  label=f'Recommended #{i+1}: L{sl}' if i < 2 else '')

    # 標記層組
    ax.axvspan(1, 4, alpha=0.1, color='green', label='low_level')
    ax.axvspan(5, 8, alpha=0.1, color='orange', label='mid_level')
    ax.axvspan(9, 12, alpha=0.1, color='purple', label='semantic')

    ax.set_xlabel('Layer Index (encoder.model[i])')
    ax.set_ylabel('Noise Sensitivity (1 - CosSim)')
    ax.set_title('Supervision Location Recommendation Based on Noise Sensitivity')
    ax.set_xticks(layers)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入原始模型（無 LoRA）
    print("Loading original model (no LoRA)...")
    model = OriginalModelWithIntermediate(WAVTOK_CONFIG, WAVTOK_CKPT, str(device))

    # 創建 dataloader
    print("Creating dataloader...")
    _, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=VAL_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=4,
        num_workers=4,
        initial_phase=1.0,
    )

    # 計算噪音敏感度
    print("Computing noise sensitivity...")
    orig_results = compute_noise_sensitivity(model, val_loader, device, max_batches=50)

    # 載入之前的訓練後結果
    trained_results_path = Path('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis/layer_distances.json')
    trained_results = None
    if trained_results_path.exists():
        with open(trained_results_path) as f:
            trained_results = json.load(f)

    # 保存結果
    output_dir = Path('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'noise_sensitivity.json', 'w') as f:
        json_results = {
            metric: {str(k): v for k, v in values.items()}
            for metric, values in orig_results.items()
        }
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir / 'noise_sensitivity.json'}")

    # 繪製比較圖
    plot_comparison(orig_results, trained_results, output_dir / 'noise_sensitivity_comparison.png')
    plot_supervision_recommendation(orig_results, output_dir / 'supervision_recommendation.png')

    # 打印結果
    print("\n" + "="*60)
    print("Noise Sensitivity Analysis (Original Model)")
    print("="*60)

    print("\nCosine Similarity (Clean vs Noisy) - Lower = More Sensitive:")
    for layer in sorted(orig_results['cos_sim'].keys()):
        cos_sim = orig_results['cos_sim'][layer]
        sensitivity = 1 - cos_sim
        print(f"  L{layer:2d}: cos_sim={cos_sim:.4f}, sensitivity={sensitivity:.4f}")

    print("\n" + "-"*60)
    print("Layer Group Summary:")
    layer_groups = {
        'input (L0)': [0],
        'low_level (L1-4)': [1, 2, 3, 4],
        'mid_level (L5-8)': [5, 6, 7, 8],
        'semantic (L9-12)': [9, 10, 11, 12],
        'abstract (L13-15)': [13, 14, 15],
    }

    for group_name, group_layers in layer_groups.items():
        group_cos = [orig_results['cos_sim'][l] for l in group_layers if l in orig_results['cos_sim']]
        if group_cos:
            avg_cos = np.mean(group_cos)
            avg_sens = 1 - avg_cos
            print(f"  {group_name:20s}: avg_cos_sim={avg_cos:.4f}, avg_sensitivity={avg_sens:.4f}")

    # 推薦監督層
    print("\n" + "-"*60)
    print("Recommended Supervision Layers (by sensitivity):")
    sorted_layers = sorted(orig_results['cos_sim'].items(), key=lambda x: x[1])
    for i, (layer, cos_sim) in enumerate(sorted_layers[:5]):
        print(f"  #{i+1}: L{layer} (cos_sim={cos_sim:.4f}, sensitivity={1-cos_sim:.4f})")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
