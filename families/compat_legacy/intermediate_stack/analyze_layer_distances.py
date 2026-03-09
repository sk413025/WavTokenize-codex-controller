"""
分析各層 Student-Teacher 特徵距離

目標:
1. 計算每層的 CosSim 和 MSE 距離
2. 繪製距離分布圖
3. 分析噪音對各層的影響
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
from families.compat_legacy.intermediate_stack.models import TeacherStudentIntermediate
from families.compat_legacy.curriculum_data.data_curriculum import create_curriculum_dataloaders


def compute_layer_distances(model, dataloader, device, max_batches=50):
    """計算各層的 Student-Teacher 距離"""
    model.eval()

    # 收集所有層的距離
    layer_cos_sims = {i: [] for i in range(18)}  # L0-L17
    layer_mse = {i: [] for i in range(18)}
    layer_l2_dist = {i: [] for i in range(18)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing distances")):
            if batch_idx >= max_batches:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # 獲取所有中間層輸出
            output = model(noisy_audio, clean_audio)

            student_intermediates = output['student_intermediates']
            teacher_intermediates = output['teacher_intermediates']

            # 計算每層的距離
            for layer_idx in student_intermediates.keys():
                s_feat = student_intermediates[layer_idx]  # (B, C, T)
                t_feat = teacher_intermediates[layer_idx]  # (B, C, T)

                # 確保維度匹配
                min_t = min(s_feat.shape[-1], t_feat.shape[-1])
                s_feat = s_feat[..., :min_t]
                t_feat = t_feat[..., :min_t]

                B, C, T = s_feat.shape

                # 1. Cosine Similarity (per time step)
                s_flat = s_feat.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
                t_flat = t_feat.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
                cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean().item()
                layer_cos_sims[layer_idx].append(cos_sim)

                # 2. MSE (normalized by dimension)
                mse = F.mse_loss(s_feat, t_feat).item()
                layer_mse[layer_idx].append(mse)

                # 3. L2 distance (normalized)
                l2_dist = torch.norm(s_feat - t_feat, p=2) / (B * C * T)
                layer_l2_dist[layer_idx].append(l2_dist.item())

    # 計算平均值
    results = {
        'cos_sim': {k: np.mean(v) for k, v in layer_cos_sims.items() if v},
        'mse': {k: np.mean(v) for k, v in layer_mse.items() if v},
        'l2_dist': {k: np.mean(v) for k, v in layer_l2_dist.items() if v},
    }

    return results


def plot_layer_distances(results, save_path):
    """繪製各層距離分析圖"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = sorted(results['cos_sim'].keys())
    cos_sims = [results['cos_sim'][l] for l in layers]
    mses = [results['mse'][l] for l in layers]
    l2_dists = [results['l2_dist'][l] for l in layers]

    # 定義層組
    layer_groups = {
        'input': [0],
        'low_level': [1, 2, 3, 4],
        'mid_level': [5, 6, 7, 8],
        'semantic': [9, 10, 11, 12],
        'abstract': [13, 14, 15, 16],
        'output': [17]
    }

    colors = []
    for l in layers:
        if l in layer_groups['input']:
            colors.append('#1f77b4')  # blue
        elif l in layer_groups['low_level']:
            colors.append('#2ca02c')  # green
        elif l in layer_groups['mid_level']:
            colors.append('#d62728')  # red (noise sensitive)
        elif l in layer_groups['semantic']:
            colors.append('#ff7f0e')  # orange
        elif l in layer_groups['abstract']:
            colors.append('#9467bd')  # purple
        else:
            colors.append('#8c564b')  # brown

    # 1. Cosine Similarity
    ax1 = axes[0, 0]
    bars1 = ax1.bar(layers, cos_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Student-Teacher Cosine Similarity by Layer\n(Higher = More Similar)')
    ax1.set_xticks(layers)
    ax1.axhline(y=np.mean(cos_sims), color='gray', linestyle='--', label=f'Mean: {np.mean(cos_sims):.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 標記最敏感的層
    min_cos_idx = np.argmin(cos_sims)
    ax1.annotate(f'Most Affected\nL{layers[min_cos_idx]}: {cos_sims[min_cos_idx]:.3f}',
                xy=(layers[min_cos_idx], cos_sims[min_cos_idx]),
                xytext=(layers[min_cos_idx]+2, cos_sims[min_cos_idx]-0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # 2. MSE Distance
    ax2 = axes[0, 1]
    bars2 = ax2.bar(layers, mses, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('MSE')
    ax2.set_title('Student-Teacher MSE by Layer\n(Lower = More Similar)')
    ax2.set_xticks(layers)
    ax2.set_yscale('log')  # log scale for MSE
    ax2.grid(axis='y', alpha=0.3)

    # 3. Cosine Similarity Loss (1 - cos_sim)
    ax3 = axes[1, 0]
    cos_losses = [1 - cs for cs in cos_sims]
    bars3 = ax3.bar(layers, cos_losses, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('1 - Cosine Similarity')
    ax3.set_title('Cosine Loss by Layer (Potential Supervision Target)\n(Higher = More Different)')
    ax3.set_xticks(layers)
    ax3.grid(axis='y', alpha=0.3)

    # 標記中間層監督位置
    supervised_layers = [3, 6]  # Exp K v2 監督的層
    for sl in supervised_layers:
        if sl in layers:
            idx = layers.index(sl)
            ax3.annotate(f'Supervised\nL{sl}',
                        xy=(sl, cos_losses[idx]),
                        xytext=(sl, cos_losses[idx]+0.05),
                        ha='center',
                        fontsize=8,
                        color='blue',
                        fontweight='bold')

    # 4. Layer Group Summary
    ax4 = axes[1, 1]
    group_names = list(layer_groups.keys())
    group_cos_sims = []
    group_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']

    for group_name in group_names:
        group_layers = layer_groups[group_name]
        group_values = [results['cos_sim'][l] for l in group_layers if l in results['cos_sim']]
        group_cos_sims.append(np.mean(group_values) if group_values else 0)

    bars4 = ax4.bar(group_names, group_cos_sims, color=group_colors, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Layer Group')
    ax4.set_ylabel('Average Cosine Similarity')
    ax4.set_title('Student-Teacher Similarity by Layer Group')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)

    # 添加數值標籤
    for bar, val in zip(bars4, group_cos_sims):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return fig


def plot_loss_contribution_analysis(results, save_path):
    """繪製 Loss 貢獻分析圖"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = sorted(results['cos_sim'].keys())
    cos_losses = [1 - results['cos_sim'][l] for l in layers]

    # 1. 當前 Exp K v2 的 Loss 設計
    ax1 = axes[0]

    # 定義各 Loss 位置和權重
    loss_positions = {
        'Feature Loss (L17)': {'layer': 17, 'weight': 1.0, 'color': '#1f77b4'},
        'Triplet Loss (L17)': {'layer': 17, 'weight': 1.0, 'color': '#ff7f0e'},
        'Intermediate L3': {'layer': 3, 'weight': 0.5, 'color': '#2ca02c'},
        'Intermediate L6': {'layer': 6, 'weight': 0.5, 'color': '#d62728'},
    }

    # 繪製層的 cosine loss 作為背景
    ax1.fill_between(layers, cos_losses, alpha=0.3, color='gray', label='Layer Noise Sensitivity')
    ax1.plot(layers, cos_losses, 'k--', alpha=0.5)

    # 標記各 Loss 位置
    for loss_name, info in loss_positions.items():
        layer = info['layer']
        if layer in layers:
            idx = layers.index(layer)
            ax1.scatter([layer], [cos_losses[idx]], s=200*info['weight'],
                       c=info['color'], marker='o', edgecolors='black',
                       linewidth=2, label=f"{loss_name} (w={info['weight']})", zorder=5)

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Loss (1 - cos_sim)')
    ax1.set_title('Current Loss Design vs Layer Noise Sensitivity')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xticks(layers)
    ax1.grid(alpha=0.3)

    # 2. 建議的 Loss 設計
    ax2 = axes[1]

    # 找出最敏感的層
    sorted_layers = sorted(enumerate(cos_losses), key=lambda x: x[1], reverse=True)
    top_sensitive = sorted_layers[:5]  # 前5個最敏感的層

    ax2.fill_between(layers, cos_losses, alpha=0.3, color='gray', label='Layer Noise Sensitivity')
    ax2.plot(layers, cos_losses, 'k--', alpha=0.5)

    # 標記建議的監督位置
    suggested_positions = {
        'Suggested L5': {'layer': 5, 'color': '#e377c2'},
        'Suggested L6': {'layer': 6, 'color': '#d62728'},
        'Suggested L7': {'layer': 7, 'color': '#bcbd22'},
        'Suggested L8': {'layer': 8, 'color': '#17becf'},
    }

    for loss_name, info in suggested_positions.items():
        layer = info['layer']
        if layer in layers:
            idx = layers.index(layer)
            ax2.scatter([layer], [cos_losses[idx]], s=150,
                       c=info['color'], marker='*', edgecolors='black',
                       linewidth=1, label=loss_name, zorder=5)

    # 標記最敏感區域
    ax2.axvspan(5, 8, alpha=0.2, color='red', label='Most Noise-Sensitive Zone')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Loss (1 - cos_sim)')
    ax2.set_title('Suggested: Focus Supervision on L5-L8')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xticks(layers)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入模型
    print("Loading model...")
    # 提取所有中間層 (model index 0-17 對應 encoder 內部的層)
    # 注意：model index 對應 encoder.model[i]
    all_layer_indices = list(range(16))  # encoder.model 有 0-15
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        intermediate_indices=all_layer_indices,
        device=str(device),
    )

    # 載入訓練好的權重 (如果有)
    checkpoint_path = Path('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/runs/exp_k_v2_20260115_020445/best_model.pt')
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 創建 dataloader
    print("Creating dataloader...")
    _, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=VAL_CACHE,  # 使用 val 數據
        val_cache_path=VAL_CACHE,
        batch_size=4,
        num_workers=4,
        initial_phase=1.0,  # 全噪音
    )

    # 計算距離
    print("Computing layer distances...")
    results = compute_layer_distances(model, val_loader, device, max_batches=50)

    # 保存結果
    output_dir = Path('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'layer_distances.json', 'w') as f:
        # Convert keys to strings for JSON
        json_results = {
            metric: {str(k): v for k, v in values.items()}
            for metric, values in results.items()
        }
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir / 'layer_distances.json'}")

    # 繪製圖表
    plot_layer_distances(results, output_dir / 'layer_distances.png')
    plot_loss_contribution_analysis(results, output_dir / 'loss_design_analysis.png')

    # 打印結果
    print("\n" + "="*60)
    print("Layer Distance Analysis Results")
    print("="*60)

    print("\nCosine Similarity (Higher = More Similar):")
    for layer in sorted(results['cos_sim'].keys()):
        cos_sim = results['cos_sim'][layer]
        cos_loss = 1 - cos_sim
        print(f"  L{layer:2d}: cos_sim={cos_sim:.4f}, cos_loss={cos_loss:.4f}")

    print("\n" + "-"*60)
    print("Layer Group Summary:")
    layer_groups = {
        'input (L0)': [0],
        'low_level (L1-4)': [1, 2, 3, 4],
        'mid_level (L5-8)': [5, 6, 7, 8],
        'semantic (L9-12)': [9, 10, 11, 12],
        'abstract (L13-16)': [13, 14, 15, 16],
        'output (L17)': [17]
    }

    for group_name, group_layers in layer_groups.items():
        group_cos_sims = [results['cos_sim'][l] for l in group_layers if l in results['cos_sim']]
        if group_cos_sims:
            avg_cos = np.mean(group_cos_sims)
            avg_loss = 1 - avg_cos
            print(f"  {group_name:20s}: avg_cos_sim={avg_cos:.4f}, avg_cos_loss={avg_loss:.4f}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
