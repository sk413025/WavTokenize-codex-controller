#!/usr/bin/env python3
"""
t-SNE 特徵空間可視化分析

目的: 驗證 Student Features 是否隨訓練向 Teacher Features 靠近

分析內容:
1. 固定 30 筆音檔，每 10 epoch 提取特徵
2. 使用 t-SNE 投影到 2D 空間
3. 觀察 Student Features 軌跡是否向 Teacher Features 收斂

使用方式:
    python feature_tsne_analysis.py --exp_name ste_baseline --num_samples 30
    python feature_tsne_analysis.py --exp_name ce_balanced --num_samples 30
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import json

# 添加路徑 - 注意順序：exp_1201 必須在 WavTokenizer 之前
exp_1201_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, exp_1201_dir)

from config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from model import TeacherStudentModel
from data import NoisyCleanPairDataset

# WavTokenizer 路徑（在 model.py 中已處理）


def load_checkpoint(checkpoint_path, device='cuda'):
    """載入 checkpoint 並返回模型"""
    print(f"Loading checkpoint: {checkpoint_path}")

    # 創建模型
    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,  # 與訓練一致
        lora_alpha=128,
        lora_dropout=0.1,
        device=device,
    )

    # 載入 checkpoint (PyTorch 2.6+ 需要 weights_only=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    return model


def extract_features(model, audio_samples, device='cuda'):
    """
    從模型提取特徵

    Args:
        model: TeacherStudentModel
        audio_samples: list of (noisy_audio, clean_audio) tuples

    Returns:
        student_features: (N, D) - 平均池化後的特徵
        teacher_features: (N, D)
    """
    student_feats = []
    teacher_feats = []

    model.eval()
    with torch.no_grad():
        for noisy_audio, clean_audio in tqdm(audio_samples, desc="Extracting features"):
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)

            # 確保格式正確 (B, T)
            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0)

            # 提取特徵
            output = model(noisy_audio, clean_audio)

            # 取平均池化 (B, D, T) -> (B, D)
            sf = output['student_features'].mean(dim=-1)  # (1, 512)
            tf = output['teacher_features'].mean(dim=-1)  # (1, 512)

            student_feats.append(sf.cpu().numpy())
            teacher_feats.append(tf.cpu().numpy())

    student_features = np.concatenate(student_feats, axis=0)  # (N, 512)
    teacher_features = np.concatenate(teacher_feats, axis=0)  # (N, 512)

    return student_features, teacher_features


def compute_feature_distances(student_features, teacher_features):
    """計算 Student 和 Teacher 特徵之間的距離統計"""
    # L2 距離
    l2_distances = np.linalg.norm(student_features - teacher_features, axis=1)

    # Cosine similarity
    s_norm = student_features / (np.linalg.norm(student_features, axis=1, keepdims=True) + 1e-8)
    t_norm = teacher_features / (np.linalg.norm(teacher_features, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.sum(s_norm * t_norm, axis=1)

    return {
        'l2_mean': np.mean(l2_distances),
        'l2_std': np.std(l2_distances),
        'l2_min': np.min(l2_distances),
        'l2_max': np.max(l2_distances),
        'cosine_mean': np.mean(cosine_sim),
        'cosine_std': np.std(cosine_sim),
    }


def main():
    parser = argparse.ArgumentParser(description='t-SNE Feature Analysis')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name (e.g., ste_baseline, ce_balanced)')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of audio samples to analyze')
    parser.add_argument('--use_val', action='store_true',
                       help='Use validation set instead of training set')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--perplexity', type=int, default=15,
                       help='t-SNE perplexity')
    args = parser.parse_args()

    # 設置路徑
    exp_dir = Path(__file__).parent.parent / 'experiments' / args.exp_name
    output_dir = Path(__file__).parent / 'results' / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    # 載入數據 - 使用 Dataset 類來處理音頻載入
    print(f"\n{'='*60}")
    print(f"Loading data...")
    print(f"{'='*60}")

    cache_path = VAL_CACHE if args.use_val else TRAIN_CACHE
    dataset = NoisyCleanPairDataset(cache_path)
    print(f"Loaded {len(dataset)} samples from {cache_path}")

    # 固定選擇樣本 (使用固定 seed 確保可重現)
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    audio_samples = []
    for i in sample_indices:
        sample = dataset[i]
        audio_samples.append((sample['noisy_audio'], sample['clean_audio']))
    print(f"Selected {len(audio_samples)} samples for analysis")

    # 找到所有 checkpoint
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoints = sorted(checkpoint_dir.glob('epoch_*_loss_*.pt'))

    if not checkpoints:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        print("Available files:", list(checkpoint_dir.glob('*')))
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    # 提取每個 epoch 的特徵
    all_student_features = {}
    all_teacher_features = {}
    distance_stats = {}

    for ckpt_path in checkpoints:
        # 解析 epoch 數
        epoch_str = ckpt_path.stem.split('_')[1]
        epoch = int(epoch_str)

        print(f"\n{'='*60}")
        print(f"Processing Epoch {epoch}")
        print(f"{'='*60}")

        # 載入模型
        model = load_checkpoint(ckpt_path, device=args.device)

        # 提取特徵
        student_feats, teacher_feats = extract_features(model, audio_samples, device=args.device)

        all_student_features[epoch] = student_feats
        all_teacher_features[epoch] = teacher_feats

        # 計算距離統計
        stats = compute_feature_distances(student_feats, teacher_feats)
        distance_stats[epoch] = stats

        print(f"  L2 Distance: {stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}")
        print(f"  Cosine Similarity: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")

        # 釋放 GPU 記憶體
        del model
        torch.cuda.empty_cache()

    # 使用最後一個 epoch 的 teacher features 作為參考
    epochs = sorted(all_student_features.keys())
    reference_teacher = all_teacher_features[epochs[-1]]

    # 合併所有特徵用於 t-SNE
    print(f"\n{'='*60}")
    print("Running t-SNE...")
    print(f"{'='*60}")

    all_features = []
    labels = []  # (epoch, 'student'/'teacher', sample_idx)

    for epoch in epochs:
        for i in range(len(audio_samples)):
            all_features.append(all_student_features[epoch][i])
            labels.append((epoch, 'student', i))

    # 添加 teacher features (只用一個 epoch 的，因為 teacher 是固定的)
    for i in range(len(audio_samples)):
        all_features.append(reference_teacher[i])
        labels.append((epochs[-1], 'teacher', i))

    all_features = np.array(all_features)
    print(f"Total features for t-SNE: {all_features.shape}")

    # 運行 t-SNE (新版 sklearn 用 max_iter 取代 n_iter)
    try:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)

    # 繪圖
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    # 獲取 teacher features 的位置（所有 epoch 共用的參考點）
    teacher_mask = [l[1] == 'teacher' for l in labels]
    teacher_points = features_2d[teacher_mask]

    # 計算統一的座標範圍（所有圖使用相同範圍以便比較）
    all_x = features_2d[:, 0]
    all_y = features_2d[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
    ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    # 圖1: 每個 epoch 獨立的 t-SNE 圖
    for idx, epoch in enumerate(epochs):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 繪製 teacher features (紅色星星，作為參考)
        ax.scatter(teacher_points[:, 0], teacher_points[:, 1], c='red', marker='*',
                  s=200, label='Teacher (Target)', edgecolors='black', linewidth=0.5, zorder=10)

        # 繪製該 epoch 的 student features
        student_mask = [(l[0] == epoch and l[1] == 'student') for l in labels]
        student_points = features_2d[student_mask]
        ax.scatter(student_points[:, 0], student_points[:, 1], c='blue',
                  alpha=0.7, s=60, label=f'Student Epoch {epoch}', edgecolors='white', linewidth=0.3)

        # 繪製每個 student 到對應 teacher 的連線
        for i in range(len(audio_samples)):
            ax.plot([student_points[i, 0], teacher_points[i, 0]],
                   [student_points[i, 1], teacher_points[i, 1]],
                   'gray', alpha=0.3, linewidth=0.5)

        # 設置統一範圍
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # 標註距離統計
        stats = distance_stats[epoch]
        stats_text = f"L2 Distance: {stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}\n"
        stats_text += f"Cosine Sim: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f't-SNE Feature Space: Epoch {epoch}\n{args.exp_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        tsne_path = output_dir / f'tsne_epoch_{epoch:03d}.png'
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved t-SNE plot: {tsne_path}")

    # 圖2: 所有 epoch 的彙總軌跡圖
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(epochs)))

    # 繪製 teacher features (參考點)
    ax.scatter(teacher_points[:, 0], teacher_points[:, 1], c='red', marker='*',
              s=200, label='Teacher (Target)', edgecolors='black', linewidth=0.5, zorder=10)

    # 繪製每個 epoch 的 student features
    for idx, epoch in enumerate(epochs):
        mask = [(l[0] == epoch and l[1] == 'student') for l in labels]
        points = features_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], c=[colors[idx]],
                  label=f'Student Epoch {epoch}', alpha=0.7, s=40, edgecolors='white', linewidth=0.2)

    # 繪製部分樣本的軌跡箭頭
    for i in range(min(5, len(audio_samples))):
        trajectory = []
        for epoch in epochs:
            mask = [(l[0] == epoch and l[1] == 'student' and l[2] == i) for l in labels]
            idx_list = [j for j, m in enumerate(mask) if m]
            if idx_list:
                trajectory.append(features_2d[idx_list[0]])

        trajectory = np.array(trajectory)
        for j in range(len(trajectory) - 1):
            ax.annotate('', xy=trajectory[j+1], xytext=trajectory[j],
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.5, lw=1.5))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f't-SNE Feature Trajectory: {args.exp_name}\n'
                f'(Arrows show feature movement over training)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_path = output_dir / 'tsne_trajectory_all.png'
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory plot: {tsne_path}")

    # 圖3: 距離變化 - 更直觀的長條圖比較
    epochs_list = sorted(distance_stats.keys())
    l2_means = [distance_stats[e]['l2_mean'] for e in epochs_list]
    l2_stds = [distance_stats[e]['l2_std'] for e in epochs_list]
    cosine_means = [distance_stats[e]['cosine_mean'] for e in epochs_list]
    cosine_stds = [distance_stats[e]['cosine_std'] for e in epochs_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 顏色漸變：越晚的 epoch 顏色越深
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(epochs_list)))

    # L2 距離 - 長條圖
    ax = axes[0]
    x_pos = np.arange(len(epochs_list))
    bars = ax.bar(x_pos, l2_means, yerr=l2_stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=1)

    # 在每個長條上方標註數值
    for i, (bar, val) in enumerate(zip(bars, l2_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + l2_stds[i] + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 Distance (Student - Teacher)', fontsize=12)
    ax.set_title('Feature L2 Distance Over Training\n(Lower = Better)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Epoch {e}' for e in epochs_list])
    ax.grid(True, alpha=0.3, axis='y')

    # 標註變化百分比
    if len(l2_means) > 1:
        change_pct = (l2_means[-1] - l2_means[0]) / l2_means[0] * 100
        change_text = f"Change: {change_pct:+.1f}%"
        change_color = 'green' if change_pct < 0 else 'red'
        ax.text(0.95, 0.95, change_text, transform=ax.transAxes, fontsize=14,
               color=change_color, ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Cosine Similarity - 長條圖
    ax = axes[1]
    bars = ax.bar(x_pos, cosine_means, yerr=cosine_stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=1)

    # 在每個長條上方標註數值
    for i, (bar, val) in enumerate(zip(bars, cosine_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cosine_stds[i] + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Feature Cosine Similarity Over Training\n(Higher = Better)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Epoch {e}' for e in epochs_list])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # 標註變化百分比
    if len(cosine_means) > 1:
        change_pct = (cosine_means[-1] - cosine_means[0]) / cosine_means[0] * 100
        change_text = f"Change: {change_pct:+.1f}%"
        change_color = 'green' if change_pct > 0 else 'red'
        ax.text(0.95, 0.95, change_text, transform=ax.transAxes, fontsize=14,
               color=change_color, ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    dist_path = output_dir / 'feature_distance_comparison.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved distance comparison plot: {dist_path}")

    # 圖4: 綜合儀表板 - 單一圖展示核心資訊
    fig = plt.figure(figsize=(16, 10))

    # 2x2 佈局
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 左上：L2 距離趨勢線
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(epochs_list,
                     np.array(l2_means) - np.array(l2_stds),
                     np.array(l2_means) + np.array(l2_stds),
                     alpha=0.3, color='blue')
    ax1.plot(epochs_list, l2_means, 'o-', color='blue', linewidth=2, markersize=10)
    for i, (e, v) in enumerate(zip(epochs_list, l2_means)):
        ax1.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('L2 Distance', fontsize=11)
    ax1.set_title('L2 Distance (Student → Teacher)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs_list)

    # 右上：Cosine Similarity 趨勢線
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(epochs_list,
                     np.array(cosine_means) - np.array(cosine_stds),
                     np.array(cosine_means) + np.array(cosine_stds),
                     alpha=0.3, color='green')
    ax2.plot(epochs_list, cosine_means, 'o-', color='green', linewidth=2, markersize=10)
    for i, (e, v) in enumerate(zip(epochs_list, cosine_means)):
        ax2.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Cosine Similarity', fontsize=11)
    ax2.set_title('Cosine Similarity (Student vs Teacher)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.8, 1.0])
    ax2.set_xticks(epochs_list)

    # 左下：總結表格
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    # 準備表格數據
    table_data = [
        ['Metric', 'First Epoch', 'Last Epoch', 'Change'],
        ['L2 Distance', f'{l2_means[0]:.4f}', f'{l2_means[-1]:.4f}',
         f'{(l2_means[-1] - l2_means[0]) / l2_means[0] * 100:+.2f}%'],
        ['Cosine Sim', f'{cosine_means[0]:.4f}', f'{cosine_means[-1]:.4f}',
         f'{(cosine_means[-1] - cosine_means[0]) / cosine_means[0] * 100:+.2f}%'],
    ]

    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # 設置表頭樣式
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 設置 Change 列的顏色
    l2_change = (l2_means[-1] - l2_means[0]) / l2_means[0] * 100
    cos_change = (cosine_means[-1] - cosine_means[0]) / cosine_means[0] * 100
    table[(1, 3)].set_facecolor('#C6EFCE' if l2_change < 0 else '#FFC7CE')
    table[(2, 3)].set_facecolor('#C6EFCE' if cos_change > 0 else '#FFC7CE')

    ax3.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # 右下：結論文字
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # 判斷結論
    is_improving = l2_means[-1] < l2_means[0]
    conclusion_title = "CONCLUSION" if is_improving else "WARNING"
    conclusion_color = 'green' if is_improving else 'red'
    conclusion_emoji = "✓" if is_improving else "✗"

    conclusion_text = f"""
{conclusion_emoji} {conclusion_title}

L2 Distance: {"DECREASING" if is_improving else "NOT DECREASING"}
             ({l2_means[0]:.4f} → {l2_means[-1]:.4f})

Features are {"CONVERGING toward Teacher" if is_improving else "NOT converging as expected"}

Experiment: {args.exp_name}
Samples: {len(audio_samples)}
Epochs: {epochs_list[0]} → {epochs_list[-1]}
"""

    ax4.text(0.5, 0.5, conclusion_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3),
            family='monospace')

    plt.suptitle(f'Feature Analysis Dashboard: {args.exp_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    dashboard_path = output_dir / 'feature_analysis_dashboard.png'
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved dashboard: {dashboard_path}")

    # 保存統計數據 (轉換 numpy 類型為 Python 原生類型)
    stats_path = output_dir / 'feature_analysis_stats.json'

    def convert_to_native(obj):
        """遞歸轉換 numpy 類型為 Python 原生類型"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    with open(stats_path, 'w') as f:
        json.dump(convert_to_native({
            'epochs': epochs_list,
            'l2_distances': {str(e): distance_stats[e] for e in epochs_list},
            'num_samples': len(audio_samples),
            'sample_indices': sample_indices.tolist(),
        }), f, indent=2)
    print(f"  Saved stats: {stats_path}")

    # 打印總結
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment: {args.exp_name}")
    print(f"Number of samples: {len(audio_samples)}")
    print(f"Epochs analyzed: {epochs_list}")
    print(f"\nFeature Distance Trend:")
    print(f"  First Epoch ({epochs_list[0]}): L2={l2_means[0]:.4f}, Cosine={cosine_means[0]:.4f}")
    print(f"  Last Epoch ({epochs_list[-1]}): L2={l2_means[-1]:.4f}, Cosine={cosine_means[-1]:.4f}")
    print(f"  Change: L2 {'↓' if l2_means[-1] < l2_means[0] else '↑'} {abs(l2_means[-1] - l2_means[0]):.4f}")
    print(f"          Cosine {'↑' if cosine_means[-1] > cosine_means[0] else '↓'} {abs(cosine_means[-1] - cosine_means[0]):.4f}")

    if l2_means[-1] < l2_means[0] and cosine_means[-1] > cosine_means[0]:
        print("\n✅ CONCLUSION: Features ARE moving toward teacher (model is learning correctly)")
    else:
        print("\n❌ CONCLUSION: Features NOT moving toward teacher (potential issue)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
