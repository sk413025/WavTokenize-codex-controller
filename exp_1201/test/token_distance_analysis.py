#!/usr/bin/env python3
"""
Token 距離分析腳本

目的: 分析 Student Token (argmax) 與 Teacher Token 的距離關係

分析內容:
1. Student features 經過 argmin 後得到的 token 與 Teacher token 的差異
2. Token 在 codebook 距離矩陣中的距離分布
3. Top-K 準確率 (student token 是否在 teacher token 的 Top-K 最近鄰中)
4. 錯誤 token 的距離分析

使用方式:
    python token_distance_analysis.py --exp_name ste_baseline --num_samples 100
    python token_distance_analysis.py --exp_name ce_balanced --num_samples 100
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from collections import Counter

# 添加路徑 - 注意順序：exp_1201 必須在 WavTokenizer 之前
exp_1201_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, exp_1201_dir)

from config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE, DISTANCE_MATRIX
from model import TeacherStudentModel
from data import NoisyCleanPairDataset

# WavTokenizer 路徑（在 model.py 中已處理）


def load_checkpoint(checkpoint_path, device='cuda'):
    """載入 checkpoint 並返回模型"""
    print(f"Loading checkpoint: {checkpoint_path}")

    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.1,
        device=device,
    )

    # PyTorch 2.6+ 需要 weights_only=False
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    return model


def extract_tokens_and_features(model, audio_samples, codebook, device='cuda'):
    """
    提取 tokens 和計算 argmin 結果

    Returns:
        student_tokens: student 預測的 tokens (通過 argmin)
        teacher_tokens: teacher 的 tokens
        student_features: student 的連續特徵
        teacher_features: teacher 的連續特徵
    """
    all_student_tokens = []
    all_teacher_tokens = []
    all_student_features = []
    all_teacher_features = []

    model.eval()
    with torch.no_grad():
        for noisy_audio, clean_audio in tqdm(audio_samples, desc="Extracting tokens"):
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)

            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0)

            output = model(noisy_audio, clean_audio)

            # 獲取特徵 (B, D, T)
            student_feats = output['student_features']  # (1, 512, T)
            teacher_feats = output['teacher_features']  # (1, 512, T)

            # Teacher tokens (從模型 VQ 層獲得)
            teacher_tokens = output['teacher_codes'].squeeze()  # (T,) or (1, T)
            if teacher_tokens.dim() == 2:
                teacher_tokens = teacher_tokens.squeeze(0)

            # Student tokens: 手動計算 argmin (模擬 VQ 選擇)
            # student_feats: (1, 512, T) -> (T, 512)
            sf = student_feats.squeeze(0).transpose(0, 1)  # (T, 512)

            # 計算到所有 codebook entries 的距離
            # codebook: (4096, 512)
            # distances: (T, 4096)
            distances = torch.cdist(sf, codebook)  # (T, 4096)

            # argmin 得到 student tokens
            student_tokens = distances.argmin(dim=1)  # (T,)

            all_student_tokens.append(student_tokens.cpu().numpy())
            all_teacher_tokens.append(teacher_tokens.cpu().numpy())
            all_student_features.append(sf.cpu().numpy())
            all_teacher_features.append(teacher_feats.squeeze(0).transpose(0, 1).cpu().numpy())

    return all_student_tokens, all_teacher_tokens, all_student_features, all_teacher_features


def analyze_token_accuracy(student_tokens_list, teacher_tokens_list, distance_matrix):
    """分析 Token 準確率和距離"""
    results = {
        'exact_match': 0,
        'total_tokens': 0,
        'top_k_accuracy': {1: 0, 5: 0, 10: 0, 50: 0, 100: 0},
        'distance_when_wrong': [],
        'distance_when_correct': [],
        'wrong_token_ranks': [],  # student token 在 teacher token 的最近鄰排名
    }

    # 預計算每個 code 的最近鄰排序
    # distance_matrix[i, j] = distance between code i and code j
    nearest_neighbors = {}
    for code in range(distance_matrix.shape[0]):
        sorted_indices = torch.argsort(distance_matrix[code])
        nearest_neighbors[code] = sorted_indices.numpy()

    for student_tokens, teacher_tokens in zip(student_tokens_list, teacher_tokens_list):
        for st, tt in zip(student_tokens, teacher_tokens):
            results['total_tokens'] += 1

            # 檢查是否完全匹配
            if st == tt:
                results['exact_match'] += 1
                results['distance_when_correct'].append(0.0)
            else:
                # 計算錯誤時的距離
                dist = distance_matrix[tt, st].item()
                results['distance_when_wrong'].append(dist)

                # 計算 student token 在 teacher 最近鄰中的排名
                rank = np.where(nearest_neighbors[tt] == st)[0][0]
                results['wrong_token_ranks'].append(rank)

            # Top-K 準確率
            for k in results['top_k_accuracy'].keys():
                if st in nearest_neighbors[tt][:k]:
                    results['top_k_accuracy'][k] += 1

    # 計算比率
    total = results['total_tokens']
    results['exact_match_rate'] = results['exact_match'] / total
    for k in results['top_k_accuracy']:
        results['top_k_accuracy'][k] = results['top_k_accuracy'][k] / total

    return results


def analyze_feature_to_codebook_distances(student_features_list, teacher_tokens_list, codebook):
    """
    分析 student features 到正確 codebook entry 的距離
    vs 到 argmin 選中的 codebook entry 的距離
    """
    correct_distances = []
    argmin_distances = []
    distance_ratios = []

    for sf_batch, tt_batch in zip(student_features_list, teacher_tokens_list):
        for sf, tt in zip(sf_batch, tt_batch):
            # sf: (512,) - student feature
            # tt: int - teacher token (正確答案)

            sf_tensor = torch.tensor(sf)

            # 到正確 code 的距離
            correct_code = codebook[tt]
            dist_to_correct = torch.norm(sf_tensor - correct_code).item()
            correct_distances.append(dist_to_correct)

            # 到 argmin code 的距離
            all_distances = torch.norm(codebook - sf_tensor.unsqueeze(0), dim=1)
            argmin_dist = all_distances.min().item()
            argmin_distances.append(argmin_dist)

            # 比率: argmin / correct
            ratio = argmin_dist / (dist_to_correct + 1e-8)
            distance_ratios.append(ratio)

    return {
        'dist_to_correct': correct_distances,
        'dist_to_argmin': argmin_distances,
        'ratio_argmin_over_correct': distance_ratios,
    }


def main():
    parser = argparse.ArgumentParser(description='Token Distance Analysis')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of audio samples to analyze')
    parser.add_argument('--use_val', action='store_true',
                       help='Use validation set')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='latest',
                       help='Checkpoint to use: latest, best, or epoch number')
    args = parser.parse_args()

    # 設置路徑
    exp_dir = Path(__file__).parent.parent / 'experiments' / args.exp_name
    output_dir = Path(__file__).parent / 'results' / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    # 載入 Distance Matrix
    print(f"\n{'='*60}")
    print("Loading distance matrix and codebook...")
    print(f"{'='*60}")

    distance_matrix = torch.load(DISTANCE_MATRIX)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # 載入數據 - 使用 Dataset 類來處理音頻載入
    print(f"\nLoading data...")
    cache_path = VAL_CACHE if args.use_val else TRAIN_CACHE
    dataset = NoisyCleanPairDataset(cache_path)
    print(f"Loaded {len(dataset)} samples")

    # 選擇樣本
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    audio_samples = []
    for i in sample_indices:
        sample = dataset[i]
        audio_samples.append((sample['noisy_audio'], sample['clean_audio']))
    print(f"Selected {len(audio_samples)} samples")

    # 確定 checkpoint
    checkpoint_dir = exp_dir / 'checkpoints'
    if args.checkpoint == 'latest':
        ckpt_path = checkpoint_dir / 'latest.pt'
    elif args.checkpoint == 'best':
        ckpt_path = checkpoint_dir / 'best.pt'
    else:
        # 尋找特定 epoch
        matches = list(checkpoint_dir.glob(f'epoch_{args.checkpoint.zfill(3)}_*.pt'))
        if matches:
            ckpt_path = matches[0]
        else:
            print(f"Checkpoint not found for epoch {args.checkpoint}")
            sys.exit(1)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # 載入模型
    print(f"\n{'='*60}")
    print(f"Loading model from {ckpt_path}")
    print(f"{'='*60}")

    model = load_checkpoint(ckpt_path, device=args.device)

    # 獲取 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.layers[0]._codebook.embed
    codebook = codebook.to(args.device)
    print(f"Codebook shape: {codebook.shape}")

    # 提取 tokens
    print(f"\n{'='*60}")
    print("Extracting tokens...")
    print(f"{'='*60}")

    student_tokens, teacher_tokens, student_features, teacher_features = \
        extract_tokens_and_features(model, audio_samples, codebook, device=args.device)

    # 分析 Token 準確率
    print(f"\n{'='*60}")
    print("Analyzing token accuracy...")
    print(f"{'='*60}")

    token_results = analyze_token_accuracy(student_tokens, teacher_tokens, distance_matrix)

    print(f"\nToken Accuracy Results:")
    print(f"  Exact Match Rate: {token_results['exact_match_rate']*100:.2f}%")
    for k, acc in token_results['top_k_accuracy'].items():
        print(f"  Top-{k} Accuracy: {acc*100:.2f}%")

    if token_results['distance_when_wrong']:
        print(f"\nDistance Statistics (when wrong):")
        print(f"  Mean: {np.mean(token_results['distance_when_wrong']):.4f}")
        print(f"  Std: {np.std(token_results['distance_when_wrong']):.4f}")
        print(f"  Median: {np.median(token_results['distance_when_wrong']):.4f}")

        print(f"\nWrong Token Rank Statistics:")
        print(f"  Mean Rank: {np.mean(token_results['wrong_token_ranks']):.1f}")
        print(f"  Median Rank: {np.median(token_results['wrong_token_ranks']):.1f}")
        print(f"  Max Rank: {np.max(token_results['wrong_token_ranks'])}")

    # 分析 Feature 到 Codebook 的距離
    print(f"\n{'='*60}")
    print("Analyzing feature-to-codebook distances...")
    print(f"{'='*60}")

    feat_results = analyze_feature_to_codebook_distances(
        student_features, teacher_tokens, codebook.cpu()
    )

    print(f"\nFeature Distance Analysis:")
    print(f"  Distance to Correct Code:")
    print(f"    Mean: {np.mean(feat_results['dist_to_correct']):.4f}")
    print(f"    Std: {np.std(feat_results['dist_to_correct']):.4f}")
    print(f"  Distance to Argmin Code:")
    print(f"    Mean: {np.mean(feat_results['dist_to_argmin']):.4f}")
    print(f"    Std: {np.std(feat_results['dist_to_argmin']):.4f}")
    print(f"  Ratio (argmin/correct):")
    print(f"    Mean: {np.mean(feat_results['ratio_argmin_over_correct']):.4f}")
    print(f"    (< 1 means argmin is closer than correct, which is expected)")

    # 可視化
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 圖1: Top-K 準確率
    ax = axes[0, 0]
    ks = list(token_results['top_k_accuracy'].keys())
    accs = [token_results['top_k_accuracy'][k] * 100 for k in ks]
    ax.bar(range(len(ks)), accs, color='steelblue')
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'Top-{k}' for k in ks])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Token Accuracy')
    ax.set_ylim([0, 100])
    for i, (k, acc) in enumerate(zip(ks, accs)):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9)

    # 圖2: 錯誤時的距離分布
    ax = axes[0, 1]
    if token_results['distance_when_wrong']:
        ax.hist(token_results['distance_when_wrong'], bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(token_results['distance_when_wrong']), color='red',
                  linestyle='--', label=f"Mean: {np.mean(token_results['distance_when_wrong']):.2f}")
        ax.set_xlabel('Distance in Codebook Space')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance Distribution (When Token is Wrong)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No wrong predictions!', ha='center', va='center', fontsize=14)
        ax.set_title('Distance Distribution (When Token is Wrong)')

    # 圖3: 錯誤 token 的排名分布
    ax = axes[1, 0]
    if token_results['wrong_token_ranks']:
        # 使用 log scale 因為排名可能很大
        ranks = token_results['wrong_token_ranks']
        ax.hist(ranks, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(ranks), color='red', linestyle='--',
                  label=f"Mean: {np.mean(ranks):.1f}")
        ax.axvline(np.median(ranks), color='green', linestyle='--',
                  label=f"Median: {np.median(ranks):.1f}")
        ax.set_xlabel('Rank of Student Token in Teacher\'s Nearest Neighbors')
        ax.set_ylabel('Frequency')
        ax.set_title('Wrong Token Rank Distribution\n(Lower is Better)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No wrong predictions!', ha='center', va='center', fontsize=14)

    # 圖4: Feature 距離比較
    ax = axes[1, 1]
    correct_dists = feat_results['dist_to_correct']
    argmin_dists = feat_results['dist_to_argmin']

    ax.scatter(correct_dists, argmin_dists, alpha=0.3, s=10)
    max_val = max(max(correct_dists), max(argmin_dists))
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (equal distance)')
    ax.set_xlabel('Distance to Correct Code')
    ax.set_ylabel('Distance to Argmin Code')
    ax.set_title('Feature Distance Comparison\n(Points below line: argmin is closer)')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plot_path = output_dir / 'token_distance_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {plot_path}")

    # 保存詳細結果
    results = {
        'experiment': args.exp_name,
        'checkpoint': str(ckpt_path),
        'num_samples': len(audio_samples),
        'total_tokens': token_results['total_tokens'],
        'exact_match_rate': token_results['exact_match_rate'],
        'top_k_accuracy': token_results['top_k_accuracy'],
        'distance_when_wrong_mean': float(np.mean(token_results['distance_when_wrong'])) if token_results['distance_when_wrong'] else None,
        'distance_when_wrong_std': float(np.std(token_results['distance_when_wrong'])) if token_results['distance_when_wrong'] else None,
        'wrong_rank_mean': float(np.mean(token_results['wrong_token_ranks'])) if token_results['wrong_token_ranks'] else None,
        'wrong_rank_median': float(np.median(token_results['wrong_token_ranks'])) if token_results['wrong_token_ranks'] else None,
        'feature_dist_to_correct_mean': float(np.mean(feat_results['dist_to_correct'])),
        'feature_dist_to_argmin_mean': float(np.mean(feat_results['dist_to_argmin'])),
        'feature_dist_ratio_mean': float(np.mean(feat_results['ratio_argmin_over_correct'])),
    }

    results_path = output_dir / 'token_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")

    # 總結
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"\n📊 Token Accuracy:")
    print(f"   Exact Match: {token_results['exact_match_rate']*100:.2f}%")
    print(f"   Top-5: {token_results['top_k_accuracy'][5]*100:.2f}%")
    print(f"   Top-10: {token_results['top_k_accuracy'][10]*100:.2f}%")

    if token_results['wrong_token_ranks']:
        print(f"\n🎯 When Wrong:")
        print(f"   Mean Distance: {np.mean(token_results['distance_when_wrong']):.4f}")
        print(f"   Median Rank: {np.median(token_results['wrong_token_ranks']):.0f}")

    print(f"\n📐 Feature Analysis:")
    ratio = np.mean(feat_results['ratio_argmin_over_correct'])
    if ratio < 1:
        print(f"   Argmin is {(1-ratio)*100:.1f}% closer than correct code (expected behavior)")
    else:
        print(f"   Correct code is {(ratio-1)*100:.1f}% closer than argmin (unexpected!)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
