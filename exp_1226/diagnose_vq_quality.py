"""
exp_1226: VQ 品質診斷腳本

診斷項目：
1. Feature Space Alignment
   - Cosine similarity (方向)
   - L2 distance (magnitude)
   - Per-dimension 分析

2. VQ Quantization Error
   - Teacher: clean_feature → VQ → reconstructed
   - Student: noisy_feature → VQ → reconstructed
   - 比較 ||original - reconstructed||

3. Token Prediction 分析
   - Token distribution (是否 collapse 到少數 codes)
   - Per-position accuracy

使用方式:
    python exp_1226/diagnose_vq_quality.py --checkpoint <path_to_checkpoint>
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
from torch.utils.data import DataLoader


def load_wavtokenizer():
    """載入 WavTokenizer"""
    from wavtokenizer.decoder.pretrained import WavTokenizer
    config_path = "/home/sbplab/ruizi/WavTokenize-self-supervised/wavtokenizer/configs/wavtokenizer_mediumdata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "/home/sbplab/ruizi/WavTokenize-self-supervised/wavtokenizer/wavtokenizer_large_speech_320_24k.ckpt"
    return WavTokenizer.from_pretrained0802(config_path, model_path)


def load_student_model(checkpoint_path, device):
    """載入 Student 模型"""
    from exp_1219.model import create_lora_wavtokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 從 checkpoint 取得 LoRA 參數
    lora_rank = checkpoint.get('lora_rank', 256)
    lora_alpha = checkpoint.get('lora_alpha', 512)
    lora_dropout = checkpoint.get('lora_dropout', 0.2)
    lora_layers = checkpoint.get('lora_layers', 'all_18')

    print(f"Loading model with LoRA: rank={lora_rank}, alpha={lora_alpha}")

    student = create_lora_wavtokenizer(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_layers=lora_layers
    )

    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()
    return student.to(device)


def analyze_feature_alignment(student_features, teacher_features, mask=None):
    """
    分析 Feature Space Alignment

    Returns:
        dict: 包含各種對齊指標
    """
    # Flatten for analysis
    if mask is not None:
        # 只分析有效位置
        B, T, D = student_features.shape
        mask_expanded = mask.unsqueeze(-1).expand_as(student_features)
        s_flat = student_features[mask_expanded].view(-1, D)
        t_flat = teacher_features[mask_expanded].view(-1, D)
    else:
        s_flat = student_features.reshape(-1, student_features.shape[-1])
        t_flat = teacher_features.reshape(-1, teacher_features.shape[-1])

    # 1. Cosine Similarity (方向)
    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)

    # 2. L2 Distance (magnitude)
    l2_dist = torch.norm(s_flat - t_flat, dim=-1)

    # 3. Magnitude 比較
    s_norm = torch.norm(s_flat, dim=-1)
    t_norm = torch.norm(t_flat, dim=-1)
    norm_ratio = s_norm / (t_norm + 1e-8)

    # 4. Per-dimension 分析
    dim_mse = ((s_flat - t_flat) ** 2).mean(dim=0)  # (D,)
    dim_correlation = []
    for d in range(s_flat.shape[-1]):
        corr = torch.corrcoef(torch.stack([s_flat[:, d], t_flat[:, d]]))[0, 1]
        dim_correlation.append(corr.item() if not torch.isnan(corr) else 0.0)

    return {
        'cosine_similarity': {
            'mean': cos_sim.mean().item(),
            'std': cos_sim.std().item(),
            'min': cos_sim.min().item(),
            'max': cos_sim.max().item(),
        },
        'l2_distance': {
            'mean': l2_dist.mean().item(),
            'std': l2_dist.std().item(),
        },
        'magnitude': {
            'student_norm_mean': s_norm.mean().item(),
            'teacher_norm_mean': t_norm.mean().item(),
            'ratio_mean': norm_ratio.mean().item(),
            'ratio_std': norm_ratio.std().item(),
        },
        'per_dimension': {
            'mse_mean': dim_mse.mean().item(),
            'mse_max': dim_mse.max().item(),
            'mse_argmax': dim_mse.argmax().item(),
            'correlation_mean': np.mean(dim_correlation),
            'correlation_min': np.min(dim_correlation),
        }
    }


def analyze_vq_quantization(features, vq_layer, codebook):
    """
    分析 VQ 量化誤差

    Args:
        features: (B, T, D) 特徵
        vq_layer: VQ layer (用於量化)
        codebook: (num_codes, D) codebook embeddings

    Returns:
        dict: VQ 量化分析結果
    """
    B, T, D = features.shape
    features_flat = features.reshape(-1, D)  # (B*T, D)

    # 計算到所有 codebook 的距離
    # distances: (B*T, num_codes)
    distances = torch.cdist(features_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)

    # 最近的 code
    min_distances, codes = distances.min(dim=-1)

    # 量化後的特徵
    quantized = codebook[codes]  # (B*T, D)

    # 量化誤差
    quant_error = torch.norm(features_flat - quantized, dim=-1)

    # Code 使用分佈
    code_counts = Counter(codes.cpu().numpy().tolist())
    num_unique_codes = len(code_counts)
    most_common = code_counts.most_common(10)

    return {
        'quantization_error': {
            'mean': quant_error.mean().item(),
            'std': quant_error.std().item(),
            'max': quant_error.max().item(),
        },
        'min_distance_to_centroid': {
            'mean': min_distances.mean().item(),
            'std': min_distances.std().item(),
        },
        'code_usage': {
            'num_unique': num_unique_codes,
            'total_codes': codebook.shape[0],
            'usage_ratio': num_unique_codes / codebook.shape[0],
            'most_common_10': most_common,
        }
    }


def analyze_token_prediction(student_codes, teacher_codes, mask=None):
    """
    分析 Token 預測
    """
    if mask is not None:
        s_valid = student_codes[mask.bool()]
        t_valid = teacher_codes[mask.bool()]
    else:
        s_valid = student_codes.flatten()
        t_valid = teacher_codes.flatten()

    # Accuracy
    correct = (s_valid == t_valid).float()
    accuracy = correct.mean().item()

    # Per-position accuracy (如果有 batch 維度)
    B, T = student_codes.shape
    pos_acc = []
    for t in range(T):
        if mask is not None:
            valid = mask[:, t].bool()
            if valid.sum() > 0:
                acc = (student_codes[valid, t] == teacher_codes[valid, t]).float().mean().item()
                pos_acc.append(acc)
        else:
            acc = (student_codes[:, t] == teacher_codes[:, t]).float().mean().item()
            pos_acc.append(acc)

    # Student code distribution
    s_counts = Counter(s_valid.cpu().numpy().tolist())
    t_counts = Counter(t_valid.cpu().numpy().tolist())

    return {
        'accuracy': accuracy,
        'position_accuracy': {
            'mean': np.mean(pos_acc) if pos_acc else 0,
            'std': np.std(pos_acc) if pos_acc else 0,
            'min': np.min(pos_acc) if pos_acc else 0,
            'max': np.max(pos_acc) if pos_acc else 0,
        },
        'student_code_usage': {
            'num_unique': len(s_counts),
            'most_common_5': s_counts.most_common(5),
        },
        'teacher_code_usage': {
            'num_unique': len(t_counts),
            'most_common_5': t_counts.most_common(5),
        },
        'code_overlap': len(set(s_counts.keys()) & set(t_counts.keys())),
    }


def run_diagnosis(checkpoint_path, num_samples=50, device='cuda'):
    """執行完整診斷"""
    print("=" * 60)
    print("VQ Quality Diagnosis")
    print("=" * 60)

    # 載入模型
    print("\n[1] Loading models...")
    teacher = load_wavtokenizer()
    teacher.eval()
    teacher = teacher.to(device)

    student = load_student_model(checkpoint_path, device)

    # 取得 VQ 相關組件
    encoder = teacher.feature_extractor
    vq_layer = teacher.backbone.quantizer  # 假設這是 VQ layer

    # 取得 codebook
    # WavTokenizer 的 codebook 位置可能不同，需要確認
    try:
        codebook = vq_layer.embed.weight  # (num_codes, D)
    except:
        try:
            codebook = vq_layer.codebook  # 另一種可能
        except:
            print("Warning: Could not find codebook, skipping VQ analysis")
            codebook = None

    # 載入資料
    print("\n[2] Loading validation data...")
    from exp_1201.config import VAL_CACHE
    dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=num_samples)
    loader = DataLoader(dataset, batch_size=4, collate_fn=aligned_collate_fn, num_workers=0)

    # 收集結果
    all_feature_analysis = []
    all_vq_analysis_student = []
    all_vq_analysis_teacher = []
    all_token_analysis = []

    print(f"\n[3] Running diagnosis on {num_samples} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths']

            # 計算有效 token 數
            encoder_stride = 320
            max_len = noisy.shape[-1]
            num_tokens = max_len // encoder_stride

            # 建立 mask
            token_lengths = lengths // encoder_stride
            mask = torch.zeros(noisy.shape[0], num_tokens, device=device)
            for i, tl in enumerate(token_lengths):
                mask[i, :tl] = 1

            # 取得 features
            noisy_unsqueezed = noisy.unsqueeze(1) if noisy.dim() == 2 else noisy
            clean_unsqueezed = clean.unsqueeze(1) if clean.dim() == 2 else clean

            # Student (noisy input)
            s_features = encoder(noisy_unsqueezed)  # (B, D, T)
            s_features = s_features.transpose(1, 2)  # (B, T, D)

            # Teacher (clean input)
            t_features = encoder(clean_unsqueezed)  # (B, D, T)
            t_features = t_features.transpose(1, 2)  # (B, T, D)

            # 對齊長度
            min_t = min(s_features.shape[1], t_features.shape[1], mask.shape[1])
            s_features = s_features[:, :min_t, :]
            t_features = t_features[:, :min_t, :]
            mask = mask[:, :min_t]

            # Feature alignment 分析
            feat_analysis = analyze_feature_alignment(s_features, t_features, mask)
            all_feature_analysis.append(feat_analysis)

            # VQ 分析 (如果有 codebook)
            if codebook is not None:
                vq_student = analyze_vq_quantization(s_features, vq_layer, codebook)
                vq_teacher = analyze_vq_quantization(t_features, vq_layer, codebook)
                all_vq_analysis_student.append(vq_student)
                all_vq_analysis_teacher.append(vq_teacher)

            # Token prediction 分析
            _, s_codes = teacher.encode(noisy_unsqueezed)
            _, t_codes = teacher.encode(clean_unsqueezed)

            s_codes = s_codes.squeeze(0) if s_codes.dim() == 3 else s_codes
            t_codes = t_codes.squeeze(0) if t_codes.dim() == 3 else t_codes

            min_t = min(s_codes.shape[1], t_codes.shape[1], mask.shape[1])
            s_codes = s_codes[:, :min_t]
            t_codes = t_codes[:, :min_t]
            mask_tokens = mask[:, :min_t]

            token_analysis = analyze_token_prediction(s_codes, t_codes, mask_tokens)
            all_token_analysis.append(token_analysis)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * 4} samples...")

    # 匯總結果
    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULTS")
    print("=" * 60)

    # Feature Alignment
    print("\n[A] Feature Space Alignment")
    print("-" * 40)
    cos_sims = [a['cosine_similarity']['mean'] for a in all_feature_analysis]
    l2_dists = [a['l2_distance']['mean'] for a in all_feature_analysis]
    mag_ratios = [a['magnitude']['ratio_mean'] for a in all_feature_analysis]

    print(f"  Cosine Similarity:  {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")
    print(f"  L2 Distance:        {np.mean(l2_dists):.4f} ± {np.std(l2_dists):.4f}")
    print(f"  Magnitude Ratio:    {np.mean(mag_ratios):.4f} ± {np.std(mag_ratios):.4f}")
    print(f"                      (1.0 = perfect match)")

    if np.mean(cos_sims) > 0.9:
        print("  → 方向對齊良好 ✓")
    else:
        print("  → ⚠️ 方向對齊不足")

    if 0.9 < np.mean(mag_ratios) < 1.1:
        print("  → Magnitude 對齊良好 ✓")
    else:
        print(f"  → ⚠️ Magnitude 偏差 (ratio={np.mean(mag_ratios):.2f})")

    # VQ Analysis
    if all_vq_analysis_student:
        print("\n[B] VQ Quantization Error")
        print("-" * 40)

        s_quant_err = [a['quantization_error']['mean'] for a in all_vq_analysis_student]
        t_quant_err = [a['quantization_error']['mean'] for a in all_vq_analysis_teacher]

        print(f"  Student (noisy):  {np.mean(s_quant_err):.4f} ± {np.std(s_quant_err):.4f}")
        print(f"  Teacher (clean):  {np.mean(t_quant_err):.4f} ± {np.std(t_quant_err):.4f}")
        print(f"  Ratio (S/T):      {np.mean(s_quant_err)/np.mean(t_quant_err):.2f}x")

        if np.mean(s_quant_err) > np.mean(t_quant_err) * 1.5:
            print("  → ⚠️ Student VQ error 顯著高於 Teacher")

    # Token Analysis
    print("\n[C] Token Prediction Analysis")
    print("-" * 40)

    accuracies = [a['accuracy'] for a in all_token_analysis]
    s_unique = [a['student_code_usage']['num_unique'] for a in all_token_analysis]
    t_unique = [a['teacher_code_usage']['num_unique'] for a in all_token_analysis]

    print(f"  Token Accuracy:     {np.mean(accuracies)*100:.2f}%")
    print(f"  Student unique codes: {np.mean(s_unique):.1f}")
    print(f"  Teacher unique codes: {np.mean(t_unique):.1f}")

    if np.mean(s_unique) < np.mean(t_unique) * 0.5:
        print("  → ⚠️ Student code 多樣性低 (可能 collapse)")

    # 最常用的 codes
    print("\n  Student most common codes:")
    all_s_codes = Counter()
    for a in all_token_analysis:
        for code, count in a['student_code_usage']['most_common_5']:
            all_s_codes[code] += count
    for code, count in all_s_codes.most_common(5):
        print(f"    Code {code}: {count}")

    print("\n  Teacher most common codes:")
    all_t_codes = Counter()
    for a in all_token_analysis:
        for code, count in a['teacher_code_usage']['most_common_5']:
            all_t_codes[code] += count
    for code, count in all_t_codes.most_common(5):
        print(f"    Code {code}: {count}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []
    if np.mean(cos_sims) < 0.9:
        issues.append("Feature 方向對齊不足")
    if not (0.9 < np.mean(mag_ratios) < 1.1):
        issues.append(f"Feature magnitude 偏差 ({np.mean(mag_ratios):.2f}x)")
    if np.mean(accuracies) < 0.1:
        issues.append("Token accuracy 極低")
    if np.mean(s_unique) < np.mean(t_unique) * 0.5:
        issues.append("Student code diversity collapse")

    if issues:
        print("\n發現的問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n未發現明顯問題，可能需要更深入分析")

    return {
        'feature_analysis': all_feature_analysis,
        'vq_student': all_vq_analysis_student,
        'vq_teacher': all_vq_analysis_teacher,
        'token_analysis': all_token_analysis,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    results = run_diagnosis(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )
