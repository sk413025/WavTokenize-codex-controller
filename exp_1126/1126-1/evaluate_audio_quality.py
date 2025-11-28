#!/usr/bin/env python3
"""
評估 Audio 重建品質和 VQ Code 分析

1. 比較 Student 和 Teacher 選到的 codes 的 L2 距離
2. 計算 Top-K Token Accuracy（放寬匹配標準）
3. 分析 Feature 相似度
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(1, '/home/sbplab/ruizi/WavTokenizer-main')

import torch
import torch.nn.functional as F
import numpy as np

def evaluate_quality():
    print("=" * 70)
    print("Audio 重建品質評估")
    print("=" * 70)

    # 載入模型
    print("\n[1] 載入模型...")
    from model import create_teacher_student_model
    from config import TrainConfig

    config = TrainConfig()
    config.lora_rank = 16
    config.lora_alpha = 32
    device = "cuda"

    model = create_teacher_student_model(config, device=device)

    # 載入訓練後的權重
    print("\n[2] 載入訓練後的 checkpoint...")
    ckpt_path = "experiments/lora_encoder_frozen_vq_v3/checkpoints/latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"    Loaded epoch {ckpt.get('epoch', 'unknown')}")

    # 載入測試數據
    print("\n[3] 載入測試數據...")
    from data import NoisyCleanPairDataset, collate_fn
    from torch.utils.data import DataLoader

    val_dataset = NoisyCleanPairDataset(
        cache_path="/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt",
        max_samples=100  # 只測試 100 個樣本
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

    num_samples = min(100, len(val_dataset))
    print(f"    Testing on {num_samples} samples (from {len(val_dataset)} total)")

    # 獲取 codebook 和 distance matrix
    teacher_codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    distance_matrix = model.distance_matrix

    # 分析指標
    all_token_acc = []
    all_top5_acc = []
    all_top10_acc = []
    all_code_distances = []
    all_feature_mse = []
    all_feature_cosine = []

    print("\n[4] 開始評估...")

    samples_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if samples_processed >= num_samples:
                break

            noisy_batch = batch['noisy_audio'].to(device)
            clean_batch = batch['clean_audio'].to(device)

            # Forward pass
            output = model(noisy_batch, clean_batch)

            student_features = output['student_features']  # (B, 512, T)
            teacher_features = output['teacher_features']  # (B, 512, T)
            student_codes = output['student_codes'].squeeze(1)  # (B, T)
            teacher_codes = output['teacher_codes'].squeeze(1)  # (B, T)

            B, _, T = student_features.shape

            # 1. Token Accuracy
            token_acc = (student_codes == teacher_codes).float().mean().item()
            all_token_acc.append(token_acc)

            # 2. Top-K Accuracy
            # 對於每個 student feature，找到最近的 K 個 codes，看 teacher code 是否在其中
            student_feat_flat = student_features.permute(0, 2, 1).reshape(-1, 512)  # (B*T, 512)
            teacher_codes_flat = teacher_codes.reshape(-1)  # (B*T,)

            # 計算到所有 codes 的距離
            distances_to_all = torch.cdist(student_feat_flat, teacher_codebook)  # (B*T, 4096)

            # Top-K indices
            _, top5_indices = distances_to_all.topk(5, dim=1, largest=False)
            _, top10_indices = distances_to_all.topk(10, dim=1, largest=False)

            top5_correct = (top5_indices == teacher_codes_flat.unsqueeze(1)).any(dim=1).float().mean().item()
            top10_correct = (top10_indices == teacher_codes_flat.unsqueeze(1)).any(dim=1).float().mean().item()

            all_top5_acc.append(top5_correct)
            all_top10_acc.append(top10_correct)

            # 3. Code Distance (選到的 codes 之間的距離)
            student_codes_flat = student_codes.reshape(-1)
            code_dist = distance_matrix[student_codes_flat, teacher_codes_flat].mean().item()
            all_code_distances.append(code_dist)

            # 4. Feature MSE
            feature_mse = F.mse_loss(student_features, teacher_features).item()
            all_feature_mse.append(feature_mse)

            # 5. Feature Cosine Similarity
            student_norm = F.normalize(student_features, dim=1)
            teacher_norm = F.normalize(teacher_features, dim=1)
            cosine_sim = (student_norm * teacher_norm).sum(dim=1).mean().item()
            all_feature_cosine.append(cosine_sim)

            samples_processed += noisy_batch.shape[0]
            if batch_idx % 2 == 0:
                print(f"    Processed {samples_processed}/{num_samples} samples...")

    # 統計結果
    print("\n" + "=" * 70)
    print("評估結果")
    print("=" * 70)

    print(f"\n📊 Token Matching:")
    print(f"    Exact Match (Top-1):  {np.mean(all_token_acc)*100:.2f}%")
    print(f"    Top-5 Match:          {np.mean(all_top5_acc)*100:.2f}%")
    print(f"    Top-10 Match:         {np.mean(all_top10_acc)*100:.2f}%")

    print(f"\n📏 Code Distance:")
    print(f"    平均 L2 距離:         {np.mean(all_code_distances):.4f}")
    print(f"    (距離越小 = 選到的 codes 越相近)")

    print(f"\n🎯 Feature Similarity:")
    print(f"    MSE:                  {np.mean(all_feature_mse):.6f}")
    print(f"    Cosine Similarity:    {np.mean(all_feature_cosine):.4f}")

    # 與初始狀態比較（如果有的話）
    print("\n" + "=" * 70)
    print("分析結論")
    print("=" * 70)

    if np.mean(all_top5_acc) > 0.15:
        print("✅ Top-5 Accuracy 較高，表示 Student 選的 code 與 Teacher 接近")
    else:
        print("⚠️ Top-5 Accuracy 較低，Student 和 Teacher 的 code 選擇差異較大")

    if np.mean(all_code_distances) < 3.0:
        print("✅ Code Distance 較小，即使選到不同 code，它們的 embedding 也相近")
    else:
        print("⚠️ Code Distance 較大，選到的 codes embedding 差異明顯")

    if np.mean(all_feature_cosine) > 0.9:
        print("✅ Feature Cosine Similarity 很高，feature 方向一致")
    elif np.mean(all_feature_cosine) > 0.8:
        print("⚠️ Feature Cosine Similarity 中等")
    else:
        print("❌ Feature Cosine Similarity 較低")

    return True


if __name__ == "__main__":
    evaluate_quality()
