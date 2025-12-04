#!/usr/bin/env python3
"""
深入分析 Student emb 為什麼不在正確的 Voronoi 區域

核心問題：
- Student emb 到 target (正確答案) 的距離: ~4.58
- Student emb 到 argmin (最近鄰) 的距離: ~0.59
- 比率 9.3x → Student 選擇的 token 幾乎都是錯的

這個腳本會分析：
1. 正確/錯誤預測時的距離分布
2. 錯誤預測時，Student 選了哪些 token？
3. 這些錯誤的 token 與正確 token 的關係
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, DISTANCE_MATRIX
from model import TeacherStudentModel
from data import NoisyCleanPairDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 載入模型
    print("Loading model...")
    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
        device=device
    )
    model.eval()

    # 獲取 codebook 和 distance matrix
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)
    print(f"Codebook shape: {codebook.shape}")  # (4096, 512)

    # 載入測試數據
    print("Loading test data...")
    dataset = NoisyCleanPairDataset(TRAIN_CACHE, max_samples=100)

    all_dist_to_target = []
    all_dist_to_argmin = []
    all_correct_mask = []
    all_wrong_predictions = []
    all_correct_targets = []

    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            noisy_audio = sample['noisy_audio'].unsqueeze(0).to(device)
            clean_audio = sample['clean_audio'].unsqueeze(0).to(device)

            # Student encoder 輸出 (VQ 前)
            audio_in = noisy_audio.unsqueeze(1)
            student_emb = model.student.feature_extractor.encodec.encoder(audio_in)

            # Teacher codes (正確答案)
            _, teacher_codes, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            if teacher_codes.dim() == 3:
                teacher_codes = teacher_codes[0]

            # 對齊長度
            T = min(student_emb.shape[-1], teacher_codes.shape[-1])
            student_emb_flat = student_emb[:, :, :T].permute(0, 2, 1).reshape(-1, 512)
            teacher_codes_flat = teacher_codes[:, :T].reshape(-1).long()

            # 計算到所有 codebook entries 的距離
            distances = torch.cdist(student_emb_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)

            # Student 的預測 (argmin)
            predictions = distances.argmin(dim=-1)

            # 到 target 和 argmin 的距離
            target_embeddings = codebook[teacher_codes_flat]
            dist_to_target = (student_emb_flat - target_embeddings).norm(dim=1)
            dist_to_argmin = distances.min(dim=1).values

            # 正確/錯誤 mask
            correct_mask = (predictions == teacher_codes_flat)

            all_dist_to_target.extend(dist_to_target.cpu().numpy())
            all_dist_to_argmin.extend(dist_to_argmin.cpu().numpy())
            all_correct_mask.extend(correct_mask.cpu().numpy())

            # 記錄錯誤預測
            wrong_mask = ~correct_mask
            all_wrong_predictions.extend(predictions[wrong_mask].cpu().numpy())
            all_correct_targets.extend(teacher_codes_flat[wrong_mask].cpu().numpy())

    all_dist_to_target = np.array(all_dist_to_target)
    all_dist_to_argmin = np.array(all_dist_to_argmin)
    all_correct_mask = np.array(all_correct_mask)
    all_wrong_predictions = np.array(all_wrong_predictions)
    all_correct_targets = np.array(all_correct_targets)

    print("\n" + "=" * 70)
    print("分析結果")
    print("=" * 70)

    # 1. 總體統計
    accuracy = all_correct_mask.mean() * 100
    print(f"\n[總體統計]")
    print(f"  Token Accuracy: {accuracy:.2f}%")
    print(f"  總 token 數: {len(all_correct_mask)}")
    print(f"  正確預測: {all_correct_mask.sum()}")
    print(f"  錯誤預測: {(~all_correct_mask).sum()}")

    # 2. 距離分析
    print(f"\n[距離分析]")
    print(f"  到 target (正確答案) 的平均距離: {all_dist_to_target.mean():.4f}")
    print(f"  到 argmin (最近鄰) 的平均距離: {all_dist_to_argmin.mean():.4f}")
    print(f"  比率 (target/argmin): {(all_dist_to_target / (all_dist_to_argmin + 1e-8)).mean():.2f}x")

    # 3. 正確 vs 錯誤預測時的距離
    correct_indices = all_correct_mask
    wrong_indices = ~all_correct_mask

    print(f"\n[正確預測時的距離]")
    if correct_indices.sum() > 0:
        print(f"  到 target 的平均距離: {all_dist_to_target[correct_indices].mean():.4f}")
        print(f"  (這時 target = argmin，所以距離相同)")

    print(f"\n[錯誤預測時的距離]")
    if wrong_indices.sum() > 0:
        print(f"  到 target (正確答案) 的平均距離: {all_dist_to_target[wrong_indices].mean():.4f}")
        print(f"  到 argmin (錯誤選擇) 的平均距離: {all_dist_to_argmin[wrong_indices].mean():.4f}")
        ratio = all_dist_to_target[wrong_indices].mean() / all_dist_to_argmin[wrong_indices].mean()
        print(f"  比率: {ratio:.2f}x → Student 選的 token 比正確答案近 {ratio:.1f} 倍")

    # 4. 分析錯誤預測的 token 關係
    print(f"\n[錯誤預測分析]")
    if len(all_wrong_predictions) > 0:
        # 計算錯誤 token 與正確 token 之間的 codebook 距離
        wrong_pred_tensor = torch.tensor(all_wrong_predictions, device=device).long()
        correct_target_tensor = torch.tensor(all_correct_targets, device=device).long()

        # 使用 distance matrix
        codebook_distances = distance_matrix[wrong_pred_tensor, correct_target_tensor].cpu().numpy()

        print(f"  錯誤預測的 token 與正確 token 的 codebook 距離:")
        print(f"    平均: {codebook_distances.mean():.4f}")
        print(f"    中位數: {np.median(codebook_distances):.4f}")
        print(f"    最小: {codebook_distances.min():.4f}")
        print(f"    最大: {codebook_distances.max():.4f}")

        # Top-K 正確率
        print(f"\n[如果使用 Soft Label / Top-K]")
        for k in [5, 10, 50, 100]:
            # 計算有多少錯誤預測落在正確 token 的 top-k 鄰居內
            in_topk = 0
            for pred, target in zip(all_wrong_predictions, all_correct_targets):
                # 找 target 的 top-k 鄰居
                target_distances = distance_matrix[target]
                topk_neighbors = target_distances.argsort()[:k]
                if pred in topk_neighbors:
                    in_topk += 1
            topk_rate = in_topk / len(all_wrong_predictions) * 100
            print(f"    錯誤預測落在正確 token 的 Top-{k} 鄰居: {topk_rate:.1f}%")

    # 5. 視覺化
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distance distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(all_dist_to_target, bins=50, alpha=0.7, label='To Target (correct)', color='red')
    ax1.hist(all_dist_to_argmin, bins=50, alpha=0.7, label='To Argmin (nearest)', color='blue')
    ax1.axvline(all_dist_to_target.mean(), color='red', linestyle='--', label=f'Target mean: {all_dist_to_target.mean():.2f}')
    ax1.axvline(all_dist_to_argmin.mean(), color='blue', linestyle='--', label=f'Argmin mean: {all_dist_to_argmin.mean():.2f}')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Student Emb Distance: Target vs Argmin')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correct vs Wrong prediction distance comparison
    ax2 = axes[0, 1]
    categories = ['Correct\nto Target', 'Wrong\nto Target', 'Wrong\nto Argmin']
    if correct_indices.sum() > 0 and wrong_indices.sum() > 0:
        means = [
            all_dist_to_target[correct_indices].mean(),
            all_dist_to_target[wrong_indices].mean(),
            all_dist_to_argmin[wrong_indices].mean()
        ]
        stds = [
            all_dist_to_target[correct_indices].std(),
            all_dist_to_target[wrong_indices].std(),
            all_dist_to_argmin[wrong_indices].std()
        ]
        colors = ['green', 'red', 'blue']
        bars = ax2.bar(categories, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax2.set_ylabel('Mean Distance')
        ax2.set_title('Correct vs Wrong Prediction Distance')
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Distance ratio distribution
    ax3 = axes[1, 0]
    ratios = all_dist_to_target / (all_dist_to_argmin + 1e-8)
    ax3.hist(ratios, bins=50, alpha=0.7, color='purple')
    ax3.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Ratio=1 (correct)')
    ax3.axvline(ratios.mean(), color='red', linestyle='--', label=f'Mean ratio: {ratios.mean():.2f}')
    ax3.set_xlabel('Ratio (Target dist / Argmin dist)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distance Ratio Distribution (>1 = wrong prediction)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Codebook distance (wrong token vs correct token)
    ax4 = axes[1, 1]
    if len(codebook_distances) > 0:
        ax4.hist(codebook_distances, bins=50, alpha=0.7, color='orange')
        ax4.axvline(codebook_distances.mean(), color='red', linestyle='--',
                   label=f'Mean: {codebook_distances.mean():.2f}')
        ax4.set_xlabel('Codebook Distance (wrong token <-> correct token)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Codebook Distance: Wrong vs Correct Token')
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203/voronoi_analysis.png', dpi=150)
    print("圖表已保存: exp_1203/voronoi_analysis.png")

    # 結論
    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    print("""
問題根源：
    Student encoder (with LoRA) 處理 noisy audio 後的輸出，
    距離正確的 codebook entry (Teacher 選的) 太遠。

    具體來說：
    1. 正確預測時：Student emb 剛好落在正確的 Voronoi 區域
    2. 錯誤預測時：Student emb 落在其他 Voronoi 區域
       → 到正確 target 的距離約 4-5
       → 到錯誤 argmin 的距離約 0.5-0.6
       → 差距 ~9 倍

    這說明 LoRA 的容量不足以讓 Student encoder 學會：
    「noisy audio → clean audio 對應的 codebook entry」這個複雜映射

可能的解決方案：
    1. 增加 LoRA rank (目前 64 → 嘗試 256, 512)
       - exp9 (rank=256) 結果顯示效果有限

    2. 使用 Soft Label 而非 Hard Label
       - 允許 Student 選擇 「接近正確答案的 token」
       - 但目前分析顯示錯誤預測通常不在 top-k 鄰居內

    3. 更強的監督信號
       - 直接監督 VQ 後的 feature (而非 VQ 前的 emb)
       - 添加對比學習損失

    4. 更激進的架構變化
       - Fine-tune 更多層 (不只是 LoRA)
       - 添加額外的 adapter layers
""")


if __name__ == "__main__":
    main()
