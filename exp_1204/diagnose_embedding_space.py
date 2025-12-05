"""
exp_1204: Embedding Space Diagnosis Tool

診斷目標:
1. 驗證是情況 A (MSE 下降但預測分散) 還是情況 B (Mode Collapse)
2. 可視化 student embedding 和 codebook 的空間分布
3. 分析 embedding 子空間問題

使用方式:
    python diagnose_embedding_space.py --checkpoint <path_to_checkpoint>
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

# 添加父目錄到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_and_data(checkpoint_path, device='cuda'):
    """載入模型和測試數據"""
    from exp_1204.model import TeacherStudentModel as DistillationModel
    from exp_1204.data import get_dataloaders

    # 載入 checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 從 checkpoint 獲取配置
    config = checkpoint.get('config', {})

    # 創建模型
    model = DistillationModel(
        lora_rank=config.get('lora_rank', 128),
        lora_alpha=config.get('lora_alpha', 256),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 載入數據
    _, val_loader = get_dataloaders(
        batch_size=config.get('batch_size', 20),
        num_workers=0,
    )

    # 載入 codebook
    codebook = model.teacher_model.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed[0]

    return model, val_loader, codebook


def collect_embeddings(model, val_loader, codebook, device='cuda', max_batches=50):
    """收集 student embedding 和 teacher codes"""
    all_student_emb = []
    all_teacher_codes = []
    all_student_predictions = []

    print(f"Collecting embeddings from {max_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            audio = batch['audio'].to(device)

            # Forward pass
            output = model(audio)
            student_emb = output['student_emb']  # (B, C, T)
            teacher_codes = output['teacher_codes']  # (n_q, B, T)

            # 處理維度
            B, C, T_emb = student_emb.shape
            if teacher_codes.dim() == 3:
                teacher_codes_2d = teacher_codes[0]  # (B, T)
            else:
                teacher_codes_2d = teacher_codes.squeeze(1)

            T_code = teacher_codes_2d.shape[1]
            T = min(T_emb, T_code)

            # Flatten
            teacher_flat = teacher_codes_2d[:, :T].reshape(-1).long()
            emb_truncated = student_emb[:, :, :T]
            emb_flat = emb_truncated.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)

            # 計算 student 預測的 token
            distances = torch.cdist(
                emb_flat.unsqueeze(0),
                codebook.unsqueeze(0)
            ).squeeze(0)  # (B*T, 4096)
            predictions = distances.argmin(dim=-1)

            # 收集
            all_student_emb.append(emb_flat.cpu())
            all_teacher_codes.append(teacher_flat.cpu())
            all_student_predictions.append(predictions.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches")

    all_student_emb = torch.cat(all_student_emb, dim=0)
    all_teacher_codes = torch.cat(all_teacher_codes, dim=0)
    all_student_predictions = torch.cat(all_student_predictions, dim=0)

    print(f"Collected {len(all_student_emb)} samples")

    return all_student_emb, all_teacher_codes, all_student_predictions, codebook.cpu()


def diagnose_mode_collapse(student_predictions, teacher_codes, num_tokens=4096):
    """
    診斷 Mode Collapse

    情況 A: 預測分散在多個 token (但可能預測錯誤)
    情況 B: 預測集中在少數 token (Mode Collapse)
    """
    print("\n" + "="*80)
    print("Mode Collapse 診斷")
    print("="*80)

    # 統計預測分布
    pred_counter = Counter(student_predictions.numpy().tolist())
    teacher_counter = Counter(teacher_codes.numpy().tolist())

    # 計算覆蓋的 unique token 數量
    unique_preds = len(pred_counter)
    unique_teachers = len(teacher_counter)

    print(f"\n1. Token 多樣性分析:")
    print(f"   - Student 預測的 unique token 數量: {unique_preds} / {num_tokens}")
    print(f"   - Teacher 的 unique token 數量: {unique_teachers} / {num_tokens}")
    print(f"   - 覆蓋率: {unique_preds / num_tokens * 100:.2f}%")

    # 分析 Top-K 預測
    top_k = 20
    most_common_preds = pred_counter.most_common(top_k)
    most_common_teachers = teacher_counter.most_common(top_k)

    print(f"\n2. Top-{top_k} 最常預測的 token:")
    total_samples = len(student_predictions)
    top_k_coverage = sum([count for _, count in most_common_preds]) / total_samples * 100
    print(f"   - Top-{top_k} token 覆蓋了 {top_k_coverage:.2f}% 的預測")

    for i, (token, count) in enumerate(most_common_preds[:10]):
        teacher_count = teacher_counter.get(token, 0)
        print(f"   [{i+1}] Token {token}: {count} ({count/total_samples*100:.2f}%), Teacher 中有 {teacher_count}")

    # 計算 entropy
    pred_probs = np.array([count / total_samples for _, count in pred_counter.most_common()])
    entropy = -np.sum(pred_probs * np.log2(pred_probs + 1e-10))
    max_entropy = np.log2(num_tokens)

    print(f"\n3. 分布 Entropy:")
    print(f"   - 預測分布 entropy: {entropy:.2f} bits")
    print(f"   - 最大可能 entropy: {max_entropy:.2f} bits")
    print(f"   - 正規化 entropy: {entropy / max_entropy * 100:.2f}%")

    # 診斷結論
    print(f"\n4. 診斷結論:")
    if unique_preds < 100:
        print(f"   ⚠️ 嚴重 Mode Collapse！預測只集中在 {unique_preds} 個 token")
        diagnosis = "Mode Collapse (嚴重)"
    elif unique_preds < 500:
        print(f"   ⚠️ 輕度 Mode Collapse，預測集中在 {unique_preds} 個 token")
        diagnosis = "Mode Collapse (輕度)"
    elif top_k_coverage > 50:
        print(f"   ⚠️ 部分 Mode Collapse，Top-{top_k} 覆蓋了 {top_k_coverage:.1f}%")
        diagnosis = "部分 Mode Collapse"
    else:
        print(f"   ✓ 預測較分散，不是典型的 Mode Collapse")
        print(f"   → 可能是情況 A: MSE 下降但預測分散到錯誤的 token")
        diagnosis = "情況 A (預測分散但錯誤)"

    return {
        'unique_predictions': unique_preds,
        'unique_teachers': unique_teachers,
        'top_k_coverage': top_k_coverage,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'diagnosis': diagnosis,
    }


def analyze_distance_distribution(student_emb, teacher_codes, codebook):
    """
    分析距離分布
    """
    print("\n" + "="*80)
    print("距離分布分析")
    print("="*80)

    # 計算到所有 codebook token 的距離
    distances = torch.cdist(student_emb, codebook)  # (N, 4096)

    # 獲取到正確 token 的距離
    correct_distances = distances[torch.arange(len(teacher_codes)), teacher_codes]

    # 獲取到最近 token 的距離
    min_distances, min_indices = distances.min(dim=1)

    # 計算正確的比例
    correct_mask = (min_indices == teacher_codes)
    accuracy = correct_mask.float().mean().item()

    print(f"\n1. 基本統計:")
    print(f"   - Token Accuracy: {accuracy * 100:.2f}%")
    print(f"   - 到正確 token 的平均距離: {correct_distances.mean():.4f}")
    print(f"   - 到最近 token 的平均距離: {min_distances.mean():.4f}")

    # 分析錯誤案例
    wrong_mask = ~correct_mask
    if wrong_mask.sum() > 0:
        wrong_correct_dist = correct_distances[wrong_mask]
        wrong_min_dist = min_distances[wrong_mask]

        print(f"\n2. 錯誤案例分析 (共 {wrong_mask.sum()} 個):")
        print(f"   - 到正確 token 的距離: {wrong_correct_dist.mean():.4f} ± {wrong_correct_dist.std():.4f}")
        print(f"   - 到最近 token 的距離: {wrong_min_dist.mean():.4f} ± {wrong_min_dist.std():.4f}")
        print(f"   - 距離差異 (correct - min): {(wrong_correct_dist - wrong_min_dist).mean():.4f}")

    # 分析正確案例
    if correct_mask.sum() > 0:
        right_correct_dist = correct_distances[correct_mask]
        right_min_dist = min_distances[correct_mask]

        print(f"\n3. 正確案例分析 (共 {correct_mask.sum()} 個):")
        print(f"   - 到正確 token 的距離: {right_correct_dist.mean():.4f} ± {right_correct_dist.std():.4f}")

    # 分析距離的 rank
    ranks = (distances <= correct_distances.unsqueeze(1)).sum(dim=1).float()

    print(f"\n4. 正確 token 的排名分析:")
    print(f"   - 平均排名: {ranks.mean():.1f}")
    print(f"   - 中位數排名: {ranks.median():.1f}")
    print(f"   - Top-1 (Acc): {(ranks == 1).float().mean() * 100:.2f}%")
    print(f"   - Top-5: {(ranks <= 5).float().mean() * 100:.2f}%")
    print(f"   - Top-10: {(ranks <= 10).float().mean() * 100:.2f}%")
    print(f"   - Top-100: {(ranks <= 100).float().mean() * 100:.2f}%")

    return {
        'accuracy': accuracy,
        'mean_correct_distance': correct_distances.mean().item(),
        'mean_min_distance': min_distances.mean().item(),
        'mean_rank': ranks.mean().item(),
        'top5_acc': (ranks <= 5).float().mean().item(),
        'top10_acc': (ranks <= 10).float().mean().item(),
    }


def visualize_embedding_space(student_emb, teacher_codes, codebook, predictions, save_dir):
    """
    使用 PCA 和 t-SNE 可視化 embedding 空間
    """
    print("\n" + "="*80)
    print("Embedding 空間可視化")
    print("="*80)

    os.makedirs(save_dir, exist_ok=True)

    # 隨機採樣（避免太多點）
    n_samples = min(5000, len(student_emb))
    indices = np.random.choice(len(student_emb), n_samples, replace=False)

    student_sample = student_emb[indices].numpy()
    teacher_sample = teacher_codes[indices].numpy()
    pred_sample = predictions[indices].numpy()
    codebook_np = codebook.numpy()

    # 選擇一些常見的 token 來著色
    unique_teachers = np.unique(teacher_sample)
    if len(unique_teachers) > 10:
        # 取最常見的 10 個
        teacher_counter = Counter(teacher_sample.tolist())
        top_tokens = [t for t, _ in teacher_counter.most_common(10)]
    else:
        top_tokens = unique_teachers.tolist()

    # ========== PCA 可視化 ==========
    print("\n1. PCA 降維...")

    # 合併 student 和 codebook 進行 PCA
    combined = np.vstack([student_sample, codebook_np])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    student_2d = combined_2d[:n_samples]
    codebook_2d = combined_2d[n_samples:]

    print(f"   - PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # 繪製 PCA 圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 圖 1: Student embedding 分布
    ax = axes[0]
    correct_mask = pred_sample == teacher_sample
    ax.scatter(student_2d[~correct_mask, 0], student_2d[~correct_mask, 1],
               c='red', alpha=0.3, s=5, label=f'Wrong ({(~correct_mask).sum()})')
    ax.scatter(student_2d[correct_mask, 0], student_2d[correct_mask, 1],
               c='green', alpha=0.3, s=5, label=f'Correct ({correct_mask.sum()})')
    ax.set_title('Student Embedding (PCA)\nGreen=Correct, Red=Wrong')
    ax.legend()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    # 圖 2: Codebook 分布
    ax = axes[1]
    ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c='blue', alpha=0.1, s=5)
    # 標記常見 token
    for token in top_tokens[:5]:
        ax.scatter(codebook_2d[token, 0], codebook_2d[token, 1],
                   s=100, marker='*', label=f'Token {token}')
    ax.set_title('Codebook Distribution (PCA)')
    ax.legend()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    # 圖 3: Student 和 Codebook 重疊
    ax = axes[2]
    ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c='blue', alpha=0.2, s=10, label='Codebook')
    ax.scatter(student_2d[:, 0], student_2d[:, 1], c='orange', alpha=0.2, s=5, label='Student')
    ax.set_title('Student vs Codebook (PCA)')
    ax.legend()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    plt.tight_layout()
    pca_path = os.path.join(save_dir, 'embedding_pca.png')
    plt.savefig(pca_path, dpi=150)
    print(f"   - Saved to {pca_path}")
    plt.close()

    # ========== t-SNE 可視化 (較慢) ==========
    print("\n2. t-SNE 降維 (這可能需要一些時間)...")

    # 只取部分樣本
    tsne_n = min(2000, n_samples)
    tsne_student = student_sample[:tsne_n]
    tsne_teacher = teacher_sample[:tsne_n]
    tsne_pred = pred_sample[:tsne_n]

    # 合併進行 t-SNE
    combined_tsne = np.vstack([tsne_student, codebook_np])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    combined_tsne_2d = tsne.fit_transform(combined_tsne)

    student_tsne_2d = combined_tsne_2d[:tsne_n]
    codebook_tsne_2d = combined_tsne_2d[tsne_n:]

    # 繪製 t-SNE 圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 圖 1: Student 按正確/錯誤著色
    ax = axes[0]
    correct_mask_tsne = tsne_pred == tsne_teacher
    ax.scatter(student_tsne_2d[~correct_mask_tsne, 0], student_tsne_2d[~correct_mask_tsne, 1],
               c='red', alpha=0.3, s=10, label=f'Wrong ({(~correct_mask_tsne).sum()})')
    ax.scatter(student_tsne_2d[correct_mask_tsne, 0], student_tsne_2d[correct_mask_tsne, 1],
               c='green', alpha=0.3, s=10, label=f'Correct ({correct_mask_tsne.sum()})')
    ax.set_title('Student Embedding (t-SNE)\nGreen=Correct, Red=Wrong')
    ax.legend()

    # 圖 2: Student 和 Codebook 重疊
    ax = axes[1]
    ax.scatter(codebook_tsne_2d[:, 0], codebook_tsne_2d[:, 1], c='blue', alpha=0.2, s=10, label='Codebook')
    ax.scatter(student_tsne_2d[:, 0], student_tsne_2d[:, 1], c='orange', alpha=0.3, s=10, label='Student')
    ax.set_title('Student vs Codebook (t-SNE)')
    ax.legend()

    plt.tight_layout()
    tsne_path = os.path.join(save_dir, 'embedding_tsne.png')
    plt.savefig(tsne_path, dpi=150)
    print(f"   - Saved to {tsne_path}")
    plt.close()

    # ========== 距離分布圖 ==========
    print("\n3. 繪製距離分布...")

    distances = torch.cdist(student_emb[indices], codebook)
    correct_distances = distances[torch.arange(n_samples), teacher_codes[indices]]
    min_distances, _ = distances.min(dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 圖 1: 到正確 token 的距離分布
    ax = axes[0]
    ax.hist(correct_distances.numpy(), bins=50, alpha=0.7, label='To correct token')
    ax.hist(min_distances.numpy(), bins=50, alpha=0.7, label='To nearest token')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()

    # 圖 2: 正確 token 的排名分布
    ax = axes[1]
    ranks = (distances <= correct_distances.unsqueeze(1)).sum(dim=1).numpy()
    ax.hist(ranks, bins=50, alpha=0.7)
    ax.axvline(x=1, color='green', linestyle='--', label='Rank 1 (correct)')
    ax.set_xlabel('Rank of correct token')
    ax.set_ylabel('Count')
    ax.set_title(f'Rank Distribution (Top-1 Acc: {(ranks==1).mean()*100:.2f}%)')
    ax.legend()

    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'distance_distribution.png')
    plt.savefig(dist_path, dpi=150)
    print(f"   - Saved to {dist_path}")
    plt.close()

    return {
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }


def analyze_subspace(student_emb, codebook):
    """
    分析 student embedding 是否被困在子空間中
    """
    print("\n" + "="*80)
    print("子空間分析")
    print("="*80)

    # 使用 PCA 分析主成分
    pca_student = PCA(n_components=min(50, student_emb.shape[1]))
    pca_student.fit(student_emb.numpy())

    pca_codebook = PCA(n_components=min(50, codebook.shape[1]))
    pca_codebook.fit(codebook.numpy())

    # 計算累積 variance
    student_cumvar = np.cumsum(pca_student.explained_variance_ratio_)
    codebook_cumvar = np.cumsum(pca_codebook.explained_variance_ratio_)

    print(f"\n1. PCA 分析:")
    print(f"   Student embedding:")
    print(f"   - 解釋 90% variance 需要的維度: {np.argmax(student_cumvar >= 0.9) + 1}")
    print(f"   - 解釋 95% variance 需要的維度: {np.argmax(student_cumvar >= 0.95) + 1}")
    print(f"   - 解釋 99% variance 需要的維度: {np.argmax(student_cumvar >= 0.99) + 1}")

    print(f"\n   Codebook:")
    print(f"   - 解釋 90% variance 需要的維度: {np.argmax(codebook_cumvar >= 0.9) + 1}")
    print(f"   - 解釋 95% variance 需要的維度: {np.argmax(codebook_cumvar >= 0.95) + 1}")
    print(f"   - 解釋 99% variance 需要的維度: {np.argmax(codebook_cumvar >= 0.99) + 1}")

    # 計算 student 的有效維度 (participation ratio)
    student_var = pca_student.explained_variance_ratio_
    student_pr = 1.0 / np.sum(student_var ** 2)

    codebook_var = pca_codebook.explained_variance_ratio_
    codebook_pr = 1.0 / np.sum(codebook_var ** 2)

    print(f"\n2. 有效維度 (Participation Ratio):")
    print(f"   - Student: {student_pr:.1f}")
    print(f"   - Codebook: {codebook_pr:.1f}")
    print(f"   - 比值: {student_pr / codebook_pr:.2f}")

    if student_pr / codebook_pr < 0.5:
        print(f"\n   ⚠️ Student embedding 的有效維度顯著低於 Codebook!")
        print(f"   → 這表明 Student 可能被困在低維子空間中")
    else:
        print(f"\n   ✓ Student 和 Codebook 的有效維度相近")

    return {
        'student_dims_90': int(np.argmax(student_cumvar >= 0.9) + 1),
        'student_dims_95': int(np.argmax(student_cumvar >= 0.95) + 1),
        'codebook_dims_90': int(np.argmax(codebook_cumvar >= 0.9) + 1),
        'codebook_dims_95': int(np.argmax(codebook_cumvar >= 0.95) + 1),
        'student_participation_ratio': float(student_pr),
        'codebook_participation_ratio': float(codebook_pr),
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose embedding space issues')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='diagnosis_results', help='Output directory')
    parser.add_argument('--max_batches', type=int, default=50, help='Max batches to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # 設定輸出目錄
    output_dir = os.path.join(os.path.dirname(args.checkpoint), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Embedding Space Diagnosis Tool")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")
    print()

    # 載入模型和數據
    model, val_loader, codebook = load_model_and_data(args.checkpoint, args.device)

    # 收集 embeddings
    student_emb, teacher_codes, predictions, codebook = collect_embeddings(
        model, val_loader, codebook, args.device, args.max_batches
    )

    # 運行診斷
    results = {}

    # 1. Mode Collapse 診斷
    results['mode_collapse'] = diagnose_mode_collapse(predictions, teacher_codes)

    # 2. 距離分布分析
    results['distance'] = analyze_distance_distribution(student_emb, teacher_codes, codebook)

    # 3. 子空間分析
    results['subspace'] = analyze_subspace(student_emb, codebook)

    # 4. 可視化
    results['visualization'] = visualize_embedding_space(
        student_emb, teacher_codes, codebook, predictions, output_dir
    )

    # 保存結果
    results_path = os.path.join(output_dir, 'diagnosis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # 總結
    print("\n" + "="*80)
    print("診斷總結")
    print("="*80)
    print(f"\n1. Mode Collapse 診斷: {results['mode_collapse']['diagnosis']}")
    print(f"2. Token Accuracy: {results['distance']['accuracy']*100:.2f}%")
    print(f"3. Top-5 Accuracy: {results['distance']['top5_acc']*100:.2f}%")
    print(f"4. Top-10 Accuracy: {results['distance']['top10_acc']*100:.2f}%")
    print(f"5. Student 有效維度: {results['subspace']['student_participation_ratio']:.1f}")
    print(f"6. Codebook 有效維度: {results['subspace']['codebook_participation_ratio']:.1f}")

    print(f"\n可視化圖片保存在: {output_dir}")
    print("  - embedding_pca.png")
    print("  - embedding_tsne.png")
    print("  - distance_distribution.png")


if __name__ == '__main__':
    main()
