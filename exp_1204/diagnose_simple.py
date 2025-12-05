"""
exp_1204: 簡化版 Embedding Space 診斷工具

診斷目標:
1. 驗證是情況 A (MSE 下降但預測分散) 還是情況 B (Mode Collapse)
2. 可視化 student embedding 和 codebook 的空間分布
3. 分析 embedding 子空間問題

使用方式:
    cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204
    python diagnose_simple.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import json

# 添加路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

# 設定
CHECKPOINT_PATH = '/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204/experiments/exp11_ce_only/checkpoints/latest.pt'
OUTPUT_DIR = '/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204/diagnosis_results'
DEVICE = 'cuda'
MAX_BATCHES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("Embedding Space Diagnosis Tool (Simplified)")
print("="*80)


# ============================================================================
# 1. 載入模型
# ============================================================================
print("\n[1] Loading model and codebook...")

# 載入 checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint.get('config', {})
print(f"    Config: lora_rank={config.get('lora_rank')}, lora_alpha={config.get('lora_alpha')}")

# 載入模型
from exp_1204.model import TeacherStudentModel
from exp_1204.wavtok_lora_patch import apply_lora_patch
from exp_1204.config import WAVTOK_CONFIG, WAVTOK_CKPT

model = TeacherStudentModel(
    wavtok_config=WAVTOK_CONFIG,
    wavtok_ckpt=WAVTOK_CKPT,
    lora_rank=config.get('lora_rank', 128),
    lora_alpha=config.get('lora_alpha', 256),
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# 獲取 codebook
codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.detach()
print(f"    Codebook shape: {codebook.shape}")  # (4096, 512)

# 載入 distance matrix
dist_mat_path = '/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204/wavtok_distance_mat_corrected.pt'
distance_matrix = torch.load(dist_mat_path, map_location=DEVICE, weights_only=False)
print(f"    Distance matrix shape: {distance_matrix.shape}")


# ============================================================================
# 2. 載入數據
# ============================================================================
print("\n[2] Loading data...")

from exp_1204.data import NoisyCleanPairDataset, collate_fn
from torch.utils.data import DataLoader

cache_path = "/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/data_with_distances/val_cache_with_distances.pt"
dataset = NoisyCleanPairDataset(cache_path=cache_path)
val_loader = DataLoader(
    dataset,
    batch_size=config.get('batch_size', 20),
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)
print(f"    Dataset size: {len(dataset)}")
print(f"    Batch size: {config.get('batch_size', 20)}")


# ============================================================================
# 3. 收集 Embeddings
# ============================================================================
print(f"\n[3] Collecting embeddings from {MAX_BATCHES} batches...")

all_student_emb = []
all_teacher_codes = []
all_student_predictions = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= MAX_BATCHES:
            break

        noisy_audio = batch['noisy_audio'].to(DEVICE)
        clean_audio = batch['clean_audio'].to(DEVICE)

        # Forward pass using forward_with_emb to get student_emb
        output = model.forward_with_emb(noisy_audio, clean_audio, compute_vq_features=False)
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
            print(f"    Processed {batch_idx + 1}/{MAX_BATCHES} batches")

all_student_emb = torch.cat(all_student_emb, dim=0)
all_teacher_codes = torch.cat(all_teacher_codes, dim=0)
all_student_predictions = torch.cat(all_student_predictions, dim=0)
codebook_cpu = codebook.cpu()

print(f"    Total samples collected: {len(all_student_emb)}")


# ============================================================================
# 4. Mode Collapse 診斷
# ============================================================================
print("\n" + "="*80)
print("Mode Collapse 診斷")
print("="*80)

pred_counter = Counter(all_student_predictions.numpy().tolist())
teacher_counter = Counter(all_teacher_codes.numpy().tolist())

unique_preds = len(pred_counter)
unique_teachers = len(teacher_counter)
num_tokens = 4096
total_samples = len(all_student_predictions)

print(f"\n1. Token 多樣性分析:")
print(f"   - Student 預測的 unique token 數量: {unique_preds} / {num_tokens}")
print(f"   - Teacher 的 unique token 數量: {unique_teachers} / {num_tokens}")
print(f"   - 覆蓋率: {unique_preds / num_tokens * 100:.2f}%")

# Top-K 分析
top_k = 20
most_common_preds = pred_counter.most_common(top_k)
top_k_coverage = sum([count for _, count in most_common_preds]) / total_samples * 100

print(f"\n2. Top-{top_k} 最常預測的 token:")
print(f"   - Top-{top_k} token 覆蓋了 {top_k_coverage:.2f}% 的預測")

for i, (token, count) in enumerate(most_common_preds[:10]):
    teacher_count = teacher_counter.get(token, 0)
    print(f"   [{i+1}] Token {token}: {count} ({count/total_samples*100:.2f}%), Teacher中有 {teacher_count}")

# Entropy 分析
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
    diagnosis = "嚴重 Mode Collapse"
    print(f"   ⚠️ 嚴重 Mode Collapse！預測只集中在 {unique_preds} 個 token")
elif unique_preds < 500:
    diagnosis = "輕度 Mode Collapse"
    print(f"   ⚠️ 輕度 Mode Collapse，預測集中在 {unique_preds} 個 token")
elif top_k_coverage > 50:
    diagnosis = "部分 Mode Collapse"
    print(f"   ⚠️ 部分 Mode Collapse，Top-{top_k} 覆蓋了 {top_k_coverage:.1f}%")
else:
    diagnosis = "情況 A (預測分散但錯誤)"
    print(f"   ✓ 預測較分散，不是典型的 Mode Collapse")
    print(f"   → 可能是情況 A: MSE 下降但預測分散到錯誤的 token")


# ============================================================================
# 5. 距離分布分析
# ============================================================================
print("\n" + "="*80)
print("距離分布分析")
print("="*80)

distances = torch.cdist(all_student_emb, codebook_cpu)  # (N, 4096)
correct_distances = distances[torch.arange(len(all_teacher_codes)), all_teacher_codes]
min_distances, min_indices = distances.min(dim=1)

correct_mask = (min_indices == all_teacher_codes)
accuracy = correct_mask.float().mean().item()

print(f"\n1. 基本統計:")
print(f"   - Token Accuracy: {accuracy * 100:.2f}%")
print(f"   - 到正確 token 的平均距離: {correct_distances.mean():.4f}")
print(f"   - 到最近 token 的平均距離: {min_distances.mean():.4f}")

# 排名分析
ranks = (distances <= correct_distances.unsqueeze(1)).sum(dim=1).float()

print(f"\n2. 正確 token 的排名分析:")
print(f"   - 平均排名: {ranks.mean():.1f}")
print(f"   - 中位數排名: {ranks.median():.1f}")
print(f"   - Top-1 (Acc): {(ranks == 1).float().mean() * 100:.2f}%")
print(f"   - Top-5: {(ranks <= 5).float().mean() * 100:.2f}%")
print(f"   - Top-10: {(ranks <= 10).float().mean() * 100:.2f}%")
print(f"   - Top-100: {(ranks <= 100).float().mean() * 100:.2f}%")


# ============================================================================
# 6. 子空間分析
# ============================================================================
print("\n" + "="*80)
print("子空間分析")
print("="*80)

pca_student = PCA(n_components=50)
pca_student.fit(all_student_emb.numpy())

pca_codebook = PCA(n_components=50)
pca_codebook.fit(codebook_cpu.numpy())

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

# Participation Ratio
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


# ============================================================================
# 7. 可視化
# ============================================================================
print("\n" + "="*80)
print("生成可視化圖表")
print("="*80)

# 採樣
n_samples = min(5000, len(all_student_emb))
indices = np.random.choice(len(all_student_emb), n_samples, replace=False)

student_sample = all_student_emb[indices].numpy()
teacher_sample = all_teacher_codes[indices].numpy()
pred_sample = all_student_predictions[indices].numpy()

# PCA 降維
print("\n1. PCA 降維...")
combined = np.vstack([student_sample, codebook_cpu.numpy()])
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)

student_2d = combined_2d[:n_samples]
codebook_2d = combined_2d[n_samples:]

# 繪製 PCA 圖
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 圖 1: Student embedding 分布
ax = axes[0]
correct_mask_vis = pred_sample == teacher_sample
ax.scatter(student_2d[~correct_mask_vis, 0], student_2d[~correct_mask_vis, 1],
           c='red', alpha=0.3, s=5, label=f'Wrong ({(~correct_mask_vis).sum()})')
ax.scatter(student_2d[correct_mask_vis, 0], student_2d[correct_mask_vis, 1],
           c='green', alpha=0.3, s=5, label=f'Correct ({correct_mask_vis.sum()})')
ax.set_title('Student Embedding (PCA)\nGreen=Correct, Red=Wrong')
ax.legend()
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

# 圖 2: Codebook 分布
ax = axes[1]
ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c='blue', alpha=0.1, s=5)
ax.set_title('Codebook Distribution (PCA)')
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
pca_path = os.path.join(OUTPUT_DIR, 'embedding_pca.png')
plt.savefig(pca_path, dpi=150)
print(f"   Saved to {pca_path}")
plt.close()

# 距離分布圖
print("\n2. 繪製距離分布...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 圖 1: 距離分布
ax = axes[0]
ax.hist(correct_distances[indices].numpy(), bins=50, alpha=0.7, label='To correct token')
ax.hist(min_distances[indices].numpy(), bins=50, alpha=0.7, label='To nearest token')
ax.set_xlabel('Distance')
ax.set_ylabel('Count')
ax.set_title('Distance Distribution')
ax.legend()

# 圖 2: 排名分布
ax = axes[1]
ranks_sample = ranks[indices].numpy()
ax.hist(ranks_sample, bins=50, alpha=0.7)
ax.axvline(x=1, color='green', linestyle='--', label='Rank 1 (correct)')
ax.set_xlabel('Rank of correct token')
ax.set_ylabel('Count')
ax.set_title(f'Rank Distribution (Top-1 Acc: {(ranks_sample==1).mean()*100:.2f}%)')
ax.legend()

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, 'distance_distribution.png')
plt.savefig(dist_path, dpi=150)
print(f"   Saved to {dist_path}")
plt.close()

# PCA explained variance 圖
print("\n3. 繪製 PCA explained variance...")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
dims = np.arange(1, 51)
ax.plot(dims, student_cumvar, 'b-', label='Student', linewidth=2)
ax.plot(dims, codebook_cumvar, 'r-', label='Codebook', linewidth=2)
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
ax.set_xlabel('Number of PCA dimensions')
ax.set_ylabel('Cumulative explained variance')
ax.set_title('PCA Explained Variance: Student vs Codebook')
ax.legend()
ax.grid(True, alpha=0.3)

pca_var_path = os.path.join(OUTPUT_DIR, 'pca_variance.png')
plt.savefig(pca_var_path, dpi=150)
print(f"   Saved to {pca_var_path}")
plt.close()


# ============================================================================
# 8. 保存結果
# ============================================================================
results = {
    'mode_collapse': {
        'unique_predictions': unique_preds,
        'unique_teachers': unique_teachers,
        'top_k_coverage': top_k_coverage,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': entropy / max_entropy,
        'diagnosis': diagnosis,
    },
    'distance': {
        'accuracy': accuracy,
        'mean_correct_distance': correct_distances.mean().item(),
        'mean_min_distance': min_distances.mean().item(),
        'mean_rank': ranks.mean().item(),
        'median_rank': ranks.median().item(),
        'top5_acc': (ranks <= 5).float().mean().item(),
        'top10_acc': (ranks <= 10).float().mean().item(),
        'top100_acc': (ranks <= 100).float().mean().item(),
    },
    'subspace': {
        'student_dims_90': int(np.argmax(student_cumvar >= 0.9) + 1),
        'student_dims_95': int(np.argmax(student_cumvar >= 0.95) + 1),
        'student_dims_99': int(np.argmax(student_cumvar >= 0.99) + 1),
        'codebook_dims_90': int(np.argmax(codebook_cumvar >= 0.9) + 1),
        'codebook_dims_95': int(np.argmax(codebook_cumvar >= 0.95) + 1),
        'codebook_dims_99': int(np.argmax(codebook_cumvar >= 0.99) + 1),
        'student_participation_ratio': float(student_pr),
        'codebook_participation_ratio': float(codebook_pr),
        'pr_ratio': float(student_pr / codebook_pr),
    },
}

results_path = os.path.join(OUTPUT_DIR, 'diagnosis_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")


# ============================================================================
# 9. 總結
# ============================================================================
print("\n" + "="*80)
print("診斷總結")
print("="*80)

print(f"""
1. Mode Collapse 診斷: {diagnosis}
   - Unique predictions: {unique_preds} / 4096
   - Top-20 coverage: {top_k_coverage:.1f}%
   - Normalized entropy: {entropy / max_entropy * 100:.1f}%

2. Token Accuracy: {accuracy*100:.2f}%
   - Top-5 Accuracy: {(ranks <= 5).float().mean()*100:.2f}%
   - Top-10 Accuracy: {(ranks <= 10).float().mean()*100:.2f}%
   - Top-100 Accuracy: {(ranks <= 100).float().mean()*100:.2f}%
   - Mean rank: {ranks.mean():.1f}

3. 子空間分析:
   - Student 有效維度: {student_pr:.1f}
   - Codebook 有效維度: {codebook_pr:.1f}
   - 比值: {student_pr / codebook_pr:.2f}
   - Student 90% variance 需 {np.argmax(student_cumvar >= 0.9) + 1} 維
   - Codebook 90% variance 需 {np.argmax(codebook_cumvar >= 0.9) + 1} 維

4. 結論:
""")

# 根據結果給出結論
if unique_preds < 500:
    print("   ⚠️ 主要問題: Mode Collapse")
    print("   → 建議: 使用對比學習或增加多樣性損失")
elif student_pr / codebook_pr < 0.5:
    print("   ⚠️ 主要問題: 子空間限制")
    print("   → 建議: 增加 LoRA rank 或使用 Full Fine-tuning")
else:
    print("   ⚠️ 主要問題: MSE 優化目標不匹配")
    print("   → 建議: 使用直接 Token 預測 (Linear projection + CE)")

print(f"\n可視化圖片保存在: {OUTPUT_DIR}")
print("  - embedding_pca.png")
print("  - distance_distribution.png")
print("  - pca_variance.png")
