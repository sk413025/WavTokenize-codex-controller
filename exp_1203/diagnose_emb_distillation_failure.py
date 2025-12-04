#!/usr/bin/env python3
"""
診斷 EmbDistillation 失敗原因

問題：Loss 下降但 Token Accuracy 也下降
假設：
1. Codebook 空間太大 (4096 codes)
2. 即使 MSE 減少，student 仍在錯誤的 Voronoi 區域
3. LoRA 容量不足

這個腳本會分析：
1. 每個 token 位置的距離分布
2. Voronoi 區域邊界距離
3. 正確 vs 錯誤預測的距離差異
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TeacherStudentModel


def analyze_distance_distribution():
    """分析 student embedding 到 codebook 的距離分布"""

    print("=" * 60)
    print("EmbDistillation 失敗診斷")
    print("=" * 60)

    # 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 使用與訓練相同的配置
    wavtok_config = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    wavtok_ckpt = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

    model = TeacherStudentModel(
        wavtok_config=wavtok_config,
        wavtok_ckpt=wavtok_ckpt,
        lora_rank=64,
        lora_alpha=128,
        device=device
    )
    model.eval()

    # 獲取 codebook (直接從 teacher 存取)
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)
    print(f"Codebook shape: {codebook.shape}")  # [4096, 512]

    # 計算 codebook 內部統計
    codebook_norms = torch.norm(codebook, dim=1)
    print(f"\nCodebook embedding norms:")
    print(f"  Mean: {codebook_norms.mean():.4f}")
    print(f"  Std:  {codebook_norms.std():.4f}")
    print(f"  Min:  {codebook_norms.min():.4f}")
    print(f"  Max:  {codebook_norms.max():.4f}")

    # 計算 codebook 之間的距離
    print("\n計算 codebook 內部距離...")
    # 隨機抽樣 100 個計算
    idx = torch.randperm(4096)[:100]
    sample_codebook = codebook[idx]

    # 計算兩兩距離
    pairwise_dist = torch.cdist(sample_codebook, sample_codebook)
    # 取上三角
    triu_idx = torch.triu_indices(100, 100, offset=1)
    distances = pairwise_dist[triu_idx[0], triu_idx[1]]

    print(f"\nCodebook 內部距離 (抽樣 100 個 code):")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std:  {distances.std():.4f}")
    print(f"  Min:  {distances.min():.4f}")
    print(f"  Max:  {distances.max():.4f}")
    print(f"  Median: {distances.median():.4f}")

    # 載入測試數據
    print("\n" + "=" * 60)
    print("載入測試數據...")

    from data import create_dataloaders

    # 創建簡單的 config
    class SimpleConfig:
        batch_size = 8
        num_workers = 0
        pin_memory = False
        num_samples = 50  # 只用少量樣本診斷

    _, val_loader = create_dataloaders(SimpleConfig())

    # 取一個 batch
    batch = next(iter(val_loader))
    noisy_audio = batch['noisy_audio'].to(device)
    clean_audio = batch['clean_audio'].to(device)

    print(f"Batch size: {noisy_audio.shape[0]}")
    print(f"Audio length: {noisy_audio.shape[1]}")

    # Forward pass
    print("\n" + "=" * 60)
    print("執行 Forward Pass...")

    with torch.no_grad():
        output = model.forward_with_emb(noisy_audio, clean_audio)

    student_emb = output['student_emb']  # [B, D, T]
    teacher_codes = output['teacher_codes']  # [B, Q, T] or [B, T]

    print(f"Student embedding shape: {student_emb.shape}")
    print(f"Teacher codes shape: {teacher_codes.shape}")

    # 轉換維度 [B, D, T] -> [B, T, D]
    B, D, T = student_emb.shape
    student_emb_flat = student_emb.permute(0, 2, 1).reshape(-1, D)  # [B*T, D]

    # 處理 teacher_codes - 形狀是 [Q, B, T] = [1, 8, 225]
    if teacher_codes.dim() == 3:
        # teacher_codes: [Q, B, T] -> 取 Q=0，變成 [B, T] -> flatten
        teacher_codes_flat = teacher_codes[0, :, :].reshape(-1)  # [B*T]
    else:
        teacher_codes_flat = teacher_codes.reshape(-1)

    print(f"Student embedding flat: {student_emb_flat.shape}")
    print(f"Teacher codes flat: {teacher_codes_flat.shape}")

    # 計算 student embedding 到所有 codebook 的距離
    print("\n" + "=" * 60)
    print("計算距離分布...")

    # 計算到所有 codebook 的距離 [B*T, 4096]
    distances_to_all = torch.cdist(student_emb_flat, codebook)

    # 找到最近的 code (student 的選擇)
    student_codes = torch.argmin(distances_to_all, dim=1)

    # 計算準確率
    correct = (student_codes == teacher_codes_flat.long()).float()
    accuracy = correct.mean().item()

    print(f"\nToken Accuracy (student argmin): {accuracy*100:.2f}%")

    # 分析正確 vs 錯誤的距離
    print("\n" + "=" * 60)
    print("正確 vs 錯誤預測分析")

    correct_mask = (student_codes == teacher_codes_flat.long())
    wrong_mask = ~correct_mask

    print(f"正確預測數量: {correct_mask.sum().item()}")
    print(f"錯誤預測數量: {wrong_mask.sum().item()}")

    # 到 teacher code 的距離
    teacher_indices = teacher_codes_flat.long()
    dist_to_teacher = distances_to_all[torch.arange(len(teacher_indices)), teacher_indices]

    # 到 student code 的距離 (最近距離)
    dist_to_student = distances_to_all.min(dim=1)[0]

    print(f"\n到 Teacher code 的距離:")
    print(f"  Mean (所有): {dist_to_teacher.mean():.4f}")
    if correct_mask.sum() > 0:
        print(f"  Mean (正確): {dist_to_teacher[correct_mask].mean():.4f}")
    if wrong_mask.sum() > 0:
        print(f"  Mean (錯誤): {dist_to_teacher[wrong_mask].mean():.4f}")

    print(f"\n到最近 code 的距離:")
    print(f"  Mean (所有): {dist_to_student.mean():.4f}")
    if correct_mask.sum() > 0:
        print(f"  Mean (正確): {dist_to_student[correct_mask].mean():.4f}")
    if wrong_mask.sum() > 0:
        print(f"  Mean (錯誤): {dist_to_student[wrong_mask].mean():.4f}")

    # 關鍵指標：teacher code 是第幾近的？
    print("\n" + "=" * 60)
    print("Teacher code 排名分析")

    # 對每個 token，計算 teacher code 在距離排名中的位置
    sorted_indices = torch.argsort(distances_to_all, dim=1)  # [B*T, 4096]

    ranks = []
    for i in range(len(teacher_indices)):
        rank = (sorted_indices[i] == teacher_indices[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)

    print(f"\nTeacher code 在距離排名中的位置:")
    print(f"  Mean: {ranks.mean():.2f}")
    print(f"  Median: {ranks.median():.2f}")
    print(f"  Min: {ranks.min():.0f}")
    print(f"  Max: {ranks.max():.0f}")
    print(f"  Rank=0 (正確): {(ranks == 0).sum().item()} ({(ranks == 0).float().mean()*100:.2f}%)")
    print(f"  Rank<=10: {(ranks <= 10).sum().item()} ({(ranks <= 10).float().mean()*100:.2f}%)")
    print(f"  Rank<=100: {(ranks <= 100).sum().item()} ({(ranks <= 100).float().mean()*100:.2f}%)")

    # Voronoi 邊界分析
    print("\n" + "=" * 60)
    print("Voronoi 邊界分析")

    # 計算到最近和次近的距離差 (margin)
    top2_dist, top2_idx = torch.topk(distances_to_all, k=2, dim=1, largest=False)
    margin = top2_dist[:, 1] - top2_dist[:, 0]  # 次近 - 最近

    print(f"\nMargin (次近距離 - 最近距離):")
    print(f"  Mean: {margin.mean():.4f}")
    print(f"  Std:  {margin.std():.4f}")
    print(f"  Min:  {margin.min():.4f}")
    print(f"  Max:  {margin.max():.4f}")

    # 小 margin 代表在邊界附近，容易切換
    small_margin = (margin < 0.01).sum().item()
    print(f"  Margin < 0.01: {small_margin} ({small_margin/len(margin)*100:.2f}%)")

    # 計算 teacher code 與最近 code 之間的距離
    print("\n" + "=" * 60)
    print("Teacher code vs Student code 距離")

    teacher_embeddings = codebook[teacher_indices]  # [B*T, D]
    student_embeddings = codebook[student_codes]  # [B*T, D]

    code_distance = torch.norm(teacher_embeddings - student_embeddings, dim=1)

    print(f"\nTeacher code 與 Student code 在 codebook 空間的距離:")
    print(f"  Mean: {code_distance.mean():.4f}")
    if wrong_mask.sum() > 0:
        print(f"  Mean (錯誤時): {code_distance[wrong_mask].mean():.4f}")

    # 結論
    print("\n" + "=" * 60)
    print("診斷結論")
    print("=" * 60)

    if ranks.mean() < 100:
        print("\n✓ Teacher code 排名通常在前 100 名")
        print("  這表示 student embedding 離 teacher code 「不算太遠」")
        print("  但在高維空間中，附近有太多競爭的 codes")

    if margin.mean() < 0.05:
        print("\n⚠ Margin 很小，embedding 在 Voronoi 邊界附近")
        print("  小的 perturbation 就會改變 argmin 結果")

    avg_codebook_dist = distances.mean().item()
    avg_mse_loss = dist_to_teacher.mean().item()

    print(f"\n📊 關鍵比值:")
    print(f"  Codebook 內部平均距離: {avg_codebook_dist:.4f}")
    print(f"  Student 到 Teacher code 平均距離: {avg_mse_loss:.4f}")
    print(f"  比值: {avg_mse_loss/avg_codebook_dist*100:.2f}%")

    if avg_mse_loss > avg_codebook_dist * 0.1:
        print("\n⚠ Student embedding 距離 Teacher code 太遠")
        print("  需要更大幅度的修正才能選對 code")

    # 建議
    print("\n" + "=" * 60)
    print("建議")
    print("=" * 60)

    print("""
1. 問題根源：4096 個 codebook 在 512 維空間
   - 平均每個 code 的 Voronoi 區域很小
   - 即使 MSE loss 下降，student 可能仍在錯誤區域

2. 可能的解決方案：
   a) 增加 LoRA rank (64 → 256)
   b) 訓練更多 encoder 層
   c) 使用 Contrastive Loss 而非 MSE
   d) 直接使用 CE Loss (分類目標)
   e) 降維：只使用 codebook 的子集

3. 為何 CE Loss 可能更好：
   - CE Loss 直接優化「選對 code」的概率
   - MSE Loss 只優化「距離」，不保證選對
    """)

    return {
        'accuracy': accuracy,
        'dist_to_teacher_mean': dist_to_teacher.mean().item(),
        'dist_to_student_mean': dist_to_student.mean().item(),
        'rank_mean': ranks.mean().item(),
        'margin_mean': margin.mean().item(),
        'codebook_dist_mean': avg_codebook_dist,
    }


if __name__ == "__main__":
    results = analyze_distance_distribution()

    # 保存結果
    output_path = Path(__file__).parent / "emb_distillation_diagnosis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已保存至: {output_path}")
