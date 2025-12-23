#!/usr/bin/env python3
"""
分析 Teacher 和 Student 特徵之間的 Cosine Similarity 分布
用於決定合適的 cosine_weight

分析內容:
1. 使用 exp_1217 的模型架構（已經有 LoRA 配置）
2. 計算隨機初始化 Student 與 Teacher 的 cos_sim 分布
3. 基於分布推薦 cosine_weight
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

import h5py
from pathlib import Path
from exp_1217.models import TeacherStudentConfigurableLoRA


def load_model(device):
    """載入 Teacher-Student 模型"""
    from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=128,
        lora_alpha=256,
        lora_dropout=0.2,
        lora_layers='all_18',
        device=str(device)
    )
    model = model.to(device)
    return model


def extract_features(model, noisy_audio, clean_audio):
    """提取 Teacher 和 Student 特徵 (encoder output)"""
    if noisy_audio.dim() == 2:
        noisy_audio = noisy_audio.unsqueeze(1)
    if clean_audio.dim() == 2:
        clean_audio = clean_audio.unsqueeze(1)

    with torch.no_grad():
        # Student: 處理 noisy audio (使用 encoder)
        student_feat = model.student.feature_extractor.encodec.encoder(noisy_audio)

        # Teacher: 處理 clean audio (使用 encoder)
        teacher_feat = model.teacher.feature_extractor.encodec.encoder(clean_audio)

    # 轉換為 [batch, time, dim] 格式
    student_feat = student_feat.transpose(1, 2)
    teacher_feat = teacher_feat.transpose(1, 2)

    return student_feat, teacher_feat


def compute_cosine_similarity(feat1, feat2):
    """計算兩個特徵之間的 cosine similarity"""
    # feat1, feat2: [batch, time, dim]
    # 計算每個 time step 的 cosine similarity
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)
    cos_sim = (feat1_norm * feat2_norm).sum(dim=-1)  # [batch, time]
    return cos_sim


def analyze_distribution(cos_sim_values, title):
    """分析 cosine similarity 分布"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    cos_sim = np.array(cos_sim_values)

    print(f"\n基本統計:")
    print(f"  樣本數: {len(cos_sim)}")
    print(f"  Mean:   {cos_sim.mean():.4f}")
    print(f"  Std:    {cos_sim.std():.4f}")
    print(f"  Min:    {cos_sim.min():.4f}")
    print(f"  Max:    {cos_sim.max():.4f}")

    # 分位數
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n分位數分布:")
    for p in percentiles:
        val = np.percentile(cos_sim, p)
        print(f"  P{p:02d}: {val:.4f}")

    # 區間分布
    print(f"\n區間分布:")
    ranges = [(-1, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in ranges:
        count = ((cos_sim >= low) & (cos_sim < high)).sum()
        pct = count / len(cos_sim) * 100
        print(f"  [{low:+.1f}, {high:+.1f}): {count:6d} ({pct:5.1f}%)")

    return cos_sim.mean(), cos_sim.std()


def add_noise(clean_audio, snr_db=10):
    """添加噪聲到音頻"""
    noise = torch.randn_like(clean_audio)
    # 計算信號和噪聲的功率
    signal_power = (clean_audio ** 2).mean()
    noise_power = (noise ** 2).mean()
    # 根據 SNR 調整噪聲幅度
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
    noisy_audio = clean_audio + noise * noise_scale
    return noisy_audio


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入模型
    print("\n載入模型...")
    model = load_model(device)

    all_cos_sim = []

    print("\n分析 Clean vs Noisy 音頻的 Cosine Similarity...")
    print("(模擬實際訓練場景: Student 處理 noisy, Teacher 處理 clean)")

    n_samples = 50
    audio_length = 24000 * 2  # 2 秒音頻

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  處理中: {i+1}/{n_samples}")

        # 生成隨機 "clean" 音頻
        clean_audio = torch.randn(1, audio_length).to(device) * 0.3

        # 添加噪聲生成 "noisy" 音頻 (SNR=10dB)
        noisy_audio = add_noise(clean_audio, snr_db=10)

        # Student 處理 noisy，Teacher 處理 clean
        student_feat, teacher_feat = extract_features(model, noisy_audio, clean_audio)

        # 計算 cosine similarity
        cos_sim = compute_cosine_similarity(student_feat, teacher_feat)
        all_cos_sim.extend(cos_sim.cpu().numpy().flatten().tolist())

    # 分析分布
    mean_cos, std_cos = analyze_distribution(
        all_cos_sim,
        "Clean vs Noisy 音頻的 Cosine Similarity 分布 (初始化)"
    )

    # 推薦 cosine_weight
    print("\n" + "="*60)
    print("Cosine Weight 推薦分析")
    print("="*60)

    initial_cosine_loss = 1 - mean_cos

    print(f"""
分析結論:

1. Clean vs Noisy 音頻的初始 cos_sim:
   - Mean = {mean_cos:.4f}
   - Std = {std_cos:.4f}

2. Cosine Loss 的目標:
   - 訓練 Student 使其 noisy 音頻特徵對齊 Teacher 的 clean 特徵
   - Loss = 1 - cos_sim
   - 初始 Cosine Loss ≈ {initial_cosine_loss:.4f}

3. 與其他 Loss 的相對規模比較:
   - Feature MSE Loss: 通常在 0.5-2.0 範圍
   - Triplet Loss: 通常在 0.1-0.5 範圍
   - Cosine Loss (初始): ≈ {initial_cosine_loss:.4f}
""")

    # 基於分析推薦
    if initial_cosine_loss > 0.5:
        recommended_weight = 0.1
        print(f"4. 推薦 cosine_weight = {recommended_weight}")
        print(f"   - 初始 Cosine Loss 較高 ({initial_cosine_loss:.4f})")
        print(f"   - 需要較低權重避免過度主導總 loss")
    elif initial_cosine_loss > 0.3:
        recommended_weight = 0.2
        print(f"4. 推薦 cosine_weight = {recommended_weight}")
        print(f"   - 初始 Cosine Loss 中等 ({initial_cosine_loss:.4f})")
    elif initial_cosine_loss > 0.1:
        recommended_weight = 0.3
        print(f"4. 推薦 cosine_weight = {recommended_weight}")
        print(f"   - 初始 Cosine Loss 較低 ({initial_cosine_loss:.4f})")
    else:
        recommended_weight = 0.5
        print(f"4. 推薦 cosine_weight = {recommended_weight}")
        print(f"   - 初始 Cosine Loss 很低 ({initial_cosine_loss:.4f})")
        print(f"   - 可使用較高權重")

    print(f"""
5. Exp51 (cosine_weight=0.5) 失敗原因分析:
   - Cosine Loss 對總 loss 的貢獻 ≈ 0.5 × {initial_cosine_loss:.4f} = {0.5 * initial_cosine_loss:.4f}
   - 相對於 Feature Loss (~1.0)，Cosine Loss 佔比可能過高
   - 導致模型過度優化方向對齊，忽略 token accuracy

6. 建議修改 exp51:
   - cosine_weight: 0.5 → {recommended_weight}
""")


if __name__ == "__main__":
    main()
