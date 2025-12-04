#!/usr/bin/env python3
"""
驗證 Baseline Token Accuracy

問題：
- exp7/exp8 初始 Token Acc 是 26-30%，遠高於 random baseline (0.024%)
- 這是異常的，需要調查原因

可能原因：
1. LoRA 初始化方式導致模型輸出接近 Teacher
2. Student encoder 初始時就接近 Teacher
3. Token Acc 計算方式有問題
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TeacherStudentModel

def verify_baseline():
    print("=" * 70)
    print("Baseline Token Accuracy 驗證")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 載入模型
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

    # 獲取 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)

    print(f"\nCodebook shape: {codebook.shape}")

    # 測試 1: 使用相同的 audio (noisy = clean)
    print("\n" + "=" * 70)
    print("測試 1: noisy_audio = clean_audio (完全相同)")
    print("=" * 70)

    torch.manual_seed(42)
    audio = torch.randn(4, 24000 * 3).to(device)  # 3 秒

    with torch.no_grad():
        output = model.forward_with_emb(audio, audio)  # 輸入相同!
        student_emb = output['student_emb']
        teacher_codes = output['teacher_codes']

        # 計算 Token Accuracy
        B, C, T = student_emb.shape
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)
        T_code = teacher_codes_2d.shape[1]
        T = min(T, T_code)

        teacher_flat = teacher_codes_2d[:, :T].reshape(-1).long()
        emb_flat = student_emb[:, :, :T].permute(0, 2, 1).reshape(-1, C)

        # 計算距離
        distances = torch.cdist(emb_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        predictions = distances.argmin(dim=-1)
        accuracy = (predictions == teacher_flat).float().mean()

        print(f"Token Accuracy (same audio): {accuracy.item()*100:.2f}%")
        print(f"預期: 接近 100% (因為 student = teacher)")

    # 測試 2: 使用不同的 audio
    print("\n" + "=" * 70)
    print("測試 2: noisy_audio ≠ clean_audio (完全不同)")
    print("=" * 70)

    torch.manual_seed(42)
    noisy_audio = torch.randn(4, 24000 * 3).to(device)
    torch.manual_seed(123)  # 不同 seed
    clean_audio = torch.randn(4, 24000 * 3).to(device)

    with torch.no_grad():
        output = model.forward_with_emb(noisy_audio, clean_audio)
        student_emb = output['student_emb']
        teacher_codes = output['teacher_codes']

        # 計算 Token Accuracy
        B, C, T = student_emb.shape
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)
        T_code = teacher_codes_2d.shape[1]
        T = min(T, T_code)

        teacher_flat = teacher_codes_2d[:, :T].reshape(-1).long()
        emb_flat = student_emb[:, :, :T].permute(0, 2, 1).reshape(-1, C)

        # 計算距離
        distances = torch.cdist(emb_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        predictions = distances.argmin(dim=-1)
        accuracy = (predictions == teacher_flat).float().mean()

        print(f"Token Accuracy (different audio): {accuracy.item()*100:.2f}%")
        print(f"預期: 接近 0.024% (random baseline)")

    # 測試 3: 檢查 LoRA 是否真的是 zero-initialized
    print("\n" + "=" * 70)
    print("測試 3: 檢查 LoRA 初始化")
    print("=" * 70)

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            print(f"{name}:")
            print(f"  shape: {param.shape}")
            print(f"  mean: {param.mean().item():.6f}")
            print(f"  std:  {param.std().item():.6f}")
            print(f"  norm: {param.norm().item():.6f}")

    # 測試 4: 比較 Student 和 Teacher 的輸出
    print("\n" + "=" * 70)
    print("測試 4: Student vs Teacher 輸出差異")
    print("=" * 70)

    torch.manual_seed(42)
    test_audio = torch.randn(2, 24000 * 3).to(device)

    with torch.no_grad():
        # Teacher 輸出
        teacher_emb = model.teacher.feature_extractor.encodec.encoder(test_audio.unsqueeze(1))

        # Student 輸出
        student_emb = model.student.feature_extractor.encodec.encoder(test_audio.unsqueeze(1))

        # 比較
        diff = (student_emb - teacher_emb).abs()
        print(f"Teacher emb shape: {teacher_emb.shape}")
        print(f"Student emb shape: {student_emb.shape}")
        print(f"Difference (abs):")
        print(f"  Mean: {diff.mean().item():.6f}")
        print(f"  Max:  {diff.max().item():.6f}")
        print(f"  Std:  {diff.std().item():.6f}")

        # 相關係數
        teacher_flat = teacher_emb.reshape(-1)
        student_flat = student_emb.reshape(-1)
        correlation = torch.corrcoef(torch.stack([teacher_flat, student_flat]))[0, 1]
        print(f"  Correlation: {correlation.item():.6f}")

        if diff.mean().item() < 0.001:
            print("\n⚠️  Student 和 Teacher 輸出幾乎相同!")
            print("   這說明 LoRA 初始化接近 0，導致初始 Token Acc 很高")

if __name__ == "__main__":
    verify_baseline()
