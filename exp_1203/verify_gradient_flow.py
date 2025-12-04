#!/usr/bin/env python3
"""
驗證梯度回傳問題：VQ 前 vs VQ 後

這個腳本直接比較：
1. Feature Loss (VQ 後) → encoder 的梯度
2. EmbDistillation (VQ 前) → encoder 的梯度

預期結果：
- Feature Loss 的梯度可能被 VQ 的 stop_gradient 阻斷
- EmbDistillation 的梯度應該正常回傳
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TeacherStudentModel


def verify_gradient_flow():
    print("=" * 70)
    print("梯度回傳驗證：VQ 前 vs VQ 後")
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

    # 確認 LoRA 參數可訓練
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower()]
    print(f"\nLoRA parameters: {len(lora_params)}")
    print(f"LoRA params require_grad: {all(p.requires_grad for p in lora_params)}")

    # 獲取 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)
    print(f"Codebook shape: {codebook.shape}")

    # 創建測試 audio
    torch.manual_seed(42)
    noisy_audio = torch.randn(2, 24000 * 3).to(device)  # 3 秒
    clean_audio = torch.randn(2, 24000 * 3).to(device)

    print("\n" + "=" * 70)
    print("測試 1: Feature Loss (VQ 後) 的梯度")
    print("=" * 70)

    model.zero_grad()
    model.train()  # 確保是訓練模式

    # Forward - 使用標準 forward (返回量化後的 features)
    output = model(noisy_audio, clean_audio)
    student_features = output['student_features']  # VQ 後
    teacher_features = output['teacher_features']  # VQ 後

    print(f"\nStudent features shape: {student_features.shape}")
    print(f"Student features requires_grad: {student_features.requires_grad}")
    print(f"Student features grad_fn: {student_features.grad_fn}")

    # Feature Loss
    feature_loss = F.mse_loss(student_features, teacher_features)
    print(f"Feature Loss: {feature_loss.item():.6f}")
    print(f"Feature Loss requires_grad: {feature_loss.requires_grad}")

    lora_grads_feature = []
    if not feature_loss.requires_grad:
        print("\n❌ Feature Loss requires_grad = False!")
        print("   → 這說明 feature_extractor 內部使用了 detach() 或 no_grad()")
        print("   → 梯度無法從 VQ 後的 features 回傳到 encoder")
        print("   → 這是 exp7 失敗的原因之一")
    else:
        # Backward
        feature_loss.backward()

        # 檢查 LoRA 梯度
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.grad is not None:
                lora_grads_feature.append((name, param.grad.norm().item()))

        print(f"\nLoRA params with gradients: {len(lora_grads_feature)}")
        if lora_grads_feature:
            print("Sample gradients (first 3):")
            for name, grad_norm in lora_grads_feature[:3]:
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
            avg_grad = sum(g for _, g in lora_grads_feature) / len(lora_grads_feature)
            print(f"Average gradient norm: {avg_grad:.6f}")
        else:
            print("❌ NO GRADIENTS! Feature Loss 梯度無法回傳到 encoder")

    print("\n" + "=" * 70)
    print("測試 2: EmbDistillation (VQ 前) 的梯度")
    print("=" * 70)

    model.zero_grad()

    # Forward - 使用 forward_with_emb (返回 VQ 前的 emb)
    output = model.forward_with_emb(noisy_audio, clean_audio)
    student_emb = output['student_emb']  # VQ 前!
    teacher_codes = output['teacher_codes']

    print(f"\nStudent emb shape: {student_emb.shape}")
    print(f"Student emb requires_grad: {student_emb.requires_grad}")

    # 處理 teacher_codes
    B, C, T = student_emb.shape
    if teacher_codes.dim() == 3:
        teacher_codes_2d = teacher_codes[0]
    else:
        teacher_codes_2d = teacher_codes.squeeze(1)
    T_code = teacher_codes_2d.shape[1]
    T = min(T, T_code)

    # EmbDistillation Loss
    teacher_flat = teacher_codes_2d[:, :T].reshape(-1).long()
    target_embeddings = codebook[teacher_flat]

    emb_truncated = student_emb[:, :, :T]
    emb_flat = emb_truncated.permute(0, 2, 1).reshape(-1, C)

    emb_loss = F.mse_loss(emb_flat, target_embeddings)
    print(f"EmbDistillation Loss: {emb_loss.item():.6f}")
    print(f"EmbDistillation Loss requires_grad: {emb_loss.requires_grad}")

    # Backward
    emb_loss.backward()

    # 檢查 LoRA 梯度
    lora_grads_emb = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.grad is not None:
            lora_grads_emb.append((name, param.grad.norm().item()))

    print(f"\nLoRA params with gradients: {len(lora_grads_emb)}")
    if lora_grads_emb:
        print("Sample gradients (first 3):")
        for name, grad_norm in lora_grads_emb[:3]:
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
        avg_grad = sum(g for _, g in lora_grads_emb) / len(lora_grads_emb)
        print(f"Average gradient norm: {avg_grad:.6f}")
    else:
        print("❌ NO GRADIENTS! EmbDistillation 梯度無法回傳到 encoder")

    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)

    feature_has_grad = len(lora_grads_feature) > 0
    emb_has_grad = len(lora_grads_emb) > 0

    print(f"\n結果對比:")
    print(f"  Feature Loss (VQ 後):    {'✅ 有梯度' if feature_has_grad else '❌ 無梯度'}")
    print(f"  EmbDistillation (VQ 前): {'✅ 有梯度' if emb_has_grad else '❌ 無梯度'}")

    if feature_has_grad and emb_has_grad:
        # 比較梯度大小
        avg_feature = sum(g for _, g in lora_grads_feature) / len(lora_grads_feature)
        avg_emb = sum(g for _, g in lora_grads_emb) / len(lora_grads_emb)
        ratio = avg_emb / avg_feature if avg_feature > 0 else float('inf')

        print(f"\n梯度大小比較:")
        print(f"  Feature Loss 平均梯度:    {avg_feature:.6f}")
        print(f"  EmbDistillation 平均梯度: {avg_emb:.6f}")
        print(f"  比例 (Emb/Feature):       {ratio:.2f}x")

        if ratio > 10:
            print(f"\n⚠️  EmbDistillation 梯度明顯更強 ({ratio:.1f}x)")
            print(f"   這說明 VQ 確實會削弱梯度傳遞")
        elif ratio < 0.1:
            print(f"\n⚠️  Feature Loss 梯度明顯更強 ({1/ratio:.1f}x)")
        else:
            print(f"\n✅ 兩種方法梯度大小相近")
    elif not feature_has_grad and emb_has_grad:
        print("\n⚠️  證實問題：Feature Loss 梯度被 VQ 阻斷!")
        print("   EmbDistillation 可以正常傳遞梯度")
    elif feature_has_grad and not emb_has_grad:
        print("\n⚠️  意外結果：Feature Loss 有梯度但 EmbDistillation 沒有")
        print("   需要進一步檢查 forward_with_emb 實現")
    else:
        print("\n❌ 兩種方法都沒有梯度，需要檢查模型設置")


def check_vq_gradient_path():
    """詳細檢查 VQ 層的梯度路徑"""
    print("\n" + "=" * 70)
    print("VQ 層梯度路徑分析")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 獲取 VQ 層
    vq_layer = model.student.feature_extractor.encodec.quantizer.vq.layers[0]
    print(f"\nVQ Layer type: {type(vq_layer)}")

    # 檢查 VQ 的 forward 方法
    print("\n檢查 VQ forward 中的操作:")

    # 模擬 VQ forward
    codebook = vq_layer._codebook.embed.to(device)
    print(f"Codebook shape: {codebook.shape}")

    # 創建測試輸入
    test_input = torch.randn(2, 512, 100, requires_grad=True, device=device)
    print(f"Test input shape: {test_input.shape}")

    # 計算距離
    input_flat = test_input.permute(0, 2, 1).reshape(-1, 512)  # [200, 512]
    distances = torch.cdist(input_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)

    # argmin (這是不可微的操作!)
    indices = distances.argmin(dim=-1)
    print(f"Indices shape: {indices.shape}")

    # 獲取量化結果
    quantized_flat = codebook[indices]
    print(f"Quantized flat shape: {quantized_flat.shape}")

    # Straight-Through Estimator (STE)
    # quantized = input + (quantized - input).detach()
    # 這個操作讓 forward 用 quantized，backward 用 input 的梯度
    quantized_ste = input_flat + (quantized_flat - input_flat).detach()
    print(f"\nSTE: quantized = input + (quantized - input).detach()")
    print(f"quantized_ste requires_grad: {quantized_ste.requires_grad}")

    # 計算一個簡單的 loss
    loss = quantized_ste.mean()
    print(f"\nLoss: {loss.item():.6f}")

    # Backward
    loss.backward()
    print(f"Test input has gradient: {test_input.grad is not None}")
    if test_input.grad is not None:
        print(f"Test input gradient norm: {test_input.grad.norm().item():.6f}")

    print("\n分析:")
    print("  VQ 使用 STE: forward 用 quantized，backward 梯度傳回 input")
    print("  所以理論上 Feature Loss 應該有梯度傳回 encoder")
    print("  但梯度的「方向」可能不是最優的（因為跳過了 argmin 的選擇邏輯）")


if __name__ == "__main__":
    verify_gradient_flow()
    check_vq_gradient_path()
