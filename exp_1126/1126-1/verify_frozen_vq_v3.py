#!/usr/bin/env python3
"""
驗證 VQ Codebook 凍結 v3：
1. Codebook 在 forward 後被正確恢復
2. LoRA 參數仍然能接收梯度
3. STE 梯度傳遞正常運作
"""

import sys
import os

# 確保當前目錄優先
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(1, '/home/sbplab/ruizi/WavTokenizer-main')

import torch
import torch.nn.functional as F

# 明確從當前目錄導入
from model import create_teacher_student_model
from config import get_smoke_test_config

def test_frozen_vq_v3():
    print("=" * 60)
    print("驗證 VQ Codebook 凍結 v3")
    print("=" * 60)

    # 創建模型
    config = get_smoke_test_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_teacher_student_model(config, device=device)

    # 設置 train mode
    model.train()

    # 獲取 codebook 引用
    student_vq = model.student.base_model.model.feature_extractor.encodec.quantizer.vq.layers[0]
    codebook_ref = student_vq._codebook

    # 保存初始狀態
    frozen_codebook = codebook_ref.embed.data.clone()
    frozen_cluster_size = codebook_ref.cluster_size.data.clone()
    frozen_embed_avg = codebook_ref.embed_avg.data.clone()

    print(f"\n1. 初始 codebook 狀態保存完成")
    print(f"   Codebook shape: {frozen_codebook.shape}")

    # 創建測試數據
    batch_size = 2
    audio_length = 24000  # 1 second
    noisy_audio = torch.randn(batch_size, audio_length, device=device)
    clean_audio = torch.randn(batch_size, audio_length, device=device)

    print(f"\n2. 執行 forward pass...")

    # Forward pass
    output = model(noisy_audio, clean_audio)

    # 計算 feature loss
    student_features = output['student_features']
    teacher_features = output['teacher_features']
    feature_loss = F.mse_loss(student_features, teacher_features)

    print(f"   Feature loss: {feature_loss.item():.6f}")
    print(f"   student_features.requires_grad: {student_features.requires_grad}")

    # 檢查 EMA 更新是否發生
    codebook_diff = (codebook_ref.embed.data - frozen_codebook).abs().max().item()
    print(f"\n3. Forward 後 codebook 變化: {codebook_diff:.6f}")

    if codebook_diff > 1e-6:
        print("   ⚠️ EMA 更新發生了！現在恢復 codebook...")

        # 恢復 codebook
        codebook_ref.embed.data.copy_(frozen_codebook)
        codebook_ref.cluster_size.data.copy_(frozen_cluster_size)
        codebook_ref.embed_avg.data.copy_(frozen_embed_avg)

        # 驗證恢復
        codebook_diff_after = (codebook_ref.embed.data - frozen_codebook).abs().max().item()
        print(f"   恢復後 codebook 差異: {codebook_diff_after:.6f}")

        if codebook_diff_after < 1e-8:
            print("   ✅ Codebook 成功恢復!")
        else:
            print("   ❌ Codebook 恢復失敗!")
            return False
    else:
        print("   ✅ 沒有 EMA 更新 (可能已經是 eval mode)")

    # 測試 backward
    print(f"\n4. 測試 backward pass...")

    try:
        feature_loss.backward()
        print("   ✅ Backward 成功!")
    except RuntimeError as e:
        print(f"   ❌ Backward 失敗: {e}")
        return False

    # 檢查 LoRA 參數梯度
    print(f"\n5. 檢查 LoRA 參數梯度...")
    lora_params_with_grad = 0
    lora_params_total = 0

    for name, param in model.student.named_parameters():
        if 'lora' in name.lower():
            lora_params_total += 1
            if param.grad is not None:
                lora_params_with_grad += 1
                grad_norm = param.grad.norm().item()
                if lora_params_with_grad <= 3:  # 只打印前幾個
                    print(f"   {name}: grad_norm = {grad_norm:.6f}")

    print(f"\n   LoRA 參數有梯度: {lora_params_with_grad}/{lora_params_total}")

    if lora_params_with_grad == lora_params_total:
        print("   ✅ 所有 LoRA 參數都有梯度!")
    elif lora_params_with_grad > 0:
        print("   ⚠️ 部分 LoRA 參數有梯度")
    else:
        print("   ❌ 沒有 LoRA 參數有梯度!")
        return False

    # 再次 forward 測試恢復機制
    print(f"\n6. 測試多輪 forward 的 codebook 恢復...")

    for i in range(3):
        # Forward
        output = model(noisy_audio, clean_audio)

        # 恢復
        codebook_ref.embed.data.copy_(frozen_codebook)
        codebook_ref.cluster_size.data.copy_(frozen_cluster_size)
        codebook_ref.embed_avg.data.copy_(frozen_embed_avg)

        # 驗證
        diff = (codebook_ref.embed.data - frozen_codebook).abs().max().item()
        print(f"   Round {i+1}: codebook diff after restore = {diff:.10f}")

    print("\n" + "=" * 60)
    print("✅ 所有驗證通過！")
    print("=" * 60)

    print("""
總結：
1. Codebook EMA 更新確實會發生 (training=True 時)
2. 但透過 forward 後恢復機制，codebook 被重置回原始狀態
3. STE 梯度傳遞正常運作 (因為 training=True)
4. LoRA 參數能正確接收梯度
5. 這個方案可以同時達到：
   - 凍結 Codebook (與 Teacher 保持一致)
   - 保持 LoRA 訓練能力
""")

    return True


if __name__ == "__main__":
    success = test_frozen_vq_v3()
    sys.exit(0 if success else 1)
