#!/usr/bin/env python3
"""
快速訓練測試：確認 v3 方案在真實訓練中正常運作
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(1, '/home/sbplab/ruizi/WavTokenizer-main')

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from model import create_teacher_student_model
from config import get_smoke_test_config
from losses import EncoderDistillationLoss

def quick_train_test():
    print("=" * 60)
    print("快速訓練測試 - 驗證 v3 方案")
    print("=" * 60)

    # 創建模型
    config = get_smoke_test_config()
    device = "cuda"
    model = create_teacher_student_model(config, device=device)

    # 創建 loss 函數
    criterion = EncoderDistillationLoss(
        feature_loss_weight=1.0,
        distance_loss_weight=0.01,
        vq_loss_weight=0.0
    )

    # 獲取 distance matrix
    distance_matrix = model.distance_matrix

    # 創建 optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    # 設置 train mode
    model.train()

    # 獲取 codebook 引用並保存初始狀態
    student_vq = model.student.base_model.model.feature_extractor.encodec.quantizer.vq.layers[0]
    codebook_ref = student_vq._codebook

    frozen_codebook = codebook_ref.embed.data.clone()
    frozen_cluster_size = codebook_ref.cluster_size.data.clone()
    frozen_embed_avg = codebook_ref.embed_avg.data.clone()

    print(f"\n初始 codebook 已保存")

    # 模擬訓練循環
    batch_size = 4
    audio_length = 24000 * 3  # 3 seconds

    print(f"\n開始訓練測試 (5 steps)...")
    print("-" * 60)

    for step in range(5):
        # 創建假數據（模擬 noisy 和 clean）
        # 讓 clean 和 noisy 有一定差異以產生有意義的 loss
        clean_audio = torch.randn(batch_size, audio_length, device=device) * 0.1
        noise = torch.randn(batch_size, audio_length, device=device) * 0.05
        noisy_audio = clean_audio + noise

        # Forward
        with autocast(enabled=True):
            output = model(noisy_audio, clean_audio)
            loss, loss_dict = criterion(output, distance_matrix)

        # 恢復 codebook（抵消 EMA 更新）
        codebook_ref.embed.data.copy_(frozen_codebook)
        codebook_ref.cluster_size.data.copy_(frozen_cluster_size)
        codebook_ref.embed_avg.data.copy_(frozen_embed_avg)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 檢查梯度
        lora_grad_norms = []
        for name, param in model.student.named_parameters():
            if 'lora' in name.lower() and param.grad is not None:
                lora_grad_norms.append(param.grad.norm().item())

        # Optimizer step
        optimizer.step()

        # 驗證 codebook 沒有改變
        codebook_diff = (codebook_ref.embed.data - frozen_codebook).abs().max().item()

        print(f"Step {step+1}: loss={loss.item():.4f}, "
              f"feat={loss_dict['feature_loss']:.4f}, "
              f"dist={loss_dict['distance_loss']:.2f}, "
              f"acc={loss_dict['code_match_rate']*100:.1f}%, "
              f"grad_norm={sum(lora_grad_norms)/len(lora_grad_norms):.6f}, "
              f"cb_diff={codebook_diff:.8f}")

    print("-" * 60)
    print("\n" + "=" * 60)
    print("✅ 訓練測試完成！")
    print("=" * 60)

    # 最終驗證
    print("\n最終驗證:")
    print(f"  - Codebook 與初始狀態差異: {codebook_diff:.10f}")
    print(f"  - LoRA 梯度範數: {sum(lora_grad_norms)/len(lora_grad_norms):.6f}")

    if codebook_diff < 1e-6 and sum(lora_grad_norms) > 0:
        print("\n✅ 方案 v3 可以正常訓練！")
        return True
    else:
        print("\n❌ 方案有問題")
        return False


if __name__ == "__main__":
    success = quick_train_test()
    sys.exit(0 if success else 1)
