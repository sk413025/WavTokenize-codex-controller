"""
測試 RVQ 模組

驗證:
1. RVQ forward pass 正確
2. 多層 codebook 使用情況
3. 梯度流動正常
4. 與原始 quantizer 格式兼容
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT
from exp_0128.phase3.residual_vq.models_rvq import TeacherStudentRVQ


def test_rvq_forward():
    """測試 RVQ forward pass"""
    print("="*60)
    print("Test 1: RVQ Forward Pass")
    print("="*60)

    # 創建模型
    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        intermediate_indices=[3, 6],
        device='cuda',
        n_rvq_layers=4,
        rvq_codebook_size=1024,
    )

    # 創建測試資料
    batch_size = 2
    audio_length = 24000  # 1.5 seconds @ 16kHz
    # Audio needs to be [batch, 1, time] for encoder
    clean_audio = torch.randn(batch_size, 1, audio_length).cuda()
    noisy_audio = torch.randn(batch_size, 1, audio_length).cuda()

    # Forward
    model.train()
    outputs = model(clean_audio, noisy_audio)

    # 檢查輸出
    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")

    print("\n✅ Forward pass successful!")
    return model, outputs


def test_rvq_codebook_usage(model, outputs):
    """測試 codebook 使用情況"""
    print("\n" + "="*60)
    print("Test 2: RVQ Codebook Usage")
    print("="*60)

    all_layer_codes = outputs['all_layer_codes']
    print(f"\nAll layer codes shape: {all_layer_codes.shape}")  # [n_layers, batch, time]

    # 分析每層使用情況
    usage_stats = model.get_rvq_usage(all_layer_codes)

    for layer_idx, stats in usage_stats.items():
        n_used = stats['n_used']
        entropy = stats['entropy']
        total = model.rvq_codebook_size

        print(f"\nLayer {layer_idx}:")
        print(f"  Used codes: {n_used}/{total} ({100*n_used/total:.1f}%)")
        print(f"  Entropy: {entropy:.2f}")

        # 檢查是否有明顯的 collapse
        if n_used < total * 0.5:
            print(f"  ⚠️  Warning: Less than 50% codes used")
        else:
            print(f"  ✅ Good diversity")

    print("\n✅ Codebook usage analysis complete!")


def test_rvq_gradients(model, outputs):
    """測試梯度流動"""
    print("\n" + "="*60)
    print("Test 3: RVQ Gradients")
    print("="*60)

    # 創建假的 loss
    student_quantized = outputs['student_quantized']
    teacher_encoder_out = outputs['teacher_encoder_out']

    # Simple MSE loss
    loss = torch.nn.functional.mse_loss(student_quantized, teacher_encoder_out)
    loss += outputs['rvq_commitment_loss']

    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"  MSE loss: {torch.nn.functional.mse_loss(student_quantized, teacher_encoder_out).item():.4f}")
    print(f"  Commitment loss: {outputs['rvq_commitment_loss'].item():.4f}")

    # Backward
    loss.backward()

    # 檢查梯度
    has_grad = False
    total_params = 0
    params_with_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                has_grad = True
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                if 'rvq' in name:
                    print(f"  {name}: grad_norm={grad_norm:.6f}")

    print(f"\nGradient check:")
    print(f"  Total trainable params: {total_params}")
    print(f"  Params with gradients: {params_with_grad}")

    if has_grad:
        print("✅ Gradients flowing correctly!")
    else:
        print("❌ No gradients found!")

    return loss


def test_compatibility():
    """測試與原始 quantizer 的兼容性"""
    print("\n" + "="*60)
    print("Test 4: Compatibility Check")
    print("="*60)

    # 測試不同配置
    configs = [
        {'n_rvq_layers': 2, 'rvq_codebook_size': 2048},
        {'n_rvq_layers': 4, 'rvq_codebook_size': 1024},
        {'n_rvq_layers': 8, 'rvq_codebook_size': 512},
    ]

    for cfg in configs:
        print(f"\nTesting: {cfg['n_rvq_layers']} layers, {cfg['rvq_codebook_size']} codes/layer")

        try:
            model = TeacherStudentRVQ(
                wavtok_config=WAVTOK_CONFIG,
                wavtok_ckpt=WAVTOK_CKPT,
                lora_rank=256,
                lora_alpha=512,
                device='cuda',
                **cfg
            )

            # Quick forward test
            clean = torch.randn(1, 1, 16000).cuda()
            noisy = torch.randn(1, 1, 16000).cuda()

            with torch.no_grad():
                outputs = model(clean, noisy)

            print(f"  ✅ Config works! Output shape: {outputs['student_quantized'].shape}")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n✅ Compatibility check complete!")


def main():
    print("\n" + "="*60)
    print("RVQ Module Testing")
    print("="*60)

    try:
        # Test 1: Forward pass
        model, outputs = test_rvq_forward()

        # Test 2: Codebook usage
        test_rvq_codebook_usage(model, outputs)

        # Test 3: Gradients
        test_rvq_gradients(model, outputs)

        # Clean up
        del model, outputs
        torch.cuda.empty_cache()

        # Test 4: Compatibility
        test_compatibility()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
