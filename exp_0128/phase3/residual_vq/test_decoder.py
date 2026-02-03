"""
測試 RVQ + Teacher Decoder 的音頻重建

驗證：
1. Teacher decoder 是凍結的
2. Teacher decoder 可以處理 RVQ quantized vectors
3. 音頻重建正常工作
"""

import torch
import torchaudio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT
from exp_0128.phase3.residual_vq.models_rvq import TeacherStudentRVQ


def test_decoder_frozen():
    """測試 teacher decoder 是否凍結"""
    print("="*60)
    print("Test 1: Verify Teacher Decoder is Frozen")
    print("="*60)

    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        device='cuda',
        n_rvq_layers=4,
        rvq_codebook_size=1024,
    )

    # 檢查 teacher decoder 參數
    decoder_params = list(model.teacher.feature_extractor.encodec.decoder.parameters())

    frozen_count = 0
    trainable_count = 0

    for param in decoder_params:
        if param.requires_grad:
            trainable_count += 1
        else:
            frozen_count += 1

    print(f"\nTeacher Decoder Parameters:")
    print(f"  Total: {len(decoder_params)}")
    print(f"  Frozen (requires_grad=False): {frozen_count}")
    print(f"  Trainable (requires_grad=True): {trainable_count}")

    if trainable_count == 0:
        print("\n✅ Teacher decoder is completely frozen!")
    else:
        print(f"\n❌ Warning: {trainable_count} parameters are trainable!")

    return model


def test_decoder_with_rvq():
    """測試 decoder 能否處理 RVQ quantized vectors"""
    print("\n" + "="*60)
    print("Test 2: Decoder with RVQ Quantized Vectors")
    print("="*60)

    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        device='cuda',
        n_rvq_layers=4,
        rvq_codebook_size=1024,
    )

    # 創建測試音頻
    batch_size = 2
    audio_length = 24000  # 1.5 seconds @ 16kHz
    clean_audio = torch.randn(batch_size, 1, audio_length).cuda()
    noisy_audio = torch.randn(batch_size, 1, audio_length).cuda()

    # Forward
    with torch.no_grad():
        outputs = model(clean_audio, noisy_audio)

        # Get RVQ quantized vectors
        student_quantized = outputs['student_quantized']
        print(f"\nRVQ quantized shape: {student_quantized.shape}")

        # Try to decode
        try:
            reconstructed = model.decode(student_quantized)
            print(f"Reconstructed audio shape: {reconstructed.shape}")
            print(f"Input audio shape: {clean_audio.shape}")

            if reconstructed.shape == clean_audio.shape:
                print("\n✅ Decoder works with RVQ quantized vectors!")
                print("   Output shape matches input shape")
            else:
                print(f"\n⚠️  Shape mismatch:")
                print(f"   Expected: {clean_audio.shape}")
                print(f"   Got: {reconstructed.shape}")

            return True, reconstructed

        except Exception as e:
            print(f"\n❌ Decoder failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None


def test_audio_quality():
    """測試音頻質量（簡單檢查）"""
    print("\n" + "="*60)
    print("Test 3: Audio Quality Check")
    print("="*60)

    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        device='cuda',
        n_rvq_layers=4,
        rvq_codebook_size=1024,
    )

    # 創建測試音頻
    clean_audio = torch.randn(1, 1, 24000).cuda()
    noisy_audio = torch.randn(1, 1, 24000).cuda()

    with torch.no_grad():
        outputs = model(clean_audio, noisy_audio)
        student_quantized = outputs['student_quantized']
        reconstructed = model.decode(student_quantized)

        # 簡單檢查
        print(f"\nAudio Statistics:")
        print(f"  Clean audio:")
        print(f"    Mean: {clean_audio.mean().item():.4f}")
        print(f"    Std:  {clean_audio.std().item():.4f}")
        print(f"    Min:  {clean_audio.min().item():.4f}")
        print(f"    Max:  {clean_audio.max().item():.4f}")

        print(f"\n  Reconstructed audio:")
        print(f"    Mean: {reconstructed.mean().item():.4f}")
        print(f"    Std:  {reconstructed.std().item():.4f}")
        print(f"    Min:  {reconstructed.min().item():.4f}")
        print(f"    Max:  {reconstructed.max().item():.4f}")

        # 檢查是否有 NaN 或 Inf
        has_nan = torch.isnan(reconstructed).any().item()
        has_inf = torch.isinf(reconstructed).any().item()

        if has_nan:
            print("\n❌ Reconstructed audio contains NaN!")
        elif has_inf:
            print("\n❌ Reconstructed audio contains Inf!")
        else:
            print("\n✅ Reconstructed audio is valid (no NaN/Inf)")

        # 簡單的 MSE
        mse = torch.nn.functional.mse_loss(reconstructed, clean_audio)
        print(f"\nMSE (reconstructed vs clean): {mse.item():.6f}")
        print("  (注意：這是隨機初始化的 RVQ，MSE 會很大)")


def test_save_audio():
    """測試保存音頻檔案"""
    print("\n" + "="*60)
    print("Test 4: Save Audio Files")
    print("="*60)

    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        device='cuda',
        n_rvq_layers=4,
        rvq_codebook_size=1024,
    )

    # 創建測試音頻
    clean_audio = torch.randn(1, 1, 16000).cuda()  # 1 second
    noisy_audio = torch.randn(1, 1, 16000).cuda()

    with torch.no_grad():
        outputs = model(clean_audio, noisy_audio)
        student_quantized = outputs['student_quantized']
        reconstructed = model.decode(student_quantized)

        # 保存音頻
        output_dir = Path('exp_0128/phase3/residual_vq/test_audio')
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            torchaudio.save(
                str(output_dir / 'test_clean.wav'),
                clean_audio[0].cpu(),
                16000
            )
            torchaudio.save(
                str(output_dir / 'test_noisy.wav'),
                noisy_audio[0].cpu(),
                16000
            )
            torchaudio.save(
                str(output_dir / 'test_reconstructed.wav'),
                reconstructed[0].cpu(),
                16000
            )

            print(f"\n✅ Audio files saved to {output_dir}/")
            print("  - test_clean.wav")
            print("  - test_noisy.wav")
            print("  - test_reconstructed.wav")
            print("\n你可以聽聽看這些檔案（雖然是隨機音頻）")

        except Exception as e:
            print(f"\n❌ Failed to save audio: {e}")


def main():
    print("\n" + "="*60)
    print("RVQ + Teacher Decoder Testing")
    print("="*60)

    try:
        # Test 1: Verify decoder is frozen
        model = test_decoder_frozen()

        # Test 2: Decoder with RVQ
        success, reconstructed = test_decoder_with_rvq()

        if success:
            # Test 3: Audio quality
            test_audio_quality()

            # Test 4: Save audio
            test_save_audio()

        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
