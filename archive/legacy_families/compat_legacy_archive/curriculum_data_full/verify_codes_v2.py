"""
驗證 model.forward() 中的 VQ 計算和 feature_extractor 是否一致
"""
import torch
import sys
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT
from exp_1217.models import TeacherStudentConfigurableLoRA


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 載入模型
    checkpoint_path = Path("exp_1217/runs/exp48_best_config/best_model.pt")

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=128,
        lora_alpha=256,
        lora_dropout=0.2,
        lora_layers='all_18',
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 載入測試音頻
    audio_dir = Path("families/compat_legacy/curriculum_data/debug_audio")
    noisy_path = audio_dir / "1_noisy_input.wav"
    clean_path = audio_dir / "2_clean_target.wav"

    noisy_np, sr = sf.read(noisy_path)
    clean_np, sr = sf.read(clean_path)
    noisy = torch.from_numpy(noisy_np).float()
    clean = torch.from_numpy(clean_np).float()

    # 確保維度正確 (B, C, T)
    # soundfile 讀取為 (T,) 或 (T, C)
    if noisy.dim() == 1:
        noisy = noisy.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    elif noisy.dim() == 2:
        noisy = noisy.T.unsqueeze(0)  # (T, C) -> (1, C, T)

    if clean.dim() == 1:
        clean = clean.unsqueeze(0).unsqueeze(0)
    elif clean.dim() == 2:
        clean = clean.T.unsqueeze(0)

    noisy = noisy.to(device)
    clean = clean.to(device)

    print(f"noisy shape: {noisy.shape}")
    print(f"clean shape: {clean.shape}")

    print("=" * 70)
    print("比較 model.forward() 和 feature_extractor() 的 codes")
    print("=" * 70)

    with torch.no_grad():
        # === 方法 1: model.forward() 的計算方式 ===
        # 輸入是 (B, C, T)
        # Student
        student_encoder_out = model.student.feature_extractor.encodec.encoder(noisy)
        student_vq = model.student.feature_extractor.encodec.quantizer(
            student_encoder_out, frame_rate=75, bandwidth=0.075
        )
        forward_student_codes = student_vq.codes

        # Teacher
        teacher_encoder_out = model.teacher.feature_extractor.encodec.encoder(clean)
        teacher_vq = model.teacher.feature_extractor.encodec.quantizer(
            teacher_encoder_out, frame_rate=75, bandwidth=0.075
        )
        forward_teacher_codes = teacher_vq.codes

        # === 方法 2: feature_extractor() ===
        # feature_extractor 期望 (B, T)，內部會 unsqueeze(1)
        noisy_for_fe = noisy.squeeze(1)  # (B, C, T) -> (B, T)
        clean_for_fe = clean.squeeze(1)
        student_feat, fe_student_codes, _ = model.student.feature_extractor(noisy_for_fe, bandwidth_id=0)
        teacher_feat, fe_teacher_codes, _ = model.teacher.feature_extractor(clean_for_fe, bandwidth_id=0)

    # 比較
    print("\n=== Student Codes 比較 ===")
    print(f"forward() codes shape: {forward_student_codes.shape}")
    print(f"feature_extractor() codes shape: {fe_student_codes.shape}")

    # 調整維度
    fsc = forward_student_codes[0] if forward_student_codes.dim() == 3 else forward_student_codes
    fesc = fe_student_codes[0] if fe_student_codes.dim() == 3 else fe_student_codes

    min_len = min(fsc.shape[-1], fesc.shape[-1])
    print(f"\nforward() first 30 codes:          {fsc[0, :30].tolist()}")
    print(f"feature_extractor() first 30 codes: {fesc[0, :30].tolist()}")

    match_rate = (fsc[..., :min_len] == fesc[..., :min_len]).float().mean().item()
    print(f"\n[Student] 兩種方法的 codes 匹配率: {match_rate*100:.2f}%")

    print("\n=== Teacher Codes 比較 ===")
    ftc = forward_teacher_codes[0] if forward_teacher_codes.dim() == 3 else forward_teacher_codes
    fetc = fe_teacher_codes[0] if fe_teacher_codes.dim() == 3 else fe_teacher_codes

    min_len_t = min(ftc.shape[-1], fetc.shape[-1])
    print(f"forward() first 30 codes:          {ftc[0, :30].tolist()}")
    print(f"feature_extractor() first 30 codes: {fetc[0, :30].tolist()}")

    match_rate_t = (ftc[..., :min_len_t] == fetc[..., :min_len_t]).float().mean().item()
    print(f"\n[Teacher] 兩種方法的 codes 匹配率: {match_rate_t*100:.2f}%")

    print("\n=== Token Accuracy ===")
    # forward() 方式
    common_len = min(fsc.shape[-1], ftc.shape[-1])
    forward_acc = (fsc[..., :common_len] == ftc[..., :common_len]).float().mean().item()

    # feature_extractor() 方式
    common_len_fe = min(fesc.shape[-1], fetc.shape[-1])
    fe_acc = (fesc[..., :common_len_fe] == fetc[..., :common_len_fe]).float().mean().item()

    print(f"forward() 方式計算的 Token Accuracy:          {forward_acc*100:.2f}%")
    print(f"feature_extractor() 方式計算的 Token Accuracy: {fe_acc*100:.2f}%")

    print("\n=== 分析 ===")
    if match_rate < 0.99 or match_rate_t < 0.99:
        print("⚠️  forward() 和 feature_extractor() 產生不同的 codes！")
        print("   這解釋了為什麼訓練報告的 Token Accuracy 和音頻品質不一致。")
    else:
        print("✓ 兩種方法產生相同的 codes")

    # === Cosine Similarity 比較 ===
    print("\n=== Cosine Similarity 比較 ===")
    with torch.no_grad():
        # Pre-VQ: encoder output
        s_enc = student_encoder_out.flatten(1)  # (B, C*T)
        t_enc = teacher_encoder_out.flatten(1)
        cos_pre_vq = torch.nn.functional.cosine_similarity(s_enc, t_enc).mean().item()

        # Post-VQ: quantized features from feature_extractor
        s_quant = student_feat.flatten(1)
        t_quant = teacher_feat.flatten(1)
        cos_post_vq = torch.nn.functional.cosine_similarity(s_quant, t_quant).mean().item()

    print(f"Pre-VQ (encoder output) Cosine Similarity:  {cos_pre_vq:.4f}")
    print(f"Post-VQ (quantized) Cosine Similarity:      {cos_post_vq:.4f}")

    print("\n=== 結論 ===")
    print(f"1. Token Accuracy = {forward_acc*100:.2f}% 是正確的")
    print(f"2. Pre-VQ Cosine Sim = {cos_pre_vq:.4f} (這是訓練時優化的)")
    print(f"3. Post-VQ Cosine Sim = {cos_post_vq:.4f} (這是解碼後的)")
    print(f"4. 即使 VQ 相同，Student 和 Teacher codes 不同 → Token Acc 低")


if __name__ == '__main__':
    main()
