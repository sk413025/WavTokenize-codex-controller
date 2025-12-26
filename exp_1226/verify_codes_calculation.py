"""
驗證 Token Accuracy 計算的兩種方式是否一致
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 載入模型
    checkpoint_path = Path("exp_1217/runs/exp48_best_config/best_model.pt")

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=128,  # Exp48 uses rank=128
        lora_alpha=256,
        lora_dropout=0.2,
        lora_layers='all_18',
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 直接從 cache 載入預處理好的音頻 tensor
    cache = torch.load(TRAIN_CACHE, weights_only=False)

    print("=" * 70)
    print("比較 model.forward() 和 feature_extractor() 的 codes")
    print("=" * 70)

    # 取前幾個樣本
    samples = cache['samples'][:5]

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")

        # 檢查 cache 內容結構
        if 'noisy_audio' in sample:
            noisy = sample['noisy_audio'].unsqueeze(0).to(device)
        elif 'noisy_waveform' in sample:
            noisy = sample['noisy_waveform'].unsqueeze(0).to(device)
        else:
            print(f"  Available keys: {sample.keys()}")
            continue

        if 'clean_audio' in sample:
            clean = sample['clean_audio'].unsqueeze(0).to(device)
        elif 'clean_waveform' in sample:
            clean = sample['clean_waveform'].unsqueeze(0).to(device)
        else:
            continue

        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)

        with torch.no_grad():
            # 方法 1: model.forward()
            output = model(noisy.squeeze(1), clean.squeeze(1))
            forward_student_codes = output['student_codes']
            forward_teacher_codes = output['teacher_codes']

            # 方法 2: feature_extractor()
            # Student
            student_feat, student_codes_fe, _ = model.student.feature_extractor(noisy, bandwidth_id=0)
            # Teacher
            teacher_feat, teacher_codes_fe, _ = model.teacher.feature_extractor(clean, bandwidth_id=0)

        # 比較
        print("\n=== Student Codes 比較 ===")
        print(f"forward() shape: {forward_student_codes.shape}")
        print(f"feature_extractor() shape: {student_codes_fe.shape}")

        # 調整維度比較
        fc = forward_student_codes[0] if forward_student_codes.dim() == 3 else forward_student_codes
        fec = student_codes_fe[0] if student_codes_fe.dim() == 3 else student_codes_fe

        # 取最小長度比較
        min_len = min(fc.shape[-1], fec.shape[-1])
        fc_trim = fc[..., :min_len]
        fec_trim = fec[..., :min_len]

        print(f"\nforward() first 20 codes: {fc_trim[0, :20].tolist()}")
        print(f"feature_extractor() first 20 codes: {fec_trim[0, :20].tolist()}")

        match_rate = (fc_trim == fec_trim).float().mean().item()
        print(f"\n兩種方法的 codes 匹配率: {match_rate*100:.2f}%")

        print("\n=== Teacher Codes 比較 ===")
        tfc = forward_teacher_codes[0] if forward_teacher_codes.dim() == 3 else forward_teacher_codes
        tfec = teacher_codes_fe[0] if teacher_codes_fe.dim() == 3 else teacher_codes_fe

        min_len_t = min(tfc.shape[-1], tfec.shape[-1])
        tfc_trim = tfc[..., :min_len_t]
        tfec_trim = tfec[..., :min_len_t]

        print(f"forward() first 20 codes: {tfc_trim[0, :20].tolist()}")
        print(f"feature_extractor() first 20 codes: {tfec_trim[0, :20].tolist()}")

        match_rate_t = (tfc_trim == tfec_trim).float().mean().item()
        print(f"\n兩種方法的 codes 匹配率: {match_rate_t*100:.2f}%")

        print("\n=== Token Accuracy 比較 ===")
        # forward() 的 accuracy
        forward_acc = (fc_trim == tfc_trim[..., :min_len]).float().mean().item()
        # feature_extractor() 的 accuracy
        fe_acc = (fec_trim == tfec_trim[..., :min(min_len, min_len_t)]).float().mean().item()

        print(f"forward() Token Accuracy: {forward_acc*100:.2f}%")
        print(f"feature_extractor() Token Accuracy: {fe_acc*100:.2f}%")

        print()


if __name__ == '__main__':
    main()
