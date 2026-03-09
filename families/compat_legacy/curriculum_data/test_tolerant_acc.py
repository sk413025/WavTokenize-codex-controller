"""
測試 Frame-Tolerant Accuracy 在真實數據上的效果
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
from families.deps.wavtokenizer_core.config import VAL_CACHE, WAVTOK_CONFIG, WAVTOK_CKPT
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1219.losses import compute_masked_accuracy
from families.compat_legacy.curriculum_data.losses_tolerant import FrameTolerantAccuracy
from torch.utils.data import DataLoader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 找 checkpoint
    checkpoint_dirs = [
        Path("exp_1219/runs/exp55_accum2_lr1e4"),
        Path("families/compat_legacy/curriculum_data/runs/exp63_vq_aware"),
    ]

    checkpoint_path = None
    for d in checkpoint_dirs:
        if d.exists():
            best_pt = d / "best_model.pt"
            if best_pt.exists():
                checkpoint_path = best_pt
                break

    if checkpoint_path is None:
        print("No checkpoint found!")
        return

    print(f"Using checkpoint: {checkpoint_path}")

    # 載入模型
    config_path = str(WAVTOK_CONFIG)
    model_path = str(WAVTOK_CKPT)

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=config_path,
        wavtok_ckpt=model_path,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
        lora_layers='all_18',
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 載入資料
    dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=30)
    loader = DataLoader(dataset, batch_size=4, collate_fn=aligned_collate_fn, num_workers=0)

    # 測試
    tolerant_acc_fn = FrameTolerantAccuracy(tolerance=1)

    strict_accs = []
    tolerant_accs = []

    print("\n" + "=" * 60)
    print("Frame-Tolerant vs Strict Accuracy")
    print("=" * 60)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)

            output = model(noisy, clean)

            s_codes = output['student_codes']
            t_codes = output['teacher_codes']

            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            # Strict accuracy
            strict_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, 320)
            strict_accs.append(strict_acc)

            # Tolerant accuracy
            tolerant_acc, info = tolerant_acc_fn(s_codes, t_codes, lengths)
            tolerant_accs.append(tolerant_acc)

            print(f"Batch {batch_idx + 1}: Strict={strict_acc*100:.2f}%, Tolerant={tolerant_acc*100:.2f}%, +{info['improvement']*100:.2f}%")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Average Strict Accuracy:   {sum(strict_accs)/len(strict_accs)*100:.2f}%")
    print(f"Average Tolerant Accuracy: {sum(tolerant_accs)/len(tolerant_accs)*100:.2f}%")
    print(f"Average Improvement:       +{(sum(tolerant_accs)-sum(strict_accs))/len(strict_accs)*100:.2f}%")


if __name__ == '__main__':
    main()
