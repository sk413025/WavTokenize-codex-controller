"""
快速檢查 Token Accuracy 計算是否正確

這個腳本會：
1. 載入最近的 checkpoint
2. 跑幾個樣本
3. 詳細印出 student_codes vs teacher_codes 的比對結果
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1219.losses import compute_masked_accuracy, create_length_mask
from torch.utils.data import DataLoader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 找最近的 checkpoint (優先使用 rank=256 的)
    checkpoint_dirs = [
        Path("exp_1219/runs/exp55_accum2_lr1e4"),  # rank=256
        Path("families/compat_legacy/curriculum_data/runs/exp63_vq_aware"),      # rank=256
    ]

    checkpoint_path = None
    for d in checkpoint_dirs:
        if d.exists():
            best_pt = d / "best_model.pt"
            if best_pt.exists():
                checkpoint_path = best_pt
                break

    if checkpoint_path is None:
        print("No checkpoint found, using fresh model for baseline check")
        use_fresh = True
    else:
        print(f"Using checkpoint: {checkpoint_path}")
        use_fresh = False

    # 載入模型 (使用 exp_1201 的配置)
    config_path = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=config_path,
        wavtok_ckpt=model_path,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
        lora_layers='all_18',
        device=device
    )

    if not use_fresh:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    model.eval()

    # 載入資料
    from families.deps.wavtokenizer_core.config import VAL_CACHE
    dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=10)
    loader = DataLoader(dataset, batch_size=2, collate_fn=aligned_collate_fn, num_workers=0)

    print("\n" + "=" * 60)
    print("Token Accuracy Detailed Check")
    print("=" * 60)

    encoder_stride = 320

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths']

            output = model(noisy, clean)

            s_codes = output['student_codes']
            t_codes = output['teacher_codes']

            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"student_codes shape: {s_codes.shape}")
            print(f"teacher_codes shape: {t_codes.shape}")

            # 處理維度
            if s_codes.dim() == 3:
                s_codes = s_codes[0]  # (B, T)
            if t_codes.dim() == 3:
                t_codes = t_codes[0]  # (B, T)

            B, T = s_codes.shape

            # 計算 masked accuracy
            max_audio_len = T * encoder_stride
            lengths = lengths.to(device)
            mask = create_length_mask(lengths, max_audio_len, encoder_stride, device=device)

            correct = (s_codes == t_codes).float()
            masked_correct = correct * mask

            for b in range(B):
                valid_len = int(mask[b].sum().item())
                s_valid = s_codes[b, :valid_len].cpu().numpy()
                t_valid = t_codes[b, :valid_len].cpu().numpy()

                num_correct = int((s_valid == t_valid).sum())
                acc = num_correct / valid_len if valid_len > 0 else 0

                print(f"\n  Sample {b + 1}:")
                print(f"    Valid tokens: {valid_len}")
                print(f"    Correct: {num_correct}")
                print(f"    Accuracy: {acc * 100:.2f}%")

                # 顯示前 20 個 tokens
                show_len = min(20, valid_len)
                print(f"    First {show_len} tokens:")
                print(f"      Student: {s_valid[:show_len].tolist()}")
                print(f"      Teacher: {t_valid[:show_len].tolist()}")
                print(f"      Match:   {['✓' if s == t else '✗' for s, t in zip(s_valid[:show_len], t_valid[:show_len])]}")

                # 統計 code 分佈
                from collections import Counter
                s_counter = Counter(s_valid.tolist())
                t_counter = Counter(t_valid.tolist())

                print(f"\n    Student top-5 codes: {s_counter.most_common(5)}")
                print(f"    Teacher top-5 codes: {t_counter.most_common(5)}")
                print(f"    Student unique codes: {len(s_counter)}")
                print(f"    Teacher unique codes: {len(t_counter)}")

            # 使用官方函數計算
            acc_official, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
            print(f"\n  Official compute_masked_accuracy: {acc_official * 100:.2f}%")

            if batch_idx >= 2:  # 只看前 3 個 batch
                break

    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print("""
如果 Student 和 Teacher 的 codes 分佈差異很大:
→ Student encoder features 經過 VQ 量化後選到不同的 centroids
→ 這表示 feature space alignment 有問題（方向對但位置不對）

如果 Student codes 集中在少數幾個值:
→ Student encoder 可能發生 mode collapse
→ 所有輸入都映射到相似的 features

如果 accuracy 接近 random guess (1/4096 ≈ 0.024%):
→ Student features 和 Teacher features 幾乎無關
→ LoRA 沒有正確學到映射
""")


if __name__ == '__main__':
    main()
