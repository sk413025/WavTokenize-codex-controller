#!/usr/bin/env python3
"""
即時檢查正在運行的實驗的 Codebook 狀態
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(1, '/home/sbplab/ruizi/WavTokenizer-main')

import torch
import glob

def check_codebook_from_checkpoint():
    """從最新的 checkpoint 檢查 Codebook"""

    # 找到最新的 checkpoint
    exp_dir = "experiments/lora_encoder_frozen_vq_v3"
    ckpt_pattern = os.path.join(exp_dir, "*.pt")
    ckpts = glob.glob(ckpt_pattern)

    if not ckpts:
        print(f"❌ 沒有找到 checkpoint: {ckpt_pattern}")
        return False

    # 按修改時間排序
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"找到最新 checkpoint: {latest_ckpt}")

    # 載入 checkpoint
    ckpt = torch.load(latest_ckpt, map_location='cpu')

    # 檢查 state_dict 中的 codebook
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        print("Warning: checkpoint 沒有 model_state_dict")
        return False

    # 找出 codebook 相關的 keys
    codebook_keys = [k for k in state_dict.keys() if 'embed' in k.lower() or 'codebook' in k.lower()]
    print(f"\n找到 {len(codebook_keys)} 個 codebook 相關的 keys")

    return True


def check_codebook_from_reference():
    """比較原始模型和正在訓練模型的 Codebook"""

    print("=" * 60)
    print("Codebook 凍結驗證")
    print("=" * 60)

    # 載入原始 distance matrix (包含原始 codebook 資訊)
    dist_mat_path = "wavtok_distance_mat_corrected.pt"
    if os.path.exists(dist_mat_path):
        print(f"\n載入 distance matrix: {dist_mat_path}")
        dist_mat = torch.load(dist_mat_path, map_location='cpu')
        print(f"Distance matrix shape: {dist_mat.shape}")

    # 從原始模型載入 codebook
    print("\n載入原始 WavTokenizer 來獲取參考 codebook...")

    from decoder.pretrained import WavTokenizer

    config_path = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    ckpt_path = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

    model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    model.eval()

    original_codebook = model.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.clone()
    print(f"原始 Codebook shape: {original_codebook.shape}")

    # 檢查是否有最新的 checkpoint
    exp_dir = "experiments/lora_encoder_frozen_vq_v3"
    best_ckpt = os.path.join(exp_dir, "best_model.pt")
    latest_ckpt = os.path.join(exp_dir, "latest_model.pt")

    ckpt_to_check = None
    if os.path.exists(latest_ckpt):
        ckpt_to_check = latest_ckpt
    elif os.path.exists(best_ckpt):
        ckpt_to_check = best_ckpt

    if ckpt_to_check:
        print(f"\n載入訓練中的 checkpoint: {ckpt_to_check}")
        ckpt = torch.load(ckpt_to_check, map_location='cpu')

        # 找出 student codebook
        state_dict = ckpt.get('model_state_dict', ckpt)

        # PEFT wrapped model 的 codebook key
        student_cb_key = None
        for key in state_dict.keys():
            if 'student' in key and '_codebook.embed' in key:
                student_cb_key = key
                break

        if student_cb_key:
            trained_codebook = state_dict[student_cb_key]
            print(f"找到 Student Codebook: {student_cb_key}")
            print(f"Trained Codebook shape: {trained_codebook.shape}")

            # 比較
            diff = (original_codebook - trained_codebook).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            entries_changed = (diff > 1e-6).any(dim=1).sum().item()

            print("\n" + "=" * 60)
            print("Codebook 比較結果")
            print("=" * 60)
            print(f"  最大差異: {max_diff:.8f}")
            print(f"  平均差異: {mean_diff:.8f}")
            print(f"  改變的條目數: {entries_changed} / {original_codebook.shape[0]}")

            if max_diff < 1e-6:
                print("\n✅ Codebook 完全相同！凍結成功！")
            else:
                print(f"\n❌ Codebook 有差異！max_diff = {max_diff}")
        else:
            print("❌ 沒有找到 Student Codebook key")
            print("可用的 keys (前 20 個):")
            for i, key in enumerate(list(state_dict.keys())[:20]):
                print(f"  {key}")
    else:
        print(f"\n⚠️ 沒有找到 checkpoint，實驗可能還沒保存")
        print(f"檢查的路徑: {exp_dir}")

    return True


if __name__ == "__main__":
    check_codebook_from_reference()
