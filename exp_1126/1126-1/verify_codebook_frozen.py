#!/usr/bin/env python3
"""
驗證 Codebook 是否確實被凍結
比較原始模型和訓練後 checkpoint 的 codebook
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(1, '/home/sbplab/ruizi/WavTokenizer-main')

import torch

def verify_codebook():
    print("=" * 70)
    print("Codebook 凍結驗證")
    print("=" * 70)

    # 1. 載入原始 codebook
    print("\n[1] 載入原始 WavTokenizer codebook...")
    from decoder.pretrained import WavTokenizer

    config_path = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    ckpt_path = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

    original_model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    original_model.eval()

    original_cb = original_model.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.clone()
    original_cluster_size = original_model.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.cluster_size.clone()
    original_embed_avg = original_model.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed_avg.clone()

    print(f"   Original codebook shape: {original_cb.shape}")

    # 2. 載入訓練後的 checkpoint
    print("\n[2] 載入訓練後的 checkpoint...")
    latest_ckpt = "experiments/lora_encoder_frozen_vq_v3/checkpoints/latest.pt"

    if not os.path.exists(latest_ckpt):
        print(f"   ❌ 找不到 checkpoint: {latest_ckpt}")
        return False

    ckpt = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    print(f"   Checkpoint loaded, epoch: {ckpt.get('epoch', 'unknown')}")

    # 3. 找出 Student codebook
    print("\n[3] 尋找 Student codebook 在 state_dict 中的 key...")

    # 搜尋可能的 key patterns
    student_cb_key = None
    student_cluster_key = None
    student_embed_avg_key = None

    for key in state_dict.keys():
        if 'student' in key:
            if '_codebook.embed' in key and 'embed_avg' not in key:
                student_cb_key = key
            elif '_codebook.cluster_size' in key:
                student_cluster_key = key
            elif '_codebook.embed_avg' in key:
                student_embed_avg_key = key

    if student_cb_key:
        print(f"   ✓ Found codebook key: {student_cb_key}")
        trained_cb = state_dict[student_cb_key]
    else:
        # 嘗試其他方式
        print("   尋找替代 key...")
        for key in state_dict.keys():
            if 'quantizer.vq.layers.0._codebook.embed' in key and 'embed_avg' not in key:
                student_cb_key = key
                trained_cb = state_dict[key]
                print(f"   ✓ Found alternative key: {key}")
                break

    if student_cb_key is None:
        print("   ❌ 找不到 codebook key!")
        print("\n   可用的 keys (包含 'embed' 或 'codebook'):")
        for key in state_dict.keys():
            if 'embed' in key.lower() or 'codebook' in key.lower():
                print(f"      {key}")
        return False

    # 4. 比較 codebook
    print("\n[4] 比較 Original vs Trained Codebook...")

    diff = (original_cb - trained_cb).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    # 統計有多少 entries 變化超過閾值
    thresh_1e6 = (diff > 1e-6).any(dim=1).sum().item()
    thresh_1e4 = (diff > 1e-4).any(dim=1).sum().item()
    thresh_1e2 = (diff > 1e-2).any(dim=1).sum().item()

    print(f"\n   {'='*50}")
    print(f"   Codebook 差異統計")
    print(f"   {'='*50}")
    print(f"   最大差異:      {max_diff:.10f}")
    print(f"   平均差異:      {mean_diff:.10f}")
    print(f"   標準差:        {std_diff:.10f}")
    print(f"   ")
    print(f"   變化 > 1e-6 的條目: {thresh_1e6:4d} / {original_cb.shape[0]}")
    print(f"   變化 > 1e-4 的條目: {thresh_1e4:4d} / {original_cb.shape[0]}")
    print(f"   變化 > 1e-2 的條目: {thresh_1e2:4d} / {original_cb.shape[0]}")

    # 5. 結論
    print(f"\n   {'='*50}")
    print(f"   結論")
    print(f"   {'='*50}")

    if max_diff < 1e-6:
        print("   ✅ Codebook 完全相同！凍結成功！")
        success = True
    elif max_diff < 1e-4:
        print("   ⚠️ Codebook 有微小差異（可能是數值精度）")
        success = True
    else:
        print(f"   ❌ Codebook 有顯著差異！")
        print(f"      max_diff = {max_diff:.6f}")
        success = False

    # 6. 額外檢查：cluster_size 和 embed_avg
    if student_cluster_key and student_embed_avg_key:
        print("\n[5] 檢查 EMA 相關 buffers...")
        trained_cluster = state_dict[student_cluster_key]
        trained_embed_avg = state_dict[student_embed_avg_key]

        cluster_diff = (original_cluster_size - trained_cluster).abs().max().item()
        embed_avg_diff = (original_embed_avg - trained_embed_avg).abs().max().item()

        print(f"   cluster_size 差異: {cluster_diff:.10f}")
        print(f"   embed_avg 差異:    {embed_avg_diff:.10f}")

    return success


if __name__ == "__main__":
    success = verify_codebook()
    print("\n" + "=" * 70)
    sys.exit(0 if success else 1)
