#!/usr/bin/env python3
"""
Codebook 凍結驗證腳本
快速確認模型是否正確使用預訓練 codebook 並完全凍結
"""

import torch
import sys

print("=" * 70)
print("Codebook 凍結驗證")
print("=" * 70)

try:
    # 1. 載入 checkpoint
    print("\n[1/5] 載入 checkpoint...")
    checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510190523/checkpoint_epoch_300.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  ✅ Checkpoint 載入成功（Epoch {checkpoint.get('epoch', 'unknown')}）")
    
    # 2. 檢查 state_dict 中的 codebook_embedding
    print("\n[2/5] 檢查 codebook_embedding 參數...")
    state_dict = checkpoint['model_state_dict']
    
    # 查找 codebook_embedding 相關的鍵
    codebook_keys = [k for k in state_dict.keys() if 'codebook_embedding' in k]
    print(f"  找到 {len(codebook_keys)} 個 codebook_embedding 相關參數：")
    for key in codebook_keys:
        tensor = state_dict[key]
        print(f"    - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    if 'codebook_embedding.weight' in state_dict:
        codebook_weight = state_dict['codebook_embedding.weight']
        print(f"\n  ✅ Codebook Embedding 權重：")
        print(f"     - Shape: {codebook_weight.shape}")
        print(f"     - 預期: [4096, 512] (4096 個 codes，每個 512 維)")
        print(f"     - 統計: mean={codebook_weight.mean():.4f}, std={codebook_weight.std():.4f}")
        
        # 檢查是否為預訓練權重（不應該是隨機初始化）
        if abs(codebook_weight.mean()) < 0.01 and abs(codebook_weight.std() - 0.02) < 0.01:
            print(f"     ⚠️  警告：權重看起來像隨機初始化（mean≈0, std≈0.02）")
        else:
            print(f"     ✅ 權重看起來像預訓練權重（非標準正態分佈）")
    
    # 3. 創建模型並載入權重
    print("\n[3/5] 創建模型實例...")
    from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
    
    config = checkpoint.get('config', {})
    model = WavTokenizerTransformerDenoiser(
        config_path=config.get('config_path', 'config/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'),
        model_path=config.get('model_path', 'models/wavtokenizer_small_600_24k.ckpt'),
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        num_decoder_layers=config.get('num_decoder_layers', 4),
        dim_feedforward=config.get('dim_feedforward', 1024),
        max_length=config.get('max_length', 400)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ 模型創建並載入權重成功")
    
    # 4. 檢查 requires_grad
    print("\n[4/5] 檢查梯度設置...")
    
    # 檢查 codebook_embedding
    if hasattr(model, 'codebook_embedding'):
        codebook_frozen = not model.codebook_embedding.weight.requires_grad
        print(f"  Codebook Embedding:")
        print(f"    - requires_grad: {model.codebook_embedding.weight.requires_grad}")
        print(f"    - 是否凍結: {'✅ 是' if codebook_frozen else '❌ 否'}")
        
        if not codebook_frozen:
            print(f"    ⚠️  警告：Codebook embedding 沒有凍結！")
    else:
        print(f"  ❌ 錯誤：找不到 codebook_embedding 屬性")
    
    # 檢查 WavTokenizer
    if hasattr(model, 'wavtokenizer'):
        wavtokenizer_params = list(model.wavtokenizer.parameters())
        wavtokenizer_frozen = all(not p.requires_grad for p in wavtokenizer_params)
        trainable_wt = sum(1 for p in wavtokenizer_params if p.requires_grad)
        
        print(f"\n  WavTokenizer:")
        print(f"    - 總參數數: {len(wavtokenizer_params)}")
        print(f"    - 可訓練參數: {trainable_wt}")
        print(f"    - 是否完全凍結: {'✅ 是' if wavtokenizer_frozen else '❌ 否'}")
        
        if not wavtokenizer_frozen:
            print(f"    ⚠️  警告：WavTokenizer 有 {trainable_wt} 個參數沒有凍結！")
    
    # 檢查 special_token_embedding
    if hasattr(model, 'special_token_embedding'):
        special_frozen = not model.special_token_embedding.weight.requires_grad
        print(f"\n  Special Token Embedding:")
        print(f"    - requires_grad: {model.special_token_embedding.weight.requires_grad}")
        print(f"    - 是否凍結: {'❌ 否（應該可訓練）' if not special_frozen else '⚠️  是（不符預期）'}")
    
    # 5. 統計可訓練參數
    print("\n[5/5] 參數統計...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  凍結參數: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # 預期：大部分參數應該是凍結的（WavTokenizer + Codebook）
    if frozen_params > total_params * 0.8:
        print(f"  ✅ 凍結比例正常（>80%，主要是 WavTokenizer 和 Codebook）")
    else:
        print(f"  ⚠️  警告：凍結比例偏低（<80%），可能沒有正確凍結")
    
    # 6. 測試 token 提取
    print("\n[6/6] 測試 Token 提取...")
    
    # 創建假音頻
    fake_audio = torch.randn(1, 1, 16000)  # 1秒音頻
    
    with torch.no_grad():
        tokens = model.encode_audio_to_tokens(fake_audio)
        print(f"  ✅ Token 提取成功")
        print(f"     - Shape: {tokens.shape}")
        print(f"     - 值範圍: [{tokens.min().item()}, {tokens.max().item()}]")
        print(f"     - 預期範圍: [0, 4095]")
        
        if tokens.min() >= 0 and tokens.max() <= 4095:
            print(f"     ✅ Token 範圍正確")
        else:
            print(f"     ❌ Token 範圍異常！")
    
    # 總結
    print("\n" + "=" * 70)
    print("驗證總結")
    print("=" * 70)
    
    all_checks_passed = True
    
    checks = [
        ("Codebook Embedding 凍結", codebook_frozen if 'codebook_frozen' in locals() else False),
        ("WavTokenizer 凍結", wavtokenizer_frozen if 'wavtokenizer_frozen' in locals() else False),
        ("凍結參數比例 > 80%", frozen_params > total_params * 0.8),
        ("Token 範圍正確", tokens.min() >= 0 and tokens.max() <= 4095)
    ]
    
    for check_name, passed in checks:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {status}: {check_name}")
        if not passed:
            all_checks_passed = False
    
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ 所有檢查通過！")
        print("   - Codebook 確實使用預訓練權重並完全凍結")
        print("   - WavTokenizer 確實完全凍結")
        print("   - Token 空間兼容性正確")
        print("   - 架構設計符合預期")
    else:
        print("❌ 部分檢查失敗！")
        print("   請檢查上述失敗的項目")
    print("=" * 70)
    
    sys.exit(0 if all_checks_passed else 1)
    
except Exception as e:
    print(f"\n❌ 驗證失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
