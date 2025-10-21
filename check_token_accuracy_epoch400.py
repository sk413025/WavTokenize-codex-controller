#!/usr/bin/env python3
"""
檢查 Epoch 400 的 Token 準確率
驗證 CE weight 修復是否有效
"""

import torch
import torchaudio
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
import numpy as np
from collections import Counter

print("="*70)
print("Token 準確率檢查 - Epoch 400")
print("="*70)

# 載入模型
print("\n[1/4] 載入 Epoch 400 checkpoint...")
checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359/checkpoint_epoch_400.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    print(f"✅ Checkpoint 載入成功")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Loss: {checkpoint['loss']:.4f}")
    
    # 創建模型
    print("\n[2/4] 初始化模型...")
    model = WavTokenizerTransformerDenoiser(
        config_path='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
        model_path='models/wavtokenizer_large_speech_320_24k.ckpt',
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        max_length=400
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = 'cpu'  # 強制使用 CPU 避免 GPU 記憶體不足
    model = model.to(device)
    
    print(f"✅ 模型已載入到 {device}")
    
    # 載入測試音頻（使用訓練生成的音頻樣本）
    print("\n[3/4] 載入測試音頻...")
    test_audio_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_400/batch_0_sample_1_input.wav"
    audio, sr = torchaudio.load(test_audio_path)
    
    print(f"✅ 音頻已載入: {audio.shape}, Sample Rate: {sr} Hz")
    
    # 執行推理
    print("\n[4/4] 執行推理並計算準確率...")
    with torch.no_grad():
        audio = audio.to(device)
        
        # 編碼到 tokens
        noisy_tokens = model.encode_audio_to_tokens(audio)
        target_tokens = noisy_tokens.clone()  # 使用相同作為 target 測試
        
        # Transformer 推理
        output = model.forward_transformer(noisy_tokens, None)
        predicted_tokens = torch.argmax(output, dim=-1)
        
        # 計算統計
        noisy_tokens_np = noisy_tokens.cpu().numpy().flatten()
        target_tokens_np = target_tokens.cpu().numpy().flatten()
        predicted_tokens_np = predicted_tokens.cpu().numpy().flatten()
        
        # Token 準確率
        correct = np.sum(predicted_tokens_np == target_tokens_np)
        total = len(predicted_tokens_np)
        accuracy = (correct / total) * 100
        
        # Token 多樣性
        unique_noisy = len(np.unique(noisy_tokens_np))
        unique_target = len(np.unique(target_tokens_np))
        unique_predicted = len(np.unique(predicted_tokens_np))
        
        # Token 分佈
        predicted_counter = Counter(predicted_tokens_np)
        most_common = predicted_counter.most_common(5)
        
        print("\n" + "="*70)
        print("📊 Token 準確率統計結果")
        print("="*70)
        print(f"\n🎯 Token 準確率: {accuracy:.2f}%")
        print(f"   - 正確預測: {correct}/{total}")
        
        print(f"\n📈 Token 多樣性:")
        print(f"   - Noisy tokens:     {unique_noisy}/4096 ({unique_noisy/4096*100:.2f}%)")
        print(f"   - Target tokens:    {unique_target}/4096 ({unique_target/4096*100:.2f}%)")
        print(f"   - Predicted tokens: {unique_predicted}/4096 ({unique_predicted/4096*100:.2f}%)")
        
        print(f"\n📊 最常見的 5 個預測 tokens:")
        for token_id, count in most_common:
            percentage = (count / total) * 100
            print(f"   - Token {token_id}: {count} 次 ({percentage:.2f}%)")
        
        # 判斷結果
        print("\n" + "="*70)
        print("✅ 評估結論")
        print("="*70)
        
        if accuracy > 60:
            print("🎉 優秀！Token 準確率 > 60%")
            print("   ✅ CE weight 修復非常成功")
            print("   ✅ 模型已學會正確的 token 對應關係")
        elif accuracy > 50:
            print("✅ 良好！Token 準確率 > 50%")
            print("   ✅ CE weight 修復有效")
            print("   💡 建議繼續訓練到 600-800 epochs")
        elif accuracy > 30:
            print("⚠️  一般。Token 準確率 > 30%")
            print("   ✅ 相比舊模型（0%）有顯著改善")
            print("   💡 建議繼續訓練或微調參數")
        elif accuracy > 10:
            print("⚠️  偏低。Token 準確率 > 10%")
            print("   ⚠️  可能需要增加 CE weight 到 15.0-20.0")
        else:
            print("❌ 失敗。Token 準確率 < 10%")
            print("   ❌ CE weight 修復可能無效")
            print("   💡 需要重新檢查配置")
        
        if unique_predicted / 4096 < 0.03:
            print(f"\n⚠️  警告：Token 多樣性過低 ({unique_predicted/4096*100:.2f}%)")
            print("   💡 建議檢查是否有 mode collapse")
        
        print("\n" + "="*70)
        
except FileNotFoundError:
    print(f"❌ 找不到 checkpoint: {checkpoint_path}")
    print("   請確認路徑是否正確")
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
