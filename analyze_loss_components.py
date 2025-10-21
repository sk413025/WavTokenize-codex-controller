#!/usr/bin/env python3
"""
分析訓練損失組成，診斷為什麼 Token 準確率為 0%

實驗編號: EXP20251021_01  
生成函式: analyze_loss_components
"""

import torch
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
import torchaudio
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F

EXP_ID = "EXP20251021_01"
DATE = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*80)
print(f"訓練損失組成分析 - {EXP_ID}")
print(f"日期時間: {DATE}")
print("="*80)

# 載入 Epoch 400 checkpoint
checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359/checkpoint_epoch_400.pth"
print(f"\n載入 checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint.get('config', {})

print(f"\n✅ Checkpoint 載入成功")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Total Loss: {checkpoint['loss']:.4f}")

print(f"\n📊 訓練配置:")
print(f"   CE weight:        {config.get('ce_weight', 'N/A')}")
print(f"   L2 weight:        {config.get('l2_embed_weight', 'N/A')}")
print(f"   Coherence weight: {config.get('coherence_weight', 'N/A')}")
print(f"   Manifold weight:  {config.get('manifold_weight', 'N/A')}")

# 載入模型
print(f"\n創建模型...")
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

# 載入測試音頻
audio_dir = Path("results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_400")
input_file = audio_dir / "batch_0_sample_1_input.wav"
target_file = audio_dir / "batch_0_sample_1_target.wav"

input_audio, sr = torchaudio.load(input_file)
target_audio, _ = torchaudio.load(target_file)

print(f"\n✅ 音頻已載入")

# 計算所有損失組件
print(f"\n" + "="*80)
print("計算損失組件")
print("="*80)

with torch.no_grad():
    # 編碼到 tokens
    noisy_tokens = model.encode_audio_to_tokens(input_audio)
    target_tokens = model.encode_audio_to_tokens(target_audio)
    
    print(f"\nToken 序列:")
    print(f"  Noisy:  {noisy_tokens.shape}")
    print(f"  Target: {target_tokens.shape}")
    
    # 切換到訓練模式以獲取 logits
    model.train()
    logits = model.forward_transformer(noisy_tokens, target_tokens)
    predicted_tokens = torch.argmax(logits, dim=-1)
    model.eval()
    
    print(f"  Logits: {logits.shape}")
    print(f"  Predicted: {predicted_tokens.shape}")
    
    # 1. Cross Entropy Loss
    ce_weight = config.get('ce_weight', 10.0)
    
    # 需要裁剪到相同長度
    min_len = min(logits.size(1), target_tokens.size(1))
    logits_trimmed = logits[:, :min_len, :]
    target_trimmed = target_tokens[:, :min_len]
    
    ce_loss = F.cross_entropy(
        logits_trimmed.reshape(-1, logits_trimmed.size(-1)),
        target_trimmed.reshape(-1),
        ignore_index=4095  # 假設 4095 是 padding token
    )
    
    print(f"\n1️⃣  Cross Entropy Loss:")
    print(f"   Raw CE Loss:      {ce_loss.item():.4f}")
    print(f"   CE Weight:        {ce_weight}")
    print(f"   Weighted CE Loss: {ce_loss.item() * ce_weight:.4f}")
    
    # 計算 token 準確率
    correct = (predicted_tokens[:, :min_len] == target_trimmed).sum().item()
    total = min_len
    accuracy = (correct / total) * 100
    print(f"   Token Accuracy:   {accuracy:.2f}% ({correct}/{total})")
    
    # 2. 獲取 embeddings 計算其他損失
    noisy_emb = model.get_token_embeddings(noisy_tokens)
    target_emb = model.get_token_embeddings(target_tokens)
    
    # 使用 transformer encoder 獲取 enhanced embeddings
    src_emb = noisy_emb * (model.d_model ** 0.5)
    src_seq_len = noisy_tokens.shape[1]
    src_emb = src_emb + model.pos_encoding[:, :src_seq_len, :]
    
    encoder_output = model.transformer.encoder(src_emb)
    
    # L2 Embedding Loss
    l2_weight = config.get('l2_embed_weight', 0.5)
    min_emb_len = min(encoder_output.size(1), target_emb.size(1))
    
    l2_loss = F.mse_loss(
        encoder_output[:, :min_emb_len, :],
        target_emb[:, :min_emb_len, :]
    )
    
    print(f"\n2️⃣  L2 Embedding Loss:")
    print(f"   Raw L2 Loss:      {l2_loss.item():.4f}")
    print(f"   L2 Weight:        {l2_weight}")
    print(f"   Weighted L2 Loss: {l2_loss.item() * l2_weight:.4f}")
    
    # 3. Coherence Loss (如果有)
    coherence_weight = config.get('coherence_weight', 0.2)
    
    # 簡化的 coherence 計算
    if encoder_output.size(1) > 1:
        diff = encoder_output[:, 1:, :] - encoder_output[:, :-1, :]
        coherence_loss = torch.mean(torch.norm(diff, dim=-1))
    else:
        coherence_loss = torch.tensor(0.0)
    
    print(f"\n3️⃣  Coherence Loss:")
    print(f"   Raw Coherence:    {coherence_loss.item():.4f}")
    print(f"   Coherence Weight: {coherence_weight}")
    print(f"   Weighted:         {coherence_loss.item() * coherence_weight:.4f}")
    
    # 4. Manifold Loss (如果有)
    manifold_weight = config.get('manifold_weight', 0.1)
    manifold_loss = torch.mean(torch.norm(encoder_output - target_emb[:, :min_emb_len, :], dim=-1))
    
    print(f"\n4️⃣  Manifold Loss:")
    print(f"   Raw Manifold:     {manifold_loss.item():.4f}")
    print(f"   Manifold Weight:  {manifold_weight}")
    print(f"   Weighted:         {manifold_loss.item() * manifold_weight:.4f}")
    
    # Total Loss
    total_loss = (ce_loss * ce_weight + 
                  l2_loss * l2_weight + 
                  coherence_loss * coherence_weight + 
                  manifold_loss * manifold_weight)
    
    print(f"\n" + "="*80)
    print("📊 損失組成總結")
    print("="*80)
    
    weighted_ce = ce_loss.item() * ce_weight
    weighted_l2 = l2_loss.item() * l2_weight
    weighted_coh = coherence_loss.item() * coherence_weight
    weighted_man = manifold_loss.item() * manifold_weight
    
    total = weighted_ce + weighted_l2 + weighted_coh + weighted_man
    
    print(f"\n各損失組件（加權後）:")
    print(f"  CE Loss:        {weighted_ce:10.4f} ({weighted_ce/total*100:5.1f}%)")
    print(f"  L2 Loss:        {weighted_l2:10.4f} ({weighted_l2/total*100:5.1f}%)")
    print(f"  Coherence:      {weighted_coh:10.4f} ({weighted_coh/total*100:5.1f}%)")
    print(f"  Manifold:       {weighted_man:10.4f} ({weighted_man/total*100:5.1f}%)")
    print(f"  " + "-"*50)
    print(f"  Total Loss:     {total:10.4f} (100.0%)")
    
    print(f"\n" + "="*80)
    print("🔍 診斷結論")
    print("="*80)
    
    ce_ratio = weighted_ce / total
    
    print(f"\nCE Loss 占比: {ce_ratio*100:.1f}%")
    
    if ce_ratio < 0.7:
        print(f"❌ CE Loss 占比過低！")
        print(f"   當前 CE weight = {ce_weight}")
        print(f"   建議增加到 {ce_weight * 2:.1f} 或更高")
        print(f"   目標：讓 CE Loss 占總損失的 85%+ ")
    elif ce_ratio < 0.85:
        print(f"⚠️  CE Loss 占比偏低")
        print(f"   建議將 CE weight 從 {ce_weight} 增加到 {ce_weight * 1.5:.1f}")
    else:
        print(f"✅ CE Loss 占比適當")
    
    if accuracy == 0:
        print(f"\n❌ Token 準確率為 0%！")
        print(f"   可能原因：")
        print(f"   1. CE Loss 權重仍然不足")
        print(f"   2. 模型容量不夠")
        print(f"   3. 訓練時間不夠（目前 Epoch 400）")
        print(f"   4. 學習率可能過低")
        print(f"\n💡 建議行動：")
        print(f"   1. 立即增加 CE weight 到 20.0-30.0")
        print(f"   2. 檢查訓練過程中 CE Loss 是否下降")
        print(f"   3. 檢查學習率是否過小")

print(f"\n" + "="*80)
