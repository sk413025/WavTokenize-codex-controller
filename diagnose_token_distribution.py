#!/usr/bin/env python3
"""
Token 分佈診斷腳本
檢查模型是否學到有意義的 token 預測
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
import torchaudio

def analyze_token_distribution(model, audio_path, label):
    """分析單個音頻的 token 分佈"""
    # 載入音頻
    waveform, sr = torchaudio.load(audio_path)
    
    # 重採樣到 24kHz
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)
    
    # 確保形狀正確 [1, 1, T]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    with torch.no_grad():
        # 提取 tokens
        tokens = model.encode_audio_to_tokens(waveform)
        
        # 統計
        unique_tokens, counts = torch.unique(tokens, return_counts=True)
        token_diversity = len(unique_tokens)
        most_common_token = unique_tokens[counts.argmax()].item()
        most_common_ratio = counts.max().item() / tokens.numel()
        
        print(f"\n{label}:")
        print(f"  Token 序列長度: {tokens.shape}")
        print(f"  Token 範圍: [{tokens.min().item()}, {tokens.max().item()}]")
        print(f"  唯一 token 數量: {token_diversity} / 4096")
        print(f"  Token 多樣性: {token_diversity/4096*100:.2f}%")
        print(f"  最常見 token: {most_common_token} (佔 {most_common_ratio*100:.2f}%)")
        
        if most_common_ratio > 0.5:
            print(f"  ⚠️  警告：超過 50% 的 tokens 都是同一個！")
        elif most_common_ratio > 0.3:
            print(f"  ⚠️  警告：Token 分佈過於集中（{most_common_ratio*100:.2f}%）")
        else:
            print(f"  ✅ Token 分佈正常")
        
        return {
            'tokens': tokens.cpu().numpy(),
            'unique_tokens': unique_tokens.cpu().numpy(),
            'counts': counts.cpu().numpy(),
            'diversity': token_diversity,
            'most_common_token': most_common_token,
            'most_common_ratio': most_common_ratio
        }

def compare_predicted_vs_target_tokens(model, noisy_audio_path, clean_audio_path):
    """比較模型預測的 tokens 與 target tokens"""
    print(f"\n{'='*70}")
    print("比較預測 Tokens vs Target Tokens")
    print('='*70)
    
    # 載入音頻
    noisy_waveform, sr1 = torchaudio.load(noisy_audio_path)
    clean_waveform, sr2 = torchaudio.load(clean_audio_path)
    
    # 重採樣
    if sr1 != 24000:
        resampler = torchaudio.transforms.Resample(sr1, 24000)
        noisy_waveform = resampler(noisy_waveform)
    if sr2 != 24000:
        resampler = torchaudio.transforms.Resample(sr2, 24000)
        clean_waveform = resampler(clean_waveform)
    
    # 確保形狀 [1, 1, T]
    if noisy_waveform.dim() == 2:
        noisy_waveform = noisy_waveform.unsqueeze(0)
    if clean_waveform.dim() == 2:
        clean_waveform = clean_waveform.unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        # 獲取模型預測的 tokens（使用推理模式）
        output = model(noisy_waveform)
        predicted_tokens = output['denoised_tokens']  # [B, T]
        
        # 獲取 target tokens
        target_tokens = model.encode_audio_to_tokens(clean_waveform)
        
        # 獲取 noisy tokens
        noisy_tokens = model.encode_audio_to_tokens(noisy_waveform)
        
        print(f"\nNoisy Tokens:")
        print(f"  Shape: {noisy_tokens.shape}")
        print(f"  範圍: [{noisy_tokens.min().item()}, {noisy_tokens.max().item()}]")
        print(f"  唯一 tokens: {len(torch.unique(noisy_tokens))}")
        
        print(f"\nTarget Tokens:")
        print(f"  Shape: {target_tokens.shape}")
        print(f"  範圍: [{target_tokens.min().item()}, {target_tokens.max().item()}]")
        print(f"  唯一 tokens: {len(torch.unique(target_tokens))}")
        
        print(f"\nPredicted Tokens:")
        print(f"  Shape: {predicted_tokens.shape}")
        print(f"  範圍: [{predicted_tokens.min().item()}, {predicted_tokens.max().item()}]")
        print(f"  唯一 tokens: {len(torch.unique(predicted_tokens))}")
        
        # 計算準確率
        # 需要調整序列長度使其匹配
        min_len = min(predicted_tokens.shape[1], target_tokens.shape[1])
        pred_trimmed = predicted_tokens[:, :min_len]
        targ_trimmed = target_tokens[:, :min_len]
        
        accuracy = (pred_trimmed == targ_trimmed).float().mean().item()
        print(f"\nToken 準確率: {accuracy*100:.2f}%")
        
        if accuracy < 0.1:
            print(f"  ❌ 準確率極低（<10%），模型幾乎沒有學到正確預測")
        elif accuracy < 0.3:
            print(f"  ⚠️  準確率偏低（<30%），模型學習效果不佳")
        elif accuracy < 0.5:
            print(f"  ⚠️  準確率中等（<50%），還有改進空間")
        else:
            print(f"  ✅ 準確率良好（≥50%）")
        
        # 檢查預測的 token 是否過度集中
        unique_pred, counts_pred = torch.unique(pred_trimmed, return_counts=True)
        most_common_pred = unique_pred[counts_pred.argmax()].item()
        most_common_ratio_pred = counts_pred.max().item() / pred_trimmed.numel()
        
        print(f"\n預測 Token 分佈:")
        print(f"  唯一 tokens: {len(unique_pred)} / 4096")
        print(f"  最常見 token: {most_common_pred} (佔 {most_common_ratio_pred*100:.2f}%)")
        
        if most_common_ratio_pred > 0.8:
            print(f"  ❌ 嚴重問題：模型總是預測同一個 token（{most_common_ratio_pred*100:.2f}%）！")
            print(f"     這表示模型沒有真正學到內容，只是記住了某個「安全」的輸出")
        elif most_common_ratio_pred > 0.5:
            print(f"  ⚠️  警告：預測過於集中（{most_common_ratio_pred*100:.2f}%）")
        
        return {
            'accuracy': accuracy,
            'predicted_tokens': predicted_tokens.cpu().numpy(),
            'target_tokens': target_tokens.cpu().numpy(),
            'noisy_tokens': noisy_tokens.cpu().numpy(),
            'most_common_ratio': most_common_ratio_pred
        }

def visualize_token_sequences(results, epoch):
    """視覺化 token 序列"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    
    # 只顯示前 100 個 tokens
    max_display = 100
    
    noisy_tokens = results['noisy_tokens'][0, :max_display]
    target_tokens = results['target_tokens'][0, :max_display]
    predicted_tokens = results['predicted_tokens'][0, :max_display]
    
    axes[0].plot(noisy_tokens, 'o-', markersize=3, linewidth=0.5, label='Noisy')
    axes[0].set_title(f"Noisy Tokens (前 {max_display} 個)")
    axes[0].set_ylabel("Token ID")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(target_tokens, 'o-', markersize=3, linewidth=0.5, label='Target', color='green')
    axes[1].set_title(f"Target Tokens (前 {max_display} 個)")
    axes[1].set_ylabel("Token ID")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(predicted_tokens, 'o-', markersize=3, linewidth=0.5, label='Predicted', color='red')
    axes[2].set_title(f"Predicted Tokens (前 {max_display} 個) - Accuracy: {results['accuracy']*100:.2f}%")
    axes[2].set_xlabel("Token Position")
    axes[2].set_ylabel("Token ID")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    output_path = f"token_analysis_epoch{epoch}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nToken 序列對比圖已儲存: {output_path}")
    plt.close()

def main():
    print("="*70)
    print("Token 分佈診斷")
    print("="*70)
    
    # 載入模型
    print("\n載入模型...")
    checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510190523/checkpoint_epoch_300.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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
    model.eval()
    print("✅ 模型載入成功")
    
    # 分析 epoch 300 的樣本
    epoch = 300
    base_path = Path(f"results/transformer_large_tokenloss_large_tokenloss_202510190523/audio_samples/epoch_{epoch}")
    
    noisy_path = base_path / "batch_0_sample_1_input.wav"
    target_path = base_path / "batch_0_sample_1_target.wav"
    enhanced_path = base_path / "batch_0_sample_1_enhanced.wav"
    
    # 1. 分析各個音頻的 token 分佈
    print(f"\n{'='*70}")
    print(f"分析音頻的 Token 分佈（Epoch {epoch}）")
    print('='*70)
    
    noisy_stats = analyze_token_distribution(model, noisy_path, "Noisy Audio")
    target_stats = analyze_token_distribution(model, target_path, "Target Audio")
    enhanced_stats = analyze_token_distribution(model, enhanced_path, "Enhanced Audio")
    
    # 2. 比較預測 vs target tokens
    results = compare_predicted_vs_target_tokens(model, noisy_path, target_path)
    
    # 3. 視覺化
    visualize_token_sequences(results, epoch)
    
    # 總結
    print("\n" + "="*70)
    print("診斷總結")
    print("="*70)
    
    if results['accuracy'] < 0.1:
        print("\n❌ 嚴重問題：Token 準確率極低")
        print("   原因：模型沒有學到從 noisy tokens 預測 clean tokens 的對應關係")
        print("   建議：")
        print("   1. 檢查 Token Loss 權重（CE 可能太低，無法學習 token 預測）")
        print("   2. 增加 CE Loss 權重從 1.0 提高到 5.0 或 10.0")
        print("   3. 減少其他損失的權重（L2_Embed, Coherence, Manifold）")
        print("   4. 確認 teacher forcing 是否正常運作")
    
    if results['most_common_ratio'] > 0.5:
        print("\n❌ 嚴重問題：模型總是預測同一個 token")
        print("   原因：模型找到了一個「安全」的輸出，避免犯錯")
        print("   建議：")
        print("   1. 增加 token prediction 的難度（減少 teacher forcing ratio）")
        print("   2. 增加 CE Loss 權重強迫模型學習正確分類")
        print("   3. 檢查是否有 label smoothing 或其他正則化技術過度使用")
    
    if enhanced_stats['diversity'] < 100:
        print(f"\n❌ 嚴重問題：Enhanced audio 的 token 多樣性極低（{enhanced_stats['diversity']}/4096）")
        print("   這解釋了為什麼音頻聽起來沒有內容")
        print("   模型產生的 tokens 過於單調，無法重建豐富的語音內容")

if __name__ == "__main__":
    main()
