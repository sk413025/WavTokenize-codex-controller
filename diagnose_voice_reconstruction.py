#!/usr/bin/env python3
"""
完整診斷音頻重建問題
- 頻譜圖比較
- 音頻相似度
- Token 準確率
- 模型輸出分析

實驗編號: EXP20251021_01
生成函式: diagnose_voice_reconstruction
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import librosa
import librosa.display
from scipy.stats import pearsonr
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser

# 設定實驗編號和日期
EXP_ID = "EXP20251021_01"
DATE = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*80)
print(f"音頻重建完整診斷 - {EXP_ID}")
print(f"日期時間: {DATE}")
print("="*80)

# 創建輸出目錄
output_dir = Path(f"results/diagnosis_{EXP_ID}_{DATE}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n✅ 輸出目錄: {output_dir}")

# ============================================================================
# 第一部分：載入音頻檔案
# ============================================================================
print("\n" + "="*80)
print("第一部分：載入音頻檔案")
print("="*80)

# Epoch 400 的音頻
audio_dir = Path("results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_400")

input_file = audio_dir / "batch_0_sample_1_input.wav"
enhanced_file = audio_dir / "batch_0_sample_1_enhanced.wav"
target_file = audio_dir / "batch_0_sample_1_target.wav"

print(f"\n載入音頻檔案:")
print(f"  Input:    {input_file.exists()} - {input_file.name}")
print(f"  Enhanced: {enhanced_file.exists()} - {enhanced_file.name}")
print(f"  Target:   {target_file.exists()} - {target_file.name}")

if not all([input_file.exists(), enhanced_file.exists(), target_file.exists()]):
    print("\n❌ 錯誤：找不到必要的音頻檔案")
    exit(1)

# 載入音頻
input_audio, sr = torchaudio.load(input_file)
enhanced_audio, _ = torchaudio.load(enhanced_file)
target_audio, _ = torchaudio.load(target_file)

print(f"\n✅ 音頻已載入 (Sample Rate: {sr} Hz)")
print(f"  Input shape:    {input_audio.shape}")
print(f"  Enhanced shape: {enhanced_audio.shape}")
print(f"  Target shape:   {target_audio.shape}")

# 轉換為 numpy 並取單聲道
input_np = input_audio[0].numpy()
enhanced_np = enhanced_audio[0].numpy()
target_np = target_audio[0].numpy()

# ============================================================================
# 第二部分：基本音頻統計
# ============================================================================
print("\n" + "="*80)
print("第二部分：基本音頻統計")
print("="*80)

def compute_audio_stats(audio, name):
    """計算音頻統計資訊"""
    rms = np.sqrt(np.mean(audio**2))
    max_amp = np.max(np.abs(audio))
    mean_amp = np.mean(np.abs(audio))
    energy = np.sum(audio**2)
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))  # Zero Crossing Rate
    
    print(f"\n{name}:")
    print(f"  RMS:              {rms:.6f}")
    print(f"  Max Amplitude:    {max_amp:.6f}")
    print(f"  Mean |Amplitude|: {mean_amp:.6f}")
    print(f"  Energy:           {energy:.2f}")
    print(f"  Zero Crossing:    {zcr:.6f}")
    
    return {
        'rms': rms,
        'max': max_amp,
        'mean': mean_amp,
        'energy': energy,
        'zcr': zcr
    }

input_stats = compute_audio_stats(input_np, "Input (Noisy)")
enhanced_stats = compute_audio_stats(enhanced_np, "Enhanced (Model)")
target_stats = compute_audio_stats(target_np, "Target (Clean)")

# ============================================================================
# 第三部分：頻譜圖比較
# ============================================================================
print("\n" + "="*80)
print("第三部分：生成頻譜圖")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f'音頻重建診斷 - {EXP_ID}\nEpoch 400 vs Target', fontsize=16)

# 波形圖
time_axis = np.arange(len(input_np)) / sr

axes[0, 0].plot(time_axis, input_np, linewidth=0.5, alpha=0.7)
axes[0, 0].set_title('Input (Noisy) - Waveform')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time_axis, target_np, linewidth=0.5, alpha=0.7, color='green')
axes[0, 1].set_title('Target (Clean) - Waveform')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# Mel 頻譜圖
hop_length = 512
n_fft = 2048

input_mel = librosa.feature.melspectrogram(y=input_np, sr=sr, n_fft=n_fft, hop_length=hop_length)
input_mel_db = librosa.power_to_db(input_mel, ref=np.max)

enhanced_mel = librosa.feature.melspectrogram(y=enhanced_np, sr=sr, n_fft=n_fft, hop_length=hop_length)
enhanced_mel_db = librosa.power_to_db(enhanced_mel, ref=np.max)

target_mel = librosa.feature.melspectrogram(y=target_np, sr=sr, n_fft=n_fft, hop_length=hop_length)
target_mel_db = librosa.power_to_db(target_mel, ref=np.max)

img1 = librosa.display.specshow(input_mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[1, 0], cmap='viridis')
axes[1, 0].set_title('Input - Mel Spectrogram')
plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

img2 = librosa.display.specshow(enhanced_mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[1, 1], cmap='viridis')
axes[1, 1].set_title('Enhanced (Model Output) - Mel Spectrogram')
plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

img3 = librosa.display.specshow(target_mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[2, 0], cmap='viridis')
axes[2, 0].set_title('Target (Ground Truth) - Mel Spectrogram')
plt.colorbar(img3, ax=axes[2, 0], format='%+2.0f dB')

# 差異圖
diff_mel_db = enhanced_mel_db - target_mel_db
img4 = librosa.display.specshow(diff_mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[2, 1], cmap='RdBu_r', vmin=-40, vmax=40)
axes[2, 1].set_title('Difference (Enhanced - Target)')
plt.colorbar(img4, ax=axes[2, 1], format='%+2.0f dB')

plt.tight_layout()
spectrogram_path = output_dir / f"spectrogram_comparison_{EXP_ID}.png"
plt.savefig(spectrogram_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 頻譜圖已儲存: {spectrogram_path}")
plt.close()

# ============================================================================
# 第四部分：頻譜相似度分析
# ============================================================================
print("\n" + "="*80)
print("第四部分：頻譜相似度分析")
print("="*80)

# 計算頻譜相似度
enhanced_flat = enhanced_mel_db.flatten()
target_flat = target_mel_db.flatten()

# Pearson 相關係數
corr, p_value = pearsonr(enhanced_flat, target_flat)
print(f"\n頻譜 Pearson 相關係數: {corr:.4f} (p={p_value:.2e})")

# MSE
mse = np.mean((enhanced_mel_db - target_mel_db)**2)
print(f"頻譜 MSE: {mse:.4f} dB²")

# 頻帶能量分佈
def compute_band_energy(mel_db):
    """計算不同頻帶的能量"""
    n_bands = 4
    band_size = mel_db.shape[0] // n_bands
    band_energies = []
    
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else mel_db.shape[0]
        band_energy = np.mean(mel_db[start:end])
        band_energies.append(band_energy)
    
    return band_energies

enhanced_bands = compute_band_energy(enhanced_mel_db)
target_bands = compute_band_energy(target_mel_db)

print(f"\n頻帶能量分佈 (dB):")
print(f"  頻帶        Enhanced    Target      差異")
print(f"  低頻 (0-25%) {enhanced_bands[0]:8.2f}  {target_bands[0]:8.2f}  {enhanced_bands[0]-target_bands[0]:+8.2f}")
print(f"  中低 (25-50%){enhanced_bands[1]:8.2f}  {target_bands[1]:8.2f}  {enhanced_bands[1]-target_bands[1]:+8.2f}")
print(f"  中高 (50-75%){enhanced_bands[2]:8.2f}  {target_bands[2]:8.2f}  {enhanced_bands[2]-target_bands[2]:+8.2f}")
print(f"  高頻 (75-100%){enhanced_bands[3]:8.2f}  {target_bands[3]:8.2f}  {enhanced_bands[3]-target_bands[3]:+8.2f}")

# ============================================================================
# 第五部分：Token 準確率診斷
# ============================================================================
print("\n" + "="*80)
print("第五部分：Token 準確率診斷")
print("="*80)

print("\n載入模型進行 Token 分析...")

try:
    # 載入 checkpoint
    checkpoint_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359/checkpoint_epoch_400.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 創建模型
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
    
    print(f"✅ 模型已載入 (Epoch {checkpoint['epoch']})")
    
    # 編碼音頻到 tokens
    with torch.no_grad():
        input_tokens = model.encode_audio_to_tokens(input_audio)
        target_tokens = model.encode_audio_to_tokens(target_audio)
        
        # Transformer 預測（推理模式直接返回 predicted tokens，不是 logits）
        # 需要切換到訓練模式才能獲得 logits
        model.train()  # 臨時切換到訓練模式以獲取 logits
        logits = model.forward_transformer(input_tokens, target_tokens)
        predicted_tokens = torch.argmax(logits, dim=-1)
        model.eval()  # 切換回評估模式
        
        print(f"\n📊 Token 統計:")
        print(f"  Input tokens shape:     {input_tokens.shape}")
        print(f"  Target tokens shape:    {target_tokens.shape}")
        print(f"  Predicted tokens shape: {predicted_tokens.shape}")
        
        # Token 準確率
        input_tokens_np = input_tokens.cpu().numpy().flatten()
        target_tokens_np = target_tokens.cpu().numpy().flatten()
        predicted_tokens_np = predicted_tokens.cpu().numpy().flatten()
        
        # 確保長度一致
        min_len = min(len(predicted_tokens_np), len(target_tokens_np))
        predicted_tokens_np = predicted_tokens_np[:min_len]
        target_tokens_np = target_tokens_np[:min_len]
        
        correct = np.sum(predicted_tokens_np == target_tokens_np)
        accuracy = (correct / min_len) * 100
        
        print(f"\n🎯 Token 準確率: {accuracy:.2f}% ({correct}/{min_len})")
        
        # Token 多樣性
        unique_input = len(np.unique(input_tokens_np))
        unique_target = len(np.unique(target_tokens_np))
        unique_predicted = len(np.unique(predicted_tokens_np))
        
        print(f"\n📈 Token 多樣性:")
        print(f"  Input:     {unique_input}/4096 ({unique_input/4096*100:.2f}%)")
        print(f"  Target:    {unique_target}/4096 ({unique_target/4096*100:.2f}%)")
        print(f"  Predicted: {unique_predicted}/4096 ({unique_predicted/4096*100:.2f}%)")
        
        # Token 分佈
        from collections import Counter
        pred_counter = Counter(predicted_tokens_np)
        most_common = pred_counter.most_common(10)
        
        print(f"\n📊 最常見的 10 個預測 tokens:")
        for i, (token_id, count) in enumerate(most_common, 1):
            percentage = (count / min_len) * 100
            print(f"  {i:2d}. Token {token_id:4d}: {count:5d} 次 ({percentage:5.2f}%)")
        
        # 檢查是否有 mode collapse
        top1_ratio = most_common[0][1] / min_len
        top5_ratio = sum([c for _, c in most_common[:5]]) / min_len
        
        print(f"\n⚠️  Mode Collapse 檢查:")
        print(f"  Top 1 token 占比: {top1_ratio*100:.2f}%")
        print(f"  Top 5 tokens 占比: {top5_ratio*100:.2f}%")
        
        if top1_ratio > 0.5:
            print(f"  ❌ 嚴重 mode collapse！單一 token 占比超過 50%")
        elif top1_ratio > 0.3:
            print(f"  ⚠️  可能有 mode collapse，單一 token 占比過高")
        elif top5_ratio > 0.8:
            print(f"  ⚠️  可能有 mode collapse，前 5 個 tokens 占比超過 80%")
        else:
            print(f"  ✅ Token 分佈較為均勻")
        
except Exception as e:
    print(f"\n❌ Token 分析失敗: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 第六部分：問題診斷總結
# ============================================================================
print("\n" + "="*80)
print("第六部分：問題診斷總結")
print("="*80)

print(f"\n{'='*80}")
print("🔍 診斷結果")
print(f"{'='*80}")

issues = []
recommendations = []

# 1. 音頻振幅檢查
if enhanced_stats['rms'] < 0.01:
    issues.append("❌ Enhanced 音頻振幅過低 (RMS < 0.01)")
    recommendations.append("💡 檢查 decoder 輸出是否正常")
elif enhanced_stats['rms'] < target_stats['rms'] * 0.5:
    issues.append(f"⚠️  Enhanced 音頻振幅偏低 (僅為 Target 的 {enhanced_stats['rms']/target_stats['rms']*100:.1f}%)")
    recommendations.append("💡 可能需要調整輸出層的縮放")
else:
    print("✅ 音頻振幅正常")

# 2. 頻譜相似度檢查
if corr < 0.3:
    issues.append(f"❌ 頻譜相關性極低 (r={corr:.3f})")
    recommendations.append("💡 模型輸出與目標頻譜完全不同，可能是訓練未收斂")
elif corr < 0.6:
    issues.append(f"⚠️  頻譜相關性偏低 (r={corr:.3f})")
    recommendations.append("💡 模型部分學到特徵，但還需繼續訓練")
else:
    print(f"✅ 頻譜相關性良好 (r={corr:.3f})")

# 3. Token 準確率檢查
try:
    if accuracy < 10:
        issues.append(f"❌ Token 準確率極低 ({accuracy:.1f}%)")
        recommendations.append("💡 CE Loss weight 可能仍不足，建議增加到 15.0-20.0")
    elif accuracy < 30:
        issues.append(f"⚠️  Token 準確率偏低 ({accuracy:.1f}%)")
        recommendations.append("💡 訓練方向正確，建議繼續訓練到 600-800 epochs")
    elif accuracy < 50:
        print(f"⚠️  Token 準確率中等 ({accuracy:.1f}%)")
        recommendations.append("💡 已有明顯改善，繼續訓練應該會更好")
    else:
        print(f"✅ Token 準確率良好 ({accuracy:.1f}%)")
    
    # Mode collapse 檢查
    if top1_ratio > 0.5:
        issues.append(f"❌ 嚴重 mode collapse (單一 token 占 {top1_ratio*100:.1f}%)")
        recommendations.append("💡 增加 diversity loss 或調整 temperature")
    elif top1_ratio > 0.3:
        issues.append(f"⚠️  可能有 mode collapse (Top 1 占 {top1_ratio*100:.1f}%)")
except:
    pass

# 4. 頻帶能量檢查
band_diffs = [abs(e - t) for e, t in zip(enhanced_bands, target_bands)]
if max(band_diffs) > 20:
    issues.append(f"⚠️  某些頻帶能量差異過大 (最大差異: {max(band_diffs):.1f} dB)")
    recommendations.append("💡 檢查是否某些頻率範圍完全沒有學到")

print(f"\n{'='*80}")
print("❌ 發現的問題:")
print(f"{'='*80}")
for issue in issues:
    print(f"  {issue}")

if not issues:
    print("  ✅ 沒有發現明顯問題")

print(f"\n{'='*80}")
print("💡 建議修復措施:")
print(f"{'='*80}")
for rec in recommendations:
    print(f"  {rec}")

if not recommendations:
    print("  ✅ 模型訓練正常，建議繼續訓練或測試更多樣本")

# ============================================================================
# 儲存診斷報告
# ============================================================================
print(f"\n{'='*80}")
print("儲存診斷報告")
print(f"{'='*80}")

report_path = output_dir / f"diagnosis_report_{EXP_ID}.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"音頻重建診斷報告 - {EXP_ID}\n")
    f.write(f"{'='*80}\n")
    f.write(f"生成時間: {DATE}\n")
    f.write(f"生成函式: diagnose_voice_reconstruction\n\n")
    
    f.write(f"基本統計:\n")
    f.write(f"  Enhanced RMS: {enhanced_stats['rms']:.6f}\n")
    f.write(f"  Target RMS:   {target_stats['rms']:.6f}\n")
    f.write(f"  頻譜相關性:   {corr:.4f}\n")
    f.write(f"  頻譜 MSE:     {mse:.4f} dB²\n\n")
    
    try:
        f.write(f"Token 統計:\n")
        f.write(f"  準確率:       {accuracy:.2f}%\n")
        f.write(f"  多樣性:       {unique_predicted}/4096 ({unique_predicted/4096*100:.2f}%)\n")
        f.write(f"  Top 1 占比:   {top1_ratio*100:.2f}%\n\n")
    except:
        pass
    
    f.write(f"發現的問題:\n")
    for issue in issues:
        f.write(f"  {issue}\n")
    f.write(f"\n")
    
    f.write(f"建議措施:\n")
    for rec in recommendations:
        f.write(f"  {rec}\n")

print(f"✅ 報告已儲存: {report_path}")

print(f"\n{'='*80}")
print(f"診斷完成！")
print(f"{'='*80}")
print(f"\n輸出檔案:")
print(f"  📊 頻譜圖: {spectrogram_path}")
print(f"  📝 報告:   {report_path}")
print(f"\n請檢視頻譜圖和報告以了解詳細問題。")
print(f"{'='*80}\n")
