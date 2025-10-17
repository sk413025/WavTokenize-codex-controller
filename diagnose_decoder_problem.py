#!/usr/bin/env python3
"""
診斷實驗：Decoder 重建能力測試

核心假設驗證：
1. 如果 enhanced tokens == target tokens，Decoder 應該能重建 clean audio
2. 如果 Decoder 被凍結且正常工作，問題在於 Transformer 訓練過程

測試方案：
Test 1: Target tokens → Decoder → 應該得到 clean audio (Inside Test - 理想情況)
Test 2: Noisy tokens → Decoder → 得到什麼？
Test 3: 訓練模型的 enhanced tokens → Decoder → 得到什麼？
Test 4: 檢查 Transformer 訓練時的 loss 行為

目的：
- 確認 Decoder 本身是否正常
- 確認問題是否在 Transformer 的訓練過程
- 找出訓練哪裡出了問題
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(audio1, audio2, name1="Audio 1", name2="Audio 2"):
    """計算音頻質量指標"""
    # 確保維度一致
    if audio1.dim() == 3:
        audio1 = audio1.squeeze(0)
    if audio2.dim() == 3:
        audio2 = audio2.squeeze(0)
    
    if audio1.dim() == 2:
        audio1 = audio1.squeeze(0)
    if audio2.dim() == 2:
        audio2 = audio2.squeeze(0)
    
    # 確保長度一致
    min_len = min(audio1.size(-1), audio2.size(-1))
    audio1 = audio1[..., :min_len]
    audio2 = audio2[..., :min_len]
    
    # 轉為 numpy
    if isinstance(audio1, torch.Tensor):
        audio1 = audio1.detach().cpu().numpy()
    if isinstance(audio2, torch.Tensor):
        audio2 = audio2.detach().cpu().numpy()
    
    # 計算指標
    mse = np.mean((audio1 - audio2) ** 2)
    signal_power = np.mean(audio2 ** 2)
    noise_power = np.mean((audio1 - audio2) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    correlation = np.corrcoef(audio1.flatten(), audio2.flatten())[0, 1]
    
    return {
        'MSE': mse,
        'SNR_dB': snr,
        'Correlation': correlation,
        'name1': name1,
        'name2': name2
    }


def save_audio(audio, path, sample_rate=24000):
    """保存音頻"""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio.squeeze(0)
    
    if audio.abs().max() > 1.0:
        audio = audio / audio.abs().max()
    
    torchaudio.save(path, audio.cpu(), sample_rate)
    logger.info(f"保存音頻: {path}")


def plot_comparison(audios, labels, save_path, title="Audio Comparison"):
    """繪製音頻對比圖"""
    fig, axes = plt.subplots(len(audios), 1, figsize=(15, 3*len(audios)))
    
    if len(audios) == 1:
        axes = [axes]
    
    for i, (audio, label) in enumerate(zip(audios, labels)):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().squeeze().numpy()
        
        time = np.arange(len(audio)) / 24000
        axes[i].plot(time, audio, linewidth=0.5)
        axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([-1.1, 1.1])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"保存對比圖: {save_path}")


def test_1_target_tokens_decoder(model, target_audio, output_dir, device):
    """
    Test 1: Target Tokens → Decoder (Inside Test - 理想情況)
    
    如果這個測試失敗，說明 Decoder 本身有問題
    如果成功，說明 Decoder 正常，問題在別處
    """
    logger.info("\n" + "="*60)
    logger.info("Test 1: Target Tokens → Decoder (Inside Test)")
    logger.info("="*60)
    
    with torch.no_grad():
        # Step 1: Target audio → Encoder → Target tokens
        target_tokens = model.encode_audio_to_tokens(target_audio)
        logger.info(f"Target tokens shape: {target_tokens.shape}")
        logger.info(f"Target tokens range: [{target_tokens.min().item()}, {target_tokens.max().item()}]")
        logger.info(f"Target tokens example: {target_tokens[0, :10].tolist()}")
        
        # Step 2: Target tokens → Decoder → Reconstructed audio
        reconstructed_audio = model.decode_tokens_to_audio(target_tokens)
        logger.info(f"Reconstructed audio shape: {reconstructed_audio.shape}")
        
        # Step 3: 比較 reconstructed vs target
        metrics = calculate_metrics(reconstructed_audio, target_audio, 
                                   "Reconstructed", "Target")
        
        logger.info(f"\n結果分析:")
        logger.info(f"  SNR: {metrics['SNR_dB']:.2f} dB")
        logger.info(f"  Correlation: {metrics['Correlation']:.4f}")
        logger.info(f"  MSE: {metrics['MSE']:.6f}")
        
        # 判斷
        if metrics['SNR_dB'] > 5 and metrics['Correlation'] > 0.8:
            logger.info(f"  ✅ Test 1 PASSED: Decoder 工作正常！")
            test1_passed = True
        else:
            logger.info(f"  ❌ Test 1 FAILED: Decoder 重建品質不佳！")
            test1_passed = False
        
        # 保存結果
        test1_dir = output_dir / "test1_target_tokens_decoder"
        test1_dir.mkdir(parents=True, exist_ok=True)
        
        save_audio(target_audio, test1_dir / "target.wav")
        save_audio(reconstructed_audio, test1_dir / "reconstructed.wav")
        
        plot_comparison(
            [target_audio, reconstructed_audio],
            ["Target (Original)", "Reconstructed (Target Tokens → Decoder)"],
            test1_dir / "comparison.png",
            f"Test 1: Target Tokens → Decoder\nSNR: {metrics['SNR_dB']:.2f} dB, Corr: {metrics['Correlation']:.4f}"
        )
        
        return test1_passed, metrics, target_tokens


def test_2_noisy_tokens_decoder(model, noisy_audio, target_audio, output_dir, device):
    """
    Test 2: Noisy Tokens → Decoder
    
    看看 Noisy tokens 解碼後是什麼樣子
    """
    logger.info("\n" + "="*60)
    logger.info("Test 2: Noisy Tokens → Decoder")
    logger.info("="*60)
    
    with torch.no_grad():
        # Step 1: Noisy audio → Encoder → Noisy tokens
        noisy_tokens = model.encode_audio_to_tokens(noisy_audio)
        logger.info(f"Noisy tokens shape: {noisy_tokens.shape}")
        logger.info(f"Noisy tokens range: [{noisy_tokens.min().item()}, {noisy_tokens.max().item()}]")
        
        # Step 2: Noisy tokens → Decoder → Decoded audio
        decoded_noisy_audio = model.decode_tokens_to_audio(noisy_tokens)
        logger.info(f"Decoded noisy audio shape: {decoded_noisy_audio.shape}")
        
        # Step 3: 比較
        metrics_vs_target = calculate_metrics(decoded_noisy_audio, target_audio,
                                             "Decoded Noisy", "Target")
        metrics_vs_noisy = calculate_metrics(decoded_noisy_audio, noisy_audio,
                                            "Decoded Noisy", "Original Noisy")
        
        logger.info(f"\n結果分析:")
        logger.info(f"  vs Target - SNR: {metrics_vs_target['SNR_dB']:.2f} dB, Corr: {metrics_vs_target['Correlation']:.4f}")
        logger.info(f"  vs Noisy  - SNR: {metrics_vs_noisy['SNR_dB']:.2f} dB, Corr: {metrics_vs_noisy['Correlation']:.4f}")
        
        # 保存結果
        test2_dir = output_dir / "test2_noisy_tokens_decoder"
        test2_dir.mkdir(parents=True, exist_ok=True)
        
        save_audio(noisy_audio, test2_dir / "noisy.wav")
        save_audio(decoded_noisy_audio, test2_dir / "decoded_noisy.wav")
        save_audio(target_audio, test2_dir / "target.wav")
        
        plot_comparison(
            [noisy_audio, decoded_noisy_audio, target_audio],
            ["Noisy (Original)", "Decoded (Noisy Tokens → Decoder)", "Target (Clean)"],
            test2_dir / "comparison.png",
            f"Test 2: Noisy Tokens → Decoder\nvs Target SNR: {metrics_vs_target['SNR_dB']:.2f} dB"
        )
        
        return metrics_vs_target, metrics_vs_noisy, noisy_tokens


def test_3_enhanced_tokens_decoder(model, noisy_audio, target_audio, output_dir, device):
    """
    Test 3: 訓練模型的 Enhanced Tokens → Decoder
    
    這是實際的 enhancement 測試
    """
    logger.info("\n" + "="*60)
    logger.info("Test 3: Enhanced Tokens (Trained Model) → Decoder")
    logger.info("="*60)
    
    with torch.no_grad():
        # Step 1: Noisy audio → Model → Enhanced audio
        output = model(noisy_audio)
        enhanced_audio = output['denoised_audio']
        enhanced_tokens = output['denoised_tokens']
        
        logger.info(f"Enhanced tokens shape: {enhanced_tokens.shape}")
        logger.info(f"Enhanced tokens range: [{enhanced_tokens.min().item()}, {enhanced_tokens.max().item()}]")
        
        # Step 2: 比較
        metrics = calculate_metrics(enhanced_audio, target_audio,
                                   "Enhanced", "Target")
        
        logger.info(f"\n結果分析:")
        logger.info(f"  SNR: {metrics['SNR_dB']:.2f} dB")
        logger.info(f"  Correlation: {metrics['Correlation']:.4f}")
        logger.info(f"  MSE: {metrics['MSE']:.6f}")
        
        if metrics['SNR_dB'] < 0:
            logger.info(f"  ❌ Enhancement 失敗：音頻變得更差！")
        elif metrics['SNR_dB'] < 5:
            logger.info(f"  ⚠️ Enhancement 不佳：改善有限")
        else:
            logger.info(f"  ✅ Enhancement 有效")
        
        # 保存結果
        test3_dir = output_dir / "test3_enhanced_tokens_decoder"
        test3_dir.mkdir(parents=True, exist_ok=True)
        
        save_audio(noisy_audio, test3_dir / "noisy.wav")
        save_audio(enhanced_audio, test3_dir / "enhanced.wav")
        save_audio(target_audio, test3_dir / "target.wav")
        
        plot_comparison(
            [noisy_audio, enhanced_audio, target_audio],
            ["Noisy", "Enhanced (Trained Model)", "Target"],
            test3_dir / "comparison.png",
            f"Test 3: Enhanced Tokens → Decoder\nSNR: {metrics['SNR_dB']:.2f} dB, Corr: {metrics['Correlation']:.4f}"
        )
        
        return metrics, enhanced_tokens


def test_4_token_comparison(target_tokens, noisy_tokens, enhanced_tokens, output_dir):
    """
    Test 4: Token 序列比較
    
    分析 tokens 之間的差異
    """
    logger.info("\n" + "="*60)
    logger.info("Test 4: Token Sequence Comparison")
    logger.info("="*60)
    
    # 確保長度一致
    min_len = min(target_tokens.size(1), noisy_tokens.size(1), enhanced_tokens.size(1))
    target_tokens = target_tokens[:, :min_len]
    noisy_tokens = noisy_tokens[:, :min_len]
    enhanced_tokens = enhanced_tokens[:, :min_len]
    
    # 計算 token accuracy
    enhanced_vs_target_acc = (enhanced_tokens == target_tokens).float().mean().item()
    noisy_vs_target_acc = (noisy_tokens == target_tokens).float().mean().item()
    enhanced_vs_noisy_acc = (enhanced_tokens == noisy_tokens).float().mean().item()
    
    logger.info(f"\nToken Accuracy:")
    logger.info(f"  Enhanced vs Target: {enhanced_vs_target_acc:.4f} ({enhanced_vs_target_acc*100:.2f}%)")
    logger.info(f"  Noisy vs Target:    {noisy_vs_target_acc:.4f} ({noisy_vs_target_acc*100:.2f}%)")
    logger.info(f"  Enhanced vs Noisy:  {enhanced_vs_noisy_acc:.4f} ({enhanced_vs_noisy_acc*100:.2f}%)")
    
    # 分析 token 變化
    target_tokens_np = target_tokens[0].cpu().numpy()
    noisy_tokens_np = noisy_tokens[0].cpu().numpy()
    enhanced_tokens_np = enhanced_tokens[0].cpu().numpy()
    
    # 繪製 token 序列
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Token sequences
    axes[0].plot(target_tokens_np, 'g-', label='Target', linewidth=1)
    axes[0].set_title('Target Tokens', fontsize=12)
    axes[0].set_ylabel('Token ID')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(noisy_tokens_np, 'r-', label='Noisy', linewidth=1)
    axes[1].set_title('Noisy Tokens', fontsize=12)
    axes[1].set_ylabel('Token ID')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(enhanced_tokens_np, 'b-', label='Enhanced', linewidth=1)
    axes[2].set_title('Enhanced Tokens (Trained Model)', fontsize=12)
    axes[2].set_ylabel('Token ID')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Token differences
    enhanced_diff = np.abs(enhanced_tokens_np - target_tokens_np)
    noisy_diff = np.abs(noisy_tokens_np - target_tokens_np)
    
    axes[3].plot(noisy_diff, 'r-', label='Noisy vs Target', linewidth=1, alpha=0.7)
    axes[3].plot(enhanced_diff, 'b-', label='Enhanced vs Target', linewidth=1, alpha=0.7)
    axes[3].set_title('Token Difference from Target', fontsize=12)
    axes[3].set_xlabel('Token Position')
    axes[3].set_ylabel('Absolute Difference')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    save_path = output_dir / "test4_token_comparison" / "token_sequences.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nToken 統計:")
    logger.info(f"  Target tokens  - mean: {target_tokens_np.mean():.1f}, std: {target_tokens_np.std():.1f}")
    logger.info(f"  Noisy tokens   - mean: {noisy_tokens_np.mean():.1f}, std: {noisy_tokens_np.std():.1f}")
    logger.info(f"  Enhanced tokens- mean: {enhanced_tokens_np.mean():.1f}, std: {enhanced_tokens_np.std():.1f}")
    
    # 關鍵發現
    logger.info(f"\n關鍵發現:")
    if enhanced_vs_target_acc < 0.3:
        logger.info(f"  ❌ Enhanced tokens 與 Target tokens 差異巨大 (accuracy < 30%)")
        logger.info(f"     → Transformer 沒有學會正確的 token 映射")
    elif enhanced_vs_target_acc < 0.7:
        logger.info(f"  ⚠️ Enhanced tokens 與 Target tokens 有一定差異 (30% < accuracy < 70%)")
        logger.info(f"     → Transformer 部分學會，但不夠準確")
    else:
        logger.info(f"  ✅ Enhanced tokens 與 Target tokens 較接近 (accuracy > 70%)")
        logger.info(f"     → Transformer 學習較好，問題可能在 Decoder")
    
    if enhanced_vs_noisy_acc > 0.8:
        logger.info(f"  ❌ Enhanced tokens 幾乎等於 Noisy tokens (similarity > 80%)")
        logger.info(f"     → Transformer 幾乎沒有做任何改變！")
    
    return {
        'enhanced_vs_target_acc': enhanced_vs_target_acc,
        'noisy_vs_target_acc': noisy_vs_target_acc,
        'enhanced_vs_noisy_acc': enhanced_vs_noisy_acc
    }


def main():
    """主診斷流程"""
    logger.info("="*80)
    logger.info("Decoder 重建能力診斷實驗")
    logger.info("="*80)
    
    # 設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 模型路徑
    model_path = "/home/sbplab/ruizi/c_code/results/wavtokenizer_tokenloss_fixed_202510150302/best_model.pth"
    data_root = "/home/sbplab/ruizi/WavTokenize/1n"
    output_dir = Path("results/decoder_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    logger.info(f"\n載入模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = WavTokenizerTransformerDenoiser(
        config_path=config['config_path'],
        model_path=config['model_path'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        max_length=config['max_length'],
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("模型載入完成")
    
    # 載入測試數據（使用一個樣本）
    logger.info(f"\n載入測試數據...")
    # 使用實際的文件命名格式
    speaker = "boy1"
    sentence_id = "001"
    
    noisy_path = Path(data_root) / f"nor_{speaker}_box_LDV_{sentence_id}.wav"
    target_path = Path(data_root) / f"nor_{speaker}_clean_{sentence_id}.wav"
    
    if not noisy_path.exists():
        logger.error(f"Noisy file not found: {noisy_path}")
        return
    if not target_path.exists():
        logger.error(f"Target file not found: {target_path}")
        return
    
    logger.info(f"Noisy: {noisy_path}")
    logger.info(f"Target: {target_path}")
    
    noisy_audio, sr1 = torchaudio.load(noisy_path)
    target_audio, sr2 = torchaudio.load(target_path)
    
    # 轉換為 24kHz
    if sr1 != 24000:
        noisy_audio = torchaudio.functional.resample(noisy_audio, sr1, 24000)
    if sr2 != 24000:
        target_audio = torchaudio.functional.resample(target_audio, sr2, 24000)
    
    # 添加 batch 維度並移到設備
    noisy_audio = noisy_audio.unsqueeze(0).to(device)
    target_audio = target_audio.unsqueeze(0).to(device)
    
    logger.info(f"Noisy audio shape: {noisy_audio.shape}")
    logger.info(f"Target audio shape: {target_audio.shape}")
    
    # 執行診斷測試
    results = {}
    
    # Test 1: Target tokens → Decoder (Inside Test)
    test1_passed, test1_metrics, target_tokens = test_1_target_tokens_decoder(
        model, target_audio, output_dir, device
    )
    results['test1'] = {'passed': test1_passed, 'metrics': test1_metrics}
    
    # Test 2: Noisy tokens → Decoder
    test2_metrics_vs_target, test2_metrics_vs_noisy, noisy_tokens = test_2_noisy_tokens_decoder(
        model, noisy_audio, target_audio, output_dir, device
    )
    results['test2'] = {
        'metrics_vs_target': test2_metrics_vs_target,
        'metrics_vs_noisy': test2_metrics_vs_noisy
    }
    
    # Test 3: Enhanced tokens → Decoder
    test3_metrics, enhanced_tokens = test_3_enhanced_tokens_decoder(
        model, noisy_audio, target_audio, output_dir, device
    )
    results['test3'] = {'metrics': test3_metrics}
    
    # Test 4: Token comparison
    test4_results = test_4_token_comparison(
        target_tokens, noisy_tokens, enhanced_tokens, output_dir
    )
    results['test4'] = test4_results
    
    # 生成診斷報告
    logger.info("\n" + "="*80)
    logger.info("診斷報告總結")
    logger.info("="*80)
    
    report_lines = []
    report_lines.append("# Decoder 重建能力診斷報告\n")
    report_lines.append(f"日期: {Path.cwd()}\n")
    report_lines.append(f"模型: {model_path}\n")
    report_lines.append(f"測試樣本: {speaker}/{sentence_id}\n\n")
    
    report_lines.append("## Test 1: Target Tokens → Decoder (Inside Test)\n")
    report_lines.append(f"**目的**: 驗證 Decoder 本身是否正常\n")
    report_lines.append(f"**結果**: {'✅ PASSED' if test1_passed else '❌ FAILED'}\n")
    report_lines.append(f"- SNR: {test1_metrics['SNR_dB']:.2f} dB\n")
    report_lines.append(f"- Correlation: {test1_metrics['Correlation']:.4f}\n")
    report_lines.append(f"- MSE: {test1_metrics['MSE']:.6f}\n\n")
    
    report_lines.append("## Test 2: Noisy Tokens → Decoder\n")
    report_lines.append(f"**目的**: 觀察 Noisy tokens 解碼結果\n")
    report_lines.append(f"- vs Target: SNR {test2_metrics_vs_target['SNR_dB']:.2f} dB, Corr {test2_metrics_vs_target['Correlation']:.4f}\n")
    report_lines.append(f"- vs Noisy:  SNR {test2_metrics_vs_noisy['SNR_dB']:.2f} dB, Corr {test2_metrics_vs_noisy['Correlation']:.4f}\n\n")
    
    report_lines.append("## Test 3: Enhanced Tokens → Decoder\n")
    report_lines.append(f"**目的**: 測試訓練模型的實際表現\n")
    report_lines.append(f"- SNR: {test3_metrics['SNR_dB']:.2f} dB\n")
    report_lines.append(f"- Correlation: {test3_metrics['Correlation']:.4f}\n")
    report_lines.append(f"- MSE: {test3_metrics['MSE']:.6f}\n\n")
    
    report_lines.append("## Test 4: Token Sequence Analysis\n")
    report_lines.append(f"- Enhanced vs Target accuracy: {test4_results['enhanced_vs_target_acc']:.4f} ({test4_results['enhanced_vs_target_acc']*100:.2f}%)\n")
    report_lines.append(f"- Noisy vs Target accuracy: {test4_results['noisy_vs_target_acc']:.4f} ({test4_results['noisy_vs_target_acc']*100:.2f}%)\n")
    report_lines.append(f"- Enhanced vs Noisy similarity: {test4_results['enhanced_vs_noisy_acc']:.4f} ({test4_results['enhanced_vs_noisy_acc']*100:.2f}%)\n\n")
    
    report_lines.append("## 診斷結論\n\n")
    
    if test1_passed:
        report_lines.append("### ✅ Decoder 本身正常\n")
        report_lines.append("Test 1 通過，說明 WavTokenizer Decoder 被凍結且工作正常。\n")
        report_lines.append("問題不在 Decoder，而在 **Transformer 的訓練過程**。\n\n")
        
        if test4_results['enhanced_vs_target_acc'] < 0.3:
            report_lines.append("### ❌ 主要問題：Transformer 沒有學會正確的 Token 映射\n")
            report_lines.append(f"Enhanced tokens 與 Target tokens 的準確率只有 {test4_results['enhanced_vs_target_acc']*100:.2f}%。\n")
            report_lines.append("**可能原因**:\n")
            report_lines.append("1. Teacher Forcing 導致訓練推理不一致\n")
            report_lines.append("2. 損失函數只優化 token-level，沒有 audio-level 監督\n")
            report_lines.append("3. Discrete token 空間限制，難以優化\n")
            report_lines.append("4. 訓練數據或訓練策略有問題\n\n")
        
        if test4_results['enhanced_vs_noisy_acc'] > 0.8:
            report_lines.append("### ⚠️ 嚴重問題：Transformer 幾乎沒有改變 Tokens\n")
            report_lines.append(f"Enhanced tokens 與 Noisy tokens 相似度高達 {test4_results['enhanced_vs_noisy_acc']*100:.2f}%。\n")
            report_lines.append("**這說明 Transformer 幾乎沒有學到任何降噪能力！**\n\n")
    else:
        report_lines.append("### ❌ Decoder 本身有問題\n")
        report_lines.append("Test 1 失敗，說明即使是正確的 Target tokens 也無法被正確重建。\n")
        report_lines.append("**需要檢查**:\n")
        report_lines.append("1. `decode_tokens_to_audio` 函數的實現\n")
        report_lines.append("2. Token 維度轉換是否正確\n")
        report_lines.append("3. `codes_to_features` 的使用方式\n")
        report_lines.append("4. bandwidth_id 設定\n\n")
    
    report_lines.append("## 建議的解決方案\n\n")
    
    if test1_passed and test4_results['enhanced_vs_target_acc'] < 0.5:
        report_lines.append("### 推薦：使用 TTT2 Token Enhancement 架構\n")
        report_lines.append("當前問題的根源在於:\n")
        report_lines.append("1. **Discrete token 空間**: 4096 個離散值限制了表達能力\n")
        report_lines.append("2. **Teacher Forcing**: 訓練推理不一致\n")
        report_lines.append("3. **缺少 Audio-level 監督**: 只優化 token loss\n\n")
        report_lines.append("TTT2 Token 的改進:\n")
        report_lines.append("- ✅ 在 **Embedding 空間** 工作（連續，非離散）\n")
        report_lines.append("- ✅ **多目標損失** (Token CE + Feature L2 + Audio L1 + Token Smooth)\n")
        report_lines.append("- ✅ **No Teacher Forcing**（訓練推理一致）\n")
        report_lines.append("- ✅ **Audio-level 監督**（直接優化音頻質量）\n\n")
        report_lines.append("**立即執行**:\n")
        report_lines.append("```bash\n")
        report_lines.append("bash run_ttt2_token.sh\n")
        report_lines.append("```\n\n")
    
    report_text = "".join(report_lines)
    
    # 保存報告
    report_path = output_dir / "DIAGNOSIS_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"\n診斷報告已保存: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("診斷完成！")
    logger.info("="*80)
    
    # 打印關鍵結論
    if test1_passed:
        logger.info("\n✅ 核心結論: Decoder 正常，問題在 Transformer 訓練")
        logger.info(f"   Enhanced vs Target Token Accuracy: {test4_results['enhanced_vs_target_acc']*100:.2f}%")
        logger.info(f"   → Transformer 需要重新設計訓練方式")
    else:
        logger.info("\n❌ 核心結論: Decoder 本身有問題")
        logger.info(f"   Target Tokens → Decoder SNR: {test1_metrics['SNR_dB']:.2f} dB")
        logger.info(f"   → 需要修復 decoder 實現")


if __name__ == '__main__':
    main()
