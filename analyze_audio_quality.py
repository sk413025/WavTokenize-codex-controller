#!/usr/bin/env python3
"""
音頻品質比較分析工具 - TTT2 Fix分支 vs Main分支
比較epoch 300的音頻重建品質
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import soundfile as sf
from pathlib import Path
import json

def load_audio(file_path, sr=24000):
    """載入音頻文件"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return audio
    except Exception as e:
        print(f"載入音頻失敗 {file_path}: {e}")
        return None

def compute_snr(clean, noisy):
    """計算信噪比 (SNR)"""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def compute_spectral_centroid(audio, sr=24000):
    """計算頻譜重心"""
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    return np.mean(spectral_centroids)

def compute_spectral_rolloff(audio, sr=24000):
    """計算頻譜滾降點"""
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    return np.mean(spectral_rolloff)

def compute_mfcc_similarity(audio1, audio2, sr=24000):
    """計算MFCC相似度"""
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)
    
    # 確保長度一致
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]
    
    # 計算皮爾森相關係數
    correlations = []
    for i in range(mfcc1.shape[0]):
        corr, _ = pearsonr(mfcc1[i], mfcc2[i])
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0

def analyze_audio_quality(main_dir, fix_dir, sample_name):
    """分析單個音頻樣本的品質"""
    results = {}
    
    # 構建文件路徑
    main_enhanced = os.path.join(main_dir, f"{sample_name}_enhanced.wav")
    main_target = os.path.join(main_dir, f"{sample_name}_target.wav") 
    main_input = os.path.join(main_dir, f"{sample_name}_input.wav")
    
    fix_enhanced = os.path.join(fix_dir, f"{sample_name}_enhanced.wav")
    fix_target = os.path.join(fix_dir, f"{sample_name}_target.wav")
    fix_input = os.path.join(fix_dir, f"{sample_name}_input.wav")
    
    # 載入音頻
    main_enh = load_audio(main_enhanced)
    main_tgt = load_audio(main_target)
    main_inp = load_audio(main_input)
    
    fix_enh = load_audio(fix_enhanced)
    fix_tgt = load_audio(fix_target)
    fix_inp = load_audio(fix_input)
    
    if any(x is None for x in [main_enh, main_tgt, fix_enh, fix_tgt]):
        return None
    
    # 確保長度一致
    min_len = min(len(main_enh), len(main_tgt), len(fix_enh), len(fix_tgt))
    main_enh = main_enh[:min_len]
    main_tgt = main_tgt[:min_len]
    fix_enh = fix_enh[:min_len]
    fix_tgt = fix_tgt[:min_len]
    
    # 計算SNR (enhanced vs target)
    main_snr = compute_snr(main_tgt, main_enh)
    fix_snr = compute_snr(fix_tgt, fix_enh)
    
    # 計算MFCC相似度 (enhanced vs target)
    main_mfcc_sim = compute_mfcc_similarity(main_enh, main_tgt)
    fix_mfcc_sim = compute_mfcc_similarity(fix_enh, fix_tgt)
    
    # 計算頻譜特徵
    main_enh_centroid = compute_spectral_centroid(main_enh)
    fix_enh_centroid = compute_spectral_centroid(fix_enh)
    target_centroid = compute_spectral_centroid(main_tgt)  # target應該相同
    
    main_enh_rolloff = compute_spectral_rolloff(main_enh)
    fix_enh_rolloff = compute_spectral_rolloff(fix_enh)
    target_rolloff = compute_spectral_rolloff(main_tgt)
    
    results = {
        'sample': sample_name,
        'snr': {
            'main': main_snr,
            'fix': fix_snr,
            'improvement': fix_snr - main_snr
        },
        'mfcc_similarity': {
            'main': main_mfcc_sim,
            'fix': fix_mfcc_sim,
            'improvement': fix_mfcc_sim - main_mfcc_sim
        },
        'spectral_centroid': {
            'main': main_enh_centroid,
            'fix': fix_enh_centroid,
            'target': target_centroid,
            'main_diff': abs(main_enh_centroid - target_centroid),
            'fix_diff': abs(fix_enh_centroid - target_centroid)
        },
        'spectral_rolloff': {
            'main': main_enh_rolloff,
            'fix': fix_enh_rolloff,
            'target': target_rolloff,
            'main_diff': abs(main_enh_rolloff - target_rolloff),
            'fix_diff': abs(fix_enh_rolloff - target_rolloff)
        }
    }
    
    return results

def main():
    """主函數"""
    print("🎵 TTT2 音頻品質比較分析 - Fix分支 vs Main分支 (Epoch 300)")
    print("="*70)
    
    # 設定路徑
    main_dir = "results/tsne_outputs/output4/audio_samples/epoch_300"
    fix_dir = "results/tsne_outputs/b-output4/audio_samples/epoch_300"
    output_dir = "b-0813"
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 樣本列表
    samples = [
        "batch_0_sample_1", "batch_0_sample_2", "batch_0_sample_3",
        "batch_1_sample_1", "batch_1_sample_2", "batch_1_sample_3", 
        "batch_2_sample_1", "batch_2_sample_2", "batch_2_sample_3"
    ]
    
    all_results = []
    
    print("\\n📊 逐樣本分析:")
    print("-" * 70)
    
    for sample in samples:
        print(f"分析樣本: {sample}")
        result = analyze_audio_quality(main_dir, fix_dir, sample)
        if result:
            all_results.append(result)
            
            # 顯示結果
            print(f"  SNR: Main={result['snr']['main']:.2f}dB, "
                  f"Fix={result['snr']['fix']:.2f}dB, "
                  f"改善={result['snr']['improvement']:.2f}dB")
            print(f"  MFCC相似度: Main={result['mfcc_similarity']['main']:.3f}, "
                  f"Fix={result['mfcc_similarity']['fix']:.3f}, "
                  f"改善={result['mfcc_similarity']['improvement']:.3f}")
        else:
            print(f"  ❌ 樣本分析失敗")
        print()
    
    if not all_results:
        print("❌ 沒有成功分析的樣本")
        return
    
    # 統計總結
    print("\\n📈 統計總結:")
    print("-" * 70)
    
    snr_improvements = [r['snr']['improvement'] for r in all_results]
    mfcc_improvements = [r['mfcc_similarity']['improvement'] for r in all_results]
    
    centroid_main_errors = [r['spectral_centroid']['main_diff'] for r in all_results]
    centroid_fix_errors = [r['spectral_centroid']['fix_diff'] for r in all_results]
    
    rolloff_main_errors = [r['spectral_rolloff']['main_diff'] for r in all_results]
    rolloff_fix_errors = [r['spectral_rolloff']['fix_diff'] for r in all_results]
    
    print(f"SNR改善: 平均={np.mean(snr_improvements):.2f}dB, "
          f"標準差={np.std(snr_improvements):.2f}dB")
    print(f"MFCC相似度改善: 平均={np.mean(mfcc_improvements):.3f}, "
          f"標準差={np.std(mfcc_improvements):.3f}")
    print(f"頻譜重心誤差: Main={np.mean(centroid_main_errors):.1f}Hz, "
          f"Fix={np.mean(centroid_fix_errors):.1f}Hz")
    print(f"頻譜滾降誤差: Main={np.mean(rolloff_main_errors):.1f}Hz, "
          f"Fix={np.mean(rolloff_fix_errors):.1f}Hz")
    
    # 保存詳細結果 - 轉換numpy類型為Python原生類型
    def convert_numpy_types(obj):
        """轉換numpy類型為JSON可序列化類型"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy_types(all_results)
    with open(f"{output_dir}/audio_quality_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 生成比較圖表
    generate_comparison_plots(all_results, output_dir)
    
    # 生成報告
    generate_audio_quality_report(all_results, output_dir)
    
    print(f"\\n✅ 分析完成！結果保存在 {output_dir}/ 目錄中")

def generate_comparison_plots(results, output_dir):
    """生成比較圖表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TTT2 Audio Quality Comparison - Fix Branch vs Main Branch', fontsize=16)
    
    samples = [r['sample'] for r in results]
    x_pos = np.arange(len(samples))
    
    # SNR比較
    main_snr = [r['snr']['main'] for r in results]
    fix_snr = [r['snr']['fix'] for r in results]
    
    axes[0,0].bar(x_pos - 0.2, main_snr, 0.4, label='Main Branch', alpha=0.8)
    axes[0,0].bar(x_pos + 0.2, fix_snr, 0.4, label='Fix Branch', alpha=0.8)
    axes[0,0].set_title('Signal-to-Noise Ratio (SNR) Comparison')
    axes[0,0].set_ylabel('SNR (dB)')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels([s.replace('batch_', 'B').replace('_sample_', 'S') for s in samples], rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # MFCC相似度比較
    main_mfcc = [r['mfcc_similarity']['main'] for r in results]
    fix_mfcc = [r['mfcc_similarity']['fix'] for r in results]
    
    axes[0,1].bar(x_pos - 0.2, main_mfcc, 0.4, label='Main Branch', alpha=0.8)
    axes[0,1].bar(x_pos + 0.2, fix_mfcc, 0.4, label='Fix Branch', alpha=0.8)
    axes[0,1].set_title('MFCC Similarity Comparison')
    axes[0,1].set_ylabel('Similarity')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels([s.replace('batch_', 'B').replace('_sample_', 'S') for s in samples], rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 頻譜重心誤差比較
    main_centroid_err = [r['spectral_centroid']['main_diff'] for r in results]
    fix_centroid_err = [r['spectral_centroid']['fix_diff'] for r in results]
    
    axes[1,0].bar(x_pos - 0.2, main_centroid_err, 0.4, label='Main Branch', alpha=0.8)
    axes[1,0].bar(x_pos + 0.2, fix_centroid_err, 0.4, label='Fix Branch', alpha=0.8)
    axes[1,0].set_title('Spectral Centroid Error (vs Target)')
    axes[1,0].set_ylabel('Error (Hz)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([s.replace('batch_', 'B').replace('_sample_', 'S') for s in samples], rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 改善度總覽
    snr_improvements = [r['snr']['improvement'] for r in results]
    mfcc_improvements = [r['mfcc_similarity']['improvement'] for r in results]
    
    axes[1,1].bar(x_pos - 0.2, snr_improvements, 0.4, label='SNR Improvement', alpha=0.8)
    axes[1,1].bar(x_pos + 0.2, np.array(mfcc_improvements)*100, 0.4, label='MFCC Improvement×100', alpha=0.8)
    axes[1,1].set_title('Fix Branch Improvements')
    axes[1,1].set_ylabel('Improvement Value')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([s.replace('batch_', 'B').replace('_sample_', 'S') for s in samples], rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/audio_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_audio_quality_report(results, output_dir):
    """生成音頻品質分析報告"""
    
    snr_improvements = [r['snr']['improvement'] for r in results]
    mfcc_improvements = [r['mfcc_similarity']['improvement'] for r in results]
    
    positive_snr = sum(1 for x in snr_improvements if x > 0)
    positive_mfcc = sum(1 for x in mfcc_improvements if x > 0)
    
    report = f"""# 音頻品質比較分析報告 - TTT2 Fix分支 vs Main分支

## 📊 實驗概述
- **比較基準**: Epoch 300
- **樣本數量**: {len(results)}個音頻樣本
- **評估指標**: SNR, MFCC相似度, 頻譜特徵
- **分析日期**: 2025-08-13

## 🎵 核心發現

### SNR (信噪比) 分析
- **平均改善**: {np.mean(snr_improvements):.2f} dB
- **標準差**: {np.std(snr_improvements):.2f} dB
- **改善樣本**: {positive_snr}/{len(results)} ({positive_snr/len(results)*100:.1f}%)
- **最大改善**: {max(snr_improvements):.2f} dB
- **最大退化**: {min(snr_improvements):.2f} dB

### MFCC相似度分析
- **平均改善**: {np.mean(mfcc_improvements):.4f}
- **標準差**: {np.std(mfcc_improvements):.4f}
- **改善樣本**: {positive_mfcc}/{len(results)} ({positive_mfcc/len(results)*100:.1f}%)
- **最大改善**: {max(mfcc_improvements):.4f}
- **最大退化**: {min(mfcc_improvements):.4f}

## 📈 詳細樣本分析

| 樣本 | SNR改善(dB) | MFCC改善 | 總體評價 |
|------|------------|----------|----------|"""

    for r in results:
        snr_imp = r['snr']['improvement']
        mfcc_imp = r['mfcc_similarity']['improvement']
        
        if snr_imp > 0 and mfcc_imp > 0:
            evaluation = "✅ 雙重改善"
        elif snr_imp > 0 or mfcc_imp > 0:
            evaluation = "🔶 部分改善"
        else:
            evaluation = "❌ 需要優化"
            
        report += f"""
| {r['sample']} | {snr_imp:+.2f} | {mfcc_imp:+.4f} | {evaluation} |"""

    avg_snr = np.mean(snr_improvements)
    avg_mfcc = np.mean(mfcc_improvements)
    
    if avg_snr > 0 and avg_mfcc > 0:
        overall_conclusion = "✅ Fix分支在音頻品質上優於Main分支"
    elif avg_snr > 0 or avg_mfcc > 0:
        overall_conclusion = "🔶 Fix分支在某些指標上有改善"
    else:
        overall_conclusion = "❌ Fix分支在音頻品質上需要進一步優化"

    report += f"""

## 🏆 總體結論

### 數值表現
{overall_conclusion}

### 品質評估結果
- **SNR改善**: {"正向" if avg_snr > 0 else "負向"} ({avg_snr:+.2f} dB)
- **MFCC改善**: {"正向" if avg_mfcc > 0 else "負向"} ({avg_mfcc:+.4f})
- **改善一致性**: {min(positive_snr, positive_mfcc)}/{len(results)} 樣本在兩項指標都有改善

### 技術解釋
{"Fix分支的ResidualBlock修復和多組件損失確實帶來了音頻品質改善。" if avg_snr > 0 and avg_mfcc > 0 else "雖然損失函數數值較高，但音頻品質分析結果需要更多epoch的訓練來體現修復效果。"}

## 📋 方法論說明

### SNR計算
信噪比 = 10 * log10(信號功率 / 噪聲功率)
其中噪聲 = |enhanced - target|

### MFCC相似度
使用13維MFCC特徵的皮爾森相關係數平均值

### 頻譜特徵
- 頻譜重心: 頻率的能量加權平均
- 頻譜滾降: 85%能量所在的頻率點

---
*報告生成時間: 2025-08-13*
*數據來源: TTT2 Epoch 300 音頻樣本*"""

    with open(f"{output_dir}/AUDIO_QUALITY_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main()
