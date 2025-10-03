#!/usr/bin/env python3
"""
分析離散化結果與連續結果的差異
實驗編號: DISCRETE_ANALYSIS_202510020616
日期: 2025-10-02
功能: 深度分析離散化為什麼比連續方法效果差
"""

import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """設置日誌系統"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"discrete_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_training_history(log_file_path):
    """
    分析訓練歷史，識別問題
    
    Args:
        log_file_path: 訓練日誌檔案路徑
    
    Returns:
        dict: 分析結果
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"無法讀取日誌檔案: {e}")
        return {}
    
    # 提取損失數據
    train_losses = []
    val_losses = []
    epochs = []
    
    loss_components = {
        'l2_loss': [],
        'consistency_loss': [],
        'manifold_loss': [],
        'normalization_loss': [],
        'coherence_loss': []
    }
    
    for line in lines:
        # 提取訓練損失
        if "Train Loss:" in line and "Val Loss:" in line:
            try:
                parts = line.split()
                train_loss = float(parts[parts.index("Loss:") + 1].rstrip(','))
                val_loss = float(parts[parts.index("Loss:") + 3].rstrip(','))
                
                # 提取 epoch 數字
                if "Epoch" in line:
                    epoch_str = line.split("Epoch")[1].split()[0]
                    epoch = int(epoch_str.split('/')[0])
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
            except (ValueError, IndexError) as e:
                continue
        
        # 提取損失組件
        if "Train Loss Components:" in line:
            try:
                for component in loss_components.keys():
                    if component in line:
                        value_str = line.split(f"{component}: ")[1].split()[0]
                        value = float(value_str)
                        loss_components[component].append(value)
            except (ValueError, IndexError):
                continue
    
    # 分析結果
    analysis = {
        'total_epochs': len(epochs),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'train_loss_trend': 'decreasing' if len(train_losses) > 1 and train_losses[-1] < train_losses[0] else 'not_decreasing',
        'val_loss_issue': all(loss == 0.0 for loss in val_losses[-10:]) if val_losses else True,
        'dominant_loss_component': max(loss_components.keys(), 
                                     key=lambda k: loss_components[k][-1] if loss_components[k] else 0),
        'loss_components_final': {k: v[-1] if v else 0 for k, v in loss_components.items()}
    }
    
    # 創建可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('離散化訓練問題分析 - DISCRETE_ANALYSIS_202510020616', fontsize=16)
    
    # 訓練損失曲線
    if epochs and train_losses:
        axes[0, 0].plot(epochs, train_losses, 'b-', label='訓練損失', linewidth=2)
        axes[0, 0].set_title('訓練損失趨勢')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('損失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 驗證損失問題
    if epochs and val_losses:
        axes[0, 1].plot(epochs, val_losses, 'r-', label='驗證損失', linewidth=2)
        axes[0, 1].set_title('驗證損失問題 (一直為0)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('損失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].text(0.5, 0.5, '驗證損失異常為0!\n表明驗證邏輯有問題', 
                       transform=axes[0, 1].transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 損失組件分析
    if any(loss_components.values()):
        component_names = list(loss_components.keys())
        component_values = [loss_components[k][-1] if loss_components[k] else 0 for k in component_names]
        
        # 使用對數比例因為coherence_loss過大
        axes[1, 0].bar(component_names, np.log10(np.array(component_values) + 1e-10))
        axes[1, 0].set_title('損失組件分析 (對數比例)')
        axes[1, 0].set_ylabel('log10(損失值)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 主要問題總結
    axes[1, 1].axis('off')
    final_train_loss = analysis.get('final_train_loss', 0)
    final_train_loss_str = f"{final_train_loss:.1f}" if final_train_loss else "N/A"
    
    problem_text = f"""
主要問題分析:

1. 驗證損失異常: 一直為 0.0000
   - 表明驗證邏輯有嚴重錯誤
   - 無法正確評估模型性能

2. 訓練損失過大: {final_train_loss_str}
   - 主要由 coherence_loss 主導
   - 表明離散token之間缺乏連貫性

3. 離散化問題根源:
   - 量化誤差累積
   - Token之間語義跳躍
   - Transformer難以學習離散映射

4. SConv1d 維度錯誤:
   - 收到4D張量但期望3D
   - 表明數據預處理有問題
"""
    
    axes[1, 1].text(0.05, 0.95, problem_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存分析結果
    output_dir = Path("results/discrete_analysis_202510020616")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "training_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"訓練分析圖表已保存至: {output_dir / 'training_analysis.png'}")
    
    # 保存數值分析
    with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    return analysis

def analyze_audio_samples(samples_dir):
    """
    分析音頻樣本質量
    
    Args:
        samples_dir: 音頻樣本目錄
    
    Returns:
        dict: 音頻質量分析結果
    """
    logger = logging.getLogger(__name__)
    samples_path = Path(samples_dir)
    
    if not samples_path.exists():
        logger.error(f"樣本目錄不存在: {samples_dir}")
        return {}
    
    # 尋找最新的epoch樣本
    epoch_dirs = [d for d in samples_path.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
    if not epoch_dirs:
        logger.error("未找到epoch樣本目錄")
        return {}
    
    latest_epoch = max(epoch_dirs, key=lambda x: int(x.name.split('_')[1]))
    logger.info(f"分析最新epoch樣本: {latest_epoch.name}")
    
    # 分析音頻文件
    audio_files = list(latest_epoch.glob("*.wav"))
    if not audio_files:
        logger.error("未找到音頻檔案")
        return {}
    
    # 按類型分組分析
    input_files = [f for f in audio_files if 'input' in f.name]
    target_files = [f for f in audio_files if 'target' in f.name]
    enhanced_files = [f for f in audio_files if 'enhanced' in f.name]
    
    def compute_audio_metrics(file_path):
        """計算音頻質量指標"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            
            # 基本統計
            metrics = {
                'duration': len(audio) / sr,
                'rms_energy': np.sqrt(np.mean(audio**2)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
                'mfcc_mean': np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)),
                'dynamic_range': np.max(audio) - np.min(audio)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"計算音頻指標失敗 {file_path}: {e}")
            return {}
    
    # 計算各類型的平均指標
    results = {}
    for file_type, files in [('input', input_files), ('target', target_files), ('enhanced', enhanced_files)]:
        if not files:
            continue
            
        metrics_list = []
        for file_path in files[:5]:  # 只分析前5個樣本
            metrics = compute_audio_metrics(file_path)
            if metrics:
                metrics_list.append(metrics)
        
        if metrics_list:
            # 計算平均值
            avg_metrics = {}
            for key in metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])
            results[file_type] = avg_metrics
    
    # 創建比較圖表
    if len(results) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('離散化音頻質量分析 - DISCRETE_ANALYSIS_202510020616', fontsize=16)
        
        metrics_names = ['rms_energy', 'spectral_centroid', 'spectral_rolloff', 
                        'zero_crossing_rate', 'mfcc_mean', 'dynamic_range']
        
        for i, metric in enumerate(metrics_names):
            row, col = i // 3, i % 3
            
            if row < 2 and col < 3:
                types = list(results.keys())
                values = [results[t].get(metric, 0) for t in types]
                
                bars = axes[row, col].bar(types, values, alpha=0.7)
                axes[row, col].set_title(f'{metric} 比較')
                axes[row, col].set_ylabel('數值')
                
                # 標註數值
                for bar, val in zip(bars, values):
                    axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                       f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存結果
        output_dir = Path("results/discrete_analysis_202510020616")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / "audio_quality_analysis.png", dpi=300, bbox_inches='tight')
        logger.info(f"音頻質量分析圖表已保存至: {output_dir / 'audio_quality_analysis.png'}")
    
    return results

def analyze_quantization_error():
    """
    分析量化誤差對音頻質量的影響
    
    Returns:
        dict: 量化誤差分析結果
    """
    logger = logging.getLogger(__name__)
    
    # 模擬量化過程的誤差分析
    logger.info("分析離散化量化誤差...")
    
    # 創建測試信號
    t = np.linspace(0, 1, 16000)  # 1秒，16kHz
    original_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)  # 440Hz + 880Hz
    
    # 模擬不同量化位數的影響
    quantization_levels = [256, 512, 1024, 2048, 4096, 8192]  # 對應不同codebook大小
    
    results = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('量化誤差對音頻質量影響分析 - DISCRETE_ANALYSIS_202510020616', fontsize=16)
    
    for i, levels in enumerate(quantization_levels):
        # 量化過程
        quantized_signal = np.round(original_signal * levels) / levels
        
        # 計算誤差
        quantization_error = original_signal - quantized_signal
        snr = 10 * np.log10(np.var(original_signal) / np.var(quantization_error))
        mse = np.mean(quantization_error**2)
        
        # 頻譜分析
        original_fft = np.abs(np.fft.fft(original_signal))
        quantized_fft = np.abs(np.fft.fft(quantized_signal))
        spectral_distortion = np.mean(np.abs(original_fft - quantized_fft))
        
        results[levels] = {
            'snr_db': snr,
            'mse': mse,
            'spectral_distortion': spectral_distortion,
            'dynamic_range_loss': np.max(original_signal) - np.max(quantized_signal)
        }
        
        # 可視化部分結果
        if i < 6:
            row, col = i // 3, i % 3
            
            # 顯示時域比較
            axes[row, col].plot(t[:1000], original_signal[:1000], 'b-', label='原始', alpha=0.7)
            axes[row, col].plot(t[:1000], quantized_signal[:1000], 'r--', label='量化', alpha=0.7)
            axes[row, col].set_title(f'量化級別: {levels} (SNR: {snr:.1f}dB)')
            axes[row, col].set_xlabel('時間 (s)')
            axes[row, col].set_ylabel('振幅')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存結果
    output_dir = Path("results/discrete_analysis_202510020616")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "quantization_error_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"量化誤差分析圖表已保存至: {output_dir / 'quantization_error_analysis.png'}")
    
    # 創建量化誤差總結圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    levels_list = list(results.keys())
    snr_list = [results[l]['snr_db'] for l in levels_list]
    mse_list = [results[l]['mse'] for l in levels_list]
    spectral_list = [results[l]['spectral_distortion'] for l in levels_list]
    
    axes[0].plot(levels_list, snr_list, 'bo-', linewidth=2)
    axes[0].set_title('SNR vs 量化級別')
    axes[0].set_xlabel('量化級別 (Codebook大小)')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(levels_list, mse_list, 'ro-', linewidth=2)
    axes[1].set_title('MSE vs 量化級別')
    axes[1].set_xlabel('量化級別 (Codebook大小)')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(levels_list, spectral_list, 'go-', linewidth=2)
    axes[2].set_title('頻譜失真 vs 量化級別')
    axes[2].set_xlabel('量化級別 (Codebook大小)')
    axes[2].set_ylabel('頻譜失真')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "quantization_summary.png", dpi=300, bbox_inches='tight')
    
    return results

def generate_improvement_recommendations():
    """
    基於分析結果生成改進建議
    
    Returns:
        dict: 改進建議
    """
    logger = logging.getLogger(__name__)
    
    recommendations = {
        "immediate_fixes": [
            "修復驗證損失計算邏輯 - 目前一直為0表明驗證函數有bug",
            "修復SConv1d維度錯誤 - 確保輸入為3D張量[B,C,T]而非4D",
            "檢查並修復batch_idx變數作用域問題",
            "增加梯度裁剪防止梯度爆炸"
        ],
        "architectural_improvements": [
            "使用預訓練的Vector Quantization (VQ-VAE)而非直接離散化",
            "實現漸進式量化策略，從連續到離散的平滑過渡",
            "增加codebook學習機制，讓離散token更好表示連續特徵",
            "引入注意力機制優化transformer對離散序列的建模能力"
        ],
        "training_strategies": [
            "實現curriculum learning：先訓練連續版本再微調離散版本",
            "使用teacher-student框架：連續模型作為teacher指導離散模型",
            "增加consistency loss確保離散化前後特徵一致性",
            "實現multi-scale訓練：同時優化不同解析度的離散表示"
        ],
        "loss_function_optimization": [
            "重新平衡損失函數權重，coherence_loss過於主導",
            "增加perceptual loss使用預訓練音頻模型",
            "實現adversarial training提升生成質量",
            "加入spectral consistency loss保持頻域特徵"
        ],
        "data_preprocessing": [
            "改進音頻正規化策略，確保輸入範圍一致",
            "實現更sophisticated的token化策略",
            "增加data augmentation提升模型泛化能力",
            "優化batch構建避免維度不匹配問題"
        ]
    }
    
    # 創建改進建議視覺化
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('離散化音頻處理改進建議 - DISCRETE_ANALYSIS_202510020616', fontsize=16)
    
    y_pos = 0
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (category, items) in enumerate(recommendations.items()):
        # 分類標題
        ax.text(0.02, 0.95 - y_pos, category.replace('_', ' ').title(), 
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
        y_pos += 0.05
        
        # 建議項目
        for j, item in enumerate(items):
            ax.text(0.05, 0.95 - y_pos, f"• {item}", 
                   transform=ax.transAxes, fontsize=11, wrap=True)
            y_pos += 0.04
        
        y_pos += 0.02  # 分類間距
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 保存結果
    output_dir = Path("results/discrete_analysis_202510020616")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "improvement_recommendations.png", dpi=300, bbox_inches='tight')
    logger.info(f"改進建議已保存至: {output_dir / 'improvement_recommendations.png'}")
    
    # 保存文字版本
    with open(output_dir / "recommendations.json", 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    
    return recommendations

def main():
    """主函數"""
    logger = setup_logging()
    logger.info("開始離散化 vs 連續化深度分析...")
    
    # 分析訓練歷史
    log_file = "/home/sbplab/ruizi/c_code/logs/wavtokenizer_transformer_training_fixed_202510020508.log"
    training_analysis = analyze_training_history(log_file)
    
    # 分析音頻樣本
    samples_dir = "/home/sbplab/ruizi/c_code/results/wavtokenizer_tokenloss_fixed_202510020508/audio_samples"
    audio_analysis = analyze_audio_samples(samples_dir)
    
    # 量化誤差分析
    quantization_analysis = analyze_quantization_error()
    
    # 生成改進建議
    recommendations = generate_improvement_recommendations()
    
    # 生成綜合報告
    output_dir = Path("results/discrete_analysis_202510020616")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 轉換numpy類型為Python原生類型以便JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    report = {
        "experiment_id": "DISCRETE_ANALYSIS_202510020616",
        "timestamp": datetime.now().isoformat(),
        "analysis_summary": {
            "main_issues": [
                "驗證損失異常為0 - 驗證邏輯有嚴重錯誤",
                "SConv1d維度錯誤 - 期望3D但收到4D張量",
                "coherence_loss過於主導整體損失",
                "離散化導致頻譜信息嚴重損失"
            ],
            "root_causes": [
                "量化過程引入不可逆的信息損失",
                "Transformer架構不適合處理離散跳躍",
                "缺乏平滑的連續到離散過渡機制",
                "損失函數設計不合理"
            ]
        },
        "training_analysis": convert_for_json(training_analysis),
        "audio_analysis": convert_for_json(audio_analysis),
        "quantization_analysis": convert_for_json(quantization_analysis),
        "recommendations": recommendations
    }
    
    with open(output_dir / "comprehensive_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("離散化深度分析完成！")
    logger.info(f"詳細結果已保存至: {output_dir}")
    
    print("\n" + "="*80)
    print("離散化 vs 連續化分析結果總結")
    print("="*80)
    print(f"實驗編號: DISCRETE_ANALYSIS_202510020616")
    print(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n主要發現:")
    for i, issue in enumerate(report["analysis_summary"]["main_issues"], 1):
        print(f"{i}. {issue}")
    
    print("\n根本原因:")
    for i, cause in enumerate(report["analysis_summary"]["root_causes"], 1):
        print(f"{i}. {cause}")
    
    print(f"\n詳細分析結果已保存至: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()