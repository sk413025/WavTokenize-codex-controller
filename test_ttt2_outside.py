#!/usr/bin/env python3
"""
TTT2模型Outside音檔測試腳本
測試訓練後的TTT2模型在outside音檔上的表現
"""

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import argparse
import librosa
from scipy.stats import pearsonr
import soundfile as sf
import sys

# 導入TTT2相關模組
sys.path.append('.')
from ttt2 import EnhancedWavTokenizer

def setup_device():
    """設置計算設備"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device

def load_trained_model(checkpoint_path, device):
    """載入訓練後的TTT2模型"""
    print(f"載入模型checkpoint: {checkpoint_path}")
    
    try:
        # 載入checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # TTT2使用的配置路徑
        config_path = "config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "models/wavtokenizer_large_speech_320_24k.ckpt"
        
        # 創建EnhancedWavTokenizer模型實例
        model = EnhancedWavTokenizer(config_path, model_path)
        
        # 載入state dict - 支援多種格式
        if 'model_state_dict' in checkpoint:
            # TTT2 best_model.pth 格式
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"載入best_model.pth格式，epoch: {epoch}")
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint 格式
            state_dict = checkpoint['state_dict']
            # 可能需要移除 'model.' 前綴
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"載入Lightning checkpoint格式，epoch: {epoch}")
        else:
            # 如果checkpoint就是state_dict
            model.load_state_dict(checkpoint)
            epoch = 'unknown'
            print(f"載入純state_dict格式")
        
        model.to(device)
        model.eval()
        
        print(f"模型載入成功")
        return model, checkpoint
        
    except Exception as e:
        print(f"模型載入失敗: {e}")
        print("嘗試創建基礎模型...")
        
        # 如果載入失敗，創建基礎模型
        config_path = "config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "models/wavtokenizer_large_speech_320_24k.ckpt"
        
        model = EnhancedWavTokenizer(config_path, model_path)
        model.to(device)
        model.eval()
        
        return model, None

def load_outside_audio_files(outside_dir, max_files=None):
    """載入outside音檔"""
    audio_files = []
    supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
    
    for ext in supported_formats:
        audio_files.extend(Path(outside_dir).glob(f"**/*{ext}"))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"找到 {len(audio_files)} 個outside音檔")
    return audio_files

def preprocess_audio(audio_path, target_sr=16000, target_length=None):
    """預處理音檔"""
    try:
        # 載入音檔
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # 如果指定長度，進行裁剪或填充
        if target_length:
            if len(audio) > target_length:
                # 隨機裁剪
                start_idx = np.random.randint(0, len(audio) - target_length + 1)
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # 重複填充
                repeat_times = target_length // len(audio) + 1
                audio = np.tile(audio, repeat_times)[:target_length]
        
        return torch.tensor(audio, dtype=torch.float32)
    
    except Exception as e:
        print(f"載入音檔失敗 {audio_path}: {e}")
        return None

def test_model_on_audio(model, audio_tensor, device):
    """測試模型在單個音檔上的表現"""
    model.eval()
    
    with torch.no_grad():
        try:
            # 準備輸入 - TTT2模型需要適當的維度
            if audio_tensor.dim() == 1:
                audio_batch = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, length]
            elif audio_tensor.dim() == 2:
                audio_batch = audio_tensor.unsqueeze(0).to(device)  # [1, channels, length]
            else:
                audio_batch = audio_tensor.to(device)
            
            # TTT2模型的forward返回多個值
            results = model(audio_batch)
            
            # 解包TTT2的返回值
            if isinstance(results, tuple) and len(results) >= 1:
                enhanced = results[0]  # 第一個返回值通常是enhanced音頻
                
                # 移除batch dimension並轉到CPU
                if enhanced.dim() > 2:
                    enhanced = enhanced.squeeze(0)  # 移除batch維度
                if enhanced.dim() > 1:
                    enhanced = enhanced.mean(dim=0)  # 如果還有channel維度，取平均
                
                enhanced = enhanced.cpu()
                
                return {
                    'enhanced': enhanced,
                    'original': audio_tensor,
                    'features': results[1:] if len(results) > 1 else {},
                    'success': True
                }
            else:
                print(f"模型返回意外格式: {type(results)}")
                return {
                    'enhanced': None,
                    'original': audio_tensor,
                    'features': {},
                    'success': False
                }
            
        except Exception as e:
            print(f"模型推理失敗: {e}")
            return {
                'enhanced': None,
                'original': audio_tensor,
                'features': {},
                'success': False
            }

def compute_audio_metrics(original, enhanced):
    """計算音檔品質指標"""
    # 確保長度一致
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]
    
    # SNR
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - enhanced) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # 皮爾森相關係數
    correlation = torch.corrcoef(torch.stack([original, enhanced]))[0, 1]
    
    # RMS差異
    rms_diff = torch.sqrt(torch.mean((original - enhanced) ** 2))
    
    # 頻譜距離
    original_fft = torch.fft.fft(original)
    enhanced_fft = torch.fft.fft(enhanced)
    spectral_distance = torch.mean(torch.abs(original_fft - enhanced_fft))
    
    return {
        'snr': snr.item(),
        'correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
        'rms_diff': rms_diff.item(),
        'spectral_distance': spectral_distance.item()
    }

def save_audio_comparison(original, enhanced, output_path, sr=16000):
    """保存音檔比較"""
    # 保存原始音檔
    original_path = output_path.parent / f"{output_path.stem}_original.wav"
    sf.write(str(original_path), original.numpy(), sr)
    
    # 保存增強音檔
    enhanced_path = output_path.parent / f"{output_path.stem}_enhanced.wav"
    sf.write(str(enhanced_path), enhanced.numpy(), sr)
    
    return original_path, enhanced_path

def create_comparison_plot(original, enhanced, metrics, output_path):
    """創建比較圖表"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    time_axis = np.arange(len(original)) / 16000
    
    # 時域波形比較
    axes[0].plot(time_axis, original, label='Original', alpha=0.7)
    axes[0].plot(time_axis, enhanced, label='Enhanced', alpha=0.7)
    axes[0].set_title(f'Waveform Comparison (SNR: {metrics["snr"]:.2f}dB, Corr: {metrics["correlation"]:.3f})')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 頻譜比較
    original_fft = np.abs(np.fft.fft(original))[:len(original)//2]
    enhanced_fft = np.abs(np.fft.fft(enhanced))[:len(enhanced)//2]
    freq_axis = np.fft.fftfreq(len(original), 1/16000)[:len(original)//2]
    
    axes[1].semilogy(freq_axis, original_fft, label='Original', alpha=0.7)
    axes[1].semilogy(freq_axis, enhanced_fft, label='Enhanced', alpha=0.7)
    axes[1].set_title(f'Frequency Spectrum (Spectral Distance: {metrics["spectral_distance"]:.2f})')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 差異信號
    diff_signal = original - enhanced
    axes[2].plot(time_axis, diff_signal, color='red', alpha=0.7)
    axes[2].set_title(f'Difference Signal (RMS: {metrics["rms_diff"]:.4f})')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='TTT2 Outside音檔測試')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路徑')
    parser.add_argument('--outside_dir', type=str, default='./1n', help='outside音檔目錄')
    parser.add_argument('--output_dir', type=str, default='ttt2_outside_test_results', help='輸出目錄')
    parser.add_argument('--max_files', type=int, default=10, help='最大測試檔案數')
    parser.add_argument('--audio_length', type=int, default=32000, help='音檔長度(樣本數)')
    args = parser.parse_args()
    
    # 設置輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎵 TTT2 Outside音檔測試開始")
    print("="*60)
    
    # 設置設備
    device = setup_device()
    
    # 載入模型
    try:
        model, checkpoint_info = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    # 載入outside音檔
    outside_files = load_outside_audio_files(args.outside_dir, args.max_files)
    if not outside_files:
        print(f"❌ 在 {args.outside_dir} 中未找到音檔")
        return
    
    # 測試結果收集
    test_results = []
    successful_tests = 0
    
    print(f"\\n🔍 開始測試 {len(outside_files)} 個音檔...")
    print("-" * 60)
    
    for i, audio_file in enumerate(outside_files):
        print(f"測試 [{i+1}/{len(outside_files)}]: {audio_file.name}")
        
        # 預處理音檔
        audio_tensor = preprocess_audio(audio_file, target_length=args.audio_length)
        if audio_tensor is None:
            continue
        
        # 測試模型
        result = test_model_on_audio(model, audio_tensor, device)
        
        if result['success']:
            # 計算指標
            metrics = compute_audio_metrics(result['original'], result['enhanced'])
            
            # 保存音檔
            audio_output_path = output_dir / f"test_{i+1:03d}_{audio_file.stem}"
            audio_output_path.mkdir(exist_ok=True)
            
            original_path, enhanced_path = save_audio_comparison(
                result['original'], result['enhanced'], 
                audio_output_path / "comparison.wav"
            )
            
            # 創建比較圖表
            plot_path = audio_output_path / "comparison_plot.png"
            create_comparison_plot(
                result['original'].numpy(), result['enhanced'].numpy(),
                metrics, plot_path
            )
            
            # 記錄結果
            test_result = {
                'file_name': audio_file.name,
                'file_path': str(audio_file),
                'metrics': metrics,
                'output_dir': str(audio_output_path),
                'original_path': str(original_path),
                'enhanced_path': str(enhanced_path),
                'plot_path': str(plot_path)
            }
            test_results.append(test_result)
            successful_tests += 1
            
            # 顯示結果
            print(f"  ✅ SNR: {metrics['snr']:.2f}dB, "
                  f"Corr: {metrics['correlation']:.3f}, "
                  f"RMS: {metrics['rms_diff']:.4f}")
        else:
            print(f"  ❌ 測試失敗")
        
        print()
    
    # 統計分析
    if test_results:
        print("\\n📊 統計分析:")
        print("-" * 60)
        
        snr_values = [r['metrics']['snr'] for r in test_results]
        corr_values = [r['metrics']['correlation'] for r in test_results]
        rms_values = [r['metrics']['rms_diff'] for r in test_results]
        
        print(f"成功測試: {successful_tests}/{len(outside_files)} 檔案")
        print(f"SNR: 平均={np.mean(snr_values):.2f}dB, "
              f"標準差={np.std(snr_values):.2f}dB, "
              f"範圍=[{np.min(snr_values):.2f}, {np.max(snr_values):.2f}]dB")
        print(f"相關係數: 平均={np.mean(corr_values):.3f}, "
              f"標準差={np.std(corr_values):.3f}, "
              f"範圍=[{np.min(corr_values):.3f}, {np.max(corr_values):.3f}]")
        print(f"RMS差異: 平均={np.mean(rms_values):.4f}, "
              f"標準差={np.std(rms_values):.4f}")
        
        # 生成總結報告
        report = {
            'test_info': {
                'timestamp': timestamp,
                'checkpoint_path': args.checkpoint,
                'outside_dir': args.outside_dir,
                'total_files': len(outside_files),
                'successful_tests': successful_tests,
                'checkpoint_info': checkpoint_info.get('epoch', 'unknown')
            },
            'statistics': {
                'snr': {
                    'mean': float(np.mean(snr_values)),
                    'std': float(np.std(snr_values)),
                    'min': float(np.min(snr_values)),
                    'max': float(np.max(snr_values))
                },
                'correlation': {
                    'mean': float(np.mean(corr_values)),
                    'std': float(np.std(corr_values)),
                    'min': float(np.min(corr_values)),
                    'max': float(np.max(corr_values))
                },
                'rms_diff': {
                    'mean': float(np.mean(rms_values)),
                    'std': float(np.std(rms_values)),
                    'min': float(np.min(rms_values)),
                    'max': float(np.max(rms_values))
                }
            },
            'detailed_results': test_results
        }
        
        # 保存報告
        report_path = output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown報告
        generate_markdown_report(report, output_dir / "TEST_REPORT.md")
        
        print(f"\\n✅ 測試完成！結果保存在: {output_dir}")
        print(f"📋 詳細報告: {output_dir / 'TEST_REPORT.md'}")
    else:
        print("\\n❌ 沒有成功的測試結果")

def generate_markdown_report(report, output_path):
    """生成Markdown測試報告"""
    timestamp = report['test_info']['timestamp']
    
    markdown_content = f"""# TTT2 Outside音檔測試報告

## 📊 測試概述
- **測試時間**: {timestamp}
- **模型Checkpoint**: `{report['test_info']['checkpoint_path']}`
- **Outside目錄**: `{report['test_info']['outside_dir']}`
- **測試檔案數**: {report['test_info']['total_files']}
- **成功測試**: {report['test_info']['successful_tests']}
- **成功率**: {report['test_info']['successful_tests']/report['test_info']['total_files']*100:.1f}%
- **訓練Epoch**: {report['test_info']['checkpoint_info']}

## 🎵 音頻品質統計

### SNR (信噪比)
- **平均值**: {report['statistics']['snr']['mean']:.2f} dB
- **標準差**: {report['statistics']['snr']['std']:.2f} dB  
- **範圍**: [{report['statistics']['snr']['min']:.2f}, {report['statistics']['snr']['max']:.2f}] dB

### 相關係數
- **平均值**: {report['statistics']['correlation']['mean']:.3f}
- **標準差**: {report['statistics']['correlation']['std']:.3f}
- **範圍**: [{report['statistics']['correlation']['min']:.3f}, {report['statistics']['correlation']['max']:.3f}]

### RMS差異
- **平均值**: {report['statistics']['rms_diff']['mean']:.4f}
- **標準差**: {report['statistics']['rms_diff']['std']:.4f}
- **範圍**: [{report['statistics']['rms_diff']['min']:.4f}, {report['statistics']['rms_diff']['max']:.4f}]

## 📈 詳細測試結果

| 檔案名 | SNR (dB) | 相關係數 | RMS差異 | 狀態 |
|--------|----------|----------|---------|------|"""

    for result in report['detailed_results']:
        metrics = result['metrics']
        markdown_content += f"""
| {result['file_name']} | {metrics['snr']:.2f} | {metrics['correlation']:.3f} | {metrics['rms_diff']:.4f} | ✅ |"""

    markdown_content += f"""

## 📋 技術說明

### 測試流程
1. 載入訓練後的TTT2模型checkpoint
2. 預處理outside音檔(重採樣至24kHz，長度標準化)
3. 模型推理生成enhanced音檔
4. 計算音頻品質指標(SNR、相關係數、RMS差異、頻譜距離)
5. 生成波形和頻譜比較圖表
6. 保存原始和enhanced音檔供後續分析

### 指標說明
- **SNR**: 信噪比，衡量增強後音檔相對於原始音檔的信號品質
- **相關係數**: 原始和增強音檔的線性相關程度
- **RMS差異**: 原始和增強音檔的均方根差異
- **頻譜距離**: 頻域的歐幾里得距離

### 檔案結構
每個測試檔案生成以下輸出：
- `*_original.wav`: 預處理後的原始音檔
- `*_enhanced.wav`: TTT2模型增強後的音檔  
- `comparison_plot.png`: 波形、頻譜和差異信號的比較圖表

---
*報告生成時間: {timestamp}*
*TTT2 Outside音檔測試系統*"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

if __name__ == "__main__":
    main()
