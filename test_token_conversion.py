#!/usr/bin/env python3
"""
Token轉換正確性測試腳本
實驗編號: token_conversion_test_20251001
目的: 驗證WavTokenizer的token轉換方法是否正確
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加模組路徑
sys.path.append('/home/sbplab/ruizi/WavTokenizer')
from decoder.pretrained import WavTokenizer

def load_wavtokenizer_model():
    """
    載入預訓練的WavTokenizer模型
    
    Returns:
        WavTokenizer: 預訓練模型實例
    """
    try:
        # 設置模型路徑 (使用現有的正確路徑)
        config_path = "/home/sbplab/ruizi/c_code/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"
        
        # 檢查文件是否存在
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return None
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
        
        # 載入預訓練模型 (參考README.md範例)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = WavTokenizer.from_pretrained0802(config_path, model_path)
        model = model.to(device)
        model.eval()
        
        print(f"✅ WavTokenizer模型載入成功")
        print(f"   - 設備: {device}")
        
        # 檢查模型基本資訊
        if hasattr(model, 'config'):
            print(f"   - 配置文件已載入")
        else:
            print(f"   - 無法訪問配置信息")
            
        return model
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        print(f"   確認模型文件是否存在:")
        print(f"   - 配置文件: {config_path}")
        print(f"   - 模型文件: {model_path}")
        return None

def load_and_preprocess_audio(audio_path, target_sr=24000, duration=3.0):
    """
    載入並預處理音檔 (參考README.md的convert_audio方法)
    
    Args:
        audio_path (str): 音檔路徑
        target_sr (int): 目標採樣率 (WavTokenizer使用24kHz)
        duration (float): 目標時長(秒)
    
    Returns:
        torch.Tensor: 預處理後的音檔張量
    """
    try:
        # 添加encoder.utils路徑
        sys.path.append('/home/sbplab/ruizi/c_code/encoder')
        from utils import convert_audio
        
        # 載入音檔
        wav, sample_rate = torchaudio.load(audio_path)
        print(f"📁 載入音檔: {Path(audio_path).name}")
        print(f"   - 原始採樣率: {sample_rate} Hz")
        print(f"   - 原始形狀: {wav.shape}")
        print(f"   - 原始時長: {wav.shape[1] / sample_rate:.2f} 秒")
        
        # 使用WavTokenizer的convert_audio函數進行預處理
        wav = convert_audio(wav, sample_rate, target_sr, 1)  # 轉為24kHz, 單聲道
        print(f"   - convert_audio後形狀: {wav.shape}")
        print(f"   - 處理後採樣率: {target_sr} Hz")
        print(f"   - 處理後時長: {wav.shape[1] / target_sr:.2f} 秒")
        
        # 截取或填充到指定時長
        target_length = int(target_sr * duration)
        if wav.shape[1] > target_length:
            wav = wav[:, :target_length]
            print(f"   - 截取到 {duration} 秒")
        elif wav.shape[1] < target_length:
            pad_length = target_length - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_length))
            print(f"   - 填充到 {duration} 秒")
        
        print(f"   - 最終形狀: {wav.shape}")
        print(f"   - 最終時長: {wav.shape[1] / target_sr:.2f} 秒")
        
        return wav
        
    except Exception as e:
        print(f"❌ 音檔處理失敗: {e}")
        return None

def test_token_conversion(model, waveform, audio_name):
    """
    測試token轉換過程 (參考README.md範例)
    
    Args:
        model: WavTokenizer模型
        waveform: 音檔張量
        audio_name: 音檔名稱
    
    Returns:
        dict: 轉換結果統計
    """
    print(f"\n🔍 開始測試token轉換: {audio_name}")
    
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    
    with torch.no_grad():
        try:
            # Step 1: 音檔 -> Token (參考README.md: encode_infer)
            print("📊 步驟1: 音檔編碼為discrete tokens...")
            bandwidth_id = torch.tensor([0]).to(device)  # 使用bandwidth_id=0
            
            # 使用encode_infer方法，返回features和discrete_code
            features, discrete_code = model.encode_infer(waveform, bandwidth_id=bandwidth_id)
            
            print(f"   - Features形狀: {features.shape}")
            print(f"   - Discrete code形狀: {discrete_code.shape}")
            print(f"   - Token總數: {discrete_code.numel()}")
            print(f"   - Token範圍: {discrete_code.min().item()} ~ {discrete_code.max().item()}")
            
            # 檢查token統計信息
            unique_tokens = torch.unique(discrete_code)
            print(f"   - 獨特token數量: {len(unique_tokens)}")
            
            # Step 2: Token -> 音檔重建 (參考README.md: decode方法)
            print("📊 步驟2: discrete tokens解碼為音檔...")
            
            # 方法1: 直接使用features解碼
            reconstructed_waveform = model.decode(features, bandwidth_id=bandwidth_id)
            print(f"   - 重建音檔形狀 (使用features): {reconstructed_waveform.shape}")
            
            # 方法2: 使用codes_to_features然後解碼
            try:
                features_from_codes = model.codes_to_features(discrete_code)
                reconstructed_from_codes = model.decode(features_from_codes, bandwidth_id=bandwidth_id)
                print(f"   - 重建音檔形狀 (使用codes): {reconstructed_from_codes.shape}")
                # 使用codes重建的結果進行後續分析
                reconstructed_waveform = reconstructed_from_codes
            except Exception as e:
                print(f"   - codes_to_features方法失敗: {e}")
                print(f"   - 使用features方法的重建結果")
            
            print(f"   - 原始音檔形狀: {waveform.shape}")
            
            # Step 3: 計算重建品質指標
            print("📊 步驟3: 計算重建品質...")
            
            # 確保形狀一致
            min_length = min(waveform.shape[-1], reconstructed_waveform.shape[-1])
            orig_trimmed = waveform[..., :min_length]
            recon_trimmed = reconstructed_waveform[..., :min_length]
            
            # MSE Loss
            mse_loss = torch.nn.functional.mse_loss(recon_trimmed, orig_trimmed).item()
            
            # L1 Loss
            l1_loss = torch.nn.functional.l1_loss(recon_trimmed, orig_trimmed).item()
            
            # 信噪比 (SNR)
            signal_power = torch.mean(orig_trimmed ** 2).item()
            noise_power = torch.mean((recon_trimmed - orig_trimmed) ** 2).item()
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
            
            # Token統計 (使用預設詞彙大小4096，如README.md所示)
            vocab_size = 4096  # 從模型名稱得知: code4096
            invalid_tokens = ((discrete_code < 0) | (discrete_code >= vocab_size)).sum().item()
            token_usage_ratio = len(unique_tokens) / vocab_size
            
            results = {
                'audio_name': audio_name,
                'token_count': discrete_code.numel(),
                'unique_tokens': len(unique_tokens),
                'token_usage_ratio': token_usage_ratio,
                'token_min': discrete_code.min().item(),
                'token_max': discrete_code.max().item(),
                'invalid_tokens': invalid_tokens,
                'vocab_size': vocab_size,
                'mse_loss': mse_loss,
                'l1_loss': l1_loss,
                'snr_db': snr_db,
                'discrete_tokens': discrete_code,
                'original_waveform': waveform,
                'reconstructed_waveform': reconstructed_waveform
            }
            
            print(f"   - MSE損失: {mse_loss:.6f}")
            print(f"   - L1損失: {l1_loss:.6f}")
            print(f"   - 信噪比: {snr_db:.2f} dB")
            print(f"   - 詞彙大小: {vocab_size}")
            print(f"   - Token使用率: {token_usage_ratio:.2%}")
            print(f"   - 無效token數量: {invalid_tokens}")
            
            return results
            
        except Exception as e:
            print(f"❌ Token轉換測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

def visualize_token_analysis(results_list, output_dir):
    """
    可視化token分析結果
    
    Args:
        results_list (list): 測試結果列表
        output_dir (str): 輸出目錄
    """
    print(f"\n📊 生成可視化分析圖表...")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置matplotlib中文字體
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Token分佈直方圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('WavTokenizer Token轉換分析結果', fontsize=16, fontweight='bold')
    
    # 收集所有音檔的token分佈
    all_tokens = []
    for result in results_list:
        if result and 'discrete_tokens' in result:
            all_tokens.extend(result['discrete_tokens'].flatten().tolist())
    
    # 繪製token分佈
    axes[0, 0].hist(all_tokens, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Token值分佈')
    axes[0, 0].set_xlabel('Token值')
    axes[0, 0].set_ylabel('頻率')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 重建品質比較
    audio_names = [r['audio_name'] for r in results_list if r]
    snr_values = [r['snr_db'] for r in results_list if r]
    mse_values = [r['mse_loss'] for r in results_list if r]
    
    x_pos = np.arange(len(audio_names))
    axes[0, 1].bar(x_pos, snr_values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('音檔重建信噪比 (SNR)')
    axes[0, 1].set_xlabel('音檔')
    axes[0, 1].set_ylabel('SNR (dB)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in audio_names], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Token使用率
    usage_ratios = [r['token_usage_ratio'] for r in results_list if r]
    axes[1, 0].bar(x_pos, usage_ratios, color='orange', alpha=0.7)
    axes[1, 0].set_title('Token詞彙使用率')
    axes[1, 0].set_xlabel('音檔')
    axes[1, 0].set_ylabel('使用率')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in audio_names], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. MSE損失比較
    axes[1, 1].bar(x_pos, mse_values, color='salmon', alpha=0.7)
    axes[1, 1].set_title('重建MSE損失')
    axes[1, 1].set_xlabel('音檔')
    axes[1, 1].set_ylabel('MSE損失')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in audio_names], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'token_conversion_analysis_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 分析圖表已保存: {plot_path}")
    
    plt.show()

def generate_test_report(results_list, output_dir):
    """
    生成測試報告
    
    Args:
        results_list (list): 測試結果列表
        output_dir (str): 輸出目錄
    """
    print(f"\n📝 生成測試報告...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'token_conversion_test_report_{timestamp}.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# WavTokenizer Token轉換正確性測試報告\n\n")
        f.write(f"**實驗編號**: token_conversion_test_{timestamp}\n")
        f.write(f"**測試時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**測試目的**: 驗證WavTokenizer的token轉換方法是否正確\n")
        f.write(f"**測試音檔來源**: /home/sbplab/ruizi/c_code/1n/\n\n")
        
        f.write("## 測試摘要\n\n")
        valid_results = [r for r in results_list if r]
        f.write(f"- **測試音檔數量**: {len(valid_results)}\n")
        f.write(f"- **成功轉換**: {len(valid_results)}\n")
        f.write(f"- **失敗轉換**: {len(results_list) - len(valid_results)}\n\n")
        
        if valid_results:
            avg_snr = np.mean([r['snr_db'] for r in valid_results])
            avg_mse = np.mean([r['mse_loss'] for r in valid_results])
            avg_usage = np.mean([r['token_usage_ratio'] for r in valid_results])
            total_invalid = sum([r['invalid_tokens'] for r in valid_results])
            
            f.write("## 整體性能指標\n\n")
            f.write(f"- **平均信噪比**: {avg_snr:.2f} dB\n")
            f.write(f"- **平均MSE損失**: {avg_mse:.6f}\n")
            f.write(f"- **平均token使用率**: {avg_usage:.2%}\n")
            f.write(f"- **無效token總數**: {total_invalid}\n\n")
            
            # 性能評估
            f.write("## 性能評估\n\n")
            if avg_snr > 20:
                f.write("✅ **重建品質**: 優秀 (SNR > 20 dB)\n")
            elif avg_snr > 15:
                f.write("✅ **重建品質**: 良好 (SNR > 15 dB)\n")
            elif avg_snr > 10:
                f.write("⚠️ **重建品質**: 可接受 (SNR > 10 dB)\n")
            else:
                f.write("❌ **重建品質**: 需要改進 (SNR < 10 dB)\n")
            
            if total_invalid == 0:
                f.write("✅ **Token有效性**: 完美 (無無效token)\n")
            elif total_invalid < len(valid_results):
                f.write("⚠️ **Token有效性**: 大部分正確 (少量無效token)\n")
            else:
                f.write("❌ **Token有效性**: 存在問題 (多個無效token)\n")
            
            if avg_usage > 0.1:
                f.write("✅ **詞彙使用**: 豐富 (使用率 > 10%)\n")
            elif avg_usage > 0.05:
                f.write("✅ **詞彙使用**: 適中 (使用率 > 5%)\n")
            else:
                f.write("⚠️ **詞彙使用**: 稀疏 (使用率 < 5%)\n")
            
            f.write("\n## 詳細測試結果\n\n")
            f.write("| 音檔名稱 | Token數量 | 獨特Token | 使用率 | SNR(dB) | MSE損失 | 無效Token |\n")
            f.write("|---------|----------|----------|--------|---------|---------|----------|\n")
            
            for result in valid_results:
                f.write(f"| {result['audio_name'][:20]}{'...' if len(result['audio_name']) > 20 else ''} | "
                       f"{result['token_count']} | {result['unique_tokens']} | "
                       f"{result['token_usage_ratio']:.2%} | {result['snr_db']:.2f} | "
                       f"{result['mse_loss']:.6f} | {result['invalid_tokens']} |\n")
            
            f.write("\n## 實驗結論\n\n")
            if avg_snr > 15 and total_invalid == 0:
                f.write("🎉 **Token轉換方法正確**: WavTokenizer能夠正確地將音檔轉換為token並重建，"
                       "重建品質良好，無無效token產生。\n\n")
            elif avg_snr > 10:
                f.write("✅ **Token轉換基本正確**: WavTokenizer的token轉換功能正常，"
                       "但可能需要調整參數以提高重建品質。\n\n")
            else:
                f.write("❌ **Token轉換需要檢查**: 重建品質較低，建議檢查模型配置或數據預處理。\n\n")
            
            f.write("## 建議\n\n")
            if total_invalid > 0:
                f.write("- 檢查token範圍限制，確保所有生成的token都在詞彙範圍內\n")
            if avg_snr < 15:
                f.write("- 考慮調整模型超參數以提高重建品質\n")
            if avg_usage < 0.05:
                f.write("- 檢查詞彙大小設置，可能存在詞彙利用不充分的問題\n")
    
    print(f"✅ 測試報告已保存: {report_path}")
    return report_path

def main():
    """主測試函數"""
    print("=" * 60)
    print("🎵 WavTokenizer Token轉換正確性測試")
    print("=" * 60)
    
    # 設置路徑
    audio_dir = "/home/sbplab/ruizi/c_code/1n"
    output_dir = "/home/sbplab/ruizi/c_code/results/token_conversion_test"
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型
    model = load_wavtokenizer_model()
    if model is None:
        print("❌ 無法載入模型，測試終止")
        return
    
    # 獲取音檔列表
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    audio_files = audio_files[:5]  # 限制測試5個音檔
    
    print(f"\n📁 找到 {len(audio_files)} 個音檔進行測試")
    
    # 測試每個音檔
    results_list = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*50}")
        print(f"📊 測試進度: {i}/{len(audio_files)}")
        print(f"{'='*50}")
        
        audio_path = os.path.join(audio_dir, audio_file)
        
        # 載入並預處理音檔
        waveform = load_and_preprocess_audio(audio_path)
        if waveform is None:
            results_list.append(None)
            continue
        
        # 測試token轉換
        result = test_token_conversion(model, waveform, audio_file)
        results_list.append(result)
    
    # 生成分析報告
    print(f"\n{'='*60}")
    print("📊 生成分析結果")
    print(f"{'='*60}")
    
    # 可視化分析
    visualize_token_analysis(results_list, output_dir)
    
    # 生成文字報告
    report_path = generate_test_report(results_list, output_dir)
    
    print(f"\n🎉 測試完成！")
    print(f"📂 結果保存在: {output_dir}")
    print(f"📝 報告文件: {report_path}")

if __name__ == "__main__":
    main()