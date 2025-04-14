import torch
import torchaudio
import os
from tsne import EnhancedWavTokenizer  # 改為從 tsne.py 導入
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import numpy as np

def ensure_shape(wav):
    """確保張量形狀為 [B, C, T]"""
    if (wav.dim() == 1):  # [T]
        wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif (wav.dim() == 2):  # [C, T]
        wav = wav.unsqueeze(0)  # [1, C, T]
    elif (wav.dim() == 3):  # [B, T, C]
        if (wav.size(1) != 1):
            wav = wav.transpose(1, 2)  # 將形狀變為 [B, C, T]
    return wav

def process_audio(audio_path, normalize=True):
    """
    處理音頻文件
    
    Parameters:
        audio_path (str): 音頻文件路徑
        normalize (bool): 是否進行正規化
        
    Returns:
        torch.Tensor: 處理後的音頻張量
    """
    # 讀取音頻
    waveform, sr = torchaudio.load(audio_path)
    
    # 重採樣到 24kHz (如果需要)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)
    
    # 轉換為單聲道 (如果是立體聲)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 正規化
    if normalize:
       (torch.max(torch.abs(waveform)) + 1e-8)
    
    return waveform

def test_model(input_path, output_path, model_config):
    """使用 tsne.py 的 EnhancedWavTokenizer 模型進行測試"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. 初始化模型 - 使用 tsne.py 的模型
        model = EnhancedWavTokenizer(
            config_path=model_config['config_path'],
            model_path=model_config['model_path']
        ).to(device)
        
        # 2. 載入訓練的權重
        checkpoint = torch.load(model_config['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {model_config['checkpoint_path']}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.6f}")
        
        # 3. 評估模式
        model.eval()
        
        print("\nProcessing input audio...")
        
        # 4. 輸入處理 - 與 tsne.py 保持一致
        input_wav = process_audio(input_path, normalize=True)  # [1, T]
        input_wav = ensure_shape(input_wav)  # [1, 1, T]
        input_wav = input_wav.to(device)
        # 移除標準化，使用與 tsne.py 相同的正規化方法
        input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
        
        # 5. 推理
        with torch.no_grad():
            # 通過模型獲取輸出和特徵
            output, input_features, enhanced_features = model(input_wav)
            
            # 正規化輸出
            output = output / (output.abs().max() + 1e-8)
            
            print("Model output shapes:")
            print(f"Output: {output.shape}")
            print(f"Input features: {input_features.shape}")
            print(f"Enhanced features: {enhanced_features.shape}")
        
        # 6. 保存結果 - 使用與 tsne.py 相同的邏輯
        print("\nSaving results...")
        with torch.no_grad():
            output_cpu = output.cpu()
            input_cpu = input_wav.cpu()
            
            # 正規化
            output_audio = output_cpu / torch.max(torch.abs(output_cpu))
            input_audio = input_cpu / torch.max(torch.abs(input_cpu))
            
            # 調整形狀
            output_audio = output_audio.squeeze(1)  # [B, T]
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存音頻
            torchaudio.save(output_path, output_audio, 24000)
            
            # 保存頻譜圖
            plot_path = os.path.join(
                os.path.dirname(output_path), 
                f'spectrogram_{os.path.basename(output_path)[:-4]}.png'
            )
            
            # 使用臨時 GPU 張量進行頻譜圖繪製
            temp_audio = output_audio.to(device)
            plot_spectrograms(
                temp_audio,
                plot_path,
                device,
                title=f'Enhanced Audio Spectrogram'
            )
            del temp_audio  # 釋放 GPU 記憶體
        
        print(f"Successfully saved to {output_path}")
        print(f"Spectrogram saved to {plot_path}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Full error trace:")
        import traceback
        traceback.print_exc()
        raise

def evaluate_metrics(original_audio, enhanced_audio, sr=24000):
    """計算音頻品質評估指標"""
    # 確保輸入是 numpy array
    if torch.is_tensor(original_audio):
        original_audio = original_audio.cpu().numpy()
    if torch.is_tensor(enhanced_audio):
        enhanced_audio = enhanced_audio.cpu().numpy()
    
    # 計算 SNR (Signal-to-Noise Ratio)
    def calculate_snr(clean, enhanced):
        noise = clean - enhanced
        snr = 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))
        return snr
    
    # 計算 PESQ (Perceptual Evaluation of Speech Quality)
    from pesq import pesq
    try:
        pesq_score = pesq(sr, original_audio.squeeze(), enhanced_audio.squeeze(), 'wb')
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        pesq_score = 0
    
    # 計算 STOI (Short-Time Objective Intelligibility)
    from pystoi import stoi
    try:
        stoi_score = stoi(original_audio.squeeze(), enhanced_audio.squeeze(), sr, extended=False)
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        stoi_score = 0
    
    return {
        'snr': calculate_snr(original_audio, enhanced_audio),
        'pesq': pesq_score,
        'stoi': stoi_score
    }

def process_directory(input_dir, output_dir, model_config, reference_dir=None):
    """增強版的批次處理函數，包含評估指標"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 載入模型
    model = EnhancedWavTokenizer(
        config_path=model_config['config_path'],
        model_path=model_config['model_path']
    ).to(device)
    
    checkpoint = torch.load(model_config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from: {model_config['checkpoint_path']}")
    model.eval()
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 取得所有 .wav 檔案
    wav_files = list(Path(input_dir).rglob("*.wav"))
    total_files = len(wav_files)
    print(f"\nFound {total_files} WAV files to process")
    
    # 儲存評估結果
    metrics_results = []
    
    # 批次處理
    for idx, wav_path in enumerate(wav_files, 1):
        try:
            print(f"\nProcessing [{idx}/{total_files}]: {wav_path}")
            
            # 保持相對路徑結構
            rel_path = wav_path.relative_to(input_dir)
            output_path = Path(output_dir) / f"enhanced_{rel_path}"
            os.makedirs(output_path.parent, exist_ok=True)
            
            # 處理音頻 - 與 tsne.py 完全一致
            input_wav = process_audio(str(wav_path), normalize=True)  # [1, T]
            input_wav = ensure_shape(input_wav)  # [1, 1, T]
            input_wav = input_wav.to(device)
            # 移除標準化，使用與 tsne.py 相同的正規化方法
            input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
            
            # 執行推理
            with torch.no_grad():
                output, input_features, enhanced_features = model(input_wav)
                
                # 如果有參考音頻，計算評估指標
                if reference_dir:
                    ref_path = Path(reference_dir) / wav_path.relative_to(input_dir)
                    if ref_path.exists():
                        ref_audio = process_audio(str(ref_path), normalize=True)
                        metrics = evaluate_metrics(ref_audio, output_audio)
                        metrics['file_name'] = str(wav_path.name)
                        metrics_results.append(metrics)
                        print(f"\nMetrics for {wav_path.name}:")
                        print(f"SNR: {metrics['snr']:.2f} dB")
                        print(f"PESQ: {metrics['pesq']:.2f}")
                        print(f"STOI: {metrics['stoi']:.2f}")
                
                # 移動到 CPU 並正規化
                output_cpu = output.cpu()
                input_cpu = input_wav.cpu()
                
                # 正規化
                output_audio = output_cpu / torch.max(torch.abs(output_cpu))
                input_audio = input_cpu / torch.max(torch.abs(input_cpu))
                
                # 調整形狀
                output_audio = output_audio.squeeze(1)  # [B, T]
                
                # 保存音頻
                torchaudio.save(output_path, output_audio, 24000)
                
                # 保存頻譜圖
                spec_path = output_path.with_suffix('.png')
                temp_audio = output_audio.to(device)  # 臨時移到 GPU
                plot_spectrograms(
                    temp_audio,
                    spec_path,
                    device,
                    title=f'Enhanced Audio Spectrogram'
                )
                del temp_audio  # 釋放 GPU 記憶體
                
        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            continue
            
    # 保存評估結果
    if metrics_results:
        import pandas as pd
        df = pd.DataFrame(metrics_results)
        results_path = Path(output_dir) / 'evaluation_results.csv'
        df.to_csv(results_path, index=False)
        
        # 計算平均指標
        print("\nOverall Evaluation Results:")
        print(f"Average SNR: {df['snr'].mean():.2f} ± {df['snr'].std():.2f} dB")
        print(f"Average PESQ: {df['pesq'].mean():.2f} ± {df['pesq'].std():.2f}")
        print(f"Average STOI: {df['stoi'].mean():.2f} ± {df['stoi'].std():.2f}")
    
    print("\nProcessing completed!")

# 使用與 tsne.py 相同的 plot_spectrograms 函數
from tsne import plot_spectrograms

if __name__ == "__main__":
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",
        'checkpoint_path': "./tout2/best_model.pth",  # 修改為 tsne.py 的輸出目錄
    }
    
    # 批次處理整個資料夾
    process_directory(
        input_dir="./tte",  # 測試集輸入資料夾
        output_dir="./tsne_enhanced_outputs",  # 增強後的輸出資料夾
        model_config=config,
        reference_dir="./tte_clean"  # 乾淨參考音頻的資料夾（如果有的話）
    )
    
    """
    # 單檔處理（已註解）
    test_model(
        input_path="./box/nor_boy1_box_LDV_100.wav",
        output_path="./enhanced_outputs/test_nor_boy1_box_LDV_100.wav",
        model_config=config
    )
    """
