import torch
import torchaudio
import os
from tqdm import tqdm

def filter_high_frequencies(audio_path, output_path, sample_rate=24000, cutoff_freq=4000):
    """
    過濾掉高於指定頻率的聲音內容
    
    Parameters:
        audio_path (str): 輸入音頻文件路徑
        output_path (str): 輸出音頻文件路徑
        sample_rate (int): 採樣率
        cutoff_freq (int): 截止頻率 (Hz)
    """
    # 讀取音頻
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    # 處理輸入張量的維度
    if waveform.dim() == 3:  # 如果是 3D 張量 (batch, channel, time)
        batch_size, num_channels, time_len = waveform.shape
        waveform = waveform.reshape(-1, time_len)  # 將 batch 和 channel 維度合併
    
    # 確保音頻是單聲道
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # STFT 參數
    n_fft = 2048
    hop_length = n_fft // 4
    
    # 計算 STFT
    stft = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=n_fft, 
        window=torch.hann_window(n_fft),
        return_complex=True
    )
    
    # 計算頻率軸
    freqs = torch.linspace(0, sample_rate//2, stft.size(1))
    
    # 創建頻率遮罩
    mask = (freqs <= cutoff_freq).float()
    
    # 應用遮罩
    masked_stft = stft * mask.unsqueeze(-1)
    
    # 執行 ISTFT
    filtered_waveform = torch.istft(
        masked_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft)
    )
    
    # 正規化音量
    filtered_waveform = filtered_waveform / torch.max(torch.abs(filtered_waveform))
    
    # 如果原始輸入是 3D，恢復原始形狀
    if waveform.dim() == 3:
        filtered_waveform = filtered_waveform.reshape(batch_size, num_channels, -1)
    
    # 保存處理後的音頻
    torchaudio.save(output_path, filtered_waveform.unsqueeze(0) if filtered_waveform.dim() == 1 else filtered_waveform, sample_rate)
    
    return filtered_waveform

def process_folder(input_folder, output_folder, sample_rate=24000, cutoff_freq=4000):
    """
    處理資料夾內所有的 .wav 檔案
    
    Parameters:
        input_folder (str): 輸入資料夾路徑
        output_folder (str): 輸出資料夾路徑
        sample_rate (int): 採樣率
        cutoff_freq (int): 截止頻率 (Hz)
    """
    # 創建輸出資料夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取所有 .wav 檔案
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    # 使用 tqdm 顯示處理進度
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        input_path = os.path.join(input_folder, wav_file)
        output_path = os.path.join(output_folder, f"filtered_{wav_file}")
        
        try:
            filter_high_frequencies(
                input_path,
                output_path,
                sample_rate=sample_rate,
                cutoff_freq=cutoff_freq
            )
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

if __name__ == "__main__":
    input_folder = "tsne_enhanced_outputs"  # 替換為你的輸入資料夾路徑
    output_folder = "tsne_enhanced_outputs"  # 替換為你的輸出資料夾路徑
    
    process_folder(
        input_folder,
        output_folder,
        sample_rate=24000,
        cutoff_freq=4000
    )
    
    print(f"Processed files saved to: {output_folder}")