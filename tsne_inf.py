import torch
import torchaudio
import os
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 導入必要的模塊和函數
from tsne import EnhancedWavTokenizer, plot_spectrograms
from encoder.utils import convert_audio

def pad_or_trim(wav, target_length):
    """將音頻填充或裁剪到指定長度"""
    current_length = wav.size(-1)
    
    if (current_length == target_length):
        return wav
    
    if (current_length > target_length):
        # 如果太長，從中間截取所需長度
        start = (current_length - target_length) // 2
        return wav[..., start:start + target_length]
    
    # 如果太短，進行填充
    padding_length = target_length - current_length
    left_pad = padding_length // 2
    right_pad = padding_length - left_pad
    return F.pad(wav, (left_pad, right_pad), mode='reflect')

def process_audio(audio_path, target_sr=24000, normalize=True):
    """處理輸入音頻"""
    try:
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, target_sr, 1)  # [1, T]
        
        if normalize:
            wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
            
        return wav
    except Exception as e:
        print(f"Error processing audio {audio_path}: {str(e)}")
        raise

def load_model(config_path, model_path, checkpoint_path, device):
    """載入訓練好的模型"""
    # 使用 EnhancedWavTokenizer 替代 EnhancedModel
    model = EnhancedWavTokenizer(config_path, model_path).to(device)
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'N/A')}")
    
    return model

def inference(input_path, output_path, spec_path, model, device, segment_length=3*24000, feature_scale=1.5):
    """推論函數 - 支援長音訊分段處理"""
    # 讀取並預處理音頻
    wav = process_audio(input_path).to(device)
    
    # 確保格式正確 [B, C, T]
    if wav.dim() == 2:  # [C, T]
        wav = wav.unsqueeze(0)  # [1, C, T]
    
    # 決定是否需要分段處理
    if wav.shape[-1] <= segment_length or segment_length <= 0:
        # 短音頻，直接處理
        with torch.no_grad():
            # 使用 tsne.py 中的模型推理邏輯
            output, _, _ = model(wav)
            output = output.cpu()
    else:
        # 長音頻，分段處理
        print(f"Processing long audio in segments (length={wav.shape[-1]}, segment={segment_length})")
        segments = []
        
        # 分段處理，使用重疊來避免拼接處的不連續性
        overlap = segment_length // 4  # 設定25%重疊
        total_length = wav.shape[-1]
        
        for i in range(0, total_length, segment_length - overlap):
            # 提取段落
            end = min(i + segment_length, total_length)
            segment = wav[:, :, i:end]
            
            # 處理段落長度
            if segment.shape[-1] < segment_length:
                if segment.shape[-1] < segment_length // 2:  # 太短的最後一段直接跳過
                    break
                segment = pad_or_trim(segment, segment_length)
            
            # 進行推理
            with torch.no_grad():
                segment_output, _, _ = model(segment)
                
                # 如果不是第一段，應用淡入效果
                if i > 0 and segments:
                    fade_len = overlap
                    fade_in = torch.linspace(0, 1, fade_len, device=device)
                    segment_output[..., :fade_len] *= fade_in
                
                # 如果不是最後一段，應用淡出效果
                if end < total_length:
                    fade_len = min(overlap, segment_output.shape[-1])
                    fade_out = torch.linspace(1, 0, fade_len, device=device)
                    segment_output[..., -fade_len:] *= fade_out
                
                # 納入有效部分
                valid_length = min(segment_output.shape[-1], end - i)
                segments.append(segment_output[..., :valid_length].cpu())
        
        # 合併結果
        if segments:
            output = torch.cat(segments, dim=-1)
            # 確保不超過原始長度
            output = output[..., :total_length]
        else:
            print("Warning: No valid segments processed!")
            return
    
    # 正規化輸出振幅
    output = output / (torch.max(torch.abs(output)) + 1e-8)
    
    # 修正: 確保輸出格式為 [C, T] 2D 張量，適合 torchaudio.save
    if output.dim() == 3:  # [B, C, T]
        output = output.squeeze(0)  # 移除批次維度，變成 [C, T]
    
    # 保存音頻文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, output, 24000)
    
    # 生成並保存頻譜圖
    if spec_path:
        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
        # 修正: 確保頻譜圖輸入格式正確
        temp_output = output.to(device)
        # 確保適合 plot_spectrograms 的格式
        if temp_output.dim() == 2:  # [C, T]
            temp_output = temp_output.unsqueeze(0)  # [1, C, T]
        plot_spectrograms(
            temp_output,
            spec_path,
            device,
            title=f'Enhanced Audio Spectrogram'
        )
        del temp_output  # 釋放 GPU 記憶體
    
    print(f"Saved to {output_path}")
    if spec_path:
        print(f"Spectrogram saved to {spec_path}")

def batch_process(input_dir, output_dir, model, device, feature_scale=1.5):
    """批次處理整個目錄的音頻文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 查找所有 WAV 文件
    wav_files = list(input_dir.glob("**/*.wav"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # 批次處理
    for audio_file in tqdm(wav_files, desc="Processing audio files"):
        # 創建相對路徑
        rel_path = audio_file.relative_to(input_dir)
        output_path = output_dir / f"enhanced_{rel_path}"
        spec_path = output_dir / f"enhanced_{rel_path.stem}_spectrogram.png"
        
        # 確保目標目錄存在
        os.makedirs(output_path.parent, exist_ok=True)
        
        # 處理單個文件
        try:
            inference(
                str(audio_file),
                str(output_path),
                str(spec_path),
                model,
                device,
                feature_scale=feature_scale
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue

def main():
    """主函數"""
    # 配置
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",
        'checkpoint_path': "./tout/best_model.pth",  # 更新為最新模型路徑
        'input_dir': "./tte",  # 輸入目錄
        'output_dir': "./ttout",  # 輸出目錄
        'feature_scale': 1.5,  # 與 tsne.py 同步
        'segment_length': 3 * 24000  # 3秒段落，避免長音頻的 VRAM 問題
    }
    
    # 設置 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 清理 GPU 快取
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before loading model: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # 載入模型
    model = load_model(
        config['config_path'],
        config['model_path'],
        config['checkpoint_path'],
        device
    )
    
    # 批次處理所有文件
    batch_process(
        config['input_dir'],
        config['output_dir'],
        model,
        device,
        feature_scale=config['feature_scale']
    )
    
    print("Processing completed!")

if __name__ == "__main__":
    main()