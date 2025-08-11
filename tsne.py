# Set matplotlib backend to non-interactive backend to avoid thread issues
import matplotlib
matplotlib.use('Agg')
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.utils.data as torch_data  # Add this line
import torchaudio
import os
import sys
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
from sklearn.manifold import TSNE  # 添加 t-SNE 導入
# Add the WavTokenizer directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from try3 import AudioDataset  # Add this import
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

def set_seed(seed=42):
    """固定隨機種子以確保每次訓練的結果都是可重現的"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 若使用多個GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"已設定隨機種子為: {seed}")

def plot_spectrograms(audio, save_path, device, title="Spectrogram"):
    """Plot and save spectrograms."""
    try:
        # 確保在同一個設備上
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=4096,
            hop_length=512,
            win_length=4096,
            n_mels=128
        ).to(device)
        
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        
        with torch.no_grad():
            mel_spec = transform(audio)
            spec_db = amplitude_to_db(mel_spec)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_spectrograms: {str(e)}")
        print(f"Audio device: {audio.device}")
        print(f"Audio shape: {audio.shape}")

from encoder.utils import convert_audio

class Encodec(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config_path = "/home/sbplab/rui.zi/WavTokenizer/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "/home/sbplab/rui.zi/WavTokenizer/models/wavtokenizer_large_speech_320_24k.ckpt"
        base_model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.encoder = base_model.feature_extractor.encodec.encoder
        self.decoder = base_model.feature_extractor.encodec.decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='gelu', leaky_relu_slope=0.1, dropout_rate=0.35):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # 可選的激活函數
        if (activation == 'gelu'):
            self.act = nn.GELU()
        elif (activation == 'leaky_relu'):
            self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # 添加 Dropout 層
        self.dropout1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # 在第一個激活函數後應用 Dropout
        out = self.dropout1(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        
        out += identity
        out = self.act(out)
        
        return out

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, config_path, model_path, num_residual_blocks=5, dropout_rate=0.25):  # 添加 dropout_rate 參數
        super().__init__()
        # 基本維度設置
        encoder_dim = 512  # 編碼器維度
        hidden_dim = 256   # 隱藏層維度
        
        # 載入基礎模型
        base_model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.wavtokenizer = base_model  # 保存完整的WavTokenizer模型
        self.encodec = base_model.feature_extractor.encodec
        
        # 凍結encoder
        for param in self.encodec.encoder.parameters():
            param.requires_grad = False
        
        # 解凍decoder (可選)
        for param in self.encodec.decoder.parameters():
            param.requires_grad = False
        
        # 特徵增強層架構
        self.input_norm = nn.LayerNorm(encoder_dim)  # 輸入歸一化
        
        # 降維卷積
        self.down_conv = nn.Sequential(
            nn.Conv1d(encoder_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 添加第一個 Dropout 層
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 殘差塊，從2個增加到3個，並傳遞 dropout_rate 參數
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation='gelu', dropout_rate=dropout_rate) 
            for _ in range(num_residual_blocks)
        ])
        
        # 上採樣卷積
        self.up_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, encoder_dim, kernel_size=1),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU()
        )
        
        # 輸出歸一化
        self.output_norm = nn.LayerNorm(encoder_dim)
        
        # 設置所有增強層參數為可訓練
        for module in [self.input_norm, self.down_conv, self.residual_blocks, self.up_conv, self.output_norm]:
            for param in module.parameters():
                param.requires_grad = True

    def ensure_feature_shape(self, features):
        """確保特徵形狀標準化為 [B, C=512, T]"""
        # 如果特徵形狀是 [B, T, C]，則轉至為 [B, C, T]
        if features.shape[1] != 512 and features.shape[2] == 512:
            print(f"轉至特徵從 [B, T, C] 到 [B, C, T]")
            features = features.transpose(1, 2)
        # 確認特徵形狀正确
        assert features.shape[1] == 512, f"特徵通道數必須是512，當前是 {features.shape[1]}"
        return features

    def forward(self, x, bandwidth_id=None):
        # 確保輸入張量具有正確的形狀：[batch_size, channels, time]
        if x.dim() == 2:  # 如果是 [batch_size, time]
            x = x.unsqueeze(1)  # 變為 [batch_size, channels=1, time]
            print(f"Reshaped 2D tensor to 3D: {x.shape}")
        elif x.dim() == 1:  # 如果是 [time]
            x = x.unsqueeze(0).unsqueeze(0)  # 變為 [batch_size=1, channels=1, time]
            print(f"Reshaped 1D tensor to 3D: {x.shape}")
        
        # 檢查形狀是否正確
        assert x.dim() == 3, f"Expected input tensor with 3 dimensions, got {x.dim()} dimensions with shape {x.shape}"
            
        # 確保bandwidth_id正確設置
        if bandwidth_id is None:
            bandwidth_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # 使用與wav_re.py相同的方法提取特徵
        with torch.no_grad():
            try:
                input_features, discrete_code = self.wavtokenizer.encode_infer(x, bandwidth_id=bandwidth_id)
                
                # 確保輸入特入特徵形狀標準化
                input_features = self.ensure_feature_shape(input_features)
                input_features = input_features.clone().detach().requires_grad_(True)
                
            except Exception as e:
                print(f"Error in encode_infer: {e}")
                print(f"Input tensor shape when error occurred: {x.shape}")
                raise
        
        # 特徵增強過程
        enhanced = input_features.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        # 克隆增強特徵，確保它可以用於反向傳播
        enhanced = enhanced.clone()
        enhanced = self.input_norm(enhanced)
        enhanced = enhanced.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # 降維處理
        enhanced = self.down_conv(enhanced)
        
        # 在第一個激活函數後應用 Dropout
        enhanced = self.dropout1(enhanced)
        
        # 應用殘差塊
        for block in self.residual_blocks:
            enhanced = block(enhanced)
        
        # 恢復原始維度
        enhanced = self.up_conv(enhanced)
        
        # 最終歸一化
        enhanced = enhanced.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        enhanced = self.output_norm(enhanced)
        enhanced = enhanced.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # 確保增强特徵形狀標準化
        enhanced = self.ensure_feature_shape(enhanced)
        
        # 通過解碼器
        decoded = self.encodec.decoder(enhanced)
        
        return decoded, input_features, enhanced, discrete_code

class EnhancedWavTokenizer(nn.Module):
    def __init__(self, config_path, model_path, dropout_rate=0.25):
        super().__init__()
        self.feature_extractor = EnhancedFeatureExtractor(config_path, model_path, dropout_rate=dropout_rate)
        
    def forward(self, x):
        # 確保輸入張量維度正確
        if x.dim() == 2:  # (batch, time)
            x = x.unsqueeze(1)  # (batch, channel, time)
        elif x.dim() == 1:  # (time,)
            x = x.unsqueeze(0).unsqueeze(0)  # (batch, channel, time)
            
        # 設置 bandwidth_id (device要與輸入張量相同)
        bandwidth_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 特徵提取和增強
        output, input_features, enhanced_features, discrete_code = self.feature_extractor(x, bandwidth_id)
            
        return output, input_features, enhanced_features, discrete_code

def compute_feature_loss(enhanced_features, target_features, device):
    """計算特徵空間的損失"""
    # 正規化特徵
    enhanced_norm = F.normalize(enhanced_features, dim=1)
    target_norm = F.normalize(target_features, dim=1)
    
    # 計算餘弦相似度
    cos_sim = torch.bmm(enhanced_norm.transpose(1, 2), target_norm)
    
    # 計算 L2 距離
    l2_dist = torch.norm(enhanced_features - target_features, dim=1)
    
    # 組合損失
    similarity_loss = -cos_sim.mean()  # 最大化相似度
    distance_loss = l2_dist.mean()     # 最小化距離
    
    #     return 0.5 * similarity_loss + 0.5 * distance_loss
    return distance_loss

def compute_voice_focused_loss(output, target, device):
    """改進的人聲專注損失函數 (完全從 try3.py 複製)"""
    # 確保輸入和目標張量大小一致
    min_length = min(output.size(-1), target.size(-1))
    output = output[..., :min_length]
    target = target[..., :min_length]
    
    # 確保輸入維度正確
    if output.dim() == 3:
        output = output.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # 基礎時域損失
    time_loss = F.l1_loss(output, target)
    
    def stft_loss(x, y, n_fft=2048):
        x_stft = torch.stft(x, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 計算頻率範圍 (關注人聲頻段 80Hz-3400Hz)
        freqs = torch.linspace(0, 12000, x_stft.shape[1], device=device)
        voice_weights = torch.ones_like(freqs, device=device)
        voice_mask = ((freqs >= 80) & (freqs <= 3400)).float()
        voice_weights = voice_weights + voice_mask * 2.0  # 人聲頻段權重加倍
        
        # 計算損失
        mag_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        phase_loss = 1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))
        weighted_loss = (mag_loss + 0.3 * phase_loss) * voice_weights.unsqueeze(-1)
        return torch.mean(weighted_loss)
    
    # 多尺度STFT損失,完全照搬try3.py的設計
    stft_loss_total = (
        stft_loss(output, target, n_fft=2048) +  # 長時窗口
        stft_loss(output, target, n_fft=1024) +  # 中等窗口 
        stft_loss(output, target, n_fft=512)     # 短時窗口
    )
    
    # 使用與try3.py相同的損失權重
    return 0.3 * time_loss + 0.7 * stft_loss_total

# 修改 compute_hybrid_loss 函數
def compute_hybrid_loss(output, target_wav, enhanced_features, target_features, device):
    """計算混合損失，主要關注特徵空間的對齊"""
    # 確保所有輸入都有正確的維度
    if enhanced_features.dim() == 2:
        enhanced_features = enhanced_features.unsqueeze(0)
    if target_features.dim() == 2:
        target_features = target_features.unsqueeze(0)
    
    # 計算特徵空間的損失，使用改進的compute_feature_loss
    feature_loss = compute_feature_loss(enhanced_features, target_features, device)
    
    return feature_loss, {
        'feature_loss': feature_loss.item(),
        'voice_loss': 0.0  # 不使用語音重建損失，專注於特徵對齊
    }

def save_sample(input_wav, output, target_wav, epoch, batch_idx, save_dir, device, model=None):
    """保存音頻樣本和頻譜圖，使用與infer_tsne.py相同的方法"""
    try:
        # 建立音頻保存目錄
        audio_dir = os.path.join(save_dir, f'epoch_{epoch+1}_samples')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 載入WavTokenizer decoder模型
        config_path = "./config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "./wavtokenizer_large_speech_320_24k.ckpt"  # 使用語音專用模型
        
        # 初始化WavTokenizer模型作為解碼器
        try:
            from decoder.pretrained import WavTokenizer
            import traceback
            
            decoder = WavTokenizer.from_pretrained0802(config_path, model_path)
            decoder = decoder.to(device)
            decoder.eval()  # 確保模型處於評估模式
            print(f"✅ 成功載入WavTokenizer解碼器")
            
            # 設定metadata中的bandwidth_id (與infer_tsne.py保持一致)
            metadata = {"bandwidth_id": 0, "sample_rate": 24000}
            bandwidth_id = torch.tensor([metadata["bandwidth_id"]], device=device)
            
            # 處理每個樣本
            for j in range(min(input_wav.size(0), 3)):  # 最多處理3個樣本
                try:
                    base_name = f"batch_{batch_idx}_sample_{j+1}"
                    print(f"\n處理樣本: {base_name}")
                    
                    with torch.no_grad():
                        # 處理輸入音頻
                        input_audio = input_wav[j].reshape(1, -1)
                        input_audio = input_audio / (torch.max(torch.abs(input_audio)) + 1e-8)
                        
                        # 處理目標音頻
                        target_audio = target_wav[j].reshape(1, -1)
                        target_audio = target_audio / (torch.max(torch.abs(target_audio)) + 1e-8)
                        
                        # 從輸出中提取增強特徵
                        enhanced_features = None
                        
                        # 檢查output是否是tuple且長度至少為3
                        if isinstance(output, tuple) and len(output) >= 3:
                            # 第三個元素通常是增強特徵
                            if output[2] is not None and j < output[2].size(0):
                                enhanced_features = output[2][j:j+1]
                                print(f"從元組中提取增強特徵，形狀: {enhanced_features.shape}")
                                
                                # 使用ensure_feature_shape方法確保特徵形狀正確
                                if model is not None and hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "ensure_feature_shape"):
                                    enhanced_features = model.feature_extractor.ensure_feature_shape(enhanced_features)
                                    print(f"標準化後的特徵形狀: {enhanced_features.shape}")
                                elif enhanced_features.shape[1] != 512 and enhanced_features.shape[2] == 512:
                                    enhanced_features = enhanced_features.transpose(1, 2)
                                    print(f"手動轉至特徵，新形狀: {enhanced_features.shape}")
                        
                        # 嘗試通過模型重新生成特徵
                        if enhanced_features is None and model is not None:
                            print("⚠️ 無法直接提取特徵，嘗試重新生成特徵")
                            try:
                                with torch.no_grad():
                                    # 重新調用模型的前向傳播
                                    temp_input = input_wav[j:j+1].to(device)
                                    _, _, new_enhanced_features, _ = model(temp_input)
                                    
                                    # 確保特徵形狀標準化
                                    if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "ensure_feature_shape"):
                                        new_enhanced_features = model.feature_extractor.ensure_feature_shape(new_enhanced_features)
                                    
                                    enhanced_features = new_enhanced_features
                                    print(f"重新生成的特徵形狀: {enhanced_features.shape}")
                            except Exception as model_err:
                                print(f"重新生成特徵失敗: {str(model_err)}")
                                traceback.print_exc()
                        elif enhanced_features is None and model is None:
                            print("⚠️ 無法重新生成特徵：模型未提供 (model=None)")
                        
                        # 如果仍然無法獲取特徵，使用原始音頻作為回退選項
                        if enhanced_features is None or enhanced_features.shape[1] != 512:
                            print("⚠️ 無法獲得有效的特徵張量，直接使用模型輸出音頻")
                            # 從模型輸出提取音頻
                            if isinstance(output, tuple) and len(output) > 0:
                                output_audio = output[0][j:j+1] if j < output[0].size(0) else input_audio  # 回退到輸入音頻
                            else:
                                output_audio = output[j:j+1] if j < output.size(0) else input_audio  # 回退到輸入音頻
                            
                            output_audio = output_audio.reshape(1, -1)
                            output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
                        else:
                            # 使用解碼器生成音頻
                            try:
                                print(f"使用解碼器解碼特徵，形狀: {enhanced_features.shape}")
                                # 確保特徵形狀正確
                                assert enhanced_features.shape[1] == 512, "特徵通道數必須是512"
                                
                                # 解碼特徵生成音頻
                                output_audio = decoder.decode(enhanced_features, bandwidth_id=bandwidth_id)
                                print(f"解碼成功，輸出音訊形狀: {output_audio.shape}")
                                
                                # 格式化音頻
                                if output_audio.dim() > 2:
                                    output_audio = output_audio.squeeze(0)
                                
                                output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
                            except Exception as decode_err:
                                print(f"❌ 解碼失敗: {str(decode_err)}")
                                traceback.print_exc()
                                # 失敗時回退到輸入音頻
                                output_audio = input_audio
                        
                        # 保存每個音頻樣本和頻譜圖
                        for audio, prefix in [
                            (output_audio, "enhanced"),
                            (input_audio, "input"),
                            (target_audio, "target")
                        ]:
                            # 保存音頻文件
                            audio_path = os.path.join(audio_dir, f"{base_name}_{prefix}.wav")
                            torchaudio.save(audio_path, audio.cpu(), 24000)
                            print(f"🔊 保存{prefix}音頻到: {audio_path}")
                            
                            # 生成頻譜圖
                            spec_path = audio_path.replace(".wav", "_spec.png")
                            plt.figure(figsize=(10, 4))
                            audio_np = audio.cpu().numpy().flatten()
                            
                            # 使用librosa生成頻譜圖
                            import librosa
                            import librosa.display
                            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
                            librosa.display.specshow(D, sr=24000, x_axis="time", y_axis="log")
                            plt.colorbar(format="%+2.0f dB")
                            plt.title(f"Epoch {epoch+1} {prefix.capitalize()} Spectrogram")
                            plt.tight_layout()
                            plt.savefig(spec_path)
                            plt.close()
                            print(f"📊 保存{prefix}頻譜圖到: {spec_path}")
                
                except Exception as e:
                    print(f"❌ 處理樣本 {j+1} 時出錯: {str(e)}")
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"❌ 解碼器模型載入或執行失敗: {str(e)}")
            traceback.print_exc()
            return False
                
    except Exception as e:
        print(f"❌ save_sample函數錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def collate_fn(batch):
    """簡化的 collate 函數"""
    input_wavs = [item[0] for item in batch]
    target_wavs = [item[1] for item in batch]
    
    # 找出最短的音訊長度
    min_len = min(
        min(wav.size(-1) for wav in input_wavs),
        min(wav.size(-1) for wav in target_wavs)
    )
    
    # 對齊長度
    input_wavs = [wav[..., :min_len] for wav in input_wavs]
    target_wavs = [wav[..., :min_len] for wav in target_wavs]
    
    return torch.stack(input_wavs), torch.stack(target_wavs)

def print_gpu_info():
    """打印 GPU 詳細信息"""
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"TensorRT: {'Enabled' if torch.cuda.is_available() and hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and torch.backends.cuda.matmul.allow_tf32 else 'Disabled'}")

def monitor_gpu_memory():
    """監控 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()/1024**2
        cached = torch.cuda.memory_reserved()/1024**2
        return allocated, cached
    return 0, 0

def plot_learning_curve(epochs, losses, lr_values, save_path):
    """繪製訓練過程中的損失和學習率曲線"""
    plt.figure(figsize=(12, 8))
    
    # 創建兩個Y軸
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 損失曲線
    line1 = ax1.plot(epochs, losses, 'b-', label='Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 學習率曲線
    line2 = ax2.plot(epochs, lr_values, 'r-', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合併圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels)
    
    plt.title('Training Loss and Learning Rate Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Learning curve saved to {save_path}")

def plot_learning_curves(epochs, train_losses, val_losses, feature_losses, voice_losses, lr_values, save_path):
    """繪製訓練和驗證過程中的損失和學習率曲線"""
    plt.figure(figsize=(12, 8))
    
    # 創建兩個Y軸
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 訓練損失曲線
    line1 = ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    # 特徵損失曲線
    line2 = ax1.plot(epochs, feature_losses, 'c-', label='Feature Loss')
    # 人聲損失曲線
    line3 = ax1.plot(epochs, voice_losses, 'm-', label='Voice Loss')

    # 驗證損失曲線
    line5 = ax1.plot(epochs, val_losses, 'g-', label='Val Loss')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 學習率曲線
    line4 = ax2.plot(epochs, lr_values, 'r-', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合併圖例
    lines = line1 + line2 + line5 + line4 
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels)
    
    plt.title('Training Losses and Learning Rate Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Learning curves saved to {save_path}")

def plot_tsne_visualization(enhanced_features, target_features, save_path, perplexity=30):
    """在最後一圈訓練中生成 t-SNE 視覺化以比較 target 和 enhanced 特徵的空間分布

    Args:
        enhanced_features: 增強特徵張量 [N, D]
        target_features: 目標特徵張量 [N, D]
        save_path: 保存路徑
        perplexity: t-SNE perplexity參數
    """
    print(f"\n正在生成 t-SNE 視覺化比較圖...")
    
    # 處理原始特徵
    enhanced_features_np = enhanced_features.detach().cpu().numpy()
    target_features_np = target_features.detach().cpu().numpy()
    
    # 確保特徵形狀是 [N, D] 兩維
    if enhanced_features_np.ndim > 2:
        enhanced_features_np = enhanced_features_np.reshape(-1, enhanced_features_np.shape[-1])
    if target_features_np.ndim > 2:
        target_features_np = target_features_np.reshape(-1, target_features_np.shape[-1])
    
    # 限制資料點數量，避免 t-SNE 計算過慢
    max_points = 3000
    if len(enhanced_features_np) > max_points:
        print(f"特徵點數量過多 ({len(enhanced_features_np)}), 隨機抽樣 {max_points} 個點...")
        idx = np.random.choice(len(enhanced_features_np), max_points, replace=False)
        enhanced_features_np = enhanced_features_np[idx]
        target_features_np = target_features_np[idx]
    
    print(f"t-SNE 計算中，特徵形狀: enhanced={enhanced_features_np.shape}, target={target_features_np.shape}")
    
    # 將兩組特徵合併用於 t-SNE 計算
    combined_features = np.vstack((enhanced_features_np, target_features_np))
    labels = np.array(['Enhanced'] * len(enhanced_features_np) + ['Target'] * len(target_features_np))
    
    # 執行 t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)
    
    # 分離結果
    enhanced_tsne = tsne_results[:len(enhanced_features_np)]
    target_tsne = tsne_results[len(enhanced_features_np):]
    
    # 繪製圖形
    plt.figure(figsize=(12, 10))
    plt.scatter(enhanced_tsne[:, 0], enhanced_tsne[:, 1], c='blue', alpha=0.5, label='Enhanced Features')
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], c='red', alpha=0.5, label='Target Features')
    
    plt.title('t-SNE: Enhanced vs Target Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存圖形
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE 視覺化圖已保存到: {save_path}")

def train_model(model, train_loader, optimizer, device, save_dir, config, num_epochs=100, scheduler=None, val_loader=None):
    """修改訓練函數以記錄和繪製訓練指標，包括驗證損失，不使用早停法"""
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建特徵保存目錄
    features_dir = os.path.join(save_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    # 創建 t-SNE 視覺化目錄
    tsne_dir = os.path.join(save_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)
    
    # 初始化早停相關的變數，但設置為無效
    best_val_loss = float('inf')  # 初始化最佳驗證損失為正無窮
    no_improve_count = 0          # 初始化無改善計數器
    early_stopping_patience = float('inf')  # 設為無限大，實際上禁用早停
    
    # 在日誌中顯示早停已禁用
    print("\n=== 早停機制已禁用 ===")
    
    # 設定最佳驗證模型的保存路徑
    best_val_model_path = os.path.join(save_dir, 'best_validation_model.pth')
    
    # 檢查是否已有最佳訓練模型
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if (os.path.exists(best_model_path)):
        checkpoint = torch.load(best_model_path, map_location=device)
        best_loss = checkpoint['loss']
        print(f"\nLoaded previous best loss: {best_loss}")
    else:
        best_loss = float('inf') # 初始化最佳訓練損失
    
    print(f"\nInitial best train loss: {best_loss}")
    print(f"Initial best val loss: {best_val_loss}")
    print(f"Early stopping: Disabled")
    
    # 記錄訓練指標
    epochs_record = []
    train_losses_record = []
    val_losses_record = []
    feature_losses_record = []
    voice_losses_record = []
    lr_values_record = []
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        total_loss = 0.0
        total_feature_loss = 0.0
        total_voice_loss = 0.0
        
        # 收集當前epoch的特徵（不僅是最後一個epoch）
        all_enhanced_features = []
        all_target_features = []
        all_target_discrete_codes = []
        all_input_discrete_codes = []
        min_seq_length = None
        min_discrete_length = None
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (input_wav, target_wav) in progress_bar:
            optimizer.zero_grad()
            
            # 移動數據到設備並進行正規化
            input_wav = input_wav.to(device)
            target_wav = target_wav.to(device)
            
            # 修改：確保輸入數據的幅度合適
            input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
            target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
            
            # 定義bandwidth_id (參考wav_re.py的做法)
            bandwidth_id = torch.zeros(input_wav.size(0), dtype=torch.long, device=input_wav.device)
            
            # 前向傳播
            output, input_features, enhanced_features, input_discrete_code = model(input_wav)
            
            # 獲取目標音頻的特徵
            with torch.no_grad():
                # 修改為使用與input相同的方法提取target特徵
                target_features, target_discrete_code = model.feature_extractor.wavtokenizer.encode_infer(target_wav, bandwidth_id=bandwidth_id)

                # 確保特徵和離散編碼的形狀正確
                enhanced_features = model.feature_extractor.ensure_feature_shape(enhanced_features)
                target_features = model.feature_extractor.ensure_feature_shape(target_features)

                # 收集所有特徵和離散編碼
                curr_seq_length = enhanced_features.size(-1)
                input_discrete_length = input_discrete_code.size(-1)
                target_discrete_length = target_discrete_code.size(-1)
                
                # 更新最短序列長度
                if min_seq_length is None:
                    min_seq_length = curr_seq_length
                else:
                    min_seq_length = min(min_seq_length, curr_seq_length)
                
                # 更新最短離散編碼長度
                if min_discrete_length is None:
                    min_discrete_length = min(input_discrete_length, target_discrete_length)
                else:
                    min_discrete_length = min(min_discrete_length, input_discrete_length, target_discrete_length)
                
                # 保存特徵和離散編碼到CPU
                all_enhanced_features.append(enhanced_features.detach().cpu())
                all_target_features.append(target_features.detach().cpu())
                all_target_discrete_codes.append(target_discrete_code.detach().cpu())
                all_input_discrete_codes.append(input_discrete_code.detach().cpu())

            # 計算損失
            loss, loss_details = compute_hybrid_loss(
                output, target_wav, enhanced_features, target_features, device
            )
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_feature_loss += loss_details["feature_loss"]
            total_voice_loss += loss_details["voice_loss"]
            
            # 監控 GPU 記憶體
            allocated, cached = monitor_gpu_memory()
            
            # 獲取當前學習率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'feat': f'{loss_details["feature_loss"]:.4f}',
                'voice': f'{loss_details["voice_loss"]:.4f}',
                'lr': f'{current_lr:.6f}',
                'GPU_MB': f'{allocated:.0f}/{cached:.0f}'
            })
        
        # 保存每個epoch收集的特徵（可自定義保存間隔）
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            # 在每個保存點處理收集的特徵
            if all_enhanced_features:
                # 創建當前epoch的特徵文件夾
                epoch_features_dir = os.path.join(features_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_features_dir, exist_ok=True)
                
                try:
                    # 將所有特徵裁剪到相同的序列長度
                    all_enhanced_features = [f[..., :min_seq_length] for f in all_enhanced_features]
                    all_target_features = [f[..., :min_seq_length] for f in all_target_features]
                    
                    # 將所有離散編碼裁剪到相同的序列長度
                    all_target_discrete_codes = [code[..., :min_discrete_length] for code in all_target_discrete_codes]
                    all_input_discrete_codes = [code[..., :min_discrete_length] for code in all_input_discrete_codes]
                    
                    # 串接所有特徵和離散編碼
                    enhanced_features = torch.cat(all_enhanced_features, dim=0)
                    target_features = torch.cat(all_target_features, dim=0)
                    target_discrete_codes = torch.cat(all_target_discrete_codes, dim=0)
                    input_discrete_codes = torch.cat(all_input_discrete_codes, dim=0)
                    
                    # 儲存特徵和離散編碼
                    torch.save(enhanced_features, os.path.join(epoch_features_dir, 'enhanced_features.pt'))
                    torch.save(target_features, os.path.join(epoch_features_dir, 'target_features.pt'))
                    torch.save(target_discrete_codes, os.path.join(epoch_features_dir, 'target_discrete_code.pt'))
                    torch.save(input_discrete_codes, os.path.join(epoch_features_dir, 'input_discrete_code.pt'))
                    
                    print(f"\nSaved features at epoch {epoch+1}:")
                    print(f"Enhanced features shape: {enhanced_features.shape}")
                    print(f"Target features shape: {target_features.shape}")
                    print(f"Target discrete codes shape: {target_discrete_codes.shape}")
                    print(f"Input discrete codes shape: {input_discrete_codes.shape}")
                    
                    # 保存特徵和離散編碼的元數據信息
                    from datetime import datetime
                    metadata = {
                        'enhanced_shape': enhanced_features.shape,
                        'target_shape': target_features.shape,
                        'target_discrete_shape': target_discrete_codes.shape,
                        'input_discrete_shape': input_discrete_codes.shape,
                        'bandwidth_id': 0,  # 與wav_re.py保持一致
                        'sample_rate': 24000,
                        'timestamp': str(datetime.now()),
                        'epoch': epoch + 1
                    }
                    torch.save(metadata, os.path.join(epoch_features_dir, 'metadata.pt'))
                except Exception as e:
                    print(f"\nError during feature/discrete code saving for epoch {epoch+1}: {str(e)}")
                    print("Attempting to save each batch individually...")
                    
                    # 嘗試單獨保存每個批次
                    for i, (enh, tgt, tgt_code, inp_code) in enumerate(zip(
                        all_enhanced_features, all_target_features, 
                        all_target_discrete_codes, all_input_discrete_codes)):
                        
                        try:
                            torch.save(enh, os.path.join(epoch_features_dir, f'enhanced_features_batch_{i}.pt'))
                            torch.save(tgt, os.path.join(epoch_features_dir, f'target_features_batch_{i}.pt'))
                            torch.save(tgt_code, os.path.join(epoch_features_dir, f'target_discrete_code_batch_{i}.pt'))
                            torch.save(inp_code, os.path.join(epoch_features_dir, f'input_discrete_code_batch_{i}.pt'))
                            print(f"Successfully saved batch {i} for epoch {epoch+1}")
                        except Exception as batch_error:
                            print(f"Error saving batch {i} for epoch {epoch+1}: {str(batch_error)}")
        
        avg_train_loss = total_loss / len(train_loader)
        avg_feature_loss = total_feature_loss / len(train_loader)
        avg_voice_loss = total_voice_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄當前epoch的訓練指標
        epochs_record.append(epoch + 1)
        train_losses_record.append(avg_train_loss)
        feature_losses_record.append(avg_feature_loss)
        voice_losses_record.append(avg_voice_loss)
        lr_values_record.append(current_lr)
        
        print(f'\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, '
              f'Feature Loss: {avg_feature_loss:.4f}, Voice Loss: {avg_voice_loss:.4f}, '
              f'Learning Rate: {current_lr:.6f}')
        
        # 驗證階段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_feature_loss = 0.0
            val_voice_loss = 0.0
            
            with torch.no_grad():
                for input_wav, target_wav in val_loader:
                    # 移動數據到設備並進行正規化
                    input_wav = input_wav.to(device)
                    target_wav = target_wav.to(device)
                    
                    # 確保輸入數據的幅度合適
                    input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
                    target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
                    
                    # 前向傳播
                    output, input_features, enhanced_features, _ = model(input_wav)
                    
                    # 修改: 使用與訓練時相同的方式提取目標特徵
                    bandwidth_id = torch.zeros(input_wav.size(0), dtype=torch.long, device=input_wav.device)
                    target_features, _ = model.feature_extractor.wavtokenizer.encode_infer(target_wav, bandwidth_id=bandwidth_id)
                    
                    # 確保特徵形狀一致
                    enhanced_features = model.feature_extractor.ensure_feature_shape(enhanced_features)
                    target_features = model.feature_extractor.ensure_feature_shape(target_features)
                    
                    # 計算驗證損失，確保與訓練階段一致
                    loss, loss_details = compute_hybrid_loss(
                        output, target_wav, enhanced_features, target_features, device
                    )
                    
                    val_loss += loss.item()
                    val_feature_loss += loss_details["feature_loss"]
                    val_voice_loss += loss_details["voice_loss"]
            
            # 計算平均損失
            avg_val_loss = val_loss / len(val_loader)
            avg_val_feature_loss = val_feature_loss / len(val_loader)
            avg_val_voice_loss = val_voice_loss / len(val_loader)
            
            val_losses_record.append(avg_val_loss)
            
            print(f'Validation Loss: {avg_val_loss:.4f}, '
                  f'Val Feature Loss: {avg_val_feature_loss:.4f}, Val Voice Loss: {avg_val_voice_loss:.4f}')
            
            # 判斷是否獲得了更好的驗證損失
            if avg_val_loss < best_val_loss:
                improvement = best_val_loss - avg_val_loss
                best_val_loss = avg_val_loss
                no_improve_count = 0  # 重置計數器
                
                # 保存基於驗證損失的最佳模型
                validation_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_loss': avg_train_loss,
                    'improvement': improvement,
                    'epochs_record': epochs_record,
                    'train_losses_record': train_losses_record,
                    'val_losses_record': val_losses_record,
                    'feature_losses_record': feature_losses_record,
                    'voice_losses_record': voice_losses_record,
                    'lr_values_record': lr_values_record
                }
                
                # 如果有排程器，也保存它的狀態
                if scheduler:
                    validation_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    
                torch.save(validation_checkpoint, best_val_model_path)
                
                print(f"\n=== New Best Validation Model Saved ===")
                print(f"Improvement: {improvement:.6f}")
                print(f"New best validation loss: {best_val_loss:.6f}")
                print(f"Saved to: {best_val_model_path}")
            else:
                no_improve_count += 1
                print(f"\nNo validation improvement for {no_improve_count} epochs")
            
        else:
            # 如果沒有驗證集，添加None以保持列表長度一致
            val_losses_record.append(None)
            
            # 如果沒有驗證集，就使用訓練損失來調整學習率
            if scheduler is not None:
                scheduler.step(avg_train_loss)
            
        # 判斷是否需要保存最佳模型 (基於訓練損失)
        if avg_train_loss < best_loss:
            improvement = best_loss - avg_train_loss
            best_loss = avg_train_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            
            # 保存檢查點
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'improvement': improvement,
                'epochs_record': epochs_record,
                'train_losses_record': train_losses_record,
                'val_losses_record': val_losses_record,
                'feature_losses_record': feature_losses_record,
                'voice_losses_record': voice_losses_record,
                'lr_values_record': lr_values_record
            }
            
            # 如果有排程器，也保存它的狀態
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
            torch.save(checkpoint_data, best_model_path)
            
            print(f"\n=== New Best Training Model Saved ===")
            print(f"Improvement: {improvement:.6f}")
            print(f"New best training loss: {best_loss:.6f}")
            print(f"Saved to: {best_model_path}")
        
        # 定期保存檢查點
        if ((epoch + 1) % 300 == 0) or (epoch == num_epochs - 1):
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'best_loss': best_loss,
                'best_val_loss': best_val_loss,
                'no_improve_count': no_improve_count,
                'epochs_record': epochs_record,
                'train_losses_record': train_losses_record,
                'val_losses_record': val_losses_record,
                'feature_losses_record': feature_losses_record,
                'voice_losses_record': voice_losses_record,
                'lr_values_record': lr_values_record
            }
            
            # 如果有排程器，也保存它的狀態
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\nCheckpoint saved to: {checkpoint_path}")
            
            # 每次保存檢查點時也保存學習曲線圖
            curve_path = os.path.join(save_dir, f'learning_curve_epoch_{epoch+1}.png')
            plot_learning_curves(
                epochs_record, train_losses_record, val_losses_record, 
                feature_losses_record, voice_losses_record, lr_values_record, 
                curve_path
            )
        
        # 每50個 epoch 繪製一次學習曲線
        if (epoch + 1) % 50 == 0:
            curve_path = os.path.join(save_dir, f'learning_curve_epoch_{epoch+1}.png')
            plot_learning_curves(
                epochs_record, train_losses_record, val_losses_record, 
                feature_losses_record, voice_losses_record, lr_values_record, 
                curve_path
            )
        
        # 每300個 epoch 以及最後一輪時保存樣本
        if (epoch + 1) % 300 == 0 or epoch == num_epochs - 1:
            print(f"\nSaving samples for epoch {epoch+1}...")
            # 保存當前 batch 的音頻樣本
            with torch.no_grad():
                for batch_idx, (input_wav, target_wav) in enumerate(train_loader):
                    input_wav = input_wav.to(device)
                    target_wav = target_wav.to(device)
                    
                    # 修改：保存完整的輸出元組，不僅僅是音頻輸出
                    output_tuple = model(input_wav)
                    
                    save_sample(
                        input_wav, output_tuple, target_wav,
                        epoch, batch_idx, save_dir, device, model
                    )
                    
                    # 只保存前幾個 batch 的樣本
                    if batch_idx >= 2:  # 只保存前2個batch
                        break
        
        # 在最後一個 epoch 生成 t-SNE 視覺化
        if epoch == num_epochs - 1:
            tsne_save_path = os.path.join(tsne_dir, f'tsne_epoch_{epoch+1}.png')
            plot_tsne_visualization(
                enhanced_features=enhanced_features,
                target_features=target_features,
                save_path=tsne_save_path
            )
    
    # 訓練結束時，再次保存整個學習曲線
    final_curve_path = os.path.join(save_dir, 'final_learning_curve.png')
    plot_learning_curves(
        epochs_record, train_losses_record, val_losses_record, 
        feature_losses_record, voice_losses_record, lr_values_record, 
        final_curve_path
    )
    
    return {
        'epochs': epochs_record,
        'train_losses': train_losses_record,
        'val_losses': val_losses_record,
        'feature_losses': feature_losses_record,
        'voice_losses': voice_losses_record,
        'lr_values': lr_values_record
    }

def worker_init_fn(worker_id):
    """初始化工作進程的隨機種子"""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # Default configuration values
    default_config = {
        'validation_strategy': 'random',
        'val_split': 0.2,
        'custom_val_split': False,
        'val_speakers': [],
        'val_materials': [],
        'test_samples': 5
    }
    
    # 添加分布式訓練參數
    parser = argparse.ArgumentParser(description="WavTokenizer訓練/特徵提取工具")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式訓練的local rank")
    parser.add_argument("--extract_only", action="store_true", help="僅提取encoder特徵，不訓練模型")
    parser.add_argument("--input_dir", type=str, default=None, help="輸入目錄，用於提取特徵")
    parser.add_argument("--save_dir", type=str, default=None, help="特徵保存目錄")
    parser.add_argument("--format", type=str, choices=["pt", "npy"], default="pt", help="特徵保存格式")
    args = parser.parse_args()

    # 初始化分布式環境
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    config = {
        'config_path': os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"),
        'model_path': os.path.join(os.getcwd(), "models", "wavtokenizer_large_speech_320_24k.ckpt"),        
        'save_dir': os.path.join(os.getcwd(), "results", "tsne_outputs", "tsne_output4"),
        'epochs': 1500,
        'batch_size': 8,             # 減小批次大小以節省記憶體
        'learning_rate': 0.003,      # 減小學習率
        'weight_decay': 0.001,
        'scheduler_patience': 5,
        'scheduler_factor': 0.99,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'feature_scale': 1.5,
        'num_workers': 2,            # 減少工作進程數
        'pin_memory': True,          
        'prefetch_factor': 2,       
        'enable_amp': True,          # 啟用自動混合精度訓練
        'grad_scaler': True,         # 啟用梯度縮放
        'T_0': 50,                  
        'T_mult': 2,               
        'eta_min': 1e-7,           
        'val_split': 0.2,
        'validation_strategy': 'speaker_only',
        'custom_val_split': True,    # 啟用自定義驗證集分割
        'val_speakers': ['girl9', 'boy7']  # 指定驗證集說話者為girl9和boy7
    }
    
    # 設定預設值
    if 'custom_val_split' not in config:
        config['custom_val_split'] = False
    
    # 設備配置
    device = torch.device(f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda')
    print(f"Using device: {device}")
    
    # GPU 記憶體配置優化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
    
    # 固定隨機種子，確保每次訓練的批次順序都相同
    set_seed(42)
    
    # 如果啟用了特徵提取模式
    if args.extract_only:
        if not args.input_dir:
            print("錯誤: 使用 --extract_only 模式時必須指定 --input_dir")
            return
        
        extract_dir = args.save_dir or os.path.join(os.getcwd(), "results", "features")
        os.makedirs(extract_dir, exist_ok=True)
        
        print(f"特徵提取模式啟動")
        print(f"輸入目錄: {args.input_dir}")
        print(f"輸出目錄: {extract_dir}")
        print(f"輸出格式: {args.format}")
        
        # 載入模型
        config_path = os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        model_path = os.path.join(os.getcwd(), "models", "wavtokenizer_large_speech_320_24k.ckpt")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        model = EnhancedWavTokenizer(config_path, model_path).to(device)
        model.eval()  # 設置為評估模式
        
        # 進行批次提取
        from extract_features import batch_extract_features
        results = batch_extract_features(
            model,
            args.input_dir,
            extract_dir,
            device,
            args.format
        )
        
        if results:
            print(f"\n處理完成，成功提取 {len(results)} 個特徵文件")
            print(f"特徵已保存到: {extract_dir}")
        
        return
    
    # 設備
    device = torch.device(f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda')
    print(f"Using device: {device}")
    
    # 打印詳細的 GPU 信息
    print_gpu_info()
    
    # GPU 記憶體設置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空 GPU 緩存
        # torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的 GPU 記憶體
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        # 為了確保結果的可重現性
        torch.backends.cudnn.deterministic = True  
    
    # 初始化模型
    model = EnhancedWavTokenizer(
        config['config_path'], 
        config['model_path']
    ).to(device)
    
    if args.local_rank != -1:
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    
    # 添加模型位置檢查
    print(f"\nModel device check:")
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")
    
    # 在訓練開始前添加測試張量檢查
    test_tensor = torch.randn(1, 1, 1000).to(device)
    print(f"Test tensor is on GPU: {test_tensor.is_cuda}")
    
    # 打印所有模型參數的設備位置
    print("\nModel parameters device check:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8,                   # 增加數值穩定性
        betas=(0.9, 0.999)          # 設置動量參數
    )
    
    # 學習率排程器設置 - 使用 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["min_lr"],
        verbose=True,
    )
    
    # 檢查是否有之前的訓練狀態
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
    if (os.path.exists(checkpoint_path)):
        print(f"\nLoading previous checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Don't load optimizer state if architecture changed
        
        # 如果存在排程器狀態，也載入它
        # if "scheduler_state_dict" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        #     print("Scheduler state loaded successfully")
    
    # 檢查目錄是否存在
    # Original input directories (commented out for reference)
    
    
    input_dirs = [
        os.path.join(os.getcwd(), "data", "raw", "box"),
        os.path.join(os.getcwd(), "data", "raw", "mac"),
        os.path.join(os.getcwd(), "data", "raw", "papercup"), 
        os.path.join(os.getcwd(), "data", "raw", "plastic") # plastic當作未知材質
    ]
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
    """

    # Modified to only use box directory
    input_dirs = [os.path.join(os.getcwd(), "data", "raw", "box")]
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
    """
    # Create full dataset
    dataset = AudioDataset(input_dirs=input_dirs, target_dir=target_dir, max_files_per_dir=None)
    
    # 不再過濾只保留5個樣本，而是使用完整的數據集
    print(f"Using all available samples: {len(dataset.paired_files)} paired files")
    
    # Initialize validation-related configurations with defaults
    validation_strategy = config.get('validation_strategy', 'random')
    val_split = config.get('val_split', 0.2)
    custom_val_split = config.get('custom_val_split', False)
    val_speakers = config.get('val_speakers', [])
    val_materials = config.get('val_materials', [])
    
    # 數據集大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 不再建立只有5個樣本的測試數據集
    # 依照驗證策略將數據分為訓練集和驗證集
    
    # 根據配置決定是使用自定義驗證集還是隨機拆分
    if config.get('custom_val_split', False):  # Using get() with default value
        # 選擇驗證策略
        validation_strategy = config.get('validation_strategy', 'speaker_only')
        print(f"Using validation strategy: {validation_strategy}")
        
        # 準備訓練和驗證索引
        train_indices = []
        val_indices = []
        
        # 統計不同材質和說話者的數量（用於分層抽樣）
        material_counts = {}
        speaker_counts = {}
        for i, pair in enumerate(dataset.paired_files):
            material = pair['material']
            speaker = pair['speaker']
            
            if material not in material_counts:
                material_counts[material] = []
            if speaker not in speaker_counts:
                speaker_counts[speaker] = []
                
            material_counts[material].append(i)
            speaker_counts[speaker].append(i)
        
        # 打印材質和說話者的分布情況
        print("\nDataset distribution:")
        print(f"Materials: {len(material_counts)} types")
        for material, indices in material_counts.items():
            print(f"  - {material}: {len(indices)} samples")
            
        print(f"Speakers: {len(speaker_counts)} persons")
        for speaker, indices in speaker_counts.items():
            print(f"  - {speaker}: {len(indices)} samples")
        
        # 根據不同的驗證策略進行數據分割
        if validation_strategy == 'speaker_only':
            # 僅基於說話者名稱的驗證集
            for i, pair in enumerate(dataset.paired_files):
                input_path = os.path.join(pair['input_dir'], pair['input'])
                is_val_sample = any(speaker in input_path for speaker in config['val_speakers'])
                
                if is_val_sample:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
                    
        elif validation_strategy == 'stratified':
            # 分層抽樣：確保每個說話者和每種材質都在驗證集中有代表
            seen_combinations = set()
            
            # 首先，確保指定的驗證說話者樣本在驗證集中
            for i, pair in enumerate(dataset.paired_files):
                # 從資料夾路徑中提取材質名稱 (box, mac, papercup, plastic 等)
                material_path = pair['input_dir']
                material = os.path.basename(material_path)  # 獲取資料夾名稱作為材質
                
                speaker = pair['speaker']
                combo = f"{speaker}_{material}"
                
                # 如果是指定的驗證說話者
                if speaker in config['val_speakers']:
                    # 確保每種材質+說話者組合只取一定比例
                    if combo not in seen_combinations:
                        val_indices.append(i)
                        seen_combinations.add(combo)
                    else:
                        # 如果已經有相同組合，則有80%的機率進入訓練集
                        if random.random() > 0.8:
                            val_indices.append(i)
                        else:
                            train_indices.append(i)
                else:
                    # 對於非指定驗證說話者，有10%的機率進入驗證集以增加多樣性
                    if random.random() < 0.1:
                        val_indices.append(i)
                    else:
                        train_indices.append(i)
                        
        elif validation_strategy == 'balanced':
            # 平衡驗證集：確保各種材質和說話者的組合都有足夠的代表性
            material_val_quota = {m: max(5, len(indices)//10) for m, indices in material_counts.items()}
            speaker_val_quota = {s: max(5, len(indices)//10) for s, indices in speaker_counts.items()}
            
            material_val_count = {m: 0 for m in material_counts}
            speaker_val_count = {s: 0 for s in speaker_counts}
            
            # 根據配額進行分配
            for i, pair in enumerate(dataset.paired_files):
                # 從資料夾路徑中提取材質名稱
                material_path = pair['input_dir']
                material = os.path.basename(material_path)  # 獲取資料夾名稱作為材質
                
                speaker = pair['speaker']
                
                # 優先確保指定的驗證說話者樣本進入驗證集
                if speaker in config['val_speakers'] and speaker_val_count[speaker] < speaker_val_quota[speaker]:
                    val_indices.append(i)
                    material_val_count[material] += 1
                    speaker_val_count[speaker] += 1
                # 其次確保所有材質類型都有足夠樣本
                elif material in config['val_materials'] and material_val_count[material] < material_val_quota[material]:
                    val_indices.append(i)
                    material_val_count[material] += 1
                    speaker_val_count[speaker] += 1
                # 對於其他樣本，確保驗證集總大小不超過總數據的25%
                elif len(val_indices) < len(dataset.paired_files) * 0.25 and random.random() < 0.15:
                    val_indices.append(i)
                    material_val_count[material] += 1
                    speaker_val_count[speaker] += 1
                else:
                    train_indices.append(i)
        else:
            # 默認回退到隨機分割
            print(f"Unknown validation strategy: {validation_strategy}")
            print("Falling back to random split...")
            config['custom_val_split'] = False
        
        # 只有在成功使用了自定義分割策略時，才創建訓練集和驗證集
        if config['custom_val_split'] and val_indices:
            from torch.utils.data import Subset
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            
            # 檢查驗證集是否合理
            if val_size < 20 or val_size > len(dataset) * 0.4:  # 驗證集太小或太大
                print(f"Warning: Validation set size is {val_size}, which may be too {'small' if val_size < 20 else 'large'}!")
                print("Falling back to random split...")
                config['custom_val_split'] = False
            else:
                print(f"\nCustom split dataset using strategy '{validation_strategy}'")
                print(f"Training samples: {train_size}")
                print(f"Validation samples: {val_size}")
                
                # 分析驗證集中的說話者和材質分布
                val_speakers = set()
                val_materials = set()
                
                for idx in val_indices:
                    pair = dataset.paired_files[idx]
                    val_speakers.add(pair['speaker'])
                    val_materials.add(pair['material'])
                
                print(f"Validation set contains {len(val_speakers)} speakers: {', '.join(sorted(val_speakers))}")
                print(f"Validation set contains {len(val_materials)} materials: {', '.join(sorted(val_materials))}")
                
                # 如果特定的驗證說話者沒有出現在驗證集中，顯示警告
                missing_speakers = [s for s in config['val_speakers'] if s not in val_speakers]
                if missing_speakers:
                    print(f"Warning: The following specified validation speakers were not found in the dataset: {', '.join(missing_speakers)}")
        else:
            # 如果驗證集為空，回退到隨機拆分
            print("Validation set is empty. Falling back to random split...")
            config['custom_val_split'] = False
    
    # 如果不使用自定義拆分或者自定義拆分失敗，則使用隨機拆分
    if not config['custom_val_split']:
        # 隨機拆分數據集
        dataset_size = len(dataset)
        val_size = int(dataset_size * config['val_split'])
        train_size = dataset_size - val_size
        
        # 使用 random_split 函數來創建訓練集和驗證集
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定隨機種子
        )
        
        print(f"\nRandom split dataset into {train_size} training samples ({train_size/dataset_size:.1%}) and {val_size} validation samples ({val_size/dataset_size:.1%})")
    
    # 創建分布式採樣器和數據加載器
    train_sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.local_rank != -1 else None

    # 使用固定的隨機種子創建訓練數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    print("Train DataLoader initialized successfully")
    
    # 創建驗證數據加載器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    print("Validation DataLoader initialized successfully")
    
    # 打印一些驗證集樣本的文件名，以便確認
    if config['custom_val_split'] and val_size > 0:
            input_path = os.path.join(pair['input_dir'], pair['input'])
            print(f"  {input_path}")
        
    print(f"\nStarting training for {config['epochs']} epochs")
    print(f"Saving outputs to: {config['save_dir']}")
    print(f"Using fixed random seed: 42")
    print(f"Using ReduceLROnPlateau scheduler with patience={config['scheduler_patience']}, factor={config['scheduler_factor']}")
    print(f"Training with {train_size} samples, validating with {val_size} samples")

    # 開始訓練，現在包含驗證集
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config['save_dir'],
        config=config,
        num_epochs=config['epochs'],
        scheduler=scheduler,
        val_loader=val_loader  # 添加驗證加載器
    )

def evaluate_model_performance(model, test_loader, device):
    """評估模型性能，計算多個指標"""
    model.eval()
    metrics = {
        'snr': [],
        'spectral_distance': [],
        'env_similarity': []
    }
    
    with torch.no_grad():
        for input_wav, target_wav in test_loader:
            input_wav = input_wav.to(device)
            target_wav = target_wav.to(device)
            
            # 生成輸出
            output = model(input_wav)
            
            # 計算各種指標
            # Placeholder for evaluate_audio_quality function
            # Define or import the function to compute metrics
            batch_metrics = {
                'snr': torch.tensor([0.0]),  # Replace with actual SNR computation
                'spectral_distance': torch.tensor([0.0]),  # Replace with actual spectral distance computation
                'env_similarity': torch.tensor([0.0])  # Replace with actual envelope similarity computation
            }
            
            # 收集指標
            for key in metrics:
                metrics[key].append(batch_metrics[key].mean().item())
    
    # 計算平均值
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    
    return avg_metrics

# 注釋掉可視化相關函數
"""
def visualize_enhanced_tsne(features, labels=None, perplexity=30, n_iter=1000, save_path=None):
    # 原函數內容已注釋
    pass

def visualize_tsne_enhanced(features, labels=None, save_path=None, title=None, 
                       perplexity=30, n_components=2, n_iter=2000):
    # 原函數內容已注釋
    pass

def visualize_feature_importance(features, labels=None, method='variance', n_components=10, save_path=None):
    # 原函數內容已注釋
    pass

def analyze_temporal_features(features, window_size=100, stride=50, save_path=None):
    # 原函數內容已注釋
    pass

def analyze_feature_stability(features, n_runs=10, perplexity=30, n_iter=1000):
    # 原函數內容已注釋
    pass
"""

if __name__ == "__main__":
    main()
