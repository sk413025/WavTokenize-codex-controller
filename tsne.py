# Set matplotlib backend to non-interactive backend to avoid thread issues
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't use Tkinter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from decoder.pretrained import WavTokenizer

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
    try :
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
            # 計算頻譜圖
            spec = transform(audio)
            spec_db = amplitude_to_db(spec)
            
            # 移到 CPU 進行繪圖
            spec_db = spec_db.cpu()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db.squeeze().numpy(), cmap='viridis', origin='lower', aspect='auto')
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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, config_path, model_path, num_residual_blocks=2):
        super().__init__()
        # 1. 首先載入預訓練的WavTokenizer模型
        base_model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.encodec = base_model.feature_extractor.encodec
        encoder_dim = 512

        # 2. 凍結encoder和decoder - 保持使用預訓練權重
        for param in self.encodec.encoder.parameters():
            param.requires_grad = False
            
        # 凍結decoder
        for param in self.encodec.decoder.parameters():
            param.requires_grad = False
            
        # 特徵增強層設置為可訓練
        self.adapter_conv = nn.Conv1d(encoder_dim, 256, kernel_size=1)
        self.adapter_bn = nn.BatchNorm1d(256)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])
        self.out_conv = nn.Conv1d(256, encoder_dim, kernel_size=1)
        self.relu = nn.ReLU()
        
        # 確保特徵增強層為可訓練狀態
        for module in [self.adapter_conv, self.adapter_bn, self.residual_blocks, self.out_conv]:
            for param in module.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        # 使用 no_grad 進行編碼
        with torch.no_grad():
            features = self.encodec.encoder(x)
        
        # 以下層進行訓練
        features = self.adapter_conv(features)
        features = self.adapter_bn(features)
        features = self.relu(features)
        features = self.residual_blocks(features)
        features = self.out_conv(features)
        features = self.relu(features)
        
        return features

class EnhancedWavTokenizer(nn.Module):
    def __init__(self, config_path, model_path):
        super().__init__()
        # 使用新的特徵增強器
        self.feature_extractor = EnhancedFeatureExtractor(config_path, model_path)
    
    def forward(self, x):
        # 1. 特徵提取和增強
        enhanced_features = self.feature_extractor(x)
        
        # 2. 解碼
        output = self.feature_extractor.encodec.decoder(enhanced_features)
        
        # 3. 返回結果
        input_features = self.feature_extractor.encodec.encoder(x)
        return output, input_features, enhanced_features

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
    
   # return 0.5 * similarity_loss + 0.5 * distance_loss
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
    """特徵空間損失計算，完全跳過 voice_loss 計算以節省時間"""
    # 1. 特徵空間損失 (tsne.py 原有的特徵比較)
    feature_loss = compute_feature_loss(enhanced_features, target_features, device)
    
    # 2. 直接使用特徵損失作為總損失，完全跳過 voice_loss 計算
    total_loss = feature_loss
    
    # 仍然返回一個包含兩種損失的字典，但 voice_loss 始終為 0
    return total_loss, {
        'feature_loss': feature_loss.item(),
        'voice_loss': 0.0  # 不計算 voice_loss，直接設為 0
    }

def save_sample(input_wav, output, target_wav, epoch, batch_idx, save_dir, device):
    """保存音頻樣本和頻譜圖"""
    try:
        audio_dir = os.path.join(save_dir, f'epoch_{epoch+1}_samples')
        os.makedirs(audio_dir, exist_ok=True)
        
        for j in range(output.size(0)):
            try:
                with torch.no_grad():
                    # 全部在 GPU 上進行正規化
                    output_audio = output[j] / (torch.max(torch.abs(output[j])) + 1e-8)
                    input_audio = input_wav[j] / (torch.max(torch.abs(input_wav[j])) + 1e-8)
                    target_audio = target_wav[j] / (torch.max(torch.abs(target_wav[j])) + 1e-8)
                    
                    # 重塑形狀
                    output_audio = output_audio.reshape(1, -1)
                    input_audio = input_audio.reshape(1, -1)
                    target_audio = target_audio.reshape(1, -1)
                    
                    # 基礎檔名
                    base_name = f"batch_{batch_idx}_sample_{j+1}"
                    
                    # 處理並保存每個音頻
                    for audio, prefix in [
                        (output_audio, 'output'),
                        (input_audio, 'input'),
                        (target_audio, 'target')
                    ]:
                        # 保存音頻文件
                        audio_path = os.path.join(audio_dir, f'{base_name}_{prefix}.wav')
                        # 只在保存時移至 CPU
                        torchaudio.save(audio_path, audio.cpu(), 24000)
                        
                        # 生成頻譜圖（保持在 GPU 上）
                        spec_path = os.path.join(audio_dir, f'{base_name}_{prefix}_spec.png')
                        # 確保音頻在正確的設備上
                        plot_spectrograms(
                            audio.to(device),  # 明確指定設備
                            spec_path,
                            device,
                            title=f'Epoch {epoch+1} {prefix.capitalize()} Spectrogram'
                        )
                
                print(f"Saved sample {j+1} from batch {batch_idx}")
                
            except Exception as e:
                print(f"Error saving sample {j+1} from batch {batch_idx}: {str(e)}")
                print(f"Device states - Output: {output.device}, Input: {input_wav.device}, Target: {target_wav.device}")
                continue
                
    except Exception as e:
        print(f"Error in save_sample function: {str(e)}")
        return False

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
    # 驗證損失曲線
    line2 = ax1.plot(epochs, val_losses, 'g-', label='Val Loss')
    # 特徵損失曲線
    line3 = ax1.plot(epochs, feature_losses, 'c-', label='Feature Loss')
    # 人聲損失曲線
    line4 = ax1.plot(epochs, voice_losses, 'm-', label='Voice Loss')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 學習率曲線
    line5 = ax2.plot(epochs, lr_values, 'r-', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合併圖例
    lines = line1 + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels)
    
    plt.title('Training and Validation Losses and Learning Rate Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Learning curves saved to {save_path}")

def train_model(model, train_loader, optimizer, device, save_dir, num_epochs=100, scheduler=None, val_loader=None):
    """修改訓練函數以記錄和繪製訓練指標，包括驗證損失"""
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        best_loss = checkpoint['loss']
        print(f"\nLoaded previous best loss: {best_loss}")
    else:
        best_loss = float('inf') # 初始化最佳損失
    no_improve_count = 0      # 添加計數器
    
    print(f"\nInitial best loss: {best_loss}")  # 添加日誌
    
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
            
            # 1. 輸入通過整個模型處理
            output, input_features, enhanced_features = model(input_wav)
            # output 是 decoder 的輸出
            # input_features 是 encoder 的輸出
            # enhanced_features 是特徵增強層的輸出

            # 2. 獲取目標音頻的特徵 (通過 encoder)
            with torch.no_grad():
                target_features = model.feature_extractor.encodec.encoder(target_wav)

            # 3. 計算混合損失
            loss, loss_details = compute_hybrid_loss(
                output,              # decoder 輸出的波形
                target_wav,         # 目標波形
                enhanced_features,   # 增強後的特徵 (decoder 輸入)
                target_features,    # 目標音頻的特徵
                device
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
            
            # 更新進度條，添加所有損失信息和學習率
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'feat': f'{loss_details["feature_loss"]:.4f}',
                'voice': f'{loss_details["voice_loss"]:.4f}',
                'lr': f'{current_lr:.6f}',
                'GPU_MB': f'{allocated:.0f}/{cached:.0f}'
            })
        
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
                    
                    # 確保輸入數據的幅度合適，使用相對標準化方法
                    input_wav = (input_wav - input_wav.mean()) / (input_wav.std() + 1e-8)
                    target_wav = (target_wav - target_wav.mean()) / (target_wav.std() + 1e-8)
                    
                    # 前向傳播
                    output, input_features, enhanced_features = model(input_wav)
                    target_features = model.feature_extractor.encodec.encoder(target_wav)
                    
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
        else:
            # 如果沒有驗證集，添加None以保持列表長度一致
            val_losses_record.append(None)
        
        # 如果使用學習率排程器，在每個epoch後更新學習率
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_train_loss)  # 使用訓練損失來調整學習率
            else:
                scheduler.step()
            print(f"Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 添加詳細的保存邏輯日誌
        print(f"Current train loss: {avg_train_loss:.4f}, Best loss: {best_loss:.4f}")
        
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
            
            print(f"\n=== New Best Model Saved ===")
            print(f"Improvement: {improvement:.6f}")
            print(f"New best loss: {best_loss:.6f}")
            print(f"Saved to: {best_model_path}")
            
            no_improve_count = 0  # 重置計數器
        else:
            no_improve_count += 1
            print(f"\nNo improvement for {no_improve_count} epochs")
        
        # 定期保存檢查點
        if ((epoch + 1) % 300 == 0) or (epoch == num_epochs - 1):
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'best_loss': best_loss,
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
                    
                    output, _, _ = model(input_wav)
                    
                    save_sample(
                        input_wav, output, target_wav,
                        epoch, batch_idx, save_dir, device
                    )
                    
                    # 只保存前幾個 batch 的樣本
                    if batch_idx >= 2:  # 只保存前2個batch
                        break
    
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
    # 固定隨機種子，確保每次訓練的批次順序都相同
    set_seed(42)
    
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",        
        'save_dir': './tout2',       # 所有訓練輸出都會儲存在這裡
        'epochs': 2000,
        'batch_size': 4,            # 修改為和 try3.py 一樣
        'learning_rate': 3e-2,      # 修改為和 try3.py 一樣
        'weight_decay': 0.001,
        'scheduler_patience': 5,
        'scheduler_factor': 0.7,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'feature_scale': 1.5,
        'tsne_weight': 0.2,          # T-SNE 損失的權重
        'num_workers': 4,           # 增加數據加載的工作進程
        'pin_memory': True,         # 啟用 pin_memory
        'prefetch_factor': 2,       # 預加載因子
        'T_0': 50,                  # CosineAnnealingWarmRestarts 的週期長度
        'T_mult': 2,                # 每個周期後週期長度的倍數
        'eta_min': 1e-7,            # 最小學習率
        'val_split': 0.2,           # 驗證集比例 (如果使用隨機拆分)
        'val_speakers': ['boy7', 'girl9'],  # 增加更多說話者到驗證集
        'val_materials': ['box', 'plastic', 'papercup'],  # 確保驗證集包含所有材質
        'validation_strategy': 'stratified',  # 'speaker_only', 'stratified', 'random'
        'custom_val_split': True,   # 是否使用自定義驗證集拆分
    }
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 打印詳細的 GPU 信息
    print_gpu_info()
    
    # GPU 記憶體設置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空 GPU 緩存
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的 GPU 記憶體
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
    
    # 學習率排程器設置 - 使用 CosineAnnealingWarmRestarts
    # 此排程器具有週期性學習率調整，適合長期訓練
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],          # 首個週期的長度
        T_mult=config['T_mult'],    # 每次重啟後週期長度的乘數
        eta_min=config['eta_min']   # 最小學習率
    )
    
    # 檢查是否有之前的訓練狀態
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
    if (os.path.exists(checkpoint_path)):
        print(f"\nLoading previous checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 如果存在排程器狀態，也載入它
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully")
    
    # 數據加載器設置
    try:
        print("\nInitializing dataset...")
        from try3 import AudioDataset, collate_fn
        
        # 檢查目錄是否存在
        input_dirs = ["./box", "./plastic", "./papercup"]
        target_dir = "./box2"
        
        for dir_path in input_dirs + [target_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            print(f"Found directory: {dir_path}")
        
        # 顯示每個目錄中的文件數量
        for dir_path in input_dirs + [target_dir]:
            files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            print(f"Files in {dir_path}: {len(files)}")
        
        # 創建數據集
        dataset = AudioDataset(
            input_dirs=input_dirs,
            target_dir=target_dir
        )
        print(f"Dataset initialized with {len(dataset)} samples")
        
        # 根據配置決定是使用自定義驗證集還是隨機拆分
        if config['custom_val_split']:
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
                    speaker = pair['speaker']
                    material = pair['material']
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
                    material = pair['material']
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
                    print(f"Validation samples: {val_size} ({val_size/len(dataset):.1%} of total data)")
                    
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
        
        # 使用固定的隨機種子創建訓練數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
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
            shuffle=False,  # 驗證集不需要打亂順序
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
            print("\nValidation set sample filenames:")
            for i in range(min(5, len(val_indices))):  # 顯示前5個樣本
                pair = dataset.paired_files[val_indices[i]]
                input_path = os.path.join(pair['input_dir'], pair['input'])
                print(f"  {input_path}")
            
    except Exception as e:
        print(f"Error initializing dataset: {str(e)}")
        raise
    
    print(f"\nStarting training for {config['epochs']} epochs")
    print(f"Saving outputs to: {config['save_dir']}")
    print(f"Using fixed random seed: 42")
    print(f"Using CosineAnnealingWarmRestarts scheduler with T_0={config['T_0']}, T_mult={config['T_mult']}")
    print(f"Training with {train_size} samples, validating with {val_size} samples")

    # 開始訓練，現在包含驗證集
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config['save_dir'],
        num_epochs=config['epochs'],
        scheduler=scheduler,
        val_loader=val_loader  # 添加驗證加載器
    )

if __name__ == "__main__":
    main()
