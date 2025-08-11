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
import traceback
import datetime
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
from sklearn.manifold import TSNE  # 添加 t-SNE 導入
# Add the WavTokenizer directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset  # Add this import
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
        
        # 凍結decoder (已修改)
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
        # 更健壯的形狀檢查和調整
        if x.dim() == 4:  # 如果是 [batch_size, channels, 1, time] (多了一個維度)
            # 檢測到4D張量，自動調整為3D
            x = x.squeeze(2)  # 移除多餘的維度，變為 [batch_size, channels, time]
            print(f"已將 4D 張量調整為 3D: {x.shape}")
        elif x.dim() == 2:  # 如果是 [batch_size, time]
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
                # 檢查bandwidth_id形狀和類型
                print(f"Debug - bandwidth_id before encode_infer: shape={bandwidth_id.shape}, dtype={bandwidth_id.dtype}, value={bandwidth_id}")
                
                # 確保bandwidth_id是單一整數（如果batch_size為1）
                if bandwidth_id.shape[0] == 1:
                    bandwidth_id = bandwidth_id.item()  # 轉換為Python整數
                    print(f"Debug - converted bandwidth_id to item: {bandwidth_id}, type={type(bandwidth_id)}")
                
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
        
        # 存儲每一層的中間特徵
        intermediate_features_list = []
        
        # 應用殘差塊，並在每次應用後保存中間特徵
        layer_enhanced = enhanced.clone()
        for block in self.residual_blocks:
            layer_enhanced = block(layer_enhanced)
            # 保存每一層的特徵
            intermediate_features_list.append(layer_enhanced.clone())
        
        # 使用最後一層的特徵作為中間特徵輸出 (向後兼容)
        intermediate_enhanced_features = layer_enhanced.clone()
        
        # 恢復原始維度
        enhanced = self.up_conv(layer_enhanced)
        
        # 最終歸一化
        enhanced = enhanced.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        enhanced = self.output_norm(enhanced)
        enhanced = enhanced.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # 確保增强特徵形狀標準化
        enhanced = self.ensure_feature_shape(enhanced)
        
        # 通過解碼器
        decoded = self.encodec.decoder(enhanced)
        
        return decoded, input_features, enhanced, discrete_code, intermediate_enhanced_features, intermediate_features_list

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
        output, input_features, enhanced_features, discrete_code, intermediate_enhanced_features, intermediate_features_list = self.feature_extractor(x, bandwidth_id)
            
        return output, input_features, enhanced_features, discrete_code, intermediate_enhanced_features, intermediate_features_list

def compute_feature_loss(enhanced_features, target_features, device, loss_type='l2', use_cosine_sim=False):
    """
    統一的特徵損失計算函數，僅使用 L2 距離損失
    
    Args:
        enhanced_features: 模型生成的增強特徵，形狀為 [batch_size, channels, time]
        target_features: 目標特徵，形狀為 [batch_size, channels, time]
        device: 計算設備
        loss_type: 保留參數，不再使用
        use_cosine_sim: 保留參數，不再使用
        
    Returns:
        torch.Tensor: 計算出的 L2 距離損失值
    """
    # 確保特徵維度正確
    if enhanced_features.dim() == 2:
        enhanced_features = enhanced_features.unsqueeze(0)
    if target_features.dim() == 2:
        target_features = target_features.unsqueeze(0)
        
    # 計算 L2 距離損失 - 只使用 L2 距離
    l2_dist = torch.norm(enhanced_features - target_features, dim=1)
    distance_loss = l2_dist.mean()  # 最小化距離
    
    # 直接返回 L2 距離損失
    return distance_loss
        
# 為了向後兼容性保留這個函數
def compute_l2_only_loss(enhanced_features, target_features, device):
    """
    只使用L2損失的特徵空間損失函數 (向後兼容)
    用於--tsne_flow_with_L2模式，與tsne.py兼容
    
    Args:
        enhanced_features: 模型生成的增強特徵
        target_features: 目標特徵
        device: 計算設備
        
    Returns:
        torch.Tensor: L2損失值
    """
    return compute_feature_loss(enhanced_features, target_features, device, loss_type='l2', use_cosine_sim=False)

# 修改 compute_hybrid_loss 函數
def compute_hybrid_loss(output, target_wav, enhanced_features, target_features, device, loss_type='l2', use_cosine_sim=False):
    """
    計算混合損失，主要關注特徵空間的對齊
    
    Args:
        output: 模型輸出的音頻波形
        target_wav: 目標音頻波形
        enhanced_features: 增強後的特徵
        target_features: 目標特徵
        device: 計算設備
        loss_type: 保留參數，不再使用
        use_cosine_sim: 保留參數，不再使用
        
    Returns:
        tuple: (損失值, 損失詳情字典)
    """
    # 計算特徵空間的損失，使用統一的compute_feature_loss
    feature_loss = compute_feature_loss(enhanced_features, target_features, device)
    
    return feature_loss, {
        'feature_loss': feature_loss.item(),
        'voice_loss': 0.0  # 不使用語音重建損失，專注於特徵對齊
    }

def compute_hybrid_loss_with_content(output, target_wav, enhanced_features, target_features, 
                           intermediate_enhanced_features, content_ids, device, alpha=0.01, beta=0.99):
    """
    計算混合損失，包含特徵損失和內容一致性損失
    
    Args:
        output (torch.Tensor): 模型輸出的音頻波形
        target_wav (torch.Tensor): 目標波形
        enhanced_features (torch.Tensor): 增強後的特徵
        target_features (torch.Tensor): 目標特徵
        intermediate_enhanced_features (torch.Tensor 或 list): 中間特徵或中間特徵列表（用於內容一致性）
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備
        alpha (float): 內容一致性損失權重
        beta (float): 特徵損失權重
        
    Returns:
        tuple: (total_loss, loss_details)，其中loss_details是包含各損失組件的字典
    """
    # 確保中間特徵有正確的維度
    if isinstance(intermediate_enhanced_features, list):
        # 如果是特徵列表，使用第二層特徵（索引1）計算內容一致性損失
        if len(intermediate_enhanced_features) > 1:
            # 明確使用第二層特徵（索引1）用於內容一致性損失
            second_layer_features = intermediate_enhanced_features[1]
            if second_layer_features.dim() == 2:
                second_layer_features = second_layer_features.unsqueeze(0)
            content_consistency_loss = compute_content_consistency_loss(second_layer_features, content_ids, device)
        else:
            content_consistency_loss = torch.tensor(0.0, device=device)
    else:
        # 若傳入單一 tensor，則直接用
        if intermediate_enhanced_features.dim() == 2:
            intermediate_enhanced_features = intermediate_enhanced_features.unsqueeze(0)
        content_consistency_loss = compute_content_consistency_loss(intermediate_enhanced_features, content_ids, device)
    
    # 計算特徵空間的L2損失（僅在最終輸出特徵上）
    feature_loss = compute_feature_loss(enhanced_features, target_features, device)

    # 計算總損失
    total_loss = alpha * content_consistency_loss + beta * feature_loss

    # 返回總損失和詳細損失組件
    return total_loss, {
        'feature_loss': feature_loss.item(),
        'content_consistency_loss': content_consistency_loss.item(),
        'voice_loss': 0.0,  # 確保包含此鍵以滿足 train_model 函數的需求
        'total_loss': total_loss.item()
    }

# 修改為與新的compute_hybrid_loss_with_content保持一致，使用第二層內容一致性損失和最終層L2損失
def compute_hybrid_loss_with_tsne_flow(output, target_wav, enhanced_features, target_features, 
                           intermediate_enhanced_features, content_ids, device, alpha=0.01, beta=0.99):
    """
    計算混合損失，按照tsne.py的處理流程，但同時納入內容一致性損失
    
    修改版的函數實現，按照要求：
    1. 內容一致性損失僅在第二層應用
    2. L2損失僅在最終層（進入decoder前）應用
    3. 中間層自由學習
    
    Args:
        output (torch.Tensor): 模型輸出的音頻波形
        target_wav (torch.Tensor): 目標波形
        enhanced_features (torch.Tensor): 增強後的特徵
        target_features (torch.Tensor): 目標特徵
        intermediate_enhanced_features (torch.Tensor或list): 中間特徵或中間特徵列表（用於內容一致性）
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備
        alpha (float): 內容一致性損失權重
        beta (float): 特徵L2損失權重
        
    Returns:
        tuple: (total_loss, loss_details)，其中loss_details是包含各損失組件的字典
    """
    return compute_hybrid_loss_with_content(
        output, target_wav, enhanced_features, target_features, 
        intermediate_enhanced_features, content_ids, device, 
        alpha=alpha, beta=beta
    )

def save_sample(input_wav, output, target_wav, epoch, batch_idx, save_dir, device, model=None):
    """保存音頻樣本和頻譜圖，與tsne.py相同的方法，修正相對路徑和音頻輸出
    
    Args:
        input_wav: 輸入音頻
        output: 模型輸出（可能是元組）
        target_wav: 目標音頻
        epoch: 當前訓練週期
        batch_idx: 批次索引
        save_dir: 保存目錄
        device: 計算設備
        model: 模型實例（可選）
    """
    try:
        # 建立音頻保存目錄 (使用固定的audio_samples目錄)
        audio_dir = os.path.join(save_dir, "audio_samples", f'epoch_{epoch+1}')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 使用絕對路徑載入WavTokenizer decoder模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        model_path = os.path.join(current_dir, "wavtokenizer_large_speech_320_24k.ckpt")
        
        # 初始化WavTokenizer模型作為解碼器
        try:
            from decoder.pretrained import WavTokenizer
            import traceback
            from encoder.utils import save_audio  # 導入 save_audio 工具函數
            
            decoder = WavTokenizer.from_pretrained0802(config_path, model_path)
            decoder = decoder.to(device)
            decoder.eval()  # 確保模型處於評估模式
            print(f"✅ 成功載入WavTokenizer解碼器")
            
            # 設定metadata中的bandwidth_id
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
                        
                        # 檢查output是否是tuple並提取增強特徵
                        if isinstance(output, tuple) and len(output) >= 3:
                            if output[2] is not None and j < output[2].size(0):
                                enhanced_features = output[2][j:j+1]
                                print(f"從元組中提取增強特徵，形狀: {enhanced_features.shape}")
                                
                                # 確保特徵形狀正確
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
                                    output_tuple = model(temp_input)
                                    
                                    if isinstance(output_tuple, tuple) and len(output_tuple) >= 3:
                                        new_enhanced_features = output_tuple[2]
                                        
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
                        
                        # 處理音頻輸出
                        if enhanced_features is None or enhanced_features.shape[1] != 512:
                            print("⚠️ 無法獲得有效的特徵張量，直接使用模型輸出音頻")
                            # 從模型輸出提取音頻
                            if isinstance(output, tuple) and len(output) > 0:
                                output_audio = output[0][j:j+1] if j < output[0].size(0) else input_audio
                            else:
                                output_audio = output[j:j+1] if j < output.size(0) else input_audio
                            
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
                            # 保存音頻文件，使用 encoder.utils 中的 save_audio 函數
                            audio_path = os.path.join(audio_dir, f"{base_name}_{prefix}.wav")
                            try:
                                save_audio(audio.cpu(), audio_path, sample_rate=24000, rescale=True)
                                print(f"🔊 保存{prefix}音頻到: {audio_path}")
                            except Exception as save_err:
                                print(f"❌ 保存音頻失敗，嘗試使用torchaudio: {str(save_err)}")
                                # 備用方案：使用torchaudio直接保存
                                torchaudio.save(audio_path, audio.cpu(), 24000)
                                print(f"🔊 使用torchaudio保存{prefix}音頻到: {audio_path}")
                            
                            # 生成頻譜圖
                            spec_path = audio_path.replace(".wav", "_spec.png")
                            try:
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
                            except Exception as spec_err:
                                print(f"❌ 生成頻譜圖時出錯: {str(spec_err)}")
                                traceback.print_exc()
                            
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

def collate_fn(batch, trim_to_shortest=True):
    """處理批次數據，支持內容ID
    
    Args:
        batch: 資料批次
        trim_to_shortest: 是否將音訊修剪到最短長度，預設為True與tsne.py保持一致
    """
    input_wavs = [item[0] for item in batch]
    target_wavs = [item[1] for item in batch]
    
    # 獲取內容ID (如果提供)
    content_ids = None
    if len(batch[0]) > 2:  # 檢查是否有內容ID
        content_ids = [item[2] for item in batch]
    
    # 找出最短的音訊長度 (與tsne.py保持一致的處理方式)
    min_len = min(
        min(wav.size(-1) for wav in input_wavs),
        min(wav.size(-1) for wav in target_wavs)
    )
    
    # 對齊長度 - 始終裁剪到最短長度，與tsne.py一致
    input_wavs = [wav[..., :min_len] for wav in input_wavs]
    target_wavs = [wav[..., :min_len] for wav in target_wavs]
    
    # 返回堆疊的張量和內容ID (如果有)
    if content_ids is not None:
        return torch.stack(input_wavs), torch.stack(target_wavs), content_ids
    else:
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

def plot_learning_curves(epochs, train_losses, val_losses, feature_losses, voice_losses, lr_values, save_path, content_losses=None):
    """繪製訓練和驗證過程中的損失和學習率曲線，包含內容一致性損失"""
    plt.figure(figsize=(14, 8))
    
    # 創建兩個Y軸
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 訓練損失曲線
    line1 = ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    # 特徵損失曲線
    line2 = ax1.plot(epochs, feature_losses, 'c-', linewidth=2, label='Feature Loss')
    # 驗證損失曲線
    line3 = ax1.plot(epochs, val_losses, 'g-', linewidth=2, label='Val Loss')
    
    # 內容一致性損失曲線 (如果提供)
    lines_content = []
    if content_losses is not None and len(content_losses) > 0:
        line_content = ax1.plot(epochs, content_losses, 'm--', linewidth=2, label='Content Consistency Loss')
        lines_content = line_content
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 學習率曲線
    line4 = ax2.plot(epochs, lr_values, 'r-', linewidth=1.5, label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合併圖例
    lines = line1 + line2 + line3 + lines_content + line4 
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, fontsize=11, loc='upper right')
    
    plt.title('Training Losses and Learning Rate Over Time', fontsize=14)
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
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 處理原始特徵
    enhanced_features_np = enhanced_features.detach().cpu().numpy() if isinstance(enhanced_features, torch.Tensor) else enhanced_features.copy()
    target_features_np = target_features.detach().cpu().numpy() if isinstance(target_features, torch.Tensor) else target_features.copy()
    
    # 打印原始形状信息以便调试
    print(f"原始特徵形狀: enhanced={enhanced_features_np.shape}, target={target_features_np.shape}")
    
    # 確保特徵形狀是 [N, D] 兩維
    if enhanced_features_np.ndim > 2:
        # 如果是多维张量(如 [batch, channel, time])，把它reshape为两维 [batch, channel*time]
        if enhanced_features_np.ndim == 3:
            # [batch, channel, time] -> [batch, channel*time]
            enhanced_features_np = enhanced_features_np.reshape(enhanced_features_np.shape[0], -1)
        else:
            # 其他情况，简单地平铺成二维
            enhanced_features_np = enhanced_features_np.reshape(-1, np.prod(enhanced_features_np.shape[1:]))
            
    if target_features_np.ndim > 2:
        # 与上面相同的处理
        if target_features_np.ndim == 3:
            target_features_np = target_features_np.reshape(target_features_np.shape[0], -1)
        else:
            target_features_np = target_features_np.reshape(-1, np.prod(target_features_np.shape[1:]))
            
    print(f"处理后特徵形狀: enhanced={enhanced_features_np.shape}, target={target_features_np.shape}")
    
    # 規範化特徵（用於更好的比較）
    from sklearn.preprocessing import StandardScaler
    try:
        # 標準化所有特徵，使它們具有相同的規模
        scaler = StandardScaler()
        enhanced_features_np = scaler.fit_transform(enhanced_features_np)
        target_features_np = scaler.transform(target_features_np)
        print("特徵已成功標準化")
    except Exception as e:
        print(f"特徵標準化失敗，使用原始特徵: {str(e)}")
    
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
    # 動態調整 perplexity，確保它小於樣本數量
    n_samples = combined_features.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)
    if adjusted_perplexity < perplexity:
        print(f"警告: 樣本數量 ({n_samples}) 小於設定的 perplexity ({perplexity})，自動調整為 {adjusted_perplexity}")
    try:
        # 嘗試使用最新的TSNE參數（使用 max_iter 而不是 n_iter）
        tsne = TSNE(
            n_components=2, 
            perplexity=adjusted_perplexity,
            max_iter=1000,  # 修改為 max_iter
            random_state=42,
            learning_rate='auto',  # 自動選擇學習率
            init='pca'  # 使用PCA初始化以加速收斂
        )
    except TypeError as e:
        print(f"使用最新參數時遇到錯誤: {e}")
        try:
            # 回退到較舊版本的參數設置
            print("回退到舊版本 t-SNE 參數 (使用 n_iter)")
            tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=1000, random_state=42)
        except TypeError as e2:
            # 如果還是不行，使用最基本的參數
            print(f"回退參數依然失敗: {e2}")
            print("使用最基本的 t-SNE 參數")
            tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
    
    # 執行t-SNE降維
    try:
        tsne_results = tsne.fit_transform(combined_features)
        
        # 分離結果
        enhanced_tsne = tsne_results[:len(enhanced_features_np)]
        target_tsne = tsne_results[len(enhanced_features_np):]
        
        # 繪製圖形 - 使用更好的視覺效果
        plt.figure(figsize=(12, 10))
        
        # 添加更好的視覺效果
        plt.scatter(enhanced_tsne[:, 0], enhanced_tsne[:, 1], c='blue', alpha=0.6, label='Enhanced Features',
                   s=60, edgecolors='white', linewidths=0.5)
        plt.scatter(target_tsne[:, 0], target_tsne[:, 1], c='red', alpha=0.6, label='Target Features',
                   s=60, edgecolors='white', linewidths=0.5)
        
        # 添加邊界和網格線以提高可讀性
        plt.title('t-SNE: Enhanced vs Target Features', fontsize=18, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=14)
        plt.ylabel('t-SNE Dimension 2', fontsize=14)
        plt.legend(fontsize=14, framealpha=0.9, loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        
        # 添加背景色以提高對比度
        ax = plt.gca()
        ax.set_facecolor('#f9f9f9')
        
        # 繪製邊界框
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # 保存高解析度圖像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ t-SNE 視覺化圖已保存到: {save_path}")
        
    except Exception as e:
        print(f"❌ t-SNE 計算或繪圖失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 嘗試使用更簡單的方式重試
        try:
            print("嘗試使用簡化的t-SNE進行重試...")
            simple_tsne = TSNE(n_components=2, perplexity=min(30, adjusted_perplexity), n_iter=500, random_state=42)
            tsne_results = simple_tsne.fit_transform(combined_features)
            
            # 繪製簡化圖形
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_results[:len(enhanced_features_np), 0], tsne_results[:len(enhanced_features_np), 1], c='blue', alpha=0.5, label='Enhanced')
            plt.scatter(tsne_results[len(enhanced_features_np):, 0], tsne_results[len(enhanced_features_np):, 1], c='red', alpha=0.5, label='Target')
            plt.title('t-SNE Visualization (Simplified)')
            plt.legend()
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"✅ 簡化的t-SNE視覺化圖已保存到: {save_path}")
        except Exception as e2:
            print(f"❌ 簡化的t-SNE也失敗了: {str(e2)}")
            # 創建一個空白圖像，避免後續流程出錯
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "t-SNE visualization failed", ha='center', va='center')
            plt.savefig(save_path)
            plt.close()

def train_model(model, train_loader, optimizer, device, save_dir, config, num_epochs=100, scheduler=None, val_loader=None, use_content_loss=True):
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
        
        # 嘗試恢復之前保存的記錄
        if 'epochs_record' in checkpoint:
            epochs_record = checkpoint['epochs_record']
            train_losses_record = checkpoint['train_losses_record']
            val_losses_record = checkpoint['val_losses_record']
            feature_losses_record = checkpoint['feature_losses_record']
            voice_losses_record = checkpoint['voice_losses_record']
            lr_values_record = checkpoint['lr_values_record']
            
            # 嘗試恢復內容一致性損失記錄
            if 'content_losses_record' in checkpoint:
                content_losses_record = checkpoint['content_losses_record']
                print(f"已恢復之前的內容一致性損失記錄，共 {len(content_losses_record)} 個記錄點")
            else:                # 如果之前沒有內容一致性損失記錄，則創建相同長度的零記錄
                content_losses_record = [0.0] * len(epochs_record)
                print("找不到之前的內容一致性損失記錄，使用零值初始化")
    else:
        best_loss = float('inf') # 初始化最佳訓練損失
        print(f"\nInitial best train loss: {best_loss}")
        print(f"Initial best val loss: {best_val_loss}")
        print(f"Early stopping: Disabled")
      # 記錄訓練指標
    # 只有在沒有恢復之前的記錄時才初始化
    if not ('epochs_record' in locals() and epochs_record):
        epochs_record = []
        train_losses_record = []
        val_losses_record = []
        feature_losses_record = []
        voice_losses_record = []
        lr_values_record = []
        content_decay_factors = []  # 新增：紀錄內容衰減因子
        content_losses_record = []  # 新增：紀錄內容一致性損失
    
    # 確保所有必要的變數都已初始化
    if 'content_decay_factors' not in locals():
        content_decay_factors = []
    if 'content_losses_record' not in locals():
        content_losses_record = []
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        total_loss = 0.0
        total_feature_loss = 0.0
        total_voice_loss = 0.0
        total_content_loss = 0.0  # 新增：記錄內容一致性損失
        
        # 記錄當前epoch的衰減因子值
        epoch_decay_factors = []
        
        # 收集當前epoch的特徵（不僅是最後一個epoch）
        all_enhanced_features = []
        all_target_features = []
        all_target_discrete_codes = []
        all_input_discrete_codes = []
        min_seq_length = None
        min_discrete_length = None
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (input_wav, target_wav, *extra_data) in progress_bar:
            # 擷取額外數據，如果有的話
            content_ids = extra_data[0] if extra_data else None
                
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
            output, input_features, enhanced_features, input_discrete_code, intermediate_enhanced_features, intermediate_features_list = model(input_wav)
            
            # 獲取目標音頻的特徵
            with torch.no_grad():
                # 修改為使用與input相同的方法提取target特徵
                target_features, target_discrete_code = model.feature_extractor.wavtokenizer.encode_infer(target_wav, bandwidth_id=bandwidth_id)

                # 確保特徵和離散編碼的形状正確
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
                all_input_discrete_codes.append(input_discrete_code.detach().cpu())            # 計算損失 - 根據不同的模式選擇不同的損失函數
            # 檢查是否啟用了分層損失和內容一致性的組合模式
            if config.get('use_layered_loss', False) and config.get('tsne_flow_with_content', False) and content_ids is not None and 'intermediate_features_list' in locals():
                # 使用分層損失：前兩層著重內容一致性，後三層著重L2特徵損失
                # 這是 --tsne_flow_with_content --use_layered_loss --first_two_blocks_only 模式
                loss, loss_details = compute_layered_hybrid_loss(
                    output, target_wav, enhanced_features, target_features, intermediate_features_list, content_ids, device,
                    current_epoch=epoch, total_epochs=num_epochs
                )
                # 存儲當前批次的衰減因子，僅保存一次每個epoch的值
                if len(epoch_decay_factors) == 0 or epoch_decay_factors[-1] != loss_details['content_decay_factor']:
                    epoch_decay_factors.append(loss_details['content_decay_factor'])
                
                print(f"\r使用分層損失+內容一致性 - 內容損失: {loss_details['avg_layer_content_loss']:.4f}, L2損失: {loss_details['avg_layer_l2_loss']:.4f}, 衰減因子: {loss_details['content_decay_factor']:.3f}", end='')
            elif config.get('use_layered_loss', False) and content_ids is not None and 'intermediate_features_list' in locals():
                # 普通分層損失模式：--use_layered_loss
                loss, loss_details = compute_layered_hybrid_loss(
                    output, target_wav, enhanced_features, target_features, intermediate_features_list, content_ids, device,
                    current_epoch=epoch, total_epochs=num_epochs
                )
                # 存儲當前批次的衰減因子
                if len(epoch_decay_factors) == 0 or epoch_decay_factors[-1] != loss_details['content_decay_factor']:
                    epoch_decay_factors.append(loss_details['content_decay_factor'])
                
                print(f"\r使用分層損失 - 內容損失: {loss_details['avg_layer_content_loss']:.4f}, L2損失: {loss_details['avg_layer_l2_loss']:.4f}, 衰減因子: {loss_details['content_decay_factor']:.3f}", end='')
            elif config.get('tsne_flow_with_L2', False):
                # 使用僅L2損失的模式：tsne.py處理流程 + 僅L2損失
                loss, loss_details = compute_hybrid_loss(
                    output, target_wav, enhanced_features, target_features, device
                )
                print(f"\r使用純L2距離損失 - 特徵損失: {loss_details['feature_loss']:.4f}", end='')
            elif config.get('tsne_flow_with_content', False) and content_ids is not None:
                # 使用混合模式：tsne.py處理流程 + 內容一致性損失
                loss, loss_details = compute_hybrid_loss_with_tsne_flow(
                    output, target_wav, enhanced_features, target_features, intermediate_enhanced_features, content_ids, device
                )
            elif content_ids is not None:
                # 使用內容感知損失函數 (標準模式)
                loss, loss_details = compute_hybrid_loss_with_content(
                    output, target_wav, enhanced_features, target_features, intermediate_enhanced_features, content_ids, device
                )
            else:
                # 在沒有內容ID的情況下也使用標準內容一致性損失，但會自動忽略內容一致性部分
                loss, loss_details = compute_hybrid_loss_with_content(
                    output, target_wav, enhanced_features, target_features, intermediate_enhanced_features, None, device
                )
              # 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_feature_loss += loss_details["feature_loss"]
            total_voice_loss += loss_details["voice_loss"]
            
            # 如果有內容一致性損失，則累積它
            if "content_consistency_loss" in loss_details or "avg_layer_content_loss" in loss_details:
                content_loss = loss_details.get("content_consistency_loss", 0.0) or loss_details.get("avg_layer_content_loss", 0.0)
                total_content_loss += content_loss
            
            # 監控 GPU 記憶體
            allocated, cached = monitor_gpu_memory()
            
            # 獲取當前學習率
            current_lr = optimizer.param_groups[0]['lr']
              # 更新進度條
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'feat': f'{loss_details["feature_loss"]:.4f}',
                'voice': f'{loss_details["voice_loss"]:.4f}',
                'lr': f'{current_lr:.6f}',
                'GPU_MB': f'{allocated:.0f}/{cached:.0f}'
            }
            
            # 如果有內容一致性損失，增加到進度條顯示
            if "content_consistency_loss" in loss_details or "avg_layer_content_loss" in loss_details:
                content_loss = loss_details.get("content_consistency_loss", 0.0) or loss_details.get("avg_layer_content_loss", 0.0)
                postfix_dict['cont'] = f'{content_loss:.4f}'
                
            progress_bar.set_postfix(postfix_dict)
        
        # 保存每個epoch收集的特徵（使用與tsne.py相同的保存間隔）
        if (epoch + 1) % 300 == 0 or epoch == num_epochs - 1:
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
                            
            # 打印內容ID信息以幫助調試
            if content_ids is not None:
                print("\n內容ID分析：")
                # 轉換為列表以便於打印
                if isinstance(content_ids, torch.Tensor):
                    content_ids_list = content_ids.cpu().tolist()
                else:
                    content_ids_list = list(content_ids)
                
                # 統計每個ID出現次數
                from collections import Counter
                id_counts = Counter(content_ids_list)
                print(f"批次中共有 {len(content_ids_list)} 個樣本，{len(id_counts)} 個不同內容ID")
                print(f"每個ID出現次數：{dict(id_counts)}")
                
                # 檢查是否有足夠的相同ID樣本用於計算內容一致性損失
                has_multiple = any(count >= 2 for count in id_counts.values())
                if not has_multiple:
                    print("⚠️ 警告：批次中沒有任何內容ID出現2次或以上，無法計算內容一致性損失！")
        
        avg_train_loss = total_loss / len(train_loader)
        avg_feature_loss = total_feature_loss / len(train_loader)
        avg_voice_loss = total_voice_loss / len(train_loader)
        avg_content_loss = total_content_loss / len(train_loader) if total_content_loss > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄當前epoch的訓練指標
        epochs_record.append(epoch + 1)
        train_losses_record.append(avg_train_loss)
        feature_losses_record.append(avg_feature_loss)
        voice_losses_record.append(avg_voice_loss)
        lr_values_record.append(current_lr)
        content_losses_record.append(avg_content_loss)  # 新增：記錄內容一致性損失
        
        # 記錄內容衰減因子 (使用當前epoch的最後一個值)
        if epoch_decay_factors:
            # 如果在此epoch期間收集了衰減因子，則保存最後一個值
            content_decay_factors.append(epoch_decay_factors[-1])
        else:            # 如果沒有收集到衰減因子（可能使用了不同的損失函數），則使用理論值
            content_decay_factor = compute_decay_factor(epoch, num_epochs)
            content_decay_factors.append(content_decay_factor)
        
        print(f'\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, '
              f'Feature Loss: {avg_feature_loss:.4f}, Voice Loss: {avg_voice_loss:.4f}, '
              f'Content Loss: {avg_content_loss:.4f}, '
              f'Learning Rate: {current_lr:.6f}')
          # 驗證階段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_feature_loss = 0.0
            val_voice_loss = 0.0
            val_content_loss = 0.0  # 新增：跟踪驗證內容一致性損失
            
            with torch.no_grad():
                for batch_data in val_loader:
                    # 处理不同长度的批次数据 (保留content_id兼容性但採用與tsne.py類似的處理方式)
                    if len(batch_data) == 3:
                        input_wav, target_wav, _ = batch_data  # 忽略content_ids
                    else:
                        input_wav, target_wav = batch_data
                    
                    # 移動數據到設備並進行正規化
                    input_wav = input_wav.to(device)
                    target_wav = target_wav.to(device)
                    
                    # 確保輸入數據的幅度合適
                    input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
                    target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
                    
                    # 前向傳播 - 獲取包括中間特徵在內的所有輸出
                    output_tuple = model(input_wav)
                    output, input_features, enhanced_features, _, intermediate_enhanced_features, intermediate_features_list = output_tuple
                    
                    # 修改: 使用與訓練時相同的方式提取目標特徵
                    bandwidth_id = torch.zeros(input_wav.size(0), dtype=torch.long, device=input_wav.device)
                    target_features, _ = model.feature_extractor.wavtokenizer.encode_infer(target_wav, bandwidth_id=bandwidth_id)
                    
                    # 確保特徵形狀一致
                    enhanced_features = model.feature_extractor.ensure_feature_shape(enhanced_features)
                    target_features = model.feature_extractor.ensure_feature_shape(target_features)
                    
                    # 根據模式選擇損失函數，與訓練階段保持一致
                    if config.get('use_layered_loss', False):
                        # 使用分層損失：前幾層著重內容一致性，後幾層著重L2特徵損失，隨訓練進度動態調整權重
                        loss, loss_details = compute_layered_hybrid_loss(
                            output, target_wav, enhanced_features, target_features, intermediate_features_list, None, device,
                            current_epoch=epoch, total_epochs=num_epochs
                        )
                    elif config.get('tsne_flow_with_L2', False):
                        # 使用僅L2損失的模式：tsne.py處理流程 + 僅L2損失
                        loss, loss_details = compute_hybrid_loss(
                            output, target_wav, enhanced_features, target_features, device
                        )
                    elif config.get('tsne_flow_with_content', False):
                        # 使用混合模式：tsne處理流程 + 內容一致性損失
                        loss, loss_details = compute_hybrid_loss_with_tsne_flow(
                            output, target_wav, enhanced_features, target_features, intermediate_enhanced_features, None, device
                        )
                    else:
                        # 在驗證階段使用包含內容一致性損失的標準計算邏輯
                        loss, loss_details = compute_hybrid_loss_with_content(
                            output, target_wav, enhanced_features, target_features, intermediate_enhanced_features, None, device
                        )
                        val_loss += loss.item()
                        val_feature_loss += loss_details["feature_loss"]
                        val_voice_loss += loss_details["voice_loss"]
                      # 累計內容一致性損失
                    if "content_consistency_loss" in loss_details or "avg_layer_content_loss" in loss_details:
                        content_loss = loss_details.get("content_consistency_loss", 0.0) or loss_details.get("avg_layer_content_loss", 0.0)
                        val_content_loss += content_loss
                  # 計算平均損失
                avg_val_loss = val_loss / len(val_loader)
                avg_val_feature_loss = val_feature_loss / len(val_loader)
                avg_val_voice_loss = val_voice_loss / len(val_loader)
                avg_val_content_loss = val_content_loss / len(val_loader) if val_content_loss > 0 else 0.0
                
                val_losses_record.append(avg_val_loss)
                
                print(f'Validation Loss: {avg_val_loss:.4f}, '
                      f'Val Feature Loss: {avg_val_feature_loss:.4f}, Val Voice Loss: {avg_val_voice_loss:.4f}, '                      f'Val Content Loss: {avg_val_content_loss:.4f}')
                  # 使用驗證損失更新學習率排程器
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)  # 使用驗證損失來調整學習率
                        print(f"Learning rate updated based on validation loss: {optimizer.param_groups[0]['lr']:.6f}")
                    else:
                        scheduler.step()
                    print(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f}")
            
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
                'lr_values_record': lr_values_record,
                'content_losses_record': content_losses_record  # 新增：保存內容一致性損失記錄
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
                'lr_values_record': lr_values_record,
                'content_losses_record': content_losses_record  # 新增：保存內容一致性損失記錄
            }
            
            # 如果有排程器，也保存它的狀態
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\nCheckpoint saved to: {checkpoint_path}")            # 每次保存檢查點時也保存學習曲線圖
            curve_path = os.path.join(save_dir, f'learning_curve_epoch_{epoch+1}.png')
            plot_learning_curves(
                epochs_record, train_losses_record, val_losses_record, 
                feature_losses_record, voice_losses_record, lr_values_record, 
                curve_path, content_losses_record
            )
        
        # 每50個 epoch 繪製一次學習曲線以及t-SNE可視化圖
        if (epoch + 1) % 50 == 0:            # 保存學習曲線
            curve_path = os.path.join(save_dir, f'learning_curve_epoch_{epoch+1}.png')
            plot_learning_curves(
                epochs_record, train_losses_record, val_losses_record, 
                feature_losses_record, voice_losses_record, lr_values_record, 
                curve_path, content_losses_record
            )
            # 創建t-SNE目錄並生成t-SNE可視化圖
            print(f"\n為 epoch {epoch+1} 生成t-SNE特徵可視化...")
            # 使用固定的tsne_visualizations目錄
            tsne_dir = os.path.join(save_dir, 'tsne_visualizations')
            os.makedirs(tsne_dir, exist_ok=True)
            tsne_save_path = os.path.join(tsne_dir, f'tsne_epoch_{epoch+1}.png')
            
            try:
                # 收集更多樣本用於t-SNE可視化 (與tsne.py保持一致的方式)
                print(f"收集特徵數據用於t-SNE可視化...")
                all_enhanced_features = []
                all_target_features = []
                
                # 只使用幾個batch的數據以避免內存問題
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(train_loader):
                        if batch_idx >= 3:  # 僅使用前3個batch
                            break
                        
                        # 解包數據，同時兼容包含content_id的數據格式
                        if len(batch_data) == 3:  # 包含content_id
                            input_wav, target_wav, _ = batch_data
                        else:  # 不包含content_id
                            input_wav, target_wav = batch_data
                            
                        # 移動數據到設備並進行正規化
                        input_wav = input_wav.to(device)
                        target_wav = target_wav.to(device)
                        
                        # 確保輸入數據的幅度合適
                        input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
                        target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
                        
                        # 獲取特徵 (處理可能返回6個值的情況)
                        model_output = model(input_wav)
                        # 檢查返回值數量，兼容兩種版本的模型
                        if isinstance(model_output, tuple) and len(model_output) >= 3:
                            batch_enhanced_features = model_output[2]  # 第三個元素應該是enhanced_features
                        
                        # 獲取目標特徵
                        bandwidth_id = torch.zeros(input_wav.size(0), dtype=torch.long, device=input_wav.device)
                        batch_target_features, _ = model.feature_extractor.wavtokenizer.encode_infer(target_wav, bandwidth_id=bandwidth_id)
                        
                        # 確保特徵形狀正確
                        batch_enhanced_features = model.feature_extractor.ensure_feature_shape(batch_enhanced_features)
                        batch_target_features = model.feature_extractor.ensure_feature_shape(batch_target_features)
                        
                        # 收集特徵
                        all_enhanced_features.append(batch_enhanced_features.detach().cpu())
                        all_target_features.append(batch_target_features.detach().cpu())
                
                # 合併批次特徵
                if all_enhanced_features and all_target_features:
                    # 找出最小的特征时间维度，确保所有张量可以连接
                    min_time_dim = float('inf')
                    for feat in all_enhanced_features + all_target_features:
                        if feat.size(-1) < min_time_dim:
                            min_time_dim = feat.size(-1)
                    
                    # 截断所有特徵到相同的時間長度
                    all_enhanced_features = [feat[..., :min_time_dim] for feat in all_enhanced_features]
                    all_target_features = [feat[..., :min_time_dim] for feat in all_target_features]
                    
                    # 现在可以安全连接
                    enhanced_features_for_tsne = torch.cat(all_enhanced_features, dim=0)
                    target_features_for_tsne = torch.cat(all_target_features, dim=0)
                    
                    print(f"為t-SNE收集了 {enhanced_features_for_tsne.size(0)} 個樣本，特徵時間維度統一為 {min_time_dim}")
                    
                    # 生成t-SNE可視化
                    plot_tsne_visualization(
                        enhanced_features=enhanced_features_for_tsne,
                        target_features=target_features_for_tsne,
                        save_path=tsne_save_path
                    )
                    print(f"✅ Epoch {epoch+1} 的特徵可視化已保存到: {tsne_save_path}")
                else:
                    print("❌ 沒有收集到特徵數據用於t-SNE可視化")
            
            except Exception as e:
                print(f"❌ 生成t-SNE可視化時出錯: {str(e)}")
                traceback.print_exc()
        
        # 每100個 epoch 以及最後一輪時保存樣本 (提高保存頻率)
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            print(f"\nSaving samples for epoch {epoch+1}...")
            # 保存當前 batch 的音頻樣本
            with torch.no_grad():
                # 基於當前 epoch 計算偏移量，確保每次採樣到不同的批次
                epoch_offset = (epoch // 100) % 10  # 每10次採樣循環一次
                
                # 創建一個臨時數據載入器，使用原始的train_loader的dataset
                # 確保獲取不同的樣本
                dataset_to_use = train_loader.dataset  # 使用train_loader的dataset而非undefined的train_dataset
                temp_loader = DataLoader(
                    dataset_to_use,
                    batch_size=config['batch_size'],
                    shuffle=True,  # 強制隨機打亂
                    num_workers=1,  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
                    collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None,
                    worker_init_fn=lambda _: random.seed(42 + epoch)  # 基於 epoch 設置隨機種子
                )
                
                # 跳過前面的批次，實現每次採樣不同的數據
                for _ in range(epoch_offset):
                    try:
                        next(iter(temp_loader))
                    except StopIteration:
                        break
                
                # 採樣並保存
                for batch_idx, batch_data in enumerate(temp_loader):
                    # 解包數據，同時兼容包含content_id的數據格式
                    if len(batch_data) == 3:  # 包含content_id
                        input_wav, target_wav, _ = batch_data
                    else:  # 不包含content_id
                        input_wav, target_wav = batch_data
                    
                    input_wav = input_wav.to(device)
                    target_wav = target_wav.to(device)
                    
                    # 修改：保存完整的輸出元組，不僅僅是音頻輸出
                    output_tuple = model(input_wav)
                    
                    # 強制啟用保存完整音頻功能，使用預設的固定目錄
                    save_sample(input_wav, output_tuple, target_wav, epoch, batch_idx, save_dir, device, model)
                    
                    # 只保存前幾個 batch 的樣本
                    if batch_idx >= 2:  # 只保存前2個batch
                        break        # 在最後一個 epoch 生成 t-SNE 視覺化（與tsne.py保持一致）
        if epoch == num_epochs - 1:
            # 確保tsne_dir存在，使用固定的目錄結構
            tsne_dir = os.path.join(save_dir, 'tsne_visualizations')
            os.makedirs(tsne_dir, exist_ok=True)
            
            # 使用特殊檔名標記最終模型的t-SNE可視化
            tsne_final_save_path = os.path.join(tsne_dir, f'tsne_final_model.png')
            plot_tsne_visualization(
                enhanced_features=enhanced_features,
                target_features=target_features,
                save_path=tsne_final_save_path
            )
            print(f"✅ 最終模型的t-SNE可視化已保存到: {tsne_final_save_path}")
    # 訓練結束時，再次保存整個學習曲線
    final_curve_path = os.path.join(save_dir, 'final_learning_curve.png')
    plot_learning_curves(
        epochs_record, train_losses_record, val_losses_record, 
        feature_losses_record, voice_losses_record, lr_values_record, 
        final_curve_path, content_losses_record
    )
    
    # 繪製內容衰減因子變化圖
    decay_factor_path = os.path.join(save_dir, 'content_decay_factor.png')
    plot_content_decay_factor(epochs_record, content_decay_factors, decay_factor_path)
    
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


class ContentAwareBatchSampler:
    """
    內容感知批次採樣器 - 直接整合在ttt.py中，無需額外檔案
    
    確保每個批次都包含足夠的相同content_id樣本以計算內容一致性損失
    
    Args:
        dataset: 包含content_id的音訊數據集
        batch_size: 每個批次的大小
        content_ratio: 每個批次中相同content_id樣本的比例 (0.0-1.0)
        min_content_samples: 每個批次中相同內容ID的最小樣本數
        shuffle: 是否隨機打亂批次順序
        drop_last: 是否丟棄最後一個不完整的批次
    """
    def __init__(self, dataset, batch_size=8, content_ratio=0.5, min_content_samples=3, 
                 shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.content_ratio = content_ratio
        self.min_content_samples = min_content_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 對所有樣本按content_id分組
        self.content_groups = self._group_by_content_id()
        
        # 檢查每個內容ID的樣本數
        self._validate_and_report_groups()
        
        # 創建批次索引
        self.batch_indices = self._create_batch_indices()
        
    def _group_by_content_id(self):
        """將數據集樣本按content_id分組"""
        content_groups = defaultdict(list)
        
        # 檢查是否為Subset類型，如果是，需要獲取原始數據集和索引
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset, 'indices'):
            # 處理PyTorch的Subset類型
            original_dataset = self.dataset.dataset
            subset_indices = self.dataset.indices
            
            # 根據Subset索引更新content_groups
            for i, idx in enumerate(subset_indices):
                if hasattr(original_dataset, 'paired_files'):
                    # AudioDataset情況
                    content_id = original_dataset.paired_files[idx].get('content_id', f"unknown_{idx}")
                    content_groups[content_id].append(i)
        else:
            # 直接處理普通數據集
            if hasattr(self.dataset, 'paired_files'):
                # AudioDataset情況
                for i, pair in enumerate(self.dataset.paired_files):
                    content_id = pair.get('content_id', f"unknown_{i}")
                    content_groups[content_id].append(i)
            else:
                # 不明確的數據集類型，使用索引作為內容ID
                for i in range(len(self.dataset)):
                    content_groups[f"idx_{i}"].append(i)
                print("警告: 無法確定數據集類型，使用索引作為內容ID")
        
        return content_groups
        
    def _validate_and_report_groups(self):
        """驗證並報告分組情況"""
        if not self.content_groups:
            print("警告: 沒有找到任何內容ID分組")
            return
            
        # 統計每個內容ID的樣本數
        id_counts = {cid: len(indices) for cid, indices in self.content_groups.items()}
        
        # 找出有足夠樣本數的內容ID
        valid_ids = {cid: count for cid, count in id_counts.items() 
                    if count >= self.min_content_samples}
        
        print(f"\n內容ID分組統計:")
        print(f"總內容ID數量: {len(self.content_groups)}")
        print(f"有效內容ID數量 (樣本數 >= {self.min_content_samples}): {len(valid_ids)}")
        
        if len(valid_ids) < len(self.content_groups):
            print(f"警告: {len(self.content_groups) - len(valid_ids)} 個內容ID樣本數不足 {self.min_content_samples}")
        
        # 統計分布情況
        sample_counts = list(id_counts.values())
        if sample_counts:
            print(f"每個內容ID的樣本數: 最小={min(sample_counts)}, "
                  f"最大={max(sample_counts)}, 平均={sum(sample_counts)/len(sample_counts):.1f}")
        
    def _create_batch_indices(self):
        """創建批次索引列表"""
        batch_indices = []
        
        # 獲取所有可用的索引集合
        available_indices = set(range(len(self.dataset)))
        
        # 內容ID組中至少有min_content_samples個樣本的ID列表
        valid_content_ids = [
            cid for cid, indices in self.content_groups.items()
            if len(indices) >= self.min_content_samples
        ]
        
        if not valid_content_ids:
            print("警告: 沒有足夠的內容ID組，使用普通批次劃分")
            # 回退到普通批次劃分
            indices_list = list(available_indices)
            if self.shuffle:
                random.shuffle(indices_list)
            
            for i in range(0, len(indices_list), self.batch_size):
                batch = indices_list[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch_indices.append(batch)
            return batch_indices
        
        # 根據內容感知策略創建批次
        while available_indices and valid_content_ids:
            batch = []
            
            # 步驟1: 隨機選擇一個有效的內容ID
            selected_cid = random.choice(valid_content_ids)
            
            # 該內容ID組的可用索引
            cid_indices = [idx for idx in self.content_groups[selected_cid] 
                          if idx in available_indices]
            
            # 如果此內容ID沒有足夠的樣本，則從有效ID列表中移除
            if len(cid_indices) < self.min_content_samples:
                valid_content_ids.remove(selected_cid)
                continue
            
            # 步驟2: 確定要從該內容ID組中選取的樣本數
            content_samples = max(
                self.min_content_samples, 
                min(len(cid_indices), int(self.batch_size * self.content_ratio))
            )
            
            # 選擇內容ID樣本
            selected_indices = random.sample(cid_indices, content_samples)
            batch.extend(selected_indices)
            
            # 從可用索引中移除已選擇的索引
            for idx in selected_indices:
                available_indices.remove(idx)
            
            # 步驟3: 用其他內容ID的樣本填滿批次
            remaining_slots = self.batch_size - len(batch)
            
            if remaining_slots > 0 and available_indices:
                # 優先選擇不同內容ID的樣本
                other_indices = [idx for idx in available_indices 
                               if all(idx not in self.content_groups[cid] 
                                      for cid in [selected_cid])]
                
                # 如果其他內容ID的樣本不足，就使用任何可用樣本
                if len(other_indices) < remaining_slots:
                    other_indices = list(available_indices)
                
                # 隨機選擇剩餘樣本
                fill_indices = random.sample(
                    other_indices, 
                    min(remaining_slots, len(other_indices))
                )
                
                batch.extend(fill_indices)
                
                # 從可用索引中移除已選擇的索引
                for idx in fill_indices:
                    available_indices.remove(idx)
            
            batch_indices.append(batch)
            
            # 更新有效內容ID列表
            valid_content_ids = [
                cid for cid in valid_content_ids
                if sum(1 for idx in self.content_groups[cid] if idx in available_indices) >= self.min_content_samples
            ]
        
        # 如果還有剩餘索引，創建額外的批次
        if available_indices and not self.drop_last:
            remaining = list(available_indices)
            if self.shuffle:
                random.shuffle(remaining)
            
            for i in range(0, len(remaining), self.batch_size):
                batch = remaining[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch_indices.append(batch)
        
        # 如果需要，打亂批次順序
        if self.shuffle:
            random.shuffle(batch_indices)
        
        # 分析批次組成，用於調試
        self._analyze_batches(batch_indices)
        
        return batch_indices
        
    def _analyze_batches(self, batch_indices):
        """分析批次組成，顯示統計信息"""
        if not batch_indices:
            return
            
        # 統計有效批次數量 (包含足夠相同內容ID樣本的批次)
        valid_batches = 0
        content_counts_per_batch = []
        
        for batch in batch_indices:
            # 計算每個批次中每個內容ID的樣本數
            batch_content_counts = defaultdict(int)
            
            for idx in batch:
                # 找出該索引對應的內容ID
                for cid, indices in self.content_groups.items():
                    if idx in indices:
                        batch_content_counts[cid] += 1
                        break
            
            # 找出該批次中樣本數最多的內容ID
            if batch_content_counts:
                max_count = max(batch_content_counts.values())
                content_counts_per_batch.append(max_count)
                
                if max_count >= self.min_content_samples:
                    valid_batches += 1
        
        # 打印分析結果
        print(f"\n批次分析:")
        print(f"總批次數: {len(batch_indices)}")
        print(f"有效批次數 (至少含{self.min_content_samples}個相同內容ID): {valid_batches} ({valid_batches/len(batch_indices)*100:.1f}%)")
        
        if content_counts_per_batch:
            avg_content_count = sum(content_counts_per_batch) / len(content_counts_per_batch)
            print(f"每批次中相同內容ID的平均最大樣本數: {avg_content_count:.2f}")
    
    def __iter__(self):
        # 返回批次索引的迭代器
        return iter(self.batch_indices)
    
    def __len__(self):
        # 返回批次數量
        return len(self.batch_indices)


def main():
    # 设置临时目录，避免使用系统默认的 /tmp 目录
    import tempfile
    import os
    
    # 在当前工作目录下创建临时目录
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    tempfile.tempdir = temp_dir
    os.environ['TMPDIR'] = temp_dir
    
    print(f"临时文件目录设置为: {temp_dir}")
    
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
    parser.add_argument("--tsne_flow_with_content", action="store_true", help="處理流程與tsne.py盡可能一致，但同時計算內容一致性損失")
    parser.add_argument("--tsne_flow_with_L2", action="store_true", help="處理流程與tsne.py盡可能一致，僅使用標準L2損失函數")
    parser.add_argument("--use_layered_loss", action="store_true", help="使用分層損失：前幾層專注於內容一致性損失，後幾層專注於L2特徵損失")
    parser.add_argument("--first_two_blocks_only", action="store_true", help="內容一致性損失僅影響前兩個residual block，後續層完全交給特徵接近的L2損失")
    args = parser.parse_args()

    # 初始化分布式環境
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    # 將模式信息添加到日誌
    if args.tsne_flow_with_content and args.tsne_flow_with_L2:
        print("\n❌ 錯誤：不能同時啟用 tsne_flow_with_content 和 tsne_flow_with_L2")
        sys.exit(1)
    elif args.tsne_flow_with_content:
        print("\n✅ 啟用了tsne處理流程與內容一致性損失混合模式")
    elif args.tsne_flow_with_L2:
        print("\n✅ 啟用了tsne處理流程僅使用L2損失模式")
    if args.use_layered_loss:
        print("\n✅ 啟用了分層損失：前幾層專注於內容一致性損失，後幾層專注於L2特徵損失")
    if args.first_two_blocks_only:
        print("\n✅ 啟用了內容一致性損失僅前兩層模式：僅前兩個residual block使用內容一致性損失，後續層完全使用L2損失")
    # 設置固定輸出目錄 (與tsne.py類似，使用固定的output3目錄)
    output_base_dir = os.path.join(os.getcwd(), "results", "tsne_outputs")
    # 確保輸出目錄存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 使用固定的output3目錄
    output_dir = os.path.join(output_base_dir, "output4")
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建子目錄
    audio_samples_dir = os.path.join(output_dir, "audio_samples")
    tsne_viz_dir = os.path.join(output_dir, "tsne_visualizations")
    os.makedirs(audio_samples_dir, exist_ok=True)
    os.makedirs(tsne_viz_dir, exist_ok=True)
    
    print(f"\n✅ 輸出目錄已設置:")
    print(f"   - 主輸出目錄: {output_dir}")
    print(f"   - 音頻樣本目錄: {audio_samples_dir}")
    print(f"   - t-SNE可視化目錄: {tsne_viz_dir}")
    
    config = {
        'config_path': os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"),
        'model_path': os.path.join(os.getcwd(), "models", "wavtokenizer_large_speech_320_24k.ckpt"),        
        'save_dir': output_dir,
        'epochs': 2,              # 設定訓練輪數
        'batch_size': 8,             # 減小批次大小以節省記憶體
        'learning_rate': 0.005,      # 適當增加學習率以加快收斂
        'weight_decay': 0.001,
        'scheduler_patience': 5,
        'scheduler_factor': 0.99,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'feature_scale': 1.5,
        'num_workers': 2,            # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
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
        'val_speakers': ['girl9', 'boy7'],  # 指定驗證集說話者為girl9和boy7
        
        # 內容感知批次採樣相關參數
        'content_aware_batching': True,      # 是否啟用內容感知批次採樣
        'content_ratio': 0.5,                # 每個批次中相同content_id的樣本比例
        'min_content_samples': 5,            # 每批次中相同content_id的最小樣本數
        'val_content_aware': True,           # 驗證集是否也使用內容感知批次
        
        # 處理模式設定
        'use_content_loss': not args.tsne_flow_with_L2,  # 使用L2損失時不使用內容一致性損失
        'tsne_flow_with_content': args.tsne_flow_with_content,  # 啟用tsne處理流程與內容一致性混合模式
        'tsne_flow_with_L2': args.tsne_flow_with_L2,  # 啟用僅使用L2損失的tsne處理流程
        'use_layered_loss': args.use_layered_loss  # 啟用分層損失計算
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
        
        device = torch.device(f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda')
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
    # 使用命令行参数指定的GPU，或者默认使用所有可用GPU
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
    # 檢查是否設置了使用僅 box 材質的環境變數
    only_box = os.environ.get('ONLY_USE_BOX_MATERIAL', '').lower() == 'true'
    
    if only_box:
        # 僅使用 box 材質
        print("\n✅ 環境變數 ONLY_USE_BOX_MATERIAL=true，僅使用 box 材質")
        input_dirs = [os.path.join(os.getcwd(), "data", "raw", "box")]
    else:
        # 使用所有材質
        input_dirs = [
            os.path.join(os.getcwd(), "data", "raw", "box"),
            os.path.join(os.getcwd(), "data", "raw", "mac"),
            os.path.join(os.getcwd(), "data", "raw", "papercup"), 
            os.path.join(os.getcwd(), "data", "raw", "plastic") # plastic當作未知材質
        ]
    
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
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
    # 依照驗證策略將數據分为訓練集和驗證集
    
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
    
    # 檢查是否啟用內容感知批次採樣
    use_content_aware = config.get('content_aware_batching', True)
    content_ratio = config.get('content_ratio', 0.5)
    min_content_samples = config.get('min_content_samples', 3)
    
    # 打印批次採樣設置
    print(f"\n批次採樣設置:")
    if use_content_aware and args.local_rank == -1:  # 非分佈式訓練時啟用
        print(f"使用內容感知批次採樣:")
        print(f"- 內容比例: {content_ratio:.2f} (相同內容ID樣本比例)")
        print(f"- 最小樣本數: {min_content_samples} (每批次中相同content_id的最小樣本數)")
    else:
        reason = "分佫式訓練" if args.local_rank != -1 else "用戶設置"
        print(f"使用標準隨機批次採樣 (原因: {reason})")
    
    # 創建分布式採樣器或內容感知批次採樣器
    if args.local_rank != -1:
        # 分佈式訓練使用標準採樣器
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        # 創建標準訓練加載器
        train_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            sampler=train_sampler,
            num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
            collate_fn=train_collate_fn,
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
            persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
            worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
        )
        print("分佈式訓練 DataLoader 初始化完成")
    else:
        # 非分佈式訓練
        train_sampler = None
        val_sampler = None
        
        if use_content_aware:
            # 使用內容感知批次採樣器
            print("\n正在創建內容感知批次索引，這可能需要一些時間...")
            
            # 創建內容感知批次採樣器
            train_batch_sampler = ContentAwareBatchSampler(
                train_dataset,
                batch_size=config['batch_size'],
                content_ratio=content_ratio,
                min_content_samples=min_content_samples,
                shuffle=True,
                drop_last=False
            )
            
            # 使用批次採樣器創建數據加載器
            train_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,  # 使用批次採樣器代替batch_size和sampler
                num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
                collate_fn=train_collate_fn,
                pin_memory=config['pin_memory'],
                prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
                persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
                worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
            )
            print("內容感知訓練 DataLoader 初始化完成")
        else:
            # 使用標準批次採樣
            train_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                sampler=None,
                num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
                collate_fn=train_collate_fn,
                pin_memory=config['pin_memory'],
                prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
                persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
                worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
            )
            print("標準訓練 DataLoader 初始化完成")
    
    # 創建驗證數據加載器
    val_content_aware = config.get('val_content_aware', False)  # 驗證集是否也使用內容感知批次
    
    if args.local_rank != -1:
        # 分佈式訓練的驗證加載器
        val_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
            collate_fn=val_collate_fn,
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
            persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
            worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
        )
        print("分佈式驗證 DataLoader 初始化完成")
    elif use_content_aware and val_content_aware:
        # 使用內容感知批次採樣器創建驗證加載器
        print("\n正在為驗證集創建內容感知批次索引...")
        
        val_batch_sampler = ContentAwareBatchSampler(
            val_dataset,
            batch_size=config['batch_size'],
            content_ratio=content_ratio,  # 使用與訓練集相同的設定
            min_content_samples=min_content_samples,
            shuffle=False,  # 驗證不需要隨機順序
            drop_last=False
        )
        
        val_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
            collate_fn=val_collate_fn,
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
            persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
            worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
        )
        print("內容感知驗證 DataLoader 初始化完成")
    else:
        # 標準驗證加載器
        val_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            sampler=None,
            num_workers=config['num_workers'],  # 設置為0以避免多進程序列化問題，特別是處理lambda函數時
            collate_fn=val_collate_fn,
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,  # 當num_workers=0時，prefetch_factor必須為None
            persistent_workers=True if config['num_workers'] > 0 else False,  # 當num_workers=0時，persistent_workers必須為False
            worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None  # 當num_workers=0時，worker_init_fn無效
        )
        print("標準驗證 DataLoader 初始化完成")
    
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
    print("使用內容一致性損失進行訓練")
    
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config['save_dir'],
        config=config,
        num_epochs=config['epochs'],
        scheduler=scheduler,
        val_loader=val_loader,  # 添加驗證加載器
        use_content_loss=True  # 始終使用內容一致性損失
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

def compute_content_consistency_loss(intermediate_features, content_ids, device):
    """
    計算內容一致性損失。
    
    對於批次中具有相同content_id的樣本，計算它們的中間特徵之間的餘弦相似度。
    目標是使相同內容的不同說話者/材質特徵更接近（餘弦相似度更高）。
    
    Args:
        intermediate_features (torch.Tensor): 中間特徵張量 [batch_size, channels, time]
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備
        
    Returns:
        torch.Tensor: 內容一致性損失值
    """
    # 確保content_ids是在正確的設備上的張量，並且是數值類型
    if content_ids is None:
        # 如果沒有content_ids，創建虛擬的不同ID
        content_ids = torch.arange(intermediate_features.size(0), device=device)
    elif not isinstance(content_ids, torch.Tensor):
        # 將字符串ID轉換為數字ID
        try:
            # 嘗試直接將字符串轉換為數字
            numeric_ids = []
            for cid in content_ids:
                # 從字符串中提取數字部分
                if isinstance(cid, str):
                    # 提取所有數字
                    digits = ''.join(c for c in cid if c.isdigit())
                    if digits:
                        numeric_ids.append(int(digits))
                    else:
                        # 如果沒有數字，使用哈希值的一部分
                        numeric_ids.append(hash(cid) % 10000)
                else:
                    # 已經是數字或其他類型，嘗試直接轉換
                    numeric_ids.append(int(cid) if cid is not None else 0)
            content_ids = torch.tensor(numeric_ids, device=device)
        except Exception as e:
            print(f"無法將content_ids轉換為張量: {e}")
            # 出錯時創建虛擬的不同ID
            content_ids = torch.arange(intermediate_features.size(0), device=device)
    
    # 獲取批次大小
    batch_size = intermediate_features.size(0)
    
    # 初始化損失為0
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 計算不同批次內相同content_id的樣本數
    unique_content_ids = torch.unique(content_ids)
    valid_groups = 0
    
    # 對於每個唯一的內容ID，找到具有該ID的所有樣本
    for content_id in unique_content_ids:
        # 找到對應此content_id的所有樣本索引
        indices = (content_ids == content_id).nonzero(as_tuple=True)[0]
        
        # 如果此內容ID只出現一次，則跳過（至少需要2個樣本才能計算一致性）
        if len(indices) < 2:
            continue
            
        # 提取這些樣本的中間特徵
        group_features = intermediate_features[indices]
        
        # 計算這組特徵的平均值
        mean_feature = torch.mean(group_features, dim=0, keepdim=True)
        
        # 正規化特徵向量，準備計算餘弦相似度
        norm_group_features = F.normalize(group_features, p=2, dim=1)
        norm_mean_feature = F.normalize(mean_feature, p=2, dim=1)
          # 計算餘弦相似度 (1 - similarity 轉為距離)
        # cos相似度範圍為[-1, 1]，1表示完全相同，-1表示完全相反，0表示正交
        # 將其轉換為[0, 2]的距離，0表示完全相似
        cos_sim = F.cosine_similarity(norm_group_features, norm_mean_feature, dim=1)
        distances = 1.0 - cos_sim  # 轉換為距離，範圍[0, 2]，0表示完全相似（我們想要最小化這個距離）
        
        # 累加這組樣本的平均距離到損失值
        group_loss = torch.mean(distances)
        loss = loss + group_loss
        
        # 計數有效的組
        valid_groups += 1
    
    # 如果有有效的組，則取平均值；否則損失為0
    if valid_groups > 0:
        loss = loss / valid_groups
        
    return loss


def compute_layered_hybrid_loss(output, target_wav, enhanced_features, target_features, 
                           intermediate_features_list, content_ids, device, current_epoch=0, total_epochs=100):
    """
    修改版分層損失函數 - 按照要求實現特定損失函數配置：
    1. 僅在第二層（索引1）應用內容一致性損失
    2. 僅在最終層（進入decoder前）應用L2損失
    3. 中間層自由學習，不施加任何損失
    
    Args:
        output (torch.Tensor): 模型輸出的音頻波形
        target_wav (torch.Tensor): 目標波形
        enhanced_features (torch.Tensor): 最終增強特徵
        target_features (torch.Tensor): 目標特徵
        intermediate_features_list (list): 包含模型中間層特徵的列表
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備
        current_epoch (int): 當前訓練的epoch
        total_epochs (int): 總訓練epochs數
        
    Returns:
        tuple: (total_loss, loss_details)，其中loss_details是包含各損失組件的字典
    """
    # 確保所有輸入都有正確的維度
    if enhanced_features.dim() == 2:
        enhanced_features = enhanced_features.unsqueeze(0)
    if target_features.dim() == 2:
        target_features = target_features.unsqueeze(0)
    
    # 初始化損失值
    content_loss = torch.tensor(0.0, device=device)
    l2_loss = torch.tensor(0.0, device=device)
    
    # 計算最終層的L2損失 (進入decoder前的特徵)
    l2_loss = compute_feature_loss(enhanced_features, target_features, device)
    
    # 計算內容一致性損失，但僅對第二層 (index 1)
    num_layers = len(intermediate_features_list)
    if num_layers > 1:  # 確保至少有第二層
        second_layer_features = intermediate_features_list[1]  # 索引1表示第二層
        
        # 檢查是否有內容ID
        if content_ids is None or len(content_ids) == 0:
            print(f"警告: 沒有內容ID提供，無法計算內容一致性損失")
            content_loss = torch.tensor(0.0, device=device)
        else:
            # 檢查是否有足夠的相同內容ID的樣本
            unique_ids = torch.unique(content_ids) if isinstance(content_ids, torch.Tensor) else set(content_ids)
            if len(unique_ids) == len(content_ids):
                print(f"警告: 批次中沒有重複的內容ID，無法計算內容一致性損失")
                content_loss = torch.tensor(0.0, device=device)
            else:
                content_loss = compute_content_consistency_loss(second_layer_features, content_ids, device)
    else:
        print(f"警告: 中間特徵層數不足 ({num_layers})，無法獲取第二層特徵")
    
    # 設定內容損失和L2損失的權重
    alpha = 0.01  # 內容一致性損失的權重
    beta = 0.99   # L2損失的權重
    
    # 計算最終損失
    total_loss = alpha * content_loss + beta * l2_loss
    
    # 停用衰減因子，固定為1.0（表示不衰減）
    content_decay_factor = 1.0
    
    # 打印詳細的損失信息和批次內容分析
    print(f"\r專用損失模式 - 第二層內容損失: {content_loss.item():.4f}, 最終層L2損失: {l2_loss.item():.4f}", end='')
    
    # 分析並打印content_ids的詳細信息，用於調試內容一致性損失
    if content_ids is not None:
        content_ids_str = ', '.join(str(cid.item() if isinstance(cid, torch.Tensor) else cid) 
                                    for cid in content_ids)
        unique_ids = set(cid.item() if isinstance(cid, torch.Tensor) else cid for cid in content_ids)
        id_counts = {}
        for uid in unique_ids:
            count = sum(1 for cid in content_ids if (cid.item() if isinstance(cid, torch.Tensor) else cid) == uid)
            id_counts[uid] = count
        
        # 每隔100個批次或內容損失為0時打印詳細信息
        if content_loss.item() == 0 or random.randint(0, 100) == 0:
            print(f"\n[DEBUG] 批次內容ID: {content_ids_str}")
            print(f"[DEBUG] 唯一內容ID: {unique_ids}")
            print(f"[DEBUG] 內容ID出現次數: {id_counts}")
            print(f"[DEBUG] 批次大小: {len(content_ids)}, 唯一ID數: {len(unique_ids)}")
            if len(unique_ids) == len(content_ids):
                print(f"[WARNING] 批次中沒有重複的內容ID，無法計算內容一致性損失!")
    
    return total_loss, {
        'feature_loss': l2_loss.item(),
        'content_consistency_loss': content_loss.item(),
        'avg_layer_content_loss': content_loss.item(),
        'avg_layer_l2_loss': l2_loss.item(),
        'content_decay_factor': content_decay_factor,
        'voice_loss': 0.0,  # 保持與現有API相容
        'total_loss': total_loss.item()
    }

def compute_decay_factor(current_epoch, total_epochs):
    """
    計算特定epoch的內容衰減因子
    
    Args:
        current_epoch (int): 當前epoch
        total_epochs (int): 總訓練epochs數
        
    Returns:
        float: 計算出的內容衰減因子（已停用衰減）
    """
    # 固定返回1.0，停用衰減功能
    return 1.0  # 返回1.0表示不衰減，保持原始權重

def plot_content_decay_factor(epochs, decay_factors, save_path='content_decay_factor.png'):
    """
    繪製內容衰減因子隨訓練進度變化的圖表
    
    Args:
        epochs (list): 紀錄的 epoch 列表
        decay_factors (list): 對應的內容衰減因子列表
        save_path (str): 圖表保存路徑
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, decay_factors, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Content Decay Factor')
    plt.title('Dynamic Content Decay Factor Over Training Progress')
    plt.grid(True, alpha=0.3)
    
    # 添加标注线
    total_epochs = max(epochs)
    plt.axvline(x=total_epochs * 0.3, color='r', linestyle='--', alpha=0.5, 
                label='Phase change: 30% of training')
    plt.axvline(x=total_epochs * 0.9, color='g', linestyle='--', alpha=0.5, 
                label='Phase change: 90% of training')
    
    # 添加衰減因子範圍
    plt.axhline(y=0.5, color='purple', linestyle=':', alpha=0.5,
                label='Initial decay factor (0.5)')
    plt.axhline(y=0.1, color='orange', linestyle=':', alpha=0.5,
                label='Final decay factor (0.1)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"內容衰減因子變化圖表已保存至: {save_path}")

# 使用说明：
# 標準模式 (包含内容一致性损失): python ttt.py
# TSNE流程 + 內容一致性模式: python ttt.py --tsne_flow_with_content
# TSNE流程 + 僅L2損失模式: python ttt.py --tsne_flow_with_L2  (使用compute_hybrid_loss函數，與tsne.py一致)
# 損失計算邏輯說明：
#
# 1. --tsne_flow_with_content --use_layered_loss --first_two_blocks_only：
#    - 前兩層 residual block 使用內容一致性損失（餘弦相似度）
#    - 後三層完全由目標特徵損失（L2距離）主導
#    - 實現函數：compute_layered_hybrid_loss
#
# 2. --use_layered_loss：
#    - 分層損失，從前到後逐漸從內容一致性損失過渡到特徵損失
#    - 實現函數：compute_layered_hybrid_loss
#
# 3. --tsne_flow_with_L2：
#    - 只使用L2特徵損失，與tsne.py相同
#    - 實現函數：compute_hybrid_loss
#
# 4. --tsne_flow_with_content：
#    - 混合內容一致性損失和特徵損失
#    - 實現函數：compute_hybrid_loss_with_tsne_flow
#
# 5. 默認模式：
#    - 使用內容一致性損失和特徵損失
#    - 實現函數：compute_hybrid_loss_with_content

if __name__ == "__main__":
    main()