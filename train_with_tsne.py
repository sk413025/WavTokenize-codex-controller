import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
import torchaudio

from try2 import (AudioDataset, EnhancedModel, process_audio, 
                 save_checkpoint, compute_voice_focused_loss, plot_spectrograms)

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

class EnhancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")  # 會顯示使用的設備
        self.setup_model()
        
    def setup_model(self):
        # 1. 載入try2.py訓練好的模型和權重
        self.model = EnhancedModel(config_path=self.config['config_path'],
                                 model_path=self.config['model_path']).to(self.device)
        checkpoint = torch.load(self.config['pretrained_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 使用 named_parameters 更精確地控制
        for name, param in self.model.named_parameters():
            if 'encodec.encoder' in name or 'encodec.decoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 添加參數檢查
        for name, param in self.model.named_parameters():
            if 'encodec.encoder' in name or 'encodec.decoder' in name:
                assert not param.requires_grad, f"{name} 應該被凍結"
                
    def calculate_tsne_distance(self, input_features, target_features):
        """計算兩組特徵之間的T-SNE距離並返回可微分的損失 (僅用於Loss計算)"""
        # 轉換為2D形式並保持梯度
        input_2d = input_features.reshape(input_features.size(0), -1)
        target_2d = target_features.reshape(target_features.size(0), -1)
        
        # 正規化特徵
        input_norm = torch.nn.functional.normalize(input_2d, dim=1)
        target_norm = torch.nn.functional.normalize(target_2d, dim=1)
        
        # 計算餘弦相似度矩陣
        similarity_matrix = torch.matmul(input_norm, target_norm.transpose(0, 1))  # [B, B]
        
        # 初始化對角線元素的損失張量
        tsne_loss_tensor = torch.zeros(input_features.size(0), device=self.device)
        
        # 迭代batch中的每個樣本以提取對角線相似度值
        for i in range(input_features.size(0)):
            tsne_loss_tensor[i] = 1 - similarity_matrix[i, i]  # 提取對角線元素
        
        tsne_loss = tsne_loss_tensor.mean() # 計算batch平均損失

        return tsne_loss, (None, None)  # 返回 tsne_loss，無需 TSNE 結果
    
    def compute_enhanced_loss(self, output, target, enhanced_features, target_features):
        """結合音訊重建損失和t-SNE特徵空間損失"""
        # 1. 音訊重建損失
        recon_loss = compute_voice_focused_loss(output, target, self.device)
        
        # 2. t-SNE 特徵空間損失
        tsne_distance, _ = self.calculate_tsne_distance(enhanced_features, target_features)
        
        # 組合損失
        total_loss = 0.7 * recon_loss + 0.3 * tsne_distance
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'tsne_loss': tsne_distance.item()
        }
    
    def train(self, train_loader, optimizer, scheduler):
        """訓練循環"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        # 移除tsne_dir相關代碼
        # tsne_dir = os.path.join(self.config['save_dir'], 'tsne_plots')
        # os.makedirs(tsne_dir, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                              desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
            
            # 在最後一個 epoch 保存所有音頻
            is_final_epoch = (epoch == self.config['epochs'] - 1)
            
            # 如果是最後一輪，創建特定目錄
            if is_final_epoch:
                final_epoch_dir = os.path.join(self.config['save_dir'], 'final_epoch')
                os.makedirs(final_epoch_dir, exist_ok=True)
                print(f"\nProcessing final epoch {epoch+1}, saving all samples...")
            
            for batch_idx, (input_wav, target_wav) in progress_bar:
                try:
                    # 注意：這裡input_wav和target_wav已經經過process_audio的預處理
                    # input_wav已經normalize=True
                    # target_wav已經normalize=False
                    
                    # 確保形狀正確
                    input_wav = ensure_shape(input_wav)
                    target_wav = ensure_shape(target_wav)
                    
                    # 移動到設備
                    input_wav = input_wav.to(self.device)
                    target_wav = target_wav.to(self.device)
                    
                    # 均值標準化
                    input_wav = (input_wav - input_wav.mean()) / (input_wav.std() + 1e-6)
                    target_wav = (target_wav - target_wav.mean()) / (target_wav.std() + 1e-6)
                    
                    # 輸出預處理信息用於調試
                    if batch_idx == 0 and epoch == 0:
                        print("\nInput wav shape:", input_wav.shape)
                        print("Input wav range:", input_wav.min().item(), "to", input_wav.max().item())
                    
                    # 數據處理流程：
                    
                    # 1. 確保維度正確的 target 特徵提取
                    with torch.no_grad():
                        target_features = self.model.feature_extractor.encodec.encoder(target_wav)  # [B, 512, T]
                
                    # 2. Input 處理流程 - 確保維度正確
                    with torch.no_grad():
                        input_features = self.model.feature_extractor.encodec.encoder(input_wav)    # [B, 512, T]
                
                    # 3. 特徵增強 - 保持維度不變
                    enhanced_features = input_features
                    for layer in [self.model.feature_extractor.adapter_conv,
                                self.model.feature_extractor.adapter_bn,
                                self.model.feature_extractor.residual_blocks,
                                self.model.feature_extractor.out_conv]:
                        enhanced_features = layer(enhanced_features)  # 維持 [B, 512, T]
                
                    # 4. 解碼
                    output = self.model.feature_extractor.encodec.decoder(enhanced_features)
                    
                    # 3. 計算組合損失
                    loss, loss_details = self.compute_enhanced_loss(
                        output.squeeze(1),
                        target_wav.squeeze(1),
                        enhanced_features,
                        target_features
                    )
                    
                    # 註解掉T-SNE視覺化保存部分
                    # if batch_idx % self.config['vis_interval'] == 0:
                    #     distance, (input_tsne, target_tsne) = self.calculate_tsne_distance(
                    #         enhanced_features,
                    #         target_features
                    #     )
                    #     tsne_path = os.path.join(
                    #         tsne_dir, 
                    #         f'tsne_epoch_{epoch+1}_batch_{batch_idx}.png'
                    #     )
                    #     self.visualize_tsne(input_tsne, target_tsne, tsne_path, epoch+1)
                    
                    # 在最後一輪保存所有batch的所有樣本
                    if is_final_epoch:
                        batch_dir = os.path.join(final_epoch_dir, f'batch_{batch_idx}')
                        os.makedirs(batch_dir, exist_ok=True)
                        
                        # 保存每個batch中的所有樣本
                        with torch.no_grad():
                            for j in range(output.size(0)):
                                sample_dir = os.path.join(batch_dir, f'sample_{j}')
                                os.makedirs(sample_dir, exist_ok=True)
                                
                                # 保存音頻和頻譜圖
                                for audio, prefix in [
                                    (input_wav[j], 'input'),
                                    (output[j], 'output'),
                                    (target_wav[j], 'target')
                                ]:
                                    # 音頻文件
                                    audio_path = os.path.join(sample_dir, f'{prefix}.wav')
                                    torchaudio.save(audio_path, audio.cpu(), 24000)
                                    
                                    # 頻譜圖
                                    spec_path = os.path.join(sample_dir, f'{prefix}_spec.png')
                                    plot_spectrograms(
                                        audio,
                                        spec_path,
                                        self.device,
                                        title=f'Final Epoch - Batch {batch_idx} - Sample {j} - {prefix}'
                                    )
                                
                                # 註解掉最後一輪的TSNE特徵視覺化
                                # if j == 0:  # 每個batch只保存第一個樣本的特徵視覺化
                                #     tsne_path = os.path.join(batch_dir, f'tsne_features.png')
                                #     distance, (input_tsne, target_tsne) = self.calculate_tsne_distance(
                                #         enhanced_features[j:j+1],
                                #         target_features[j:j+1]
                                #     )
                                #     self.visualize_tsne(
                                #         input_tsne, 
                                #         target_tsne, 
                                #         tsne_path,
                                #         f"Final Epoch - Batch {batch_idx}"
                                #     )
                                
                                print(f"Saved sample {j+1} from batch {batch_idx}")
                    
                    # 反向傳播 (只更新特徵增強層)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 更新進度條，修正這裡的引用
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'tsne_loss': f'{loss_details["tsne_loss"]:.4f}'  # 修正鍵名
                    })
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Epoch結束後的處理
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.6f}")
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    self.model, optimizer, epoch+1, avg_loss,
                    self.config['save_dir'], is_best=True
                )
                print(f"New best model saved! Loss: {best_loss:.6f}")

def collate_fn(batch):
    """處理不同長度的音頻批次"""
    input_wavs = [item[0] for item in batch]
    target_wavs = [item[1] for item in batch]
    
    # 找出最長的音訊長度
    max_len = max(max(wav.size(-1) for wav in input_wavs),
                 max(wav.size(-1) for wav in target_wavs))
    
    # 將所有音訊補齊到相同長度
    padded_inputs = []
    padded_targets = []
    
    for input_wav, target_wav in zip(input_wavs, target_wavs):
        if input_wav.size(-1) < max_len:
            input_wav = torch.nn.functional.pad(input_wav, (0, max_len - input_wav.size(-1)))
        if target_wav.size(-1) < max_len:
            target_wav = torch.nn.functional.pad(target_wav, (0, max_len - target_wav.size(-1)))
        padded_inputs.append(input_wav)
        padded_targets.append(target_wav)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)

def main():
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",
        'pretrained_path': "./self_output2/best_model.pth",  # 使用test2.py訓練的模型
        'save_dir': './tsne_output',
        'epochs': 800,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.001,
        'scheduler_patience': 5,
        'scheduler_factor': 0.7,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'feature_scale': 1.5,
        'tsne_weight': 0.2,  # T-SNE損失的權重
        # 移除不需要的vis_interval配置
        # 'vis_interval': 50   # T-SNE可視化間隔
    }
    
    # 初始化訓練器
    trainer = EnhancedTrainer(config)
    
    # 使用 try2.py 中的 AudioDataset
    dataset = AudioDataset(
        input_dirs=[           # 改為列表形式，包含多個輸入目錄
            "./box",
            "./plastic",
            "./papercup"
        ],    
        target_dir="./box2"  # target目錄保持不變
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn  # 添加自定義的 collate_fn
    )
    
    # 優化器和調度器
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # 開始訓練
    trainer.train(train_loader, optimizer, scheduler)

if __name__ == "__main__":
    main()
