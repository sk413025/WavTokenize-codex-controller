#!/usr/bin/env python3
"""
Token Denoising Transformer 訓練腳本
完全凍結 WavTokenizer Codebook，只訓練 Transformer

使用方式:
    python train_token_denoising.py --config <config> --model_path <model> --output_dir <dir>
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset
from ttt2 import collate_fn
from token_denoising_transformer import TokenDenoisingTransformer, PositionalEncoding

# 創建 logger（不在這裡配置，在 main 中配置）
logger = logging.getLogger(__name__)


class TokenDenoisingDataset(Dataset):
    """Token Denoising 數據集
    
    將音訊轉換為 Token IDs，用於 Transformer 訓練
    """
    
    def __init__(self, audio_dataset, wavtokenizer, device='cuda'):
        """
        Args:
            audio_dataset: AudioDataset 實例
            wavtokenizer: WavTokenizer 實例 (用於編碼音訊)
            device: 計算設備
        """
        self.audio_dataset = audio_dataset
        self.wavtokenizer = wavtokenizer
        self.device = device
        
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        """返回 noisy 和 clean 的 token IDs"""
        noisy_audio, clean_audio = self.audio_dataset[idx]
        
        # 移到設備
        noisy_audio = noisy_audio.to(self.device)
        clean_audio = clean_audio.to(self.device)
        
        # 編碼為 Token IDs
        with torch.no_grad():
            _, noisy_tokens = self.wavtokenizer.encode_infer(
                noisy_audio.unsqueeze(0), 
                bandwidth_id=torch.tensor([0], device=self.device)
            )
            _, clean_tokens = self.wavtokenizer.encode_infer(
                clean_audio.unsqueeze(0), 
                bandwidth_id=torch.tensor([0], device=self.device)
            )
            
            noisy_tokens = noisy_tokens[0].squeeze(0)  # (T,)
            clean_tokens = clean_tokens[0].squeeze(0)  # (T,)
        
        return noisy_tokens, clean_tokens


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (noisy_tokens, clean_tokens) in enumerate(progress_bar):
        noisy_tokens = noisy_tokens.to(device)
        clean_tokens = clean_tokens.to(device)
        
        # Forward pass
        logits = model(noisy_tokens, return_logits=True)  # (B, T, 4096)
        
        # Reshape for loss computation
        B, T, vocab_size = logits.shape
        logits_flat = logits.reshape(B * T, vocab_size)
        targets_flat = clean_tokens.reshape(B * T)
        
        # Compute loss
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Token accuracy
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == clean_tokens).sum().item()
        total_correct += correct
        total_tokens += B * T
        
        # Update progress bar
        current_acc = (total_correct / total_tokens) * 100
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """驗證一個 epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for noisy_tokens, clean_tokens in tqdm(dataloader, desc="Validation"):
            noisy_tokens = noisy_tokens.to(device)
            clean_tokens = clean_tokens.to(device)
            
            # Forward pass
            logits = model(noisy_tokens, return_logits=True)
            
            # Reshape for loss computation
            B, T, vocab_size = logits.shape
            logits_flat = logits.reshape(B * T, vocab_size)
            targets_flat = clean_tokens.reshape(B * T)
            
            # Compute loss
            loss = criterion(logits_flat, targets_flat)
            
            total_loss += loss.item()
            
            # Token accuracy
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == clean_tokens).sum().item()
            total_correct += correct
            total_tokens += B * T
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None, 
                   train_losses=None, val_losses=None, val_accuracies=None):
    """保存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    if train_losses is not None:
        checkpoint['train_losses'] = train_losses
    if val_losses is not None:
        checkpoint['val_losses'] = val_losses
    if val_accuracies is not None:
        checkpoint['val_accuracies'] = val_accuracies
        
    torch.save(checkpoint, save_path)
    logger.info(f"檢查點已保存到: {save_path}")


def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 損失曲線
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 準確率曲線
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Token Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"訓練歷史圖已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Token Denoising Transformer 訓練')
    
    # WavTokenizer 參數
    parser.add_argument('--config', type=str, 
                        default='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                        help='WavTokenizer 配置文件路徑')
    parser.add_argument('--model_path', type=str, 
                        default='models/wavtokenizer_large_speech_320_24k.ckpt',
                        help='WavTokenizer 預訓練模型路徑')
    
    # Transformer 參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 模型維度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力頭數')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='前饋網絡維度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=1000, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='權重衰減')
    parser.add_argument('--save_every', type=int, default=100, help='每幾個 epoch 保存一次')
    
    # 數據參數
    parser.add_argument('--output_dir', type=str, default='results/token_denoising_frozen_codebook',
                        help='輸出目錄')
    parser.add_argument('--val_speakers', nargs='+', default=['girl9', 'boy7'],
                        help='驗證集語者')
    parser.add_argument('--train_speakers', nargs='+', default=None,
                        help='訓練集語者')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=None,
                        help='每位語者最大句子數')
    
    # 設備參數
    parser.add_argument('--device', type=str, default='cuda', help='訓練設備')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定日誌（清除現有 handlers 並重新配置）
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除所有現有 handlers
    
    # 添加文件和控制台 handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("Token Denoising Transformer - Frozen Codebook Training")
    logger.info("=" * 80)
    
    # 設定設備
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 載入 WavTokenizer
    logger.info("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config, args.model_path)
    wavtokenizer.eval()
    wavtokenizer.to(device)
    
    # 凍結 WavTokenizer
    for param in wavtokenizer.parameters():
        param.requires_grad = False
    logger.info("✓ WavTokenizer 已凍結")
    
    # 提取 Codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")  # (4096, 512)
    
    # 準備數據集
    logger.info("準備音頻資料集...")
    
    # 獲取項目根目錄（train_token_denoising.py 的父目錄的父目錄）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dirs = [os.path.join(project_root, "data", "raw", "box")]
    target_dir = os.path.join(project_root, "data", "clean", "box2")
    
    # 準備允許的語者列表
    allowed_speakers = set()
    if args.train_speakers:
        allowed_speakers.update(args.train_speakers)
    if args.val_speakers:
        allowed_speakers.update(args.val_speakers)
    
    # 創建音頻數據集
    audio_dataset = AudioDataset(input_dirs, target_dir, 
                                max_sentences_per_speaker=args.max_sentences_per_speaker,
                                allowed_speakers=allowed_speakers if allowed_speakers else None)
    
    logger.info(f"數據集大小: {len(audio_dataset)} 個音頻對")
    
    # 按語者分割數據集
    train_indices = []
    val_indices = []
    
    for idx in range(len(audio_dataset)):
        # AudioDataset 使用 paired_files 屬性
        filename = audio_dataset.paired_files[idx]['input']
        # 文件名格式: nor_boy10_box_LDV_001.wav
        # 提取語者名稱 (第二個部分)
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 2:
            speaker = parts[1]  # boy10, girl9, etc.
        else:
            speaker = parts[0]  # fallback
        
        if speaker in args.val_speakers:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    
    logger.info(f"訓練集大小: {len(train_indices)}, 驗證集大小: {len(val_indices)}")
    
    # 創建 Token 數據集
    train_audio_dataset = torch.utils.data.Subset(audio_dataset, train_indices)
    val_audio_dataset = torch.utils.data.Subset(audio_dataset, val_indices)
    
    # 注意: TokenDenoisingDataset 會在 __getitem__ 中進行 Token 編碼
    # 這裡我們直接使用音頻數據集，在訓練循環中進行 Token 編碼
    
    # 創建數據載入器
    def token_collate_fn(batch):
        """Collate function for token data with dynamic padding
        
        WavTokenizer 返回的 discrete_codes 形狀為 (num_quantizers, B, T)
        我們只使用第一層量化器 (index 0)
        """
        noisy_tokens_list = []
        clean_tokens_list = []
        
        for noisy_audio, clean_audio, content_id in batch:
            # 移到設備
            noisy_audio = noisy_audio.to(device).unsqueeze(0)
            clean_audio = clean_audio.to(device).unsqueeze(0)
            
            # 編碼為 Token IDs
            with torch.no_grad():
                _, noisy_tokens = wavtokenizer.encode_infer(
                    noisy_audio, 
                    bandwidth_id=torch.tensor([0], device=device)
                )
                _, clean_tokens = wavtokenizer.encode_infer(
                    clean_audio, 
                    bandwidth_id=torch.tensor([0], device=device)
                )
                
                # discrete_codes 形狀: (num_quantizers, 1, seq_len)
                # 只使用第一層量化器: (1, seq_len)
                noisy_tokens_list.append(noisy_tokens[0])  # [1, seq_len]
                clean_tokens_list.append(clean_tokens[0])  # [1, seq_len]
        
        # 找出最大長度（考慮 noisy 和 clean）
        max_len = max(
            max(t.shape[1] for t in noisy_tokens_list),
            max(t.shape[1] for t in clean_tokens_list)
        )
        
        # Pad 所有 tokens 到相同長度
        padded_noisy = []
        padded_clean = []
        
        for noisy_t, clean_t in zip(noisy_tokens_list, clean_tokens_list):
            # Pad noisy tokens [1, seq_len] -> [max_len]
            curr_noisy = noisy_t.squeeze(0)  # [seq_len]
            if curr_noisy.shape[0] < max_len:
                pad_size = max_len - curr_noisy.shape[0]
                curr_noisy = torch.nn.functional.pad(curr_noisy, (0, pad_size), value=0)
            padded_noisy.append(curr_noisy)  # [max_len]
            
            # Pad clean tokens [1, seq_len] -> [max_len]
            curr_clean = clean_t.squeeze(0)  # [seq_len]
            if curr_clean.shape[0] < max_len:
                pad_size = max_len - curr_clean.shape[0]
                curr_clean = torch.nn.functional.pad(curr_clean, (0, pad_size), value=0)
            padded_clean.append(curr_clean)  # [max_len]
        
        # Stack tokens: [batch_size, seq_len]
        noisy_tokens_batch = torch.stack(padded_noisy, dim=0)
        clean_tokens_batch = torch.stack(padded_clean, dim=0)
        
        return noisy_tokens_batch, clean_tokens_batch
    
    train_loader = DataLoader(train_audio_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0, collate_fn=token_collate_fn)
    val_loader = DataLoader(val_audio_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0, collate_fn=token_collate_fn)
    
    # 創建模型
    logger.info("創建 Token Denoising Transformer...")
    model = TokenDenoisingTransformer(
        codebook=codebook,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 確認 Codebook 凍結
    assert not model.codebook.requires_grad, "Codebook 必須凍結！"
    
    # 計算參數量（正確方式：parameters 包含可訓練，buffers 包含凍結）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(b.numel() for b in model.buffers())  # Codebook 在 buffers 中
    total_params = trainable_params + frozen_params
    
    logger.info(f"模型總參數數量: {total_params:,}")
    logger.info(f"  - 可訓練參數 (Transformer + Output): {trainable_params:,}")
    logger.info(f"  - 凍結參數 (Codebook): {frozen_params:,}")
    logger.info(f"  - Codebook 形狀: {model.codebook.shape}")
    
    # 創建優化器和損失函數
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # 訓練循環
    logger.info("開始訓練...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # 驗證
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 更新學習率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, save_path, vars(args),
                          train_losses, val_losses, val_accuracies)
            logger.info(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 定期保存
        if epoch % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, save_path, vars(args),
                          train_losses, val_losses, val_accuracies)
        
        # 繪製訓練歷史
        if epoch % 10 == 0:
            plot_path = os.path.join(args.output_dir, 'training_history.png')
            plot_training_history(train_losses, val_losses, val_accuracies, plot_path)
    
    logger.info("訓練完成！")
    
    # 最終保存
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs, val_losses[-1], final_path, vars(args),
                   train_losses, val_losses, val_accuracies)


if __name__ == "__main__":
    main()
