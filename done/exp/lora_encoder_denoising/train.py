"""
LoRA Encoder Denoising Training Script

使用 Teacher-Student 架構訓練去噪 LoRA 模型
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import json
from datetime import datetime

# 添加必要路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # 添加當前目錄

try:
    from .model import TeacherStudentModel
    from .losses import EncoderDistillationLoss
    from .data import create_dataloaders
    from .config import get_train_config, WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
    from .wavtok_lora_patch import apply_lora_patch
except ImportError:
    from model import TeacherStudentModel
    from losses import EncoderDistillationLoss
    from data import create_dataloaders
    from config import get_train_config, WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
    from wavtok_lora_patch import apply_lora_patch


class Trainer:
    """
    訓練器類別，處理完整的訓練流程
    """

    def __init__(self, config, exp_name):
        self.config = config
        self.exp_name = exp_name

        # 設置裝置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 創建輸出目錄
        self.setup_directories()

        # 載入 distance matrix
        self.distance_matrix = self.load_distance_matrix()

        # 套用 WavTokenizer-LoRA 相容性 patch
        apply_lora_patch()

        # 創建模型
        self.model = self.create_model()

        # 創建 dataloaders
        self.train_loader, self.val_loader = create_dataloaders(config)

        # 創建 loss function
        self.criterion = EncoderDistillationLoss(
            feature_loss_weight=config.feature_loss_weight,
            distance_loss_weight=config.distance_loss_weight,
            vq_loss_weight=config.vq_loss_weight
        )

        # 創建 optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 創建 learning rate scheduler (cosine with warmup)
        self.scheduler = self.create_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # 訓練狀態
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_checkpoints = []  # List of (val_loss, checkpoint_path) tuples

    def setup_directories(self):
        """創建必要的目錄"""
        self.exp_dir = Path('experiments') / self.exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # 保存 config (轉換 Path 為 string)
        config_path = self.exp_dir / 'config.json'
        config_dict = vars(self.config).copy()
        # 轉換所有 Path 對象為字符串
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, list) and value and isinstance(value[0], Path):
                config_dict[key] = [str(v) for v in value]
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Experiment directory: {self.exp_dir}")

    def load_distance_matrix(self):
        """載入 VQ codebook distance matrix"""
        print(f"Loading distance matrix from {DISTANCE_MATRIX}...")
        distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True)
        distance_matrix = distance_matrix.to(self.device)
        print(f"Distance matrix shape: {distance_matrix.shape}")
        return distance_matrix

    def create_model(self):
        """創建 Teacher-Student 模型"""
        print(f"\nCreating Teacher-Student model...")
        print(f"  WavTokenizer config: {WAVTOK_CONFIG}")
        print(f"  WavTokenizer checkpoint: {WAVTOK_CKPT}")
        print(f"  LoRA rank: {self.config.lora_rank}")
        print(f"  LoRA alpha: {self.config.lora_alpha}")

        model = TeacherStudentModel(
            wavtok_config=WAVTOK_CONFIG,
            wavtok_ckpt=WAVTOK_CKPT,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
        ).to(self.device)

        # 打印可訓練參數統計
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.4f}%")

        return model

    def create_scheduler(self):
        """創建 learning rate scheduler with cosine annealing + warmup"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                return self.config.min_lr / self.config.learning_rate + \
                       (1 - self.config.min_lr / self.config.learning_rate) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch):
        """訓練一個 epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_distance_loss = 0.0
        epoch_vq_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy_audio = batch['noisy_audio'].to(self.device)  # (B, T)
            clean_audio = batch['clean_audio'].to(self.device)  # (B, T)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                output = self.model(noisy_audio, clean_audio)
                loss, loss_dict = self.criterion(output, self.distance_matrix)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # 統計
            epoch_loss += loss.item()
            epoch_feature_loss += loss_dict['feature_loss']
            epoch_distance_loss += loss_dict['distance_loss']
            epoch_vq_loss += loss_dict['vq_loss']

            # 更新進度條
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Tensorboard logging (每 N 步)
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/feature_loss', loss_dict['feature_loss'], self.global_step)
                self.writer.add_scalar('train/distance_loss', loss_dict['distance_loss'], self.global_step)
                self.writer.add_scalar('train/vq_loss', loss_dict['vq_loss'], self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

            self.global_step += 1

        # Epoch 統計
        n_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / n_batches,
            'feature_loss': epoch_feature_loss / n_batches,
            'distance_loss': epoch_distance_loss / n_batches,
            'vq_loss': epoch_vq_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, epoch):
        """驗證"""
        self.model.eval()

        val_loss = 0.0
        val_feature_loss = 0.0
        val_distance_loss = 0.0
        val_vq_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validating"):
            noisy_audio = batch['noisy_audio'].to(self.device)
            clean_audio = batch['clean_audio'].to(self.device)

            with autocast(enabled=self.config.use_amp):
                output = self.model(noisy_audio, clean_audio)
                loss, loss_dict = self.criterion(output, self.distance_matrix)

            val_loss += loss.item()
            val_feature_loss += loss_dict['feature_loss']
            val_distance_loss += loss_dict['distance_loss']
            val_vq_loss += loss_dict['vq_loss']

        n_batches = len(self.val_loader)

        metrics = {
            'loss': val_loss / n_batches,
            'feature_loss': val_feature_loss / n_batches,
            'distance_loss': val_distance_loss / n_batches,
            'vq_loss': val_vq_loss / n_batches,
        }

        # Tensorboard logging
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

        return metrics

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': vars(self.config),
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存 latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # 保存 best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best checkpoint (val_loss={val_metrics['loss']:.4f})")

        # Top-K checkpoints management
        val_loss = val_metrics['loss']
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch:03d}_loss_{val_loss:.4f}.pt'
        torch.save(checkpoint, checkpoint_path)

        self.best_checkpoints.append((val_loss, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by loss

        # 保留 top-K
        while len(self.best_checkpoints) > self.config.save_top_k:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists() and old_path != best_path:
                old_path.unlink()

    def train(self):
        """主訓練循環"""
        print(f"\n{'='*70}")
        print(f"{'Starting Training':^70}")
        print(f"{'='*70}")
        print(f"Experiment: {self.exp_name}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # 訓練
            train_metrics = self.train_epoch(epoch)

            # 驗證
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.validate(epoch)

                # 打印統計
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Feature Loss: {val_metrics['feature_loss']:.4f}")
                print(f"  Val Distance Loss: {val_metrics['distance_loss']:.4f}")
                print(f"  Val VQ Loss: {val_metrics['vq_loss']:.4f}")

                # 保存 checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']

                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            else:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

        print(f"\n{'='*70}")
        print(f"{'Training Completed!':^70}")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='LoRA Encoder Denoising Training')

    # Experiment
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name (用於保存 checkpoints 和 logs)')

    # Training
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')

    # Loss weights
    parser.add_argument('--feature_loss_weight', type=float, default=1.0,
                       help='Feature loss weight')
    parser.add_argument('--distance_loss_weight', type=float, default=0.1,
                       help='Distance loss weight')
    parser.add_argument('--vq_loss_weight', type=float, default=0.01,
                       help='VQ loss weight')

    # Misc
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create config
    config = get_train_config(**vars(args))

    # Create trainer
    trainer = Trainer(config, args.exp_name)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
