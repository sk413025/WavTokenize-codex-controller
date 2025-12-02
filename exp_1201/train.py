"""
exp_1201: Soft Distance Loss Training Script

改進 exp_1128:
- 使用可微的 Soft Distance Loss (softmax 取代 argmax)
- 新增頻譜圖儲存功能
- 每 10 epoch 儲存 loss 圖、音檔、頻譜圖
"""

import torch
import torch.nn as nn
import torchaudio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import json
import numpy as np
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

        # 載入 codebook (for Soft Distance Loss)
        self.codebook = self.load_codebook()

        # 創建 loss function (支持 soft/gumbel/ste/ce/margin 模式)
        self.criterion = EncoderDistillationLoss(
            feature_loss_weight=config.feature_loss_weight,
            soft_dist_loss_weight=config.soft_dist_loss_weight,
            vq_loss_weight=config.vq_loss_weight,
            temperature=config.temperature,
            distance_loss_mode=getattr(config, 'distance_loss_mode', 'gumbel'),
            gumbel_hard=getattr(config, 'gumbel_hard', True),
            margin=getattr(config, 'margin', 1.0),
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )
        print(f"Distance loss mode: {getattr(config, 'distance_loss_mode', 'gumbel')}")

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

        # 訓練歷史記錄 (用於繪圖)
        self.history = {
            'train_loss': [], 'train_feature_loss': [], 'train_distance_loss': [],
            'train_vq_loss': [], 'train_token_acc': [],
            'val_loss': [], 'val_feature_loss': [], 'val_distance_loss': [],
            'val_vq_loss': [], 'val_token_acc': [],
            'epochs': [], 'learning_rates': []
        }

    def setup_directories(self):
        """創建必要的目錄"""
        self.exp_dir = Path('experiments') / self.exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.audio_dir = self.exp_dir / 'audio_samples'
        self.plot_dir = self.exp_dir / 'plots'
        self.spectrogram_dir = self.exp_dir / 'spectrograms'  # 新增頻譜圖目錄

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.spectrogram_dir.mkdir(exist_ok=True)

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

    def load_codebook(self):
        """載入 VQ codebook (for Soft Distance Loss)"""
        print("Loading VQ codebook from model...")
        try:
            codebook = self.model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
            codebook = codebook.to(self.device)
            print(f"Codebook shape: {codebook.shape}")  # (4096, 512)
            return codebook
        except AttributeError:
            print("Warning: Cannot access codebook, using random initialization")
            return torch.randn(4096, 512, device=self.device)

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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 關鍵修正 v3：凍結 Student VQ 的 EMA 更新 (同時保持 STE 梯度傳遞)
        #
        # 問題分析：
        #   1. VQ 的 EMA 更新和 STE 梯度傳遞都依賴 self.training flag
        #   2. 設置 eval() 會同時關閉兩者，但 STE 是梯度反傳必需的！
        #      - core_vq.py:217-229: EMA 更新 (我們想關閉)
        #      - core_vq.py:301-302: STE 梯度傳遞 (我們需要保持!)
        #
        # 解決方案：
        #   保持 training=True (讓 STE 正常運作)
        #   但在每次 forward 前後恢復 codebook，抵消 EMA 更新
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # 獲取 Student VQ 的 codebook 引用
        student_vq = self.model.student.base_model.model.feature_extractor.encodec.quantizer.vq.layers[0]
        codebook_ref = student_vq._codebook

        # 保存原始 codebook 狀態 (用於每次 forward 後恢復)
        frozen_codebook = codebook_ref.embed.data.clone()
        frozen_cluster_size = codebook_ref.cluster_size.data.clone()
        frozen_embed_avg = codebook_ref.embed_avg.data.clone()

        epoch_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_distance_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_token_acc = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy_audio = batch['noisy_audio'].to(self.device)  # (B, T)
            clean_audio = batch['clean_audio'].to(self.device)  # (B, T)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                output = self.model(noisy_audio, clean_audio)
                # 使用 Soft Distance Loss (需要 codebook)
                loss, loss_dict = self.criterion(output, self.distance_matrix, self.codebook)

            # ━━━ 恢復 codebook (抵消 EMA 更新) ━━━
            codebook_ref.embed.data.copy_(frozen_codebook)
            codebook_ref.cluster_size.data.copy_(frozen_cluster_size)
            codebook_ref.embed_avg.data.copy_(frozen_embed_avg)

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

            # 統計 (使用 soft_distance_loss)
            epoch_loss += loss.item()
            epoch_feature_loss += loss_dict['feature_loss']
            epoch_distance_loss += loss_dict.get('soft_distance_loss', loss_dict.get('hard_distance_loss', 0.0))
            epoch_vq_loss += loss_dict['vq_loss']
            epoch_token_acc += loss_dict['code_match_rate']

            # 更新進度條
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_dict['code_match_rate']*100:.1f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Tensorboard logging (每 N 步)
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/feature_loss', loss_dict['feature_loss'], self.global_step)
                self.writer.add_scalar('train/soft_distance_loss', loss_dict.get('soft_distance_loss', 0.0), self.global_step)
                self.writer.add_scalar('train/hard_distance_loss', loss_dict.get('hard_distance_loss', 0.0), self.global_step)
                self.writer.add_scalar('train/vq_loss', loss_dict['vq_loss'], self.global_step)
                self.writer.add_scalar('train/token_acc', loss_dict['code_match_rate'], self.global_step)
                self.writer.add_scalar('train/soft_code_entropy', loss_dict.get('soft_code_entropy', 0.0), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

            self.global_step += 1

        # Epoch 統計
        n_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / n_batches,
            'feature_loss': epoch_feature_loss / n_batches,
            'distance_loss': epoch_distance_loss / n_batches,
            'vq_loss': epoch_vq_loss / n_batches,
            'token_acc': epoch_token_acc / n_batches,
        }

    @torch.no_grad()
    def validate(self, epoch):
        """驗證"""
        self.model.eval()

        val_loss = 0.0
        val_feature_loss = 0.0
        val_distance_loss = 0.0
        val_vq_loss = 0.0
        val_token_acc = 0.0

        for batch in tqdm(self.val_loader, desc="Validating"):
            noisy_audio = batch['noisy_audio'].to(self.device)
            clean_audio = batch['clean_audio'].to(self.device)

            with autocast(enabled=self.config.use_amp):
                output = self.model(noisy_audio, clean_audio)
                # 使用 Soft Distance Loss (需要 codebook)
                loss, loss_dict = self.criterion(output, self.distance_matrix, self.codebook)

            val_loss += loss.item()
            val_feature_loss += loss_dict['feature_loss']
            val_distance_loss += loss_dict.get('soft_distance_loss', loss_dict.get('hard_distance_loss', 0.0))
            val_vq_loss += loss_dict['vq_loss']
            val_token_acc += loss_dict['code_match_rate']

        n_batches = len(self.val_loader)

        metrics = {
            'loss': val_loss / n_batches,
            'feature_loss': val_feature_loss / n_batches,
            'distance_loss': val_distance_loss / n_batches,
            'vq_loss': val_vq_loss / n_batches,
            'token_acc': val_token_acc / n_batches,
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
        best_path = self.checkpoint_dir / 'best.pt'  # 修復：確保 best_path 有定義
        while len(self.best_checkpoints) > self.config.save_top_k:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists() and old_path != best_path:
                old_path.unlink()

    def plot_training_curves(self, epoch):
        """
        繪製訓練曲線並保存

        Args:
            epoch: 當前 epoch 數
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建 2x3 子圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Curves - Epoch {epoch+1}', fontsize=14)
        
        epochs = self.history['epochs']
        
        # Loss 曲線
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True)
        
        # Feature Loss
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_feature_loss'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_feature_loss'], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Loss')
        ax.set_title('Feature Loss (MSE)')
        ax.legend()
        ax.grid(True)
        
        # Distance Loss
        ax = axes[0, 2]
        ax.plot(epochs, self.history['train_distance_loss'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_distance_loss'], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance Loss')
        ax.set_title('Distance Loss')
        ax.legend()
        ax.grid(True)
        
        # VQ Loss
        ax = axes[1, 0]
        ax.plot(epochs, self.history['train_vq_loss'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_vq_loss'], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('VQ Loss')
        ax.set_title('VQ Loss')
        ax.legend()
        ax.grid(True)
        
        # Token Accuracy
        ax = axes[1, 1]
        ax.plot(epochs, [acc*100 for acc in self.history['train_token_acc']], 'b-', label='Train')
        ax.plot(epochs, [acc*100 for acc in self.history['val_token_acc']], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Token Accuracy (%)')
        ax.set_title('Token Accuracy')
        ax.legend()
        ax.grid(True)
        
        # Learning Rate
        ax = axes[1, 2]
        ax.plot(epochs, self.history['learning_rates'], 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存圖表
        plot_path = self.plot_dir / f'training_curves_epoch_{epoch+1:03d}_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved training curves to {plot_path}")
        
        # 也保存訓練歷史為 JSON
        history_path = self.exp_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def save_audio_samples(self, epoch, num_samples=3):
        """
        保存訓練和驗證音頻樣本，包含:
        - noisy: 原始噪音音頻
        - clean: 目標乾淨音頻
        - student_pred: Student 模型預測的音頻 (noisy → student encoder → decoder)
        - teacher_recon: Teacher 模型重建的音頻 (clean → teacher encoder → decoder)

        Args:
            epoch: 當前 epoch 數
            num_samples: 每個集合保存的樣本數
        """
        self.model.eval()
        epoch_audio_dir = self.audio_dir / f'epoch_{epoch+1:03d}'
        epoch_audio_dir.mkdir(exist_ok=True)

        sample_rate = 24000

        def save_sample_set(data_iter, prefix, num):
            """保存一組樣本 (noisy, clean, student_pred, teacher_recon)"""
            for i in range(num):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                noisy_audio = batch['noisy_audio'][:1].to(self.device)
                clean_audio = batch['clean_audio'][:1].to(self.device)

                # 確保音頻是 2D (C, T) 格式
                if noisy_audio.dim() == 1:
                    noisy_audio = noisy_audio.unsqueeze(0)
                if clean_audio.dim() == 1:
                    clean_audio = clean_audio.unsqueeze(0)

                # 1. 保存原始 noisy 音頻
                noisy_path = epoch_audio_dir / f'{prefix}_{i+1}_noisy.wav'
                torchaudio.save(str(noisy_path), noisy_audio.cpu(), sample_rate)

                # 2. 保存 clean 音頻 (target)
                clean_path = epoch_audio_dir / f'{prefix}_{i+1}_clean.wav'
                torchaudio.save(str(clean_path), clean_audio.cpu(), sample_rate)

                # 3. Student prediction: noisy → student encoder → decoder
                # 使用正確的 API: feature_extractor(audio, bandwidth_id=0)
                # 輸入是 (B, T) 格式
                noisy_input = noisy_audio.squeeze(0) if noisy_audio.shape[0] == 1 else noisy_audio
                if noisy_input.dim() == 1:
                    noisy_input = noisy_input.unsqueeze(0)  # (1, T)
                student_features, _, _ = self.model.student.feature_extractor(noisy_input, bandwidth_id=0)
                # 使用 teacher 的 decoder (因為 student 只有 encoder 有 LoRA)
                # decode 需要 bandwidth_id 參數
                student_pred_audio = self.model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(self.device))
                if student_pred_audio.dim() == 3:
                    student_pred_audio = student_pred_audio.squeeze(1)
                student_pred_path = epoch_audio_dir / f'{prefix}_{i+1}_student_pred.wav'
                torchaudio.save(str(student_pred_path), student_pred_audio.cpu(), sample_rate)

                # 4. Teacher reconstruction: clean → teacher encoder → decoder
                clean_input = clean_audio.squeeze(0) if clean_audio.shape[0] == 1 else clean_audio
                if clean_input.dim() == 1:
                    clean_input = clean_input.unsqueeze(0)  # (1, T)
                teacher_features, _, _ = self.model.teacher.feature_extractor(clean_input, bandwidth_id=0)
                teacher_recon_audio = self.model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(self.device))
                if teacher_recon_audio.dim() == 3:
                    teacher_recon_audio = teacher_recon_audio.squeeze(1)
                teacher_recon_path = epoch_audio_dir / f'{prefix}_{i+1}_teacher_recon.wav'
                torchaudio.save(str(teacher_recon_path), teacher_recon_audio.cpu(), sample_rate)

        try:
            # 保存訓練樣本
            train_iter = iter(self.train_loader)
            save_sample_set(train_iter, 'train', min(num_samples, len(self.train_loader)))

            # 保存驗證樣本
            val_iter = iter(self.val_loader)
            save_sample_set(val_iter, 'val', min(num_samples, len(self.val_loader)))

            print(f"  🔊 Saved audio samples to {epoch_audio_dir}")
            print(f"      (noisy, clean, student_pred, teacher_recon) x {num_samples} train + {num_samples} val")
        except Exception as e:
            import traceback
            print(f"  ⚠️ Failed to save audio samples: {e}")
            traceback.print_exc()

    @torch.no_grad()
    def save_spectrograms(self, epoch, num_samples=3):
        """
        保存頻譜圖比較

        每個樣本生成一張圖，包含:
        - noisy 音頻的頻譜圖
        - clean 音頻的頻譜圖
        - student_pred 的頻譜圖
        - teacher_recon 的頻譜圖
        """
        self.model.eval()
        epoch_spec_dir = self.spectrogram_dir / f'epoch_{epoch+1:03d}'
        epoch_spec_dir.mkdir(exist_ok=True)

        sample_rate = 24000
        n_fft = 1024
        hop_length = 256

        def compute_spectrogram(audio):
            """計算 mel spectrogram"""
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.dim() == 3:
                audio = audio.squeeze(1)

            # 使用 torchaudio 計算 mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=80,
            ).to(audio.device)

            mel = mel_transform(audio)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
            return mel_db[0].cpu().numpy()  # (n_mels, time)

        def save_comparison_spectrogram(noisy, clean, student_pred, teacher_recon, path, title_prefix):
            """保存 4 張頻譜圖的比較"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{title_prefix} - Spectrogram Comparison', fontsize=14)

            specs = [
                (noisy, 'Noisy Input', axes[0, 0]),
                (clean, 'Clean Target', axes[0, 1]),
                (student_pred, 'Student Prediction', axes[1, 0]),
                (teacher_recon, 'Teacher Reconstruction', axes[1, 1]),
            ]

            for audio, label, ax in specs:
                spec = compute_spectrogram(audio)
                im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
                ax.set_title(label)
                ax.set_xlabel('Time')
                ax.set_ylabel('Mel Bin')
                plt.colorbar(im, ax=ax, format='%+2.0f dB')

            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()

        def process_samples(data_iter, prefix, num):
            """處理樣本並保存頻譜圖"""
            for i in range(num):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                noisy_audio = batch['noisy_audio'][:1].to(self.device)
                clean_audio = batch['clean_audio'][:1].to(self.device)

                # 確保格式為 (B, T)
                if noisy_audio.dim() == 1:
                    noisy_audio = noisy_audio.unsqueeze(0)
                if clean_audio.dim() == 1:
                    clean_audio = clean_audio.unsqueeze(0)

                # Student prediction: 使用正確的 feature_extractor API
                student_features, _, _ = self.model.student.feature_extractor(noisy_audio, bandwidth_id=0)
                student_pred = self.model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(self.device))

                # Teacher reconstruction
                teacher_features, _, _ = self.model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
                teacher_recon = self.model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(self.device))

                # 保存頻譜圖
                spec_path = epoch_spec_dir / f'{prefix}_{i+1}_comparison.png'
                save_comparison_spectrogram(
                    noisy_audio, clean_audio, student_pred, teacher_recon,
                    spec_path, f'{prefix.capitalize()} Sample {i+1}'
                )

        try:
            # Train 樣本
            train_iter = iter(self.train_loader)
            process_samples(train_iter, 'train', min(num_samples, len(self.train_loader)))

            # Val 樣本
            val_iter = iter(self.val_loader)
            process_samples(val_iter, 'val', min(num_samples, len(self.val_loader)))

            print(f"  📊 Saved spectrograms to {epoch_spec_dir}")
        except Exception as e:
            import traceback
            print(f"  ⚠️ Failed to save spectrograms: {e}")
            traceback.print_exc()

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

                # 更新訓練歷史
                self.history['epochs'].append(epoch + 1)
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_feature_loss'].append(train_metrics['feature_loss'])
                self.history['train_distance_loss'].append(train_metrics['distance_loss'])
                self.history['train_vq_loss'].append(train_metrics['vq_loss'])
                self.history['train_token_acc'].append(train_metrics['token_acc'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_feature_loss'].append(val_metrics['feature_loss'])
                self.history['val_distance_loss'].append(val_metrics['distance_loss'])
                self.history['val_vq_loss'].append(val_metrics['vq_loss'])
                self.history['val_token_acc'].append(val_metrics['token_acc'])
                # 確保 learning rate 是 float 而非 Tensor
                lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rates'].append(float(lr) if hasattr(lr, 'item') else lr)

                # 打印統計
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Token Acc: {train_metrics['token_acc']*100:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Token Acc: {val_metrics['token_acc']*100:.2f}%")
                print(f"  Val Feature Loss: {val_metrics['feature_loss']:.4f}")
                print(f"  Val Distance Loss: {val_metrics['distance_loss']:.4f}")
                print(f"  Val VQ Loss: {val_metrics['vq_loss']:.4f}")

                # 保存 checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']

                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=is_best)
                    # 繪製訓練曲線
                    self.plot_training_curves(epoch)
                    # 保存音頻樣本 (train + val)
                    num_samples = getattr(self.config, 'num_audio_samples', 3)
                    self.save_audio_samples(epoch, num_samples=num_samples)
                    # 保存頻譜圖 (train + val)
                    if getattr(self.config, 'save_spectrograms', True):
                        self.save_spectrograms(epoch, num_samples=num_samples)
            else:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['token_acc']*100:.2f}%")

        # 訓練結束時保存最終曲線
        self.plot_training_curves(self.config.num_epochs - 1)
        # 最終音頻和頻譜圖
        self.save_audio_samples(self.config.num_epochs - 1, num_samples=getattr(self.config, 'num_audio_samples', 3))
        if getattr(self.config, 'save_spectrograms', True):
            self.save_spectrograms(self.config.num_epochs - 1, num_samples=getattr(self.config, 'num_audio_samples', 3))

        print(f"\n{'='*70}")
        print(f"{'Training Completed!':^70}")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"Plots saved to: {self.plot_dir}")
        print(f"Audio samples saved to: {self.audio_dir}")
        print(f"Spectrograms saved to: {self.spectrogram_dir}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='exp_1201: Soft Distance Loss Training')

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
    parser.add_argument('--lora_rank', type=int, default=64,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=128,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')

    # Loss weights (使用 Soft Distance Loss!)
    parser.add_argument('--feature_loss_weight', type=float, default=1.0,
                       help='Feature loss weight')
    parser.add_argument('--soft_dist_loss_weight', type=float, default=0.1,
                       help='Distance loss weight (可微!)')
    parser.add_argument('--vq_loss_weight', type=float, default=0.0,
                       help='VQ loss weight (保持 0，codebook 凍結)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax temperature for distance loss')
    parser.add_argument('--distance_loss_mode', type=str, default='gumbel',
                       choices=['soft', 'gumbel', 'ste', 'ce', 'margin'],
                       help='Distance loss mode: soft, gumbel, ste, ce (Cross-Entropy), margin')
    parser.add_argument('--gumbel_hard', type=bool, default=True,
                       help='Use hard codes in Gumbel mode (only for gumbel mode)')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for margin loss mode')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing for CE loss mode')

    # Logging and Saving
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log every N batches')

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
