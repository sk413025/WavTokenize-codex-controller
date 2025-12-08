"""
exp16: Feature Loss + Cross-Entropy Loss 訓練

核心洞察（來自 exp_1204 診斷）：
- mean_correct_distance: 3.75  (到正確 token 的距離)
- mean_min_distance: 0.45      (到最近錯誤 token 的距離)
- 問題：MSE 讓特徵「靠近」但不保證「跨過 Voronoi 邊界」

解決方案：
- Feature Loss: 確保特徵結構穩定（不要崩壞）
- CE Loss: 強迫特徵跨回正確的 Voronoi Cell

Loss = λ_feature * MSE(z_student, z_teacher) + λ_ce * CE(logits, teacher_tokens)

其中 logits = -||z_student - codebook||^2 (負距離作為 logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import math
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer

# 導入 exp_1201 的工具
from exp_1201.data import create_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1201.wavtok_lora_patch import apply_lora_patch

# 應用 LoRA patch
apply_lora_patch()


class TeacherStudentWithCE(nn.Module):
    """
    Teacher-Student 模型，支援 Cross-Entropy Loss

    關鍵：需要訪問 codebook 來計算 CE Loss
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Teacher: 凍結
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)

        # Student: LoRA
        print("Loading Student (with LoRA)...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_target_modules = [
            "feature_extractor.encodec.encoder.model.0.conv.conv",
            "feature_extractor.encodec.encoder.model.3.conv.conv",
            "feature_extractor.encodec.encoder.model.6.conv.conv",
            "feature_extractor.encodec.encoder.model.9.conv.conv",
        ]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()
        self.student = self.student.to(device)

        # 獲取 codebook (從 VQ layer)
        # WavTokenizer 的 codebook 在 feature_extractor.encodec.quantizer
        self.codebook = self._get_codebook()
        print(f"Codebook shape: {self.codebook.shape}")  # 應該是 (4096, 512)

    def _get_codebook(self):
        """從 Teacher 模型獲取 codebook"""
        # WavTokenizer 的結構：
        # quantizer (ResidualVectorQuantizer)
        #   └── vq (LanguageVectorQuantization)
        #       └── layers (ModuleList)
        #           └── [0] (VectorQuantization)
        #               └── codebook: (4096, 512)
        try:
            quantizer = self.teacher.feature_extractor.encodec.quantizer
            # 正確路徑：quantizer.vq.layers[0].codebook
            codebook = quantizer.vq.layers[0].codebook.detach()
            # shape: (num_codes, dim) = (4096, 512)
            return codebook
        except Exception as e:
            print(f"Warning: Could not get codebook directly: {e}")
            print("Trying alternative method...")
            # 備用方法：從 distance matrix 反推或使用已知路徑
            raise RuntimeError("Cannot access codebook. Please check WavTokenizer structure.")

    def forward(self, noisy_audio, clean_audio):
        """
        Forward pass - 分離 encoder 和 VQ 步驟，以獲取 VQ前 特徵

        Returns:
            dict with features, codes, and encoder outputs (VQ前)
        """
        # 確保格式
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        # Teacher: clean audio (frozen, no grad)
        with torch.no_grad():
            # 分步執行以獲取 VQ前 特徵
            clean_audio_3d = clean_audio.unsqueeze(1) if clean_audio.dim() == 2 else clean_audio
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio_3d)
            # VQ
            quantizer = self.teacher.feature_extractor.encodec.quantizer
            teacher_vq_result = quantizer(teacher_encoder_out, frame_rate=75, bandwidth=0.075)
            teacher_features = teacher_vq_result.quantized  # VQ後
            teacher_codes = teacher_vq_result.codes

        # Student: 手動分步執行
        # Step 1: Encoder (VQ前)
        noisy_audio_3d = noisy_audio.unsqueeze(1) if noisy_audio.dim() == 2 else noisy_audio
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio_3d)
        # shape: (B, C, T) where C=512

        # Step 2: VQ
        quantizer = self.student.feature_extractor.encodec.quantizer
        student_vq_result = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_features = student_vq_result.quantized  # VQ後
        student_codes = student_vq_result.codes
        vq_loss = student_vq_result.penalty

        return {
            'student_features': student_features,      # VQ後
            'teacher_features': teacher_features,      # VQ後
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'student_encoder_out': student_encoder_out,  # VQ前
            'teacher_encoder_out': teacher_encoder_out,  # VQ前（新增！）
            'vq_loss': vq_loss,
        }

    def compute_ce_logits(self, encoder_out):
        """
        計算 Cross-Entropy logits

        encoder_out: (B, C, T) - Student encoder 輸出
        codebook: (num_codes, C)

        logits = -||z - c||^2 for each code c
               = -z^2 + 2*z*c - c^2
               = 2*z*c - c^2  (忽略 z^2，因為對 softmax 無影響)

        Returns:
            logits: (B, T, num_codes)
        """
        B, C, T = encoder_out.shape
        num_codes = self.codebook.shape[0]

        # encoder_out: (B, C, T) -> (B, T, C)
        z = encoder_out.permute(0, 2, 1)  # (B, T, C)

        # codebook: (num_codes, C)
        # 計算 2*z*c: (B, T, C) @ (C, num_codes) -> (B, T, num_codes)
        logits = 2 * torch.matmul(z, self.codebook.t())

        # 減去 c^2: (num_codes,) broadcast
        c_sq = (self.codebook ** 2).sum(dim=1)  # (num_codes,)
        logits = logits - c_sq.unsqueeze(0).unsqueeze(0)

        return logits  # (B, T, num_codes)


def compute_losses(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature=0.1):
    """
    計算所有 losses

    Returns:
        dict with:
        - total_loss: weighted sum (參與訓練)
        - feature_loss: MSE(z_student, z_teacher) - 使用 VQ前 特徵！
        - ce_loss: CrossEntropy(logits, teacher_tokens)
        - distance_loss: 僅監控
        - vq_loss: 僅監控
        - token_acc: 僅監控

    Note:
        ce_temperature: 用於縮放 logits，避免 softmax 飽和
        - 較小的 temperature (如 0.1) 會使機率分布更尖銳
        - 這有助於在距離很大時仍然保持有意義的梯度
    """
    # 使用 VQ前 特徵（關鍵修改！）
    student_encoder_out = output['student_encoder_out']  # VQ前
    teacher_encoder_out = output['teacher_encoder_out']  # VQ前
    student_codes = output['student_codes']
    teacher_codes = output['teacher_codes']
    vq_loss = output['vq_loss']

    # 1. Feature Loss (使用 VQ前 特徵！)
    # 這樣梯度可以直接流回 Student encoder
    feature_loss = F.mse_loss(student_encoder_out, teacher_encoder_out)

    # 2. Cross-Entropy Loss
    # 計算 logits
    logits = model.compute_ce_logits(student_encoder_out)  # (B, T, num_codes)

    # 獲取 teacher tokens
    if teacher_codes.dim() == 3:
        t_codes = teacher_codes[0]  # (B, T)
    else:
        t_codes = teacher_codes.squeeze(1)

    # CE Loss with temperature scaling
    # Temperature scaling 可以避免 softmax 飽和
    # 當距離很大時，logits 差異會很大，導致 softmax 趨近於 one-hot
    # 除以 temperature 可以緩解這個問題
    B, T, num_codes = logits.shape
    logits_scaled = logits / ce_temperature  # Temperature scaling
    logits_flat = logits_scaled.reshape(B * T, num_codes)  # (B*T, num_codes)
    targets_flat = t_codes.reshape(B * T).long()    # (B*T,)
    ce_loss = F.cross_entropy(logits_flat, targets_flat)

    # 3. Total Loss (參與訓練)
    total_loss = feature_weight * feature_loss + ce_weight * ce_loss

    # 4. 監控指標 (不參與訓練)
    with torch.no_grad():
        if student_codes.dim() == 3:
            s_codes = student_codes[0]
        else:
            s_codes = student_codes.squeeze(1)

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        distances = distance_matrix[s_flat, t_flat]
        distance_loss = distances.mean().item()

        token_acc = (s_codes == t_codes).float().mean().item()

    vq_loss_val = vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss

    return {
        'total_loss': total_loss,
        'feature_loss': feature_loss,
        'ce_loss': ce_loss,
        'distance_loss': distance_loss,
        'vq_loss': vq_loss_val,
        'token_acc': token_acc,
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                distance_matrix, feature_weight, ce_weight, ce_temperature=0.1,
                scaler=None, use_amp=True):
    """訓練一個 epoch"""
    model.train()

    total_loss = 0
    total_feature_loss = 0
    total_ce_loss = 0
    total_distance_loss = 0
    total_vq_loss = 0
    total_token_acc = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        # Forward with AMP
        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            losses = compute_losses(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature)
            loss = losses['total_loss']

        # Backward with AMP
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 累計
        total_loss += losses['total_loss'].item()
        total_feature_loss += losses['feature_loss'].item()
        total_ce_loss += losses['ce_loss'].item()
        total_distance_loss += losses['distance_loss']
        total_vq_loss += losses['vq_loss']
        total_token_acc += losses['token_acc']
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'feat': f"{losses['feature_loss'].item():.4f}",
            'ce': f"{losses['ce_loss'].item():.2f}",
            'acc': f"{losses['token_acc']*100:.1f}%",
        })

    # 確保 lr 是 float
    if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
        if hasattr(lr, 'item'):
            lr = lr.item()
    else:
        # 從 optimizer 取得固定學習率
        lr = optimizer.param_groups[0]['lr']

    return {
        'total_loss': total_loss / n_batches,
        'feature_loss': total_feature_loss / n_batches,
        'ce_loss': total_ce_loss / n_batches,
        'distance_loss': total_distance_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'token_acc': total_token_acc / n_batches,
        'lr': float(lr),
    }


@torch.no_grad()
def validate(model, dataloader, device, distance_matrix, feature_weight, ce_weight,
             ce_temperature=0.1, use_amp=True):
    """驗證"""
    model.eval()

    total_loss = 0
    total_feature_loss = 0
    total_ce_loss = 0
    total_distance_loss = 0
    total_vq_loss = 0
    total_token_acc = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            losses = compute_losses(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature)

        total_loss += losses['total_loss'].item()
        total_feature_loss += losses['feature_loss'].item()
        total_ce_loss += losses['ce_loss'].item()
        total_distance_loss += losses['distance_loss']
        total_vq_loss += losses['vq_loss']
        total_token_acc += losses['token_acc']
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'feature_loss': total_feature_loss / n_batches,
        'ce_loss': total_ce_loss / n_batches,
        'distance_loss': total_distance_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'token_acc': total_token_acc / n_batches,
    }


def plot_training_curves(history, exp_dir, epoch):
    """
    繪製訓練曲線

    8 個子圖：
    1. Total Loss
    2. Feature Loss
    3. CE Loss
    4. Distance Loss (Monitor Only)
    5. VQ Loss (Monitor Only)
    6. Token Accuracy
    7. Learning Rate
    8. Loss Breakdown (stacked)
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Training Curves - Epoch {epoch}', fontsize=14)

    epochs = list(range(1, len(history['train_total_loss']) + 1))

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_total_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Feature Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_feature_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_feature_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CE Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_ce_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_ce_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CE')
    ax.set_title('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distance Loss (Monitor Only)
    ax = axes[0, 3]
    ax.plot(epochs, history['train_distance_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_distance_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Distance Loss (Monitor Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. VQ Loss (Monitor Only)
    ax = axes[1, 0]
    ax.plot(epochs, history['train_vq_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_vq_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VQ Loss')
    ax.set_title('VQ Loss (Monitor Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Token Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, [acc*100 for acc in history['train_token_acc']], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [acc*100 for acc in history['val_token_acc']], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Token Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Learning Rate
    ax = axes[1, 2]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 8. Loss Breakdown
    ax = axes[1, 3]
    ax.plot(epochs, history['train_feature_loss'], 'b-', label='Feature', linewidth=2)
    ax.plot(epochs, history['train_ce_loss'], 'r-', label='CE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Breakdown (Train)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = exp_dir / f'training_curves_epoch_{epoch:03d}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  📊 Saved training curves to {plot_path}")


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=3, split='val'):
    """
    保存音檔樣本
    
    Args:
        model: 模型
        dataloader: 資料載入器
        device: 裝置
        exp_dir: 實驗目錄
        epoch: 當前 epoch
        num_samples: 樣本數量
        split: 'train' 或 'val'，用於分開儲存
    """
    model.eval()
    audio_dir = exp_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)

    torch.cuda.empty_cache()

    for i in range(min(num_samples, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        # 1. 保存 noisy
        noisy_path = audio_dir / f'sample_{i+1}_noisy.wav'
        torchaudio.save(str(noisy_path), noisy_audio.cpu(), sample_rate)

        # 2. 保存 clean
        clean_path = audio_dir / f'sample_{i+1}_clean.wav'
        torchaudio.save(str(clean_path), clean_audio.cpu(), sample_rate)

        try:
            # 3. Student reconstruction
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            student_path = audio_dir / f'sample_{i+1}_student_recon.wav'
            torchaudio.save(str(student_path), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            # 4. Teacher reconstruction
            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            teacher_path = audio_dir / f'sample_{i+1}_teacher_recon.wav'
            torchaudio.save(str(teacher_path), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠️ OOM when saving audio sample {i+1}, skipping reconstruction")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  🔊 Saved {min(num_samples, len(dataloader))} {split} audio samples to {audio_dir}")


def plot_spectrogram(waveform, sample_rate, title, ax):
    """繪製單個頻譜圖"""
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    n_fft = 1024
    hop_length = 256
    ax.specgram(waveform, NFFT=n_fft, Fs=sample_rate, noverlap=n_fft-hop_length,
                cmap='viridis', scale='dB')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')


@torch.no_grad()
def save_spectrogram_comparison(model, val_loader, device, exp_dir, epoch, num_samples=3):
    """保存頻譜圖比較"""
    model.eval()
    spec_dir = exp_dir / 'spectrograms' / f'epoch_{epoch:03d}'
    spec_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    val_iter = iter(val_loader)

    torch.cuda.empty_cache()

    for i in range(min(num_samples, len(val_loader))):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Spectrogram Comparison - Sample {i+1} (Epoch {epoch})', fontsize=14)

        plot_spectrogram(noisy_audio[0], sample_rate, 'Noisy Input', axes[0, 0])
        plot_spectrogram(clean_audio[0], sample_rate, 'Clean Target', axes[0, 1])

        try:
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            plot_spectrogram(student_recon[0], sample_rate, 'Student Recon', axes[1, 0])
            del student_features, student_recon
            torch.cuda.empty_cache()

            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            plot_spectrogram(teacher_recon[0], sample_rate, 'Teacher Recon', axes[1, 1])
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠️ OOM when generating spectrogram {i+1}")
                axes[1, 0].text(0.5, 0.5, 'OOM', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, 'OOM', ha='center', va='center', transform=axes[1, 1].transAxes)
                torch.cuda.empty_cache()
            else:
                raise e

        plt.tight_layout()
        spec_path = spec_dir / f'sample_{i+1}_spectrogram.png'
        plt.savefig(spec_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  📊 Saved {min(num_samples, len(val_loader))} spectrograms to {spec_dir}")


def main():
    parser = argparse.ArgumentParser(description='exp16: Feature + CE Loss Training')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp16_feature_ce',
                       help='Experiment name')

    # Training
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0,
                       help='Weight for Feature MSE Loss')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                       help='Weight for Cross-Entropy Loss')
    parser.add_argument('--ce_temperature', type=float, default=0.1,
                       help='Temperature for CE logits scaling (smaller = sharper distribution)')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Logging
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--plot_interval', type=int, default=10)
    parser.add_argument('--audio_interval', type=int, default=10)
    parser.add_argument('--num_audio_samples', type=int, default=3)

    # Data
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load distance matrix (for monitoring only)
    print(f"\nLoading distance matrix from {DISTANCE_MATRIX}...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)

    model = TeacherStudentWithCE(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
    )

    # Create dataloader
    print("\nLoading data...")

    class SimpleConfig:
        def __init__(self):
            self.use_hdf5 = False
            self.batch_size = args.batch_size
            self.num_workers = args.num_workers
            self.pin_memory = True

    train_loader, val_loader = create_dataloaders(SimpleConfig())
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler (disabled - using constant learning rate)
    # 設為 None 表示不使用 scheduler，保持固定學習率
    scheduler = None
    print(f"Scheduler: DISABLED (constant lr={args.learning_rate})")

    # AMP
    use_amp = True
    scaler = GradScaler() if use_amp else None
    print(f"Using AMP: {use_amp}")

    # Training history
    history = {
        'train_total_loss': [],
        'train_feature_loss': [],
        'train_ce_loss': [],
        'train_distance_loss': [],
        'train_vq_loss': [],
        'train_token_acc': [],
        'val_total_loss': [],
        'val_feature_loss': [],
        'val_ce_loss': [],
        'val_distance_loss': [],
        'val_vq_loss': [],
        'val_token_acc': [],
        'learning_rate': [],
    }

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Loss configuration:")
    print(f"  - Feature Loss weight: {args.feature_weight}")
    print(f"  - CE Loss weight: {args.ce_weight}")
    print(f"  - CE Temperature: {args.ce_temperature}")
    print(f"  - Total = {args.feature_weight}*Feature + {args.ce_weight}*CE")
    print("="*60 + "\n")

    best_val_token_acc = 0

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, distance_matrix,
            args.feature_weight, args.ce_weight, args.ce_temperature,
            scaler=scaler, use_amp=use_amp
        )

        # Validate
        val_results = validate(model, val_loader, device, distance_matrix,
                              args.feature_weight, args.ce_weight, args.ce_temperature,
                              use_amp=use_amp)

        # Record history
        history['train_total_loss'].append(train_results['total_loss'])
        history['train_feature_loss'].append(train_results['feature_loss'])
        history['train_ce_loss'].append(train_results['ce_loss'])
        history['train_distance_loss'].append(train_results['distance_loss'])
        history['train_vq_loss'].append(train_results['vq_loss'])
        history['train_token_acc'].append(train_results['token_acc'])
        history['val_total_loss'].append(val_results['total_loss'])
        history['val_feature_loss'].append(val_results['feature_loss'])
        history['val_ce_loss'].append(val_results['ce_loss'])
        history['val_distance_loss'].append(val_results['distance_loss'])
        history['val_vq_loss'].append(val_results['vq_loss'])
        history['val_token_acc'].append(val_results['token_acc'])
        history['learning_rate'].append(train_results['lr'])

        # Print
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train: Total={train_results['total_loss']:.4f}, "
              f"Feat={train_results['feature_loss']:.4f}, "
              f"CE={train_results['ce_loss']:.2f}, "
              f"Acc={train_results['token_acc']*100:.2f}%")
        print(f"  Val:   Total={val_results['total_loss']:.4f}, "
              f"Feat={val_results['feature_loss']:.4f}, "
              f"CE={val_results['ce_loss']:.2f}, "
              f"Acc={val_results['token_acc']*100:.2f}%")
        print(f"  LR: {train_results['lr']:.2e}")

        # Save best
        if val_results['token_acc'] > best_val_token_acc:
            best_val_token_acc = val_results['token_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_token_acc': val_results['token_acc'],
            }, exp_dir / 'best.pt')
            print(f"  ✓ New best! Token Acc={val_results['token_acc']*100:.2f}%")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_dir / f'epoch_{epoch:03d}.pt')

        # Plot training curves
        if epoch % args.plot_interval == 0 or epoch == args.num_epochs:
            plot_training_curves(history, exp_dir, epoch)

        # Save audio samples and spectrograms (分別儲存 train 和 val)
        if epoch % args.audio_interval == 0 or epoch == args.num_epochs:
            # Train samples
            save_audio_samples(model, train_loader, device, exp_dir, epoch, args.num_audio_samples, split='train')
            # Val samples
            save_audio_samples(model, val_loader, device, exp_dir, epoch, args.num_audio_samples, split='val')
            # Spectrograms (使用 val)
            save_spectrogram_comparison(model, val_loader, device, exp_dir, epoch, args.num_audio_samples)

        # Save history
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Final summary
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best Val Token Accuracy: {best_val_token_acc*100:.2f}%")
    print(f"Results saved to: {exp_dir}")

    print("\n" + "="*60)
    print("Analysis")
    print("="*60)
    print(f"Epoch 1 Token Accuracy: {history['train_token_acc'][0]*100:.2f}%")
    print(f"Final Token Accuracy:   {history['train_token_acc'][-1]*100:.2f}%")
    print(f"Change: {(history['train_token_acc'][-1] - history['train_token_acc'][0])*100:+.2f}%")

    if history['train_token_acc'][-1] > history['train_token_acc'][0]:
        print("\n✅ Token Accuracy improved! Feature + CE Loss works!")
    else:
        print("\n❌ Token Accuracy did not improve")


if __name__ == '__main__':
    main()
