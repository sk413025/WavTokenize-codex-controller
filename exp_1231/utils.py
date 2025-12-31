"""
Exp1231 共用工具函數

包含:
- plot_metrics: 繪製訓練曲線
- save_audio_samples: 保存 train/val 音檔樣本
"""

import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_metrics(history: Dict[str, List], exp_dir: Path, exp_type: str = 'general'):
    """
    繪製訓練曲線

    Args:
        history: 訓練歷史字典
        exp_dir: 實驗目錄
        exp_type: 實驗類型 ('soft_token', 'contrastive', 'adapter', 'progressive', 'two_stage', 'general')
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # === 第一行 ===
    # Loss
    if 'train_loss' in history and history['train_loss']:
        axes[0, 0].plot(history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Val', color='orange')
    # Two-stage 特殊處理
    if 'stage1_loss' in history and history['stage1_loss']:
        axes[0, 0].plot(history['stage1_loss'], label='Stage1', color='green')
    if 'stage2_loss' in history and history['stage2_loss']:
        x_offset = len(history.get('stage1_loss', []))
        x = list(range(x_offset, x_offset + len(history['stage2_loss'])))
        axes[0, 0].plot(x, history['stage2_loss'], label='Stage2', color='blue')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Token Accuracy
    if 'train_acc' in history and history['train_acc']:
        axes[0, 1].plot([a * 100 for a in history['train_acc']], label='Train', color='blue')
    if 'val_acc' in history and history['val_acc']:
        axes[0, 1].plot([a * 100 for a in history['val_acc']], label='Val', color='orange')
    if 'stage2_acc' in history and history['stage2_acc']:
        x_offset = len(history.get('stage1_loss', []))
        x = list(range(x_offset, x_offset + len(history['stage2_acc'])))
        axes[0, 1].plot(x, [a * 100 for a in history['stage2_acc']], label='Stage2', color='blue')
    axes[0, 1].set_title('Token Accuracy (%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 第三個圖 - 根據實驗類型
    if exp_type == 'soft_token' and 'train_soft_acc' in history:
        axes[0, 2].plot([a * 100 for a in history['train_soft_acc']], label='Train', color='blue')
        axes[0, 2].plot([a * 100 for a in history['val_soft_acc']], label='Val', color='orange')
        axes[0, 2].set_title('Soft Token Accuracy (%)')
    elif exp_type == 'contrastive' and 'train_contrastive_acc' in history:
        axes[0, 2].plot([a * 100 for a in history['train_contrastive_acc']], label='Train', color='blue')
        axes[0, 2].plot([a * 100 for a in history['val_contrastive_acc']], label='Val', color='orange')
        axes[0, 2].set_title('Contrastive Accuracy (%)')
    elif exp_type == 'progressive' and 'feature_weight' in history:
        axes[0, 2].plot(history['feature_weight'], label='Feature', color='blue')
        axes[0, 2].plot(history['soft_token_weight'], label='Soft Token', color='orange')
        axes[0, 2].set_title('Loss Weights Schedule')
    else:
        # 顯示 LR
        if 'lr' in history and history['lr']:
            axes[0, 2].plot(history['lr'], color='green')
            axes[0, 2].set_title('Learning Rate')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # === 第二行 ===
    # Feature Loss
    if 'train_feature_loss' in history and history['train_feature_loss']:
        axes[1, 0].plot(history['train_feature_loss'], label='Train', color='blue')
        axes[1, 0].plot(history['val_feature_loss'], label='Val', color='orange')
        axes[1, 0].set_title('Feature Loss')
    else:
        axes[1, 0].set_visible(False)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 第五個圖 - 根據實驗類型
    if exp_type == 'soft_token' and 'train_soft_token_loss' in history:
        axes[1, 1].plot(history['train_soft_token_loss'], label='Train', color='blue')
        axes[1, 1].plot(history['val_soft_token_loss'], label='Val', color='orange')
        axes[1, 1].set_title('Soft Token Loss (KLD)')
    elif exp_type == 'contrastive' and 'train_contrastive_loss' in history:
        axes[1, 1].plot(history.get('train_contrastive_loss', []), label='Train', color='blue')
        axes[1, 1].plot(history.get('val_contrastive_loss', []), label='Val', color='orange')
        axes[1, 1].set_title('Contrastive Loss (InfoNCE)')
    elif 'train_triplet_loss' in history and history['train_triplet_loss']:
        axes[1, 1].plot(history['train_triplet_loss'], label='Train', color='blue')
        axes[1, 1].plot(history['val_triplet_loss'], label='Val', color='orange')
        axes[1, 1].set_title('Triplet Loss')
    else:
        axes[1, 1].set_visible(False)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Learning Rate (如果還沒顯示)
    if 'lr' in history and history['lr'] and exp_type in ['soft_token', 'contrastive']:
        axes[1, 2].plot(history['lr'], color='green')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


@torch.no_grad()
def save_audio_samples(
    model,
    train_loader,
    val_loader,
    device: str,
    exp_dir: Path,
    epoch: int,
    num_samples: int = 3,
    sample_rate: int = 24000,
):
    """
    保存 train 和 val 的音檔樣本，包含 Teacher 和 Student 的重建

    保存結構:
    exp_dir/
      audio_samples/
        epoch_X/
          train/
            sample_0_noisy.wav           # 原始帶噪音輸入
            sample_0_clean.wav           # 原始乾淨目標
            sample_0_student.wav         # Student 去噪重建
            sample_0_teacher.wav         # Teacher 重建 (VQ 上界)
          val/
            ...

    Args:
        model: 訓練模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        device: 設備
        exp_dir: 實驗目錄
        epoch: 當前 epoch
        num_samples: 每個 split 保存的樣本數
        sample_rate: 音訊採樣率
    """
    model.eval()

    audio_dir = exp_dir / 'audio_samples' / f'epoch_{epoch}'

    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        split_dir = audio_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for batch in loader:
            if saved >= num_samples:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # 取得 model output
            output = model(noisy_audio, clean_audio)

            # 取得 features
            student_features = output['student_encoder_out']
            teacher_features = output['teacher_encoder_out']

            # 檢查模型結構並重建
            if hasattr(model, 'teacher'):
                # TeacherStudentConfigurableLoRA 結構
                bandwidth_id = torch.tensor([0]).to(device)
                student_recon = model.teacher.decode(student_features, bandwidth_id=bandwidth_id)
                teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=bandwidth_id)
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'teacher'):
                # TeacherStudentWithAdapter 結構
                bandwidth_id = torch.tensor([0]).to(device)
                student_recon = model.base_model.teacher.decode(student_features, bandwidth_id=bandwidth_id)
                teacher_recon = model.base_model.teacher.decode(teacher_features, bandwidth_id=bandwidth_id)
            else:
                print(f"Warning: Cannot reconstruct audio for {split_name}")
                continue

            # 處理維度
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            if noisy_audio.dim() == 3:
                noisy_audio = noisy_audio.squeeze(1)
            if clean_audio.dim() == 3:
                clean_audio = clean_audio.squeeze(1)

            # 對齊長度
            min_len = min(
                student_recon.shape[-1],
                teacher_recon.shape[-1],
                noisy_audio.shape[-1],
                clean_audio.shape[-1]
            )

            # 保存每個樣本
            batch_size = min(noisy_audio.shape[0], num_samples - saved)
            for i in range(batch_size):
                sample_idx = saved + i

                # 截取
                noisy_wav = noisy_audio[i, :min_len].cpu()
                clean_wav = clean_audio[i, :min_len].cpu()
                student_wav = student_recon[i, :min_len].cpu()
                teacher_wav = teacher_recon[i, :min_len].cpu()

                # 正規化並保存
                for wav, name in [
                    (noisy_wav, 'noisy'),
                    (clean_wav, 'clean'),
                    (student_wav, 'student'),
                    (teacher_wav, 'teacher')
                ]:
                    wav = wav / (wav.abs().max() + 1e-8)
                    wav = wav.unsqueeze(0)  # [1, T]
                    torchaudio.save(
                        split_dir / f'sample_{sample_idx}_{name}.wav',
                        wav,
                        sample_rate
                    )

            saved += batch_size

    print(f"Audio samples saved to {audio_dir}")


def compute_masked_accuracy(student_codes, teacher_codes, lengths, encoder_stride=320):
    """計算有效區域的 token 準確率"""
    batch_size = student_codes.shape[0]
    total_correct = 0
    total_tokens = 0

    for i in range(batch_size):
        if lengths is not None and i < len(lengths):
            valid_frames = lengths[i].item() // encoder_stride
            valid_frames = min(valid_frames, student_codes.shape[1])
        else:
            valid_frames = student_codes.shape[1]

        if valid_frames > 0:
            s = student_codes[i, :valid_frames]
            t = teacher_codes[i, :valid_frames]
            total_correct += (s == t).sum().item()
            total_tokens += valid_frames

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return accuracy, total_correct, total_tokens
