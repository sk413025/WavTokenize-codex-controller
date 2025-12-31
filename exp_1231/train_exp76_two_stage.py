"""
Exp76: Two-Stage Training

核心改進：
Stage 1: 先訓練 waveform-level 去噪 (不經過 VQ)
Stage 2: 用去噪後的音訊去匹配 VQ tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use('Agg')
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1231.losses import CombinedLossExp71
from exp_1231.utils import plot_metrics, save_audio_samples


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_masked_accuracy(student_codes, teacher_codes, lengths, encoder_stride=320):
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
    return total_correct / total_tokens if total_tokens > 0 else 0, total_correct, total_tokens


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss for waveform quality"""

    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[50, 120, 240],
                 win_lengths=[240, 600, 1200]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, pred, target):
        loss = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT
            pred_stft = torch.stft(pred.squeeze(1), fft_size, hop_size, win_length,
                                   return_complex=True, window=torch.hann_window(win_length).to(pred.device))
            target_stft = torch.stft(target.squeeze(1), fft_size, hop_size, win_length,
                                     return_complex=True, window=torch.hann_window(win_length).to(target.device))

            # Spectral convergence loss
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

            # Magnitude loss
            mag_loss = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

            loss += sc_loss + mag_loss

        return loss / len(self.fft_sizes)


def train_stage1(model, dataloader, optimizer, device, epoch,
                 scaler=None, use_amp=True, grad_clip=1.0):
    """Stage 1: Waveform-level denoising"""
    model.train()
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Stage1 Epoch {epoch}", leave=False)
    for batch in pbar:
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # 通過 student encoder 和 decoder
            student_features, _, _ = model.student.feature_extractor(noisy_audio.squeeze(1), bandwidth_id=0)
            denoised = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))

            if denoised.dim() == 3:
                denoised = denoised.squeeze(1)

            # 長度對齊
            min_len = min(denoised.shape[-1], clean_audio.shape[-1])
            denoised = denoised[..., :min_len]
            clean_target = clean_audio.squeeze(1)[..., :min_len]

            # L1 + STFT Loss
            l1_loss = F.l1_loss(denoised, clean_target)
            stft_loss = stft_loss_fn(denoised.unsqueeze(1), clean_target.unsqueeze(1))
            loss = l1_loss + stft_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / n_batches


def train_stage2(model, dataloader, optimizer, loss_fn, device, epoch,
                 encoder_stride=320, scaler=None, use_amp=True,
                 grad_clip=1.0, gradient_accumulation_steps=1):
    """Stage 2: Token matching"""
    model.train()
    metrics = {'total_loss': 0, 'masked_acc': 0}
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Stage2 Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                encoder_stride=encoder_stride,
            )
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        metrics['total_loss'] += loss.item() * gradient_accumulation_steps
        metrics['masked_acc'] += acc
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item()*gradient_accumulation_steps:.4f}", 'acc': f"{acc*100:.2f}%"})

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, encoder_stride=320, use_amp=True):
    model.eval()
    metrics = {'total_loss': 0, 'masked_acc': 0}
    n_batches = 0

    for batch in dataloader:
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            loss, _ = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                encoder_stride=encoder_stride,
            )

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        metrics['total_loss'] += loss.item()
        metrics['masked_acc'] += acc
        n_batches += 1

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Exp76: Two-Stage Training')
    parser.add_argument('--exp_name', type=str, default='exp76_two_stage')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Stage 1
    parser.add_argument('--stage1_epochs', type=int, default=100)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)

    # Stage 2
    parser.add_argument('--stage2_epochs', type=int, default=200)
    parser.add_argument('--stage2_lr', type=float, default=5e-5)
    parser.add_argument('--freeze_stage1', action='store_true', default=False)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Exp76: Two-Stage Training")
    print("=" * 60)
    print(f"Stage 1: Waveform Denoising ({args.stage1_epochs} epochs, lr={args.stage1_lr})")
    print(f"Stage 2: Token Matching ({args.stage2_epochs} epochs, lr={args.stage2_lr})")
    print("=" * 60)

    # Data
    from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
    from torch.utils.data import DataLoader
    train_dataset = AlignedNoisyCleanPairDataset(TRAIN_CACHE)
    val_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn)

    # Model
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG), wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_layers=args.lora_layers, device=device,
    )

    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    history = {'stage1_loss': [], 'stage2_loss': [], 'stage2_acc': [], 'val_loss': [], 'val_acc': []}

    scaler = GradScaler(enabled=args.use_amp)

    # ==================== Stage 1: Waveform Denoising ====================
    print("\n" + "=" * 60)
    print("Stage 1: Waveform Denoising")
    print("=" * 60)

    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr, weight_decay=0.05
    )

    for epoch in range(1, args.stage1_epochs + 1):
        loss = train_stage1(model, train_loader, optimizer1, device, epoch, scaler, args.use_amp, args.grad_clip)
        history['stage1_loss'].append(loss)
        print(f"Stage1 Epoch {epoch}/{args.stage1_epochs}: Loss={loss:.4f}")

    # 保存 Stage 1 模型
    torch.save({
        'epoch': args.stage1_epochs,
        'model_state_dict': model.state_dict(),
        'stage': 1,
        'config': config
    }, exp_dir / 'stage1_model.pt')
    print("Stage 1 completed and saved!")

    # ==================== Stage 2: Token Matching ====================
    print("\n" + "=" * 60)
    print("Stage 2: Token Matching")
    print("=" * 60)

    # Stage 2 Loss
    loss_fn = CombinedLossExp71(
        feature_weight=1.0,
        triplet_weight=1.0,
        soft_token_weight=1.0,
        triplet_margin=0.2,
        soft_token_temperature=1.0,
        vq_commitment_weight=0.1,
        vq_distortion_weight=0.1,
    )

    optimizer2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage2_lr, weight_decay=0.05
    )

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(1, args.stage2_epochs + 1):
        train_metrics = train_stage2(model, train_loader, optimizer2, loss_fn, device, epoch,
                                     args.encoder_stride, scaler, args.use_amp,
                                     args.grad_clip, args.gradient_accumulation_steps)
        val_metrics = validate(model, val_loader, loss_fn, device, args.encoder_stride, args.use_amp)

        history['stage2_loss'].append(train_metrics['total_loss'])
        history['stage2_acc'].append(train_metrics['masked_acc'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['val_acc'].append(val_metrics['masked_acc'])

        print(f"Stage2 Epoch {epoch}/{args.stage2_epochs}: "
              f"Train Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}% | "
              f"Val Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save({
                'epoch': args.stage1_epochs + epoch,
                'model_state_dict': model.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'stage': 2,
                'config': config
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best: {best_val_acc*100:.2f}%")
            # 保存最佳模型的音檔樣本
            save_audio_samples(model, train_loader, val_loader, device, exp_dir, args.stage1_epochs + epoch)

        # 繪製訓練曲線
        plot_metrics(history, exp_dir, exp_type='two_stage')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    torch.save({
        'epoch': args.stage1_epochs + args.stage2_epochs,
        'model_state_dict': model.state_dict(),
        'val_masked_acc': val_metrics['masked_acc'],
        'stage': 2,
        'config': config
    }, exp_dir / 'last_model.pt')

    print("\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Val Acc: {best_val_acc*100:.2f}% @ Stage2 Epoch {best_epoch}")
    print("=" * 60)


if __name__ == '__main__':
    main()
