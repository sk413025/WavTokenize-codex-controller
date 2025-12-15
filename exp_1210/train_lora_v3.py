"""
exp_1210: 修復版訓練腳本

修復問題:
1. Teacher 意外進入 train 模式
2. Codebook EMA 漂移
3. 添加 codebook 安全檢查（每個 epoch 檢查一次）

架構:
    Noisy Audio → Encoder(LoRA, rank 可調) → VQ(凍結) → tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.data import create_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1210.models import TeacherStudentExpandedLoRA, CodebookDriftError
from exp_1210.losses import CombinedLossV2, compute_token_accuracy


def set_seed(seed: int = 42):
    """固定隨機種子"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trainable_params(model):
    """獲取可訓練參數"""
    return (p for p in model.parameters() if p.requires_grad)


def verify_model_state(model, stage: str):
    """
    驗證模型狀態是否正確

    Args:
        model: TeacherStudentExpandedLoRA 模型
        stage: 當前階段描述（用於錯誤訊息）
    """
    # 檢查 Teacher 狀態
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")

    if model.teacher.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Teacher quantizer 意外進入 train 模式!")

    # 檢查 Student quantizer 狀態
    if model.student.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Student quantizer 意外進入 train 模式!")

    # 檢查 codebook 是否有梯度
    teacher_cb = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    student_cb = model.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    if teacher_cb.requires_grad:
        raise RuntimeError(f"[{stage}] Teacher codebook requires_grad=True!")

    if student_cb.requires_grad:
        raise RuntimeError(f"[{stage}] Student codebook requires_grad=True!")


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn: CombinedLossV2,
    device: str,
    epoch: int,
    distance_matrix: torch.Tensor,
    scaler=None,
    use_amp: bool = True,
    check_interval: int = 100,
) -> dict:
    """
    訓練一個 epoch

    Args:
        check_interval: 每隔多少 batch 檢查一次模型狀態
    """
    model.train()

    # 驗證模型狀態
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'dw_loss': 0, 'soft_ce_loss': 0,
        'token_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)

                loss, loss_info = loss_fn(
                    student_out=output['student_encoder_out'],
                    teacher_out=output['teacher_encoder_out'],
                    codebook=output['codebook'],
                    teacher_codes=output['teacher_codes'],
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_out=output['student_encoder_out'],
                teacher_out=output['teacher_encoder_out'],
                codebook=output['codebook'],
                teacher_codes=output['teacher_codes'],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), 1.0)
            optimizer.step()

        # 定期檢查模型狀態
        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx + 1}")

        # 累計 metrics
        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['dw_loss'] += loss_info.get('dw_loss', 0)
        metrics['soft_ce_loss'] += loss_info.get('soft_ce_loss', 0)

        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        # Distance loss
        with torch.no_grad():
            s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
            t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{token_acc*100:.1f}%"
        })

    # 平均
    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: CombinedLossV2,
    device: str,
    distance_matrix: torch.Tensor,
    use_amp: bool = True,
) -> dict:
    """驗證模型"""
    model.eval()

    # 驗證模型狀態
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'dw_loss': 0, 'soft_ce_loss': 0,
        'token_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_out=output['student_encoder_out'],
                teacher_out=output['teacher_encoder_out'],
                codebook=output['codebook'],
                teacher_codes=output['teacher_codes'],
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['dw_loss'] += loss_info.get('dw_loss', 0)
        metrics['soft_ce_loss'] += loss_info.get('soft_ce_loss', 0)

        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


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

    每個樣本包含：
    - noisy: 原始噪音音頻
    - clean: 目標乾淨音頻
    - student_recon: Student encoder → Teacher decoder
    - teacher_recon: Teacher encoder → Teacher decoder
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

        # 確保格式
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
            # 3. Student reconstruction: noisy → student encoder → teacher decoder
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            student_path = audio_dir / f'sample_{i+1}_student_recon.wav'
            torchaudio.save(str(student_path), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            # 4. Teacher reconstruction: clean → teacher encoder → teacher decoder
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
                print(f"  OOM when saving audio sample {i+1}, skipping reconstruction")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples to {audio_dir}")


def plot_spectrogram(waveform, sample_rate, title, ax):
    """繪製單個頻譜圖"""
    import numpy as np

    # 確保是 numpy array
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()

    # 確保是 1D
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # 計算 spectrogram
    n_fft = 1024
    hop_length = 256

    ax.specgram(waveform, NFFT=n_fft, Fs=sample_rate, noverlap=n_fft-hop_length,
                cmap='viridis', scale='dB')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')


@torch.no_grad()
def save_spectrogram_comparison(model, val_loader, device, exp_dir, epoch, num_samples=3):
    """
    保存頻譜圖比較

    每個樣本生成一張圖，包含 4 個子圖：
    - noisy: 原始噪音音頻
    - clean: 目標乾淨音頻
    - student_recon: Student encoder → Teacher decoder
    - teacher_recon: Teacher encoder → Teacher decoder
    """
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

        # 創建 2x2 子圖
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Spectrogram Comparison - Sample {i+1} (Epoch {epoch})', fontsize=14)

        # 1. Noisy
        plot_spectrogram(noisy_audio[0], sample_rate, 'Noisy Input', axes[0, 0])

        # 2. Clean
        plot_spectrogram(clean_audio[0], sample_rate, 'Clean Target', axes[0, 1])

        try:
            # 3. Student reconstruction
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            plot_spectrogram(student_recon[0], sample_rate, 'Student Recon (noisy→student→decoder)', axes[1, 0])
            del student_features, student_recon
            torch.cuda.empty_cache()

            # 4. Teacher reconstruction
            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            plot_spectrogram(teacher_recon[0], sample_rate, 'Teacher Recon (clean→teacher→decoder)', axes[1, 1])
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when generating spectrogram {i+1}")
                axes[1, 0].text(0.5, 0.5, 'OOM', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, 'OOM', ha='center', va='center', transform=axes[1, 1].transAxes)
                torch.cuda.empty_cache()
            else:
                raise e

        plt.tight_layout()
        spec_path = spec_dir / f'sample_{i+1}_spectrogram.png'
        plt.savefig(spec_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {min(num_samples, len(val_loader))} spectrograms to {spec_dir}")


def save_plots(history: dict, exp_dir: Path):
    """保存訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_total_loss'], label='Train')
    axes[0, 0].plot(history['val_total_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    # Feature Loss
    axes[0, 1].plot(history['train_feature_loss'], label='Train')
    axes[0, 1].plot(history['val_feature_loss'], label='Val')
    axes[0, 1].set_title('Feature Loss')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')

    # Triplet Loss
    axes[0, 2].plot(history['train_triplet_loss'], label='Train')
    axes[0, 2].plot(history['val_triplet_loss'], label='Val')
    axes[0, 2].set_title('Triplet Loss')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Epoch')

    # Token Accuracy
    axes[1, 0].plot([x * 100 for x in history['train_token_acc']], label='Train')
    axes[1, 0].plot([x * 100 for x in history['val_token_acc']], label='Val')
    axes[1, 0].set_title('Token Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Epoch')

    # VQ Distance
    axes[1, 1].plot(history['train_distance_loss'], label='Train')
    axes[1, 1].plot(history['val_distance_loss'], label='Val')
    axes[1, 1].set_title('VQ Distance')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch')

    # DW Loss (if used)
    if any(x > 0 for x in history.get('train_dw_loss', [0])):
        axes[1, 2].plot(history['train_dw_loss'], label='Train')
        axes[1, 2].plot(history['val_dw_loss'], label='Val')
        axes[1, 2].set_title('DW Loss')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Epoch')
    else:
        axes[1, 2].text(0.5, 0.5, 'DW Loss not used', ha='center', va='center')
        axes[1, 2].set_title('DW Loss')

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Exp1210: Fixed LoRA Training')
    parser.add_argument('--exp_name', type=str, required=True, help='實驗名稱')
    parser.add_argument('--lora_rank', type=int, default=128, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=256, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--feature_weight', type=float, default=1.0, help='Feature Loss 權重')
    parser.add_argument('--triplet_weight', type=float, default=0.5, help='Triplet Loss 權重')
    parser.add_argument('--triplet_margin', type=float, default=0.2, help='Triplet margin')
    parser.add_argument('--dw_weight', type=float, default=0.0, help='Distance-Weighted Loss 權重')
    parser.add_argument('--dw_temperature', type=float, default=1.0, help='Distance-Weighted temperature')
    parser.add_argument('--soft_ce_weight', type=float, default=0.0, help='Soft CE Loss 權重')
    parser.add_argument('--soft_ce_temperature', type=float, default=2.0, help='Soft CE temperature')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='訓練 epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')
    parser.add_argument('--check_interval', type=int, default=100, help='模型狀態檢查間隔')
    args = parser.parse_args()

    # 設置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using device: {device}")

    # 實驗目錄
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp1210 (Fixed): {args.exp_name}")
    print(f"{'='*60}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, 18 layers")
    print(f"Loss weights: feature={args.feature_weight}, triplet={args.triplet_weight}")
    print(f"             dw={args.dw_weight}, soft_ce={args.soft_ce_weight}")
    print(f"Optimizer: lr={args.lr}, weight_decay={args.weight_decay}")
    print(f"Codebook safety check interval: {args.check_interval} batches")
    print(f"{'='*60}\n")

    # 載入 distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)

    # 創建模型
    print("Creating model...")
    model = TeacherStudentExpandedLoRA(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    # 初始驗證
    print("\n[初始驗證] 檢查模型狀態...")
    verify_model_state(model, "初始化後")
    model.check_codebook_integrity(raise_error=True)
    print("[初始驗證] 通過!\n")

    # 創建 loss function
    loss_fn = CombinedLossV2(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        dw_weight=args.dw_weight,
        dw_temperature=args.dw_temperature,
        soft_ce_weight=args.soft_ce_weight,
        soft_ce_temperature=args.soft_ce_temperature,
    )

    # 只優化可訓練參數 (LoRA)
    trainable_params = list(get_trainable_params(model))
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # DataLoader
    class DataConfig:
        def __init__(self, batch_size, num_workers):
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = True

    train_loader, val_loader = create_dataloaders(DataConfig(args.batch_size, args.num_workers))
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 訓練歷史
    history = {
        'train_total_loss': [], 'train_feature_loss': [], 'train_triplet_loss': [],
        'train_dw_loss': [], 'train_soft_ce_loss': [],
        'train_token_acc': [], 'train_distance_loss': [],
        'val_total_loss': [], 'val_feature_loss': [], 'val_triplet_loss': [],
        'val_dw_loss': [], 'val_soft_ce_loss': [],
        'val_token_acc': [], 'val_distance_loss': [],
    }

    best_val_acc = 0

    # 訓練迴圈
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, scaler, args.use_amp, args.check_interval
        )

        # 驗證
        val_metrics = validate(model, val_loader, loss_fn, device, distance_matrix, args.use_amp)

        # Epoch 結束時檢查 codebook 完整性
        print(f"\n[Epoch {epoch}] Codebook 完整性檢查...")
        integrity = model.check_codebook_integrity(raise_error=True)
        print(f"  Teacher drift: {integrity['teacher_drift']:.8f}")
        print(f"  Student drift: {integrity['student_drift']:.8f}")
        print(f"  狀態: OK")

        # 記錄
        for key in train_metrics:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])

        # 打印
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Feature={train_metrics['feature_loss']:.4f}, "
              f"Triplet={train_metrics['triplet_loss']:.4f}, "
              f"Acc={train_metrics['token_acc']*100:.2f}%, "
              f"Dist={train_metrics['distance_loss']:.4f}")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Feature={val_metrics['feature_loss']:.4f}, "
              f"Triplet={val_metrics['triplet_loss']:.4f}, "
              f"Acc={val_metrics['token_acc']*100:.2f}%, "
              f"Dist={val_metrics['distance_loss']:.4f}")

        # 保存最佳模型
        if val_metrics['token_acc'] > best_val_acc:
            best_val_acc = val_metrics['token_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, exp_dir / 'best_model.pt')
            print(f"  -> New best! Val Acc: {best_val_acc*100:.2f}%")

        # 保存歷史
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # 保存圖表
        save_plots(history, exp_dir)

        # 每 5 個 epoch 保存音檔和頻譜圖
        if epoch % 5 == 0 or epoch == 1:
            print(f"\n[Epoch {epoch}] 保存音檔和頻譜圖...")
            save_audio_samples(model, train_loader, device, exp_dir, epoch, num_samples=2, split='train')
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=2, split='val')
            save_spectrogram_comparison(model, val_loader, device, exp_dir, epoch, num_samples=2)

    # 完成
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
