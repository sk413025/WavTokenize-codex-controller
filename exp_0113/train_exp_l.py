"""
exp_0113/train_exp_l.py: Exp L - 多位置 Adapter 去噪訓練

核心設計:
- 在多個噪音敏感層 (L2, L4, L6, L8) 插入 Adapter
- 使用較大的 Bottleneck (input_dim // 2) 增加容量
- 目標: ~70K 可訓練參數，比 Exp J (8K) 增加約 8 倍

配置:
- Adapter 位置: L2, L4, L6, L8
- Bottleneck: input_dim // 2 (reduction_factor=2)
- 初始 scale: 0.01 (小值，漸進生效)

基於 Exp J 經驗:
- Exp J 聽感好但數值差 → 可能是容量不足
- 多位置 + 大容量 → 期望改善數值同時保持聽感

執行:
    python exp_0113/train_exp_l.py --exp_name exp_l_multi_adapter
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX, TRAIN_CACHE, VAL_CACHE
from exp_0113.models import TeacherStudentMultiAdapter
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


def set_seed(seed: int = 42):
    """
    設定隨機種子以確保實驗可重現性

    Args:
        seed: 隨機種子值
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trainable_params(model):
    """
    獲取模型中所有可訓練的參數

    Args:
        model: 模型實例

    Returns:
        可訓練參數的 generator
    """
    return (p for p in model.parameters() if p.requires_grad)


def verify_model_state(model, stage: str):
    """
    驗證模型狀態，確保 Teacher 和 Quantizer 保持凍結

    Args:
        model: TeacherStudentMultiAdapter 模型
        stage: 當前階段描述字串

    Raises:
        RuntimeError: 如果模型狀態異常
    """
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")
    if model.teacher.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Teacher quantizer 意外進入 train 模式!")
    if model.student.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Student quantizer 意外進入 train 模式!")


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                distance_matrix, encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    """
    執行一個 epoch 的訓練

    Args:
        model: TeacherStudentMultiAdapter 模型
        dataloader: 訓練資料載入器
        optimizer: 優化器
        loss_fn: 損失函數
        device: 計算設備
        epoch: 當前 epoch 數
        distance_matrix: Codebook 距離矩陣
        encoder_stride: Encoder 步長
        scaler: AMP GradScaler
        use_amp: 是否使用混合精度
        check_interval: 狀態檢查間隔
        grad_clip: 梯度裁剪閾值
        gradient_accumulation_steps: 梯度累積步數

    Returns:
        metrics: 訓練指標字典
    """
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
        'avg_snr': 0,
    }

    # Adapter scales for each position
    adapter_scale_metrics = {}

    n_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx + 1}")

        # 更新 metrics
        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        # 記錄各 Adapter 的 scale
        for key, val in output.get('adapter_scales', {}).items():
            if key not in adapter_scale_metrics:
                adapter_scale_metrics[key] = 0
            adapter_scale_metrics[key] += val

        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        with torch.no_grad():
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1

        # 顯示 progress bar
        scales_str = ', '.join([f"{k}:{v:.3f}" for k, v in output.get('adapter_scales', {}).items()])
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
            'scales': scales_str[:30]  # 截斷避免太長
        })

    # 平均化 metrics
    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    for key in adapter_scale_metrics:
        adapter_scale_metrics[key] /= n_batches

    metrics['adapter_scales'] = adapter_scale_metrics

    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, distance_matrix,
             encoder_stride=320, use_amp=True):
    """
    執行驗證

    Args:
        model: TeacherStudentMultiAdapter 模型
        dataloader: 驗證資料載入器
        loss_fn: 損失函數
        device: 計算設備
        distance_matrix: Codebook 距離矩陣
        encoder_stride: Encoder 步長
        use_amp: 是否使用混合精度

    Returns:
        metrics: 驗證指標字典
    """
    model.eval()
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    adapter_scale_metrics = {}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
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
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        for key, val in output.get('adapter_scales', {}).items():
            if key not in adapter_scale_metrics:
                adapter_scale_metrics[key] = 0
            adapter_scale_metrics[key] += val

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    for key in adapter_scale_metrics:
        adapter_scale_metrics[key] /= n_batches

    metrics['adapter_scales'] = adapter_scale_metrics

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
    """
    保存音頻樣本用於聽感評估

    Args:
        model: 模型實例
        dataloader: 資料載入器
        device: 計算設備
        exp_dir: 實驗目錄
        epoch: 當前 epoch
        num_samples: 保存樣本數量
        split: 資料集類型 ('train' 或 'val')
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

        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        try:
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_student_recon.wav'), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_metrics(history, exp_dir, adapter_positions):
    """
    繪製訓練曲線

    Args:
        history: 訓練歷史字典
        exp_dir: 實驗目錄
        adapter_positions: Adapter 位置列表
    """
    n_adapter_plots = min(len(adapter_positions), 4)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Total Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Masked Accuracy
    ax = axes[0, 1]
    ax.plot([x * 100 for x in history['train_masked_acc']], label='Train')
    ax.plot([x * 100 for x in history['val_masked_acc']], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 2]
    ax.plot(history['train_feature_loss'], label='Train')
    ax.plot(history['val_feature_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[0, 3]
    ax.plot(history['train_triplet_loss'], label='Train')
    ax.plot(history['val_triplet_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Train-Val Gap
    ax = axes[1, 0]
    gap = [t - v for t, v in zip(history['train_masked_acc'], history['val_masked_acc'])]
    ax.plot([g * 100 for g in gap], color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (%)')
    ax.set_title('Train-Val Accuracy Gap')
    ax.grid(True)

    # Distance
    ax = axes[1, 1]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    # Adapter Scales (all positions)
    ax = axes[1, 2]
    if 'adapter_scales' in history and len(history['adapter_scales']) > 0:
        # 取得所有 position 的 scale history
        all_keys = set()
        for scales in history['adapter_scales']:
            all_keys.update(scales.keys())

        for key in sorted(all_keys):
            values = [scales.get(key, 0) for scales in history['adapter_scales']]
            ax.plot(values, label=key)
        ax.legend(fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Scale')
    ax.set_title('Adapter Scale Parameters')
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 3]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate')
    ax.grid(True)

    plt.suptitle('Exp L: Multi-Position Adapter Denoising', fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    """
    主訓練函數

    執行 Exp L: 多位置 Adapter 去噪訓練
    """
    parser = argparse.ArgumentParser(description='Exp 0113: 多位置 Adapter 去噪 (Exp L)')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_l_multi_adapter')
    parser.add_argument('--output_dir', type=str, default=None)

    # Multi-Adapter Config
    parser.add_argument('--adapter_positions', type=str, default='1,4,7,10',
                        help='Adapter 插入位置 (ResBlock 後: L1, L4, L7, L10)')
    parser.add_argument('--reduction_factor', type=int, default=2,
                        help='Bottleneck 降維倍數 (2 = input_dim // 2)')
    parser.add_argument('--adapter_dropout', type=float, default=0.1)
    parser.add_argument('--adapter_init_scale', type=float, default=0.01,
                        help='Adapter 初始 scale (小值使初期影響小)')

    # Learning Rate
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for Adapters')

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # Curriculum Learning
    parser.add_argument('--curriculum_mode', type=str, default='curriculum',
                        choices=['curriculum', 'anti_curriculum'])
    parser.add_argument('--initial_phase', type=float, default=0.3)
    parser.add_argument('--phase_increment', type=float, default=0.1)
    parser.add_argument('--phase_advance_epochs', type=int, default=30)

    # Training
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--check_interval', type=int, default=100)

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Gradient Accumulation
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 解析 Adapter 位置
    adapter_positions = [int(x) for x in args.adapter_positions.split(',')]

    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['experiment_type'] = 'Exp L: Multi-Position Adapter Denoising'
    config['adapter_positions_list'] = adapter_positions

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Exp 0113: {config['experiment_type']}")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"\nMulti-Adapter Configuration:")
    print(f"  Positions: {adapter_positions}")
    print(f"  Reduction Factor: {args.reduction_factor} (bottleneck = input_dim // {args.reduction_factor})")
    print(f"  Init scale: {args.adapter_init_scale}")
    print(f"  LR: {args.lr}")
    print("=" * 60)

    # Load data
    print("\n載入資料 (with curriculum)...")
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        curriculum_mode=args.curriculum_mode,
        initial_phase=args.initial_phase,
        phase_increment=args.phase_increment,
        compute_snr=False,
    )
    print(f"  Initial curriculum: {len(curriculum_sampler)} samples ({args.initial_phase:.0%})")
    print(f"  Val batches: {len(val_loader)}")

    # Load distance matrix
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # Create model
    print("\n創建模型...")
    model = TeacherStudentMultiAdapter(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        adapter_positions=adapter_positions,
        reduction_factor=args.reduction_factor,
        dropout=args.adapter_dropout,
        init_scale=args.adapter_init_scale,
        device=device,
    )

    # 只優化 Adapter 參數
    adapter_params = model.get_adapter_params()
    optimizer = torch.optim.AdamW(
        adapter_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_adapter_params = sum(p.numel() for p in adapter_params)
    print(f"\nOptimizer: AdamW")
    print(f"  Adapter params: {total_adapter_params:,}")
    print(f"  LR: {args.lr}")

    # Loss Function
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        encoder_stride=args.encoder_stride,
    )

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'adapter_scales': [],
        'curriculum_phase': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    print("\n開始訓練...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} (Curriculum: {curriculum_sampler.current_phase:.0%})")
        print(f"{'='*60}")

        if epoch > 1 and (epoch - 1) % args.phase_advance_epochs == 0:
            curriculum_sampler.advance_phase()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, scaler, args.use_amp,
            args.check_interval, args.grad_clip, args.gradient_accumulation_steps
        )

        # Record LR
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

        val_metrics = validate(
            model, val_loader, loss_fn, device,
            distance_matrix, args.encoder_stride, args.use_amp
        )

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_masked_acc'].append(train_metrics['masked_acc'])
        history['val_masked_acc'].append(val_metrics['masked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_dist'].append(train_metrics['distance_loss'])
        history['val_dist'].append(val_metrics['distance_loss'])
        history['train_avg_snr'].append(train_metrics.get('avg_snr', 0))
        history['adapter_scales'].append(train_metrics.get('adapter_scales', {}))
        history['curriculum_phase'].append(curriculum_sampler.current_phase)

        train_val_gap = train_metrics['masked_acc'] - val_metrics['masked_acc']

        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Masked Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Masked Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"Gap:   {train_val_gap*100:.2f}%")

        # Print adapter scales
        print("Adapter Scales:")
        for key, val in train_metrics.get('adapter_scales', {}).items():
            print(f"  {key}: {val:.4f}")

        # Check codebook integrity
        try:
            cb_check = model.check_codebook_integrity(raise_error=True)
        except Exception as e:
            print(f"ERROR: {e}")
            break

        # Save best model
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch

            # 保存所有 Adapter 的狀態
            adapter_states = {
                f"adapter_{k}": v.state_dict()
                for k, v in model.get_adapters().items()
            }
            torch.save({
                'epoch': epoch,
                'adapter_states': adapter_states,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best! Val Acc: {best_val_acc*100:.2f}%")

        # Save audio samples periodically
        if epoch % 50 == 0 or epoch == 1:
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=2, split='val')
            save_audio_samples(model, train_loader, device, exp_dir, epoch, num_samples=2, split='train')

        # Save history and plot
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == 1:
            plot_metrics(history, exp_dir, adapter_positions)

    # Final summary
    print("\n" + "=" * 60)
    print("訓練完成!")
    print("=" * 60)
    print(f"Best Val Acc: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")
    print(f"Total Adapter params: {total_adapter_params:,}")

    # Final adapter scales
    print("\nFinal Adapter Scales:")
    for key, val in model.get_adapter_scales().items():
        print(f"  {key}: {val:.4f}")

    print(f"\n結果保存於: {exp_dir}")

    # Final plots
    plot_metrics(history, exp_dir, adapter_positions)

    # Save final model
    adapter_states = {
        f"adapter_{k}": v.state_dict()
        for k, v in model.get_adapters().items()
    }
    torch.save({
        'epoch': args.num_epochs,
        'adapter_states': adapter_states,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['masked_acc'],
        'config': config,
    }, exp_dir / 'final_model.pt')


if __name__ == '__main__':
    main()
