"""
exp_0112_intermediate: Exp K v6 - TracIn 支援版本

V6 改進重點 (支援 TracIn 診斷分析):
1. 定期保存 checkpoint (每 N epochs)
2. 調整 L4 權重: 1.0 → 0.5 (解決 L4 過擬合問題)
3. 保持 300 epochs 完整訓練
4. 保存 LoRA gradients 用於 TracIn

encoder.model 結構:
  model[3]: SConv1d (Downsample 1) - 監督目標
  model[4]: SEANetResnetBlock (ResBlock 2) - 監督目標 (降低權重)
  model[5]: ELU (激活函數，監督無效!)
  model[6]: SConv1d (Downsample 2) - 監督目標

配置:
- L3/model[3] (0.3): Downsample, Cosine Loss
- L4/model[4] (0.5): ResBlock, Cosine Loss (降低自 1.0)
- L6/model[6] (0.5): Downsample, Cosine Loss
- intermediate_weight: 0.5 → 動態衰減至 0.25
- checkpoint 保存: 每 10 epochs

執行:
    bash exp_0112_intermediate/run_exp_k_v6.sh
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
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


class IntermediateSupervisionLossV6(nn.Module):
    """
    V6: 動態權重中間層監督 Loss (調整 L4 權重)

    改進:
    - L4 權重降低: 1.0 → 0.5 (解決過擬合)
    - 支援動態權重衰減 (warmdown)
    - 監督 L3, L4, L6
    - 全部使用 Cosine Loss
    """

    def __init__(self, layer_weights: dict = None, target_scale: float = 1.0):
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                3: 0.3,   # model[3]: Downsample 1
                4: 0.5,   # model[4]: ResBlock 2 (降低自 1.0)
                6: 0.5,   # model[6]: Downsample 2
            }

        self.layer_weights = layer_weights
        self.layer_indices = sorted(layer_weights.keys())
        self.target_scale = target_scale

        print(f"[V6] Intermediate Supervision Config:")
        print(f"     Layers: {self.layer_indices}")
        print(f"     Weights: {layer_weights}")
        print(f"     Loss: All Cosine")
        print(f"     Feature: TracIn checkpoint support")

    def forward(self, student_features: dict, teacher_features: dict,
                layer_scale: float = 1.0):
        """
        計算中間層監督 Loss

        Args:
            student_features: Student 中間層特徵
            teacher_features: Teacher 中間層特徵
            layer_scale: 動態縮放因子 (用於 warmdown)

        Returns:
            total_loss: 總 Loss
            layer_losses: 各層 Loss dict
        """
        total_loss = 0.0
        layer_losses = {}

        for idx in self.layer_indices:
            # Support both int and str keys
            if idx in student_features:
                student_feat = student_features[idx]
                teacher_feat = teacher_features[idx]
            elif str(idx) in student_features:
                student_feat = student_features[str(idx)]
                teacher_feat = teacher_features[str(idx)]
            else:
                continue

            # 確保維度匹配 (B, T, C) or (B, C, T)
            if student_feat.shape != teacher_feat.shape:
                min_len = min(student_feat.shape[-1], teacher_feat.shape[-1])
                student_feat = student_feat[..., :min_len]
                teacher_feat = teacher_feat[..., :min_len]

            # Flatten to (B*T, C) for cosine similarity
            if student_feat.dim() == 3:
                B, C, T = student_feat.shape
                student_flat = student_feat.permute(0, 2, 1).reshape(-1, C)
                teacher_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)
            else:
                student_flat = student_feat
                teacher_flat = teacher_feat

            # Cosine Loss (with eps for numerical stability, prevent NaN from zero-norm)
            student_norm = F.normalize(student_flat, dim=-1, eps=1e-8)
            teacher_norm = F.normalize(teacher_flat * self.target_scale, dim=-1, eps=1e-8)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
            loss = 1 - cos_sim

            # 動態縮放
            weight = self.layer_weights[idx] * layer_scale
            total_loss = total_loss + weight * loss
            layer_losses[f'intermediate_L{idx}_loss'] = loss.item()

        layer_losses['layer_scale'] = layer_scale

        return total_loss, layer_losses


def compute_dynamic_intermediate_weight(epoch: int, curriculum_epochs: int,
                                         base_weight: float, min_weight: float,
                                         warmdown_epochs: int) -> float:
    """
    計算動態中間層權重

    策略:
    1. Curriculum 階段: 保持 base_weight
    2. Warmdown 階段: 從 base_weight 線性衰減到 min_weight
    3. 穩定階段: 保持 min_weight

    Args:
        epoch: 當前 epoch
        curriculum_epochs: Curriculum 結束的 epoch
        base_weight: 基礎權重
        min_weight: 最小權重
        warmdown_epochs: Warmdown 持續的 epoch 數

    Returns:
        dynamic_weight: 當前 epoch 的中間層權重
    """
    if epoch <= curriculum_epochs:
        # Curriculum 階段: 保持 base_weight
        return base_weight
    elif epoch <= curriculum_epochs + warmdown_epochs:
        # Warmdown 階段: 線性衰減
        progress = (epoch - curriculum_epochs) / warmdown_epochs
        return base_weight - progress * (base_weight - min_weight)
    else:
        # 穩定階段: 保持 min_weight
        return min_weight


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trainable_params(model):
    return (p for p in model.parameters() if p.requires_grad)


def get_lora_state_dict(model):
    """只提取 LoRA 參數"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state[name] = param.data.clone()
    return lora_state


def verify_model_state(model, stage: str):
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")
    if model.teacher.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Teacher quantizer 意外進入 train 模式!")
    if model.student.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Student quantizer 意外進入 train 模式!")


def train_epoch(model, dataloader, optimizer, loss_fn, intermediate_loss_fn, device, epoch,
                distance_matrix, intermediate_weight, encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
        'avg_snr': 0,
        'effective_intermediate_weight': 0,
    }
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

                # Final output loss
                final_loss, final_loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )

                # Intermediate supervision loss
                inter_loss, inter_loss_info = intermediate_loss_fn(
                    student_features=output['student_intermediates'],
                    teacher_features=output['teacher_intermediates'],
                )

                # Combined loss (使用動態權重)
                total_loss = final_loss + intermediate_weight * inter_loss
                total_loss = total_loss / gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio)

            final_loss, final_loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            inter_loss, inter_loss_info = intermediate_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )

            total_loss = final_loss + intermediate_weight * inter_loss
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        # Update metrics
        metrics['total_loss'] += total_loss.item() * gradient_accumulation_steps
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss
        metrics['effective_intermediate_weight'] += intermediate_weight

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        # Compute masked accuracy
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, correct, total_frames = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += correct
        metrics['total_frames'] += total_frames

        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item()*gradient_accumulation_steps:.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
            'inter': f"{inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss:.4f}",
            'iw': f"{intermediate_weight:.3f}"
        })

        if batch_idx % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx}")

    # Average metrics
    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, intermediate_loss_fn, device, epoch,
                   distance_matrix, intermediate_weight, encoder_stride=320):
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0, 'distance': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy_audio, clean_audio)

        final_loss, final_loss_info = loss_fn(
            student_features=output['student_encoder_out'],
            teacher_features=output['teacher_encoder_out'],
            teacher_codes=output['teacher_codes'],
            codebook=output['codebook'],
            lengths=lengths,
        )

        inter_loss, inter_loss_info = intermediate_loss_fn(
            student_features=output['student_intermediates'],
            teacher_features=output['teacher_intermediates'],
        )

        total_loss = final_loss + intermediate_weight * inter_loss

        metrics['total_loss'] += total_loss.item()
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, correct, total_frames = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        metrics['distance'] += 0

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
    """保存音檔樣本"""
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


def plot_training_curves_v6(history, save_path):
    """繪製 V6 訓練曲線"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Exp K v6: TracIn Support + L4 Weight Adjustment', fontsize=14)

    epochs = range(1, len(history['train_loss']) + 1)

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Masked Token Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [x*100 for x in history['train_masked_acc']], 'b-', label='Train')
    ax.plot(epochs, [x*100 for x in history['val_masked_acc']], 'r-', label='Val')
    ax.set_title('Masked Token Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_feature_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_feature_loss'], 'r-', label='Val')
    ax.set_title('Feature Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_triplet_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_triplet_loss'], 'r-', label='Val')
    ax.set_title('Triplet Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Intermediate Supervision Loss (Total)
    ax = axes[1, 1]
    ax.plot(epochs, history['train_intermediate_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_intermediate_loss'], 'r-', label='Val')
    ax.set_title('Intermediate Supervision Loss (Total)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Dynamic Intermediate Weight
    ax = axes[1, 2]
    if 'effective_intermediate_weight' in history:
        ax.plot(epochs, history['effective_intermediate_weight'], 'purple', linewidth=2)
    ax.set_title('V6: Dynamic Intermediate Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.grid(True)

    # Per-Layer Intermediate Loss (Train)
    ax = axes[2, 0]
    ax.plot(epochs, history['train_intermediate_L3_loss'], 'b-', label='L3 (w=0.3)')
    ax.plot(epochs, history['train_intermediate_L4_loss'], 'g--', label='L4 (w=0.5)')  # 更新標籤
    ax.plot(epochs, history['train_intermediate_L6_loss'], 'r-.', label='L6 (w=0.5)')
    ax.set_title('Per-Layer Intermediate Loss (Train)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Curriculum Phase
    ax = axes[2, 1]
    if 'curriculum_phase' in history:
        ax.plot(epochs, history['curriculum_phase'], 'orange', linewidth=2)
    ax.set_title('Curriculum Phase')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Max Noise Ratio')
    ax.set_ylim(0, 1.1)
    ax.grid(True)

    # Learning Rate
    ax = axes[2, 2]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'b-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp K v6: TracIn Support + L4 Weight Adjustment')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_k_v6')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA Config
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)

    # Learning Rate
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # V6: Dynamic Intermediate Supervision (調整 L4)
    parser.add_argument('--intermediate_weight', type=float, default=0.5,
                        help='Base intermediate loss weight')
    parser.add_argument('--intermediate_weight_min', type=float, default=0.25,
                        help='Minimum intermediate loss weight after warmdown')
    parser.add_argument('--warmdown_epochs', type=int, default=50,
                        help='Epochs for intermediate weight warmdown after curriculum')
    parser.add_argument('--intermediate_L3_weight', type=float, default=0.3)
    parser.add_argument('--intermediate_L4_weight', type=float, default=0.5,
                        help='L4 weight (reduced from 1.0 to fix overfitting)')
    parser.add_argument('--intermediate_L6_weight', type=float, default=0.5)
    parser.add_argument('--target_scale', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # Training
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

    # V6: Checkpoint saving for TracIn
    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                        help='Save checkpoint every N epochs for TracIn')
    parser.add_argument('--save_lora_only', action='store_true', default=True,
                        help='Only save LoRA parameters in checkpoints (smaller files)')

    # Curriculum
    parser.add_argument('--curriculum_start', type=float, default=0.3,
                        help='Initial curriculum phase (easy samples only)')
    parser.add_argument('--curriculum_end', type=float, default=0.85,
                        help='Final curriculum phase (exclude hardest 15%)')
    parser.add_argument('--curriculum_epochs', type=int, default=200,
                        help='Extended curriculum period')

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Audio samples
    parser.add_argument('--save_audio_interval', type=int, default=50)

    # Random seed
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path('exp_0112_intermediate/runs') / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints directory
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['v6_info'] = {
        'description': 'TracIn support + L4 weight adjustment',
        'changes': [
            'Checkpoint saving every N epochs for TracIn',
            'L4 weight reduced: 1.0 → 0.5 (fix overfitting)',
            'Same curriculum: 200 epochs, end=0.85',
            'Same warmdown: 50 epochs',
            'LoRA-only checkpoint option for smaller files',
        ],
        'motivation': 'V5 showed L4 val loss kept increasing (overfitting). '
                      'V6 reduces L4 weight and adds checkpoint saving for TracIn diagnosis.'
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp K v6: TracIn Support + L4 Weight Adjustment")
    print(f"{'='*60}")
    print(f"Output: {run_dir}")
    print(f"Config: L3(w={args.intermediate_L3_weight}) + L4(w={args.intermediate_L4_weight}) + L6(w={args.intermediate_L6_weight})")
    print(f"Intermediate weight: {args.intermediate_weight} -> {args.intermediate_weight_min} (warmdown)")
    print(f"Curriculum: {args.curriculum_start} -> {args.curriculum_end} over {args.curriculum_epochs} epochs")
    print(f"Warmdown: {args.warmdown_epochs} epochs after curriculum")
    print(f"Checkpoint: every {args.save_checkpoint_every} epochs (LoRA only: {args.save_lora_only})")
    print(f"{'='*60}\n")

    # Load distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)

    # Create model (L3, L4, L6)
    intermediate_indices = [3, 4, 6]

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        intermediate_indices=intermediate_indices,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
            return max(args.min_lr / args.lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Calculate curriculum phase advance interval
    num_phases = int((args.curriculum_end - args.curriculum_start) / 0.05) + 1
    phase_advance_epochs = max(1, args.curriculum_epochs // num_phases)
    phase_increment = (args.curriculum_end - args.curriculum_start) / num_phases

    print(f"Curriculum: {num_phases} phases, advance every {phase_advance_epochs} epochs")
    print(f"Phase increment: {phase_increment:.3f}")

    # Create data loaders
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        curriculum_mode='curriculum',
        initial_phase=args.curriculum_start,
        phase_increment=phase_increment,
        compute_snr=False,
    )

    # Create loss functions
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
    )

    layer_weights = {
        3: args.intermediate_L3_weight,
        4: args.intermediate_L4_weight,
        6: args.intermediate_L6_weight,
    }
    intermediate_loss_fn = IntermediateSupervisionLossV6(
        layer_weights=layer_weights,
        target_scale=args.target_scale,
    )

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_intermediate_loss': [], 'val_intermediate_loss': [],
        'train_intermediate_L3_loss': [], 'val_intermediate_L3_loss': [],
        'train_intermediate_L4_loss': [], 'val_intermediate_L4_loss': [],
        'train_intermediate_L6_loss': [], 'val_intermediate_L6_loss': [],
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'curriculum_phase': [],
        'effective_intermediate_weight': [],
        'lr': [],
        'checkpoints_saved': [],  # Track saved checkpoints
    }

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Update curriculum phase (only during curriculum period)
        if epoch > 1 and epoch <= args.curriculum_epochs and epoch % phase_advance_epochs == 0:
            if curriculum_sampler.current_phase < args.curriculum_end:
                curriculum_sampler.advance_phase()
                # Clamp to curriculum_end
                if curriculum_sampler.current_phase > args.curriculum_end:
                    curriculum_sampler.current_phase = args.curriculum_end

        current_phase = curriculum_sampler.current_phase

        # 計算動態中間層權重
        effective_intermediate_weight = compute_dynamic_intermediate_weight(
            epoch=epoch,
            curriculum_epochs=args.curriculum_epochs,
            base_weight=args.intermediate_weight,
            min_weight=args.intermediate_weight_min,
            warmdown_epochs=args.warmdown_epochs,
        )

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Curriculum Phase: {current_phase:.0%}")
        print(f"  Intermediate Weight: {effective_intermediate_weight:.3f}")
        if epoch > args.curriculum_epochs:
            warmdown_progress = min(1.0, (epoch - args.curriculum_epochs) / args.warmdown_epochs)
            print(f"  Warmdown Progress: {warmdown_progress:.0%}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, intermediate_loss_fn,
            device, epoch, distance_matrix, effective_intermediate_weight,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, intermediate_loss_fn,
            device, epoch, distance_matrix, effective_intermediate_weight
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_masked_acc'].append(train_metrics['masked_acc'])
        history['val_masked_acc'].append(val_metrics['masked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_intermediate_loss'].append(train_metrics['intermediate_loss'])
        history['val_intermediate_loss'].append(val_metrics['intermediate_loss'])

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            history[f'train_{key}'].append(train_metrics.get(key, 0))
            history[f'val_{key}'].append(val_metrics.get(key, 0))

        history['val_dist'].append(val_metrics['distance'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['curriculum_phase'].append(current_phase)
        history['effective_intermediate_weight'].append(effective_intermediate_weight)
        history['lr'].append(current_lr)

        # Print summary
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"Inter: L3={val_metrics.get('intermediate_L3_loss', 0):.4f}, "
              f"L4={val_metrics.get('intermediate_L4_loss', 0):.4f}, "
              f"L6={val_metrics.get('intermediate_L6_loss', 0):.4f}")
        print(f"LR: {current_lr:.6f}, IntWeight: {effective_intermediate_weight:.3f}")

        # Save best model
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, run_dir / 'best_model.pt')
            print(f"* New best Val Acc: {best_val_acc*100:.3f}%")

        # V6: Save checkpoint for TracIn
        if epoch % args.save_checkpoint_every == 0:
            ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch:03d}.pt'
            if args.save_lora_only:
                # Only save LoRA parameters (smaller file)
                lora_state = get_lora_state_dict(model)
                torch.save({
                    'epoch': epoch,
                    'lora_state_dict': lora_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['masked_acc'],
                    'train_acc': train_metrics['masked_acc'],
                    'val_loss': val_metrics['total_loss'],
                    'train_loss': train_metrics['total_loss'],
                }, ckpt_path)
                print(f"  Saved LoRA checkpoint: {ckpt_path.name}")
            else:
                # Save full model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['masked_acc'],
                    'train_acc': train_metrics['masked_acc'],
                    'val_loss': val_metrics['total_loss'],
                    'train_loss': train_metrics['total_loss'],
                }, ckpt_path)
                print(f"  Saved full checkpoint: {ckpt_path.name}")
            history['checkpoints_saved'].append(epoch)

        # Save audio samples
        if epoch % args.save_audio_interval == 0 or epoch == 1:
            print(f"\nSaving audio samples...")
            save_audio_samples(model, val_loader, device, run_dir, epoch, num_samples=2, split='val')
            save_audio_samples(model, train_loader, device, run_dir, epoch, num_samples=2, split='train')

        # Save history and plot
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == args.num_epochs:
            plot_training_curves_v6(history, run_dir / 'training_curves.png')

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc*100:.3f}% @ Epoch {best_epoch}")
    print(f"Checkpoints saved: {len(history['checkpoints_saved'])} (epochs: {history['checkpoints_saved']})")
    print(f"Output directory: {run_dir}")

    # Final plot
    plot_training_curves_v6(history, run_dir / 'training_curves.png')


if __name__ == '__main__':
    main()
