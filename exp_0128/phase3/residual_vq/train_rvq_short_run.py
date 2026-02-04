"""
exp_0128 Phase 3: Residual Vector Quantization (RVQ) Training

目的：
- 使用多層 RVQ 替代單層 VQ，強制 codebook diversity
- 通過架構層面改變解決 token collapse 問題
- Short-run: 1000 steps 驗證有效性

方法：
    z → q0 → residual0 → q1 → residual1 → q2 → ...
    最終: z_q = q0 + q1 + q2 + ...

成功判準（比 baseline 更嚴格）：
- Val entropy > 6.5 (baseline: 6.07)
- Val top-10 mass < 15% (baseline: 19.7%)
- Val strict acc >= 0.82% (90% of 0.91%)

執行：
    # Exp 5a: 2 層 RVQ (溫和)
    CUDA_VISIBLE_DEVICES=0 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
        --steps 1000 \
        --batch_size 8 \
        --n_rvq_layers 2 \
        --rvq_codebook_size 2048 \
        --output_dir exp_0128/phase3/residual_vq/run_exp5a \
        --seed 42

    # Exp 5b: 4 層 RVQ (推薦)
    CUDA_VISIBLE_DEVICES=0 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
        --steps 1000 \
        --batch_size 8 \
        --n_rvq_layers 4 \
        --rvq_codebook_size 1024 \
        --output_dir exp_0128/phase3/residual_vq/run_exp5b \
        --seed 42

    # Exp 5c: 8 層 RVQ (激進)
    CUDA_VISIBLE_DEVICES=0 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
        --steps 1000 \
        --batch_size 8 \
        --n_rvq_layers 8 \
        --rvq_codebook_size 512 \
        --output_dir exp_0128/phase3/residual_vq/run_exp5c \
        --seed 42
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0128.phase3.residual_vq.models_rvq import TeacherStudentRVQ
from exp_0112_intermediate.train_v6 import IntermediateSupervisionLossV6
from exp_1226.data_curriculum import create_curriculum_dataloaders


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Masked MSE over valid frames. lengths are audio sample lengths at 24kHz."""
    if lengths is None:
        return F.mse_loss(student, teacher)

    # 24kHz and quantizer frame_rate=75 => hop ~ 320 samples/frame.
    hop = 320
    T = student.shape[-1]
    frame_lens = (lengths + hop - 1) // hop  # ceil
    frame_lens = torch.clamp(frame_lens, min=0, max=T)

    frame_idx = torch.arange(T, device=student.device).unsqueeze(0)  # [1, T]
    mask = frame_idx < frame_lens.unsqueeze(1)  # [B, T]
    mask = mask.unsqueeze(1).to(student.dtype)  # [B, 1, T]

    sq = (student - teacher) ** 2
    sq = sq * mask

    denom = mask.sum() * student.shape[1]
    return sq.sum() / denom.clamp(min=1.0)


def evaluate_collapse_metrics(model, val_loader, device, max_batches=50):
    """
    評估 RVQ collapse metrics (針對 RVQ 架構修正)

    注意：不再比較 teacher codes vs student codes (不同 codebook space)
    改為評估：
    1. RVQ Layer 0 的多樣性 (與 baseline 比較)
    2. Joint code diversity (所有層組合)
    3. Feature space alignment (quantized vs teacher encoder)

    Returns:
        dict with RVQ-specific metrics
    """
    model.eval()

    all_layer0_codes = []  # 1D tensors of codes (masked, no padding)
    all_layer_codes_list = []  # [n_layers, n_tokens] tensors (masked, no padding)
    feature_distances = []  # Feature space alignment

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            lengths = batch.get('lengths', None)  # (B,) in audio samples (24kHz)
            if lengths is not None:
                lengths = lengths.cpu()

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Dataset may provide [B, T]; WavTokenizer expects [B, 1, T]
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0).unsqueeze(0)
            elif clean_audio.dim() == 2:
                clean_audio = clean_audio.unsqueeze(1)
            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0).unsqueeze(0)
            elif noisy_audio.dim() == 2:
                noisy_audio = noisy_audio.unsqueeze(1)

            output = model(clean_audio, noisy_audio)

            # Extract RVQ layer codes: [n_layers, batch, time]
            all_layer_codes = output['all_layer_codes'].cpu()
            n_layers, batch_size, time = all_layer_codes.shape

            # Mask out padding using audio lengths.
            # Dataset is 24kHz and quantizer frame_rate=75 => hop ~ 24000/75 = 320 samples/frame.
            if lengths is not None:
                hop = 320
                frame_lens = (lengths + hop - 1) // hop  # ceil(length/hop)
                frame_lens = torch.clamp(frame_lens, min=0, max=time)

                for b in range(batch_size):
                    L = int(frame_lens[b].item())
                    if L <= 0:
                        continue
                    all_layer0_codes.append(all_layer_codes[0, b, :L])
                    all_layer_codes_list.append(all_layer_codes[:, b, :L])
            else:
                # No lengths available: include all tokens
                all_layer0_codes.append(all_layer_codes[0].reshape(-1))
                all_layer_codes_list.append(all_layer_codes.reshape(n_layers, -1))

            # Feature space distance
            student_quantized = output['student_quantized']  # [batch, 512, time]
            teacher_encoder_out = output['teacher_encoder_out']  # [batch, 512, time]
            if lengths is not None:
                mse_sum = 0.0
                n_valid = 0
                for b in range(student_quantized.shape[0]):
                    L = int(frame_lens[b].item())
                    if L <= 0:
                        continue
                    mse_sum += F.mse_loss(
                        student_quantized[b, :, :L],
                        teacher_encoder_out[b, :, :L],
                    ).item()
                    n_valid += 1
                feature_distances.append(mse_sum / max(1, n_valid))
            else:
                feature_distances.append(
                    F.mse_loss(student_quantized, teacher_encoder_out).item()
                )

    # ===== Layer 0 Diversity Metrics =====
    all_layer0_codes = torch.cat(all_layer0_codes, dim=0).flatten()  # [total_tokens]
    codebook_size = model.rvq_codebook_size

    # Count code frequencies for Layer 0
    code_counts = torch.bincount(all_layer0_codes, minlength=codebook_size)
    code_probs = code_counts.float() / code_counts.sum()
    nonzero_probs = code_probs[code_probs > 0]

    # Entropy (Layer 0)
    entropy_layer0 = -(nonzero_probs * torch.log2(nonzero_probs)).sum().item()

    # Top-10 mass (Layer 0)
    top_10_probs, _ = torch.topk(code_probs, k=min(10, len(nonzero_probs)))
    top_10_mass_layer0 = top_10_probs.sum().item()

    # Used codes (Layer 0)
    used_codes_layer0 = (code_counts > 0).sum().item()

    # ===== Joint Code Diversity =====
    # Combine all layers into joint codes: [n_layers, total_tokens]
    all_layer_codes_flat = torch.cat(all_layer_codes_list, dim=1)
    joint_codes = all_layer_codes_flat.t().contiguous()  # [total_tokens, n_layers]

    unique_joint = torch.unique(joint_codes, dim=0).shape[0]
    total_joint = joint_codes.shape[0]
    joint_diversity = float(unique_joint) / float(total_joint)

    # ===== Feature Space Alignment =====
    mean_feature_dist = sum(feature_distances) / len(feature_distances)

    model.train()

    return {
        # Layer 0 metrics (comparable to baseline single VQ)
        'layer0_entropy': entropy_layer0,
        'layer0_top10_mass': top_10_mass_layer0,
        'layer0_used_codes': used_codes_layer0,
        'layer0_total_codes': codebook_size,
        'layer0_usage_pct': 100.0 * used_codes_layer0 / codebook_size,

        # Joint diversity (RVQ-specific)
        'joint_unique_codes': unique_joint,
        'joint_total_codes': total_joint,
        'joint_diversity': joint_diversity,  # Higher = better

        # Feature space alignment (primary training objective)
        'feature_mse': mean_feature_dist,  # Lower = better

        # Legacy for reference (but not meaningful for comparison)
        'entropy': entropy_layer0,  # For backward compatibility
        'top_10_mass': top_10_mass_layer0,
        'used_codes': used_codes_layer0,
    }


def analyze_rvq_layer_usage(model, all_layer_codes):
    """
    分析 RVQ 每層的使用情況

    Args:
        model: TeacherStudentRVQ model
        all_layer_codes: [n_layers, batch, time]

    Returns:
        dict mapping layer_idx → metrics
    """
    usage_stats = model.get_rvq_usage(all_layer_codes)

    layer_metrics = {}
    for layer_idx, stats in usage_stats.items():
        n_used = stats['n_used']
        entropy = stats['entropy']
        total = model.rvq_codebook_size

        layer_metrics[f'layer_{layer_idx}_used'] = n_used
        layer_metrics[f'layer_{layer_idx}_entropy'] = entropy
        layer_metrics[f'layer_{layer_idx}_usage_pct'] = 100.0 * n_used / total

    return layer_metrics


def save_audio_samples(model, data_loader, device, output_dir, step, num_samples=2, split='val'):
    """
    保存音頻樣本

    使用 Teacher decoder 重建 RVQ quantized vectors
    """
    model.eval()

    audio_dir = output_dir / 'audio_samples' / f'step_{step:06d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break

            clean_audio = batch['clean_audio'].to(device)
            noisy_audio = batch['noisy_audio'].to(device)

            # Forward pass
            outputs = model(clean_audio, noisy_audio)

            # Get quantized vectors from RVQ
            student_quantized = outputs['student_quantized']  # [batch, dim, time]

            # Use teacher's decoder to reconstruct audio
            # Teacher decoder can handle quantized vectors directly
            try:
                # Reconstruct using teacher decoder (via model.decode method)
                reconstructed = model.decode(student_quantized)

                # Save audio files
                batch_size = min(clean_audio.shape[0], 2)  # Save max 2 per batch
                for b in range(batch_size):
                    sample_idx = i * batch_size + b

                    # Save clean (target)
                    torchaudio.save(
                        audio_dir / f'{split}_sample{sample_idx}_clean.wav',
                        clean_audio[b].cpu(),
                        16000
                    )

                    # Save noisy (input)
                    torchaudio.save(
                        audio_dir / f'{split}_sample{sample_idx}_noisy.wav',
                        noisy_audio[b].cpu(),
                        16000
                    )

                    # Save reconstructed (RVQ output)
                    torchaudio.save(
                        audio_dir / f'{split}_sample{sample_idx}_reconstructed.wav',
                        reconstructed[b].cpu(),
                        16000
                    )

            except Exception as e:
                print(f"Warning: Could not reconstruct audio at step {step}: {e}")
                print("Continuing without audio samples...")

    model.train()
    print(f"Saved {num_samples} audio samples to {audio_dir}")


def plot_loss_curves(loss_history, metrics_history, output_dir):
    """繪製 loss 和 metrics 曲線（RVQ-specific）"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Extract data
    steps = [x['step'] for x in loss_history]
    train_losses = [x['total_loss'] for x in loss_history]
    loss_quant = [x['loss_quant'] for x in loss_history]
    loss_pre = [x['loss_pre'] for x in loss_history]
    loss_inter = [x['loss_inter'] for x in loss_history]
    loss_commit = [x['loss_commit'] for x in loss_history]
    loss_codebook = [x['loss_codebook'] for x in loss_history]

    # Extract evaluation metrics (only eval steps have these)
    eval_metrics = [x for x in loss_history if 'layer0_entropy' in x]
    eval_steps = [x['step'] for x in eval_metrics]

    # RVQ-specific metrics
    layer0_entropy = [x['layer0_entropy'] for x in eval_metrics]
    layer0_top10 = [x['layer0_top10_mass'] * 100 for x in eval_metrics]
    joint_diversity = [x['joint_diversity'] * 100 for x in eval_metrics]
    feature_mse = [x['feature_mse'] for x in eval_metrics]

    # Plot 1: Total loss
    axes[0, 0].plot(steps, train_losses, label='Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss components
    axes[0, 1].plot(steps, loss_quant, label='L_quant', linewidth=2)
    axes[0, 1].plot(steps, loss_pre, label='L_pre', linewidth=2)
    axes[0, 1].plot(steps, loss_inter, label='L_inter', linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: RVQ losses (commit/codebook)
    axes[0, 2].plot(steps, loss_commit, label='L_commit', linewidth=2, color='purple')
    axes[0, 2].plot(steps, loss_codebook, label='L_codebook', linewidth=2, color='gray')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('RVQ Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Layer 0 Entropy
    axes[1, 0].plot(eval_steps, layer0_entropy, 'o-', label='Layer0 Entropy', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=6.07, color='r', linestyle='--', label='Baseline (6.07)', linewidth=2)
    axes[1, 0].axhline(y=6.5, color='g', linestyle='--', label='Target (6.5)', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy (bits)')
    axes[1, 0].set_title('Layer 0 Entropy (vs Baseline)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Layer 0 Top-10 Mass
    axes[1, 1].plot(eval_steps, layer0_top10, 'o-', label='Layer0 Top-10', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=19.7, color='r', linestyle='--', label='Baseline (19.7%)', linewidth=2)
    axes[1, 1].axhline(y=15.0, color='g', linestyle='--', label='Target (15%)', linewidth=2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Top-10 Mass (%)')
    axes[1, 1].set_title('Layer 0 Top-10 Mass (vs Baseline)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Joint Diversity (RVQ-specific)
    axes[1, 2].plot(eval_steps, joint_diversity, 'o-', label='Joint Diversity', linewidth=2, markersize=6, color='purple')
    axes[1, 2].axhline(y=70.0, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Joint Diversity (%)')
    axes[1, 2].set_title('Joint Code Diversity (RVQ-specific)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 7: Feature MSE (primary objective)
    axes[2, 0].plot(eval_steps, feature_mse, 'o-', label='Feature MSE', linewidth=2, markersize=6, color='orange')
    axes[2, 0].axhline(y=0.1, color='g', linestyle='--', label='Target (0.1)', linewidth=2)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('MSE')
    axes[2, 0].set_title('Feature Space Alignment (student vs teacher)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Layer 0 Used Codes
    eval_used = [x.get('layer0_used_codes', 0) for x in eval_metrics]
    axes[2, 1].plot(eval_steps, eval_used, 'o-', label='Layer0 Used', linewidth=2, markersize=6)
    axes[2, 1].axhline(y=740, color='r', linestyle='--', label='Baseline (~740/4096)', linewidth=2)
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Number of Codes')
    axes[2, 1].set_title('Layer 0 Codebook Usage')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 9: Per-layer entropy (if available)
    if 'layer_0_entropy' in eval_metrics[0]:
        # infer number of layers from keys
        layer_ids = sorted({
            int(k.split('_')[1])
            for k in eval_metrics[0].keys()
            if k.startswith('layer_') and k.endswith('_entropy')
        })
        for i in layer_ids:
            layer_ent = [x.get(f'layer_{i}_entropy', 0) for x in eval_metrics]
            axes[2, 2].plot(eval_steps, layer_ent, 'o-', label=f'Layer {i}', linewidth=2, markersize=4)
        axes[2, 2].set_xlabel('Step')
        axes[2, 2].set_ylabel('Entropy (bits)')
        axes[2, 2].set_title('Per-Layer Entropy')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    else:
        # Placeholder if per-layer not available
        axes[2, 2].text(0.5, 0.5, 'Per-layer metrics\n(see RVQ analysis)',
                       ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('RVQ Layer Analysis')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='exp_0128 Phase 3-2: RVQ Fix Training')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_rvq_layers', type=int, default=4, help='Number of RVQ layers')
    parser.add_argument('--rvq_codebook_size', type=int, default=1024, help='Codebook size per layer')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')

    # Phase 3-2: Loss weights
    parser.add_argument('--lambda_quant', type=float, default=1.0, help='Weight for L_quant (z_q vs t_e)')
    parser.add_argument('--lambda_pre', type=float, default=0.0, help='Weight for L_pre (z_e vs t_e)')
    parser.add_argument('--lambda_inter', type=float, default=0.5, help='Weight for L_inter (intermediate supervision)')
    parser.add_argument('--beta_commit', type=float, default=0.25, help='Weight for L_commit (encoder commitment)')
    parser.add_argument('--lambda_codebook', type=float, default=1.0, help='Weight for L_codebook (codebook loss; grad mode only)')

    # Phase 3-2: Quantizer update mode
    parser.add_argument('--rvq_update', type=str, default='grad', choices=['grad', 'ema'], help='Codebook update mode')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay (rvq_update=ema)')
    parser.add_argument('--ema_eps', type=float, default=1e-5, help='EMA eps (rvq_update=ema)')
    parser.add_argument('--ema_dead_code_threshold', type=int, default=0, help='Dead-code threshold (rvq_update=ema; 0 disables)')
    parser.add_argument('--ema_usage_penalty', type=float, default=0.0, help='Usage penalty weight using log(EMA cluster_size) (rvq_update=ema)')

    # Phase 3-2: training controls
    parser.add_argument('--inter_warmup_steps', type=int, default=0, help='Warmup steps before enabling intermediate loss')
    parser.add_argument('--early_stop_on_collapse', action='store_true', help='Early stop at step 200 if collapse_flag=true')

    args = parser.parse_args()

    # Set random seed (與 baseline 一致)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("exp_0128 Phase 3-2: RVQ Fix Training")
    print("=" * 60)
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"RVQ layers: {args.n_rvq_layers}")
    print(f"Codebook size per layer: {args.rvq_codebook_size}")
    print(f"Total expressiveness: {args.rvq_codebook_size}^{args.n_rvq_layers}")
    print(f"RVQ update: {args.rvq_update}")
    if args.rvq_update == 'ema':
        print(f"  EMA decay: {args.ema_decay}")
        print(f"  EMA eps: {args.ema_eps}")
        print(f"  Dead-code threshold: {args.ema_dead_code_threshold}")
        print(f"  Usage penalty (log cluster_size): {args.ema_usage_penalty}")
    print(f"Loss weights: λ_quant={args.lambda_quant}, λ_pre={args.lambda_pre}, λ_inter={args.lambda_inter}, β_commit={args.beta_commit}, λ_codebook={args.lambda_codebook}")
    print(f"Inter warmup steps: {args.inter_warmup_steps}")
    print(f"Early stop on collapse: {args.early_stop_on_collapse}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Create dataloaders (standard curriculum dataset)
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        filter_clean_to_clean=True,
        compute_snr=False,  # Skip SNR for faster loading
    )

    # Create model with RVQ
    device = torch.device(args.device)
    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        intermediate_indices=[3, 6],
        device=device,
        n_rvq_layers=args.n_rvq_layers,
        rvq_codebook_size=args.rvq_codebook_size,
        rvq_update=args.rvq_update,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        ema_dead_code_threshold=args.ema_dead_code_threshold,
        ema_usage_penalty=args.ema_usage_penalty,
    )

    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={3: 0.5, 6: 0.5},
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training loop
    model.train()
    scaler = GradScaler()

    step = 0
    pbar = tqdm(total=args.steps, desc="Training")

    # Initial evaluation
    print("\n--- Initial Evaluation (Step 0) ---")
    metrics_init = evaluate_collapse_metrics(model, val_loader, device)
    print(f"Initial metrics:")
    for k, v in metrics_init.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save initial metrics
    metrics_history = [{'step': 0, **metrics_init}]
    loss_history = []

    train_iter = iter(train_loader)
    optimizer.zero_grad()
    last_eval_loss_idx = 0
    metrics_at_step200 = None
    early_stop_reason = None

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        # Dataset may provide [B, T]; WavTokenizer expects [B, 1, T]
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0).unsqueeze(0)
        elif clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0).unsqueeze(0)
        elif noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        lengths = batch.get('lengths', None)
        if lengths is not None:
            lengths = lengths.to(device)

        # Forward
        with autocast():
            output = model(clean_audio, noisy_audio)

            # Phase 3-2: Quantized alignment (primary) + optional pre-quant alignment
            loss_quant = masked_mse(
                student=output['student_quantized'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )
            loss_pre = masked_mse(
                student=output['student_encoder_out'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )

            # Intermediate loss (V6 API returns: (loss_tensor, layer_losses_dict))
            loss_inter_raw, _ = inter_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )
            loss_inter = loss_inter_raw if step >= args.inter_warmup_steps else (loss_inter_raw * 0.0)

            # RVQ losses (commit/codebook)
            loss_commit = output['rvq_loss_commit']
            loss_codebook = output['rvq_loss_codebook']

            # Total loss
            total_loss = (
                args.lambda_quant * loss_quant +
                args.lambda_pre * loss_pre +
                args.lambda_inter * loss_inter +
                args.beta_commit * loss_commit +
                args.lambda_codebook * loss_codebook
            )

            # Scale for gradient accumulation
            total_loss = total_loss / args.grad_accum

        # Backward
        scaler.scale(total_loss).backward()

        # Update weights every grad_accum steps
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Log
        loss_history.append({
            'step': step,
            'total_loss': (total_loss.item() * args.grad_accum),
            'loss_quant': loss_quant.item(),
            'loss_pre': loss_pre.item(),
            'loss_inter': loss_inter.item(),
            'loss_commit': loss_commit.item(),
            'loss_codebook': loss_codebook.item(),
        })

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item() * args.grad_accum:.4f}",
            'quant': f"{loss_quant.item():.4f}",
            'commit': f"{loss_commit.item():.4f}",
        })
        pbar.update(1)
        step += 1

        # Evaluation
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n--- Evaluation at Step {step} ---")
            metrics = evaluate_collapse_metrics(model, val_loader, device)

            # Average losses since last evaluation (for Phase 3-2 logging)
            if len(loss_history) > last_eval_loss_idx:
                window = loss_history[last_eval_loss_idx:]
                metrics['avg_total_loss'] = sum(x['total_loss'] for x in window) / len(window)
                for k in ['loss_quant', 'loss_pre', 'loss_inter', 'loss_commit', 'loss_codebook']:
                    metrics[f'avg_{k}'] = sum(x[k] for x in window) / len(window)
            last_eval_loss_idx = len(loss_history)

            # Analyze RVQ layer usage
            # Get a batch for layer analysis
            val_batch = next(iter(val_loader))
            val_clean = val_batch['clean_audio'].to(device)
            val_noisy = val_batch['noisy_audio'].to(device)
            # Dataset may provide [B, T]; WavTokenizer expects [B, 1, T]
            if val_clean.dim() == 1:
                val_clean = val_clean.unsqueeze(0).unsqueeze(0)
            elif val_clean.dim() == 2:
                val_clean = val_clean.unsqueeze(1)
            if val_noisy.dim() == 1:
                val_noisy = val_noisy.unsqueeze(0).unsqueeze(0)
            elif val_noisy.dim() == 2:
                val_noisy = val_noisy.unsqueeze(1)
            with torch.no_grad():
                val_output = model(val_clean, val_noisy)
                layer_metrics = analyze_rvq_layer_usage(
                    model,
                    val_output['all_layer_codes']
                )
                metrics.update(layer_metrics)

            # Phase 3-2 acceptance gates (P1 @ step200)
            K = model.rvq_codebook_size
            p1_used_thr = max(int(0.02 * K), 20)
            p1_pass = (
                metrics['layer0_top10_mass'] <= 0.95 and
                metrics['layer0_used_codes'] >= p1_used_thr and
                metrics['feature_mse'] <= 0.1
            )
            collapse_flag = (
                metrics['layer0_top10_mass'] > 0.95 and
                metrics['layer0_used_codes'] < (0.01 * K)
            )

            if step == 200:
                metrics['p1_pass'] = bool(p1_pass)
                metrics['collapse_flag'] = bool(collapse_flag)
                metrics_at_step200 = {'step': step, **metrics}

            metrics_history.append({'step': step, **metrics})
            loss_history[-1].update(metrics)

            print(f"Metrics at step {step}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            # Save checkpoint
            checkpoint_dir = output_dir / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step:06d}.pt'

            # Save model state (RVQ + student LoRA parameters)
            checkpoint = {
                'step': step,
                'rvq_state_dict': model.rvq.state_dict(),
                'student_state_dict': model.student.state_dict(),
                'metrics': metrics,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path.name}")

            # Print Phase 3-2 status (P1 gate at step 200)
            if step == 200:
                print("\n[P1 @ step200]")
                print(f"  layer0_top10_mass: {metrics['layer0_top10_mass']:.4f} (pass<=0.95)")
                print(f"  layer0_used_codes: {metrics['layer0_used_codes']} (pass>={p1_used_thr}, K={K})")
                print(f"  feature_mse: {metrics['feature_mse']:.4f} (pass<=0.1)")
                print(f"  P1 pass: {p1_pass}")
                print(f"  collapse_flag: {collapse_flag} (early-stop condition)")

                if collapse_flag and args.early_stop_on_collapse:
                    early_stop_reason = "collapse_flag@step200"
                    print("\n🛑 Early stop triggered (collapse_flag@step200).")
                    break

    pbar.close()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    # Reuse last eval metrics (eval already runs at step==args.steps or step==200 for P1 gate)
    final_eval = metrics_history[-1] if len(metrics_history) > 0 else {'step': 0, **metrics_init}
    final_metrics = {k: v for k, v in final_eval.items() if k != 'step'}
    final_step = int(final_eval.get('step', step))
    print("Final metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    K = args.rvq_codebook_size
    p0_pass = True

    # P1 (step200) gate
    if metrics_at_step200 is not None:
        p1_used_thr = max(int(0.02 * K), 20)
        p1_pass = (
            metrics_at_step200['layer0_top10_mass'] <= 0.95 and
            metrics_at_step200['layer0_used_codes'] >= p1_used_thr and
            metrics_at_step200['feature_mse'] <= 0.1
        )
        collapse_flag = bool(metrics_at_step200.get('collapse_flag', False))
    else:
        p1_pass = False
        collapse_flag = False

    # P2 (final) gate
    p2_pass = (
        final_metrics['layer0_entropy'] >= 5.0 and
        final_metrics['layer0_top10_mass'] <= 0.5 and
        final_metrics['layer0_used_codes'] >= int(0.10 * K) and
        final_metrics.get('joint_diversity', 0.0) >= 0.30 and
        final_metrics['feature_mse'] <= 0.1
    )

    # P3 (stretch; original Phase 3 strict targets)
    p3_pass = (
        final_metrics['layer0_entropy'] > 6.5 and
        final_metrics['layer0_top10_mass'] < 0.15 and
        final_metrics.get('joint_diversity', 0.0) > 0.7 and
        final_metrics['feature_mse'] < 0.1
    )

    summary = {
        'config': config,
        'final_step': final_step,
        'early_stop_reason': early_stop_reason,
        'acceptance': {
            'P0_pass': bool(p0_pass),
            'P1_pass': bool(p1_pass),
            'P2_pass': bool(p2_pass),
            'P3_pass': bool(p3_pass),
            'collapse_flag': bool(collapse_flag),
        },
        'step200_metrics': metrics_at_step200,
        'final_metrics': final_metrics,
        'baseline': {
            'single_vq_entropy': 6.07,
            'single_vq_top10_mass': 0.197,
            'single_vq_used_codes': 740,
            'note': 'Baseline uses single VQ with 4096 codebook, not directly comparable to RVQ'
        },
        'success_criteria': {
            'P1': {
                'layer0_top10_mass': '<= 0.95',
                'layer0_used_codes': f'>= max(0.02*K, 20) (K={K})',
                'feature_mse': '<= 0.1',
                'collapse_flag': '(top10_mass > 0.95) and (used_codes < 0.01*K)',
            },
            'P2': {
                'layer0_entropy': '>= 5.0',
                'layer0_top10_mass': '<= 0.5',
                'layer0_used_codes': f'>= 0.10*K (K={K})',
                'joint_diversity': '>= 0.30',
                'feature_mse': '<= 0.1',
            },
            'P3': {
                'layer0_entropy': '> 6.5',
                'layer0_top10_mass': '< 0.15',
                'joint_diversity': '> 0.7',
                'feature_mse': '< 0.1',
            },
        },
        # For Phase 3-2, "success" means passing P2 (worth continuing RVQ).
        'success': bool(p2_pass),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)

    with open(output_dir / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)

    # Plot curves
    plot_loss_curves(loss_history, metrics_history, output_dir)

    # Save final model checkpoint
    final_checkpoint_path = output_dir / 'final_model.pt'
    final_checkpoint = {
        'step': final_step,
        'rvq_state_dict': model.rvq.state_dict(),
        'student_state_dict': model.student.state_dict(),
        'metrics': final_metrics,
        'config': config,
    }
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"\nFinal model saved: {final_checkpoint_path.name}")

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
