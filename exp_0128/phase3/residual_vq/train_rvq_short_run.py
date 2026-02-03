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
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import create_curriculum_dataloaders


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

    all_layer0_codes = []  # Layer 0 codes for diversity check
    all_layer_codes_list = []  # All layers for joint analysis
    feature_distances = []  # Feature space alignment

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            output = model(clean_audio, noisy_audio)

            # Extract RVQ layer codes: [n_layers, batch, time]
            all_layer_codes = output['all_layer_codes']

            # Layer 0 codes (for primary diversity metric)
            layer0_codes = all_layer_codes[0]  # [batch, time]
            all_layer0_codes.append(layer0_codes.cpu())

            # All layers for joint analysis
            all_layer_codes_list.append(all_layer_codes.cpu())

            # Feature space distance
            student_quantized = output['student_quantized']  # [batch, 512, time]
            teacher_encoder_out = output['teacher_encoder_out']  # [batch, 512, time]
            feat_dist = F.mse_loss(student_quantized, teacher_encoder_out).item()
            feature_distances.append(feat_dist)

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
    # Combine all layers into joint codes
    all_layer_codes_cat = torch.cat(all_layer_codes_list, dim=1)  # [n_layers, total_batch, time]
    n_layers = all_layer_codes_cat.shape[0]

    # Create joint codes by treating each (layer0, layer1, ...) tuple as unique
    # Flatten to [n_layers, n_tokens]
    all_layer_codes_flat = all_layer_codes_cat.reshape(n_layers, -1)  # [n_layers, n_tokens]

    # Count unique tuples
    # Convert to hashable format for unique counting
    joint_codes_str = ['_'.join(map(str, all_layer_codes_flat[:, i].tolist()))
                       for i in range(all_layer_codes_flat.shape[1])]
    unique_joint = len(set(joint_codes_str))
    total_joint = len(joint_codes_str)
    joint_diversity = unique_joint / total_joint

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
    main_losses = [x['main_loss'] for x in loss_history]
    inter_losses = [x['inter_loss'] for x in loss_history]
    rvq_losses = [x['rvq_commitment_loss'] for x in loss_history]

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
    axes[0, 1].plot(steps, main_losses, label='Main Loss', linewidth=2)
    axes[0, 1].plot(steps, inter_losses, label='Intermediate Loss', linewidth=2)
    axes[0, 1].plot(steps, rvq_losses, label='RVQ Commitment', linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: RVQ Commitment Loss
    axes[0, 2].plot(steps, rvq_losses, label='RVQ Commitment', linewidth=2, color='purple')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('RVQ Commitment Loss')
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
        for i in range(4):  # Assume 4 layers for exp5b
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
    parser = argparse.ArgumentParser(description='exp_0128 Phase 3: RVQ Training')
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
    print("exp_0128 Phase 3: RVQ Training")
    print("=" * 60)
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"RVQ layers: {args.n_rvq_layers}")
    print(f"Codebook size per layer: {args.rvq_codebook_size}")
    print(f"Total expressiveness: {args.rvq_codebook_size}^{args.n_rvq_layers}")
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
    )

    # Create loss functions (same as baseline)
    loss_fn = MaskedCombinedLossV2(
        feature_weight=1.0,
        # RVQ Phase 3: start without triplet (teacher-codebook alignment) to avoid
        # accidentally re-imposing single-VQ collapse dynamics; enable as ablation.
        triplet_weight=0.0,
        triplet_margin=0.2,
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

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch.get('lengths', None)
        if lengths is not None:
            lengths = lengths.to(device)

        # Forward
        with autocast():
            output = model(clean_audio, noisy_audio)

            # Main loss
            loss_main, _ = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=model.codebook,
                lengths=lengths,
            )

            # Intermediate loss
            loss_inter = inter_loss_fn(
                student_intermediates=output['student_intermediates'],
                teacher_intermediates=output['teacher_intermediates'],
            )

            # RVQ commitment loss
            loss_rvq = output['rvq_commitment_loss']

            # Total loss
            total_loss = loss_main + loss_inter + loss_rvq

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
            'main_loss': loss_main.item(),
            'inter_loss': loss_inter.item(),
            'rvq_commitment_loss': loss_rvq.item(),
        })

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item() * args.grad_accum:.4f}",
            'main': f"{loss_main.item():.4f}",
            'rvq': f"{loss_rvq.item():.4f}",
        })
        pbar.update(1)
        step += 1

        # Evaluation
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n--- Evaluation at Step {step} ---")
            metrics = evaluate_collapse_metrics(model, val_loader, device)

            # Analyze RVQ layer usage
            # Get a batch for layer analysis
            val_batch = next(iter(val_loader))
            with torch.no_grad():
                val_output = model(
                    val_batch['clean_audio'].to(device),
                    val_batch['noisy_audio'].to(device)
                )
                layer_metrics = analyze_rvq_layer_usage(
                    model,
                    val_output['all_layer_codes']
                )
                metrics.update(layer_metrics)

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

            # Check success criteria (RVQ-specific)
            # 成功判準：
            # 1. Layer 0 多樣性優於 baseline
            # 2. Joint diversity 高（RVQ 特有）
            # 3. Feature space alignment 好
            success = (
                metrics['layer0_entropy'] > 6.5 and          # Layer 0 entropy > baseline (6.07)
                metrics['layer0_top10_mass'] < 0.15 and     # Layer 0 top-10 < baseline (19.7%)
                metrics['joint_diversity'] > 0.7 and         # Joint diversity > 70%
                metrics['feature_mse'] < 0.1                 # Feature alignment good
            )
            if success:
                print(f"\n✅ SUCCESS! All criteria met at step {step}")
            else:
                print(f"\nCurrent status:")
                print(f"  Layer0 Entropy: {metrics['layer0_entropy']:.2f} {'✅' if metrics['layer0_entropy'] > 6.5 else '❌'} (target > 6.5, baseline: 6.07)")
                print(f"  Layer0 Top-10: {metrics['layer0_top10_mass']*100:.1f}% {'✅' if metrics['layer0_top10_mass'] < 0.15 else '❌'} (target < 15%, baseline: 19.7%)")
                print(f"  Joint Diversity: {metrics['joint_diversity']*100:.1f}% {'✅' if metrics['joint_diversity'] > 0.7 else '❌'} (target > 70%)")
                print(f"  Feature MSE: {metrics['feature_mse']:.4f} {'✅' if metrics['feature_mse'] < 0.1 else '❌'} (target < 0.1)")

    pbar.close()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_metrics = evaluate_collapse_metrics(model, val_loader, device)
    print("Final metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    summary = {
        'config': config,
        'final_metrics': final_metrics,
        'baseline': {
            'single_vq_entropy': 6.07,
            'single_vq_top10_mass': 0.197,
            'single_vq_used_codes': 740,
            'note': 'Baseline uses single VQ with 4096 codebook, not directly comparable to RVQ'
        },
        'success_criteria': {
            'layer0_entropy': {'target': '> 6.5', 'baseline': 6.07},
            'layer0_top10_mass': {'target': '< 0.15', 'baseline': 0.197},
            'joint_diversity': {'target': '> 0.7', 'note': 'RVQ-specific'},
            'feature_mse': {'target': '< 0.1', 'note': 'Feature alignment'},
        },
        'success': (
            final_metrics['layer0_entropy'] > 6.5 and
            final_metrics['layer0_top10_mass'] < 0.15 and
            final_metrics['joint_diversity'] > 0.7 and
            final_metrics['feature_mse'] < 0.1
        ),
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
        'step': args.steps,
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
