"""
exp_0128 Phase 2: Entropy Regularization (實驗 3)

目的：
- 通過 entropy regularization 明確懲罰低 entropy 分佈
- 鼓勵使用多樣化的 codebook tokens
- Short-run: 1000 steps 驗證有效性

方法：
    total_loss = intermediate_loss + entropy_loss
    entropy_loss = -lambda_entropy * entropy  # Maximize entropy

成功判準：
- Val entropy > 6.07
- Val top-10 mass < 19.7%
- Val strict acc >= 0.82% (90% of 0.91%)

執行：
    CUDA_VISIBLE_DEVICES=0 python exp_0128/phase2/entropy_regularization/train_short_run.py \
        --steps 1000 \
        --batch_size 8 \
        --lambda_entropy 0.01 \
        --output_dir exp_0128/phase2/entropy_regularization/run_exp3a \
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
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_0112_intermediate.train_v6 import IntermediateSupervisionLossV6
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1219.data import create_curriculum_dataloaders


def compute_entropy_regularization(student_codes, codebook_size=4096):
    """
    計算 entropy regularization loss

    Args:
        student_codes: [B, T] token indices
        codebook_size: vocabulary size (default 4096)

    Returns:
        entropy: scalar entropy value
        entropy_loss: -lambda * entropy (to be minimized)
    """
    # Count token frequencies in the batch
    batch_size, seq_len = student_codes.shape
    total_tokens = batch_size * seq_len

    # Compute token distribution
    token_counts = torch.bincount(student_codes.flatten(), minlength=codebook_size).float()
    token_probs = token_counts / total_tokens

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    token_probs = token_probs + epsilon

    # Compute entropy: H = -sum(p * log(p))
    entropy = -(token_probs * torch.log(token_probs)).sum()

    return entropy


def evaluate_collapse_metrics(model, val_loader, device):
    """
    評估 collapse metrics

    Returns:
        metrics: dict with entropy, top_k_mass, strict_acc
    """
    model.eval()
    all_student_tokens = []
    all_teacher_tokens = []
    total_frames = 0
    correct_frames = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Forward
            output = model(noisy_audio, clean_audio)

            # Collect tokens (flatten each batch before appending)
            all_student_tokens.append(output['student_codes'].cpu().flatten())
            all_teacher_tokens.append(output['teacher_codes'].cpu().flatten())

            # Strict accuracy
            correct = (output['student_codes'] == output['teacher_codes']).sum().item()
            total = output['student_codes'].numel()
            correct_frames += correct
            total_frames += total

    # Concatenate all tokens
    all_student_tokens = torch.cat(all_student_tokens, dim=0)
    all_teacher_tokens = torch.cat(all_teacher_tokens, dim=0)

    # Compute metrics
    # 1. Entropy
    student_counts = torch.bincount(all_student_tokens, minlength=2048)
    student_probs = student_counts.float() / student_counts.sum()
    student_probs = student_probs[student_probs > 0]  # Remove zeros
    student_entropy = -(student_probs * torch.log(student_probs)).sum().item()

    # 2. Top-10 mass
    top_10_mass = student_probs.topk(10).values.sum().item()

    # 3. Strict accuracy
    strict_acc = correct_frames / total_frames

    # 4. Unique tokens
    unique_tokens = (student_counts > 0).sum().item()

    metrics = {
        'entropy': student_entropy,
        'top_10_mass': top_10_mass,
        'strict_acc': strict_acc,
        'unique_tokens': unique_tokens,
    }

    return metrics


def save_audio_samples(model, dataloader, device, output_dir, step, num_samples=2, split='val'):
    """保存音檔樣本"""
    model.eval()
    audio_dir = output_dir / 'audio_samples' / split / f'step_{step:04d}'
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
    model.train()  # Restore training mode


def plot_loss_curves(loss_history, save_path):
    """繪製訓練 Loss 曲線"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle('Exp 0128 Phase 2: Entropy Regularization', fontsize=12)

    steps = [x['step'] for x in loss_history]
    main_loss = [x['loss_main'] for x in loss_history]
    inter_loss = [x['loss_inter'] for x in loss_history]
    entropy_loss = [x.get('loss_entropy', 0) for x in loss_history]
    total_loss = [x['total_loss'] for x in loss_history]

    # Loss curves
    ax = axes[0]
    ax.plot(steps, total_loss, 'k-', label='Total Loss', linewidth=2)
    ax.plot(steps, main_loss, 'b-', label='Main Loss', alpha=0.7)
    ax.plot(steps, inter_loss, 'r-', label='Intermediate Loss', alpha=0.7)
    ax.plot(steps, entropy_loss, 'g-', label='Entropy Loss', alpha=0.7)
    ax.set_title('Training Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Collapse metrics
    ax = axes[1]
    metrics_steps = [x['step'] for x in loss_history if 'entropy' in x]
    entropy = [x['entropy'] for x in loss_history if 'entropy' in x]
    top_10_mass = [x['top_10_mass'] * 100 for x in loss_history if 'top_10_mass' in x]

    ax2 = ax.twinx()
    line1 = ax.plot(metrics_steps, entropy, 'g-', label='Entropy', marker='o')
    line2 = ax2.plot(metrics_steps, top_10_mass, 'orange', label='Top-10 Mass (%)', marker='s')

    ax.set_title('Collapse Metrics')
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy', color='g')
    ax2.set_ylabel('Top-10 Mass (%)', color='orange')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

    # Entropy loss over time
    ax = axes[2]
    ax.plot(steps, entropy_loss, 'g-', linewidth=2)
    ax.set_title('Entropy Regularization Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy Loss (should be negative)')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Exp 0128 Phase 2 - Entropy Regularization')
    parser.add_argument('--steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_entropy', type=float, default=0.01,
                        help='Entropy regularization weight (0.01, 0.05, 0.10)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--eval_interval', type=int, default=200, help='Eval every N steps')
    parser.add_argument('--save_checkpoint_every', type=int, default=200, help='Save checkpoint every N steps')
    parser.add_argument('--save_audio_interval', type=int, default=500, help='Save audio every N steps')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory and checkpoint subdirectory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # Save config
    config = vars(args)
    config['experiment'] = 'exp_0128_phase2_entropy_regularization'
    config['baseline'] = 'exp_k_v6'
    config['timestamp'] = datetime.now().isoformat()
    config['method'] = 'entropy_regularization'

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Exp 0128 Phase 2: Entropy Regularization")
    print("=" * 60)
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Lambda entropy: {args.lambda_entropy}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Create dataloaders (standard curriculum dataset with random sampling)
    train_loader, val_loader = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        filter_clean_to_clean=True,  # Same as baseline
    )

    # Create model (same as exp_k v6)
    device = torch.device(args.device)
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
        intermediate_indices=[3, 4, 6],
        device=device,
    )

    # Create loss functions (same as exp_k v6)
    loss_fn = MaskedCombinedLossV2(
        feature_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.2,
    )

    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={3: 0.3, 4: 0.5, 6: 0.5},
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
    print("\n--- Initial Evaluation ---")
    metrics_init = evaluate_collapse_metrics(model, val_loader, device)
    print(f"Initial metrics: {metrics_init}")

    # Save initial metrics and audio
    metrics_history = [{'step': 0, **metrics_init}]
    loss_history = []

    # Save initial audio samples
    print(f"\nSaving initial audio samples...")
    save_audio_samples(model, val_loader, device, output_dir, step=0, num_samples=2, split='val')
    save_audio_samples(model, train_loader, device, output_dir, step=0, num_samples=2, split='train')

    train_iter = iter(train_loader)

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
            output = model(noisy_audio, clean_audio)

            # Main loss
            loss_main, _ = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            # Intermediate loss (固定 weight=0.5，如 exp_k v6)
            loss_inter, _ = inter_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )

            # Entropy regularization (NEW!)
            entropy = compute_entropy_regularization(
                output['student_codes'],
                codebook_size=4096
            )
            loss_entropy = -args.lambda_entropy * entropy  # Maximize entropy (negative sign)

            # Total loss
            intermediate_weight = 0.5
            loss = (
                intermediate_weight * loss_inter +
                (1 - intermediate_weight) * loss_main +
                loss_entropy
            )

            # Gradient accumulation
            loss = loss / args.grad_accum

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track loss
        loss_history.append({
            'step': step,
            'total_loss': (loss.item() * args.grad_accum),
            'loss_main': loss_main.item(),
            'loss_inter': loss_inter.item(),
            'loss_entropy': loss_entropy.item(),
            'entropy': entropy.item(),
        })

        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item() * args.grad_accum:.4f}',
            'main': f'{loss_main.item():.4f}',
            'inter': f'{loss_inter.item():.4f}',
            'ent': f'{loss_entropy.item():.4f}',
            'H': f'{entropy.item():.2f}',
        })
        pbar.update(1)

        step += 1

        # Save checkpoint
        if step % args.save_checkpoint_every == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f'checkpoint_step{step:04d}.pt'
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"\n  Saved checkpoint: {ckpt_path.name}")

        # Save audio samples
        if step % args.save_audio_interval == 0 or step == args.steps:
            print(f"\nSaving audio samples at step {step}...")
            save_audio_samples(model, val_loader, device, output_dir, step, num_samples=2, split='val')
            save_audio_samples(model, train_loader, device, output_dir, step, num_samples=2, split='train')

        # Periodic evaluation
        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            print(f"\n--- Evaluation at step {step} ---")
            metrics = evaluate_collapse_metrics(model, val_loader, device)
            print(f"Metrics: {metrics}")

            # Add metrics to loss history
            loss_history[-1].update(metrics)
            metrics_history.append({'step': step, **metrics})

            # Save metrics and loss history
            with open(output_dir / 'metrics_history.json', 'w') as f:
                json.dump(metrics_history, f, indent=2)
            with open(output_dir / 'loss_history.json', 'w') as f:
                json.dump(loss_history, f, indent=2)

            # Plot curves
            plot_loss_curves(loss_history, output_dir / 'training_curves.png')

            model.train()

    pbar.close()

    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    metrics_final = evaluate_collapse_metrics(model, val_loader, device)
    print(f"Final metrics: {metrics_final}")

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics_final,
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")

    # Compare with baseline
    print("\n--- Comparison with Baseline (exp_k v6) ---")
    baseline = {
        'entropy': 6.07,
        'top_10_mass': 0.197,
        'strict_acc': 0.0091,
    }

    print(f"Metric          | Baseline | Final   | Change")
    print(f"----------------|----------|---------|----------")
    print(f"Entropy         | {baseline['entropy']:.2f}     | {metrics_final['entropy']:.2f}   | "
          f"{metrics_final['entropy'] - baseline['entropy']:+.2f}")
    print(f"Top-10 Mass     | {baseline['top_10_mass']*100:.1f}%   | {metrics_final['top_10_mass']*100:.1f}% | "
          f"{(metrics_final['top_10_mass'] - baseline['top_10_mass'])*100:+.1f}%")
    print(f"Strict Acc      | {baseline['strict_acc']*100:.2f}%  | {metrics_final['strict_acc']*100:.2f}% | "
          f"{(metrics_final['strict_acc'] - baseline['strict_acc'])*100:+.2f}%")

    # Save final summary
    summary = {
        'experiment': config['experiment'],
        'method': 'entropy_regularization',
        'lambda_entropy': args.lambda_entropy,
        'baseline': baseline,
        'final': metrics_final,
        'improvement': {
            'entropy': metrics_final['entropy'] - baseline['entropy'],
            'top_10_mass': metrics_final['top_10_mass'] - baseline['top_10_mass'],
            'strict_acc': metrics_final['strict_acc'] - baseline['strict_acc'],
        },
        'success': (
            metrics_final['entropy'] > baseline['entropy'] and
            metrics_final['top_10_mass'] < baseline['top_10_mass'] and
            metrics_final['strict_acc'] >= baseline['strict_acc'] * 0.9
        ),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'✅' if summary['success'] else '❌'} Experiment complete! Results saved to {output_dir}")
    print(f"Success: {summary['success']}")


if __name__ == '__main__':
    main()
